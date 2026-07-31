#!/usr/bin/env python3
"""
单片机通信状态机 —— 握手(PING/PONG) + 题目录像(TASK START/STOP -> ACK) + 钢珠 X 坐标持续发送
环境: /home/hao/vision_env/bin/python3（需 pyserial + opencv，vision_env 都有）

对应文档: Formal_code/通信协议/单片机树莓派通信协议.md
被 v1.1_beta.py import 使用，不单独运行。

串口热插拔连接管理直接复用 code/task_code/serial_link.py 的 SerialLink（找口/等口/
断线重连不重写），本模块只负责帧收发节奏和 START/STOP/录像/PONG 这套状态机。
"""

import os
import queue
import threading
import time
from datetime import datetime

import cv2

import mcu_protocol
from serial_link import SerialLink, Disconnected  # 由调用方(v1.1_beta.py)先插好 sys.path


class _VideoRecorder:
    """
    后台写盘线程 + 有界队列：把 mp4 编码(cv2.VideoWriter.write)从主循环搬到独立线程，
    录像时不再拖低"采集+检测+发串口"的主循环帧率（保留录像功能又尽量不掉帧）。
    VideoWriter 由本线程【独占】创建/写/释放（OpenCV VideoWriter 非线程安全，绝不让
    主线程碰 writer 对象），主线程只通过队列递交帧和控制消息。

    生命周期：McuLink 构造时起一个常驻 daemon 线程；每次 TASK 录像：
        open(w,h,fps,task_id)  ->  submit(frame) * N  ->  close()
    open/submit/close 都【非阻塞】，不卡主循环；close() 只入队一个哨兵，worker 把队列里
    剩余帧写完再 release + 落盘打印。McuLink.close() 调 stop() 停线程并 join(超时兜底)。

    队列【有界】：满了就丢帧 + 计数告警（宁可丢也绝不阻塞主循环）。正常负载下
    mp4v@640x480 在树莓派5 跟得上、队列近空、不丢帧；只有编码短暂跟不上才会丢。

    【录像倍速偏快的修复：预热测真实 fps，而不是直接用调用方传的瞬时值】
    open() 传入的 fps 只是"这一刻"的主循环 EMA 快照，实测发现它有时会明显偏高于这段
    录像自己实际能达到的帧率（比如相机 SDK 内部缓冲短暂"连续秒读"把 EMA 拉高），拿这个
    偏高值去建 VideoWriter，会导致 帧数/fps 算出的播放时长比真实经过时间短、看起来倍速。
    改法：收到 open() 后先不建 writer，进入"预热"状态，把最先来的一批帧(连同各自到达
    时间)缓冲在内存里(不丢、不写盘)，攒够 WARMUP_MIN_FRAMES 帧且经过 WARMUP_MIN_TIME
    (或达到 WARMUP_MAX_FRAMES 兜底上限)后，用这批帧自己的时间戳算出
    real_fps=(帧数-1)/(首尾时间差)，再真正建 writer、把缓冲的帧一次性冲下去、之后转入
    正常"收到就写"。STOP 如果在预热完成前就到了(录像很短)，就用已收集到的帧提前收尾。
    代价：每段录像开头有 ≤WARMUP_MIN_TIME 秒的决策延迟，但帧全部保留在缓冲区，不丢。

    【2026-08-01 进一步修复：USB 相机帧"突发式到达"仍会让预热窗口测出虚高 fps】
    实测录像 1x 播放比真实快 ~1.67 倍（用户要 0.6x 才贴合）：预热窗口只有 0.5s，撞上
    USB 相机的"攒一批→一次吐一批"突刺就把帧率测虚高（比如实际 30fps 测成 50fps）。
    两层修复：
    1. 预热窗口拉长到 WARMUP_MIN_TIME=2s（跨多个突刺周期取平均，压低突刺影响）。
    2. 收尾时用【整段录像】的首末帧墙钟时间算真实平均 fps，与封装 fps 偏差 >8% 就
       读回 mp4 用真实 fps 重封装替换（短录像几秒内完成，1x 播放精确贴合真实时间）。
       即使预热仍偏差（突刺周期 >2s 等），收尾校正兜底；不丢帧、不依赖 ffmpeg。
    """

    WARMUP_MIN_FRAMES = 60   # 至少攒这么多帧才够算一个靠谱的 real_fps
    WARMUP_MIN_TIME = 2.0    # 且至少经过这么久（跨多个突刺周期取平均，压掉相机突发性）
    WARMUP_MAX_FRAMES = 150  # 兜底上限：极端情况下也不能无限攒下去不开始写

    def __init__(self, video_dir, fallback_fps=30, maxsize=60, log=print):
        self.video_dir = video_dir
        self.fallback_fps = fallback_fps
        self.log = log
        self._q = queue.Queue(maxsize=maxsize)
        self._dropped = 0
        self._last_drop_log = 0.0
        self._alive = True
        self._thread = threading.Thread(target=self._run, name="video-writer", daemon=True)
        self._thread.start()

    # -------------------- 主线程侧 API（全部非阻塞） --------------------

    def open(self, w, h, fps, task_id):
        """
        开一段新录像（真正建 writer 延后到 worker 线程预热完成，见类文档"倍速偏快的修复"）。
        fps 这个参数现在只是个提示值，实际封装用的 fps 由 worker 自己预热这段录像的真实
        帧间隔算出来，传 0/None 也没关系。
        """
        self._put(("open", int(w), int(h), float(fps) if fps else 0.0, task_id))

    def submit(self, frame):
        """递交一帧【干净】画面；队列满则丢帧 + 计数（不阻塞主循环）。"""
        try:
            self._q.put_nowait(("frame", frame))
        except queue.Full:
            self._dropped += 1
            now = time.time()
            if now - self._last_drop_log >= 1.0:  # 丢帧告警限流，别刷屏
                self._last_drop_log = now
                self.log(f"⚠️ 录像写盘跟不上，累计丢弃 {self._dropped} 帧（编码速度 < 采集速度）。")

    def close(self):
        """结束当前这段录像：入队关闭哨兵，worker 写完剩余帧后 release + 落盘。"""
        self._put(("close",))

    def stop(self, join_timeout=3.0):
        """停后台线程（退出时调），先兜底关掉可能还开着的 writer 再 join。"""
        if not self._alive:
            return
        self._alive = False
        self._put(("quit",))
        self._thread.join(timeout=join_timeout)

    def _put(self, item):
        # 控制类消息(open/close/quit)很少，尽量别丢；满了阻塞一小会儿（不会真卡住主循环）。
        try:
            self._q.put(item, timeout=1.0)
        except queue.Full:
            self.log("⚠️ 录像控制消息入队超时，已忽略。")

    # -------------------------- worker 线程侧 --------------------------

    def _run(self):
        writer = None
        rec_path = None
        task_id = None
        state = "idle"        # idle / warming(攒帧算真实fps) / open(writer已建好)
        warmup = []            # [(frame, t_perf_counter), ...]，只在 warming 态使用
        pending_wh = None      # (w, h)，warming 态建 writer 时要用
        # 整段录像的计数/首末帧时间戳/封装fps：收尾时用整段真实时长核对封装帧率，
        # 偏差过大就重封装校正（见 _maybe_reencode）。每段 open 时重置。
        rec_meta = None        # {"count", "first", "last", "used_fps"}
        while True:
            item = self._q.get()
            kind = item[0]

            if kind == "frame":
                frame = item[1]
                now = time.perf_counter()
                if rec_meta is not None:
                    if rec_meta["first"] is None:
                        rec_meta["first"] = now
                    rec_meta["last"] = now
                    rec_meta["count"] += 1
                if state == "open" and writer is not None:
                    writer.write(self._watermark(frame, task_id))
                elif state == "warming":
                    # 预热阶段要把帧攒住等一小段时间才写，跟"open"态里帧一到就立即写掉不同，
                    # 必须 .copy()——不然如果上游相机/主线程复用同一块缓冲区，攒着的这些帧
                    # 会在真正写盘前被后面的帧数据覆盖掉。
                    warmup.append((frame.copy(), now))
                    elapsed = now - warmup[0][1]
                    ready = (len(warmup) >= self.WARMUP_MIN_FRAMES and elapsed >= self.WARMUP_MIN_TIME) \
                        or len(warmup) >= self.WARMUP_MAX_FRAMES
                    if ready:
                        writer, rec_path = self._finish_warmup(warmup, pending_wh, task_id, rec_meta)
                        warmup = []
                        state = "open" if writer is not None else "idle"

            elif kind == "open":
                # 异常路径：上一段没正常 close 就来了新 open（无论上一段是"已建 writer"还是
                # 还在"预热"），先把它当作要结束的一段收尾，避免帧丢在半空或 writer 泄漏。
                if state == "warming" and warmup:
                    w0, rp0 = self._finish_warmup(warmup, pending_wh, task_id, rec_meta)
                    if w0 is not None:
                        w0.release()
                        self.log(f"💾 录像已保存 → {rp0}")
                        self._maybe_reencode(rp0, rec_meta)
                    warmup = []
                if writer is not None:
                    writer.release()
                    self.log(f"💾 录像已保存 → {rec_path}")
                    self._maybe_reencode(rec_path, rec_meta)
                writer, rec_path = None, None
                _, w, h, _fps_hint, task_id = item
                pending_wh = (w, h)
                rec_meta = {"count": 0, "first": None, "last": None, "used_fps": None}
                state = "warming"

            elif kind == "close":
                if state == "warming":
                    writer, rec_path = self._finish_warmup(warmup, pending_wh, task_id, rec_meta)
                    warmup = []
                if writer is not None:
                    writer.release()
                    self.log(f"💾 录像已保存 → {rec_path}")
                    self._maybe_reencode(rec_path, rec_meta)
                writer, rec_path = None, None
                state = "idle"

            elif kind == "quit":
                if state == "warming":
                    writer, rec_path = self._finish_warmup(warmup, pending_wh, task_id, rec_meta)
                    warmup = []
                if writer is not None:
                    writer.release()
                    self.log(f"💾 录像已保存 → {rec_path}")
                    self._maybe_reencode(rec_path, rec_meta)
                break

    def _watermark(self, frame, task_id):
        # 水印画在【自己的 copy】上，不动主线程递交进来的共享帧（那帧主线程还可能拿去画
        # MJPEG 叠加层）；这份 copy 在 worker 线程做，不占主循环时间。
        rec = frame.copy()
        cv2.putText(rec, f"TASK {task_id}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
        return rec

    def _finish_warmup(self, warmup, wh, task_id, rec_meta=None):
        """
        预热结束（攒够/STOP提前来）：用缓冲帧自己的时间戳算出这段录像的真实 fps，
        建 writer，把缓冲帧按顺序一次性冲下去。不足 2 帧算不出间隔，退回 fallback_fps。
        """
        if not warmup:
            return None, None
        w, h = wh
        if len(warmup) >= 2:
            elapsed = warmup[-1][1] - warmup[0][1]
            real_fps = (len(warmup) - 1) / elapsed if elapsed > 0 else self.fallback_fps
        else:
            real_fps = self.fallback_fps
        writer, rec_path = self._open_writer(w, h, real_fps, task_id, rec_meta)
        if writer is not None:
            for frame, _t in warmup:
                writer.write(self._watermark(frame, task_id))
        return writer, rec_path

    def _open_writer(self, w, h, fps, task_id, rec_meta=None):
        os.makedirs(self.video_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.video_dir, f"{stamp}.mp4")
        # fps 是这段录像自己预热实测出的真实帧率（夹到合理区间）：固定/瞬时偏高的值去封装
        # 实际采到的帧，会让录像时长被压短；用真实帧率封装，播放时长才和实际一致。
        use_fps = fps if (fps and fps > 0) else self.fallback_fps
        use_fps = max(1.0, min(float(use_fps), 120.0))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, use_fps, (int(w), int(h)))
        if not writer.isOpened():
            self.log(f"❌ 无法创建录像文件 {path}（mp4v 编码不可用？）")
            return None, None
        if rec_meta is not None:
            rec_meta["used_fps"] = use_fps
        self.log(f"🎥 开始录像 → {path}（TASK {task_id}, {use_fps:.1f}fps 实测）")
        return writer, path

    def _maybe_reencode(self, path, rec_meta):
        """
        收尾校验：用整段录像的真实时长（首帧~末帧墙钟）核对封装 fps，偏差 >8% 就重封装校正，
        让 1x 播放速度贴合真实世界（实测 USB 相机突发到达会让预热 fps 虚高 ~1.67x）。
        短录像重封装几秒内完成、不阻塞主循环；失败保留原文件 + 告警，不崩。
        """
        if not path or rec_meta is None:
            return
        count = rec_meta.get("count") or 0
        used = rec_meta.get("used_fps") or 0
        first, last = rec_meta.get("first"), rec_meta.get("last")
        if count < 2 or first is None or last is None or last <= first:
            return
        if not (0 < used <= 120):
            return
        real = (count - 1) / (last - first)
        if not (0 < real <= 120):
            return
        if abs(real - used) / used < 0.08:
            return  # 封装 fps 已贴合真实，跳过
        self.log(f"📼 录像帧率校正: {used:.1f}→{real:.1f}fps（整段 {count} 帧 / 真实 "
                 f"{(last - first):.1f}s，按真实时长重封装）")
        if self._reencode_fps(path, real):
            self.log(f"  已替换 → {path}")
        else:
            self.log(f"  ⚠️ 重封装失败，保留原文件（fps {used:.1f}）")

    def _reencode_fps(self, path, fps):
        """读回已落盘的 mp4，按指定 fps 重写一份再原子替换（mp4v 重编码，短录像很快）。
        临时文件放在 video_dir 下的隐藏子目录里（保留 .mp4 后缀让 OpenCV 认出容器，实测
        无后缀/陌生后缀 VideoWriter 打不开；子目录不会被网页录像列表扫到），重编码成功后
        os.replace 原子替换。失败返回 False 保留原文件 + 清理临时目录。"""
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            cap.release()
            return False
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = max(1.0, min(float(fps), 120.0))
        tmpdir = os.path.join(self.video_dir, "_reencode_tmp")
        os.makedirs(tmpdir, exist_ok=True)
        tmp = os.path.join(tmpdir, os.path.basename(path))
        vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        if not vw.isOpened():
            cap.release()
            self._cleanup_tmpdir(tmpdir)
            return False
        try:
            while True:
                ok, fr = cap.read()
                if not ok:
                    break
                vw.write(fr)
        finally:
            cap.release()
            vw.release()
        if not os.path.exists(tmp) or os.path.getsize(tmp) <= 0:
            self._cleanup_tmpdir(tmpdir)
            return False
        os.replace(tmp, path)  # 同文件系统，原子替换
        self._cleanup_tmpdir(tmpdir)
        return True

    def _cleanup_tmpdir(self, tmpdir):
        """删掉重封装临时目录（含可能残留的半成品文件），失败静默忽略。"""
        try:
            for fn in os.listdir(tmpdir):
                os.unlink(os.path.join(tmpdir, fn))
            os.rmdir(tmpdir)
        except OSError:
            pass


class McuLink:
    """
    用法（主循环每帧调用）：
        mcu = McuLink(video_dir=..., video_fps=60, port=args.mcu_port)
        mcu.try_open()                 # 非阻塞，不卡视觉启动
        ...初始化完相机/检测/显示后...
        mcu.ready = True               # 协议要求：这之后才回 PONG
        while True:
            mcu.try_open()
            mcu.poll_incoming()        # 处理 PING/TASK，自动回 PONG/ACK
            mcu.send_x(x_px_or_None)   # 按协议节流发 $X / $X,NA
            # 每帧都调用！方法内部按需判断要不要写；measured_fps 传主循环实测帧率(fps_ema)，
            # 让录像文件用真实帧率封装、播放时长与实际一致。写盘在后台线程，不拖慢本循环。
            mcu.write_video_frame(clean_frame, measured_fps=fps_ema)
        mcu.close()                    # 退出时停后台写盘线程(兜底 release) + 关串口
    """

    def __init__(self, video_dir, video_fps=30, port=None, any_ok=True, log=print):
        self.link = SerialLink(port=port, any_ok=any_ok, log=log)
        self.video_dir = video_dir
        self.video_fps = video_fps
        self.log = log

        self.buf = bytearray()
        self.ready = False  # 相机/识别/显示/录像都初始化完成后才置 True，才回 PONG
        self.last_ping_t = 0.0

        self.current_run_id = None
        self.current_task_id = None
        # 录像写盘搬到后台线程（见 _VideoRecorder）。writer 归后台线程独占，主线程这边只留
        # 一个"是否正在录"的标志 + "该开新文件了"的标志；fallback_fps 用 video_fps。
        self._recorder = _VideoRecorder(video_dir, fallback_fps=video_fps, log=log)
        self._rec_active = False   # 主线程视角：当前是否处于一段录像中（START~STOP）
        self._pending_open = False  # 收到 START 后置真，第一帧到了才真正开文件（要帧尺寸）

        self.lost = True
        self._last_x_tx = 0.0
        self._last_na_tx = 0.0
        self.tx_x_count = 0
        self._tx_print_last = {}  # 按 throttle_key 记录上次打印时间，控制高频帧(X)的刷屏
        # 联调已跑通，终端不再刷"→ 串口发送"这行；发送本身（写串口/计数）完全不受影响，
        # 只是不打印。以后要再核对发送内容，把这个改回 True 即可，不用改别处代码。
        self.log_tx = False

    # ---------------------------- 连接管理（透传） ----------------------------

    @property
    def connected(self):
        return self.link.connected

    @property
    def mcu_online(self):
        """仅供本地 HUD 展示：3s 内收到过 PING 就认为 MCU 在线（协议里判定视觉离线的镜像逻辑）。"""
        return self.last_ping_t > 0 and (time.time() - self.last_ping_t) < 3.0

    @property
    def recording(self):
        # 主线程视角的"是否正在录"标志（真正的 writer 归后台线程独占，这里不看它）。
        return self._rec_active

    def try_open(self, min_interval=1.0):
        return self.link.try_open(min_interval)

    def wait_and_open(self, poll=0.5):
        self.link.wait_and_open(poll)

    # ------------------------------- 接收解析 -------------------------------

    def poll_incoming(self):
        """非阻塞读一次串口，切帧解析并分发处理；主循环每帧调一次。"""
        if not self.link.connected:
            return
        try:
            # 非阻塞：无数据立即返回，避免每帧白等一个串口 timeout(0.2s) 把视觉主循环
            # （连同发坐标频率）拖到个位数 Hz。见 serial_link.read_available_nonblocking。
            data = self.link.read_available_nonblocking()
        except Disconnected:
            return
        if data:
            self.buf += data
        while b"\n" in self.buf:
            idx = self.buf.index(b"\n")
            line = bytes(self.buf[:idx])
            del self.buf[:idx + 1]
            self._handle_line(line)

    def _handle_line(self, line):
        parsed = mcu_protocol.parse_frame(line)
        if parsed is None:
            self.log(f"← 串口接收(校验失败/无法解析): {line!r}")
            return
        msg_type, fields = parsed
        self.log(f"← 串口接收: {line.decode('ascii', 'replace').strip()}")
        if msg_type == "PING" and len(fields) == 1:
            self._on_ping(fields[0])
        elif msg_type == "TASK" and len(fields) == 3:
            self._on_task(fields[0], fields[1], fields[2])

    def _write(self, frame_bytes, log_always=True, throttle_key=None, throttle_interval=0.0):
        """
        写串口 + 打印，供 _on_ping/_send_ack/send_x 复用。
        log_always=False 时按 throttle_key 限流【打印】（不影响实际发送节奏），
        用于 $X 这种最高 60Hz 的高频帧，避免刷屏。
        返回是否真的写成功，供调用方决定要不要计数（保持和之前 try/except 一致的行为）。
        """
        try:
            self.link.write(frame_bytes)
        except Disconnected:
            return False
        if self.log_tx:
            text = frame_bytes.decode().strip()
            if log_always:
                self.log(f"→ 串口发送: {text}")
            else:
                now = time.time()
                if now - self._tx_print_last.get(throttle_key, 0.0) >= throttle_interval:
                    self._tx_print_last[throttle_key] = now
                    self.log(f"→ 串口发送: {text}")
        return True

    def _on_ping(self, ping_id):
        self.last_ping_t = time.time()
        if not self.ready:
            return  # 视觉还没初始化完成，按协议不回 PONG（MCU 会按 500ms 继续重发）
        self._write(mcu_protocol.build_pong_frame(ping_id))

    def _on_task(self, run_id, task_id, phase):
        if phase == "START":
            self._start_task(run_id, task_id)
        elif phase == "STOP":
            self._stop_task(run_id, task_id)

    def _start_task(self, run_id, task_id):
        if self.current_run_id == run_id and self._rec_active:
            # 同一 run_id 的重复 START：不重复创建录像文件，只重复回 ACK
            self._send_ack(run_id, task_id, "START")
            return
        if self._rec_active:
            self.log(f"⚠️ 收到新 run_id={run_id} 的 START，但仍在录 run_id={self.current_run_id}，先关闭旧文件。")
            self._recorder.close()
        self.current_run_id = run_id
        self.current_task_id = task_id
        self._rec_active = True
        self._pending_open = True  # 真正建文件延后到拿到帧尺寸的 write_video_frame()
        self._send_ack(run_id, task_id, "START")

    def _stop_task(self, run_id, task_id):
        if self._rec_active:
            self._recorder.close()  # 非阻塞：后台线程写完队列里剩余帧再 release + 落盘打印
        self._rec_active = False
        self._pending_open = False
        self._send_ack(run_id, task_id, "STOP")

    def _send_ack(self, run_id, task_id, phase):
        self._write(mcu_protocol.build_ack_frame(run_id, task_id, phase))

    # -------------------------------- 录像 --------------------------------

    def write_video_frame(self, frame, measured_fps=None):
        """
        主循环【每帧无条件调用】，传去畸变后的【干净】画面（不要传叠了调参可视化的那份）。
        measured_fps：主循环实测端到端帧率(fps_ema)，第一帧建 writer 时用来当录像帧率，
        让播放时长与实际一致（不传/无效则回退构造时的 video_fps）。

        实际写盘在【后台线程】(_VideoRecorder)：本方法只把帧【非阻塞入队】，几乎不占主循环
        时间，录像时也不拖低采集/检测/发串口的帧率。水印由后台线程画在自己的副本上。

        【踩坑，实测复现过】调用方不能用 `if mcu.recording: mcu.write_video_frame(frame)`
        包一层——START 后只置了标志，真正开文件要等本方法拿到第一帧的尺寸才做；早期
        `recording` 曾定义成 `writer is not None`，包一层会导致 writer 永远建不起来、录像
        文件永远不生成（握手却看似正常）。现在 `recording` 看主线程标志、本方法内部自带判断，
        调用方直接每帧调用即可，不要在外面再加 `if mcu.recording` 门槛。
        """
        if not self._rec_active:
            return
        if self._pending_open:
            # 第一帧到了才真正开文件（此刻才知道帧尺寸，且 fps_ema 已收敛可用）
            self._recorder.open(frame.shape[1], frame.shape[0], measured_fps, self.current_task_id)
            self._pending_open = False
        self._recorder.submit(frame)

    # ------------------------------ X 坐标发送 ------------------------------

    def send_x(self, x_val):
        """
        x_val: 当前帧检测到的主目标 x 坐标（int），None=丢球。
        有效坐标：【每读到一帧有效数据就立即发一次】，不做频率节流——发送快慢直接由视觉
        主循环的帧率决定（尽快发）。协议 §6 的 30~60Hz 只是"建议范围"，主循环本来就在这
        区间内，无需再人为限速。NA(丢球)：刚丢立即发一次，持续丢球每 200ms 一次（§7）。
        注：本项目当前发的是【全画面绝对像素 x】，不是协议定义的 mm——摆杆两端像素->毫米
        标定还没做，等标定好了只需把调用方传进来的 x_val 换成换算后的 mm 值，本方法不用改。
        """
        now = time.time()
        if x_val is not None:
            self.lost = False
            # 打印限流(0.3s)，只是别刷屏；发送本身每帧都发，不受打印限流影响。
            ok = self._write(mcu_protocol.build_x_frame(int(x_val)),
                              log_always=False, throttle_key="x", throttle_interval=0.3)
            if ok:
                self._last_x_tx = now
                self.tx_x_count += 1
        else:
            just_lost = not self.lost
            self.lost = True
            if just_lost or (now - self._last_na_tx) >= 0.2:
                if self._write(mcu_protocol.build_x_na_frame()):
                    self._last_na_tx = now

    # --------------------------------- 收尾 ---------------------------------

    def close(self):
        self._recorder.stop()  # 停后台写盘线程，兜底 release 掉可能还开着的录像文件
        self.link.close()
