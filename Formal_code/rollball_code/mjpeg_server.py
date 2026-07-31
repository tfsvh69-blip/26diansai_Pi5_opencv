#!/usr/bin/env python3
"""
局域网 MJPEG 实时画面服务 —— 浏览器打开 http://<本机局域网IP>:<port>/ 即可看当前摄像头画面。
被 v1.1_beta.py import 使用，不单独运行。

只用标准库 http.server(ThreadingHTTPServer) + cv2.imencode 编码 JPEG，不引入 Flask 等额外依赖
（比赛现场树莓派 vision_env 不一定装了 Flask，标准库最省心）。

两路画面流：
  /stream      主画面(彩色，检测框+HUD)
  /stream/bin  球体二值化掩膜(黑白，调参观测用)，同一套 _Stream 机制克隆一份。
外加一个只读+可写的检测参数 JSON API，配合网页滑块用：
  GET  /api/params   -> {"param2": 25, "hi_v": 121, ...}
  POST /api/params   body={"name": value, ...} -> 校验通过的立即生效(+触发调用方持久化)

外加一套录像回放 API（构造时传 replay_dir 才启用；FMP4 编码浏览器 <video> 原生放不了，
用 OpenCV 读帧 + 复用 _Stream 推 MJPEG，任何浏览器都能看；per-connection 独立流，互不干扰）：
  GET /api/recordings          -> {"ok":true, "files":[{name,size,mtime},...]}（按 mtime 倒序）
  GET /api/replay/info?file=x  -> {"ok":true, fps,frames,duration,w,h}（按 name+mtime 缓存）
  GET /replay?file=x&t=秒&speed=倍速  -> MJPEG 流：OpenCV 读 mp4、seek 到 t、按 fps*speed 节奏喂帧
  GET /replay/frame?file=x&t=秒       -> 单帧 JPEG 静图（暂停/拖进度即时预览）
未传 replay_dir 时这四个接口全部优雅降级（空列表/报错/占位帧），不影响直播。

用法（主循环里）：
    server = MjpegServer(port=8080)
    server.param_spec = [("param2", 1, 100, 1, "Hough严格度"), ...]
    server.get_params = lambda: {k: getattr(detector, k) for k, *_ in server.param_spec}
    server.set_param = lambda name, val: setattr(detector, name, val)
    server.doc_html = "<h3>...</h3><p>...</p>"  # "使用说明"按钮弹出的说明面板，纯 HTML 片段
    server.start()
    print(f"http://{get_lan_ip()}:{server.port}/")
    ...
    if server.has_clients:                   # 没人看主画面就不用浪费 CPU 编码/画叠加层
        server.update_frame(overlay_bgr)
    if server.has_bin_clients:                # 二值化窗口同理，独立门槛
        server.update_bin_frame(mask_bgr)
    ...
    server.stop()                             # 退出时收尾
"""

import json
import os
import socket
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

_BOUNDARY = b"FRAME"


def get_lan_ip():
    """
    猜本机在局域网里对外可见的 IP（不是 127.0.0.1）。
    用 UDP connect 到一个公网地址的技巧：不需要真的联网/发包，只是让操作系统按路由表
    选出会用哪张网卡的 IP，本机断网也不报错（探测失败时兜底退回 127.0.0.1）。
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def _placeholder_jpeg(w=640, h=480, quality=80, text="waiting for camera..."):
    """服务刚启动、主循环还没推第一帧时的占位画面，避免客户端一连上就因为没帧而断开。"""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(img, text, (max(10, w // 2 - 150), h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return buf.tobytes() if ok else b""


class _Stream:
    """
    单路 MJPEG 流的状态机：latest-wins 后台编码线程 + 条件变量推流。
    main/bin 两路各持有一个独立实例，互不干扰、各自的 has_clients 独立判断。
    """

    def __init__(self, quality=80, placeholder_text="waiting for camera..."):
        self.quality = quality
        # _cond/_lock：保护【已编码好的 JPEG】(_jpg/_frame_id)，推流处理线程在此等待新帧。
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._jpg = _placeholder_jpeg(quality=quality, text=placeholder_text)
        self._frame_id = 0
        self._clients = 0
        # 【性能】JPEG 编码(cv2.imencode)从主循环搬到这个后台【编码线程】：主循环 update()
        # 只存下最新一帧的引用 + 唤醒编码线程（非阻塞、latest-wins，丢弃中间帧对预览无影响），
        # imencode 的开销就不再压在"采集+检测+发串口"的主循环上，看网页时也不掉帧。
        self._in_lock = threading.Lock()
        self._in_cond = threading.Condition(self._in_lock)
        self._latest_bgr = None
        self._latest_seq = 0        # 主循环每递交一帧 +1
        self._running = False
        self._encoder = None

    @property
    def has_clients(self):
        with self._lock:
            return self._clients > 0

    def client_connected(self):
        with self._lock:
            self._clients += 1

    def client_disconnected(self):
        with self._lock:
            self._clients = max(0, self._clients - 1)

    def wait_frame(self, last_id, timeout=2.0):
        with self._cond:
            got = self._cond.wait_for(lambda: self._frame_id != last_id, timeout=timeout)
            if not got:
                return self._jpg, last_id  # 超时心跳：重发上一帧，防止中间代理判定连接已死
            return self._jpg, self._frame_id

    def update(self, bgr):
        """主循环调用【非阻塞】：只存下最新一帧的引用 + 唤醒后台编码线程。"""
        with self._in_cond:
            self._latest_bgr = bgr
            self._latest_seq += 1
            self._in_cond.notify()

    def _encoder_run(self):
        """后台编码线程：等主循环递交新帧 → cv2.imencode → 更新 _jpg 唤醒推流线程。"""
        last = 0
        while True:
            with self._in_cond:
                self._in_cond.wait_for(
                    lambda: self._latest_seq != last or not self._running, timeout=1.0)
                if not self._running:
                    break
                if self._latest_seq == last:
                    continue  # 超时且没有新帧，继续等
                bgr = self._latest_bgr
                last = self._latest_seq
            # imencode 放在锁外做，别占着输入锁
            ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
            if not ok:
                continue
            with self._cond:
                self._jpg = buf.tobytes()
                self._frame_id += 1
                self._cond.notify_all()

    def start(self):
        self._running = True
        self._encoder = threading.Thread(target=self._encoder_run, name="mjpeg-encoder", daemon=True)
        self._encoder.start()

    def stop(self):
        # 先停编码线程，避免它在下面把 _jpg 置 None 之后又写回一帧真图（推流线程靠 None 退出）。
        with self._in_cond:
            self._running = False
            self._in_cond.notify_all()
        if self._encoder is not None:
            self._encoder.join(timeout=2.0)
            self._encoder = None
        # 再通知推流处理线程收工（_jpg=None 让它们 break 出循环）
        with self._cond:
            self._jpg = None
            self._frame_id += 1
            self._cond.notify_all()


class _ReplayReader:
    """
    回放流读取线程：按固定节奏把 mp4 帧喂进一个 _Stream（与主循环喂直播帧是同一个 update()
    接口，_Stream 本身一行不用改）。per-connection：一条 /replay 连接一个实例，连接断开时
    stop() 收工；本连接专属的 _Stream/编码线程也由调用方随之 stop()。

    pacing 用单调时钟累加【绝对目标时刻】（deadline = t0 + idx*interval），只累加不重算，
    平均速率精确不漂移；内层 sleep 分片 ≤50ms，能及时响应 stop。某帧处理超时导致 now 已过
    deadline 时直接读下一帧（轻微快进追平），不会永久偏移。
    """

    def __init__(self, cap, stream, interval, log=print):
        self._cap = cap          # 已 seek 到起点、可正常读的 cv2.VideoCapture
        self._stream = stream    # 本连接专属的 _Stream
        self._interval = interval  # 每帧间隔秒 = 1/(fps*speed)
        self._log = log
        self._running = False
        self._thread = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, name="replay-reader", daemon=True)
        self._thread.start()

    def stop(self, join_timeout=2.0):
        """先置停止标志、join 等 reader 退出，再 release cap——避免在 cap.read() 进行中
        从另一线程释放（OpenCV 未定义行为）。本地文件 read 很快，join 通常 <50ms。"""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=join_timeout)
            self._thread = None
        self._cap.release()

    def _run(self):
        t0 = time.monotonic()
        idx = 0
        while self._running:
            ok, bgr = self._cap.read()
            if not ok:
                break  # EOF：跳出后靠 wait_frame 的超时心跳保住最后一帧/占位帧
            self._stream.update(bgr)
            idx += 1
            deadline = t0 + idx * self._interval
            while True:
                now = time.monotonic()
                if now >= deadline or not self._running:
                    break
                time.sleep(min(deadline - now, 0.05))


class _ReplayPlayer:
    """
    回放管理：列录像目录、读单文件元数据（按 (size,mtime) 缓存，录制中 mtime 变自动失效）、
    开流（打开+seek+探测）、取单帧 JPEG。只读 + 缓存带锁，多连接并发安全，无需互斥。
    目录/文件打不开一律返回可安全处理的失败值，不抛异常、不崩服务。
    """

    def __init__(self, replay_dir, quality=80, log=print):
        self.replay_dir = replay_dir
        self.quality = quality
        self.log = log
        self._info_cache = {}    # name -> ((size, mtime), info_dict)
        self._info_lock = threading.Lock()

    # ------------------------------ 安全路径 ------------------------------

    def _resolve(self, name):
        """只允许 mp4 文件名，拒绝路径穿越（含 ../、子目录）；非法返回 None。"""
        if not isinstance(name, str) or not name:
            return None
        if name != os.path.basename(name):
            return None
        if not name.lower().endswith(".mp4"):
            return None
        return os.path.join(self.replay_dir, name)

    # ------------------------------- 列表 --------------------------------

    def list_recordings(self):
        """-> [{"name","size","mtime"}, ...]，按 mtime 倒序；目录不存在/为空返回 []。"""
        if not self.replay_dir or not os.path.isdir(self.replay_dir):
            return []
        out = []
        try:
            for fn in os.listdir(self.replay_dir):
                if not fn.lower().endswith(".mp4"):
                    continue
                p = os.path.join(self.replay_dir, fn)
                try:
                    st = os.stat(p)
                except OSError:
                    continue
                out.append({"name": fn, "size": st.st_size, "mtime": st.st_mtime})
        except OSError:
            return []
        out.sort(key=lambda f: f["mtime"], reverse=True)
        return out

    # ------------------------------- 元数据 -------------------------------

    def _read_info(self, path):
        """打开一次读元数据；文件打不开/frames<=0/fps 异常返回 None。"""
        cap = cv2.VideoCapture(path)
        try:
            if not cap.isOpened():
                return None
            fps = cap.get(cv2.CAP_PROP_FPS)
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if not (0 < fps <= 500) or frames <= 0:
                return None
            return {
                "fps": fps,
                "frames": frames,
                "duration": frames / fps,
                "w": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "h": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            }
        finally:
            cap.release()

    def info(self, name):
        """-> {"ok":bool, ...}。缓存键 = (size, mtime)；正在写入的文件 mtime 持续变、缓存自然失效。
        frames<=0（moov 未落盘）判为"正在写入或已损坏"。"""
        path = self._resolve(name)
        if path is None:
            return {"ok": False, "error": "非法文件名"}
        try:
            st = os.stat(path)
        except OSError:
            return {"ok": False, "error": "文件不存在"}
        key = (st.st_size, st.st_mtime)
        with self._info_lock:
            cached = self._info_cache.get(name)
            if cached is not None and cached[0] == key:
                return cached[1]
        meta = self._read_info(path)
        if meta is None:
            result = {"ok": False, "error": "文件正在写入或已损坏"}
        else:
            result = {"ok": True, "name": name, **meta}
        with self._info_lock:
            self._info_cache[name] = (key, result)
        return result

    # ------------------------------- 开流 -------------------------------

    def open_stream(self, name, t, speed):
        """打开 + seek(ms) + 探测 read + 复位，返回 (cap, interval) 或 None。
        seek 用 CAP_PROP_POS_MSEC（录像 fps 是实测非整数，按帧号算有累积误差）。
        探测 read 是必须的：正在录制的 mp4 moov 未落盘，isOpened() 不可靠，read() 才是
        "能否真播"的判据；t 超界/文件被删/损坏都会在这里失败。"""
        path = self._resolve(name)
        if path is None:
            return None
        cap = cv2.VideoCapture(path)
        try:
            if not cap.isOpened():
                cap.release()
                return None
            fps = cap.get(cv2.CAP_PROP_FPS)
            if not (0 < fps <= 500):
                fps = 30.0  # 兜底异常值，避免除零
            t = max(0.0, t)
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, _probe = cap.read()
            if not ok:
                cap.release()
                return None
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)  # 探测消费了一帧，复位到 t
            return cap, 1.0 / (fps * speed)
        except Exception:
            cap.release()
            return None

    def frame_jpeg(self, name, t):
        """单帧 JPEG bytes（暂停/拖进度的静图）。失败返回占位图，不抛异常。"""
        res = self.open_stream(name, t, 1.0)
        if res is None:
            return _placeholder_jpeg(quality=self.quality, text="回放: 文件无法打开")
        cap, _ = res
        try:
            ok, bgr = cap.read()
            if not ok:
                return _placeholder_jpeg(quality=self.quality, text="回放: 文件无法打开")
            ok2, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
            if not ok2:
                return _placeholder_jpeg(quality=self.quality, text="回放: 编码失败")
            return buf.tobytes()
        finally:
            cap.release()


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # 静默访问日志（默认会刷到 stderr），避免刷屏拖慢主程序打印

    def do_GET(self):
        mjpeg = self.server.mjpeg
        if self.path in ("/", "/index.html"):
            body = mjpeg.index_html().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/stream":
            self._serve_stream(mjpeg.main)
            return
        if self.path == "/stream/bin":
            self._serve_stream(mjpeg.bin)
            return
        if self.path == "/api/params":
            self._serve_params_get(mjpeg)
            return
        # 回放路由带 query（?file=&t=&speed=），用 path_only 精确匹配，query 留给各 serve 方法解析。
        path_only = urllib.parse.urlparse(self.path).path
        if path_only == "/api/recordings":
            self._serve_recordings(mjpeg)
            return
        if path_only == "/api/replay/info":
            self._serve_replay_info(mjpeg)
            return
        if path_only == "/replay/frame":
            self._serve_replay_frame(mjpeg)
            return
        if path_only == "/replay":
            self._serve_replay(mjpeg)
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        mjpeg = self.server.mjpeg
        if self.path == "/api/params":
            self._serve_params_post(mjpeg)
            return
        if self.path == "/api/save":
            self._serve_save(mjpeg)
            return
        self.send_response(404)
        self.end_headers()

    def _serve_stream(self, stream):
        self.send_response(200)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={_BOUNDARY.decode()}")
        self.end_headers()
        stream.client_connected()
        try:
            last_id = -1
            while True:
                jpg, last_id = stream.wait_frame(last_id)
                if jpg is None:
                    break  # 服务在关停 (stop())
                self.wfile.write(b"--" + _BOUNDARY + b"\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpg)}\r\n\r\n".encode())
                self.wfile.write(jpg)
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass  # 客户端关了浏览器/断网，安静退出这个连接线程
        finally:
            stream.client_disconnected()

    def _serve_recordings(self, mjpeg):
        files = mjpeg.replay_player.list_recordings() if mjpeg.replay_player is not None else []
        body = json.dumps({"ok": True, "files": files}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_replay_info(self, mjpeg):
        q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        name = q.get("file", [""])[0]
        if mjpeg.replay_player is not None and name:
            info = mjpeg.replay_player.info(name)
        else:
            info = {"ok": False, "error": "回放未启用"}
        body = json.dumps(info).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_replay_frame(self, mjpeg):
        q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        name = q.get("file", [""])[0]
        try:
            t = max(0.0, float(q.get("t", ["0"])[0]))
        except ValueError:
            t = 0.0
        if mjpeg.replay_player is not None and name:
            body = mjpeg.replay_player.frame_jpeg(name, t)
        else:
            body = _placeholder_jpeg(text="回放未启用")
        # 静图必须禁缓存：seek 到不同 t 要重新取到对应的帧，不能让浏览器复用旧图。
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _serve_replay(self, mjpeg):
        """
        回放 MJPEG 流：per-connection 独立 _Stream（独享编码线程，多客户端互不串台）。
        直接复用 _serve_stream 作内层推流（内含 multipart 头 + client_connected/disconnected），
        finally 里先停 reader（先 join 再 release cap）再停 _Stream。文件打不开只显示
        "回放: 加载中…"占位帧 + log，不崩；reader 到 EOF 后靠 wait_frame 超时心跳保持连接。
        """
        q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        name = q.get("file", [""])[0]
        try:
            t = max(0.0, float(q.get("t", ["0"])[0]))
        except ValueError:
            t = 0.0
        try:
            speed = max(0.1, min(float(q.get("speed", ["1"])[0]), 8.0))
        except ValueError:
            speed = 1.0

        player = mjpeg.replay_player
        stream = _Stream(quality=mjpeg.quality, placeholder_text="回放: 加载中…")
        reader = None
        try:
            if player is not None and name:
                res = player.open_stream(name, t, speed)
                if res is not None:
                    cap, interval = res
                    reader = _ReplayReader(cap, stream, interval)
            # 先登记再 start：保证 server.stop() 一定能找到并停掉这条回放连接。
            mjpeg._replay_started(reader, stream)
            stream.start()
            if reader is not None:
                reader.start()
            elif name and player is not None:
                player.log(f"回放打开失败: {name!r} t={t} speed={speed}（正在写入/不存在/损坏）")
            self._serve_stream(stream)
        finally:
            mjpeg._replay_finished(reader, stream)
            if reader is not None:
                reader.stop()   # 先 join 等 reader 退出，再 release cap
            stream.stop()       # 停编码线程，_jpg=None 让 wait_frame 退出，_serve_stream 才返回

    def _serve_save(self, mjpeg):
        """立即落盘当前参数(不等 1s 防抖)，供网页"保存参数"按钮用，明确反馈成功与否。"""
        ok = bool(mjpeg.on_save()) if mjpeg.on_save else False
        body = json.dumps({"ok": ok}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_params_get(self, mjpeg):
        data = mjpeg.get_params() if mjpeg.get_params else {}
        body = json.dumps(data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_params_post(self, mjpeg):
        length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(length) if length > 0 else b""
        try:
            payload = json.loads(raw.decode("utf-8")) if raw else {}
        except (json.JSONDecodeError, UnicodeDecodeError):
            self.send_response(400)
            self.end_headers()
            return
        if not isinstance(payload, dict):
            self.send_response(400)
            self.end_headers()
            return

        # 服务端按 param_spec 白名单校验+夹值，不相信前端传什么就设什么。
        spec = {name: (lo, hi) for name, lo, hi, *_ in mjpeg.param_spec}
        applied = {}
        for name, val in payload.items():
            if name not in spec or mjpeg.set_param is None:
                continue
            lo, hi = spec[name]
            try:
                val = float(val)
            except (TypeError, ValueError):
                continue
            val = max(lo, min(hi, val))
            if float(val).is_integer():
                val = int(val)
            mjpeg.set_param(name, val)
            applied[name] = val

        body = json.dumps(applied).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class _Server(ThreadingHTTPServer):
    daemon_threads = True  # 主程序退出时不用等还在推流的客户端线程


class MjpegServer:
    def __init__(self, port=8080, quality=80, replay_dir=None):
        self.port = port
        # 原来没存 quality，现在建 per-connection 回放 _Stream 要用，补存一份。
        self.quality = quality
        self.main = _Stream(quality=quality, placeholder_text="waiting for camera...")
        self.bin = _Stream(quality=quality, placeholder_text="")
        self._httpd = None
        self._thread = None
        # 录像回放：传入录像目录才启用（None 时 /api/recordings 等接口全部优雅降级）。
        self.replay_player = _ReplayPlayer(replay_dir, quality=quality) if replay_dir else None
        # 活跃回放连接登记（reader + per-connection _Stream）。server.stop() 时统一收干净，
        # 避免主程序退出时还挂着的回放 daemon 线程带着 cv2.VideoCapture 被强杀 → OpenCV abort
        # （实测复现过 "FATAL: exception not rethrown" 核心转储）。
        self._replay_lock = threading.Lock()
        self._replay_readers = set()
        self._replay_streams = set()

        # 检测参数面板：由外部(v1.1_beta.py)注入，本模块不知道 BallDetector 长什么样。
        # param_spec: [(name, min, max, step, 中文label), ...]
        self.param_spec = []
        self.get_params = None   # () -> dict
        self.set_param = None    # (name, value) -> None
        self.on_save = None      # () -> bool，网页"保存参数"按钮触发的立即落盘
        # 使用说明面板：外部(v1.1_beta.py)直接注入一段现成的 HTML 片段（本模块不关心内容
        # 结构，只负责渲染 + 一个显示/隐藏的按钮），说明每个参数是什么、什么情况下怎么调。
        self.doc_html = ""

    @property
    def has_clients(self):
        return self.main.has_clients

    @property
    def has_bin_clients(self):
        return self.bin.has_clients

    def index_html(self):
        sliders = "".join(
            f"""
            <div class="row">
              <label for="p_{name}">{label}</label>
              <input type="range" id="p_{name}" min="{lo}" max="{hi}" step="{step}">
              <span id="v_{name}"></span>
            </div>"""
            for name, lo, hi, step, label in self.param_spec
        )
        # 回放面板 HTML/JS 用普通字符串（非 f-string），JS 里的 { } 不用转义；
        # 再通过 {replay_panel}/{replay_js} 占位符嵌进下面的 f-string。
        replay_panel = """<div id="replayPanel" style="display:none; background:#1c1c1c; padding:12px 16px; max-width:760px; margin:0 auto;">
  <div class="row">
    <select id="recSelect" style="flex:1;"></select>
    <button id="playBtn" onclick="playPause()" disabled>播放</button>
    <select id="speedSel" onchange="onSpeed()">
      <option value="0.5">0.5x</option><option value="1" selected>1x</option>
      <option value="2">2x</option><option value="4">4x</option>
    </select>
  </div>
  <img id="replayImg" style="max-width:100%; max-height:55vh; background:#000;">
  <div class="row">
    <input type="range" id="seekBar" min="0" max="1000" step="1" value="0"
           oninput="onSeekInput()" onchange="onSeekChange()" style="flex:1;">
    <span id="timeLabel">0:00 / 0:00</span>
  </div>
  <div id="replayMsg" style="font-size:13px; color:#f88;"></div>
</div>"""
        replay_js = """
var rep = {file:null, fps:1, duration:0, t:0, speed:1, playing:false,
           playStart:0, playStartT:0, timer:null, infoOk:false};
function toggleReplay() {
  var p = document.getElementById("replayPanel");
  var show = (p.style.display === "none" || p.style.display === "");
  p.style.display = show ? "block" : "none";
  if (show) loadReplayList(); else stopPlayback();
}
function loadReplayList() {
  fetch("/api/recordings").then(function(r){ return r.json(); }).then(function(d){
    var sel = document.getElementById("recSelect");
    sel.innerHTML = "";
    if (!d.ok || !d.files || !d.files.length) { setMsg("暂无录像（或回放未启用）"); return; }
    d.files.forEach(function(f){
      var o = document.createElement("option");
      o.value = f.name;
      o.textContent = f.name + "  (" + fmtSize(f.size) + ")";
      sel.appendChild(o);
    });
    sel.onchange = selectReplay;
    selectReplay();
  });
}
function selectReplay() {
  rep.file = document.getElementById("recSelect").value;
  stopPlayback();
  fetch("/api/replay/info?file=" + encodeURIComponent(rep.file)).then(function(r){ return r.json(); }).then(function(info){
    if (!info.ok) {
      rep.infoOk = false;
      setMsg("无法读取该文件（可能正在写入或已损坏）");
      document.getElementById("playBtn").disabled = true;
      return;
    }
    rep.infoOk = true; rep.fps = info.fps; rep.duration = info.duration; rep.t = 0;
    document.getElementById("playBtn").disabled = false;
    document.getElementById("seekBar").max = Math.round(info.duration * 1000);
    document.getElementById("seekBar").value = 0;
    document.getElementById("replayImg").src = "/replay/frame?file=" + encodeURIComponent(rep.file) + "&t=0";
    updateTime();
  });
}
function playPause() {
  if (!rep.infoOk) return;
  if (rep.playing) { pauseReplay(); return; }
  if (rep.t >= rep.duration - 0.05) rep.t = 0;
  rep.playing = true; rep.playStart = performance.now(); rep.playStartT = rep.t;
  document.getElementById("playBtn").textContent = "暂停";
  document.getElementById("replayImg").src = "/replay?file=" + encodeURIComponent(rep.file) + "&t=" + rep.t.toFixed(3) + "&speed=" + rep.speed;
  rep.timer = setInterval(tickReplay, 100);
}
function pauseReplay() {
  clearInterval(rep.timer); rep.playing = false;
  document.getElementById("playBtn").textContent = "播放";
  document.getElementById("replayImg").src = "/replay/frame?file=" + encodeURIComponent(rep.file) + "&t=" + rep.t.toFixed(3);
  updateTime();
}
function tickReplay() {
  rep.t = rep.playStartT + (performance.now() - rep.playStart) / 1000 * rep.speed;
  if (rep.t >= rep.duration) { rep.t = rep.duration; endReplay(); return; }
  document.getElementById("seekBar").value = Math.round(rep.t * 1000);
  updateTime();
}
function endReplay() {
  clearInterval(rep.timer); rep.playing = false;
  document.getElementById("playBtn").textContent = "播放";
  document.getElementById("replayImg").src = "/replay/frame?file=" + encodeURIComponent(rep.file) + "&t=" + rep.duration.toFixed(3);
  document.getElementById("seekBar").value = document.getElementById("seekBar").max;
  updateTime();
}
function onSeekInput() {
  var v = document.getElementById("seekBar").value / 1000;
  document.getElementById("timeLabel").textContent = fmtTime(v) + " / " + fmtTime(rep.duration);
}
function onSeekChange() {
  rep.t = document.getElementById("seekBar").value / 1000;
  if (rep.playing) { restartStream(); } else {
    document.getElementById("replayImg").src = "/replay/frame?file=" + encodeURIComponent(rep.file) + "&t=" + rep.t.toFixed(3);
  }
  updateTime();
}
function restartStream() {
  rep.playStart = performance.now(); rep.playStartT = rep.t;
  document.getElementById("replayImg").src = "/replay?file=" + encodeURIComponent(rep.file) + "&t=" + rep.t.toFixed(3) + "&speed=" + rep.speed;
}
function onSpeed() {
  rep.speed = parseFloat(document.getElementById("speedSel").value);
  if (rep.playing) restartStream();
}
function stopPlayback() {
  if (rep.playing) { clearInterval(rep.timer); rep.playing = false; }
  document.getElementById("playBtn").textContent = "播放";
  document.getElementById("replayImg").src = "";
}
function fmtSize(n){ return n > 1048576 ? (n/1048576).toFixed(1) + "MB" : Math.round(n/1024) + "KB"; }
function fmtTime(t){ t = Math.max(0, t); return Math.floor(t/60) + ":" + String(Math.floor(t%60)).padStart(2, "0"); }
function updateTime(){ document.getElementById("timeLabel").textContent = fmtTime(rep.t) + " / " + fmtTime(rep.duration); }
function setMsg(s){ document.getElementById("replayMsg").textContent = s; }
"""
        return f"""<!doctype html><html><head><meta charset='utf-8'>
<title>RollBall 实时画面</title>
<style>
  body {{ margin:0; background:#111; color:#eee; font-family:sans-serif; }}
  .views {{ display:flex; flex-wrap:wrap; justify-content:center; gap:4px; }}
  .views img {{ max-width:100%; max-height:70vh; }}
  #btnBar {{ position:fixed; top:8px; right:8px; z-index:10; display:flex; gap:8px; }}
  #btnBar button {{ padding:8px 14px; font-size:14px; }}
  #panel {{ display:none; background:#1c1c1c; padding:12px 16px; }}
  #panel .row {{ display:flex; align-items:center; gap:8px; margin:6px 0; }}
  #panel label {{ width:110px; font-size:13px; }}
  #panel input[type=range] {{ flex:1; }}
  #panel span {{ width:48px; text-align:right; font-size:13px; }}
  #saveBtn {{ margin-top:10px; padding:8px 14px; font-size:14px; }}
  #saveMsg {{ margin-left:10px; font-size:13px; color:#8f8; }}
  #docPanel {{ display:none; background:#1c1c1c; padding:12px 20px; max-width:720px;
               margin:0 auto; line-height:1.6; font-size:13px; }}
  #docPanel h3 {{ color:#9cf; margin:16px 0 6px; }}
  #docPanel h3:first-child {{ margin-top:0; }}
  #docPanel b {{ color:#fd6; }}
  #docPanel p {{ margin:4px 0 10px; }}
</style></head>
<body>
<div id="btnBar">
  <button id="docBtn" onclick="toggleDoc()">使用说明</button>
  <button id="toggleBtn" onclick="togglePanel()">调参数</button>
  <button id="replayBtn" onclick="toggleReplay()">回放</button>
</div>
<div class="views">
  <img src="/stream">
  <img src="/stream/bin">
</div>
<div id="docPanel">
  {self.doc_html}
</div>
<div id="panel">
  {sliders}
  <div class="row">
    <button id="saveBtn" onclick="saveParams()">保存参数（永久写入配置文件）</button>
    <span id="saveMsg"></span>
  </div>
</div>
{replay_panel}
<script>
function togglePanel() {{
  var p = document.getElementById("panel");
  p.style.display = (p.style.display === "none" || p.style.display === "") ? "block" : "none";
}}
function toggleDoc() {{
  var d = document.getElementById("docPanel");
  d.style.display = (d.style.display === "none" || d.style.display === "") ? "block" : "none";
}}
var debounceTimers = {{}};
function onSlide(name) {{
  var el = document.getElementById("p_" + name);
  document.getElementById("v_" + name).textContent = el.value;
  clearTimeout(debounceTimers[name]);
  debounceTimers[name] = setTimeout(function() {{
    var body = {{}};
    body[name] = Number(el.value);
    fetch("/api/params", {{method: "POST", body: JSON.stringify(body)}});
  }}, 150);
}}
function saveParams() {{
  var msg = document.getElementById("saveMsg");
  msg.textContent = "保存中…";
  fetch("/api/save", {{method: "POST"}}).then(function(r) {{ return r.json(); }}).then(function(data) {{
    msg.textContent = data.ok ? "已保存 ✓" : "保存失败";
    setTimeout(function() {{ msg.textContent = ""; }}, 2000);
  }}).catch(function() {{
    msg.textContent = "保存失败";
  }});
}}
window.addEventListener("DOMContentLoaded", function() {{
  document.querySelectorAll('#panel input[type=range]').forEach(function(el) {{
    var name = el.id.slice(2);
    el.addEventListener("input", function() {{ onSlide(name); }});
  }});
  fetch("/api/params").then(function(r) {{ return r.json(); }}).then(function(data) {{
    for (var name in data) {{
      var el = document.getElementById("p_" + name);
      if (!el) continue;
      el.value = data[name];
      var v = document.getElementById("v_" + name);
      if (v) v.textContent = data[name];
    }}
  }});
}});
{replay_js}
</script>
</body></html>"""

    def update_frame(self, bgr):
        """主循环调用【非阻塞】，主画面(彩色检测叠加层)。"""
        self.main.update(bgr)

    def update_bin_frame(self, bgr):
        """主循环调用【非阻塞】，球体二值化掩膜(建议先 cvtColor 成 BGR 再传)。"""
        self.bin.update(bgr)

    def _replay_started(self, reader, stream):
        with self._replay_lock:
            if reader is not None:
                self._replay_readers.add(reader)
            self._replay_streams.add(stream)

    def _replay_finished(self, reader, stream):
        with self._replay_lock:
            if reader is not None:
                self._replay_readers.discard(reader)
            self._replay_streams.discard(stream)

    def start(self):
        self.main.start()
        self.bin.start()
        self._httpd = _Server(("0.0.0.0", self.port), _Handler)
        self._httpd.mjpeg = self
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def stop(self):
        if self._httpd is None:
            return
        self.main.stop()
        self.bin.stop()
        # 收掉仍挂着的回放连接：先停 reader（join 再 release cap），再停 per-connection 流
        # （_jpg=None 让 handler 的 wait_frame 退出），让每个回放线程都干净收场而不是被强杀。
        with self._replay_lock:
            readers = list(self._replay_readers)
            streams = list(self._replay_streams)
            self._replay_readers.clear()
            self._replay_streams.clear()
        for rd in readers:
            rd.stop()
        for st in streams:
            st.stop()
        self._httpd.shutdown()
        self._httpd.server_close()
        self._httpd = None
