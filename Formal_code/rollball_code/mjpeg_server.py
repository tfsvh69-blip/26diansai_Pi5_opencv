#!/usr/bin/env python3
"""
局域网 MJPEG 实时画面服务 —— 浏览器打开 http://<本机局域网IP>:<port>/ 即可看当前摄像头画面。
被 v1.1_beta.py import 使用，不单独运行。

只用标准库 http.server(ThreadingHTTPServer) + cv2.imencode 编码 JPEG，不引入 Flask 等额外依赖
（比赛现场树莓派 vision_env 不一定装了 Flask，标准库最省心）。

用法（主循环里）：
    server = MjpegServer(port=8080)
    server.start()
    print(f"http://{get_lan_ip()}:{server.port}/")
    ...
    if server.has_clients:               # 没人看就不用浪费 CPU 编码/画叠加层
        server.update_frame(overlay_bgr)  # 主循环按需调用，推一帧新画面
    ...
    server.stop()                         # 退出时收尾
"""

import socket
import threading
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


def _placeholder_jpeg(w=640, h=480, quality=80):
    """服务刚启动、主循环还没推第一帧时的占位画面，避免客户端一连上就因为没帧而断开。"""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(img, "waiting for camera...", (max(10, w // 2 - 150), h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return buf.tobytes() if ok else b""


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
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={_BOUNDARY.decode()}")
            self.end_headers()
            mjpeg._client_connected()
            try:
                last_id = -1
                while True:
                    jpg, last_id = mjpeg._wait_frame(last_id)
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
                mjpeg._client_disconnected()
            return
        self.send_response(404)
        self.end_headers()


class _Server(ThreadingHTTPServer):
    daemon_threads = True  # 主程序退出时不用等还在推流的客户端线程


class MjpegServer:
    def __init__(self, port=8080, quality=80):
        self.port = port
        self.quality = quality
        # _cond/_lock：保护【已编码好的 JPEG】(_jpg/_frame_id)，推流处理线程在此等待新帧。
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._jpg = _placeholder_jpeg(quality=quality)
        self._frame_id = 0
        self._clients = 0
        self._httpd = None
        self._thread = None
        # 【性能】JPEG 编码(cv2.imencode)从主循环搬到这个后台【编码线程】：主循环 update_frame
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

    def _client_connected(self):
        with self._lock:
            self._clients += 1

    def _client_disconnected(self):
        with self._lock:
            self._clients = max(0, self._clients - 1)

    def _wait_frame(self, last_id, timeout=2.0):
        with self._cond:
            got = self._cond.wait_for(lambda: self._frame_id != last_id, timeout=timeout)
            if not got:
                return self._jpg, last_id  # 超时心跳：重发上一帧，防止中间代理判定连接已死
            return self._jpg, self._frame_id

    def index_html(self):
        return (
            "<!doctype html><html><head><meta charset='utf-8'>"
            "<title>RollBall 实时画面</title></head>"
            "<body style=\"margin:0;background:#111;display:flex;"
            "justify-content:center;align-items:center;height:100vh;\">"
            "<img src='/stream' style='max-width:100%;max-height:100%;'>"
            "</body></html>"
        )

    def update_frame(self, bgr):
        """
        主循环调用【非阻塞】：只存下最新一帧的引用 + 唤醒后台编码线程，imencode 由编码线程做。
        latest-wins：编码线程忙不过来时，中间帧被覆盖丢弃（预览无所谓），不阻塞主循环。
        传进来的 bgr 是主循环里 build_detection_overlay 产出的独立副本，跨线程只读安全。
        """
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
        self._httpd = _Server(("0.0.0.0", self.port), _Handler)
        self._httpd.mjpeg = self
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def stop(self):
        if self._httpd is None:
            return
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
        self._httpd.shutdown()
        self._httpd.server_close()
        self._httpd = None
