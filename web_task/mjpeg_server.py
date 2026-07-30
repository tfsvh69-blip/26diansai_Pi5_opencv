#!/usr/bin/env python3
"""
零依赖 MJPEG 局域网推流服务 —— 只用标准库 http.server，不装 flask
环境: /home/hao/vision_env/bin/python3

用法（被 v1.0.py import，不单独运行）：
    import mjpeg_server
    mjpeg_server.start_server(port=8000)   # 起一个 daemon 线程跑 HTTP 服务
    ...主循环里每帧...
    mjpeg_server.push_frame(frame)          # frame 是 BGR ndarray，已画好检测框/HUD

浏览器/手机打开 http://<树莓派IP>:<port>/ 看极简网页（<img src="/stream.mjpg">），
或者直接访问 /stream.mjpg 看 multipart/x-mixed-replace 原始流。

线程安全：_FrameStore 用锁保护最新一帧 JPEG bytes；ThreadingHTTPServer 允许多个
客户端同时连接，各自独立读最新帧，互不阻塞。服务线程是 daemon，主程序退出时
随进程一起结束，不用显式 shutdown。
"""

import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

BOUNDARY = "frame"

_INDEX_HTML = b"""<!doctype html>
<html><head><meta charset="utf-8"><title>Ball Detect Live</title>
<style>body{margin:0;background:#111;display:flex;justify-content:center;align-items:center;height:100vh}
img{max-width:100%;max-height:100%}</style></head>
<body><img src="/stream.mjpg"></body></html>
"""


class _FrameStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._jpeg = None

    def update(self, jpeg_bytes):
        with self._lock:
            self._jpeg = jpeg_bytes

    def get(self):
        with self._lock:
            return self._jpeg


_store = _FrameStore()


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # 静默，不然每帧访问都刷屏

    def do_GET(self):
        if self.path == "/stream.mjpg":
            self._serve_stream()
        elif self.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(_INDEX_HTML)))
            self.end_headers()
            self.wfile.write(_INDEX_HTML)
        else:
            self.send_response(404)
            self.end_headers()

    def _serve_stream(self):
        self.send_response(200)
        self.send_header(
            "Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY}"
        )
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.end_headers()
        try:
            while True:
                jpeg = _store.get()
                if jpeg is None:
                    time.sleep(0.05)
                    continue
                self.wfile.write(f"--{BOUNDARY}\r\n".encode())
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
                time.sleep(0.03)  # 上限约 ~30fps 推送，避免空转占满 CPU
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass  # 客户端关掉页面/断网，正常现象，不用报错


def start_server(port=8000):
    """起后台 daemon 线程跑 HTTP 服务，返回 server 实例（一般不需要用到）。"""
    server = ThreadingHTTPServer(("0.0.0.0", port), _Handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server


def push_frame(bgr_frame, quality=80):
    """把一帧 BGR ndarray 编码成 JPEG 存入共享缓冲，供 /stream.mjpg 推给所有客户端。"""
    ok, buf = cv2.imencode(".jpg", bgr_frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if ok:
        _store.update(buf.tobytes())
