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
    def __init__(self, port=8080, quality=80):
        self.port = port
        self.main = _Stream(quality=quality, placeholder_text="waiting for camera...")
        self.bin = _Stream(quality=quality, placeholder_text="")
        self._httpd = None
        self._thread = None

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
</script>
</body></html>"""

    def update_frame(self, bgr):
        """主循环调用【非阻塞】，主画面(彩色检测叠加层)。"""
        self.main.update(bgr)

    def update_bin_frame(self, bgr):
        """主循环调用【非阻塞】，球体二值化掩膜(建议先 cvtColor 成 BGR 再传)。"""
        self.bin.update(bgr)

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
        self._httpd.shutdown()
        self._httpd.server_close()
        self._httpd = None
