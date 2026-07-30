#!/usr/bin/env python3
"""
局域网网页预览 V1.0 —— 纯 OpenCV 检测小钢珠 + 串口发下位机 + 本地窗口 + 网页 MJPEG 同时预览
环境: /home/hao/vision_env/bin/python3
用法:
    /home/hao/vision_env/bin/python3 web_task/v1.0.py
      --port 8000   网页服务端口（默认 8000）
      --any         串口找不到固定口时放宽到任意 CH340/ttyUSB
      --no-serial   只看检测+网页预览、不发串口

在 code/opencv_code/v1.0.py 已跑通的串口链路基础上，加一路网页 MJPEG 预览。
检测算法、相机热插拔、去畸变、串口热插拔、通信协议、参数持久化全部复用
opencv_code / ready_code / task_code 现有模块，不改一行。

同一局域网内的手机/电脑浏览器打开脚本启动时打印的地址（http://<树莓派IP>:端口/）
即可看到与本地 imshow 窗口一致的实时检测画面（含检测框和坐标 HUD）。

按键:  u=切换 去畸变/原始    q/ESC=退出
本地窗口需在有显示器/桌面的会话里跑（有 cv2.imshow 窗口）；网页预览不受此限制。
"""

import argparse
import os
import socket
import sys
import time

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "code", "ready_code"))
sys.path.insert(0, os.path.join(_ROOT, "code", "task_code"))
sys.path.insert(0, os.path.join(_ROOT, "code", "opencv_code"))
sys.path.insert(0, _HERE)

import camera_common as cc                          # noqa: E402
from camera_link import CameraLink, CameraLost       # noqa: E402
from serial_link import SerialLink, Disconnected     # noqa: E402
import protocol                                      # noqa: E402
from ball_detector import BallDetector, draw         # noqa: E402
import config                                        # noqa: E402
import mjpeg_server                                  # noqa: E402


def _local_ip():
    """猜一个局域网 IP 给用户看；拿不到就退化提示手动查。"""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return None
    finally:
        s.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000, help="网页服务端口（默认 8000）")
    ap.add_argument("--any", action="store_true", help="串口放宽到任意 CH340/ttyUSB")
    ap.add_argument("--no-serial", action="store_true", help="不发串口，只看检测+网页预览")
    args = ap.parse_args()

    cfg = config.load()
    det = BallDetector().load_dict(cfg["detector"])
    print(f"⚙️  配置: {config.CONFIG_FILE}")
    print(f"    去畸变={cfg['undistort']}  param2={det.param2} r=[{det.min_radius},{det.max_radius}]")

    cam = CameraLink()
    w, h = cam.wait_and_open()
    print("   (MJPG, 固定曝光/增益/白平衡)")

    calib = cc.load_calibration()

    def build_maps(size):
        if calib is None:
            return None
        return cc.build_undistort_maps(
            calib["camera_matrix"], calib["dist_coeffs"], size, alpha=0.0)

    maps = build_maps((w, h))
    if calib is not None:
        print(f"   已载入标定 (RMS={calib['rms']:.3f}px)。")
    else:
        print("   ⚠️ 未找到 camera_calib.npz，只能用原始画面。")
    undistort_pref = cfg["undistort"]
    undistort_on = undistort_pref and maps is not None

    def persist():
        cfg["undistort"] = undistort_pref
        cfg["detector"] = det.as_dict()
        config.save(cfg)

    link = None
    if not args.no_serial:
        link = SerialLink(any_ok=args.any)
        link.try_open()
    print(f"   串口: {'关闭(--no-serial)' if link is None else '开'}")

    mjpeg_server.start_server(args.port)
    ip = _local_ip()
    if ip:
        print(f"   🌐 网页预览: http://{ip}:{args.port}/")
    else:
        print(f"   🌐 网页预览: http://<本机IP>:{args.port}/  (查 IP 用 hostname -I)")
    print("   [u]去畸变  [q]退出")

    win = "OpenCV Ball  detect+serial+web"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, w, h)

    fps_ema = None
    tx_count = 0
    last_frame_str = ""
    while True:
        if not cam.connected:
            if cam.try_open():
                nw, nh = cam.size
                if (nw, nh) != (w, h):
                    w, h = nw, nh
                    maps = build_maps((w, h))
                    cv2.resizeWindow(win, w, h)
                print("✅ 摄像头已重连")
            else:
                placeholder = np.zeros((h, w, 3), dtype=np.uint8)
                cv2.putText(placeholder, "waiting for camera...",
                            (20, h // 2), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0, 0, 255), 2, cv2.LINE_AA)
                mjpeg_server.push_frame(placeholder)
                cv2.imshow(win, placeholder)
                if (cv2.waitKey(100) & 0xFF) in (ord("q"), 27):
                    break
                continue

        try:
            frame = cam.read()
        except CameraLost:
            print("⚠️ 摄像头掉线，等待重连…")
            continue
        if frame is None:
            continue

        if undistort_on and maps is not None:
            frame = cv2.remap(frame, maps[0], maps[1], cv2.INTER_LINEAR)

        t0 = time.perf_counter()
        cands = det.detect(frame)
        prim = det.pick_primary(cands)
        loop_ms = (time.perf_counter() - t0) * 1e3
        fps = 1000.0 / loop_ms if loop_ms > 0 else 0.0
        fps_ema = fps if fps_ema is None else 0.9 * fps_ema + 0.1 * fps

        found = prim is not None
        cx, cy = (prim[0], prim[1]) if found else (0, 0)

        serial_hud = "OFF"
        if link is not None:
            if link.try_open():
                out = protocol.build_ball_frame(found, cx, cy, len(cands))
                last_frame_str = out.decode(errors="replace").strip()
                try:
                    link.write(out)
                    tx_count += 1
                    serial_hud = f"TX#{tx_count}"
                except Disconnected:
                    serial_hud = "DISCONN"
            else:
                serial_hud = "WAIT"

        draw(frame, cands, prim)
        conf_pct = int(det._confidence * 100) if hasattr(det, '_confidence') else 0
        hud = (f"{fps_ema:5.1f}FPS det={len(cands)} "
               f"CONF={conf_pct}% "
               f"{'UNDIST' if undistort_on else 'RAW'} | serial:{serial_hud} | web:{args.port}")
        cv2.putText(frame, hud, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 255), 2, cv2.LINE_AA)
        if last_frame_str:
            cv2.putText(frame, last_frame_str, (10, h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        mjpeg_server.push_frame(frame)
        cv2.imshow(win, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("u"):
            if maps is None:
                print("   无标定，无法切去畸变。")
            else:
                undistort_pref = not undistort_pref
                undistort_on = undistort_pref
                persist()
                print(f"   切换为: {'去畸变' if undistort_on else '原始'}（已保存）")

    persist()
    cam.close()
    cv2.destroyAllWindows()
    if link is not None:
        link.close()
    print(f"已退出。共发送 {tx_count} 帧串口数据。设置已保存到 {os.path.basename(config.CONFIG_FILE)}。")


if __name__ == "__main__":
    main()
