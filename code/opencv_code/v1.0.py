#!/usr/bin/env python3
"""
题目 V1.0（纯 OpenCV 版）—— 实时检测小钢珠 + 把球心 X/Y 经串口发给下位机
环境: /home/hao/vision_env/bin/python3
用法:
    /home/hao/vision_env/bin/python3 code/opencv_code/v1.0.py
      --any         串口找不到固定口时放宽到任意 CH340/ttyUSB
      --no-serial   只看检测、不发串口（纯视觉调试）

为什么有这个版本：YOLO NCNN 只有 ~17FPS，太慢。钢珠是镜面金属圆球，用
【HoughCircles 找圆 + 金属高光/对比度确认】即可，纯检测 ~6ms(~160FPS)，
相机 30FPS 满帧无压力。检测算法在 ball_detector.py（唯一事实来源，含迭代记录）。

复用现有框架（与 task_code 一致，下位机无需改）：
  - camera_common.py 固定曝光/增益/白平衡 + 【默认去畸变】（项目约定）；
  - task_code/camera_link.py / serial_link.py 热插拔自动重连；
  - task_code/protocol.py 通信格式 $BALL,found,x,y,n*CHK（主目标=分最高的球）。

参数持久化：检测参数与是否去畸变存 detector_config.json；先用 tune_detector.py
现场调好保存，本程序启动自动载入。运行时按 u 切去畸变会自动保存。

按键:  u=切换 去畸变/原始    q/ESC=退出
需在有显示器/桌面的会话里跑（有 cv2.imshow 窗口）。
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_CODE, "ready_code"))
sys.path.insert(0, os.path.join(_CODE, "task_code"))
sys.path.insert(0, _HERE)

import camera_common as cc                          # noqa: E402
from camera_link import CameraLink, CameraLost      # noqa: E402
from serial_link import SerialLink, Disconnected    # noqa: E402
import protocol                                     # noqa: E402
from ball_detector import BallDetector, draw        # noqa: E402
import config                                       # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--any", action="store_true", help="串口放宽到任意 CH340/ttyUSB")
    ap.add_argument("--no-serial", action="store_true", help="不发串口，只看检测")
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
    print(f"   串口: {'关闭(--no-serial)' if link is None else '开'}   [u]去畸变  [q]退出")

    win = "OpenCV Ball  detect+serial"
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
               f"{'UNDIST' if undistort_on else 'RAW'} | serial:{serial_hud}")
        cv2.putText(frame, hud, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 255), 2, cv2.LINE_AA)
        if last_frame_str:
            cv2.putText(frame, last_frame_str, (10, h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

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
