#!/usr/bin/env python3
"""
题目 V1.0 —— 实时检测球 + 把球心 X/Y 通过串口发给下位机
环境: /home/hao/vision_env/bin/python3
用法:
    /home/hao/vision_env/bin/python3 code/task_code/v1.0.py [--size 320|416|640]
    #   --size 320      初始模型规格(默认 320，最流畅；运行时 1/2/3 可切)
    #   --conf 0.25     置信度阈值
    #   --any           串口找不到固定口时放宽到任意 CH340/ttyUSB
    #   --no-serial     只看检测、不发串口(纯视觉调试)

功能：
  - 复用 code/ready_code/camera_common.py 固定曝光/增益/白平衡 + 【默认去畸变】(项目约定)；
  - 复用 code/detect_code/yolo_ncnn.py 跑 NCNN 检测，画框；
  - 复用 serial_link.py 锁定固定物理 USB 口、热插拔自动重连；
  - 每帧选【面积最大的球】作为主目标，按 protocol.py 的格式发一帧 X/Y。
    通信格式见 protocol.py（$BALL,found,x,y,n*CHK\\r\\n）。

按键:
  1 / 2 / 3 : 切换模型规格 320 / 416 / 640
  u         : 切换 去畸变 / 原始画面
  q / ESC   : 退出
需在有显示器/桌面的会话里跑(有 cv2.imshow 窗口)。
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
sys.path.insert(0, os.path.join(_CODE, "detect_code"))
sys.path.insert(0, _HERE)

import camera_common as cc          # noqa: E402
from yolo_ncnn import YoloNcnn, draw_detections, MODELS_DIR  # noqa: E402
from serial_link import SerialLink, Disconnected            # noqa: E402
from camera_link import CameraLink, CameraLost              # noqa: E402
import protocol                     # noqa: E402

SIZES = (320, 416, 640)


def get_model(cache, size):
    if size not in cache:
        print(f"   载入模型 best_ncnn_{size} ...")
        cache[size] = YoloNcnn(os.path.join(MODELS_DIR, f"best_ncnn_{size}"))
    return cache[size]


def pick_primary(dets):
    """从检测结果里选主目标=面积最大的球，返回 (cx, cy) 或 None。"""
    if not dets:
        return None
    best = max(dets, key=lambda d: (d[2] - d[0]) * (d[3] - d[1]))
    x1, y1, x2, y2, _conf = best
    return (x1 + x2) // 2, (y1 + y2) // 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=320, choices=SIZES)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--any", action="store_true", help="串口放宽到任意 CH340/ttyUSB")
    ap.add_argument("--no-serial", action="store_true", help="不发串口，只看检测")
    args = ap.parse_args()

    cam = CameraLink()            # 锁定 /dev/video0，热插拔自动重连
    w, h = cam.wait_and_open()    # 没插摄像头就阻塞等待，插上自动连
    print("   (MJPG, 固定曝光/增益/白平衡)")

    # 默认去畸变。maps 依赖分辨率，重连后若分辨率变则重建。
    calib = cc.load_calibration()

    def build_maps(size):
        if calib is None:
            return None
        return cc.build_undistort_maps(
            calib["camera_matrix"], calib["dist_coeffs"], size, alpha=0.0)

    maps = build_maps((w, h))
    if calib is not None:
        print(f"   已载入标定 (RMS={calib['rms']:.3f}px)，默认去畸变。")
    else:
        print("   ⚠️ 未找到 camera_calib.npz，用原始画面。")
    undistort_on = maps is not None

    cache = {}
    cur_size = args.size
    det = get_model(cache, cur_size)
    det.conf_thres = args.conf

    link = None
    if not args.no_serial:
        link = SerialLink(any_ok=args.any)   # 锁定固定物理口，非阻塞重连
        link.try_open()                      # 先尝试连一次（连不上不阻塞，循环里再试）
    print(f"   当前模型: {cur_size}   [1/2/3]切规格  [u]去畸变  [q]退出")
    print(f"   串口: {'关闭(--no-serial)' if link is None else '开，' + protocol.build_ball_frame(True,320,240,1).decode().strip() + ' 这类帧'}")

    win = "Task V1.0  detect+serial"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, w, h)

    fps_ema = None
    tx_count = 0
    last_frame_str = ""
    while True:
        # 摄像头掉线：非阻塞重连；没连上就显示占位画面并保持按键响应
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
        if frame is None:            # 偶发单帧失败，跳过
            continue

        if undistort_on and maps is not None:
            frame = cv2.remap(frame, maps[0], maps[1], cv2.INTER_LINEAR)

        t0 = time.perf_counter()
        dets, timing = det.detect(frame)
        loop_ms = (time.perf_counter() - t0) * 1e3
        fps = 1000.0 / loop_ms if loop_ms > 0 else 0.0
        fps_ema = fps if fps_ema is None else 0.9 * fps_ema + 0.1 * fps

        center = pick_primary(dets)
        found = center is not None
        cx, cy = center if found else (0, 0)

        # 发串口（每帧一帧；断开则本帧跳过，下帧非阻塞重连）
        serial_hud = "OFF"
        if link is not None:
            if link.try_open():
                out = protocol.build_ball_frame(found, cx, cy, len(dets))
                last_frame_str = out.decode(errors="replace").strip()
                try:
                    link.write(out)
                    tx_count += 1
                    serial_hud = f"TX#{tx_count}"
                except Disconnected:
                    serial_hud = "DISCONN"
            else:
                serial_hud = "WAIT"

        # 画面标注
        draw_detections(frame, dets)
        if found:
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(frame, f"({cx},{cy})", (cx + 8, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        hud = (f"size={cur_size} {fps_ema:4.1f}FPS det={len(dets)} "
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
                undistort_on = not undistort_on
                print(f"   切换为: {'去畸变' if undistort_on else '原始'}")
        elif key in (ord("1"), ord("2"), ord("3")):
            new_size = SIZES[key - ord("1")]
            if new_size != cur_size:
                cur_size = new_size
                det = get_model(cache, cur_size)
                det.conf_thres = args.conf
                fps_ema = None
                print(f"   切换模型: {cur_size}")

    cam.close()
    cv2.destroyAllWindows()
    for m in cache.values():
        m.release()
    if link is not None:
        link.close()
    print(f"已退出。共发送 {tx_count} 帧串口数据。")


if __name__ == "__main__":
    main()
