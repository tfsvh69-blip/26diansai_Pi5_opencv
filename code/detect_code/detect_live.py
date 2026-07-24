#!/usr/bin/env python3
"""
YOLO11n NCNN 实时检测预览
环境: /home/hao/vision_env/bin/python3
用法:
    /home/hao/vision_env/bin/python3 code/detect_code/detect_live.py [--size 320|416|640]
    默认 --size 320（树莓派上最流畅；按 1/2/3 可运行时切换三种规格对比）。

遵循本项目约定：
  - 复用 code/ready_code/camera_common.py 的固定曝光/增益/白平衡参数（唯一事实来源）；
  - 【默认去畸变】：有 camera_calib.npz 就默认喂去畸变画面（与 v1.2 / record_dataset 一致，
    部署与训练几何一致），按 u 可临时切回原始对比；
  - 需在有显示器/桌面的会话里跑（有 cv2.imshow 窗口），纯 SSH 无 X 转发会失败。

按键:
  1 / 2 / 3 : 切换模型规格 320 / 416 / 640
  u         : 切换 去畸变 / 原始画面
  q / ESC   : 退出
"""

import argparse
import os
import sys
import time

import cv2

# 复用 ready_code 下的公共模块（固定相机参数、开摄像头、去畸变映射）
READY_CODE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ready_code"
)
sys.path.insert(0, READY_CODE)
import camera_common as cc  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yolo_ncnn import YoloNcnn, draw_detections, MODELS_DIR  # noqa: E402

SIZES = (320, 416, 640)


def get_model(cache, size):
    """按需加载并缓存指定规格的模型。"""
    if size not in cache:
        print(f"   载入模型 best_ncnn_{size} ...")
        cache[size] = YoloNcnn(os.path.join(MODELS_DIR, f"best_ncnn_{size}"))
    return cache[size]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=320, choices=SIZES,
                    help="初始模型输入规格 (默认 320)")
    ap.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    args = ap.parse_args()

    cap = cc.open_camera()
    if cap is None:
        print("❌ 无法打开摄像头 /dev/video0")
        return
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"✅ 摄像头已打开: {actual_w}x{actual_h} (MJPG, 固定曝光/增益/白平衡)")

    # 默认去畸变（有标定就开）
    calib = cc.load_calibration()
    maps = None
    if calib is not None:
        maps = cc.build_undistort_maps(
            calib["camera_matrix"], calib["dist_coeffs"], (actual_w, actual_h), alpha=0.0
        )
        print(f"   已载入标定 (RMS={calib['rms']:.3f}px)，默认去畸变。")
    else:
        print("   ⚠️ 未找到 camera_calib.npz，只能用原始画面。")
    undistort_on = maps is not None

    cache = {}
    cur_size = args.size
    det = get_model(cache, cur_size)
    det.conf_thres = args.conf
    print(f"   当前模型: {cur_size}   [1/2/3]切换规格  [u]去畸变  [q]退出")

    win = "YOLO NCNN Detect"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, actual_w, actual_h)

    fps_ema = None  # 端到端 FPS 的指数滑动平均
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ 读取帧失败")
            break
        if undistort_on and maps is not None:
            frame = cv2.remap(frame, maps[0], maps[1], cv2.INTER_LINEAR)

        t0 = time.perf_counter()
        dets, timing = det.detect(frame)
        loop_ms = (time.perf_counter() - t0) * 1e3
        fps = 1000.0 / loop_ms if loop_ms > 0 else 0.0
        fps_ema = fps if fps_ema is None else 0.9 * fps_ema + 0.1 * fps

        draw_detections(frame, dets)
        hud = (f"size={cur_size}  {fps_ema:4.1f} FPS  "
               f"infer={timing['infer_ms']:.0f}ms  det={len(dets)}  "
               f"{'UNDIST' if undistort_on else 'RAW'}")
        cv2.putText(frame, hud, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 255), 2, cv2.LINE_AA)

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

    cap.release()
    cv2.destroyAllWindows()
    for m in cache.values():
        m.release()
    print("已退出")


if __name__ == "__main__":
    main()
