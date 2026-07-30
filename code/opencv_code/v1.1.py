#!/usr/bin/env python3
"""
题目 V1.1（纯 OpenCV 版）—— 在 v1.0 的 Hough 检测上叠加 BoT-SORT(无 ReID)+GMC+
卡尔曼滤波+One Euro Filter，解决"能测到、但很不连续"的问题
环境: /home/hao/vision_env/bin/python3
用法:
    /home/hao/vision_env/bin/python3 code/opencv_code/v1.1.py
      --any         串口找不到固定口时放宽到任意 CH340/ttyUSB
      --no-serial   只看检测、不发串口（纯视觉调试）
      --no-gmc      关闭相机运动补偿
      --no-euro     关闭 One Euro Filter，直接发跟踪原始坐标

不动 v1.0.py / ball_detector.py：本文件只是【新增】一个使用方式，检测核心
（HoughCircles + 多因素评分 + NMS，唯一事实来源 ball_detector.py）完全复用、
一行没改；v1.0.py 也原样保留作对比。

为什么"能测到但不连续"，以及这里怎么解决：
  v1.0 每帧调用 BallDetector.detect() 后直接 pick_primary()=取分最高的候选做
  EMA 平滑。问题在于：
    1. 没有"目标身份"概念——如果画面里同时有两个候选圆（真球 + 一个高光/反光
       误检的假候选），哪个分高完全看当帧噪声，主目标可能来回跳；
    2. HoughCircles 逐帧独立找圆，圆心本身就有几像素的量化抖动，EMA 只是低通，
       抖动应对不算好，而且是"事后"平滑、没用运动模型；
    3. BallDetector 内部虽然有一套速度惯性+置信度衰减的兜底（详见 ball_detector.py
       "情况 B"），但那是【单目标】的、没有多候选关联，也不管相机自己会不会晃。
  这里把 BallDetector.detect() 返回的候选（真实检测时是多个打分候选；置信度
  兜底时是 1 个"假"候选、score=0~1 的置信度值）当成【原始检测框】喂给
  task_code/bot_sort.py 的 BotSort：
    - 用 (cx-r,cy-r,cx+r,cy+r,score) 把圆转成 bbox 接入通用的 ByteTrack式两级
      关联 + GMC + 卡尔曼滤波（细节见 bot_sort.py 顶部注释）；
    - 阈值（HIGH_THRESH/LOW_THRESH，见下）特意设得远高于 1.0——BallDetector 自己
      兜底时返回的 score 是 0~1 的置信度，真实 Hough 候选的评分通常在几十到
      一百出头，两者数量级天然分得开。也就是说 BallDetector 自己的"置信度衰减
      续命"分支会被这里的阈值自然滤掉，不会和 BoT-SORT 的卡尔曼预测互相打架——
      画面上看，target 丢失后接管续命的是 BoT-SORT 的橙色 LOST 框，不是
      BallDetector 内部那套（两者不冲突，但功能上有重叠，说明一下避免看代码时困惑）；
    - 主目标锁定：跟 v1.0 一样"分最高的球"当主目标，但只在【当前没有锁定的
      轨迹】时才重新选，选到之后除非轨迹真丢了（超过宽限期），否则不切换。

配置持久化：复用 v1.0 的 detector_config.json（undistort + detector 参数），
GMC/One Euro 开关是命令行参数，不持久化。

按键: u=切换 去畸变/原始（会重置跟踪器）  r=放弃当前主目标  q/ESC=退出
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
from ball_detector import BallDetector              # noqa: E402
from bot_sort import BotSort, TRACKED               # noqa: E402
from one_euro_filter import OneEuroFilter           # noqa: E402
import config                                       # noqa: E402

# 现场实测纯检测 ~160FPS、端到端 ~89FPS（见 CLAUDE.md），仅用来给"丢失宽限期"
# 换算帧数、给 One Euro Filter 一个采样频率初值，不必精确。
FPS_HINT = 89.0
TRACK_BUFFER_SEC = 1.5      # 轨迹丢失后最多"凭卡尔曼预测续命"多久才真正丢弃

# BallDetector 真实检测评分量级大致 0~120（亮度30+高光30+对比度20+空间一致性40），
# 自身置信度兜底的 score 是 0~1；这两个阈值必须显著大于 1，才能让兜底候选被
# 自然滤掉（见上方模块说明），数值本身现场用 HUD 上的 score 观察后可调。
HIGH_THRESH = 35.0
LOW_THRESH = 12.0

ONE_EURO_MIN_CUTOFF = 1.0   # 越小，球静止/慢动时越平滑
ONE_EURO_BETA = 0.02        # 越大，球快速移动时跟得越紧（抖动也越多）


def to_bbox_dets(cands):
    """BallDetector 的 (cx,cy,r,score) 候选 -> BotSort 要的 (x1,y1,x2,y2,score)。"""
    return [(cx - r, cy - r, cx + r, cy + r, score) for (cx, cy, r, score) in cands]


def make_tracker(use_gmc):
    tracker = BotSort(high_thresh=HIGH_THRESH, low_thresh=LOW_THRESH,
                       frame_rate=FPS_HINT, track_buffer_sec=TRACK_BUFFER_SEC,
                       use_gmc=use_gmc)
    return tracker


def draw_tracks(frame, tracks, primary_id):
    for t in tracks:
        x1, y1, x2, y2 = t.tlbr
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        r = max(1, int(round((x2 - x1 + y2 - y1) / 4.0)))
        is_primary = (t.track_id == primary_id)
        if t.state == TRACKED:
            color = (0, 0, 255) if is_primary else (0, 200, 0)
            thick = 3 if is_primary else 2
            label = f"#{t.track_id} {t.score:.0f}"
        else:
            color = (0, 165, 255)
            thick = 1
            label = f"#{t.track_id} LOST"
        cv2.circle(frame, (int(round(cx)), int(round(cy))), r, color, thick, cv2.LINE_AA)
        cv2.putText(frame, label, (int(cx) + r + 4, int(cy) + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--any", action="store_true", help="串口放宽到任意 CH340/ttyUSB")
    ap.add_argument("--no-serial", action="store_true", help="不发串口，只看检测")
    ap.add_argument("--no-gmc", action="store_true", help="关闭相机运动补偿")
    ap.add_argument("--no-euro", action="store_true", help="关闭 One Euro Filter 平滑")
    args = ap.parse_args()
    use_gmc = not args.no_gmc
    use_euro = not args.no_euro

    cfg = config.load()
    det = BallDetector().load_dict(cfg["detector"])
    print(f"⚙️  配置: {config.CONFIG_FILE}")
    print(f"    去畸变={cfg['undistort']}  param2={det.param2} r=[{det.min_radius},{det.max_radius}]"
          f"  GMC={'开' if use_gmc else '关'}  OneEuro={'开' if use_euro else '关'}")

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

    tracker = make_tracker(use_gmc)
    primary_id = None
    euro_x = OneEuroFilter(freq=FPS_HINT, min_cutoff=ONE_EURO_MIN_CUTOFF, beta=ONE_EURO_BETA)
    euro_y = OneEuroFilter(freq=FPS_HINT, min_cutoff=ONE_EURO_MIN_CUTOFF, beta=ONE_EURO_BETA)

    def rebuild_tracker():
        """坐标系变了（切去畸变/摄像头分辨率变），旧轨迹坐标不再有效，整体重开。"""
        nonlocal tracker, primary_id
        tracker = make_tracker(use_gmc)
        primary_id = None
        euro_x.reset()
        euro_y.reset()

    def persist():
        cfg["undistort"] = undistort_pref
        cfg["detector"] = det.as_dict()
        config.save(cfg)

    link = None
    if not args.no_serial:
        link = SerialLink(any_ok=args.any)
        link.try_open()
    print(f"   串口: {'关闭(--no-serial)' if link is None else '开'}"
          f"   [u]去畸变  [r]放弃当前主目标  [q]退出")

    win = "OpenCV Ball V1.1  detect+BoT-SORT+GMC+Kalman"
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
                    rebuild_tracker()
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
        tracks = tracker.update(to_bbox_dets(cands), frame)
        loop_ms = (time.perf_counter() - t0) * 1e3
        fps = 1000.0 / loop_ms if loop_ms > 0 else 0.0
        fps_ema = fps if fps_ema is None else 0.9 * fps_ema + 0.1 * fps

        n_high = sum(1 for d in cands if d[3] >= HIGH_THRESH)
        max_score = max((d[3] for d in cands), default=0.0)

        by_id = {t.track_id: t for t in tracks}
        if primary_id is not None and primary_id in by_id:
            primary = by_id[primary_id]
        else:
            candidates = [t for t in tracks if t.state == TRACKED]
            primary = max(candidates, key=lambda t: t.score) if candidates else None
            new_id = primary.track_id if primary else None
            if new_id != primary_id:
                euro_x.reset()
                euro_y.reset()
            primary_id = new_id

        found = primary is not None
        if found:
            x1, y1, x2, y2 = primary.tlbr
            raw_cx, raw_cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            if use_euro:
                cx, cy = euro_x(raw_cx), euro_y(raw_cy)
            else:
                cx, cy = raw_cx, raw_cy
            cx, cy = int(round(cx)), int(round(cy))
        else:
            cx, cy = 0, 0

        serial_hud = "OFF"
        if link is not None:
            if link.try_open():
                out = protocol.build_ball_frame(found, cx, cy, n_high)
                last_frame_str = out.decode(errors="replace").strip()
                try:
                    link.write(out)
                    tx_count += 1
                    serial_hud = f"TX#{tx_count}"
                except Disconnected:
                    serial_hud = "DISCONN"
            else:
                serial_hud = "WAIT"

        draw_tracks(frame, tracks, primary_id)
        if found:
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
            cv2.putText(frame, f"({cx},{cy})", (cx + 8, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        gmc_hud = ("--" if not use_gmc else ("OK" if tracker.last_gmc_ok else "INIT"))
        hud = (f"{fps_ema:5.1f}FPS det={len(cands)} maxScore={max_score:.0f} trk={len(tracks)} "
               f"pid={primary_id} GMC:{gmc_hud} {'UNDIST' if undistort_on else 'RAW'} | serial:{serial_hud}")
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
                rebuild_tracker()
                print(f"   切换为: {'去畸变' if undistort_on else '原始'}（已保存，跟踪器已重置）")
        elif key == ord("r"):
            primary_id = None
            euro_x.reset()
            euro_y.reset()
            print("   已放弃当前主目标，下一帧重新按分数最高挑选。")

    persist()
    cam.close()
    cv2.destroyAllWindows()
    if link is not None:
        link.close()
    print(f"已退出。共发送 {tx_count} 帧串口数据。设置已保存到 {os.path.basename(config.CONFIG_FILE)}。")


if __name__ == "__main__":
    main()
