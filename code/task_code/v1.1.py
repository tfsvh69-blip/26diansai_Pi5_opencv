#!/usr/bin/env python3
"""
题目 V1.1 —— YOLO11n + BoT-SORT(无 ReID) + GMC + 卡尔曼滤波，保证框选连续性
环境: /home/hao/vision_env/bin/python3
用法:
    /home/hao/vision_env/bin/python3 code/task_code/v1.1.py [--size 320|416|640]
    #   --size 320      模型规格；不给则用上次保存的配置（沿用 v1.0 的 task_config.json）
    #   --conf 0.25     置信度阈值（同时作为 BoT-SORT 的高置信度阈值）；不给用配置
    #   --any           串口找不到固定口时放宽到任意 CH340/ttyUSB
    #   --no-serial     只看检测、不发串口(纯视觉调试)
    #   --no-gmc        关闭相机运动补偿(GMC)，调试用/若性能吃紧
    #   --no-euro       关闭 One Euro Filter，直接发跟踪原始坐标

相对 v1.0 的核心变化 —— 为什么要跟踪而不是"每帧独立选最大球"：
  v1.0 每帧独立选面积最大的球当主目标，一旦漏检一帧就直接报 found=0，
  或者出现第二个更大的球时主目标会【瞬间跳过去】，对下位机的控制环很不友好。
  v1.1 引入跟踪，把"选哪个球当主目标"和"这一帧有没有测到"解耦：
    - YOLO 检测 -> BoT-SORT(无 ReID) 关联，给每个球分配持续的轨迹 id；
      内部是 ByteTrack 式高/低两级置信度关联 + 相机运动补偿(GMC) + 用 (cx,cy,w,h)
      建模的卡尔曼滤波，细节见 bot_sort.py 顶部注释；
    - 主目标一旦锁定某个轨迹 id，只要该轨迹还活着（tracked，或刚丢失、在宽限期内
      靠卡尔曼预测续着）就【不换目标】；轨迹真正超时消失了才重新按"面积最大"选一个；
    - 主目标坐标最后过一道 One Euro Filter（见 one_euro_filter.py）再发串口，
      去掉检测框本身的像素抖动，给下位机一个更干净、低延迟的位置信号。
  ReID（外观特征关联）按要求先关闭：单类球本身没什么可辨识的外观特征，
  且树莓派5 上跑 ReID 特征网络划不来；后续如控制端确有需要区分"很像的多个目标"
  再考虑加。

配置持久化：复用 v1.0 的 task_config.json（size/undistort/conf），语义不变。

按键:
  1 / 2 / 3 : 切换模型规格 320 / 416 / 640（会重建跟踪器，坐标系不受影响时轨迹尽量保留id计数重置）
  u         : 切换 去畸变 / 原始画面（坐标系变了，跟踪器会重置）
  r         : 主动放弃当前主目标，下一帧重新按"面积最大"挑（人工纠错用）
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
from yolo_ncnn import YoloNcnn, MODELS_DIR  # noqa: E402
from serial_link import SerialLink, Disconnected            # noqa: E402
from camera_link import CameraLink, CameraLost              # noqa: E402
from bot_sort import BotSort, TRACKED                        # noqa: E402
from one_euro_filter import OneEuroFilter                   # noqa: E402
import protocol                     # noqa: E402
import config                       # noqa: E402

SIZES = (320, 416, 640)
# 现场实测端到端 FPS（见 docs/模型性能测试记录.md），用来把"丢失宽限期"换算成帧数、
# 给 One Euro Filter 一个合理的采样频率初值；并非精确值，够用即可。
SIZE_FPS_HINT = {320: 17.0, 416: 12.0, 640: 8.0}

TRACK_BUFFER_SEC = 0.5      # 轨迹丢失后最多"凭卡尔曼预测续命"多久才真正丢弃
LOW_CONF_RATIO = 0.4        # ByteTrack 低置信度阈值 = 高阈值(conf) * 此比例
MIN_LOW_CONF = 0.05
ONE_EURO_MIN_CUTOFF = 1.0   # 越小，目标静止/慢动时越平滑
ONE_EURO_BETA = 0.02        # 越大，目标快速运动时跟得越紧（抖动也越多）


def get_model(cache, size):
    if size not in cache:
        print(f"   载入模型 best_ncnn_{size} ...")
        cache[size] = YoloNcnn(os.path.join(MODELS_DIR, f"best_ncnn_{size}"))
    return cache[size]


def two_stage_detect(det, frame, low_thresh):
    """一次前向，按低阈值解码一次，供 BoT-SORT 内部再拆成高/低两级。"""
    h, w = frame.shape[:2]
    t0 = time.perf_counter()
    mat, scale, left, top = det.preprocess(frame)
    t1 = time.perf_counter()
    out = det.forward(mat)
    t2 = time.perf_counter()
    dets = det.decode(out, scale, left, top, w, h, conf_thres=low_thresh)
    t3 = time.perf_counter()
    timing = {"pre_ms": (t1 - t0) * 1e3, "infer_ms": (t2 - t1) * 1e3,
              "post_ms": (t3 - t2) * 1e3, "total_ms": (t3 - t0) * 1e3}
    return dets, timing


def make_tracker(high_thresh, size, use_gmc):
    low_thresh = max(MIN_LOW_CONF, high_thresh * LOW_CONF_RATIO)
    fps_hint = SIZE_FPS_HINT.get(size, 15.0)
    tracker = BotSort(high_thresh=high_thresh, low_thresh=low_thresh,
                       frame_rate=fps_hint, track_buffer_sec=TRACK_BUFFER_SEC,
                       use_gmc=use_gmc)
    # 外层解码阈值必须 <= tracker 内部（可能被 clamp 过）的 low_thresh，否则极端阈值下
    # 会把 tracker 想要的"低置信度候选"提前过滤掉；用 tracker.low_thresh 做唯一事实来源。
    return tracker, tracker.low_thresh, fps_hint


def draw_tracks(frame, tracks, primary_id):
    for t in tracks:
        x1, y1, x2, y2 = (int(round(v)) for v in t.tlbr)
        is_primary = (t.track_id == primary_id)
        if t.state == TRACKED:
            color = (0, 0, 255) if is_primary else (0, 255, 0)
            thickness = 3 if is_primary else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            label = f"#{t.track_id} {t.score:.2f}"
        else:  # LOST：卡尔曼预测续着，画细一点的橙色框提示"这是预测出来的"
            color = (0, 165, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
            label = f"#{t.track_id} LOST"
        cv2.putText(frame, label, (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=None, choices=SIZES,
                    help="YOLO 规格；不给则用上次保存的配置")
    ap.add_argument("--conf", type=float, default=None, help="置信度阈值；不给则用配置")
    ap.add_argument("--any", action="store_true", help="串口放宽到任意 CH340/ttyUSB")
    ap.add_argument("--no-serial", action="store_true", help="不发串口，只看检测")
    ap.add_argument("--no-gmc", action="store_true", help="关闭相机运动补偿")
    ap.add_argument("--no-euro", action="store_true", help="关闭 One Euro Filter 平滑")
    args = ap.parse_args()

    cfg = config.load()
    if args.size is not None:
        cfg["size"] = args.size
    if args.conf is not None:
        cfg["conf"] = args.conf
    use_gmc = not args.no_gmc
    use_euro = not args.no_euro
    print(f"⚙️  配置文件: {config.CONFIG_FILE}")
    print(f"    载入设置: size={cfg['size']} undistort={cfg['undistort']} conf={cfg['conf']}"
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

    cache = {}
    cur_size = cfg["size"]
    det = get_model(cache, cur_size)
    det.conf_thres = cfg["conf"]

    tracker, low_thresh, fps_hint = make_tracker(cfg["conf"], cur_size, use_gmc)
    primary_id = None
    euro_x = OneEuroFilter(freq=fps_hint, min_cutoff=ONE_EURO_MIN_CUTOFF, beta=ONE_EURO_BETA)
    euro_y = OneEuroFilter(freq=fps_hint, min_cutoff=ONE_EURO_MIN_CUTOFF, beta=ONE_EURO_BETA)

    def rebuild_tracker():
        """坐标系或阈值变了（切规格/切去畸变），旧轨迹坐标不再有效，整体重开。"""
        nonlocal tracker, low_thresh, fps_hint, primary_id
        tracker, low_thresh, fps_hint = make_tracker(cfg["conf"], cur_size, use_gmc)
        primary_id = None
        euro_x.freq = fps_hint
        euro_y.freq = fps_hint
        euro_x.reset()
        euro_y.reset()

    def persist():
        cfg["size"] = cur_size
        cfg["undistort"] = undistort_pref
        config.save(cfg)

    link = None
    if not args.no_serial:
        link = SerialLink(any_ok=args.any)
        link.try_open()
    print(f"   当前模型: {cur_size}   [1/2/3]切规格  [u]去畸变  [r]放弃当前主目标  [q]退出")
    print(f"   串口: {'关闭(--no-serial)' if link is None else '开，' + protocol.build_ball_frame(True,320,240,1).decode().strip() + ' 这类帧'}")

    win = "Task V1.1  YOLO+BoT-SORT(noReID)+GMC+Kalman"
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
                    rebuild_tracker()   # 分辨率变了，坐标系变了
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
        raw_dets, _timing = two_stage_detect(det, frame, low_thresh)
        tracks = tracker.update(raw_dets, frame)
        loop_ms = (time.perf_counter() - t0) * 1e3
        fps = 1000.0 / loop_ms if loop_ms > 0 else 0.0
        fps_ema = fps if fps_ema is None else 0.9 * fps_ema + 0.1 * fps

        n_high = sum(1 for d in raw_dets if d[4] >= cfg["conf"])   # 与 v1.0 的 n 语义对齐

        by_id = {t.track_id: t for t in tracks}
        if primary_id is not None and primary_id in by_id:
            primary = by_id[primary_id]
        else:
            candidates = [t for t in tracks if t.state == TRACKED]
            primary = max(candidates, key=lambda t: (t.tlbr[2]-t.tlbr[0])*(t.tlbr[3]-t.tlbr[1])) \
                if candidates else None
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
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(frame, f"({cx},{cy})", (cx + 8, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        gmc_hud = ("--" if not use_gmc else ("OK" if tracker.last_gmc_ok else "INIT"))
        hud = (f"size={cur_size} {fps_ema:4.1f}FPS trk={len(tracks)} pid={primary_id} "
               f"GMC:{gmc_hud} {'UNDIST' if undistort_on else 'RAW'} | serial:{serial_hud}")
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
            print("   已放弃当前主目标，下一帧重新按面积最大挑选。")
        elif key in (ord("1"), ord("2"), ord("3")):
            new_size = SIZES[key - ord("1")]
            if new_size != cur_size:
                cur_size = new_size
                det = get_model(cache, cur_size)
                det.conf_thres = cfg["conf"]
                fps_ema = None
                persist()
                rebuild_tracker()
                print(f"   切换模型: {cur_size}（已保存，跟踪器已重置）")

    persist()
    cam.close()
    cv2.destroyAllWindows()
    for m in cache.values():
        m.release()
    if link is not None:
        link.close()
    print(f"已退出。共发送 {tx_count} 帧串口数据。设置已保存到 {os.path.basename(config.CONFIG_FILE)}。")


if __name__ == "__main__":
    main()
