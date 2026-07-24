#!/usr/bin/env python3
"""
三规格 NCNN 模型性能基准测试（树莓派 5）
环境: /home/hao/vision_env/bin/python3
用法:
    /home/hao/vision_env/bin/python3 code/detect_code/benchmark.py
    # 常用参数:
    #   --iters 200      每个模型计时的帧数（默认 200）
    #   --warmup 15      预热帧数（默认 15，排除首帧建图/缓存冷启动）
    #   --video <path>   用于取帧的视频（默认那段训练录像，真实 640x480 含球场景）
    #   --threads 4      ncnn 线程数（默认 4，树莓派 5 四核）
    #   --sizes 320,416,640

不依赖摄像头/显示器（用录像帧回放做负载），结果可复现，适合纯 SSH 里跑。
测量口径：
  - pre  = letterbox + 归一化预处理
  - infer= ncnn 前向
  - post = 解码 + NMS + 反 letterbox
  - total= 上面三项之和（≈ detect_live 里一帧的检测耗时）
  - undist = 去畸变 remap 的单帧开销（部署默认开，单列出来便于估算真实帧率）
端到端 FPS 用 total 的均值换算；叠加 undist 后是"默认去畸变部署"的估算上限
（真实还要再加摄像头读帧，通常几毫秒，视 USB/MJPG 解码而定）。
"""

import argparse
import os
import sys
import statistics
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yolo_ncnn import YoloNcnn, MODELS_DIR  # noqa: E402

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_VIDEO = os.path.join(
    _PROJECT_ROOT, "code", "record_code", "output", "dataset_20260723_185257.mp4"
)


def load_frames(video, n):
    """从视频均匀取 n 帧（若视频不够则循环补齐），返回 BGR 帧列表。"""
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"打不开视频: {video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or n
    frames = []
    for k in range(n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * k / n) % total)
        ok, fr = cap.read()
        if ok:
            frames.append(fr)
    cap.release()
    if not frames:
        raise RuntimeError("未能从视频读到任何帧")
    return frames


def measure_undistort(frames):
    """测去畸变 remap 的单帧均值耗时(ms)；无标定则返回 None。"""
    sys.path.insert(0, os.path.join(_PROJECT_ROOT, "code", "ready_code"))
    import camera_common as cc
    calib = cc.load_calibration()
    if calib is None:
        return None
    h, w = frames[0].shape[:2]
    m1, m2, _ = cc.build_undistort_maps(
        calib["camera_matrix"], calib["dist_coeffs"], (w, h), alpha=0.0
    )
    ts = []
    for fr in frames[:100]:
        t0 = time.perf_counter()
        cv2.remap(fr, m1, m2, cv2.INTER_LINEAR)
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.mean(ts)


def bench_model(size, frames, iters, warmup, threads):
    det = YoloNcnn(os.path.join(MODELS_DIR, f"best_ncnn_{size}"), num_threads=threads)
    nf = len(frames)

    # 预热（首帧会建图/分配，排除掉）
    for i in range(warmup):
        det.detect(frames[i % nf])

    stages = {"pre_ms": [], "infer_ms": [], "post_ms": [], "total_ms": []}
    ndet = []
    for i in range(iters):
        dets, t = det.detect(frames[i % nf])
        for k in stages:
            stages[k].append(t[k])
        ndet.append(len(dets))
    det.release()

    def stat(xs):
        xs_sorted = sorted(xs)
        return {
            "mean": statistics.mean(xs),
            "p50": xs_sorted[len(xs) // 2],
            "p95": xs_sorted[min(len(xs) - 1, int(len(xs) * 0.95))],
        }

    res = {k: stat(v) for k, v in stages.items()}
    res["fps"] = 1000.0 / res["total_ms"]["mean"]
    res["avg_det"] = statistics.mean(ndet)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=15)
    ap.add_argument("--video", default=DEFAULT_VIDEO)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--sizes", default="320,416,640")
    args = ap.parse_args()

    sizes = [int(s) for s in args.sizes.split(",")]
    print(f"取帧视频: {args.video}")
    frames = load_frames(args.video, max(args.iters, args.warmup) + 5)
    print(f"载入 {len(frames)} 帧 ({frames[0].shape[1]}x{frames[0].shape[0]})，"
          f"ncnn 线程={args.threads}，计时 {args.iters} 帧/模型，预热 {args.warmup}\n")

    undist_ms = measure_undistort(frames)

    results = {}
    for sz in sizes:
        print(f"跑 best_ncnn_{sz} ...", flush=True)
        results[sz] = bench_model(sz, frames, args.iters, args.warmup, args.threads)

    # 输出 Markdown 表（可直接贴进文档）
    print("\n" + "=" * 72)
    print(f"| 规格 | 纯推理 infer(ms) | 端到端 total(ms) | 检测 FPS | pre(ms) | post(ms) | 均检测数 |")
    print(f"|---|---|---|---|---|---|---|")
    for sz in sizes:
        r = results[sz]
        print(f"| {sz} "
              f"| {r['infer_ms']['mean']:.1f} (p95 {r['infer_ms']['p95']:.1f}) "
              f"| {r['total_ms']['mean']:.1f} "
              f"| **{r['fps']:.1f}** "
              f"| {r['pre_ms']['mean']:.1f} "
              f"| {r['post_ms']['mean']:.1f} "
              f"| {r['avg_det']:.1f} |")
    print("=" * 72)

    if undist_ms is not None:
        print(f"\n去畸变 remap 单帧: {undist_ms:.1f} ms")
        print("叠加去畸变后的估算 FPS（total+undist，未含摄像头读帧）:")
        for sz in sizes:
            r = results[sz]
            fps2 = 1000.0 / (r["total_ms"]["mean"] + undist_ms)
            print(f"  {sz}: {fps2:.1f} FPS")
    else:
        print("\n(无 camera_calib.npz，跳过去畸变开销测量)")


if __name__ == "__main__":
    main()
