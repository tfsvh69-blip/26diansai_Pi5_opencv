#!/usr/bin/env python3
"""
数据采集录像 - 为训练模型采集视频（固定曝光/增益/白平衡, 640x480, MJPG）
环境: /home/hao/vision_env/bin/python3
用法:
    /home/hao/vision_env/bin/python3 code/record_code/record_dataset.py [输出文件.mp4]
    不给参数时，输出到 code/record_code/output/dataset_<时间戳>.mp4

设计要点:
  - 复用 code/ready_code/camera_common.py 的固定曝光/增益/白平衡参数（唯一事实来源），
    保证采集画面稳定一致，不另写一份相机参数。
  - 一次运行只产出【一个】MP4：分多段拍摄（空格开/关），全部写进同一个
    VideoWriter，暂停的部分不写入，最终自然拼接成单个文件。
  - 写入文件的是【干净的画面帧】，屏幕上的 REC/时长等提示只画在显示副本上，
    不会污染训练数据。
  - 【默认去畸变】：有 camera_calib.npz 就默认录制去畸变后的画面（与 v1.2 一致，
    部署时同样喂去畸变画面，训练/推理几何一致）；没有标定文件则退回原始画面。
    运行时按 u 可临时切换 去畸变/原始。
  - VideoWriter 的帧率用预览阶段实测的真实帧率，保证回放速度自然。

按键:
  空格      : 开始 / 暂停 录制（可反复切换，多段都写进同一个文件）
  u         : 切换 去畸变 / 原始画面（默认去畸变）
  q / ESC   : 退出（自动收尾并保存 MP4）
"""

import os
import sys
import time
from datetime import datetime

import cv2

# 复用 ready_code 下的公共模块（固定相机参数、开摄像头流程）
READY_CODE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ready_code"
)
sys.path.insert(0, READY_CODE)
import camera_common as cc  # noqa: E402

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
DEFAULT_FPS = 30.0  # 实测不足时的兜底帧率


def make_writer(path, fps, size):
    """按 mp4v 编码创建 VideoWriter，返回 writer（失败返回 None）。"""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, size)
    return writer if writer.isOpened() else None


def main():
    # 输出路径：命令行给了就用，否则用时间戳，保证不覆盖以前采集的数据
    if len(sys.argv) > 1:
        out_path = os.path.abspath(sys.argv[1])
    else:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(OUTPUT_DIR, f"dataset_{stamp}.mp4")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    cap = cc.open_camera()
    if cap is None:
        print("❌ 无法打开摄像头 /dev/video0")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"✅ 摄像头已打开: {w}x{h} (MJPG, 固定曝光/增益/白平衡)")

    # 默认去畸变：有标定文件就构建去畸变映射（cv2.remap 实时用）
    calib = cc.load_calibration()
    if calib is not None:
        map1, map2, _ = cc.build_undistort_maps(
            calib["camera_matrix"], calib["dist_coeffs"], (w, h), alpha=0.0
        )
        undistort = True
        print("   去畸变: 开 (默认)   [u] 切换 去畸变/原始")
    else:
        map1 = map2 = None
        undistort = False
        print("   ⚠ 未找到 camera_calib.npz，按【原始画面】录制（先跑 calibrate_camera.py 可启用去畸变）")

    print("   [空格] 开始/暂停录制   [q/ESC] 退出并保存")
    print(f"   输出文件: {out_path}")

    cv2.namedWindow("Record Dataset", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Record Dataset", w, h)

    writer = None            # 第一次开始录制时才创建，避免产生空文件
    recording = False
    written_frames = 0       # 已写入文件的总帧数（跨多段累计）

    # 实测帧率估算（用于建 writer 时确定回放速度）
    fps_est = DEFAULT_FPS
    last_t = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ 读取帧失败")
            break

        # 默认去畸变：写入文件和显示的都是去畸变后的画面
        if undistort and map1 is not None:
            frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)

        # 平滑估算真实帧率（指数移动平均）
        now = time.time()
        dt = now - last_t
        last_t = now
        if 0 < dt < 1.0:
            inst_fps = 1.0 / dt
            fps_est = 0.9 * fps_est + 0.1 * inst_fps

        # 录制中：把【干净原始帧】写入文件
        if recording and writer is not None:
            writer.write(frame)
            written_frames += 1

        # 显示副本上叠加提示（不影响写入文件的帧）
        display = frame.copy()
        total_sec = written_frames / fps_est if fps_est > 0 else 0
        if recording:
            cv2.circle(display, (22, 24), 9, (0, 0, 255), -1)
            cv2.putText(display, "REC", (38, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(display, "PAUSED  [SPACE] to record", (12, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        mode = "undistort" if undistort else "raw"
        cv2.putText(display, f"saved: {written_frames}f / {total_sec:.1f}s  [{mode}]",
                    (12, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        cv2.imshow("Record Dataset", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):
            if not recording:
                # 首次开始：用实测帧率创建唯一的 writer
                if writer is None:
                    fps = max(5.0, min(60.0, fps_est))
                    writer = make_writer(out_path, fps, (w, h))
                    if writer is None:
                        print("❌ 无法创建视频文件（mp4v 编码不可用？）")
                        break
                    print(f"▶ 开始录制 @ {fps:.1f}fps")
                else:
                    print("▶ 继续录制（追加到同一文件）")
                recording = True
            else:
                recording = False
                print(f"⏸ 暂停（已保存 {written_frames} 帧）")
        elif key == ord("u"):
            if map1 is not None:
                undistort = not undistort
                print(f"切换画面: {'去畸变' if undistort else '原始'}")
            else:
                print("⚠ 无标定文件，无法去畸变")
        elif key in (ord("q"), 27):
            break

    # 收尾
    if writer is not None:
        writer.release()
    cap.release()
    cv2.destroyAllWindows()

    if written_frames > 0:
        secs = written_frames / max(fps_est, 1e-6)
        print(f"✅ 已保存: {out_path}")
        print(f"   共 {written_frames} 帧，约 {secs:.1f} 秒")
    else:
        # 一帧都没录，删掉可能产生的空文件
        if writer is not None and os.path.exists(out_path):
            os.remove(out_path)
        print("未录制任何内容，已退出（无文件生成）")


if __name__ == "__main__":
    main()
