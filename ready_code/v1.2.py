#!/usr/bin/env python3
"""
USB摄像头调用 - 实时预览（固定曝光/增益/白平衡 + 可切换去畸变）
环境: /home/hao/vision_env/bin/python3
用法: /home/hao/vision_env/bin/python3 v1.2.py

相比 v1.1.py 的变化:
  - 复用 camera_common.py 里的固定参数和开摄像头流程（同一份参数，不再各写一份）。
  - 载入 calibrate_camera.py 生成的 camera_calib.npz，运行时可用【u】键
    自由切换「原始画面」和「去畸变(校准)画面」，方便对比或按需使用。
  - 去畸变用预计算映射 + cv2.remap()，比每帧 cv2.undistort() 快，适合树莓派实时。
  - 若还没标定(找不到 camera_calib.npz)，脚本照常运行，只是不能切到去畸变，
    并提示先跑 calibrate_camera.py。

按键:
  u        : 切换 原始 / 去畸变 画面
  q / ESC  : 退出
"""

import cv2

import camera_common as cc


def main():
    cap = cc.open_camera()
    if cap is None:
        print("❌ 无法打开摄像头 /dev/video0")
        return

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"✅ 摄像头已打开: {actual_w}x{actual_h} @ {actual_fps}fps (MJPG, 固定曝光/增益/白平衡)")

    # 载入标定结果（可能没有）
    calib = cc.load_calibration()
    maps = None
    if calib is not None:
        maps = cc.build_undistort_maps(
            calib["camera_matrix"], calib["dist_coeffs"], (actual_w, actual_h), alpha=0.0
        )
        print(f"   已载入标定 (RMS={calib['rms']:.3f}px)，按 [u] 切换 原始/去畸变。")
    else:
        print("   ⚠️ 未找到 camera_calib.npz，只能显示原始画面。先跑 calibrate_camera.py 做标定。")

    undistort_on = maps is not None  # 有标定就默认开去畸变
    print("   [u] 切换去畸变   [q/ESC] 退出")

    win = "USB Camera"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, actual_w, actual_h)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ 读取帧失败")
            break

        if undistort_on and maps is not None:
            frame = cv2.remap(frame, maps[0], maps[1], cv2.INTER_LINEAR)
            label = "UNDISTORTED (u to toggle)"
        else:
            label = "ORIGINAL (u to toggle)"

        cv2.putText(frame, label, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow(win, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("u"):
            if maps is None:
                print("   还没标定，无法切换。先跑 calibrate_camera.py。")
            else:
                undistort_on = not undistort_on
                print(f"   切换为: {'去畸变' if undistort_on else '原始'} 画面")

    cap.release()
    cv2.destroyAllWindows()
    print("已退出")


if __name__ == "__main__":
    main()
