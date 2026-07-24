#!/usr/bin/env python3
"""
USB摄像头调用 - 实时预览（固定曝光/固定增益/固定白平衡）
环境: /home/hao/vision_env/bin/python3
用法: python3 v1.1.py

设计原则:
  - 关闭自动曝光/自动增益/自动白平衡，改用固定数值，避免光线变化或物体
    移动时画面忽亮忽暗、颜色跳变，保证 OpenCV 算法拿到的输入稳定一致。
  - 固定数值集中在 camera_common.py（唯一事实来源），由 tune_camera.py 现场
    调出并自动写回；本脚本直接复用，不再自带一份副本。换光照环境后重跑
    tune_camera.py 即可，本脚本无需改动。
  - 用 v4l2-ctl 直接下发 V4L2 控制项（见 camera_common.py），因为
    exposure_time_absolute / white_balance_temperature 在 OpenCV 属性映射里
    不完全可靠。
  - 该摄像头(Integrated_Webcam_HD)是定焦镜头，没有自动对焦硬件，
    清晰度需要靠物理调整摄像头到目标的距离。
  - MJPG 模式下只支持 1280x720 / 640x480 两档分辨率，用最低档 640x480。

说明:
  - v1.1 是固定参数预览基线，不做去畸变；需要去畸变请用 v1.2.py（默认去畸变）。

按键:
  q / ESC : 退出
"""

import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import camera_common as cc  # noqa: E402


def main():
    cap = cc.open_camera()
    if cap is None:
        print("❌ 无法打开摄像头 /dev/video0")
        return

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"✅ 摄像头已打开: {actual_w}x{actual_h} @ {actual_fps}fps (MJPG, 固定曝光/增益/白平衡)")
    print(f"   曝光={cc.EXPOSURE_TIME_ABSOLUTE} 增益={cc.GAIN} "
          f"白平衡色温={cc.WHITE_BALANCE_TEMPERATURE} 亮度={cc.BRIGHTNESS}")
    print("   [q/ESC] 退出")

    cv2.namedWindow("USB Camera", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("USB Camera", actual_w, actual_h)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ 读取帧失败")
            break

        cv2.putText(frame, "Fixed Exposure / Gain / WB", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("USB Camera", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("已退出")


if __name__ == "__main__":
    main()
