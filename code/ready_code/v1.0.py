#!/usr/bin/env python3
"""
USB摄像头调用 - 实时预览（自动曝光/自动白平衡）
环境: /home/hao/vision_env/bin/python3
用法: python3 v1.0.py

设计原则:
  - 开启自动曝光、自动白平衡，由摄像头硬件自动调节，无需手动干预。
  - 该摄像头(Integrated_Webcam_HD)是定焦镜头，没有自动对焦硬件，
    因此不提供对焦功能；清晰度需要靠物理调整摄像头到目标的距离。
  - 用 MJPG 编码代替默认 YUYV，同样是 1280x720 分辨率下帧率从
    ~10fps 提升到 30fps，这是画面发糊(运动模糊)的主要原因之一。

按键:
  q / ESC : 退出
"""

import cv2

FRAME_W, FRAME_H, FRAME_FPS = 1280, 720, 30


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("❌ 无法打开摄像头 /dev/video0")
        return

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_FPS, FRAME_FPS)

    # 自动模式：开启自动曝光、开启自动白平衡
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)  # 3 = 自动曝光
    cap.set(cv2.CAP_PROP_AUTO_WB, 1)        # 1 = 自动白平衡

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"✅ 摄像头已打开: {actual_w}x{actual_h} @ {actual_fps}fps (MJPG, 自动曝光/白平衡)")
    print("   [q/ESC] 退出")

    cv2.namedWindow("USB Camera", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("USB Camera", actual_w, actual_h)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ 读取帧失败")
            break

        cv2.putText(frame, "Auto Exposure / Auto WB", (10, 25),
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
