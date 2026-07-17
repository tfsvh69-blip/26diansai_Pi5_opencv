#!/usr/bin/env python3
"""
USB摄像头调用 - 实时预览（固定曝光/固定增益/固定白平衡）
环境: /home/hao/vision_env/bin/python3
用法: python3 v1.1.py

设计原则:
  - 关闭自动曝光/自动增益/自动白平衡，改用固定数值，避免光线变化或物体
    移动时画面忽亮忽暗、颜色跳变，保证 OpenCV 算法拿到的输入稳定一致。
  - 固定数值是用 tune_camera.py 在实际使用光照环境下现场调出来的，
    换了拍摄环境/光照后需要重新跑一次 tune_camera.py 并更新下面的常量。
  - 用 v4l2-ctl 直接下发 V4L2 控制项（而不是只用 cv2.VideoCapture.set()），
    因为 exposure_time_absolute / white_balance_temperature 这类控制项
    在 OpenCV 的属性映射里不完全可靠。
  - 该摄像头(Integrated_Webcam_HD)是定焦镜头，没有自动对焦硬件，
    因此不提供对焦功能；清晰度需要靠物理调整摄像头到目标的距离。
  - 用 MJPG 编码代替默认 YUYV，帧率更高，画面更不容易发糊(运动模糊)。
  - 该摄像头 MJPG 模式下只支持 1280x720 / 640x480 两档分辨率(v4l2-ctl
    --list-formats-ext 实测)，没有 640x360，因此用最低档 640x480。

按键:
  q / ESC : 退出
"""

import subprocess
import cv2

DEVICE = "/dev/video0"
FRAME_W, FRAME_H, FRAME_FPS = 640, 480, 30

# 固定参数：由 tune_camera.py 现场调出，2026-07-17 记录
EXPOSURE_TIME_ABSOLUTE = 13     # 曝光时间，范围 10~626
GAIN = 2                        # 增益，范围 1~8
WHITE_BALANCE_TEMPERATURE = 4532  # 白平衡色温，范围 2800~6500
BRIGHTNESS = -11                # 亮度，范围 -64~64


def set_ctrl(name, value):
    subprocess.run(
        ["v4l2-ctl", "-d", DEVICE, f"--set-ctrl={name}={value}"],
        check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def main():
    # 先切手动模式，否则 exposure_time_absolute / white_balance_temperature
    # 处于 inactive 状态，设置了也不生效
    set_ctrl("auto_exposure", 1)             # 1 = Manual Mode
    set_ctrl("white_balance_automatic", 0)   # 关闭自动白平衡
    set_ctrl("exposure_time_absolute", EXPOSURE_TIME_ABSOLUTE)
    set_ctrl("gain", GAIN)
    set_ctrl("white_balance_temperature", WHITE_BALANCE_TEMPERATURE)
    set_ctrl("brightness", BRIGHTNESS)

    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("❌ 无法打开摄像头 /dev/video0")
        return

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_FPS, FRAME_FPS)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"✅ 摄像头已打开: {actual_w}x{actual_h} @ {actual_fps}fps (MJPG, 固定曝光/增益/白平衡)")
    print(f"   曝光={EXPOSURE_TIME_ABSOLUTE} 增益={GAIN} 白平衡色温={WHITE_BALANCE_TEMPERATURE} 亮度={BRIGHTNESS}")
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
