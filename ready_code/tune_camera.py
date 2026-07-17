#!/usr/bin/env python3
"""
USB摄像头交互式调参工具 - 固定曝光/增益/白平衡
环境: /home/hao/vision_env/bin/python3
用法: python3 tune_camera.py
注意: 需要能看到画面窗口，请在有显示器/桌面的会话里运行（不要用纯 SSH 无 X 转发）。

用滑条实时调节 曝光/增益/白平衡色温/亮度，每次滑条变化就用 v4l2-ctl 直接
下发给 V4L2 驱动（不依赖 cv2.VideoCapture.set()，部分 UVC 控制项在 OpenCV
里的属性映射不完全可靠）。

摄像头: Integrated_Webcam_HD (0c45:64ab)，实测控制项范围
(v4l2-ctl -d /dev/video0 --list-ctrls-menus)：
  auto_exposure            : 1=Manual Mode, 3=Aperture Priority Mode (默认 3)
  exposure_time_absolute   : 10~626  (默认 156，单位 100us)
  white_balance_automatic  : 0/1 (默认 1)
  white_balance_temperature: 2800~6500 (默认 4600)
  gain                     : 1~8 (默认 1)
  brightness               : -64~64 (默认 0)

按键:
  q / ESC : 退出，并在终端打印最终数值（记下来填进 v1.1.py）
"""

import subprocess
import cv2

DEVICE = "/dev/video0"
# 该摄像头 MJPG 模式下只支持 1280x720 / 640x480 两档分辨率
# (v4l2-ctl --list-formats-ext 实测)，没有 640x360，因此用最低档 640x480。
FRAME_W, FRAME_H, FRAME_FPS = 640, 480, 30

# name: (min, max, default)
CTRLS = {
    "exposure_time_absolute": (10, 626, 156),
    "gain": (1, 8, 1),
    "white_balance_temperature": (2800, 6500, 4600),
    "brightness": (-64, 64, 0),
}


def set_ctrl(name, value):
    subprocess.run(
        ["v4l2-ctl", "-d", DEVICE, f"--set-ctrl={name}={value}"],
        check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def main():
    # 先切手动模式，否则 exposure_time_absolute / white_balance_temperature
    # 处于 inactive 状态，改了也不生效
    set_ctrl("auto_exposure", 1)             # 1 = Manual Mode
    set_ctrl("white_balance_automatic", 0)   # 关闭自动白平衡

    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("❌ 无法打开摄像头 /dev/video0")
        return

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_FPS, FRAME_FPS)

    win = "Camera Tune (q/ESC exit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, FRAME_W, FRAME_H)
    for name, (lo, hi, default) in CTRLS.items():
        # cv2 trackbar 不支持负数下限，统一平移到 [0, hi-lo] 区间
        cv2.createTrackbar(name, win, default - lo, hi - lo, lambda v: None)

    last_values = {name: None for name in CTRLS}

    print("✅ 摄像头已打开，拖动滑条调节，画面正常后按 q/ESC 退出")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ 读取帧失败")
            break

        y = 20
        for name, (lo, hi, _default) in CTRLS.items():
            pos = cv2.getTrackbarPos(name, win)
            value = pos + lo
            if value != last_values[name]:
                set_ctrl(name, value)
                last_values[name] = value
            cv2.putText(frame, f"{name}={value}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            y += 20

        cv2.imshow(win, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("\n最终数值（填进 v1.1.py 里）：")
    for name, value in last_values.items():
        print(f"  {name} = {value}")


if __name__ == "__main__":
    main()
