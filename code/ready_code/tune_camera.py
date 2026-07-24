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
  q / ESC : 退出，退出时【自动把最终数值写回 camera_common.py】（唯一事实来源），
            调完即生效，无需手抄。calibrate_camera.py / v1.2.py / record_dataset.py
            都从 camera_common.py 读参数，所以只写这一处即可。
            (v1.0/v1.1 是自带副本的旧版，不受影响，需要的话自行同步。)
"""

import os
import re
import subprocess
import cv2

DEVICE = "/dev/video0"

# 调好后写回的目标：camera_common.py 里的四个常量（唯一事实来源）
CAMERA_COMMON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "camera_common.py")
CTRL_TO_CONST = {
    "exposure_time_absolute": "EXPOSURE_TIME_ABSOLUTE",
    "gain": "GAIN",
    "white_balance_temperature": "WHITE_BALANCE_TEMPERATURE",
    "brightness": "BRIGHTNESS",
}


def save_to_common(values):
    """把调好的数值写回 camera_common.py，只改常量值、保留注释。"""
    try:
        with open(CAMERA_COMMON, "r", encoding="utf-8") as f:
            text = f.read()
    except OSError as e:
        print(f"⚠ 无法读取 {CAMERA_COMMON}: {e}")
        return

    missing = []
    for ctrl, const in CTRL_TO_CONST.items():
        v = values.get(ctrl)
        if v is None:
            continue
        # 只替换 "常量名 = 数字" 里的数字，行尾注释原样保留
        pattern = re.compile(rf"^({const}\s*=\s*)(-?\d+)", re.MULTILINE)
        text, n = pattern.subn(lambda m: m.group(1) + str(v), text)
        if n == 0:
            missing.append(const)

    if missing:
        print(f"⚠ 未在 camera_common.py 找到这些常量，未写入: {', '.join(missing)}")

    try:
        with open(CAMERA_COMMON, "w", encoding="utf-8") as f:
            f.write(text)
    except OSError as e:
        print(f"⚠ 写回 {CAMERA_COMMON} 失败: {e}")
        return
    print(f"✅ 已写回 {CAMERA_COMMON}")
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

    print("\n最终数值：")
    for name, value in last_values.items():
        print(f"  {name} = {value}")
    save_to_common(last_values)


if __name__ == "__main__":
    main()
