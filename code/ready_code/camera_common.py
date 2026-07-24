#!/usr/bin/env python3
"""
USB摄像头公共模块 - 被 calibrate_camera.py / v1.2.py 复用
环境: /home/hao/vision_env/bin/python3

集中管理三样东西，避免各脚本各写一份、参数走样：
  1. 固定曝光/增益/白平衡等 V4L2 控制项（现场用 tune_camera.py 调出的值）；
  2. 打开摄像头的标准流程（MJPG / 640x480 / 手动模式）；
  3. 相机标定结果的读写，以及实时去畸变映射的构建。

设计说明:
  - 固定参数与 v1.1.py 保持一致；换了拍摄环境/光照后需重新跑 tune_camera.py，
    并同步更新下面的常量（此文件是唯一事实来源，改这里即可）。
  - 用 v4l2-ctl 直接下发控制项，而不是只用 cv2.VideoCapture.set()，
    因为 exposure_time_absolute / white_balance_temperature 这类控制项
    在 OpenCV 属性映射里不完全可靠。
"""

import os
import subprocess

import cv2
import numpy as np

DEVICE = "/dev/video0"
FRAME_W, FRAME_H, FRAME_FPS = 640, 480, 30

# 固定参数：由 tune_camera.py 现场调出，2026-07-17 记录（与 v1.1.py 一致）
EXPOSURE_TIME_ABSOLUTE = 10       # 曝光时间，范围 10~626
GAIN = 1                          # 增益，范围 1~8
WHITE_BALANCE_TEMPERATURE = 3399  # 白平衡色温，范围 2800~6500
BRIGHTNESS = 10                  # 亮度，范围 -64~64

# 标定板参数：实测 chessboard_a4.png 为 9x6 内角点、方格 25mm
CHESSBOARD_SIZE = (9, 6)   # (列内角点数, 行内角点数)
SQUARE_SIZE_MM = 25.0      # 单格边长(毫米)，只影响外参尺度，不影响去畸变

# 标定结果文件（与本脚本同目录）
CALIB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "camera_calib.npz")


def set_ctrl(name, value):
    """用 v4l2-ctl 下发单个 V4L2 控制项。"""
    subprocess.run(
        ["v4l2-ctl", "-d", DEVICE, f"--set-ctrl={name}={value}"],
        check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def apply_fixed_params():
    """切到手动模式并下发固定的曝光/增益/白平衡/亮度。"""
    # 先切手动模式，否则 exposure_time_absolute / white_balance_temperature
    # 处于 inactive 状态，设置了也不生效
    set_ctrl("auto_exposure", 1)             # 1 = Manual Mode
    set_ctrl("white_balance_automatic", 0)   # 关闭自动白平衡
    set_ctrl("exposure_time_absolute", EXPOSURE_TIME_ABSOLUTE)
    set_ctrl("gain", GAIN)
    set_ctrl("white_balance_temperature", WHITE_BALANCE_TEMPERATURE)
    set_ctrl("brightness", BRIGHTNESS)


def open_camera():
    """按固定参数打开摄像头，返回 cap（失败返回 None）。"""
    apply_fixed_params()

    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cap.isOpened():
        return None

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_FPS, FRAME_FPS)
    return cap


def save_calibration(camera_matrix, dist_coeffs, image_size, rms):
    """保存标定结果到 CALIB_FILE。"""
    np.savez(
        CALIB_FILE,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        image_size=np.array(image_size),
        rms=rms,
    )


def load_calibration():
    """读取标定结果，返回 dict；文件不存在返回 None。"""
    if not os.path.exists(CALIB_FILE):
        return None
    data = np.load(CALIB_FILE)
    return {
        "camera_matrix": data["camera_matrix"],
        "dist_coeffs": data["dist_coeffs"],
        "image_size": tuple(int(x) for x in data["image_size"]),
        "rms": float(data["rms"]),
    }


def build_undistort_maps(camera_matrix, dist_coeffs, image_size, alpha=0.0):
    """
    预计算去畸变映射，供 cv2.remap() 实时使用（比每帧 cv2.undistort() 快得多）。

    alpha: 0 = 裁掉去畸变后产生的黑边(画面会略微放大)，
           1 = 保留全部像素(边缘会有黑色弯曲区域)。默认 0，画面干净。
    返回 (map1, map2, new_camera_matrix)。
    """
    w, h = image_size
    new_camera_matrix, _roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), alpha, (w, h)
    )
    map1, map2 = cv2.initUndistortRectifyMap(
        camera_matrix, dist_coeffs, None, new_camera_matrix, (w, h), cv2.CV_16SC2
    )
    return map1, map2, new_camera_matrix
