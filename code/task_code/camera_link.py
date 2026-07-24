#!/usr/bin/env python3
"""
摄像头链路封装 - 自动等待/热插拔重连（对标 serial_link.py）
环境: /home/hao/vision_env/bin/python3

复用 code/ready_code/camera_common.py 的 open_camera()（固定曝光/增益/白平衡、MJPG、
640x480），在其外面包一层"设备不在就循环等、掉线就自动重连"，供 v1.0.py 用。
USB 摄像头临时掉线时脚本不崩、不退，等它回来自动继续。

判定掉线：/dev/video0 消失，或 cap.read() 连续失败。
"""

import os
import sys
import time

import cv2

# 复用 ready_code 下的 camera_common（固定参数、开摄像头流程）
_READY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ready_code")
if _READY not in sys.path:
    sys.path.insert(0, _READY)
import camera_common as cc  # noqa: E402


class CameraLost(Exception):
    """读帧时发现摄像头已掉线。"""


class CameraLink:
    """
    会自动重连的摄像头。用法：
        cam = CameraLink()
        w, h = cam.wait_and_open()     # 阻塞直到摄像头就绪
        frame = cam.read()             # 掉线抛 CameraLost
    在实时循环里也可用 try_open()（非阻塞）配合，掉线时保持界面响应。
    """

    def __init__(self, log=print):
        self.cap = None
        self.log = log
        self.size = None            # (w, h)
        self._last_try = 0.0
        self._read_fail = 0

    @property
    def connected(self):
        return self.cap is not None

    @staticmethod
    def _device_present():
        return os.path.exists(cc.DEVICE)

    def _open_once(self):
        """尝试打开一次；成功返回 (w,h)，失败返回 None。"""
        if not self._device_present():
            return None
        cap = cc.open_camera()
        if cap is None:
            return None
        # 读一帧确认真的能出图（有些情况下 isOpened 为真但读不出）
        ok, _ = cap.read()
        if not ok:
            cap.release()
            return None
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.cap = cap
        self.size = (w, h)
        self._read_fail = 0
        return (w, h)

    def wait_and_open(self, poll=0.5):
        """阻塞循环等待摄像头就绪并打开，返回 (w,h)。期间每隔几秒提示一次。"""
        announced = False
        waited = 0.0
        while True:
            size = self._open_once()
            if size is not None:
                self.log(f"✅ 摄像头已连接: {size[0]}x{size[1]} @ {cc.DEVICE}")
                return size
            if not announced:
                self.log(f"⌛ 等待摄像头 {cc.DEVICE} …（插好 USB 摄像头）")
                announced = True
            time.sleep(poll)
            waited += poll
            if waited >= 5.0:
                self.log(f"   …仍未发现 {cc.DEVICE}，检查 USB 摄像头是否插好（lsusb 应有 0c45:64ab）。")
                waited = 0.0

    def try_open(self, min_interval=1.0):
        """非阻塞尝试连接一次（供实时循环调用）。已连或连上返回 True。"""
        if self.cap is not None:
            return True
        now = time.time()
        if now - self._last_try < min_interval:
            return False
        self._last_try = now
        return self._open_once() is not None

    def read(self):
        """读一帧 BGR；掉线抛 CameraLost；偶发单帧失败返回 None（调用方跳过即可）。"""
        if self.cap is None:
            raise CameraLost()
        ok, frame = self.cap.read()
        if ok and frame is not None:
            self._read_fail = 0
            return frame
        # 读失败：设备没了就判掉线；否则算偶发，连续多次也判掉线
        self._read_fail += 1
        if not self._device_present() or self._read_fail >= 15:
            self._drop()
            raise CameraLost()
        return None

    def _drop(self):
        try:
            if self.cap:
                self.cap.release()
        except Exception:
            pass
        self.cap = None
        self._read_fail = 0

    def close(self):
        self._drop()
