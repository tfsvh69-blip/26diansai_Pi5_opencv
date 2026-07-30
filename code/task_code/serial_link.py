#!/usr/bin/env python3
"""
串口链路封装 - 锁定固定物理 USB 口 + 自动等待/热插拔重连
环境: /home/hao/vision_env/bin/python3（需 pyserial）

被 serial_test.py（收）和后续 v1.0.py（发坐标）复用，把"找口/等口/断线重连"这套
逻辑收在一处，避免各脚本各写一份。

为什么用 by-path 而不是 /dev/ttyUSB0：
  ttyUSB 的编号会随插拔顺序变，而 /dev/serial/by-path/... 绑定的是【物理 USB 口】，
  只要一直插同一个口，路径就固定。这样"指定只插这一个口"才可靠。
  查看本机当前口：  ls -l /dev/serial/by-path/
  换了口就把下面 PREFERRED_BYPATH 改成新的那条。
"""

import glob
import os
import time

import serial  # pyserial

# 固定选用的物理 USB 口（2026-07-24 实测：CH340 插在 xhci-hcd.1 的 usb-0:2 口）
PREFERRED_BYPATH = "/dev/serial/by-path/platform-xhci-hcd.1-usb-0:2:1.0-port0"


class Disconnected(Exception):
    """读写时发现设备已拔出。"""


def resolve_port(preferred=PREFERRED_BYPATH, any_ok=False):
    """
    解析出一个具体的串口设备路径；找不到返回 None。
      preferred: 首选的固定 by-path（存在就用它，实现"只认这一个口"）。
      any_ok=True: 首选口不在时，放宽到"任意 CH340 / 任意 ttyUSB"，方便临时换口调试。
    """
    if preferred and os.path.exists(preferred):
        return os.path.realpath(preferred)
    if any_ok:
        byid = (glob.glob("/dev/serial/by-id/*1a86*")
                or glob.glob("/dev/serial/by-id/*USB_Serial*"))
        if byid:
            return os.path.realpath(byid[0])
        tty = sorted(glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*"))
        if tty:
            return tty[0]
    return None


class SerialLink:
    """
    一条会自动重连的串口链路。用法：
        link = SerialLink(baud=115200)
        link.wait_and_open()          # 阻塞直到设备就绪并打开
        data = link.read_available()  # 读；设备被拔会抛 Disconnected
        link.write(b"...")            # 写；被拔会抛 Disconnected
    捕获到 Disconnected 后再调 wait_and_open() 即可重连。
    """

    def __init__(self, port=None, baud=115200, any_ok=False, log=print):
        self.explicit_port = port      # 用户 --port 显式指定时优先
        self.baud = baud
        self.any_ok = any_ok
        self.log = log
        self.ser = None
        self.dev = None
        self._last_try = 0.0

    @property
    def connected(self):
        return self.ser is not None

    def _target(self):
        if self.explicit_port:
            return self.explicit_port if os.path.exists(self.explicit_port) else None
        return resolve_port(any_ok=self.any_ok)

    def wait_and_open(self, poll=0.5):
        """循环等待目标设备出现并打开；期间每隔几秒提示一次。永不返回失败（除非 Ctrl-C）。"""
        waited = 0.0
        announced = False
        while True:
            dev = self._target()
            if dev is not None:
                try:
                    self.ser = serial.Serial(
                        port=dev, baudrate=self.baud,
                        bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE,
                        stopbits=serial.STOPBITS_ONE, timeout=0.2,
                    )
                    self.ser.reset_input_buffer()
                    self.dev = dev
                    self.log(f"✅ 已连接串口: {dev} @ {self.baud} 8N1")
                    return
                except (serial.SerialException, OSError) as e:
                    self.log(f"   打开 {dev} 失败({e})，继续等待…")
            else:
                if not announced:
                    tgt = self.explicit_port or PREFERRED_BYPATH
                    self.log(f"⌛ 等待串口设备出现：{tgt}"
                             + ("" if self.explicit_port else "（--any 可放宽到任意 CH340/ttyUSB）"))
                    announced = True
            time.sleep(poll)
            waited += poll
            if dev is None and waited >= 5.0:
                self.log("   …仍未发现设备，请插上 USB-TTL（或确认插的是同一个口）。")
                waited = 0.0

    def try_open(self, min_interval=1.0):
        """
        非阻塞地尝试连接一次（供实时循环调用，不会卡住画面）。
        已连接直接返回 True；未连接则最多每 min_interval 秒试一次，成功返回 True。
        """
        if self.ser is not None:
            return True
        now = time.time()
        if now - self._last_try < min_interval:
            return False
        self._last_try = now
        dev = self._target()
        if dev is None:
            return False
        try:
            self.ser = serial.Serial(
                port=dev, baudrate=self.baud,
                bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE, timeout=0.2,
            )
            self.dev = dev
            self.log(f"✅ 串口已连接: {dev} @ {self.baud}")
            return True
        except (serial.SerialException, OSError):
            self.ser = None
            return False

    def read_available(self):
        """读走当前缓冲里的数据（至少阻塞一个 timeout）；设备被拔出抛 Disconnected。"""
        if self.ser is None:
            raise Disconnected()
        try:
            n = self.ser.in_waiting
            return self.ser.read(n if n > 0 else 1)
        except (serial.SerialException, OSError, TypeError):
            self._drop()
            raise Disconnected()

    def read_available_nonblocking(self):
        """
        非阻塞版：只取此刻 OS 缓冲里已有的字节，没有就立刻返回 b""，绝不等超时。
        设备被拔出抛 Disconnected。

        为什么单独加一个：read_available() 在无数据时会 self.ser.read(1) 阻塞一个 timeout
        （0.2s），对【每帧都要 poll 一次串口】的实时循环（如视觉主循环）是致命的——下位机
        不是一直在发数据，大多数帧 in_waiting==0，于是每帧白等最多 0.2s，把整个循环（连同
        串口发坐标的频率）拖到个位数 Hz。实测单次 read_available() 均值 ~117ms。实时循环改
        用这个方法后，无数据帧几乎零耗时。serial_test.py 那种专门的收包循环仍用会阻塞的
        read_available()（靠它的 timeout 天然限速、不空转），故本方法【新增】而不改旧的。
        """
        if self.ser is None:
            raise Disconnected()
        try:
            n = self.ser.in_waiting
            if n <= 0:
                return b""
            return self.ser.read(n)
        except (serial.SerialException, OSError, TypeError):
            self._drop()
            raise Disconnected()

    def write(self, data):
        """写数据；设备被拔出抛 Disconnected。"""
        if self.ser is None:
            raise Disconnected()
        try:
            return self.ser.write(data)
        except (serial.SerialException, OSError) as e:
            self._drop()
            raise Disconnected() from e

    def _drop(self):
        try:
            if self.ser:
                self.ser.close()
        except Exception:
            pass
        self.ser = None
        self.dev = None

    def close(self):
        self._drop()
