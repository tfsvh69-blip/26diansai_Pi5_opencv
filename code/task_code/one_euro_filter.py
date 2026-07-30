#!/usr/bin/env python3
"""
One Euro Filter —— 低延迟的一维信号平滑滤波器
环境: /home/hao/vision_env/bin/python3（纯 Python，无依赖）

用途：v1.1.py 里对主目标（跟踪后的）像素坐标 x/y 做最后一步平滑，
去掉检测框本身的像素级抖动，给下位机控制环一个更干净的位置信号，
同时比固定窗口的滑动平均延迟更低（目标快速移动时截止频率自动升高、跟得更紧；
目标静止/慢动时截止频率自动降低、抖动压得更狠）。原理见论文
"1€ Filter: A Simple Speed-based Low-pass Filter for Noisy Input in
Interactive Systems" (Casiez et al., 2012)。

用法：
    f = OneEuroFilter(freq=30.0, min_cutoff=1.0, beta=0.0)
    y = f(x, timestamp=None)   # timestamp 不给则用内部估的采样间隔
"""

import time


class _LowPass:
    """一阶低通滤波器，外部传入截止频率对应的平滑系数 alpha。"""

    def __init__(self):
        self._y = None
        self._initialized = False

    def filter(self, x, alpha):
        if not self._initialized:
            self._y = x
            self._initialized = True
        else:
            self._y = alpha * x + (1.0 - alpha) * self._y
        return self._y

    def reset(self):
        self._initialized = False
        self._y = None


def _alpha(cutoff, dt):
    """截止频率 -> 一阶低通的平滑系数（标准 RC 低通离散化公式）。"""
    tau = 1.0 / (2 * 3.141592653589793 * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """
    一维 One Euro Filter。多维信号（如 x、y）各建一个独立实例即可。

    参数：
      freq       : 预期采样频率(Hz)，仅在 timestamp 未提供时用于估计 dt 的兜底值。
      min_cutoff : 最小截止频率——越小，目标静止/慢动时越平滑（但快速运动时滞后越明显）。
      beta       : 速度系数——越大，目标快速运动时截止频率升得越高（越跟手，抖动也越多）。
      d_cutoff   : 用于平滑"速度估计"本身的截止频率，一般不用调，默认 1.0 即可。
    """

    def __init__(self, freq=30.0, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x_filt = _LowPass()
        self._dx_filt = _LowPass()
        self._last_time = None

    def reset(self):
        """目标切换/重新捕获时调用，避免把不相关的历史值平滑进新目标。"""
        self._x_filt.reset()
        self._dx_filt.reset()
        self._last_time = None

    def __call__(self, x, timestamp=None):
        t = time.monotonic() if timestamp is None else timestamp
        if self._last_time is None:
            dt = 1.0 / self.freq
        else:
            dt = max(t - self._last_time, 1e-6)
        self._last_time = t

        # 先估计速度（对速度也做一次低通，抑制速度估计本身的噪声）
        prev_x = self._x_filt._y if self._x_filt._initialized else x
        dx = (x - prev_x) / dt
        edx = self._dx_filt.filter(dx, _alpha(self.d_cutoff, dt))

        # 速度越大，截止频率越高（beta 项），响应更快；速度小时截止频率趋近 min_cutoff，更平滑
        cutoff = self.min_cutoff + self.beta * abs(edx)
        return self._x_filt.filter(x, _alpha(cutoff, dt))
