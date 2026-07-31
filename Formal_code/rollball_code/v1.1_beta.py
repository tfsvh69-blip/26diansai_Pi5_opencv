#!/usr/bin/env python3
"""
车载平衡滚球 —— 视觉识别 v1.1 beta（形状+高光找钢珠 + 锁定相机成像 + 矩形 ROI）
环境: /home/hao/vision_env/bin/python3

场景：摄像头装在摆杆正上方、光轴垂直向下；25cm 白色 PPR 水管凹槽里放一颗 φ1cm 银色反光
钢珠（有高光点），背景基本纯白。目标：稳定、可复现、对光照变化鲁棒地把球找出来。

【为什么换掉背景差分（历史）】早先这一版用"静态背景差分"（当前帧 vs 一次性拍的
background.png 做 absdiff）。原理性缺陷：任何全局光照变化会让整帧都偏离存好的背景（不
只是球），mask 泛白 → 检测崩；而且相机自动曝光/自动白平衡没锁，每次开机像素值都不一样，
上次调好的阈值必然失配。这两点导致"每次进去都不一定识别得到、光照变一点就崩"。现已改：
  · 相机成像锁定（见下）解决"每次参数不一样"；
  · 检测换成不依赖背景图的形状+高光方案，结构上干掉最脆的一环。
背景差分基线保留在 v1.1_beta.bgsub.bak.py 备查。

本版已实现：
  1. 可替换的摄像头接口：USB(cv2.VideoCapture+V4L2) / CSI(picamera2) /
     RealSense(pyrealsense2，只用彩色流当 2D 摄像头，不用深度)，用 --source 选，
     三个后端都吐 640x480 BGR 帧；换后端不改主流程。当前默认 realsense。
  2. 【相机成像锁定，可复现的地基】RealSense 彩色 sensor 默认自动曝光+自动白平衡全开，
     每次开机/光照微变都会重新自适应、像素值就变。本版起流后【先关自动、再设固定
     曝光/增益/白平衡】（顺序同 camera_common 的 V4L2 约定：auto 必须先切手动），
     exposure/gain/white_balance 三个值都能在 Tuning 窗口滑条实时调、s 存盘，
     "到一个新环境调一次就稳定"。掉线重连自动按最新值重设。
  3. 【检测：形状+高光，逐帧原始值，不做时间维度平滑】直接复用
     code/opencv_code/ball_detector.py 的 BallDetector（HoughCircles 找圆 + HSV 的 V 通道
     高光/亮度/对比度多因素评分），不依赖任何背景图。BallDetector 本身还内建了 EMA 平滑
     (pick_primary()) 和漏检时的速度外推续帧，但本文件【不使用这两项】——发给下位机的必须
     是"这一帧真实检测到的值"，不要平滑/预测，那部分滤波交给下位机做。做法：不调
     pick_primary()，取本帧最高分原始候选；每帧检测完立即 reset() 检测器的跟踪状态，让
     "漏检续帧"这条分支永远不会被触发。没检测到就是没检测到，立即发 NA。
  4. 矩形 ROI：用【上/下/左/右】四个滑条框出感兴趣区域（限制 Hough 搜索范围、排除管外
     杂圆、更快），拖动即生效。
  5. 关键参数在 Tuning 窗口滑条实时可调（滑条名用参数英文名 ASCII——本机 Qt 缺字体，
     中文滑条名显示不出来；中文含义启动时打印到终端，见 SLIDER_HELP）。s 保存回 JSON。

还没做（后面几步再加）：
  - 丢球状态机（连丢超阈值输出 valid=0）
  - 摆杆左右端标定 → 毫米坐标换算 / 旋转投影 / 到中心线垂直距离
  - 高光评分阈值自适应化（锁定曝光后暂用绝对阈值，跨差异大的环境再改成相对帧统计）

用法:
  /home/hao/vision_env/bin/python3 code/rollball_code/v1.1_beta.py            # 默认 RealSense 彩色流
  /home/hao/vision_env/bin/python3 code/rollball_code/v1.1_beta.py --source usb
  /home/hao/vision_env/bin/python3 code/rollball_code/v1.1_beta.py --source csi
  /home/hao/vision_env/bin/python3 code/rollball_code/v1.1_beta.py --selftest # 无窗口，跑通+测帧率
  /home/hao/vision_env/bin/python3 code/rollball_code/v1.1_beta.py --headless # 无窗口比赛/部署，串口发X最快，Ctrl-C 退出

按键（主窗口聚焦时）:
  c = 清空ROI（恢复整幅画面）    u = 切换去畸变
  s = 保存参数                   q/ESC = 退出
带窗口版需在有显示器/桌面的会话里跑（有 cv2 窗口）；--selftest / --headless 除外。
"""

import argparse
import glob
import json
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(HERE, "rollball_config.json")

PROJECT_ROOT = os.path.dirname(os.path.dirname(HERE))

# 复用 code/ready_code/camera_common 做相机标定读写 + 去畸变映射（唯一事实来源，不另写一份）
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code", "ready_code"))
import camera_common as cc  # noqa: E402

# 复用 code/opencv_code/ball_detector 的钢珠检测核心（HoughCircles + V通道高光多因素评分），
# 不依赖背景图、对光照更鲁棒。它自带的 EMA 平滑/续帧预测本文件不用（见下方主循环注释），
# 只借用逐帧的形状+高光打分本身。
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code", "opencv_code"))
from ball_detector import BallDetector  # noqa: E402

# 复用 code/task_code/serial_link 做串口热插拔连接管理（找口/等口/断线重连不重写）
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code", "task_code"))

from mcu_link import McuLink  # noqa: E402
from mjpeg_server import MjpegServer, get_lan_ip  # noqa: E402

# ---- 可调参数默认值（滑条会覆盖，s 保存回 JSON）----
# 注意：检测器(BallDetector)的一堆参数不摊在这里，整包存在 cfg["detector"] 里（复用它自带的
# as_dict/load_dict 序列化）。这里只放 ROI、去畸变、和 RealSense 成像固定值。
DEFAULT_CFG = {
    "roi_top": 0,          # ROI 上边界(像素行, 整幅去畸变图坐标)，0 表示待运行时初始化为整幅
    "roi_bottom": 0,       # ROI 下边界；roi_bottom<=roi_top 时视为"未设置"，用整幅高度
    "roi_left": 0,         # ROI 左边界(像素列)，0 表示待运行时初始化为整幅宽度
    "roi_right": 0,        # ROI 右边界；roi_right<=roi_left 时视为"未设置"，用整幅宽度
    "undistort": True,     # 项目约定：有 camera_calib.npz 就默认去畸变（u 键运行时切换）
    # RealSense 彩色成像固定值：起流后【先关自动曝光/自动白平衡、再设这三个】，让每次开机成像
    # 一致、可复现。滑条可实时调、s 存盘；换环境调一次即可。具体好值要上机调（曝光单位是微秒，
    # 和 camera_common 的 V4L2 刻度不是一套，别照抄）。应用时会按 sensor 实际量程夹紧，越界不报错。
    "rs_exposure": 150,        # 微秒(D4xx 色流)，控高光亮度——最关键
    "rs_gain": 16,             # 增益，越大越亮也越噪
    "rs_white_balance": 4600,  # 白平衡色温(K)
    # BallDetector 参数整包（param2/min_vmax/hi_v/半径/跟踪/置信度…），空=用其内置默认。
    "detector": {},
}

# 做成滑条的【检测器】参数（其余 BallDetector 参数只在 cfg["detector"] 里，靠改 JSON 调）。
# (滑条名 = BallDetector 属性名, 滑条最大值)
_DET_TRACKBARS = [
    ("param2", 100),      # HoughCircles 累加器阈值：越大越严（漏检↑误检↓）
    ("min_vmax", 255),    # 高光/亮度门槛：钢珠高光越暗就调低
    ("hi_v", 255),        # "最亮"阈值(高光分用)
    ("body_v", 255),      # 球体轮廓阈值(比hi_v低，圈出整个球体，二值化窗口/质心细化用)
    ("min_radius", 40),   # 圆最小半径(px)
    ("max_radius", 60),   # 圆最大半径(px)
]

# 做成滑条的【RealSense 成像】参数 (滑条名 = cfg 键, 滑条最大值)。仅 realsense 源有意义。
_RS_TRACKBARS = [
    ("rs_exposure", 2000),
    ("rs_gain", 128),
    ("rs_white_balance", 6500),
]


# ============================ 配置读写 ============================

def load_cfg():
    cfg = dict(DEFAULT_CFG)
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                for k in DEFAULT_CFG:
                    if k in data and data[k] is not None:
                        cfg[k] = data[k]
        except (json.JSONDecodeError, OSError, ValueError):
            pass  # 坏了就用默认，不让脚本崩
    return cfg


def save_cfg(cfg):
    try:
        tmp = CONFIG_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        os.replace(tmp, CONFIG_FILE)
        return True
    except OSError:
        return False


# ======================= 可替换的摄像头接口 =======================
# 统一约定：read() 返回一帧 BGR ndarray（失败返回 None），release() 释放。
# 想换别的采集来源（比如网络流），照这个接口再写一个类即可，主流程不用动。

class BaseCamera:
    def read(self):
        raise NotImplementedError

    def release(self):
        pass


def resolve_usb_video_path(index=None):
    """
    解析出实际可用的 USB 摄像头设备路径。
    【踩坑，实测复现过】摄像头掉线重连（或任何 USB 重新枚举）后 /dev/videoN 的编号
    会漂移——比如从 /dev/video0 变成了 /dev/video1，固定按 index 找会导致重连逻辑
    永远连不上。优先用 udev 生成的 /dev/v4l/by-id/*-video-index0 稳定 symlink
    （只要插的是同一个物理摄像头，路径不随编号漂移而变，"-video-index0" specifically
    是真正的 Video Capture 节点，不是 metadata 节点），原理同 task_code/serial_link.py
    用 by-path 锁定物理串口口。
    index 非 None 表示用户显式指定了 --index，直接用 /dev/video{index}，跳过自动解析。
    """
    if index is not None:
        return f"/dev/video{index}"
    hits = sorted(glob.glob("/dev/v4l/by-id/*-video-index0"))
    if hits:
        return hits[0]
    return "/dev/video0"  # 没有 by-id symlink（非 udev 环境）就退回默认编号


class UsbCamera(BaseCamera):
    """USB 摄像头：MJPG + 640x480，尽量高帧率。设备路径优先按 by-id 稳定解析（见上）。"""

    def __init__(self, index=None, width=640, height=480, fps=60):
        device = resolve_usb_video_path(index)
        self.device = device
        self.cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"打不开 USB 摄像头 {device}")
        # 注意顺序：先 MJPG 再设分辨率/帧率，否则默认 YUYV 只有 15fps。
        # 【踩坑】本机这颗 USB 摄像头 + V4L2 后端下设 CAP_PROP_BUFFERSIZE=1 会把
        # 采集帧率从 30 砍到 15，故不设缓冲大小。
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

    def read(self):
        ok, frame = self.cap.read()
        return frame if ok else None

    def release(self):
        try:
            self.cap.release()
        except Exception:
            pass


class CsiCamera(BaseCamera):
    """CSI 摄像头（树莓派 5 用 picamera2）。未装 picamera2 会给出明确提示。"""

    def __init__(self, width=640, height=480, fps=60):
        try:
            from picamera2 import Picamera2
        except ImportError as e:
            raise RuntimeError(
                "CSI 需要 picamera2，但当前环境未安装。\n"
                "  树莓派上一般用系统包：sudo apt install -y python3-picamera2\n"
                "  （注意 picamera2 依赖系统 libcamera，vision_env 里 pip 装通常不可用；\n"
                "   先用 --source usb 跑通，CSI 后端等装好 picamera2 再切。）"
            ) from e
        self.picam = Picamera2()
        conf = self.picam.create_video_configuration(
            main={"size": (width, height), "format": "RGB888"},
            controls={"FrameRate": fps},
        )
        self.picam.configure(conf)
        self.picam.start()
        time.sleep(0.3)  # 等自动增益/曝光稳定

    def read(self):
        arr = self.picam.capture_array()   # picamera2 的 RGB888 实际按 BGR 排布
        if arr is None:
            return None
        if arr.ndim == 3 and arr.shape[2] == 3:
            return arr  # 已是 BGR 顺序，直接用
        return arr

    def release(self):
        try:
            self.picam.stop()
        except Exception:
            pass


class RealSenseCamera(BaseCamera):
    """
    Intel RealSense 深度相机——只用它的【彩色流】当普通 2D 摄像头，不用深度/对齐/内参。
    起流写法参考 /home/hao/Desktop/luhao/ruikang/my/v2.8.py 的 start_pipeline()，
    但去掉了深度流、rs.align、intrinsics 这些只有用深度才需要的部分。
    """

    def __init__(self, width=640, height=480, fps=30,
                 exposure=None, gain=None, white_balance=None):
        try:
            import pyrealsense2 as rs
        except ImportError as e:
            raise RuntimeError(
                "RealSense 需要 pyrealsense2，但当前环境未安装。\n"
                "  /home/hao/vision_env/bin/python3 -m pip install pyrealsense2\n"
            ) from e
        self._rs = rs
        self._exposure = exposure
        self._gain = gain
        self._white_balance = white_balance
        self._color_sensor = None
        self.pipeline = rs.pipeline()

        # 【踩坑，实测复现过：这是"只有15fps"的真凶，不是网站也不是检测算法】
        # 本机这颗 D435 在 640x480 下【压根没有 bgr8 @60fps 这个档位】(SDK 只给 6/15/30fps)，
        # 请求 60fps 必然让 pipeline.start() 抛 "Couldn't resolve requests"。原来的降级策略是
        # 直接问 SDK 要"完全不设限的默认配置"——实测这个默认配置协商到的是 640x480 【rgb8】
        # @15fps，实测交付速率只有 ~12.6fps；而同一分辨率、正确格式(bgr8，不用再转 RGB2BGR)
        # 下其实是有 30fps 档位的，实测交付 ~25.9fps——整整快了一倍。所以降级时不要一步跳到
        # "完全不设限"，先按【同分辨率 + bgr8】从请求 fps 开始往下试几个常见档位(30/15/6)，
        # 试到能起流的最高档为止，全部失败了才真正退到完全不设限的兜底。
        profile = None
        tried_fps = []
        for try_fps in dict.fromkeys([fps, 30, 15, 6]):    # 去重但保留顺序
            cfg = rs.config()
            cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, try_fps)
            try:
                profile = self.pipeline.start(cfg)
                if try_fps != fps:
                    print(f"   ⚠️ RealSense {width}x{height}@{fps}fps(bgr8) 无此档位，"
                          f"改用同分辨率 {width}x{height}@{try_fps}fps(bgr8)")
                break
            except RuntimeError:
                tried_fps.append(try_fps)
                continue   # 同一个 pipeline 对象可以直接重试，不用重建
        if profile is None:
            print(f"   ⚠️ RealSense {width}x{height}(bgr8) 在 {tried_fps} 几档 fps 都协商失败，"
                  f"退回设备完全默认配置（可能更慢/格式不同，仅兜底用）…")
            cfg = rs.config()
            cfg.enable_stream(rs.stream.color)
            profile = self.pipeline.start(cfg)
        self.profile = profile
        # 起流成功后再锁成像（各分支都要锁）。
        self._lock_color_imaging()
        self._print_negotiated_profile(width, height, fps)

    def _print_negotiated_profile(self, req_w, req_h, req_fps):
        """
        【诊断】打印 SDK 实际协商到的彩色流分辨率/帧率，和请求值对照。
        帧率上不去时先看这行：如果协商到的 fps 本身就远低于请求值(比如请求60、协商到只有
        15~20)，那是 USB 带宽/驱动层面的限制（640x480 bgr8 未压缩每帧接近 1MB，USB2 口很
        可能扛不住高 fps），不是代码/检测算法的问题，换 USB3 口或降分辨率才有用；如果协商到
        的 fps 本身就够高，瓶颈就在别处（比如 detect() 耗时，见主循环里的 detect_ms_ema）。
        """
        try:
            vprofile = self.profile.get_stream(self._rs.stream.color).as_video_stream_profile()
            w, h, f = vprofile.width(), vprofile.height(), vprofile.fps()
            tag = "" if (w, h, f) == (req_w, req_h, req_fps) else "  ⚠️ 与请求值不同"
            print(f"   📷 RealSense 协商到的彩色流: {w}x{h}@{f}fps"
                  f"（请求 {req_w}x{req_h}@{req_fps}fps）{tag}")
        except Exception as e:
            print(f"   ⚠️ 读取 RealSense 实际协商流参数失败({e})，跳过此诊断打印。")

    def _get_color_sensor(self):
        """拿彩色 sensor（优先 first_color_sensor，退而扫 query_sensors 找支持 exposure 的）。"""
        rs = self._rs
        try:
            dev = self.profile.get_device()
        except Exception:
            return None
        try:
            s = dev.first_color_sensor()
            if s is not None:
                return s
        except Exception:
            pass
        try:
            for s in dev.query_sensors():
                if s.supports(rs.option.exposure):
                    return s
        except Exception:
            pass
        return None

    def _lock_color_imaging(self):
        """
        【关键：可复现的地基】关自动曝光 + 自动白平衡，再设固定 曝光/增益/白平衡。
        顺序同 camera_common.py 的 V4L2 约定：auto 必须先切手动，否则手动项 inactive 被静默忽略。
        每步都 supports()+try/except 防御（照 v2.8.py 的风格），设不了也不炸。
        """
        rs = self._rs
        s = self._get_color_sensor()
        self._color_sensor = s
        if s is None:
            print("   ⚠️ 未取到 RealSense 彩色 sensor，成像未锁定（自动曝光/白平衡仍开着）。")
            return
        for opt, val in ((rs.option.enable_auto_exposure, 0),
                         (rs.option.enable_auto_white_balance, 0)):
            try:
                if s.supports(opt):
                    s.set_option(opt, val)
            except Exception:
                pass
        self.set_color_options(self._exposure, self._gain, self._white_balance)
        print(f"   ✅ RealSense 成像已锁定：auto_exposure/auto_wb 关，"
              f"exposure={self._exposure} gain={self._gain} white_balance={self._white_balance}")

    def set_color_options(self, exposure=None, gain=None, white_balance=None):
        """
        运行时设固定曝光/增益/白平衡（供滑条实时下发）。None=不动该项。值按 sensor 实际量程
        夹紧再设，越界不报错（拖滑条到无效区间也不刷屏）。同时记住最新值，掉线重连后重设。
        """
        rs = self._rs
        s = self._color_sensor
        if s is None:
            return
        for opt, val, attr in ((rs.option.exposure, exposure, "_exposure"),
                               (rs.option.gain, gain, "_gain"),
                               (rs.option.white_balance, white_balance, "_white_balance")):
            if val is None:
                continue
            setattr(self, attr, val)
            try:
                if not s.supports(opt):
                    continue
                rng = s.get_option_range(opt)
                v = max(rng.min, min(float(val), rng.max))
                s.set_option(opt, v)
            except Exception:
                pass

    def read(self):
        try:
            ok, frames = self.pipeline.try_wait_for_frames(1000)
        except Exception:
            return None
        if not ok:
            return None
        color_frame = frames.get_color_frame()
        if not color_frame:
            return None
        img = np.asanyarray(color_frame.get_data()).copy()
        # 【踩坑，实测复现过：录像红蓝互换】上面主起流请求的是 bgr8，但 640x480@fps 协商
        # 失败会落到 __init__ 里那条【无格式】降级分支(cfg.enable_stream(rs.stream.color))，
        # 驱动此时可能给的是 rgb8。所以不能想当然按 bgr8 直接用，要按【实际协商到的格式】
        # 判断：只有真是 rgb8 时才转 BGR，否则原样返回。bgr8 不转、rgb8 才转，主路径/降级
        # 路径都正确——这套判断照抄仓库里唯一可用的参考实现 ruikang/my/v2.8.py 的约定。
        if color_frame.profile.format() == self._rs.format.rgb8:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    def release(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass


def open_camera(source, index, width, height, fps, rs_opts=None):
    """rs_opts=(exposure, gain, white_balance) 仅对 realsense 源生效（锁定成像用）。"""
    if source == "csi":
        return CsiCamera(width, height, fps)
    if source == "realsense":
        e, g, w = rs_opts if rs_opts else (None, None, None)
        return RealSenseCamera(width, height, fps, exposure=e, gain=g, white_balance=w)
    return UsbCamera(index, width, height, fps)


class CameraLost(Exception):
    """读帧时发现摄像头已掉线（供实时循环 catch 住继续等重连，不崩不退）。"""


class ResilientCamera(BaseCamera):
    """
    在任意 BaseCamera 后端外面包一层"设备不在就等、掉线就自动重连"（与
    code/task_code/camera_link.py::CameraLink 同一套思路，适配这里可替换的
    USB/CSI 摄像头接口）。摆杆小车行驶时 USB 口容易接触不良，这层保证掉线
    不炸程序、插回自动续上。

    用法：
        cam = ResilientCamera(lambda: open_camera(args.source, args.index, w, h, fps))
        cam.wait_and_open()          # 阻塞直到摄像头就绪（一开始没插好也不报错退出）
        frame = cam.read()           # 掉线抛 CameraLost
    实时循环里配合非阻塞的 try_open()（节流重试）使用，界面不会卡死。
    """

    def __init__(self, factory, log=print):
        self.factory = factory     # 无参可调用，返回一个新的 BaseCamera 子类实例
        self.log = log
        self.cam = None
        self._last_try = 0.0
        self._read_fail = 0

    @property
    def connected(self):
        return self.cam is not None

    def _open_once(self):
        """尝试新建一个底层摄像头实例并确认真的能读出一帧；成功返回 True。"""
        try:
            cam = self.factory()
        except Exception:
            return False
        try:
            frame = cam.read()
        except Exception:
            frame = None
        if frame is None:
            try:
                cam.release()
            except Exception:
                pass
            return False
        self.cam = cam
        self._read_fail = 0
        return True

    def wait_and_open(self, poll=0.5):
        """阻塞循环等待摄像头就绪并打开。期间每隔几秒提示一次。"""
        announced = False
        waited = 0.0
        while True:
            if self._open_once():
                self.log("✅ 摄像头已连接")
                return
            if not announced:
                self.log("⌛ 等待摄像头就绪…（检查 USB 摄像头是否插好）")
                announced = True
            time.sleep(poll)
            waited += poll
            if waited >= 5.0:
                self.log("   …仍未连接摄像头，请检查 USB 接口/连线。")
                waited = 0.0

    def try_open(self, min_interval=1.0):
        """非阻塞尝试连接一次（供实时循环调用）。已连或连上返回 True。"""
        if self.cam is not None:
            return True
        now = time.time()
        if now - self._last_try < min_interval:
            return False
        self._last_try = now
        return self._open_once()

    def read(self):
        """读一帧 BGR；掉线抛 CameraLost；偶发单帧失败返回 None（调用方跳过即可）。"""
        if self.cam is None:
            raise CameraLost()
        try:
            frame = self.cam.read()
        except Exception:
            frame = None
        if frame is not None:
            self._read_fail = 0
            return frame
        self._read_fail += 1
        if self._read_fail >= 15:
            self._drop()
            raise CameraLost()
        return None

    def _drop(self):
        try:
            if self.cam:
                self.cam.release()
        except Exception:
            pass
        self.cam = None
        self._read_fail = 0

    def set_color_options(self, exposure=None, gain=None, white_balance=None):
        """把 RealSense 成像设置透传给底层 cam（其它后端没有此方法就静默跳过）。"""
        if self.cam is not None and hasattr(self.cam, "set_color_options"):
            self.cam.set_color_options(exposure, gain, white_balance)

    def release(self):
        self._drop()


# ============================ 小工具 ============================

def undistort_frame(frame, maps, on):
    """去畸变：预计算映射 + cv2.remap（不逐帧 undistort，为实时性能）。on=False 或无映射时原样返回。"""
    if on and maps is not None:
        return cv2.remap(frame, maps[0], maps[1], cv2.INTER_LINEAR)
    return frame


def apply_rect_roi(frame, top, bottom, left, right):
    """
    矩形 ROI：上下边界裁通道高度（摆杆是横向管道，凹槽居中），左右边界裁通道内感兴趣的
    一段（比如只关心摆杆中段，排除两端支架/背景干扰）。返回 (子图, (x0, y0))。
    """
    H, W = frame.shape[:2]
    y0 = max(0, min(int(top), H - 2))
    y1 = max(y0 + 1, min(int(bottom), H))
    x0 = max(0, min(int(left), W - 2))
    x1 = max(x0 + 1, min(int(right), W))
    return frame[y0:y1, x0:x1], (x0, y0)


# ======================= 检测：形状+高光（复用 BallDetector） =======================
# 检测核心整套复用 code/opencv_code/ball_detector.py 的 BallDetector：
#   cands = detector.detect(roi_bgr)   -> [(cx,cy,r,score), ...]（ROI 内坐标）
# BallDetector 本身是有状态的（内建 EMA 平滑 pick_primary() + 漏检时的速度外推续帧），但
# 本文件的主循环【不用这两个特性】——项目要求发给下位机的是每帧真实检测值，不要平滑/预测，
# 滤波交给下位机做。做法：不调用 pick_primary()，改用 cands[0]（本帧原始最高分候选）；
# 且每帧检测完立即 detector.reset()，让跟踪状态不跨帧存活，从根上掐掉续帧续帧行为。
# 详见主循环里 cands = detector.detect(sub) 那一段的注释。


# ============================ 滑条窗口 (Tuning window) ============================
# 每个滑条名字直接用参数的英文名(ASCII)，从上到下顺序 = 下面 _TRACKBAR_ORDER 的顺序。
# 原来那块 PIL 画的【中文说明面板】已移除：实测单帧 25~40ms、占 CPU 70%+，还拖慢串口发送；
# 现在只保留 图像+掩膜 预览 + 本滑条窗口，中文含义启动时打印到终端(见 SLIDER_HELP)。

TUNE_WIN = "Tuning (rollball v1.1b)"

# 【性能】GUI 绘制+imshow 会抢 CPU、拖慢"每读到一帧有效数据就发 X"的主循环；重的画面显示
# 不需要跟检测/发送同步，限流到这个间隔画一次(~15Hz)，把 CPU 让给采集+检测+串口。
# 完全不要窗口(最高发送频率)用 --headless。按键(b/c/u/s/q)也在这个节奏里轮询，够灵敏。
DISPLAY_REFRESH_INTERVAL = 1.0 / 15.0

# ROI 四条边界滑条名（最大值运行时按帧尺寸定）。检测器/成像滑条见 _DET_TRACKBARS/_RS_TRACKBARS。
_ROI_TRACKBARS = ("roi_top", "roi_bottom", "roi_left", "roi_right")

# 启动时打印到终端的滑条说明（中文；顺序同 Tuning 窗口从上到下）。
# 注意：滑条本身在窗口里的名字必须是英文/ASCII——本机 Qt 高亮组件缺字体，中文滑条名显示不出来。
SLIDER_HELP = [
    "Tuning 窗口滑条说明(从上到下):",
    "  roi_top/bottom/left/right  ROI 上/下/左/右边界(像素)",
    "  param2       圆检测严格度(HoughCircles param2)：越大越严(漏检↑误检↓)",
    "  min_vmax     高光/亮度门槛：钢珠高光越暗就调低(否则good圆被拒)",
    "  hi_v         '最亮'阈值(高光分用)",
    "  min_radius   圆最小半径(px)",
    "  max_radius   圆最大半径(px)",
    "  rs_exposure  RealSense 曝光(微秒)：控高光亮度，最关键(仅 realsense 源)",
    "  rs_gain      RealSense 增益(越大越亮也越噪)",
    "  rs_white_balance  RealSense 白平衡色温(K)",
]

# 网页"使用说明"面板的内容——纯 HTML 片段，注入 MjpegServer.doc_html，点"使用说明"按钮
# 弹出。跟 SLIDER_HELP(打印到终端) 覆盖同一批参数，但写得更详细：每个参数是什么、什么情况
# 下往哪个方向调。集中写在这里，改参数含义/调参经验时只改这一处，网页/终端不用分别改两份。
WEB_DOC_HTML = """
<h3>ROI（感兴趣区域）</h3>
<p><b>roi_top / roi_bottom / roi_left / roi_right</b> —— 画面里参与检测的矩形范围(像素)，
外面的区域完全不送进 Hough 圆检测，画面上用黄色框线标出。作用：缩小搜索范围排除管道外的
干扰圆、加快检测速度。调法：把黄框拖到刚好框住摆杆管道的凹槽区域即可，改完立即生效；
若管道装配位置变了或摄像头挪动过，需要重新框一次。</p>

<h3>圆检测 (Hough) 参数</h3>
<p><b>param1</b> —— Canny 边缘检测高阈值。太低会把噪声也当边缘、误检变多；太高会丢失偏弱的
边缘、导致漏检。一般不用大改，管道内壁反光复杂时可以适当调高排除杂边缘。</p>
<p><b>param2</b>（Hough严格度） —— 圆心累加器阈值，是最常调的一个：<b>调大</b>=更严格
(误检↓但漏检↑，球有时会检测不到)；<b>调小</b>=更宽松(漏检↓但误检↑，容易把别的亮斑当成球)。
球老是丢检，第一个先试着调小这个。</p>
<p><b>min_radius / max_radius</b> —— 圆半径搜索范围(像素)，要覆盖球在画面里实际呈现的像素
半径(离镜头越近半径越大)。范围卡得越窄，速度越快、误检越少，但离镜头远近变化大时要留一点
余量，否则太远/太近的球会因半径超出范围被直接过滤掉。</p>
<p><b>blur_ksize</b> —— 中值模糊核大小(奇数)，检测前先去噪声。太小噪声压不住导致误检；
太大会把球的边缘也模糊掉导致漏检，一般 5 附近就够。</p>

<h3>高光/亮度打分参数（决定候选圆的"像不像球"）</h3>
<p><b>hi_v</b>（最亮阈值） —— 高于这个亮度的像素比例算作"高光分"，钢珠反光越强这个可以设
高一些、更精准锁定高光点；反光弱/环境暗时调低，否则高光分永远拿不到分。</p>
<p><b>min_vmax</b>（最低亮度门槛，硬拒绝） —— 候选区域里最大亮度低于这个值就直接判负、
不管其他因素多好。<b>球一直检测不到，先检查这个是不是设太高了</b>——环境变暗后这个门槛
没跟着降，会把所有候选都拒掉。</p>
<p><b>min_vstd</b>（最低对比度门槛，硬拒绝） —— 候选区域亮度标准差(对比度)低于这个值就直接
判负，用来排除"一片均匀亮度、没有球体轮廓起伏"的假阳性(比如反光板本身)。背景本身很亮很
均匀、总是误检成球时可以调高这个。</p>
<p><b>body_v</b>（球体轮廓阈值） —— 比 hi_v 低很多，用来圈出整个球体轮廓(不只是高光点)，
只用于二值化窗口显示 + 主目标质心细化，不参与候选打分。<b>右边(或网页第二路)的黑白掩膜
画面就是这个阈值二值化后的效果</b>——白色区域应该正好覆盖整颗球、不多不少：白色区域明显小于
球体就调低，明显把背景也圈进来了(一片白)就调高。</p>

<h3>NMS（去重）</h3>
<p><b>nms_iou</b> —— 多个重叠候选圆合并去重的距离阈值(按半径归一化)。调小=去重更激进，
容易把两个靠得很近的圆错误合并成一个；调大=去重更宽松，容易在同一颗球上保留多个重复候选。
一般不用大改。</p>

<h3>RealSense 相机成像参数（仅 --source realsense 生效）</h3>
<p><b>rs_exposure</b>（曝光，微秒） —— <b>全场最关键的一个参数</b>，直接决定画面整体亮度
和球面高光的强弱。太暗看不清高光、检测不到球；太亮容易过曝，球面高光糊成一片白反而丢失
形状/对比度信息。<b>换了光照环境(比如换场地、换灯光)，第一个先调这个</b>，配合右侧黑白
掩膜画面观察球体轮廓是否清晰。</p>
<p><b>rs_gain</b>（增益） —— 让画面整体变亮，但同时会引入更多噪点。曝光已经调到上限还是
不够亮时，再考虑加增益；能靠曝光解决就优先调曝光，增益是补充手段。</p>
<p><b>rs_white_balance</b>（白平衡色温，K） —— 控制画面色调偏冷/偏暖。检测本身只看 HSV
的 V(亮度)通道，理论上白平衡对检测结果影响很小，主要是让人眼看画面顺眼、颜色不失真；
画面明显偏黄/偏蓝时调这个。</p>

<h3>关于保存</h3>
<p>拖动滑块会立即生效（相机/检测器实时应用），并在停止拖动约 1 秒后自动写入配置文件
(rollball_config.json)，无需手动操作；也可以随时点"调参数"面板里的"保存参数"按钮立即
落盘。程序下次启动、或摄像头掉线重连时都会自动读取这份配置并重新应用。</p>
"""


def setup_trackbars(cfg, detector, H, W, with_rs):
    cv2.namedWindow(TUNE_WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(TUNE_WIN, 500, 620)
    _n = lambda v: None
    roi_max = {"roi_top": max(1, H - 1), "roi_bottom": H,
               "roi_left": max(1, W - 1), "roi_right": W}
    for name in _ROI_TRACKBARS:
        cv2.createTrackbar(name, TUNE_WIN, int(cfg[name]), roi_max[name], _n)
    # 检测器滑条：初值取自 detector 当前属性（已由 cfg["detector"] 载入）
    for name, mx in _DET_TRACKBARS:
        cv2.createTrackbar(name, TUNE_WIN, int(getattr(detector, name)), mx, _n)
    if with_rs:
        for name, mx in _RS_TRACKBARS:
            cv2.createTrackbar(name, TUNE_WIN, int(cfg[name]), mx, _n)


def read_trackbars(cfg, detector, with_rs):
    """读滑条：ROI 写 cfg；检测器参数实时写进 detector 对象；成像参数写 cfg（应用在主循环里做）。"""
    top = cv2.getTrackbarPos("roi_top", TUNE_WIN)
    bottom_raw = cv2.getTrackbarPos("roi_bottom", TUNE_WIN)
    cfg["roi_top"] = top
    cfg["roi_bottom"] = max(top + 10, bottom_raw)  # 至少 10px 高度，避免空 ROI
    left = cv2.getTrackbarPos("roi_left", TUNE_WIN)
    right_raw = cv2.getTrackbarPos("roi_right", TUNE_WIN)
    cfg["roi_left"] = left
    cfg["roi_right"] = max(left + 10, right_raw)   # 至少 10px 宽度
    for name, _mx in _DET_TRACKBARS:
        setattr(detector, name, cv2.getTrackbarPos(name, TUNE_WIN))
    if detector.param2 < 1:
        detector.param2 = 1  # HoughCircles param2 不能为 0
    if detector.max_radius <= detector.min_radius:
        detector.max_radius = detector.min_radius + 1
    if with_rs:
        for name, _mx in _RS_TRACKBARS:
            cfg[name] = cv2.getTrackbarPos(name, TUNE_WIN)


# ============================ 绘制叠加 ============================

def draw_roi_highlight(canvas, top, bottom, left, right, dim=True):
    """
    画四条黄色 ROI 边界线（上/下/左/右）。dim=True 时额外把 ROI 外区域调暗突出重点
    （本地调参窗口用，习惯了这个效果不改）；dim=False 时画面保持原始真实亮度不处理，
    只叠黄框做参考（网页主画面用——网页要看到的是"真实画面里球在哪"，调暗了反而看不清
    ROI 外的实际情况）。
    """
    H, W = canvas.shape[:2]
    if dim:
        dimmed = canvas.astype(np.float32) * 0.35
        mask = np.ones((H, W), dtype=bool)
        mask[top:bottom, left:right] = False   # ROI 内不调暗
        canvas[mask] = dimmed[mask].astype(np.uint8)
    cv2.line(canvas, (0, top), (W, top), (0, 255, 255), 2, cv2.LINE_AA)
    cv2.line(canvas, (0, min(bottom, H - 1)), (W, min(bottom, H - 1)), (0, 255, 255), 2, cv2.LINE_AA)
    cv2.line(canvas, (left, 0), (left, H), (0, 255, 255), 2, cv2.LINE_AA)
    cv2.line(canvas, (min(right, W - 1), 0), (min(right, W - 1), H), (0, 255, 255), 2, cv2.LINE_AA)
    return canvas


def draw_detections(canvas, cands, primary, x_offset, y_offset):
    """
    把 ROI 内坐标的候选/主目标画到整幅画面上（加 (x_offset,y_offset) 换算成绝对坐标）。
    cands/primary 是 BallDetector 的 (cx,cy,r,score) 元组（primary 可为 None）。
    """
    for (cx, cy, r, sc) in cands:
        cv2.circle(canvas, (int(cx + x_offset), int(cy + y_offset)), int(r),
                    (0, 200, 0), 1, cv2.LINE_AA)
    if primary is not None:
        bx, by = int(primary[0] + x_offset), int(primary[1] + y_offset)
        cv2.circle(canvas, (bx, by), int(primary[2]), (0, 255, 0), 2, cv2.LINE_AA)
        cv2.circle(canvas, (bx, by), 3, (0, 255, 0), -1)
        cv2.putText(canvas, f"({bx},{by}) s={primary[3]:.1f}",
                    (bx + 8, by), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
    return canvas


def draw_hud(canvas, fps, primary, detect_ms=None):
    state = f"BALL s={primary[3]:.1f}" if primary is not None else "NO BALL"
    det_s = f"  det={detect_ms:.0f}ms" if detect_ms is not None else ""
    cv2.putText(canvas, f"{fps:4.1f}FPS  {state}{det_s}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 255, 255), 2, cv2.LINE_AA)
    return canvas


def build_detection_overlay(frame, cfg, cands, primary, x0, y0, fps, mcu, detect_ms=None, dim_roi=True):
    """
    整幅画面 + ROI 高亮 + 检测结果 + HUD。detect_ms：detector.detect() 单帧耗时(ms)，
    HUD 上显示，定位帧率瓶颈用。dim_roi：透传给 draw_roi_highlight，True=本地 GUI 左图
    的老样子(ROI 外调暗)，False=网页主画面(真实原始亮度+黄框参考线，不调暗)。
    """
    canvas = frame.copy()
    draw_roi_highlight(canvas, cfg["roi_top"], cfg["roi_bottom"], cfg["roi_left"], cfg["roi_right"], dim=dim_roi)
    draw_detections(canvas, cands, primary, x0, y0)
    draw_hud(canvas, fps, primary, detect_ms)
    if mcu is not None:
        mcu_hud = (f"MCU:{'ON' if mcu.mcu_online else '--'} "
                   f"REC:{('TASK' + str(mcu.current_task_id)) if mcu.recording else '-'} "
                   f"TXx#{mcu.tx_x_count}")
        cv2.putText(canvas, mcu_hud, (8, 42), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 255, 255), 2, cv2.LINE_AA)
    return canvas


def build_ball_mask_overlay(detector, cfg, cands, primary, x0, y0, sub_shape, H, W):
    """
    球体二值化掩膜整幅画布：ball_detector.BallDetector._refine_center() 对主目标做局部
    V 通道阈值(body_v)+闭运算+连通域筛出的完整球体轮廓(球=白/背景=黑)，供【本地 GUI
    右图】和【局域网 /stream/bin 推流】共用一份，不重复画两次。没有主目标/细化失败时
    last_mask_full 是 None，画一块纯黑。
    """
    if detector.last_mask_full is not None:
        ball_mask = detector.last_mask_full
    else:
        ball_mask = np.zeros(sub_shape, np.uint8)
    canvas = np.zeros((H, W, 3), np.uint8)
    canvas[y0:y0 + sub_shape[0], x0:x0 + sub_shape[1]] = cv2.cvtColor(ball_mask, cv2.COLOR_GRAY2BGR)
    draw_detections(canvas, cands, primary, x0, y0)  # 复用同款绝对坐标画法，圆圈/坐标标签跟左图一致
    cv2.line(canvas, (0, cfg["roi_top"]), (W, cfg["roi_top"]), (0, 255, 255), 1, cv2.LINE_AA)
    cv2.line(canvas, (0, cfg["roi_bottom"] - 1), (W, cfg["roi_bottom"] - 1), (0, 255, 255), 1, cv2.LINE_AA)
    cv2.line(canvas, (cfg["roi_left"], 0), (cfg["roi_left"], H), (0, 255, 255), 1, cv2.LINE_AA)
    cv2.line(canvas, (cfg["roi_right"] - 1, 0), (cfg["roi_right"] - 1, H), (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"BALL MASK (V>body_v={detector.body_v})", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    return canvas




# ============================ 主程序 ============================

def run_gui(args):
    cfg = load_cfg()
    gui_on = not getattr(args, "headless", False)   # --headless: 不开任何窗口/滑条
    with_rs = (args.source == "realsense")          # 成像滑条/锁定只对 realsense 源有意义
    print(f"⚙️  配置: {CONFIG_FILE}")
    if not gui_on:
        print("🖥️  headless 模式：不开窗口/滑条，参数用 rollball_config.json 已保存值；退出按 Ctrl-C。")

    # 检测器：整套复用 ball_detector.BallDetector（形状+高光，不依赖背景图）。参数整包从
    # cfg["detector"] 载入（没有就用其内置默认），滑条实时改它、s 存回。
    detector = BallDetector().load_dict(cfg.get("detector", {}))

    # 局域网实时画面：浏览器打开这个地址就能看当前摄像头画面，不用接显示器/开 GUI 窗口
    # （headless 比赛模式尤其有用）。只用标准库 http.server，不装额外依赖。
    # 【default 开，2026-07-30 实测确认】曾经因为怀疑它拖慢帧率，一度改成默认关、
    # 要 --stream 手动开。后来查明真凶其实是 RealSense 640x480@60fps(bgr8) 请求不到、
    # 协商降级成 rgb8@15fps（见 RealSenseCamera 里的降级策略注释），跟网站无关；修好相机
    # 协商后 headless 稳定 30fps，上机实测【开着网站也一样有 30 多帧】——因为 JPEG 编码在
    # 后台线程做（cv2.imencode 会释放 GIL，能吃到 Pi5 的另一个核），且只有 has_clients=True
    # （真有人在看）时才会被触发去编码，平时几乎零开销。确认不是瓶颈后改回默认开，用 --no-stream
    # 显式关掉（比如要绝对榨干最后一点余量、或者不想暴露局域网服务时）。
    stream_server = None
    if not args.no_stream:
        stream_server = MjpegServer(port=args.stream_port)
        stream_server.start()
        print(f"🌐 局域网实时画面: http://{get_lan_ip()}:{args.stream_port}/ "
              f"（同一局域网/热点下的手机、电脑浏览器打开即可，摄像头就绪前先显示占位画面）")
    else:
        print("🌐 局域网实时画面: 已禁用 (--no-stream)")

    # 网页参数滑块接线：get/set 直接读写 detector 属性(立即生效)，POST 落地的改动记一个
    # "脏"时间戳，主循环里距上次改动静默 ≥1s 才真正 save_cfg() 落盘——跟本地按 s 键存盘
    # 是同一个 save_cfg，只是触发时机换成"滑块停下来 1s"，避免拖动滑块时每次都写文件。
    params_dirty = {"since": None}
    # ROI 四个边界 + RealSense 成像三个值都存在 cfg（不是 detector）里，跟检测器参数走
    # 同一套网页 get/set 接口，_set_param 里按 name 属于哪个集合分流去改 cfg 还是 detector。
    _ROI_KEYS = {"roi_top", "roi_bottom", "roi_left", "roi_right"}
    # 只有 realsense 源才有这三个成像选项（曝光/增益/白平衡），USB/CSI 源没有对应硬件接口，
    # 不摆上网页面板，避免调了也没用。
    _RS_KEYS = {"rs_exposure", "rs_gain", "rs_white_balance"} if with_rs else set()
    if stream_server is not None:
        # 【只列真正对 v1.1_beta.py 这套用法生效的参数】BallDetector.TUNABLE 里还有一批
        # 跟踪/EMA 相关的参数(track_max_dist/match_dist_factor/vel_alpha/vel_decay/
        # conf_inc/conf_dec/ema_alpha)——本文件每帧检测完立即 detector.reset()、也从不调
        # pick_primary()（见主循环里那段"不做时间维度平滑"的注释），这些参数在这套用法下
        # 根本走不到对应代码分支，摆上滑块调了也不会有任何效果，所以不放进来，以免误导。
        # 下面这 10 个是 detect()→_score() 真正每帧都会用到的，中间 4 个 ROI 边界决定
        # Hough 只在画面哪个矩形范围内找圆，最后 3 个是 RealSense 成像固定值（仅 realsense
        # 源才追加，见 _RS_TRACKBARS 里同样的量程）：
        stream_server.param_spec = [
            ("param1", 1, 300, 1, "Canny高阈值"),
            ("param2", 1, 100, 1, "Hough严格度"),
            ("min_radius", 1, 60, 1, "最小半径"),
            ("max_radius", 1, 80, 1, "最大半径"),
            ("blur_ksize", 1, 15, 2, "中值模糊核"),
            ("hi_v", 0, 255, 1, "最亮阈值(高光分)"),
            ("min_vmax", 0, 255, 1, "最低亮度门槛(硬拒绝)"),
            ("min_vstd", 0, 60, 1, "最低对比度门槛(硬拒绝)"),
            ("body_v", 0, 255, 1, "球体轮廓阈值"),
            ("nms_iou", 0.0, 1.0, 0.01, "NMS重叠阈值"),
            ("roi_top", 0, args.height, 1, "ROI上边界(y)"),
            ("roi_bottom", 0, args.height, 1, "ROI下边界(y)"),
            ("roi_left", 0, args.width, 1, "ROI左边界(x)"),
            ("roi_right", 0, args.width, 1, "ROI右边界(x)"),
        ] + ([
            ("rs_exposure", 0, 2000, 1, "曝光(微秒)"),
            ("rs_gain", 0, 128, 1, "增益"),
            ("rs_white_balance", 0, 6500, 1, "白平衡色温(K)"),
        ] if with_rs else [])
        stream_server.get_params = lambda: {
            name: (cfg[name] if name in _ROI_KEYS or name in _RS_KEYS else getattr(detector, name))
            for name, *_ in stream_server.param_spec
        }

        def _set_param(name, value):
            if name in _ROI_KEYS:
                cfg[name] = int(round(value))
                # 跟本地 ROI 滑条(read_trackbars)一样的最小尺寸保护：网页顺序拖两个边界时
                # 中间状态可能短暂 top>=bottom 或 left>=right，这里每次都重新校验一遍，
                # 避免拿一个空/负的 ROI 去裁图把后面的检测搞崩。
                if cfg["roi_bottom"] <= cfg["roi_top"]:
                    cfg["roi_bottom"] = cfg["roi_top"] + 10
                if cfg["roi_right"] <= cfg["roi_left"]:
                    cfg["roi_right"] = cfg["roi_left"] + 10
            elif name in _RS_KEYS:
                # 只落 cfg，不在这里直接下发给相机——主循环每圈都会比较 cfg 与 last_rs
                # 是否变化、变了才调用 cam.set_color_options()（见主循环那段，gui/headless
                # 两种模式都会跑到），这里不用重复一份下发逻辑。
                cfg[name] = int(round(value))
            else:
                setattr(detector, name, value)
            params_dirty["since"] = time.time()
            # 【踩坑】带 GUI 窗口时，主循环每帧都会用本地 Tuning 滑条(检测器参数+ROI+成像
            # 都有)的当前位置覆盖回 cfg/detector（见 read_trackbars）——网页这里刚设完，
            # 下一帧(~30ms后)就被本地滑条的旧值冲掉，网页操作看起来"没生效"。这里把本地
            # 滑条位置也同步过去，两边就不会互相打架了。headless 模式没有这个窗口、或者
            # 这个参数没有对应本地滑条(比如 param1/blur_ksize/min_vstd/nms_iou，只在网页
            # 上暴露)，try/except 兜底忽略。
            if gui_on:
                try:
                    sync_val = cfg[name] if (name in _ROI_KEYS or name in _RS_KEYS) else value
                    cv2.setTrackbarPos(name, TUNE_WIN, int(round(sync_val)))
                except cv2.error:
                    pass

        stream_server.set_param = _set_param

        def _save_now():
            cfg["detector"] = detector.as_dict()
            ok = save_cfg(cfg)
            params_dirty["since"] = None  # 已经落盘，撤销待定的防抖持久化
            return ok

        stream_server.on_save = _save_now
        stream_server.doc_html = WEB_DOC_HTML

    def camera_factory():
        # 把 cfg 里的成像固定值传给 RealSense（掉线重连也会用这些值重新锁定）。
        rs_opts = (cfg["rs_exposure"], cfg["rs_gain"], cfg["rs_white_balance"])
        return open_camera(args.source, args.index, args.width, args.height, args.fps, rs_opts)

    cam = ResilientCamera(camera_factory)
    print(f"📷 摄像头: source={args.source} index={args.index} 目标 {args.width}x{args.height}@{args.fps}")
    cam.wait_and_open()   # 阻塞等到摄像头就绪；支持一开始没插好的情况

    # 拿一帧确定尺寸
    frame = None
    for _ in range(30):
        try:
            frame = cam.read()
        except CameraLost:
            cam.wait_and_open()
            continue
        if frame is not None:
            break
    if frame is None:
        print("❌ 摄像头读不到帧，退出。")
        cam.release()
        return
    H, W = frame.shape[:2]
    if cfg["roi_bottom"] <= cfg["roi_top"]:
        cfg["roi_top"], cfg["roi_bottom"] = 0, H
    if cfg["roi_right"] <= cfg["roi_left"]:
        cfg["roi_left"], cfg["roi_right"] = 0, W
    print(f"   实际帧尺寸 {W}x{H}，ROI=x[{cfg['roi_left']},{cfg['roi_right']}] y[{cfg['roi_top']},{cfg['roi_bottom']}]")

    # 单片机通信：握手(PING/PONG) + 题目录像(TASK START/STOP->ACK) + 持续发 X 坐标。
    # 非阻塞尝试连接，不卡视觉启动；真正开始回 PONG 要等下面主循环前置初始化都做完。
    mcu = None
    if not args.no_mcu:
        mcu = McuLink(video_dir=os.path.join(PROJECT_ROOT, "Formal_code", "mp4"),
                      video_fps=args.fps, port=args.mcu_port)
        print(f"🔌 单片机串口: port={args.mcu_port or '自动探测(CH340/ttyUSB)'}")
        mcu.try_open()
    else:
        print("🔌 单片机串口: 已禁用 (--no-mcu)")

    # 去畸变：复用 camera_common 的标定（camera_calib.npz），预计算映射，喂给检测/背景的
    # 都是去畸变后的画面，几何一致（项目约定：有标定就默认开，u 键运行时切换）。
    calib = cc.load_calibration()
    undist_maps = cc.build_undistort_maps(
        calib["camera_matrix"], calib["dist_coeffs"], (W, H), alpha=0.0) if calib else None
    undist_on = bool(cfg.get("undistort", True)) and undist_maps is not None
    if calib is not None:
        cs = calib.get("image_size")
        if cs is not None and tuple(cs) != (W, H):
            print(f"   ⚠️ 标定分辨率 {tuple(cs)} 与当前 {W}x{H} 不一致，去畸变可能不准。")
        print(f"   已载入标定 (RMS={calib['rms']:.3f}px)，去畸变={'开' if undist_on else '关'}（u 键切换）")
    else:
        print("   ⚠️ 未找到 camera_calib.npz，用原始画面（无法去畸变）。")

    frame = undistort_frame(frame, undist_maps, undist_on)

    PANEL_W = W * 2   # 主窗口 = 左:整幅彩色(带高亮+检测) | 右:球体二值掩膜(黑白，调参用)

    main_win = "RollBall detect (v1.1 beta)"
    if gui_on:
        cv2.namedWindow(main_win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(main_win, PANEL_W, H)
        setup_trackbars(cfg, detector, H, W, with_rs)

    # 记住上次下发给相机的成像值，滑条一变才重新 set_option（不每帧白设）。
    last_rs = (cfg["rs_exposure"], cfg["rs_gain"], cfg["rs_white_balance"])

    fps_ema = 0.0
    detect_ms_ema = 0.0   # 【诊断】detector.detect() 单帧耗时的 EMA(毫秒)，定位 15fps 瓶颈用
    if gui_on:
        print("   窗口: 主窗口 = [左: 画面+ROI+检测结果(彩色)] | [右: 球体二值掩膜 V>body_v(黑白，调参用)]")
        for _l in SLIDER_HELP:      # 滑条中文含义打印到终端（顺序同 Tuning 窗口从上到下）
            print("   " + _l)
        print("   按键: [c]清空ROI(恢复整幅) [u]切换去畸变 [s]保存参数 [q]退出")
    print("   摄像头掉线会自动等待重连，不用重启程序。")

    if mcu is not None:
        mcu.ready = True  # 相机/检测/显示都已就绪，满足协议"初始化完成才回PONG"的前提

    prev_t = time.perf_counter()
    last_draw = 0.0    # GUI 渲染限流计时（见 DISPLAY_REFRESH_INTERVAL）
    last_stat = 0.0    # headless 下周期性打印状态的计时
    try:
     while True:
        if not cam.connected:
            if cam.try_open():
                print("✅ 摄像头已重连")
                try:
                    nf = cam.read()
                except CameraLost:
                    nf = None
                if nf is not None and nf.shape[:2] != (H, W):
                    nh, nw = nf.shape[:2]
                    print(f"   ⚠️ 重连后分辨率变为 {nw}x{nh}（原 {W}x{H}），重建去畸变映射，"
                          f"ROI 重置为整幅。")
                    W, H = nw, nh
                    undist_maps = cc.build_undistort_maps(
                        calib["camera_matrix"], calib["dist_coeffs"], (W, H), alpha=0.0) if calib else None
                    undist_on = bool(cfg.get("undistort", True)) and undist_maps is not None
                    detector.reset()   # 分辨率变了，跟踪的历史位置作废，清空重来
                    cfg["roi_top"], cfg["roi_bottom"] = 0, H
                    cfg["roi_left"], cfg["roi_right"] = 0, W
                    PANEL_W = W * 2
                    if gui_on:
                        cv2.resizeWindow(main_win, PANEL_W, H)
                        cv2.setTrackbarPos("roi_top", TUNE_WIN, 0)
                        cv2.setTrackbarPos("roi_bottom", TUNE_WIN, H)
                        cv2.setTrackbarPos("roi_left", TUNE_WIN, 0)
                        cv2.setTrackbarPos("roi_right", TUNE_WIN, W)
            else:
                if gui_on:
                    placeholder = np.zeros((H, PANEL_W, 3), dtype=np.uint8)
                    cv2.putText(placeholder, "waiting for camera... (check USB)",
                                (20, H // 2), cv2.FONT_HERSHEY_SIMPLEX,
                                0.9, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.imshow(main_win, placeholder)
                    if (cv2.waitKey(100) & 0xFF) in (ord("q"), 27):
                        break
                else:
                    time.sleep(0.1)  # headless 无 waitKey，睡一下避免空转烧 CPU
                continue

        try:
            frame = cam.read()
        except CameraLost:
            print("⚠️ 摄像头掉线，等待重连…")
            continue
        if frame is None:
            continue
        frame = undistort_frame(frame, undist_maps, undist_on)
        if gui_on:
            read_trackbars(cfg, detector, with_rs)  # headless 无滑条，用 JSON 载入的 cfg/detector
        if with_rs:
            # 【放在 gui_on 之外】cfg 里的成像值不仅可能来自本地滑条(read_trackbars)，也可能
            # 来自网页参数面板(_set_param 直接改 cfg，见上文)——headless 模式没有本地滑条，
            # 全靠网页改 cfg，这里必须每圈都检查一遍变没变，不能只在 gui_on 分支里做。
            rs_now = (cfg["rs_exposure"], cfg["rs_gain"], cfg["rs_white_balance"])
            if rs_now != last_rs:
                cam.set_color_options(*rs_now)  # 成像值一变才实时下发给相机
                last_rs = rs_now
        sub, (x0, y0) = apply_rect_roi(frame, cfg["roi_top"], cfg["roi_bottom"],
                                        cfg["roi_left"], cfg["roi_right"])

        # 形状+高光检测（复用 BallDetector，不依赖背景图）。cands 是 ROI 内坐标。
        #
        # 【不用 pick_primary()，也不让状态跨帧存活——要"最实时"而不是"类卡尔曼"】
        # BallDetector 内建两层时间维度上的平滑/预测，这里都不要：
        #   ① pick_primary() 会做 EMA 平滑（新值和上次平滑值按 ema_alpha 混合），位置会拖尾；
        #   ② detect() 内部若本帧没找到候选、但历史置信度>0，会用【速度外推的预测位置】顶
        #      替续报——这也是一种类卡尔曼的"续帧"效果，发出去的坐标不是真的这一帧看到的。
        # 直接取本帧打分最高的原始候选(cands[0])当 primary，并且每帧结束前 reset() 掉检测器
        # 的跟踪状态（位置/速度/置信度全清空）：下一帧进 detect() 时 self._px 是 None，②那条
        # "续帧"分支的触发条件(self._px is not None)恒为假，永远不会被走到。这样每一次发给
        # 下位机的坐标都对应"这一帧确确实实检测到了球"；没检测到就是没检测到，立即发 NA，
        # 不在树莓派这边做任何时间维度上的平滑/外推——留给下位机自己处理。
        # (副作用：_score() 的"空间一致性奖励"也失去了跨帧参考，评分会比启用跟踪时略低，
        #  但这只影响打分权重，不影响 Hough 找圆本身；对稳定检出没有实质影响。)
        _det_t0 = time.perf_counter()
        cands = detector.detect(sub)
        primary = cands[0] if cands else None
        detector.reset()
        _det_ms = (time.perf_counter() - _det_t0) * 1000.0
        detect_ms_ema = _det_ms if detect_ms_ema == 0 else 0.9 * detect_ms_ema + 0.1 * _det_ms

        # 真实端到端帧率（含采集+去畸变+检测），不是只算管线，避免虚高到上千
        now = time.perf_counter()
        dt = now - prev_t
        prev_t = now
        fps = 1.0 / dt if dt > 0 else 0.0
        fps_ema = fps if fps_ema == 0 else 0.9 * fps_ema + 0.1 * fps

        # 单片机通信：非阻塞重连 -> 处理收到的 PING/TASK(自动回PONG/ACK) -> 按协议节流发
        # 主目标 X 坐标(丢球发NA) -> 若正在录像(TASK进行中)把【干净帧】(未叠加下面这些
        # 调参可视化)写入录像文件。当前 send_x 发的是全画面绝对像素 x，不是协议里的 mm
        # （摆杆两端标定还没做，见 mcu_link.McuLink.send_x 的说明）。
        if mcu is not None:
            mcu.try_open()
            mcu.poll_incoming()
            mcu.send_x(int(primary[0]) + x0 if primary is not None else None)
            # 【无条件每帧调用，不要包 if mcu.recording】——是否要真的写由方法内部判断，
            # 外面加门槛会导致 START 后录像文件永远不会生成。详见 mcu_link.py::write_video_frame。
            # 传实测帧率 fps_ema：录像用真实帧率封装、播放时长才和实际一致（不再被固定 60 压短）。
            # 实际写盘在后台线程，本调用只非阻塞入队，不拖慢主循环。
            mcu.write_video_frame(frame, measured_fps=fps_ema)

        # 网页滑块改动的防抖持久化：静默 ≥1s 才写盘，避免拖动滑块时每次改动都写一次文件。
        if params_dirty["since"] is not None and (time.time() - params_dirty["since"]) >= 1.0:
            cfg["detector"] = detector.as_dict()
            save_cfg(cfg)
            params_dirty["since"] = None

        # ---- 显示 & 按键 ----
        # GUI 模式：重的绘制/imshow 限流到 DISPLAY_REFRESH_INTERVAL(~15Hz)，把 CPU 让给
        # 上面每圈都跑的采集+检测+串口发送，让 X 反馈频率尽量接近相机上限；按键也在这个
        # 节奏里轮询。headless 模式：完全不画，只周期性打印一行状态。
        if gui_on:
            now_draw = time.perf_counter()
            if (now_draw - last_draw) >= DISPLAY_REFRESH_INTERVAL:
                last_draw = now_draw
                # 左：整幅画面 + ROI 高亮(调暗) + 检测结果（绝对坐标）——本地窗口老样子不变
                left = build_detection_overlay(frame, cfg, cands, primary, x0, y0, fps_ema, mcu, detect_ms_ema)
                if stream_server is not None and stream_server.has_clients:
                    # 网页主画面单独画一版：不调暗 ROI 外区域，看到的是真实原始画面(只叠黄框
                    # 参考线+绿色检测圈)，跟本地调参窗口的"调暗突出"风格分开，算法/坐标不受影响。
                    web_left = build_detection_overlay(frame, cfg, cands, primary, x0, y0, fps_ema, mcu,
                                                        detect_ms_ema, dim_roi=False)
                    stream_server.update_frame(web_left)  # 没人在看局域网画面就不用白编码浪费 CPU

                # 右：球体二值掩膜（见 build_ball_mask_overlay 说明）。
                # 【踩坑，实测复现过：左右面板坐标对不上】曾经在这里贴图前就用 ball_detector.draw()
                # 在 ROI 局部坐标系的子图上画圈+印文字——文字是局部坐标(比如"(262,17)")，左图
                # 用的是 draw_detections() 换算过的整幅画面绝对坐标(比如"(301,217)")，同一个球
                # 两个面板显示两个不同的数字，看起来像识别出了两个不同位置，其实是同一份检测
                # 结果、只是标签坐标系不一致。build_ball_mask_overlay 内部【先贴图、再在绝对
                # 坐标系里画】，和左图共用同一个 draw_detections()，两个面板完全一致。
                mid = build_ball_mask_overlay(detector, cfg, cands, primary, x0, y0, sub.shape[:2], H, W)
                if stream_server is not None and stream_server.has_bin_clients:
                    stream_server.update_bin_frame(mid)  # 没人看二值化窗口就不用白编码

                # 主窗口 = 左:画面+检测(彩色) | 右:球体二值掩膜(黑白，调参用)
                panel = np.hstack([left, mid])
                cv2.imshow(main_win, panel)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                elif key == ord("u"):
                    if undist_maps is None:
                        print("   无标定，无法去畸变。")
                    else:
                        undist_on = not undist_on
                        cfg["undistort"] = undist_on
                        save_cfg(cfg)
                        detector.reset()  # 去畸变改变几何映射，跟踪历史位置作废，清空重来
                        print(f"   去畸变={'开' if undist_on else '关'}（已保存）。")
                elif key == ord("c"):
                    cfg["roi_top"], cfg["roi_bottom"] = 0, H
                    cfg["roi_left"], cfg["roi_right"] = 0, W
                    cv2.setTrackbarPos("roi_top", TUNE_WIN, 0)
                    cv2.setTrackbarPos("roi_bottom", TUNE_WIN, H)
                    cv2.setTrackbarPos("roi_left", TUNE_WIN, 0)
                    cv2.setTrackbarPos("roi_right", TUNE_WIN, W)
                    print("   ROI 清为整幅画面。")
                elif key == ord("s"):
                    cfg["undistort"] = undist_on
                    cfg["detector"] = detector.as_dict()  # 检测器参数整包存回
                    ok = save_cfg(cfg)
                    print(f"   参数已{'保存' if ok else '保存失败'} → {CONFIG_FILE}")
        else:
            # headless 无本地 GUI，但局域网画面仍可能有人在看：限流到 DISPLAY_REFRESH_INTERVAL
            # (~15Hz) 才画叠加层+编码，且只在真有客户端连着时才做，不白白占用主循环的 CPU。
            if stream_server is not None and (stream_server.has_clients or stream_server.has_bin_clients):
                now_draw = time.perf_counter()
                if (now_draw - last_draw) >= DISPLAY_REFRESH_INTERVAL:
                    last_draw = now_draw
                    if stream_server.has_clients:
                        # headless 没有本地窗口，直接就是给网页看的，不调暗 ROI 外区域。
                        overlay = build_detection_overlay(frame, cfg, cands, primary, x0, y0, fps_ema, mcu,
                                                           detect_ms_ema, dim_roi=False)
                        stream_server.update_frame(overlay)
                    if stream_server.has_bin_clients:
                        mid = build_ball_mask_overlay(detector, cfg, cands, primary, x0, y0, sub.shape[:2], H, W)
                        stream_server.update_bin_frame(mid)
            # headless：每 2s 打印一行状态，确认程序还活着、看当前反馈频率(FPS≈每秒发X次数)
            now_stat = time.perf_counter()
            if (now_stat - last_stat) >= 2.0:
                last_stat = now_stat
                bx = int(primary[0]) + x0 if primary is not None else None
                mcu_s = "--" if mcu is None else ("ON" if mcu.mcu_online else "off")
                txn = 0 if mcu is None else mcu.tx_x_count
                print(f"[headless] {fps_ema:4.1f}FPS  det={detect_ms_ema:.0f}ms  MCU:{mcu_s}  TXx#{txn}  "
                      f"ball_x={bx if bx is not None else 'NA'}")
    except KeyboardInterrupt:
        print("\n⛔ 收到 Ctrl-C，正在退出…")

    cfg["undistort"] = undist_on
    cfg["detector"] = detector.as_dict()  # 退出时也把检测器参数存回，别丢了本次调的值
    save_cfg(cfg)
    if mcu is not None:
        mcu.close()  # 兜底：万一退出时 MCU 还没发 STOP，也要 release 掉录像文件
    if stream_server is not None:
        stream_server.stop()
    cam.release()
    if gui_on:
        cv2.destroyAllWindows()
    print("已退出，参数已保存。")


def run_selftest(args):
    """无窗口自检：开摄像头、跑形状+高光检测管线、报帧率与检出率（验证管线不报错）。"""
    cfg = load_cfg()
    rs_opts = (cfg["rs_exposure"], cfg["rs_gain"], cfg["rs_white_balance"])
    cam = open_camera(args.source, args.index, args.width, args.height, args.fps, rs_opts)
    print(f"[selftest] source={args.source} index={args.index}")

    warm = None
    for _ in range(30):
        f = cam.read()
        if f is not None:
            warm = f
    if warm is None:
        print("[selftest] ❌ 读不到帧")
        cam.release()
        return
    H, W = warm.shape[:2]
    calib = cc.load_calibration()
    maps = cc.build_undistort_maps(
        calib["camera_matrix"], calib["dist_coeffs"], (W, H), alpha=0.0) if calib else None
    undist_on = bool(cfg.get("undistort", True)) and maps is not None
    top, bottom = (cfg["roi_top"], cfg["roi_bottom"]) if cfg["roi_bottom"] > cfg["roi_top"] else (0, H)
    left, right = (cfg["roi_left"], cfg["roi_right"]) if cfg["roi_right"] > cfg["roi_left"] else (0, W)
    detector = BallDetector().load_dict(cfg.get("detector", {}))
    print(f"[selftest] 帧 {W}x{H}, ROI=x[{left},{right}] y[{top},{bottom}], 去畸变={'开' if undist_on else '关'}, "
          f"检测器 param2={detector.param2} min_vmax={detector.min_vmax} r=[{detector.min_radius},{detector.max_radius}]")

    n, det_frames, t0 = 0, 0, time.perf_counter()
    while n < 120:
        f = cam.read()
        if f is None:
            continue
        f = undistort_frame(f, maps, undist_on)
        sub, (x0, y0) = apply_rect_roi(f, top, bottom, left, right)
        cands = detector.detect(sub)
        if detector.pick_primary(cands) is not None:
            det_frames += 1
        n += 1
    dt = time.perf_counter() - t0
    print(f"[selftest] {n} 帧用 {dt:.2f}s → {n/dt:.1f} FPS(采集+检测)，"
          f"其中 {det_frames} 帧锁到目标（有球且参数合适时应较高，无球时接近 0）")
    cam.release()
    print("[selftest] ✅ 通过")


def main():
    ap = argparse.ArgumentParser(description="车载平衡滚球视觉识别 v1.1 beta")
    ap.add_argument("--source", choices=["realsense", "usb", "csi"], default="realsense",
                    help="摄像头类型（默认 realsense：只用深度相机的彩色流当2D摄像头）")
    ap.add_argument("--index", type=int, default=None,
                    help="USB 摄像头 index（/dev/videoN）；不填=按 by-id 自动解析物理设备，"
                         "更抗拔插后编号漂移，一般不用填")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--selftest", action="store_true", help="无窗口跑通并测帧率")
    ap.add_argument("--headless", action="store_true",
                    help="无窗口运行（比赛/部署用）：不开任何 GUI/滑条/预览，只跑 采集→检测→串口发坐标。"
                         "省下 GUI 抢的 CPU，串口 X 反馈频率可从带窗口的 ~33Hz 提到 ~42Hz(相机驱动硬上限)。"
                         "检测器/成像参数用 rollball_config.json 已保存的值，需提前用带窗口版调好。"
                         "退出按 Ctrl-C。")
    ap.add_argument("--no-mcu", action="store_true", help="不接单片机串口，纯视觉调试")
    ap.add_argument("--mcu-port", default=None,
                    help="单片机串口设备路径，手动指定覆盖自动探测（默认按 CH340/ttyUSB 自动找）")
    ap.add_argument("--no-stream", action="store_true",
                    help="禁用局域网实时画面 HTTP 服务（默认开启）。浏览器打开 "
                         "http://<本机局域网IP>:<port>/ 即可看当前画面，不用接显示器。"
                         "实测确认它不是帧率瓶颈（JPEG 编码在后台线程做，没人看时"
                         "has_clients=False 也不会触发编码），默认开；需要绝对榨干"
                         "最后一点余量或不想暴露局域网服务时用这个关掉。")
    ap.add_argument("--stream-port", type=int, default=8080,
                    help="局域网实时画面 HTTP 服务端口，默认 8080")
    args = ap.parse_args()

    if args.selftest:
        run_selftest(args)
    else:
        run_gui(args)


if __name__ == "__main__":
    main()
