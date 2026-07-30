#!/usr/bin/env python3
"""
车载平衡滚球 —— 视觉识别 v1.1 beta（第一步：背景差分找白/亮球 + 矩形 ROI + 可调阈值）
环境: /home/hao/vision_env/bin/python3

场景：摄像头装在摆杆正上方、光轴垂直向下；25cm 白色 PPR 水管凹槽里放一颗 φ1cm 钢珠，
背景基本纯白。这一版只做"稳定地把球找出来"，为后面加卡尔曼/丢球状态机/毫米换算打底。

本版已实现（对应总体需求里的基础部分）：
  1. 可替换的摄像头接口：USB(cv2.VideoCapture+V4L2) / CSI(picamera2) /
     RealSense(pyrealsense2，只用彩色流当 2D 摄像头，不用深度)，用 --source 选，
     三个后端都吐 640x480 BGR 帧；换后端不改主流程。当前默认 realsense
     （树莓派上插的是深度相机，USB 彩色摄像头已停用，参考
     /home/hao/Desktop/luhao/ruikang/my/v2.8.py 里 RealSense pipeline 的起流写法，
     但去掉了对齐/深度流/内参那一套，因为这里不用深度）。
  2. 空水管背景标定模式：按 b 取下钢珠后连续采 N 帧取【中值】生成稳定背景，存 background.png。
  3. 背景差分主流程：矩形ROI → 灰度 → 3x3 高斯 → cv2.absdiff(背景) → 阈值二值化 →
     椭圆 3x3 开运算去噪 → 椭圆 5x5 闭运算连通 → 外轮廓 → 按面积/圆度筛候选 → 取最优。
  4. 矩形 ROI：摆杆在画面里是一条横向管道，凹槽在管道中间。用【上/下/左/右】四个
     滑条框出感兴趣区域（滑块交互式调节，全部实时可调、拖动即生效），
     既能只裁出中间通道高度，也能进一步裁掉左右两端不关心的支架/背景。
  5. 全部关键阈值在一个【滑条窗口 Tuning】里实时可调，滑条名本身是参数英文名(ASCII)——
     本机 Qt 高亮组件缺字体，滑条名用中文会显示不出来（环境限制）；每个滑条的中文含义
     在启动时打印到终端(SLIDER_HELP)对照着看。原来那块 PIL 画的中文说明面板已移除——
     实测它单帧要 25~40ms、占 CPU 70%+，还会拖慢串口发坐标，得不偿失。
  6. 阈值支持"自动(Otsu)"模式，缓解不同光照下手动阈值失效的问题（auto_thresh 滑条）。

还没做（后面几步再加，代码已留位置）：
  - α-β / 卡尔曼滤波 + 用预测位置辅助匹配
  - 丢球状态机（连丢超阈值输出 valid=0）
  - 摆杆左右端标定 → 毫米坐标换算 / 旋转投影 / 到中心线垂直距离
  - Scharr/Sobel 梯度、局部标准差、黑帽等辅助特征评分
  - 霍夫圆作为可选的检测恢复

用法:
  /home/hao/vision_env/bin/python3 code/rollball_code/v1.1_beta.py            # 默认 RealSense 彩色流
  /home/hao/vision_env/bin/python3 code/rollball_code/v1.1_beta.py --source usb
  /home/hao/vision_env/bin/python3 code/rollball_code/v1.1_beta.py --source csi
  /home/hao/vision_env/bin/python3 code/rollball_code/v1.1_beta.py --selftest # 无窗口，跑通+测帧率
  /home/hao/vision_env/bin/python3 code/rollball_code/v1.1_beta.py --headless # 无窗口比赛/部署，串口发X最快，Ctrl-C 退出

按键（主窗口聚焦时）:
  b = 背景标定（先取下钢珠！）   c = 清空ROI（恢复整幅画面）
  u = 切换去畸变                 s = 保存参数        q/ESC = 退出
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
BG_FILE = os.path.join(HERE, "background.png")

PROJECT_ROOT = os.path.dirname(os.path.dirname(HERE))

# 复用 code/ready_code/camera_common 做相机标定读写 + 去畸变映射（唯一事实来源，不另写一份）
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code", "ready_code"))
import camera_common as cc  # noqa: E402

# 复用 code/task_code/serial_link 做串口热插拔连接管理（找口/等口/断线重连不重写）
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code", "task_code"))

from mcu_link import McuLink  # noqa: E402
from mjpeg_server import MjpegServer, get_lan_ip  # noqa: E402

# ---- 可调参数默认值（滑条会覆盖，s 保存回 JSON）----
DEFAULT_CFG = {
    "roi_top": 0,          # 条带上边界(像素行, 整幅去畸变图坐标)，0 表示待运行时初始化为整幅
    "roi_bottom": 0,       # 条带下边界；roi_bottom<=roi_top 时视为"未设置"，用整幅高度
    "roi_left": 0,         # ROI 左边界(像素列)，0 表示待运行时初始化为整幅宽度
    "roi_right": 0,        # ROI 右边界；roi_right<=roi_left 时视为"未设置"，用整幅宽度
    "thresh": 30,          # 差分二值化阈值（越大越严，抗噪但易漏；auto_thresh=1 时忽略）
    "auto_thresh": 0,      # 1=用 Otsu 自动阈值（不同光照下更鲁棒），0=用上面的固定 thresh
    "blur_k": 3,           # 高斯核（奇数）
    "open_k": 3,           # 开运算核（去小噪点）
    "close_k": 5,          # 闭运算核（连断裂）
    "min_area": 50,        # 候选最小面积(px^2)
    "max_area": 3000,      # 候选最大面积(px^2)
    "min_circ": 40,        # 最小圆度 *100（圆=100）
    "undistort": True,     # 项目约定：有 camera_calib.npz 就默认去畸变（u 键运行时切换）
}


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

    def __init__(self, width=640, height=480, fps=30):
        try:
            import pyrealsense2 as rs
        except ImportError as e:
            raise RuntimeError(
                "RealSense 需要 pyrealsense2，但当前环境未安装。\n"
                "  /home/hao/vision_env/bin/python3 -m pip install pyrealsense2\n"
            ) from e
        self._rs = rs
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        try:
            self.profile = self.pipeline.start(cfg)
        except RuntimeError as e:
            # 同参考文件的降级策略：指定分辨率/帧率协商失败就退回设备默认彩色流配置。
            print(f"   ⚠️ RealSense 指定 {width}x{height}@{fps} 启动失败({e})，退回默认配置…")
            self.pipeline = rs.pipeline()
            cfg = rs.config()
            cfg.enable_stream(rs.stream.color)
            self.profile = self.pipeline.start(cfg)

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


def open_camera(source, index, width, height, fps):
    if source == "csi":
        return CsiCamera(width, height, fps)
    if source == "realsense":
        return RealSenseCamera(width, height, fps)
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

    def release(self):
        self._drop()


# ============================ 小工具 ============================

def odd(v, lo=1):
    """把滑条值规整成 >=lo 的奇数（形态学/高斯核要奇数）。"""
    v = max(lo, int(v))
    return v if v % 2 == 1 else v + 1


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


# ======================= 背景差分检测主流程 =======================

def detect(band_bgr, bg_gray, cfg):
    """
    背景差分找球。返回 (candidates, best, mask, thresh_used)。
      candidates: [{cx,cy,r,area,circ}, ...]（ROI 内坐标，(0,0)=ROI 左上角）
      best: 面积最大的合法候选（这一版先用面积，评分融合下一步再做）或 None
      mask: 二值掩膜，用于可视化调参
      thresh_used: 实际用的阈值（auto_thresh=1 时是 Otsu 算出来的值，供面板显示）
    """
    gray = cv2.cvtColor(band_bgr, cv2.COLOR_BGR2GRAY)
    k = odd(cfg["blur_k"])
    gray = cv2.GaussianBlur(gray, (k, k), 0)

    diff = cv2.absdiff(gray, bg_gray)
    if cfg.get("auto_thresh"):
        thresh_used, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        thresh_used = float(cfg["thresh"])
        _, mask = cv2.threshold(diff, int(cfg["thresh"]), 255, cv2.THRESH_BINARY)

    ok = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (odd(cfg["open_k"]),) * 2)
    ck = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (odd(cfg["close_k"]),) * 2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ok)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ck)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cands = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < cfg["min_area"] or area > cfg["max_area"]:
            continue
        per = cv2.arcLength(c, True)
        if per <= 1e-3:
            continue
        circ = 4.0 * np.pi * area / (per * per)   # 1.0=完美圆
        if circ * 100.0 < cfg["min_circ"]:
            continue
        (cx, cy), r = cv2.minEnclosingCircle(c)
        cands.append({"cx": float(cx), "cy": float(cy), "r": float(r),
                      "area": float(area), "circ": float(circ)})

    best = max(cands, key=lambda d: d["area"]) if cands else None
    return cands, best, mask, float(thresh_used)


# ============================ 滑条窗口 (Tuning window) ============================
# 每个滑条名字直接用参数的英文名(ASCII)，从上到下顺序 = 下面 _TRACKBAR_ORDER 的顺序。
# 原来那块 PIL 画的【中文说明面板】已移除：实测单帧 25~40ms、占 CPU 70%+，还拖慢串口发送；
# 现在只保留 图像+掩膜 预览 + 本滑条窗口，中文含义启动时打印到终端(见 SLIDER_HELP)。

TUNE_WIN = "Tuning (rollball v1.1b)"

# 【性能】GUI 绘制+imshow 会抢 CPU、拖慢"每读到一帧有效数据就发 X"的主循环；重的画面显示
# 不需要跟检测/发送同步，限流到这个间隔画一次(~15Hz)，把 CPU 让给采集+检测+串口。
# 完全不要窗口(最高发送频率)用 --headless。按键(b/c/u/s/q)也在这个节奏里轮询，够灵敏。
DISPLAY_REFRESH_INTERVAL = 1.0 / 15.0

# (滑条名=cfg键名, 滑条最大值或 None=运行时按帧尺寸决定)
_TRACKBAR_ORDER = [
    ("roi_top", None),
    ("roi_bottom", None),
    ("roi_left", None),
    ("roi_right", None),
    ("thresh", 255),
    ("auto_thresh", 1),
    ("blur_k", 15),
    ("open_k", 15),
    ("close_k", 15),
    ("min_area", 5000),
    ("max_area", 20000),
    ("min_circ", 100),
]

# 启动时打印到终端的滑条说明（中文；顺序同 Tuning 窗口从上到下）。
# 注意：滑条本身在窗口里的名字(roi_top等)必须是英文/ASCII——本机 Qt 高亮组件缺字体，
# 滑条名若用中文会显示不出来（环境限制），所以窗口里滑条名保留英文，含义靠这里的中文对照。
SLIDER_HELP = [
    "Tuning 窗口滑条说明(从上到下):",
    "  roi_top      ROI 上边界 (像素行)",
    "  roi_bottom   ROI 下边界 (像素行)",
    "  roi_left     ROI 左边界 (像素列)",
    "  roi_right    ROI 右边界 (像素列)",
    "  thresh       差分二值化阈值 (auto_thresh=1 时此项不生效)",
    "  auto_thresh  1=用Otsu自动阈值(更抗光照变化), 0=用上面的thresh",
    "  blur_k       高斯模糊核大小 (奇数)",
    "  open_k       开运算核 (去小噪点)",
    "  close_k      闭运算核 (连接断裂区域)",
    "  min_area     候选区域最小面积 (像素^2)",
    "  max_area     候选区域最大面积 (像素^2)",
    "  min_circ     最小圆度 x100 (100=完美圆)",
]


def setup_trackbars(cfg, H, W):
    cv2.namedWindow(TUNE_WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(TUNE_WIN, 480, 460)
    _n = lambda v: None
    maxes = {
        "roi_top": max(1, H - 1), "roi_bottom": H,
        "roi_left": max(1, W - 1), "roi_right": W,
    }
    for name, default_max in _TRACKBAR_ORDER:
        mx = maxes.get(name, default_max)
        cv2.createTrackbar(name, TUNE_WIN, int(cfg[name]), mx, _n)


def read_trackbars(cfg):
    top = cv2.getTrackbarPos("roi_top", TUNE_WIN)
    bottom_raw = cv2.getTrackbarPos("roi_bottom", TUNE_WIN)
    cfg["roi_top"] = top
    cfg["roi_bottom"] = max(top + 10, bottom_raw)  # 至少 10px 通道高度，避免空条带
    left = cv2.getTrackbarPos("roi_left", TUNE_WIN)
    right_raw = cv2.getTrackbarPos("roi_right", TUNE_WIN)
    cfg["roi_left"] = left
    cfg["roi_right"] = max(left + 10, right_raw)  # 至少 10px 宽度，避免空区域
    cfg["thresh"] = cv2.getTrackbarPos("thresh", TUNE_WIN)
    cfg["auto_thresh"] = cv2.getTrackbarPos("auto_thresh", TUNE_WIN)
    cfg["blur_k"] = cv2.getTrackbarPos("blur_k", TUNE_WIN)
    cfg["open_k"] = cv2.getTrackbarPos("open_k", TUNE_WIN)
    cfg["close_k"] = cv2.getTrackbarPos("close_k", TUNE_WIN)
    cfg["min_area"] = cv2.getTrackbarPos("min_area", TUNE_WIN)
    cfg["max_area"] = max(cfg["min_area"] + 1, cv2.getTrackbarPos("max_area", TUNE_WIN))
    cfg["min_circ"] = cv2.getTrackbarPos("min_circ", TUNE_WIN)


# ============================ 背景标定 ============================

def calibrate_background(cam, maps=None, undist_on=False, n_frames=45, preview_win=None):
    """
    取下钢珠后连采 n 帧，对【整幅画面】灰度取中值生成稳定背景（与检测同样先去畸变）。
    背景按整幅存，不按当前 ROI 存——这样拖动 ROI 边界只是在同一张背景图上换一段来比对，
    ROI 边界怎么调都不会和背景尺寸对不上。返回 bg_gray(整幅) 或 None。
    """
    print(f"📸 背景标定：请确认已取下钢珠，正在采集 {n_frames} 帧…")
    stack = []
    grabbed = 0
    tries = 0
    while grabbed < n_frames and tries < n_frames * 4:
        tries += 1
        try:
            frame = cam.read()
        except CameraLost:
            print("   ⚠️ 标定过程中摄像头掉线，中止本次标定。")
            break
        if frame is None:
            continue
        frame = undistort_frame(frame, maps, undist_on)
        stack.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        grabbed += 1
        if preview_win is not None:
            show = frame.copy()
            cv2.putText(show, f"CALIB {grabbed}/{n_frames} (remove ball!)",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imshow(preview_win, show)
            cv2.waitKey(1)
    if grabbed < 5:
        print("   ⚠️ 采到的有效帧太少，背景标定失败。")
        return None
    bg = np.median(np.stack(stack, axis=0), axis=0).astype(np.uint8)
    cv2.imwrite(BG_FILE, bg)
    print(f"   ✅ 背景已生成并保存: {BG_FILE}（{bg.shape[1]}x{bg.shape[0]}，整幅）")
    return bg


def quick_background(cam, maps=None, undist_on=False, n=30):
    """快速抓一张【临时】整幅背景（中值），不落盘、不带钢珠提示。用于让画面立刻有反馈可调参。"""
    stack = []
    tries = 0
    while len(stack) < n and tries < n * 4:
        tries += 1
        try:
            f = cam.read()
        except CameraLost:
            break
        if f is None:
            continue
        f = undistort_frame(f, maps, undist_on)
        stack.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    if len(stack) < 5:
        return None
    return np.median(np.stack(stack, axis=0), axis=0).astype(np.uint8)


def load_background(expect_shape=None):
    """载入已存的整幅背景；shape 与当前整幅画面不符（换分辨率/去畸变开关）则视为无效。"""
    if not os.path.exists(BG_FILE):
        return None
    bg = cv2.imread(BG_FILE, cv2.IMREAD_GRAYSCALE)
    if bg is None:
        return None
    if expect_shape is not None and bg.shape[:2] != expect_shape[:2]:
        print("   ⚠️ 已存背景与当前画面尺寸不符，需要重新按 b 标定。")
        return None
    return bg


# ============================ 绘制叠加 ============================

def draw_roi_highlight(canvas, top, bottom, left, right):
    """矩形 ROI 外区域调暗，凸显感兴趣区域，并画四条黄色边界线（上/下/左/右）。"""
    H, W = canvas.shape[:2]
    dim = canvas.astype(np.float32) * 0.35
    mask = np.ones((H, W), dtype=bool)
    mask[top:bottom, left:right] = False   # ROI 内不调暗
    canvas[mask] = dim[mask].astype(np.uint8)
    cv2.line(canvas, (0, top), (W, top), (0, 255, 255), 2, cv2.LINE_AA)
    cv2.line(canvas, (0, min(bottom, H - 1)), (W, min(bottom, H - 1)), (0, 255, 255), 2, cv2.LINE_AA)
    cv2.line(canvas, (left, 0), (left, H), (0, 255, 255), 2, cv2.LINE_AA)
    cv2.line(canvas, (min(right, W - 1), 0), (min(right, W - 1), H), (0, 255, 255), 2, cv2.LINE_AA)
    return canvas


def draw_detections(canvas, cands, best, x_offset, y_offset):
    """把 ROI 内坐标的候选/主目标画到整幅画面上（加 (x_offset,y_offset) 换算成绝对坐标）。"""
    for d in cands:
        cv2.circle(canvas, (int(d["cx"] + x_offset), int(d["cy"] + y_offset)), int(d["r"]),
                    (0, 200, 0), 1, cv2.LINE_AA)
    if best is not None:
        bx, by = int(best["cx"] + x_offset), int(best["cy"] + y_offset)
        cv2.circle(canvas, (bx, by), int(best["r"]), (0, 255, 0), 2, cv2.LINE_AA)
        cv2.circle(canvas, (bx, by), 3, (0, 255, 0), -1)
        cv2.putText(canvas, f"({bx},{by}) A={int(best['area'])} C={best['circ']:.2f}",
                    (bx + 8, by), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
    return canvas


def draw_hud(canvas, cfg, fps, bg_state):
    hud = f"{fps:4.1f}FPS  BG:{bg_state}  th={cfg['thresh']}{'(auto)' if cfg.get('auto_thresh') else ''}"
    cv2.putText(canvas, hud, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 255, 255), 2, cv2.LINE_AA)
    return canvas


def build_detection_overlay(frame, cfg, cands, best, x0, y0, fps, bg_state, mcu):
    """
    整幅画面 + ROI 高亮 + 检测结果 + HUD，供【本地 GUI 左图】和【局域网 MJPEG 推流】共用一份，
    不重复画两次。
    """
    canvas = frame.copy()
    draw_roi_highlight(canvas, cfg["roi_top"], cfg["roi_bottom"], cfg["roi_left"], cfg["roi_right"])
    draw_detections(canvas, cands, best, x0, y0)
    draw_hud(canvas, cfg, fps, bg_state)
    if mcu is not None:
        mcu_hud = (f"MCU:{'ON' if mcu.mcu_online else '--'} "
                   f"REC:{('TASK' + str(mcu.current_task_id)) if mcu.recording else '-'} "
                   f"TXx#{mcu.tx_x_count}")
        cv2.putText(canvas, mcu_hud, (8, 42), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 255, 255), 2, cv2.LINE_AA)
    return canvas




# ============================ 主程序 ============================

def run_gui(args):
    cfg = load_cfg()
    gui_on = not getattr(args, "headless", False)   # --headless: 不开任何窗口/滑条
    print(f"⚙️  配置: {CONFIG_FILE}")
    if not gui_on:
        print("🖥️  headless 模式：不开窗口/滑条，参数用已保存值、背景用 background.png；退出按 Ctrl-C。")

    # 局域网实时画面：浏览器打开这个地址就能看当前摄像头画面，不用接显示器/开 GUI 窗口
    # （headless 比赛模式尤其有用）。只用标准库 http.server，不装额外依赖。
    stream_server = None
    if not args.no_stream:
        stream_server = MjpegServer(port=args.stream_port)
        stream_server.start()
        print(f"🌐 局域网实时画面: http://{get_lan_ip()}:{args.stream_port}/ "
              f"（同一局域网/热点下的手机、电脑浏览器打开即可，摄像头就绪前先显示占位画面）")
    else:
        print("🌐 局域网实时画面: 已禁用 (--no-stream)")

    def camera_factory():
        return open_camera(args.source, args.index, args.width, args.height, args.fps)

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
    # 背景按【整幅画面】载入/抓取，与当前 ROI 边界无关——拖动 ROI 边界只是在这张
    # 整幅背景上换一段来比对，不会因为 ROI 尺寸变了就找不到背景（这是之前的 bug 根因）。
    bg_full = load_background(expect_shape=frame.shape)
    bg_is_temp = False
    if bg_full is not None:
        print(f"   已载入背景 {BG_FILE}")
    else:
        # 没有正式背景时先自动抓一张【临时】背景，保证一进来阈值/掩膜就有反馈可调；
        # 正式使用请取下钢珠后按 b 重新标定，效果最好。
        bg_full = quick_background(cam, undist_maps, undist_on)
        bg_is_temp = bg_full is not None
        print("   没有正式背景：已自动抓一张临时背景让画面有反馈；正式用请取下钢珠按 b 标定。")

    PANEL_W = W * 2   # 主窗口 = 左:整幅(带高亮) | 中:掩膜(整幅对齐)（中文说明面板已移除）

    main_win = "RollBall detect (v1.1 beta)"
    if gui_on:
        cv2.namedWindow(main_win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(main_win, PANEL_W, H)
        setup_trackbars(cfg, H, W)

    fps_ema = 0.0
    if gui_on:
        print("   窗口: 主窗口 = [左: 画面+ROI+检测结果] | [右: 掩膜]")
        for _l in SLIDER_HELP:      # 滑条中文含义打印到终端（顺序同 Tuning 窗口从上到下）
            print("   " + _l)
        print("   按键: [b]标定背景 [c]清空ROI(恢复整幅) [u]切换去畸变 [s]保存参数 [q]退出")
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
                          f"背景/通道需要重新设置。")
                    W, H = nw, nh
                    undist_maps = cc.build_undistort_maps(
                        calib["camera_matrix"], calib["dist_coeffs"], (W, H), alpha=0.0) if calib else None
                    undist_on = bool(cfg.get("undistort", True)) and undist_maps is not None
                    bg_full = None
                    bg_is_temp = False
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
            read_trackbars(cfg)  # headless 无滑条，直接用 rollball_config.json 载入的 cfg
        sub, (x0, y0) = apply_rect_roi(frame, cfg["roi_top"], cfg["roi_bottom"],
                                        cfg["roi_left"], cfg["roi_right"])

        if bg_full is not None and bg_full.shape[:2] == frame.shape[:2]:
            bg_band = bg_full[y0:y0 + sub.shape[0], x0:x0 + sub.shape[1]]
            cands, best, mask, thresh_used = detect(sub, bg_band, cfg)
            bg_state = "TEMP" if bg_is_temp else "OK"
        else:
            cands, best, mask, thresh_used = [], None, np.zeros(sub.shape[:2], np.uint8), 0.0
            bg_state = "none"

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
            mcu.send_x(int(best["cx"]) + x0 if best is not None else None)
            # 【无条件每帧调用，不要包 if mcu.recording】——是否要真的写由方法内部判断，
            # 外面加门槛会导致 START 后录像文件永远不会生成。详见 mcu_link.py::write_video_frame。
            # 传实测帧率 fps_ema：录像用真实帧率封装、播放时长才和实际一致（不再被固定 60 压短）。
            # 实际写盘在后台线程，本调用只非阻塞入队，不拖慢主循环。
            mcu.write_video_frame(frame, measured_fps=fps_ema)

        # ---- 显示 & 按键 ----
        # GUI 模式：重的绘制/imshow 限流到 DISPLAY_REFRESH_INTERVAL(~15Hz)，把 CPU 让给
        # 上面每圈都跑的采集+检测+串口发送，让 X 反馈频率尽量接近相机上限；按键也在这个
        # 节奏里轮询。headless 模式：完全不画，只周期性打印一行状态。
        if gui_on:
            now_draw = time.perf_counter()
            if (now_draw - last_draw) >= DISPLAY_REFRESH_INTERVAL:
                last_draw = now_draw
                # 左：整幅画面 + ROI 高亮 + 检测结果（绝对坐标）
                left = build_detection_overlay(frame, cfg, cands, best, x0, y0, fps_ema, bg_state, mcu)
                if stream_server is not None and stream_server.has_clients:
                    stream_server.update_frame(left)  # 没人在看局域网画面就不用白编码浪费 CPU

                # 中：掩膜嵌回整幅大小的画布，行列都和左图对齐，方便对照
                mask_full = np.zeros((H, W), np.uint8)
                mask_full[y0:y0 + mask.shape[0], x0:x0 + mask.shape[1]] = mask
                mid = cv2.cvtColor(mask_full, cv2.COLOR_GRAY2BGR)
                cv2.line(mid, (0, cfg["roi_top"]), (W, cfg["roi_top"]), (0, 255, 255), 1, cv2.LINE_AA)
                cv2.line(mid, (0, cfg["roi_bottom"] - 1), (W, cfg["roi_bottom"] - 1), (0, 255, 255), 1, cv2.LINE_AA)
                cv2.line(mid, (cfg["roi_left"], 0), (cfg["roi_left"], H), (0, 255, 255), 1, cv2.LINE_AA)
                cv2.line(mid, (cfg["roi_right"] - 1, 0), (cfg["roi_right"] - 1, H), (0, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(mid, "MASK (absdiff>thresh)", (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 255), 1, cv2.LINE_AA)

                # 主窗口 = 左:图像 | 右:掩膜（原中文说明面板已移除，省 CPU、不拖慢串口发送；
                # 滑条英文含义已在启动时打印到终端，见 SLIDER_HELP）
                panel = np.hstack([left, mid])
                cv2.imshow(main_win, panel)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                elif key == ord("b"):
                    newbg = calibrate_background(cam, undist_maps, undist_on, preview_win=main_win)
                    if newbg is not None:
                        bg_full = newbg
                        bg_is_temp = False
                elif key == ord("u"):
                    if undist_maps is None:
                        print("   无标定，无法去畸变。")
                    else:
                        undist_on = not undist_on
                        cfg["undistort"] = undist_on
                        save_cfg(cfg)
                        # 去畸变改变了整幅画面的几何映射，旧背景作废，抓一张临时背景续上反馈；
                        # 正式用再按 b。（ROI 边界不影响背景，不用因为这个重新抓）
                        bg_full = quick_background(cam, undist_maps, undist_on)
                        bg_is_temp = bg_full is not None
                        print(f"   去畸变={'开' if undist_on else '关'}（已保存，临时背景已更新，正式用请按 b）。")
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
                    ok = save_cfg(cfg)
                    print(f"   参数已{'保存' if ok else '保存失败'} → {CONFIG_FILE}")
        else:
            # headless 无本地 GUI，但局域网画面仍可能有人在看：限流到 DISPLAY_REFRESH_INTERVAL
            # (~15Hz) 才画叠加层+编码，且只在真有客户端连着时才做，不白白占用主循环的 CPU。
            if stream_server is not None and stream_server.has_clients:
                now_draw = time.perf_counter()
                if (now_draw - last_draw) >= DISPLAY_REFRESH_INTERVAL:
                    last_draw = now_draw
                    overlay = build_detection_overlay(frame, cfg, cands, best, x0, y0, fps_ema, bg_state, mcu)
                    stream_server.update_frame(overlay)
            # headless：每 2s 打印一行状态，确认程序还活着、看当前反馈频率(FPS≈每秒发X次数)
            now_stat = time.perf_counter()
            if (now_stat - last_stat) >= 2.0:
                last_stat = now_stat
                bx = int(best["cx"]) if best is not None else None
                mcu_s = "--" if mcu is None else ("ON" if mcu.mcu_online else "off")
                txn = 0 if mcu is None else mcu.tx_x_count
                print(f"[headless] {fps_ema:4.1f}FPS  MCU:{mcu_s}  TXx#{txn}  "
                      f"best_x={bx if bx is not None else 'NA'}  bg={bg_state}")
    except KeyboardInterrupt:
        print("\n⛔ 收到 Ctrl-C，正在退出…")

    cfg["undistort"] = undist_on
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
    """无窗口自检：开摄像头、用前若干帧中值当背景、跑管线、报帧率与检测数。"""
    cfg = load_cfg()
    cam = open_camera(args.source, args.index, args.width, args.height, args.fps)
    print(f"[selftest] source={args.source} index={args.index}")

    warm = []
    for _ in range(30):
        f = cam.read()
        if f is not None:
            warm.append(f)
    if not warm:
        print("[selftest] ❌ 读不到帧")
        cam.release()
        return
    H, W = warm[-1].shape[:2]
    calib = cc.load_calibration()
    maps = cc.build_undistort_maps(
        calib["camera_matrix"], calib["dist_coeffs"], (W, H), alpha=0.0) if calib else None
    undist_on = bool(cfg.get("undistort", True)) and maps is not None
    if cfg["roi_bottom"] <= cfg["roi_top"]:
        top, bottom = 0, H
    else:
        top, bottom = cfg["roi_top"], cfg["roi_bottom"]
    if cfg["roi_right"] <= cfg["roi_left"]:
        left, right = 0, W
    else:
        left, right = cfg["roi_left"], cfg["roi_right"]
    # 背景按整幅合成（和 run_gui 一样解耦于 ROI 边界），再按 ROI 切一段来测。
    grays = [cv2.cvtColor(undistort_frame(f, maps, undist_on), cv2.COLOR_BGR2GRAY) for f in warm]
    bg_full = np.median(np.stack(grays, axis=0), axis=0).astype(np.uint8)
    print(f"[selftest] 帧 {W}x{H}, ROI=x[{left},{right}] y[{top},{bottom}], 去畸变={'开' if undist_on else '关'}, "
          f"合成背景 {bg_full.shape[1]}x{bg_full.shape[0]}(整幅)")

    n, det_frames, t0 = 0, 0, time.perf_counter()
    while n < 120:
        f = cam.read()
        if f is None:
            continue
        f = undistort_frame(f, maps, undist_on)
        sub, (x0, y0) = apply_rect_roi(f, top, bottom, left, right)
        bg_band = bg_full[y0:y0 + sub.shape[0], x0:x0 + sub.shape[1]]
        cands, best, _, _ = detect(sub, bg_band, cfg)
        if best is not None:
            det_frames += 1
        n += 1
    dt = time.perf_counter() - t0
    print(f"[selftest] {n} 帧用 {dt:.2f}s → {n/dt:.1f} FPS(采集+管线)，"
          f"其中 {det_frames} 帧检出候选（用自合成背景，正常应接近 0，只验证管线不报错）")
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
                         "参数用 rollball_config.json 已保存的值、背景用 background.png，需提前调好/标定好。"
                         "退出按 Ctrl-C。")
    ap.add_argument("--no-mcu", action="store_true", help="不接单片机串口，纯视觉调试")
    ap.add_argument("--mcu-port", default=None,
                    help="单片机串口设备路径，手动指定覆盖自动探测（默认按 CH340/ttyUSB 自动找）")
    ap.add_argument("--stream-port", type=int, default=8080,
                    help="局域网实时画面 HTTP 服务端口，默认 8080；浏览器打开 "
                         "http://<本机局域网IP>:<port>/ 查看")
    ap.add_argument("--no-stream", action="store_true", help="禁用局域网实时画面服务")
    args = ap.parse_args()

    if args.selftest:
        run_selftest(args)
    else:
        run_gui(args)


if __name__ == "__main__":
    main()
