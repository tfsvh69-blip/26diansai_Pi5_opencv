#!/usr/bin/env python3
"""
YOLO11n NCNN 推理封装 - 被 detect_live.py / benchmark.py 复用
环境: /home/hao/vision_env/bin/python3（需 `pip install ncnn`）

职责（把"如何跑一个 ncnn YOLO 模型"这件事收在一处，避免各脚本各写一份）：
  1. 载入 models/best_ncnn_<sz>/ 下的 model.ncnn.param + model.ncnn.bin；
  2. letterbox 预处理（保持长宽比缩放 + 灰边填充到 imgsz）；
  3. 前向推理；
  4. 解码输出 + NMS，把框还原回【原图】坐标。

关于模型输出（已实测）：
  ncnn 导出的 out0 形状是 [5, N]，5 = [cx, cy, w, h, 置信度]，
  框坐标【已经是 imgsz 输入尺度下的像素值】（ultralytics 导出时把 DFL/anchor
  解码烘焙进了计算图），且最后一维已过 Sigmoid。所以这里不需要手写 anchor 解码，
  只做：阈值筛选 → xywh→xyxy → 反 letterbox → NMS。

预处理喂法与 ultralytics 的 NCNN 后端一致：把 RGB / CHW / 除以 255 的 float32
numpy 直接构造 ncnn.Mat（实测 ncnn.Mat((3,H,W) float) 会被解释为 c=3,h=H,w=W）。
"""

import os
import re
import time

import cv2
import numpy as np
import ncnn

# models/ 目录（本文件在 code/detect_code/，上两级即项目根）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(_PROJECT_ROOT, "models")

# 单类模型，类别名
CLASS_NAMES = ["ball"]


def _read_imgsz(model_dir):
    """从 metadata.yaml 或目录名推断方形输入边长 imgsz。"""
    meta = os.path.join(model_dir, "metadata.yaml")
    if os.path.exists(meta):
        with open(meta, "r", encoding="utf-8") as f:
            txt = f.read()
        # metadata.yaml 里 imgsz 是一个两元素列表，取第一个数字即可
        m = re.search(r"imgsz:\s*\n\s*-\s*(\d+)", txt)
        if m:
            return int(m.group(1))
    # 退回目录名后缀，如 best_ncnn_320 -> 320
    m = re.search(r"(\d+)\s*$", os.path.basename(model_dir.rstrip("/")))
    if m:
        return int(m.group(1))
    raise ValueError(f"无法从 {model_dir} 推断 imgsz")


def letterbox(img, new_size, color=(114, 114, 114)):
    """
    保持长宽比把 img 缩放并填充到 new_size×new_size 方形。
    返回 (padded, scale, pad_left, pad_top)，用于把检测框还原回原图。
    """
    h, w = img.shape[:2]
    scale = min(new_size / w, new_size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    pad_w, pad_h = new_size - nw, new_size - nh
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return padded, scale, left, top


class YoloNcnn:
    """一个 YOLO11n NCNN 模型的推理器。"""

    def __init__(self, model_dir, conf_thres=0.25, iou_thres=0.45,
                 num_threads=4, use_fp16=True):
        self.model_dir = model_dir
        self.imgsz = _read_imgsz(model_dir)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        self.net = ncnn.Net()
        # 树莓派 4 核，多线程 + fp16 提高吞吐
        self.net.opt.num_threads = num_threads
        self.net.opt.use_fp16_packed = use_fp16
        self.net.opt.use_fp16_storage = use_fp16
        self.net.opt.use_fp16_arithmetic = use_fp16
        self.net.load_param(os.path.join(model_dir, "model.ncnn.param"))
        self.net.load_model(os.path.join(model_dir, "model.ncnn.bin"))

    def preprocess(self, frame):
        """BGR 原图 -> (ncnn.Mat, scale, pad_left, pad_top)。"""
        padded, scale, left, top = letterbox(frame, self.imgsz)
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        # HWC -> CHW，归一化到 [0,1]
        chw = np.ascontiguousarray(rgb.transpose(2, 0, 1).astype(np.float32) / 255.0)
        # 注意：ncnn.Mat(numpy) 只是【包装】numpy 缓冲区、不拷贝；chw 一旦离开本函数
        # 作用域被回收，Mat 就成了悬空指针（会段错误）。用 .clone() 让 Mat 拥有自己的内存。
        return ncnn.Mat(chw).clone(), scale, left, top

    def forward(self, mat):
        """跑一次前向，返回 out0 的 numpy 副本，形状 (5, N)。"""
        with self.net.create_extractor() as ex:
            ex.input("in0", mat)   # mat 已是 preprocess 里 clone 出的自有内存
            _, out0 = ex.extract("out0")
            return np.array(out0).copy()

    def decode(self, out, scale, left, top, orig_w, orig_h):
        """
        out: (5, N)，把满足阈值的框解码 + NMS + 反 letterbox 到原图坐标。
        返回 list[(x1, y1, x2, y2, conf)]（int 坐标）。
        """
        out = out.T  # (N, 5)
        conf = out[:, 4]
        keep = conf >= self.conf_thres
        out, conf = out[keep], conf[keep]
        if out.shape[0] == 0:
            return []

        cx, cy, bw, bh = out[:, 0], out[:, 1], out[:, 2], out[:, 3]
        # letterbox 坐标 -> 原图坐标
        x1 = (cx - bw / 2 - left) / scale
        y1 = (cy - bh / 2 - top) / scale
        x2 = (cx + bw / 2 - left) / scale
        y2 = (cy + bh / 2 - top) / scale

        boxes = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1)  # xywh for NMS
        idxs = cv2.dnn.NMSBoxes(
            boxes.tolist(), conf.tolist(), self.conf_thres, self.iou_thres
        )
        if len(idxs) == 0:
            return []
        idxs = np.array(idxs).flatten()

        results = []
        for i in idxs:
            rx1 = int(max(0, min(orig_w - 1, x1[i])))
            ry1 = int(max(0, min(orig_h - 1, y1[i])))
            rx2 = int(max(0, min(orig_w - 1, x2[i])))
            ry2 = int(max(0, min(orig_h - 1, y2[i])))
            results.append((rx1, ry1, rx2, ry2, float(conf[i])))
        return results

    def detect(self, frame):
        """端到端检测一帧 BGR，返回 (detections, timing_dict)。"""
        h, w = frame.shape[:2]
        t0 = time.perf_counter()
        mat, scale, left, top = self.preprocess(frame)
        t1 = time.perf_counter()
        out = self.forward(mat)
        t2 = time.perf_counter()
        dets = self.decode(out, scale, left, top, w, h)
        t3 = time.perf_counter()
        timing = {
            "pre_ms": (t1 - t0) * 1e3,
            "infer_ms": (t2 - t1) * 1e3,
            "post_ms": (t3 - t2) * 1e3,
            "total_ms": (t3 - t0) * 1e3,
        }
        return dets, timing

    def release(self):
        self.net.clear()


def draw_detections(frame, dets):
    """在帧上画框和置信度。"""
    for (x1, y1, x2, y2, conf) in dets:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{CLASS_NAMES[0]} {conf:.2f}", (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return frame
