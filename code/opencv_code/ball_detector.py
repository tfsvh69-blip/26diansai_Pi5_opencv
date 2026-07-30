#!/usr/bin/env python3
"""
纯 OpenCV 小钢珠检测核心 v2 —— 检测 + 跟踪融合
环境: /home/hao/vision_env/bin/python3

核心思路：
  不再单纯依赖每帧都完美检测到钢珠，而是维护一个跟踪滤波器：
  - 检测到了 → 更新位置、提高置信度
  - 没检测到 → 按速度惯性前推、置信度逐步衰减
  - 置信度 > 0 就继续报位置，串口持续发 found=1

改进要点（对比 v1）：
  1. 打分机制从"硬阈值否决"改为"多因素综合评分"，降低漏检率
     - 亮度分(30) + 高光分(30) + 对比度分(20) + 空间一致性奖励(40)
     - 即使某个因素弱，其他因素补上，总分为正就通过
  2. 加入速度预测，丢失期间不是静止不动而是按速度惯性滑行
  3. 置信度衰减替代硬 10 帧保持：满置信度可撑 ~1.3s(~40帧)才丢锁
  4. 保持锁定期间串口依然发 found=1，下位机不受干扰

接口兼容 v1：detect() 返回候选列表，pick_primary() 返回主目标。
替换 ball_detector.py 后，v1.0.py 和 tune_detector.py 无需修改调用方式。
"""

import cv2
import numpy as np


def _nms_cluster(circles, iou_thresh=0.35):
    """NMS 聚类合并重叠圆。"""
    if not circles:
        return []
    kept = []
    sorted_c = sorted(circles, key=lambda c: c[3], reverse=True)
    while sorted_c:
        best = sorted_c.pop(0)
        kept.append(best)
        remaining = []
        for c in sorted_c:
            d = np.hypot(c[0] - best[0], c[1] - best[1])
            if d >= iou_thresh * (c[2] + best[2]):
                remaining.append(c)
        sorted_c = remaining
    return kept


class BallDetector:
    """
    钢珠检测器 v2 —— 融合检测 + 跟踪。

    用法与 v1 一致：
      det = BallDetector().load_dict(cfg["detector"])
      cands = det.detect(frame)       # -> [(cx,cy,r,score), ...]
      prim  = det.pick_primary(cands) # -> (cx,cy,r,score) or None

    运行状态（_px, _py, _vx, _vy, _confidence, _smoothed）不持久化。
    新加跟踪参数会自动从旧版 config 中刨除（load_dict 只认 TUNABLE 内的 key）。
    """

    def __init__(self):
        # === Hough 参数 ===
        self.param1 = 120
        self.param2 = 22          # 配合打分防误检，比旧版 18 略收
        self.min_radius = 5
        self.max_radius = 22
        self.blur_ksize = 5

        # === 金属确认参数（已融入多因素评分，不再是硬否决）===
        self.hi_v = 160           # "最亮"阈值（高光分用到）
        self.min_vmax = 100       # 亮度分用到，不单独否决
        self.min_vstd = 5         # 对比度分用到，不单独否决

        # === NMS ===
        self.nms_iou = 0.35

        # === 跟踪滤波 ===
        self.track_max_dist = 40.0         # 检测/预测距离超此→不算同一目标
        self.match_dist_factor = 1.8       # 匹配窗口 = factor * 半径
        self.vel_alpha = 0.35              # 速度更新平滑
        self.vel_decay = 0.92              # 每帧速度衰减

        # === 置信度 ===
        self.conf_inc = 0.25               # 检出一帧 +0.25
        self.conf_dec = 0.025              # 漏一帧 -0.025
        # 从 1.0 衰减到 0 需要 40 帧（~1.3s）
        # 从 0 升到 1.0 需要 4 帧 连续检出

        # === 输出平滑 ===
        self.ema_alpha = 0.3

        # === 运行状态（不持久化）===
        self.reset()

    TUNABLE = (
        "param1", "param2", "min_radius", "max_radius", "blur_ksize",
        "hi_v", "min_vmax", "min_vstd",
        "nms_iou",
        "track_max_dist", "match_dist_factor", "vel_alpha", "vel_decay",
        "conf_inc", "conf_dec",
        "ema_alpha",
    )

    def as_dict(self):
        return {k: getattr(self, k) for k in self.TUNABLE}

    def load_dict(self, d):
        if isinstance(d, dict):
            for k in self.TUNABLE:
                if k in d and d[k] is not None:
                    setattr(self, k, d[k])
        return self

    def reset(self):
        """清空跟踪状态（目标明确移走后调用）。"""
        self._px = None        # 预测 x
        self._py = None        # 预测 y
        self._pr = None        # 预测 半径
        self._vx = 0.0         # 速度 x
        self._vy = 0.0         # 速度 y
        self._confidence = 0.0
        self._smoothed = None

    # ---- 内部打分 ----

    def _score(self, hsv, x, y, r, px=None, py=None, pr=None):
        """
        多因素综合评分。返回 >=0 为通过候选，-1 为明确拒绝。

        评分组成（满分 120）：
          亮度分(0~30) + 高光分(0~30) + 对比度分(0~20) + 非平坦惩罚(-10~0)
          + 空间一致性奖励(0~40)
        """
        H, W = hsv.shape[:2]
        x0 = max(0, x - r)
        x1 = min(W, x + r)
        y0 = max(0, y - r)
        y1 = min(H, y + r)
        if x1 - x0 < 6 or y1 - y0 < 6:
            return -1.0

        sub = hsv[y0:y1, x0:x1]
        cxl = x - x0
        cyl = y - y0
        yy, xx = np.ogrid[:sub.shape[0], :sub.shape[1]]
        inner = (xx - cxl)**2 + (yy - cyl)**2 <= (r * 0.85)**2
        if inner.sum() < 6:
            return -1.0

        V = sub[:, :, 2].astype(np.float32)
        vin = V[inner]

        vmax = float(vin.max())
        vstd = float(vin.std())

        # === 硬否决：完全没有内部变化 + 亮度极低 → 不是钢珠 ===
        if vstd < 3.0 and vmax < 80:
            return -1.0

        # ===== 1) 亮度分 (0~30) =====
        brightness = min(30.0, vmax / 4.0)

        # ===== 2) 高光分 (0~30) =====
        highlight_ratio = float((vin > self.hi_v).mean())
        specular = 30.0 * highlight_ratio * min(1.0, vmax / 200.0)

        # ===== 3) 对比度分 (0~20) =====
        contrast = min(20.0, vstd * 1.2)

        # ===== 4) 非平坦惩罚 (-10~0) =====
        flat_penalty = 0.0
        if vstd < 8.0:
            flat_penalty = -10.0 * (1.0 - vstd / 8.0)

        base = brightness + specular + contrast + flat_penalty

        # ===== 5) 空间一致性奖励 (0~40) =====
        # 检测位置离预测位置越近，加分越多
        spatial_bonus = 0.0
        if px is not None and py is not None and pr is not None:
            dist = np.hypot(x - px, y - py)
            match_radius = max(pr * self.match_dist_factor, 15.0)
            if dist < match_radius:
                # 在匹配范围内：平方衰减
                ratio = 1.0 - dist / match_radius
                spatial_bonus = 40.0 * ratio * ratio
            elif dist < self.track_max_dist:
                # 超出匹配范围但在可接受范围：少量奖励
                remainder = self.track_max_dist - match_radius
                if remainder > 1:
                    ratio = 1.0 - (dist - match_radius) / remainder
                    spatial_bonus = 5.0 * max(0.0, ratio)

        return base + spatial_bonus

    # ---- 核心检测 + 跟踪 ----

    def detect(self, bgr):
        """
        检测钢珠并更新跟踪状态。

        返回 [(cx, cy, r, score), ...] 列表：
        - 有实际检测到：返回 Hough + 打分后的候选
        - 没检测到但置信度 > 0：返回预测位置（score = 当前置信度）
        - 完全丢失：返回空列表

        调用方用 pick_primary() 取主目标（含 EMA 平滑）。
        """
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        k = self.blur_ksize | 1
        gray = cv2.medianBlur(gray, k)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        # Hough 找圆
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=12,
            param1=self.param1, param2=self.param2,
            minRadius=self.min_radius, maxRadius=self.max_radius,
        )

        # 逐一打分
        scored = []
        if circles is not None:
            for (x, y, r) in np.uint16(np.around(circles[0])):
                sc = self._score(
                    hsv, int(x), int(y), int(r),
                    self._px, self._py, self._pr,
                )
                if sc > 0:
                    scored.append((int(x), int(y), int(r), float(sc)))

        # NMS 去重
        dedup = _nms_cluster(scored, iou_thresh=self.nms_iou)

        # ===== 情况 A：检测到候选 =====
        if dedup:
            best = dedup[0]  # 评分最高
            bx, by, br = best[0], best[1], best[2]

            # 更新速度
            if self._px is not None and self._py is not None:
                self._vx = (1.0 - self.vel_alpha) * self._vx \
                           + self.vel_alpha * (bx - self._px)
                self._vy = (1.0 - self.vel_alpha) * self._vy \
                           + self.vel_alpha * (by - self._py)

            # 更新预测位置
            self._px = float(bx)
            self._py = float(by)
            self._pr = float(br)

            # 提升置信度
            self._confidence = min(1.0, self._confidence + self.conf_inc)

            return dedup

        # ===== 情况 B：无检测 → 用跟踪预测 =====
        if self._px is not None and self._confidence > 0:
            # 按速度惯性前推
            self._px += self._vx
            self._py += self._vy
            self._vx *= self.vel_decay
            self._vy *= self.vel_decay

            # 半径稳定在合理范围
            if self._pr is not None:
                self._pr = float(np.clip(
                    self._pr, self.min_radius, self.max_radius
                ))

            # 置信度衰减
            self._confidence -= self.conf_dec

            if self._confidence > 0:
                # 返回预测位置
                r_val = int(round(self._pr)) if self._pr is not None \
                        else int((self.min_radius + self.max_radius) / 2)
                return [(
                    int(self._px), int(self._py), r_val,
                    self._confidence,
                )]
            else:
                # 置信度归零 → 彻底丢失
                self._confidence = 0.0
                self._px = self._py = self._pr = None
                self._vx = self._vy = 0.0
                self._smoothed = None

        return []

    # ---- 主目标选取 + EMA 平滑 ----

    def pick_primary(self, cands):
        """
        从候选列表中取主目标（NMS 后评分最高的），并做 EMA 平滑。

        返回 (x, y, r, score) 或 None。
        """
        if not cands:
            self._smoothed = None
            return None

        cx, cy, r, sc = cands[0]

        if self._smoothed is None:
            self._smoothed = (cx, cy, r, sc)
        else:
            a = self.ema_alpha
            fx = int(a * cx + (1 - a) * self._smoothed[0])
            fy = int(a * cy + (1 - a) * self._smoothed[1])
            fr = int(a * r + (1 - a) * self._smoothed[2])
            fs = a * sc + (1 - a) * self._smoothed[3]
            self._smoothed = (fx, fy, fr, fs)

        return self._smoothed


# ---- 绘图辅助 ----

def draw(bgr, cands, primary=None):
    """画绿圈：所有候选绿线，主目标粗绿 + 中心点 + 坐标。"""
    for (x, y, r, sc) in cands:
        is_primary = (
            primary is not None
            and (x, y) == (primary[0], primary[1])
        )
        color = (0, 200, 0) if not is_primary else (0, 255, 0)
        thick = 2 if not is_primary else 3
        cv2.circle(bgr, (x, y), r, color, thick, cv2.LINE_AA)
        if is_primary:
            cv2.circle(bgr, (x, y), 3, (0, 255, 0), -1)
            cv2.putText(bgr, f"({x},{y})", (x + r + 4, y + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (0, 255, 0), 1, cv2.LINE_AA)
    return bgr
