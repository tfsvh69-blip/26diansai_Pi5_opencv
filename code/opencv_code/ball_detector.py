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

        # === 金属确认参数 ===
        self.hi_v = 160           # "最亮"阈值（高光分用到）
        self.min_vmax = 100       # 硬门槛：候选区域最大亮度低于此值直接拒绝(排除暗淡假阳性)
        self.min_vstd = 5         # 硬门槛：候选区域亮度标准差(对比度)低于此值直接拒绝(排除平坦假阳性)

        # === 球体轮廓阈值（比 hi_v 低很多，圈出整个球体而非只有高光点）===
        # 用于主目标的亚像素质心细化 + 二值化调试/网页窗口显示，不参与 _score() 打分。
        # 默认值是起点，需要现场对着实体钢珠用滑块调（各环境光照/球面反光程度不同）。
        self.body_v = 50

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

        # 主目标最近一次的球体轮廓掩膜（整幅画面尺寸，球=255/背景=0），给调试窗口/网页
        # 二值化预览用，不参与检测逻辑；不受 reset() 影响，只在 detect() 里被覆盖更新。
        # 没有主目标或细化失败时是 None。
        self.last_mask_full = None

        # === 运行状态（不持久化）===
        self.reset()

    TUNABLE = (
        "param1", "param2", "min_radius", "max_radius", "blur_ksize",
        "hi_v", "min_vmax", "min_vstd", "body_v",
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

    # ---- 主目标球体轮廓质心细化 ----

    def _refine_center(self, hsv, x, y, r):
        """
        在 Hough 给出的 (x,y,r) 附近做一次局部二值化+质心细化，得到更稳的亚像素圆心。

        Hough 的圆心直接被上游 int 化，量化噪声会让静止目标在相邻帧的整数坐标间跳动；
        这里改用"局部阈值分割出整个球体轮廓 + 图像矩算加权质心"，比单次 Hough 投票更
        抗噪声，还顺带产出一张完整的球体二值掩膜供调试/网页窗口显示。

        返回 (cx_f, cy_f, mask_patch, (x0,y0)) 或 None（细化失败，调用方回退用原始
        Hough 坐标，不影响现有正确性）。mask_patch 是 uint8 0/255 局部掩膜，(x0,y0) 是
        它左上角在整幅画面里的偏移，供拼回整幅画布。
        """
        H, W = hsv.shape[:2]
        half = max(int(round(r * 1.6)), 4)
        x0 = max(0, x - half)
        x1 = min(W, x + half)
        y0 = max(0, y - half)
        y1 = min(H, y + half)
        if x1 - x0 < 6 or y1 - y0 < 6:
            return None

        V = hsv[y0:y1, x0:x1, 2]
        _, mask = cv2.threshold(V, self.body_v, 255, cv2.THRESH_BINARY)
        mask = mask.astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num <= 1:
            return None  # 全黑，没有前景

        # 只保留离 Hough 圆心最近的连通域（排除背景杂散高光/噪声误入）
        cxl, cyl = x - x0, y - y0
        best_lbl, best_dist = None, None
        for lbl in range(1, num):
            dx = centroids[lbl][0] - cxl
            dy = centroids[lbl][1] - cyl
            dist = dx * dx + dy * dy
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_lbl = lbl

        area = stats[best_lbl, cv2.CC_STAT_AREA]
        expected = np.pi * r * r
        if area < 0.3 * expected or area > 3.0 * expected:
            return None  # 面积明显不像一个球，细化不可信

        comp_mask = np.where(labels == best_lbl, 255, 0).astype(np.uint8)
        m = cv2.moments(comp_mask, binaryImage=True)
        if m["m00"] <= 0:
            return None

        cx_f = x0 + m["m10"] / m["m00"]
        cy_f = y0 + m["m01"] / m["m00"]
        return cx_f, cy_f, comp_mask, (x0, y0)

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

        # === 硬否决：不够亮 或 不够有对比度 → 不是钢珠 ===
        # min_vmax/min_vstd 两个独立下限，任一没达标就直接拒绝（原来是"两者都极低才拒绝"
        # 的弱组合条件，几乎挡不住背景纹理/反光凑出来的假阳性候选，导致 Hough 到处乱报圆时
        # 打分照样能通过、框到处跳。现在接成真正的硬门槛，配合 hi_v/body_v 用滑条现场调。
        if vmax < self.min_vmax or vstd < self.min_vstd:
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

            # 对主目标做局部球体轮廓质心细化：比原始 Hough 整数圆心更抗噪声，顺带产出
            # 完整球体二值掩膜供调试/网页窗口显示。细化失败就原样保留 Hough 坐标。
            refined = self._refine_center(hsv, bx, by, br)
            if refined is not None:
                cx_f, cy_f, mask_patch, (mx0, my0) = refined
                bx, by = int(round(cx_f)), int(round(cy_f))
                full = np.zeros(hsv.shape[:2], dtype=np.uint8)
                full[my0:my0 + mask_patch.shape[0], mx0:mx0 + mask_patch.shape[1]] = mask_patch
                self.last_mask_full = full
                dedup[0] = (bx, by, br, best[3])
            else:
                self.last_mask_full = None

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
        # 本帧没有真实检测，之前那张球体掩膜不再对应当前画面，清掉避免调试/网页窗口
        # 显示一张过时位置的白色轮廓。
        self.last_mask_full = None
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
