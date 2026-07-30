#!/usr/bin/env python3
"""
BoT-SORT（无 ReID 轻量版）—— 保证检测框在帧间的"连续性"
环境: /home/hao/vision_env/bin/python3（仅依赖 numpy + opencv，本机没装 scipy/lap/
ultralytics/filterpy，所以匈牙利分配、卡尔曼滤波、相机运动补偿全部手写，不引入新依赖）

背景 & 取舍：
  真正的 BoT-SORT = ByteTrack 的高/低置信度两级关联 + 相机运动补偿(GMC)
  + 用 (cx,cy,w,h) 而非长宽比 'a' 建模的改良卡尔曼滤波 + 可选 ReID 外观特征。
  这里按用户要求【关闭 ReID】（球是单类、无外观可辨识度，且树莓派5跑 ReID CNN
  划不来），所以本质是"ByteTrack 关联 + GMC + 改良卡尔曼"，与官方 BoT-SORT
  在 --with-reid False 模式下完全等价的算法结构。
  匈牙利分配用手写 O(n^3) 实现（不依赖 scipy.optimize / lap.lapjv）——球的数量
  通常个位数，性能完全够用。

核心流程（每帧 BotSort.update(dets, frame)）：
  1. 所有轨迹先用卡尔曼滤波 predict() 前推一步；
  2. GMC 用稀疏光流估计"上一帧->这一帧"的相机仿射运动，修正第1步的预测位置
     （否则镜头一晃，预测框和新检测框对不上，误判成"丢失重新起号"）；
  3. 第一阶段：高置信度检测 vs 全部轨迹(含 lost 宽限期内的)，IoU 匈牙利匹配；
  4. 第二阶段：低置信度检测 vs 第一阶段仍未匹配、但上一帧还是 tracked 的轨迹
     （ByteTrack 的关键点：低分检测不足以【起新轨迹】，但足以【延续已有轨迹】，
     用来找回运动模糊/部分遮挡时置信度掉下去的框）；
  5. 两阶段都没匹配上的轨迹进入/保持 lost，超过 max_time_lost 帧才真正丢弃；
  6. 剩下没匹配上的高置信度检测，达到 new_track_thresh 才起新轨迹。

调用方（v1.1.py）额外做的事：
  - 维护 primary_id：只要该 id 还在返回列表里（tracked 或宽限期内的 lost）就一直
    认它是主目标，不因为"某一帧别的球恰好面积更大"就跳目标；
  - 对 lost 状态的主目标，用卡尔曼预测位置继续输出（画面上会标注 LOST），
    真正超时被移除后才重新按"面积最大"挑新的主目标。
"""

import cv2
import numpy as np

TRACKED = "tracked"
LOST = "lost"


# ----------------------------- 匈牙利分配 -----------------------------

def _hungarian_core(cost):
    """cost: (n, m) ndarray, n <= m，最小化总代价。返回长度 n 的 list：第 i 行分配到的列。
    经典 O(n^2 * m) 势函数实现（Kuhn-Munkres），1-indexed 内部计算，结果转回 0-indexed。"""
    n, m = cost.shape
    INF = float("inf")
    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)      # p[j] = 分配到列 j 的行号（1-indexed），0 表示空
    way = [0] * (m + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = -1
            for j in range(1, m + 1):
                if not used[j]:
                    cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    row_to_col = [-1] * n
    for j in range(1, m + 1):
        if p[j] != 0:
            row_to_col[p[j] - 1] = j - 1
    return row_to_col


def linear_sum_assignment(cost):
    """对齐 scipy.optimize.linear_sum_assignment 的接口：返回 (row_ind, col_ind)。"""
    cost = np.asarray(cost, dtype=float)
    n, m = cost.shape
    if n == 0 or m == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    if n <= m:
        row_to_col = _hungarian_core(cost)
        return np.arange(n), np.array(row_to_col, dtype=int)
    else:
        col_to_row = _hungarian_core(cost.T)
        return np.array(col_to_row, dtype=int), np.arange(m)


def iou_batch(boxes_a, boxes_b):
    """两组 xyxy 框两两 IoU，返回 (len(a), len(b)) 矩阵。"""
    a = np.asarray(boxes_a, dtype=float)
    b = np.asarray(boxes_b, dtype=float)
    if a.size == 0 or b.size == 0:
        return np.zeros((len(a), len(b)), dtype=float)
    ax1, ay1, ax2, ay2 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    area_a = np.clip(ax2 - ax1, 0, None) * np.clip(ay2 - ay1, 0, None)
    area_b = np.clip(bx2 - bx1, 0, None) * np.clip(by2 - by1, 0, None)
    ix1 = np.maximum(ax1[:, None], bx1[None, :])
    iy1 = np.maximum(ay1[:, None], by1[None, :])
    ix2 = np.minimum(ax2[:, None], bx2[None, :])
    iy2 = np.minimum(ay2[:, None], by2[None, :])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


# ----------------------------- 卡尔曼滤波 (x,y,w,h) -----------------------------

class KalmanFilterXYWH:
    """
    等速运动模型，状态 = [cx, cy, w, h, vcx, vcy, vw, vh]。
    用 w、h 直接建模（而非原始 SORT/DeepSORT 的长宽比 a），是 BoT-SORT 相对
    DeepSORT 的改进点之一：目标尺度变化时更稳定。噪声标准差按当前 w/h 缩放，
    是 DeepSORT/BoT-SORT 一脉相承的做法（框越大，允许的像素级噪声也越大）。
    """

    def __init__(self):
        ndim, dt = 4, 1.0
        self._motion_mat = np.eye(2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement):
        mean = np.r_[measurement, np.zeros(4)]
        w, h = measurement[2], measurement[3]
        std = [
            2 * self._std_weight_position * w, 2 * self._std_weight_position * h,
            2 * self._std_weight_position * w, 2 * self._std_weight_position * h,
            10 * self._std_weight_velocity * w, 10 * self._std_weight_velocity * h,
            10 * self._std_weight_velocity * w, 10 * self._std_weight_velocity * h,
        ]
        return mean, np.diag(np.square(std))

    def predict(self, mean, cov):
        w, h = mean[2], mean[3]
        std = [
            self._std_weight_position * w, self._std_weight_position * h,
            self._std_weight_position * w, self._std_weight_position * h,
            self._std_weight_velocity * w, self._std_weight_velocity * h,
            self._std_weight_velocity * w, self._std_weight_velocity * h,
        ]
        motion_cov = np.diag(np.square(std))
        mean = self._motion_mat @ mean
        cov = self._motion_mat @ cov @ self._motion_mat.T + motion_cov
        return mean, cov

    def project(self, mean, cov):
        w, h = mean[2], mean[3]
        std = [self._std_weight_position * w, self._std_weight_position * h,
               self._std_weight_position * w, self._std_weight_position * h]
        innovation_cov = np.diag(np.square(std))
        mean = self._update_mat @ mean
        cov = self._update_mat @ cov @ self._update_mat.T + innovation_cov
        return mean, cov

    def update(self, mean, cov, measurement):
        proj_mean, proj_cov = self.project(mean, cov)
        kalman_gain = (cov @ self._update_mat.T) @ np.linalg.inv(proj_cov)
        innovation = measurement - proj_mean
        new_mean = mean + kalman_gain @ innovation
        new_cov = cov - kalman_gain @ proj_cov @ kalman_gain.T
        return new_mean, new_cov


# ----------------------------- 相机运动补偿 (GMC) -----------------------------

class GMC:
    """
    稀疏光流版全局运动补偿（对标 BoT-SORT 的 GMC "sparseOptFlow" 方案）。
    在缩小分辨率的灰度图上找角点、光流跟踪到当前帧、RANSAC 估仿射变换，
    再把平移量换算回原图分辨率。检测框区域会被抠掉不选特征点，避免把
    球自己的运动当成"背景/相机运动"。
    """

    def __init__(self, downscale=0.5, max_corners=200, min_points=8):
        self.downscale = downscale
        self.max_corners = max_corners
        self.min_points = min_points
        self.prev_gray = None
        self.prev_pts = None

    def _to_gray_small(self, frame):
        if self.downscale != 1.0:
            frame = cv2.resize(frame, None, fx=self.downscale, fy=self.downscale,
                                interpolation=cv2.INTER_LINEAR)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def _detect_points(self, gray, dets_xyxy):
        mask = np.full(gray.shape, 255, dtype=np.uint8)
        s = self.downscale
        gh, gw = gray.shape
        for det in (dets_xyxy or []):
            x1, y1, x2, y2 = det[0], det[1], det[2], det[3]
            mx1 = max(0, int(x1 * s) - 2)
            my1 = max(0, int(y1 * s) - 2)
            mx2 = min(gw, int(x2 * s) + 2)
            my2 = min(gh, int(y2 * s) + 2)
            mask[my1:my2, mx1:mx2] = 0
        return cv2.goodFeaturesToTrack(
            gray, maxCorners=self.max_corners, qualityLevel=0.01,
            minDistance=15, mask=mask)

    def apply(self, frame, dets_xyxy=None):
        """返回 2x3 仿射矩阵 M（原图分辨率下，当前 ≈ M @ [上一帧点;1]）；
        首帧或估计失败返回 None（等价于"相机没动"）。"""
        gray = self._to_gray_small(frame)
        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_pts = self._detect_points(gray, dets_xyxy)
            return None

        M = None
        if self.prev_pts is not None and len(self.prev_pts) >= self.min_points:
            curr_pts, status, _err = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, self.prev_pts, None,
                winSize=(21, 21), maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
            if status is not None:
                status = status.reshape(-1).astype(bool)
                prev_ok = self.prev_pts.reshape(-1, 2)[status]
                curr_ok = curr_pts.reshape(-1, 2)[status]
                if len(prev_ok) >= self.min_points:
                    m_small, _inliers = cv2.estimateAffinePartial2D(
                        prev_ok, curr_ok, method=cv2.RANSAC,
                        ransacReprojThreshold=3, maxIters=500, confidence=0.95)
                    if m_small is not None:
                        M = m_small.astype(float)
                        M[:, 2] /= self.downscale   # 缩小坐标系下的平移换算回原图分辨率

        self.prev_gray = gray
        self.prev_pts = self._detect_points(gray, dets_xyxy)
        return M

    def reset(self):
        self.prev_gray = None
        self.prev_pts = None


# ----------------------------- 单条轨迹 -----------------------------

class STrack:
    shared_kf = KalmanFilterXYWH()
    _next_id = 1

    @classmethod
    def reset_ids(cls):
        cls._next_id = 1

    @classmethod
    def _alloc_id(cls):
        i = cls._next_id
        cls._next_id += 1
        return i

    def __init__(self, xyxy, score):
        x1, y1, x2, y2 = xyxy
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        w, h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
        self.mean, self.cov = self.shared_kf.initiate(np.array([cx, cy, w, h], dtype=float))
        self.score = float(score)
        self.track_id = self._alloc_id()
        self.state = TRACKED
        self.time_since_update = 0
        self.hits = 1

    @property
    def tlbr(self):
        cx, cy, w, h = self.mean[:4]
        return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)

    def predict(self):
        self.mean, self.cov = self.shared_kf.predict(self.mean, self.cov)

    def apply_gmc(self, M):
        """相机运动补偿：位置做仿射变换，速度只跟着旋转/缩放部分转（不叠加平移）。"""
        r = M[:2, :2]
        t = M[:2, 2]
        r8 = np.eye(8)
        r8[0:2, 0:2] = r
        r8[4:6, 4:6] = r
        self.mean = r8 @ self.mean
        self.mean[0:2] += t
        self.cov = r8 @ self.cov @ r8.T

    def update(self, xyxy, score):
        x1, y1, x2, y2 = xyxy
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        w, h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
        self.mean, self.cov = self.shared_kf.update(
            self.mean, self.cov, np.array([cx, cy, w, h], dtype=float))
        self.score = float(score)
        self.state = TRACKED
        self.time_since_update = 0
        self.hits += 1

    def mark_lost(self):
        self.state = LOST
        self.time_since_update += 1


def _match(tracks, dets, dist_thresh):
    """tracks: list[STrack]，dets: list[(x1,y1,x2,y2,conf)]。
    返回 (matched[(ti,di)], unmatched_track_idx, unmatched_det_idx)。"""
    if not tracks or not dets:
        return [], list(range(len(tracks))), list(range(len(dets)))
    iou = iou_batch([t.tlbr for t in tracks], [d[:4] for d in dets])
    cost = 1.0 - iou
    row_ind, col_ind = linear_sum_assignment(cost)
    matched, mt, md = [], set(), set()
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] <= dist_thresh:
            matched.append((r, c))
            mt.add(r)
            md.add(c)
    unmatched_tracks = [i for i in range(len(tracks)) if i not in mt]
    unmatched_dets = [i for i in range(len(dets)) if i not in md]
    return matched, unmatched_tracks, unmatched_dets


# ----------------------------- 跟踪器 -----------------------------

class BotSort:
    """
    ByteTrack 式两级关联 + GMC + 改良卡尔曼，ReID 关闭。
    用法：
        tracker = BotSort(high_thresh=cfg['conf'], frame_rate=17)
        tracks = tracker.update(dets, frame)   # dets 需在【低阈值】下解出（见 low_thresh）
        for t in tracks:
            t.track_id, t.tlbr, t.score, t.state  # state: 'tracked' / 'lost'
    """

    def __init__(self, high_thresh=0.25, low_thresh=0.1, new_track_thresh=None,
                 match_thresh_high=0.8, match_thresh_low=0.5,
                 frame_rate=17.0, track_buffer_sec=1.5,
                 use_gmc=True, gmc_downscale=0.5):
        self.high_thresh = high_thresh
        self.low_thresh = min(low_thresh, max(0.01, high_thresh - 0.05))
        self.new_track_thresh = (new_track_thresh if new_track_thresh is not None
                                  else min(0.95, high_thresh + 0.1))
        self.match_thresh_high = match_thresh_high
        self.match_thresh_low = match_thresh_low
        self.max_time_lost = max(1, int(round(frame_rate * track_buffer_sec)))
        self.gmc = GMC(downscale=gmc_downscale) if use_gmc else None
        self.last_gmc_ok = False

        self.frame_id = 0
        self.tracked_stracks = []
        self.lost_stracks = []
        STrack.reset_ids()

    def update(self, dets, frame):
        self.frame_id += 1
        dets = dets or []
        high = [d for d in dets if d[4] >= self.high_thresh]
        low = [d for d in dets if self.low_thresh <= d[4] < self.high_thresh]

        M = None
        if self.gmc is not None:
            M = self.gmc.apply(frame, dets)
        self.last_gmc_ok = M is not None

        pool = self.tracked_stracks + self.lost_stracks
        for t in pool:
            t.predict()
            if M is not None:
                t.apply_gmc(M)

        # 第一阶段：高置信度检测 vs 全部轨迹
        matched_a, unmatched_tracks_a, unmatched_high = _match(pool, high, self.match_thresh_high)

        # 第二阶段：低置信度检测 只找回"上一帧仍是 tracked"的剩余轨迹
        r_tracked = [pool[i] for i in unmatched_tracks_a if pool[i].state == TRACKED]
        matched_b, _unmatched_tracks_b, _unmatched_low = _match(r_tracked, low, self.match_thresh_low)

        matched_ids = set()
        for ti, di in matched_a:
            pool[ti].update(high[di][:4], high[di][4])
            matched_ids.add(id(pool[ti]))
        for ri, di in matched_b:
            r_tracked[ri].update(low[di][:4], low[di][4])
            matched_ids.add(id(r_tracked[ri]))

        still_alive = []
        for t in pool:
            if id(t) not in matched_ids:
                t.mark_lost()
            if t.time_since_update <= self.max_time_lost:
                still_alive.append(t)

        for di in unmatched_high:
            if high[di][4] >= self.new_track_thresh:
                still_alive.append(STrack(high[di][:4], high[di][4]))

        self.tracked_stracks = [t for t in still_alive if t.state == TRACKED]
        self.lost_stracks = [t for t in still_alive if t.state == LOST]
        return still_alive
