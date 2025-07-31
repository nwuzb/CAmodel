#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run_3DModel_H5D.py
=====================
使用新的 5 维 EKF (u,v,Z,dZ,H) 追踪行人，动态估计头高。
与原 Run_3DModel_5D.py 类似，但加载 `Physical3DKF_H5D`。
"""
from __future__ import annotations
import os, sys, importlib, importlib.util, shutil
from pathlib import Path
import cv2

SCRIPT_DIR = Path(__file__).resolve().parent       # CAmodel/deep_sort
PROJECT_ROOT = SCRIPT_DIR.parent                   # CAmodel
INNER_PKG_DIR = SCRIPT_DIR / "deep_sort"          # 真正包目录

for p in (PROJECT_ROOT, SCRIPT_DIR, INNER_PKG_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0,str(p))

# 手动导入 kalman_filter, tracker，确保引用一致

def _manual_import(name:str, path:Path):
    spec = importlib.util.spec_from_file_location(name,str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules[name]=mod
    return mod

_manual_import("deep_sort.kalman_filter", INNER_PKG_DIR/"kalman_filter.py")
_manual_import("deep_sort.tracker",         INNER_PKG_DIR/"tracker.py")
# 同步引用
sys.modules["deep_sort.tracker"].kalman_filter = sys.modules["deep_sort.kalman_filter"]

print("\n>>> 已将 DeepSORT KalmanFilter 替换为 动态头高 5D EKF (H5D) <<<\n")

DEFAULT_HEAD = 0.30 # 默认头高，yolo检测的也是头部框，也是头高。

# 摄像机参数（可按需修改）

F_PIXELS, CX, CY = 795.0, 268.0, 137.0 # 2k 0.6裁剪右侧 相机内参 A
# F_PIXELS, CX, CY = 795.0, 1268.0, 137.0 # 2k 0.6裁剪左侧 相机内参B

# F_PIXELS, CX, CY = 1600.3, 515.0, 153.0 # 4k 0.6裁剪右侧 相机内参 C
# F_PIXELS, CX, CY = 1600.3, 1789.0, 153.0 # 4k 0.6裁剪左侧 相机内参 D

# 输入输出路径
# FOLDER = "/Users/binzeng/MA/GT_videos/gt_60_videos/test_gt_60_2_clip_02_right_47_test" #2k 0.6裁剪右侧 相机内参 A ⭐️
# FOLDER = "/Users/binzeng/MA/GT_videos/gt_60_videos/test_60fps_gt_4_clip_02_rightpart_99_test" # 调试中⭐️

FOLDER = "/Users/binzeng/MA/GT_videos/gt_60_videos/gt_60_2_clip_03_right_124" #2k 0.6裁剪右侧 相机内参 A
# FOLDER = "/Users/binzeng/MA/GT_videos/gt_60_videos/gt_60_2_clip_02_right_47" #2k 0.6裁剪右侧 相机内参 A
# FOLDER = "/Users/binzeng/MA/GT_videos/gt_60_videos/gt_60_5_clip_03_right_32" #2k 0.6裁剪右侧 相机内参 A

# FOLDER = "/Users/binzeng/MA/GT_videos/gt_60_videos/gt_60_5_clip_01_left_28" #2k 0.6裁剪左侧 相机内参 B

# FOLDER = "/Users/binzeng/MA/GT_videos/gt_60_videos/gt_60_4_clip_02_right_100" #4k 0.6裁剪右侧 相机内参 C
# FOLDER = "/Users/binzeng/MA/GT_videos/gt_60_videos/gt_60_1_clip_01_right_54" #4k 0.6裁剪右侧 相机内参 C

# FOLDER = "/Users/binzeng/MA/GT_videos/gt_60_videos/gt_60_6_clip_06_left_46" #4k 0.6裁剪左侧 相机内参 D
# FOLDER = "/Users/binzeng/MA/GT_videos/gt_60_videos/gt_60_10_clip_01_left_53" #4k 0.6裁剪左侧 相机内参 D
# FOLDER = "/Users/binzeng/MA/GT_videos/gt_60_videos/gt_60_13_clip_01_left_16" #4k 0.6裁剪左侧 相机内参 D
# FOLDER = "/Users/binzeng/MA/GT_videos/gt_60_videos/gt_60_13_clip03_left_9" #4k 0.6裁剪左侧 相机内参 D

video_candidates = [f for f in os.listdir(FOLDER) if f.lower().endswith((".mp4",))]
if not video_candidates:
    raise FileNotFoundError("未找到视频")
VIDEO = os.path.join(FOLDER, video_candidates[0])
GT   = os.path.join(FOLDER, "gt.txt")
OUT_DIR = FOLDER + "-results-平滑H5D"
if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)
os.makedirs(OUT_DIR,exist_ok=True)

# 读取 FPS
cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
cap.release()
DT = 1.0/fps
print(f"[H5D] Detected FPS={fps:.2f}, Δt={DT:.4f}s")

# 导入 H5D KF
# 额外依赖
import numpy as np

from physical3dkf_h5d import Physical3DKF_H5D

# ============ 使 KF 与上方相机参数保持一致 ============
class CustomPhysical3DKF_H5D(Physical3DKF_H5D):
    """包装 Physical3DKF_H5D，使其默认使用当前文件中 F_PIXELS/CX/CY 变量。"""
    def __init__(self, f: float = F_PIXELS, cx: float = CX, cy: float = CY,
                 H0: float = DEFAULT_HEAD, dt: float = DT, max_decel: float = 1.0):
        super().__init__(f=f, cx=cx, cy=cy, H0=H0, dt=dt, max_decel=max_decel)

# 注入 KF
kf_mod = sys.modules["deep_sort.kalman_filter"]
# 用自定义 KF 替换默认 KalmanFilter
kf_mod.KalmanFilter = CustomPhysical3DKF_H5D
kf_mod.AccelerationKalmanFilter = CustomPhysical3DKF_H5D

# ---- DeepSORT 下游适配补丁 ----
import deep_sort.linear_assignment as _la
import importlib as _ibl

# ---- 重写 gate_cost_matrix 为 3D (u,v,h_px) 门控 ----
_orig_gate = _la.gate_cost_matrix  # 备份

def _gate_cost_matrix_3d(kf, cost_matrix, tracks, detections, track_indices, detection_indices,
                         gated_cost=_la.INFTY_COST):
    """对 DeepSORT 原函数的裁剪：使用纯位置 (u,v) 2D 马氏距离进行门控。"""
    gating_dim = 2  # 纯位置(u,v)
    gating_threshold = kf.chi2inv95[gating_dim]
    measurements = np.asarray([detections[i].to_xyah() for i in detection_indices])
    for row, track_idx in enumerate(track_indices):
        track = tracks[track_idx]
        gating_distance = kf.gating_distance(track.mean, track.covariance, measurements, only_position=True)
        cost_matrix[row, gating_distance > gating_threshold] = gated_cost
    return cost_matrix

# ========= 基于历史直线走廊的方向门控 ==========
CORRIDOR_HISTORY = 5       # 拟合直线所用的最近帧数,但是要注意低速或者静止时的影响
CORRIDOR_HALF_WIDTH = 20   # 走廊半宽 (像素)
BACKWARD_TOL = 5.0         # 允许轻微后退 (像素)

def _apply_corridor_gating(cost_matrix,
                           tracks,
                           detections,
                           track_indices,
                           detection_indices,
                           corridor_half_width: float = CORRIDOR_HALF_WIDTH,
                           backward_tol: float = BACKWARD_TOL,
                           min_hist_len: int = 2,
                           debug: bool = False, **kwargs):
    """使用“直线走廊”规则过滤匹配组合。

    1. 最近 CORRIDOR_HISTORY 帧中心点 -> 拟合主方向直线 (首尾向量)。
    2. 对检测中心点计算垂距 w 与前向位移 s。
    3. 若 w > corridor_half_width 或 s < -backward_tol，则置为 ∞ 代价。
    """
    # 兼容旧方向门控的参数名
    backward_tol = kwargs.pop('backward_tol', backward_tol)
    min_hist_len = kwargs.pop('min_age', min_hist_len)
    debug = kwargs.pop('debug', debug)

    for r, track_idx in enumerate(track_indices):
        track = tracks[track_idx]
        hist: list[np.ndarray] | None = getattr(track, "_center_hist", None)
        if hist is None or len(hist) < min_hist_len:
            # 历史不足 ⇒ 不做限制
            continue
        p_start = hist[0]
        p_end   = hist[-1]
        dir_vec = p_end - p_start
        norm = np.linalg.norm(dir_vec)
        if norm < 1e-3:
            continue
        dir_unit = dir_vec / norm
        for c, det_idx in enumerate(detection_indices):
            u, v, *_ = detections[det_idx].to_xyah()
            det_pt = np.asarray([u, v], dtype=float)
            diff   = det_pt - p_end
            s      = float(np.dot(diff, dir_unit))
            w      = float(np.linalg.norm(diff - s * dir_unit))
            if w > corridor_half_width or s < -backward_tol:
                cost_matrix[r, c] = _la.INFTY_COST
    return cost_matrix

# ====== Gating switches ======
ENABLE_MAHALANOBIS_GATING = False    # True → 使用马氏距离门控
GATE_USE_3D              = True       # True → (u,v,h_px)  False → 原 4D (u,v,a,h)
ENABLE_DIRECTION_GATING  = True      # True → 使用直线走廊门控
# -------------------------------------------------
_selected_gate = _gate_cost_matrix_3d if GATE_USE_3D else _orig_gate
if ENABLE_MAHALANOBIS_GATING:
    _la.gate_cost_matrix = _selected_gate
else:
    _la.gate_cost_matrix = lambda kf, cost_matrix, *a, **k: cost_matrix

# ------- 方向门控 + 外观阈值(按高度) 组合 -------
_tracker_mod = _ibl.import_module("deep_sort.tracker")

SMALL_OBJ_H = 30        # < px  → 外观阈值 small_thr
APPEAR_SMALL_THR = 0.1
APPEAR_LARGE_THR = 0.3


def _apply_height_threshold(cost_matrix, detections, detection_indices,
                            small_thr: float = APPEAR_SMALL_THR,
                            large_thr: float = APPEAR_LARGE_THR,
                            cut_h: int = SMALL_OBJ_H):
    """根据检测框高度对外观成本矩阵做阈值过滤。"""
    for col, det_idx in enumerate(detection_indices):
        h_px = detections[det_idx].to_xyah()[3]
        thr = small_thr if h_px < cut_h else large_thr
        # 把高于阈值的成本置为无穷
        cost_matrix[:, col] = np.where(cost_matrix[:, col] > thr, _la.INFTY_COST, cost_matrix[:, col])
    return cost_matrix


def _direction_height_gating(cost_matrix,
                              tracks,
                              detections,
                              track_indices,
                              detection_indices,
                              **kwargs):
    # 1) 方向走廊
    if ENABLE_DIRECTION_GATING:
        cost_matrix = _apply_corridor_gating(cost_matrix, tracks, detections, track_indices, detection_indices, **kwargs)
    # 2) 高度依赖的外观阈值
    cost_matrix = _apply_height_threshold(cost_matrix, detections, detection_indices)
    return cost_matrix

# 注入替换
_tracker_mod.apply_direction_gating = _direction_height_gating

# 2) bbox 计算重写及中心点历史
_track = _ibl.import_module("deep_sort.track")
_orig_update = _track.Track.update

def _to_tlwh_h5d(self):
    u, v, Z, _, H = self.mean[:5]
    h_px = F_PIXELS * H / Z
    a = getattr(self, "last_aspect", 0.5)
    w = a * h_px
    return np.array([u - w / 2, v - h_px / 2, w, h_px], dtype=float)

def _to_tlbr_h5d(self):
    tlwh = _to_tlwh_h5d(self)
    tlbr = tlwh.copy(); tlbr[2:]+=tlbr[:2]
    return tlbr

_track.Track.to_tlwh = _to_tlwh_h5d
_track.Track.to_tlbr = _to_tlbr_h5d

# 3) 在 update 记录宽高比 + 中心历史

def _update_h5d(self, kf, detection):
    xyah = detection.to_xyah()
    self.last_aspect = float(xyah[2])
    _orig_update(self, kf, detection)
    # -------- 记录中心历史，用于走廊门控 --------
    center = self.mean[:2].copy()
    hist = getattr(self, "_center_hist", [])
    hist.append(center)
    if len(hist) > CORRIDOR_HISTORY:
        hist.pop(0)
    self._center_hist = hist

_track.Track.update = _update_h5d

# -------- 运行评估脚本 --------
orig = SCRIPT_DIR/"评估追踪器性能yolo.py"
_spec = importlib.util.spec_from_file_location("eval_h5d", orig)
_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_eval)

_eval.FOLDER_PATH, _eval.VIDEO_PATH, _eval.GT_PATH, _eval.OUTPUT_DIR = FOLDER, VIDEO, GT, OUT_DIR
_eval.TRACKING_PARAMS.update({"use_acceleration":True})

if __name__ == "__main__":
    _eval.main()