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
FOLDER = "/Users/binzeng/MA/GT_videos/gt_60_videos/test_gt_60_2_clip_02_right_47_test" #2k 0.6裁剪右侧 相机内参 A ⭐️
# FOLDER = "/Users/binzeng/MA/GT_videos/gt_60_videos/test_60fps_gt_4_clip_02_rightpart_99_test" # 调试中⭐️

# FOLDER = "/Users/binzeng/MA/GT_videos/gt_60_videos/gt_60_2_clip_03_right_124" #2k 0.6裁剪右侧 相机内参 A
# FOLDER = "/Users/binzeng/MA/GT_videos/gt_60_videos/gt_60_2_clip_02_right_47" #2k 0.6裁剪右侧 相机内参 A
# FOLDER = "/Users/binzeng/MA/GT_videos/gt_60_videos/gt_60_5_clip_03_right_32" #2k 0.6裁剪右侧 相机内参 A

# FOLDER = "/Users/binzeng/MA/GT_videos/gt_60_videos/gt_60_5_clip_01_left_28" #2k 0.6裁剪左侧 相机内参 B

# FOLDER = "/Users/binzeng/MA/GT_videos/gt_60_videos/gt_60_4_clip_02_right_100" #4k 0.6裁剪右侧 相机内参 C
# FOLDER = "/Users/binzeng/MA/GT_videos/gt_60_videos/gt_60_1_clip_01_right_54" #4k 0.6裁剪右侧 相机内参 C

# FOLDER = "/Users/binzeng/MA/GT_videos/gt_60_videos/gt_60_6_clip_06_left_46" #4k 0.6裁剪左侧 相机内参 D
# FOLDER = "/Users/binzeng/MA/GT_videos/gt_60_videos/gt_60_10_clip_01_left_53" #4k 0.6裁剪左侧 相机内参 D

video_candidates = [f for f in os.listdir(FOLDER) if f.lower().endswith((".mp4",))]
if not video_candidates:
    raise FileNotFoundError("未找到视频")
VIDEO = os.path.join(FOLDER, video_candidates[0])
GT   = os.path.join(FOLDER, "gt.txt")
OUT_DIR = FOLDER + "-results-H5D"
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
    """对 DeepSORT 原函数的裁剪：使用 (u,v,h_px) 三维马氏距离进行门控。"""
    gating_dim = 3
    gating_threshold = kf.chi2inv95[gating_dim]
    measurements = np.asarray([detections[i].to_xyah() for i in detection_indices])
    for row, track_idx in enumerate(track_indices):
        track = tracks[track_idx]
        gating_distance = kf.gating_distance(track.mean, track.covariance, measurements, only_position=False)
        cost_matrix[row, gating_distance > gating_threshold] = gated_cost
    return cost_matrix

# 开关：控制是否使用自定义 3D gate
USE_3D_GATE = False  # True=3D gate, False=原4D gate ⭐️
if USE_3D_GATE:
    _la.gate_cost_matrix = _gate_cost_matrix_3d
else:
    _la.gate_cost_matrix = _orig_gate

# ---- 强制关闭所有门控（仅用于测试外观匹配） ---- ⚠️ ⚠️ ⚠️
DISABLE_GATING = True  # ⚠️ 调试完后务必改回 False
if DISABLE_GATING:
    # 1) 禁用马氏距离 / Chi2 门控：直接返回原 cost_matrix
    _la.gate_cost_matrix = lambda kf, cost_matrix, tracks, detections, track_indices, detection_indices, **kwargs: cost_matrix
    # 2) 禁用方向门控：apply_direction_gating 恒等返回
    _tracker_mod = _ibl.import_module("deep_sort.tracker")
    _tracker_mod.apply_direction_gating = lambda cost_matrix, *args, **kwargs: cost_matrix

# 2) bbox 计算重写
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

# 3) 在 update 记录宽高比
import numpy as np

def _update_h5d(self, kf, detection):
    xyah = detection.to_xyah()
    self.last_aspect = float(xyah[2])
    _orig_update(self, kf, detection)

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