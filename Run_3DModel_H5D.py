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

# 摄像机参数（可按需修改）
F_PIXELS, CX, CY = 795.0, 268.0, 137.0
DEFAULT_HEAD = 0.30

# 输入输出路径
FOLDER = "/Users/binzeng/MA/GT_videos/gt_60_videos/test_gt_60_2_clip_02_right_47_test"
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

# 注入 KF
kf_mod = sys.modules["deep_sort.kalman_filter"]
kf_mod.KalmanFilter            = Physical3DKF_H5D
kf_mod.AccelerationKalmanFilter = Physical3DKF_H5D

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

# 覆盖
_la.gate_cost_matrix = _gate_cost_matrix_3d

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