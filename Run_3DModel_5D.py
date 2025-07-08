#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run_3DModel_5D.py
====================
本脚本与 ``Run_3DModel.py`` 基本一致，只是把 Deep SORT 的 KalmanFilter
替换为 **5 维状态向量** 版本 `Physical3DKF5D`，方便与原 8D 模型做并行对比。
运行方法：::

    python deep_sort/Run_3DModel_5D.py

其余输入输出、摄像机参数等保持不变，可在脚本顶部手动修改。
"""

from __future__ import annotations
import os, sys, importlib, importlib.util, shutil
from pathlib import Path
import numpy as np
import cv2

SCRIPT_DIR = Path(__file__).resolve().parent       # CAmodel/deep_sort
PROJECT_ROOT = SCRIPT_DIR.parent                   # CAmodel
INNER_PKG_DIR = SCRIPT_DIR / "deep_sort"          # 真正包目录 (含 __init__.py)

# 把必要路径写入 sys.path
for p in (PROJECT_ROOT, SCRIPT_DIR, INNER_PKG_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# --- 手动导入工具函数 --------------------------------------------------------

def _manual_import(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules[name] = mod
    return mod

# 手动导入内部 kalman_filter 并注册到模块表
_manual_import("deep_sort.kalman_filter", INNER_PKG_DIR / "kalman_filter.py")

# 覆盖 KalmanFilter 之前就先导入 tracker，确保 tracker 中引用同一份 kalman_filter
_manual_import("deep_sort.tracker", INNER_PKG_DIR / "tracker.py")
# 同步引用（保险起见）
sys.modules["deep_sort.tracker"].kalman_filter = sys.modules["deep_sort.kalman_filter"]

# ---------------- 摄像机参数 ----------------
print("\n>>> 已将 DeepSORT KalmanFilter 替换为 5D Physical3DKF (物理运动模型) <<<\n")
F_PIXELS, CX, CY, HEAD_H = 795.0, 268.0, 137.0, 0.3  # 根据 2_clip_02_right 标定

# 提前占位，稍后在读取视频后覆盖 fps / DT
fps = 25.0
DT = 1.0 / fps

# ---------------- 输入输出 -------------------
FOLDER = "/Users/binzeng/MA/GT_videos/gt_60_videos/test_gt_60_2_clip_02_right_47_test"
_video_exts = (".mp4",)
video_candidates = [f for f in os.listdir(FOLDER) if f.lower().endswith(_video_exts)]
if not video_candidates:
    raise FileNotFoundError(f"在 {FOLDER} 中未找到视频文件{_video_exts}")
if len(video_candidates) > 1:
    raise RuntimeError(f"在 {FOLDER} 中发现多个视频文件: {video_candidates}")
VIDEO = os.path.join(FOLDER, video_candidates[0])

GT = os.path.join(FOLDER, "gt.txt")
OUT_DIR = FOLDER + "-results-3DModel-5D"
if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

# 读取 FPS
cap = cv2.VideoCapture(VIDEO)
fps_read = cap.get(cv2.CAP_PROP_FPS)
cap.release()
if fps_read and fps_read > 1:
    fps = fps_read
    DT = 1.0 / fps
print(f"[3DModel-5D] Detected FPS={fps:.2f}, Δt={DT:.4f}s")

# ---------------- 5D KF 类导入 ----------------
from physical3dkf_5d import Physical3DKF5D  # ⭐️ 关键：5 维 KF

# ---------------- 仍保留 8D/UKF 类（用于对照，可省略） ----------------
# 这里直接复用原脚本中定义的 Physical3DKF / Physical3DUKF，如需参考可复制。

# -------- 注入 5D KF --------
kf_mod = sys.modules.get("deep_sort.kalman_filter")
if kf_mod is None:
    kf_mod = _manual_import("deep_sort.kalman_filter", INNER_PKG_DIR / "kalman_filter.py")

kf_mod.KalmanFilter = Physical3DKF5D
kf_mod.AccelerationKalmanFilter = Physical3DKF5D

# -------- 额外猴子补丁：确保 5D KF 与 DeepSORT 完美衔接 --------
import deep_sort.linear_assignment as _la
import importlib as _ibl

# 1) 仅基于 (u,v) 做门控，避免 4 维卡方分布缺失导致的 KeyError
_orig_gate_cost = _la.gate_cost_matrix
_la.gate_cost_matrix = lambda *args, **kwargs: _orig_gate_cost(*args, **kwargs, only_position=True)

# 2) 修补 Track 的 bbox 计算逻辑（5D 状态向量无宽高比 a）
_track_mod = _ibl.import_module("deep_sort.track")

# 保存原始 update 以便复用
_orig_track_update = _track_mod.Track.update

def _to_tlwh_5d(self):
    """根据 5D KF 的状态(mean) 计算 (x,y,w,h)。
    mean = [u,v,h,Z,dZ]，此处近似采用最近一次观测到的宽高比 a；若无则默认为 0.5。
    """
    u, v, h = self.mean[:3]
    a = getattr(self, "last_aspect", 0.5)  # 宽高比 w/h
    w = a * h
    return np.array([u - w / 2, v - h / 2, w, h], dtype=float)

# tlbr 直接由 tlwh 推导
def _to_tlbr_5d(self):
    tlwh = _to_tlwh_5d(self)
    tlbr = tlwh.copy()
    tlbr[2:] += tlbr[:2]
    return tlbr

_track_mod.Track.to_tlwh = _to_tlwh_5d
_track_mod.Track.to_tlbr = _to_tlbr_5d

# 3) 在每次 update 时记录当前检测框的宽高比，供 bbox 计算使用

def _update_5d(self, kf, detection):
    # 先记录宽高比 (a) 以便后续 bbox 计算
    xyah = detection.to_xyah()
    self.last_aspect = float(xyah[2]) if len(xyah) > 2 else 0.5
    # 调用原始 update 完成状态更新等逻辑
    _orig_track_update(self, kf, detection)

_track_mod.Track.update = _update_5d

# 4) 为 5D KF 增补 4 自由度的卡方阈值，防止潜在访问错误
Physical3DKF5D.chi2inv95[4] = 9.4877

# -------- 运行原评估脚本 --------
orig = SCRIPT_DIR / "评估追踪器性能yolo.py"
spec = importlib.util.spec_from_file_location("eval5d", orig)
mod = importlib.util.module_from_spec(spec)
# execute module to load its main() and other functions
spec.loader.exec_module(mod)

# 覆盖其全局参数
mod.FOLDER_PATH, mod.VIDEO_PATH, mod.GT_PATH, mod.OUTPUT_DIR = FOLDER, VIDEO, GT, OUT_DIR
mod.TRACKING_PARAMS.update({
    'use_acceleration': True,  # 加速度追踪器，内部 KF 已指向 5D
})

if __name__ == "__main__":
    mod.main() 