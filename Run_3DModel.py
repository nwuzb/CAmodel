#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run_3DModel.py
=================
位于 CAmodel/deep_sort/ 目录下，以确保同级可以直接 import 待用模块。
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

def _manual_import(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules[name] = mod
    return mod

# 手动导入内部 kalman_filter 并注册到 shallow 名称 (先导入，方便覆盖类)
_manual_import("deep_sort.kalman_filter", INNER_PKG_DIR / "kalman_filter.py")

# 覆盖 KalmanFilter 和 AccelerationKalmanFilter 为 3D 版本后，再导入 tracker
_manual_import("deep_sort.tracker", INNER_PKG_DIR / "tracker.py")
# 确保 tracker 内部引用指向新的 kalman_filter 模块
sys.modules["deep_sort.tracker"].kalman_filter = sys.modules["deep_sort.kalman_filter"]

# ---------------- 摄像机参数 ----------------
print("\n>>> 已将 DeepSORT KalmanFilter 替换为基于针孔相机的 Physical3DKF (3D 运动模型) <<<\n")
F_PIXELS, CX, CY, HEAD_H = 1344.0, 286.0, 137.0, 0.30

# 提前占位，稍后在 VIDEO 定义后重新计算 FPS
fps = 25.0
DT = 1.0 / fps

# ---------------- 输入输出 -------------------
# FOLDER = "/Users/binzeng/MA/GT_videos/gt_60_videos/gt_60_2_clip_02_right_47"
FOLDER = "/Users/binzeng/MA/GT_videos/gt_60_videos/test_gt_60_2_clip_02_right_47_test"

# 在文件夹中自动查找唯一的视频文件
_video_exts = (".mp4")
video_candidates = [f for f in os.listdir(FOLDER) if f.lower().endswith(_video_exts)]
if len(video_candidates) == 0:
    raise FileNotFoundError(f"在 {FOLDER} 中未找到视频文件 {_video_exts}")
elif len(video_candidates) > 1:
    raise RuntimeError(f"在 {FOLDER} 中发现多个视频文件: {video_candidates}，请确保仅保留一个视频文件")
VIDEO = os.path.join(FOLDER, video_candidates[0])

GT = os.path.join(FOLDER, "gt.txt")
OUT_DIR = FOLDER + "-results-3DModel"
if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

# --- 重新读取 FPS，覆盖默认值 ---
cap = cv2.VideoCapture(VIDEO)
fps_read = cap.get(cv2.CAP_PROP_FPS)
cap.release()
if fps_read and fps_read > 1:
    fps = fps_read
    DT = 1.0 / fps
print(f"[3DModel] Detected FPS={fps:.2f}, Δt={DT:.4f}s")

# ---------------- 3D EKF ---------------------
class Physical3DKF:
    chi2inv95 = {1:3.8415,2:5.9915,3:7.8147,4:9.4877}
    def __init__(self,f=F_PIXELS,cx=CX,cy=CY,head=HEAD_H):
        print("[Physical3DKF] 初始化 3D EKF")
        import numpy as np; self.np=np
        self.f,self.cx,self.cy,self.H=f,cx,cy,head
        # 过程噪声重新调优: 位置 1/8, 速度 1/40
        self._std_pos=1/8
        self._std_vel=1/40
        # 前 warmup 帧完全信赖观测, 不衰减速度
        self._warmup = 1  # 约2秒, 可调 ⭐️
    def initiate(self,m):
        np=self.np;u,v,a,h=m;Z=self.f*self.H/max(h,1e-6);X=(u-self.cx)*Z/self.f;Y=(v-self.cy)*Z/self.f
        # 初始速度 dZ 估成 0.5m/s (近似列车匀速)；如需更精确可按两帧差分再覆盖
        mean=np.array([u,v,a,h,Z,0.5, X,Y])
        std=[2*self._std_pos*h]*2+[1e-2,2*self._std_pos*h]+[5]*4
        # print("3d EKF initiated")
        return mean, np.diag(np.square(std))
    
    def predict(self,mean,cov):
        np=self.np;Z,dZ,X,Y=mean[4],mean[5],mean[6],mean[7]
        # 根据 warmup 决定阻尼
        damping = 1.0 if self._warmup > 0 else 0.9
        Z2 = Z + dZ*DT
        dZ *= damping
        if self._warmup > 0:
            self._warmup -= 1
        mean[:4]=[self.f*X/Z2+self.cx, self.f*Y/Z2+self.cy, mean[2], self.f*self.H/Z2]
        mean[4:6]=[Z2,dZ]
        pos_std=self._std_pos*mean[3]
        vel_std=self._std_vel*mean[3]
        cov += np.diag(np.square([pos_std,pos_std,1e-2,pos_std,5,vel_std,vel_std,vel_std]))
        # print("3d EKF predicted")
        return mean,cov
    
    def project(self,mean,cov):
        np=self.np;proj=mean[:4];std=[self._std_pos*mean[3]]*2+[1e-1,self._std_pos*mean[3]]
        return proj, cov[:4,:4]+np.diag(np.square(std))
    def update(self,mean,cov,meas):
        np=self.np;proj,P=self.project(mean,cov)
        # 预测前的深度用于计算速度
        Z_prev = mean[4]
        K = cov[:,:4] @ np.linalg.inv(P)
        mean += K @ (meas - proj)
        cov  -= K @ P @ K.T

        # 重新投影得到新的 3D 位置
        u,v,h = mean[0], mean[1], mean[3]
        Z     = self.f * self.H / max(h,1e-6)
        X     = (u - self.cx) * Z / self.f
        Y     = (v - self.cy) * Z / self.f

        # 速度估计：差分 + EMA
        raw_dZ = (Z - Z_prev) / DT if DT else 0.0
        alpha  = 0.9  # 平滑系数，可调 ⭐️
        mean[5] = alpha * mean[5] + (1 - alpha) * raw_dZ

        mean[4] = Z
        mean[6] = X
        mean[7] = Y
        # print("3d EKF updated")
        return mean,cov
    def gating_distance(self,mean,cov,meas,only_position=False):
        np=self.np;proj,P=self.project(mean,cov)
        if only_position:proj,P,meas=proj[:2],P[:2,:2],meas[:,:2]
        z=np.linalg.solve(np.linalg.cholesky(P),(meas-proj).T).T;return np.sum(z*z,1)
    

# -------- 注入 --------
kf_mod = sys.modules.get("deep_sort.kalman_filter")
if kf_mod is None:
    kf_mod = _manual_import("deep_sort.kalman_filter", INNER_PKG_DIR / "kalman_filter.py")

kf_mod.KalmanFilter = Physical3DKF
kf_mod.AccelerationKalmanFilter = Physical3DKF

# ---------------- 运行原脚本 ----------------
orig = SCRIPT_DIR/"评估追踪器性能yolo.py"
spec=importlib.util.spec_from_file_location("eval3d",orig)
mod=importlib.util.module_from_spec(spec)
# execute module to load functions
spec.loader.exec_module(mod)
# now override configuration variables
mod.FOLDER_PATH,mod.VIDEO_PATH,mod.GT_PATH,mod.OUTPUT_DIR=FOLDER,VIDEO,GT,OUT_DIR
mod.TRACKING_PARAMS.update({
    'use_acceleration': True,  # 使用 AccelerationTracker 但其内部KF已被替换
})

if __name__=="__main__":
    mod.main() 