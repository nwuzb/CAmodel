#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run_3DModel.py
=================
本脚本在运行时 **猴子补丁** Deep SORT，把二维 Kalman 滤波器替换为
具有物理意义的 3D 版本 (`Physical3DKF / UKF`)。

坐标系与零点说明
------------------
1. **像素坐标 (u,v,h)**
   • 原点在左上角；u 轴→右，v 轴→下。
   • 检测框高度 *h* 用于深度估计。

2. **相机/世界坐标 (X,Y,Z)**
   • 原点：相机光心。
   • +Z 轴：沿光轴 **向远处**（离相机越远 Z 越大）。
   • +X 轴：向右，与像素 u 轴同向。
   • +Y 轴：向下，与像素 v 轴同向。（采用"相机朝前"常见的左手系）

   转换公式（针孔模型）：
   ```math
   Z = \frac{f\,H}{h},\qquad
   X = \frac{(u-c_x)\,Z}{f},\qquad
   Y = \frac{(v-c_y)\,Z}{f}
   ```

3. **速度与加速度**
   • 仅在 Z 轴上建模：`dZ < 0` 表示向相机靠近；`dZ > 0` 为远离。
   • 加速度 `a = -min(a_max, dZ^2 / (2Z)) * sign(dZ)` —— 恒减速刹车模型。

本文件、`analyze_3d_motion.py` 及 `3DModel*.md` 已统一遵循上述约定。
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
# 相机内参
# F_PIXELS    # 焦距 (像素)"像素高度 ↔ 实际距离"互换的比例尺
# cx,cy 把 3D 点映射到正确的像素位置。
F_PIXELS, CX, CY, HEAD_H = 795.0, 268.0, 137.0, 0.3 # 2k 0.6裁剪右侧 相机内参
# F_PIXELS, CX, CY, HEAD_H = 795.0, 1268.0, 137.0, 0.3 # 2k 0.6裁剪左侧 相机内参

# F_PIXELS, CX, CY, HEAD_H = 1600.3, 515.0, 153.0, 0.3 # 4k 0.6裁剪右侧 相机内参 49.69
# F_PIXELS, CX, CY, HEAD_H = 1600.3, 1789.0, 153.0, 0.3 # 4k 0.6裁剪左侧 相机内参 

# 提前占位，稍后在 VIDEO 定义后重新计算 FPS
fps = 25.0
DT = 1.0 / fps

# ---------------- 输入输出 -------------------
# FOLDER = "/Users/binzeng/MA/GT_videos/gt_60_videos/gt_60_2_clip_03_right_124" #2k 0.6裁剪右侧 相机内参
FOLDER = "/Users/binzeng/MA/GT_videos/gt_60_videos/test_gt_60_2_clip_02_right_47_test"
# FOLDER = "/Users/binzeng/MA/GT_videos/gt_60_videos/gt_60_2_clip_02_right_47"2k 0.6裁剪右侧 相机内参
# FOLDER = "/Users/binzeng/MA/GT_videos/gt_60_videos/gt_60_4_clip_02_right_100" #4k 0.6裁剪右侧 相机内参
# FOLDER = "/Users/binzeng/MA/GT_videos/gt_60_videos/gt_60_1_clip_01_right_54" #4k 0.6裁剪右侧 相机内参
# FOLDER = "/Users/binzeng/MA/GT_videos/gt_60_videos/gt_60_5_clip_03_right_32" #2k 0.6裁剪右侧 相机内参

# FOLDER = "/Users/binzeng/MA/GT_videos/gt_60_videos/gt_60_6_clip_06_left_46" #4k 0.6裁剪左侧 相机内参
# FOLDER = "/Users/binzeng/MA/GT_videos/gt_60_videos/gt_60_10_clip_01_left_53" #4k 0.6裁剪左侧 相机内参
# FOLDER = "/Users/binzeng/MA/GT_videos/gt_60_videos/gt_60_5_clip_01_left_28" #2k 0.6裁剪左侧 相机内参

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

# ---------------- 3D EKF with Acceleration ---------------------
class Physical3DKF:

    chi2inv95 = {1:3.8415,2:5.9915,3:7.8147,4:9.4877}
    def __init__(self,f=F_PIXELS,cx=CX,cy=CY,head=HEAD_H):
        print("[Physical3DKF] 初始化 3D EKF with Acceleration")
        import numpy as np; self.np=np
        self.f,self.cx,self.cy,self.H=f,cx,cy,head
        # 过程噪声重新调优: 位置 1/8, 速度 1/40
        self._std_pos=1/8
        self._std_vel=1/40
        self._std_acc=1/100  # 加速度噪声
        
        # 加速度参数
        # 最大允许减速度 (m/s²)
        self.max_decel = 1.0  # 可按实际列车制动能力调整
        
        # 前 warmup 帧完全信赖观测, 不衰减速度
        self._warmup = 1  # 约2秒, 可调 ⭐️
        
        # 速度历史用于自适应加速度估计
        self.velocity_history = []
        self.max_history = 5
        
    def initiate(self,m):
        np=self.np;u,v,a,h=m;Z=self.f*self.H/max(h,1e-6);X=(u-self.cx)*Z/self.f;Y=(v-self.cy)*Z/self.f
        # 初始速度 dZ 估成 0.5m/s (近似列车匀速)；如需更精确可按两帧差分再覆盖
        mean=np.array([u,v,a,h,Z,-13.8, X,Y])
        std=[2*self._std_pos*h]*2+[1e-2,2*self._std_pos*h]+[5]*4
        
        # 初始化速度历史
        self.velocity_history = [-0.7]
        
        # print("3d EKF with acceleration initiated")
        return mean, np.diag(np.square(std))
    
    def estimate_acceleration(self, current_velocity):
        """恒定减速度模型：直接返回固定 deceleration"""
        return self.max_decel
    
    def predict(self,mean,cov):
        np=self.np;Z,dZ,X,Y=mean[4],mean[5],mean[6],mean[7]
        
        # 🚄 动态加速度：根据"刹车距离公式 a = v^2 / (2*s)"
        #   这里 s≈Z 表示距离终点(相机)的剩余距离。
        #   限制到最大制动力 self.max_decel。
        if Z > 0.05:  # 避免除零
            accel_mag = min(self.max_decel, (dZ**2) / (2*Z))
        else:
            accel_mag = self.max_decel
        # 方向取速度反向
        acceleration = -np.sign(dZ) * accel_mag
        
        # 🚄 带加速度的运动方程
        Z2 = Z + dZ*DT + 0.5*acceleration*DT*DT  # 位置更新
        dZ2 = dZ + acceleration*DT              # 速度更新

        # -------- 约束 1: 只能向镜头靠拢 (Z 不得增大) --------
        if Z2 > Z:
            Z2 = Z            # 保持原深度
            dZ2 = min(dZ2, 0.0)  # 若速度朝远离方向，置零/保持负值
        
        # 限制速度范围（避免数值爆炸）, 保留方向符号
        dZ2 = np.clip(dZ2, -20.0, -0.1) if dZ2 < 0 else 0.0
        
        if self._warmup > 0:
            self._warmup -= 1
            
        # 重新投影到2D
        if Z2 > 0.1:  # 避免过近距离
            mean[:4]=[self.f*X/Z2+self.cx, self.f*Y/Z2+self.cy, mean[2], self.f*self.H/Z2]
        else:
            # 保持当前投影
            pass
            
        mean[4:6]=[Z2,dZ2]
        
        # 🚄 过程噪声（恒定减速度，无需额外加速度不确定性）
        pos_std=self._std_pos*mean[3]
        vel_std=self._std_vel*mean[3]
        process_noise = np.square([
            pos_std, pos_std, 1e-2, pos_std,  # u,v,a,h
            vel_std, vel_std,                # Z, dZ
            vel_std, vel_std                 # X, Y
        ])
        
        cov += np.diag(process_noise)
        # print(f"3d EKF predicted with acceleration: {acceleration:.3f} m/s²")
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

        # -------- 约束 2: 观测不得让 Z 变大 --------
        if Z > Z_prev:  # 观测让目标远离，裁剪
            Z = Z_prev
            h = self.f * self.H / Z
            mean[3] = h
            # 重新计算 3D 坐标
            X = (u - self.cx) * Z / self.f
            Y = (v - self.cy) * Z / self.f

        # 🚄 速度估计：差分 + EMA (考虑加速度)
        raw_dZ = (Z - Z_prev) / DT if DT else 0.0
        
        # 自适应平滑系数（加速度场景下需要更快适应）
        if abs(raw_dZ - mean[5]) > 0.5:  # 速度变化较大
            alpha = 0.7  # 更快适应
        else:
            alpha = 0.9  # 正常平滑
            
        mean[5] = alpha * mean[5] + (1 - alpha) * raw_dZ

        mean[4] = Z
        mean[6] = X
        mean[7] = Y
        # print("3d EKF with acceleration updated")
        return mean,cov
        
    def gating_distance(self,mean,cov,meas,only_position=False):
        np=self.np;proj,P=self.project(mean,cov)
        if only_position:proj,P,meas=proj[:2],P[:2,:2],meas[:,:2]
        z=np.linalg.solve(np.linalg.cholesky(P),(meas-proj).T).T;return np.sum(z*z,1)

# ==================== UKF 版本 ====================

try:
    import numpy as _np
except ImportError:
    import numpy as _np


class Physical3DUKF:
    """Unscented Kalman Filter 版本，接口保持与 deep_sort.kalman_filter 一致"""

    chi2inv95 = {1:3.8415,2:5.9915,3:7.8147,4:9.4877}

    # --- UKF 参数 ---
    _alpha, _beta, _kappa = 1e-1, 2.0, 0.0

    def __init__(self, f=F_PIXELS, cx=CX, cy=CY, head=HEAD_H):
        self.f, self.cx, self.cy, self.H = f, cx, cy, head

        # 维度
        self.n = 8          # 状态维度
        self.m = 4          # 观测维度

        # 缩放参数
        lam = self._alpha ** 2 * (self.n + self._kappa) - self.n
        self.Wm = _np.full(2 * self.n + 1, 1.0 / (2*(self.n + lam)))
        self.Wc = self.Wm.copy()
        self.Wm[0] = lam / (self.n + lam)
        self.Wc[0] = self.Wm[0] + (1 - self._alpha ** 2 + self._beta)
        self.c = _np.sqrt(self.n + lam)

        # 过程 / 测量噪声
        self.Q = _np.diag([1e-2,1e-2,1e-3,1e-2, 0.1,0.1, 0.05,0.05])
        self.R = _np.diag([10.,10.,0.1,10.])

    # ----------------- sigma points -----------------
    def _sigma_points(self, mean, cov):
        A = self.c * _np.linalg.cholesky(cov)
        sp = [mean]
        for i in range(self.n):
            sp.append(mean + A[:, i])
            sp.append(mean - A[:, i])
        return _np.stack(sp)

    # ----------------- motion model f(x) -----------------
    def _fx(self, x):
        u, v, a, h, Z, dZ, X, Y = x
        # 恒定减速度方向与速度反向
        acc = -0.7 * _np.sign(dZ if dZ != 0 else -1)  # 恒定减速度 0.7 m/s² (方向与速度相反)
        Z2 = Z + dZ * DT + 0.5 * acc * DT * DT
        dZ2 = dZ + acc * DT

        # -------- 约束: 只能向镜头靠拢 (Z 不得增大) --------
        if Z2 > Z:
            Z2 = Z
            dZ2 = min(dZ2, 0.0)

        # clip speed
        dZ2 = _np.clip(dZ2, -20.0, -0.1) if dZ2 < 0 else 0.0

        u2 = self.f * X / Z2 + self.cx
        v2 = self.f * Y / Z2 + self.cy
        h2 = self.f * self.H / Z2
        return _np.array([u2, v2, a, h2, Z2, dZ2, X, Y])

    # ----------------- measurement model h(x) -----------------
    def _hx(self, x):
        return x[:4]

    # ----------------- EKF 接口 -----------------
    def initiate(self, meas):
        u,v,a,h = meas
        Z = self.f * self.H / max(h,1e-6)
        X = (u - self.cx) * Z / self.f
        Y = (v - self.cy) * Z / self.f
        mean = _np.array([u,v,a,h,Z,-13.8,X,Y])
        std  = _np.array([h/4,h/4,1e-2,h/4, 2.0,1.0, 1.0,1.0])
        cov  = _np.diag(std ** 2)
        return mean, cov

    def predict(self, mean, cov):
        scale = (mean[4] / 20.0)**2          # Z_ref 可取 20 m
        self.Q[:4,:4] *= scale                # u,v,h 部分
        self.R[:2,:2] *= scale                # u,v 测量噪声
        sigmas = self._sigma_points(mean, cov)
        sigmas_f = _np.array([self._fx(s) for s in sigmas])
        mean_pred = _np.sum(self.Wm[:,None] * sigmas_f, axis=0)
        diff = sigmas_f - mean_pred
        cov_pred = diff.T @ (_np.diag(self.Wc) @ diff) + self.Q
        return mean_pred, cov_pred

    def project(self, mean, cov):
        sigmas = self._sigma_points(mean, cov)
        sigmas_h = sigmas[:, :4]   # h(x) is linear first 4 components
        z_pred = _np.sum(self.Wm[:,None] * sigmas_h, axis=0)
        diff_z = sigmas_h - z_pred
        S = diff_z.T @ (_np.diag(self.Wc) @ diff_z) + self.R
        return z_pred, S

    def update(self, mean, cov, meas):
        sigmas = self._sigma_points(mean, cov)
        sigmas_f = _np.array([self._fx(s) for s in sigmas])
        mean_pred = _np.sum(self.Wm[:,None] * sigmas_f, axis=0)
        diff_x = sigmas_f - mean_pred

        # measurement prediction
        sigmas_h = sigmas_f[:, :4]
        z_pred = _np.sum(self.Wm[:,None] * sigmas_h, axis=0)
        diff_z = sigmas_h - z_pred

        S = diff_z.T @ (_np.diag(self.Wc) @ diff_z) + self.R
        Pxz = diff_x.T @ (_np.diag(self.Wc) @ diff_z)
        K = Pxz @ _np.linalg.inv(S)

        mean_new = mean_pred + K @ (meas - z_pred)
        cov_new = cov - K @ S @ K.T
        return mean_new, cov_new

    def gating_distance(self, mean, cov, meas, only_position=False):
        z_pred, S = self.project(mean, cov)
        if only_position:
            z_pred = z_pred[:2]; S = S[:2,:2]; meas = meas[:,:2]
        diff = meas - z_pred
        # 使用马氏距离: 先做 Cholesky 分解以保持数值稳定
        z = _np.linalg.solve(_np.linalg.cholesky(S), diff.T).T
        return _np.sum(z * z, axis=1)

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

def uvh_to_xyz(u, v, h, f=795, cx=268, cy=137, H=0.3):
    Z = f * H / h
    X = (u - cx) * Z / f
    Y = (v - cy) * Z / f
    return X, Y, Z

def xyz_to_uvh(X, Y, Z, f=795, cx=268, cy=137, H=0.3):
    u = f * X / Z + cx
    v = f * Y / Z + cy
    h = f * H / Z
    return u, v, h

if __name__=="__main__":
    mod.main() 