import numpy as np

class Physical3DKF_H5D:
    """基于针孔模型的 5 维 EKF

    状态 x = [u, v, Z, dZ, H]^T
        u, v : 检测框中心像素 (px)
        Z    : 头顶深度 (m)，朝相机方向为正
        dZ   : 深度方向速度 (m/s)，负值靠近相机
        H    : 头部实际高度 (m) —— 动态估计目标身高

    观测 z = [u, v, h_px]^T
        h_px : 头部在像素坐标系中的高度，以检测框 h 近似

    由于 h_px = f * H / Z 为非线性，因此采用 EKF 更新。
    """

    chi2inv95 = {1:3.8415,2:5.9915,3:7.8147}

    def __init__(self, f:float=795.0, cx:float=268.0, cy:float=137.0,
                 H0:float=0.30, dt:float=1/30, max_decel:float=1.0):
        self.f, self.cx, self.cy = f, cx, cy
        self.DT = float(dt)
        self.max_decel = max_decel
        # 过程噪声权重
        self._std_uv = 1.0   # 像素
        self._std_Z  = 5.0   # m
        self._std_dZ = 5.0   # m/s
        self._std_H  = 0.05  # m （身高在短时内基本不变）
        # 观测噪声（像素）
        self._r_uv = 2.0
        self._r_h  = 3.0
        # 初始头高先验
        self.H0 = H0

        # --- 额外: 图像平面横向速度估计 ---
        self._u_vel = 0.0
        self._v_vel = 0.0
        self._vel_alpha = 0.7  # EMA 系数

    # ---------------- helper ----------------
    def _xyz_to_uv(self, X:float, Y:float, Z:float):
        u = self.f * X / Z + self.cx
        v = self.f * Y / Z + self.cy
        return u, v

    # ---------------- KF API ----------------
    def initiate(self, meas):
        """meas: (u,v,h) 或 (u,v,a,h)，a 被忽略"""
        if len(meas)==4:
            u,v,_,h_px = meas
        else:
            u,v,h_px = meas
        Z_init = self.f * self.H0 / max(h_px,1e-6)
        dZ_init = -0.7
        H_init = self.H0
        mean = np.array([u,v,Z_init,dZ_init,H_init], dtype=float)
        std = np.array([
            self._std_uv,
            self._std_uv,
            self._std_Z,
            self._std_dZ,
            self._std_H
        ], dtype=float)
        cov = np.diag(std**2)

        # 初始化速度为 0
        self._u_vel = 0.0
        self._v_vel = 0.0
        return mean, cov

    def predict(self, mean, cov):
        # 解包
        u,v,Z,dZ,H = mean
        # 深度方向匀减速模型（与 old KF 保持一致）
        if Z>0.05:
            accel_mag = min(self.max_decel, (dZ**2)/(2*Z))
        else:
            accel_mag = self.max_decel
        a_Z = -np.sign(dZ)*accel_mag
        Z2 = Z + dZ*self.DT + 0.5*a_Z*self.DT**2
        dZ2 = dZ + a_Z*self.DT
        if Z2>Z:  # 不允许远离相机
            Z2, dZ2 = Z, min(dZ2,0.0)
        dZ2 = np.clip(dZ2,-20.0,-0.1) if dZ2<0 else 0.0
        # --- 像素平面匀速模型 ---
        u2 = u + self._u_vel * self.DT
        v2 = v + self._v_vel * self.DT

        mean[:5] = [u2, v2, Z2, dZ2, H]
        # 过程噪声
        Q = np.diag([
            self._std_uv**2,
            self._std_uv**2,
            (self._std_Z)**2,
            (self._std_dZ)**2,
            (self._std_H)**2
        ])
        cov += Q
        return mean, cov

    def _project(self, mean, cov):
        u,v,Z,_,H = mean
        h_px = self.f * H / Z
        z_pred = np.array([u,v,h_px], dtype=float)
        # 雅可比 H_jac (3x5)
        H_jac = np.zeros((3,5))
        H_jac[0,0] = 1
        H_jac[1,1] = 1
        H_jac[2,2] = -self.f*H/(Z**2)
        H_jac[2,4] =  self.f/Z
        # 观测噪声
        R = np.diag([self._r_uv**2, self._r_uv**2, self._r_h**2])
        S = H_jac @ cov @ H_jac.T + R
        return z_pred, S, H_jac

    def project(self, mean, cov):
        z_pred, S, _ = self._project(mean,cov)
        return z_pred, S

    def update(self, mean, cov, meas):
        if len(meas)==4:
            u_m,v_m,_,h_m = meas
            z = np.array([u_m,v_m,h_m],dtype=float)
        else:
            u_m,v_m,h_m = meas
            z = meas
        z_pred, S, H_jac = self._project(mean,cov)
        Z_prev = float(mean[2])  # 预测深度，用于单调约束
        K = cov @ H_jac.T @ np.linalg.inv(S)  # (5x3)
        innovation = z - z_pred
        mean_prev_uv = mean[:2].copy()
        mean = mean + K @ innovation
        # --- 深度单调性约束：不允许远离相机 ---
        if mean[2] > Z_prev:
            mean[2] = Z_prev
            if mean[3] > 0:
                mean[3] = 0.0  # 截断正向速度
        cov = cov - K @ S @ K.T

        # --- 一致性修正：确保投影后像素高与观测一致 ---
        # 1) 像素位置对齐到观测，避免滞后
        mean[0] = z[0]
        mean[1] = z[1]

        # 2) 更新横向速度 EMA
        raw_du = (z[0] - mean_prev_uv[0]) / self.DT
        raw_dv = (z[1] - mean_prev_uv[1]) / self.DT
        self._u_vel = self._vel_alpha * self._u_vel + (1 - self._vel_alpha) * raw_du
        self._v_vel = self._vel_alpha * self._v_vel + (1 - self._vel_alpha) * raw_dv

        # 3) 高度一致性修正
        Z_new = mean[2]
        H_new = h_m * Z_new / self.f
        mean[4] = H_new

        return mean, cov

    def gating_distance(self, mean, cov, meas, only_position=False):
        if only_position:
            z_pred = mean[:2]
            S = cov[:2,:2] + np.diag([self._r_uv**2,self._r_uv**2])
            if meas.ndim==1:
                dz = meas[:2]-z_pred
                d = np.linalg.solve(np.linalg.cholesky(S), dz)
                return np.sum(d*d)
            else:
                dz = meas[:,:2]-z_pred
                d = np.linalg.solve(np.linalg.cholesky(S), dz.T).T
                return np.sum(d*d,axis=1)
        # full 3D gating
        z_pred,S,_ = self._project(mean,cov)
        if meas.ndim==1:
            if len(meas)==4:
                z = np.array([meas[0],meas[1],meas[3]])
            else:
                z = meas
            dz = z - z_pred
            d = np.linalg.solve(np.linalg.cholesky(S), dz)
            return np.sum(d*d)
        else:
            if meas.shape[1]==4:
                z = np.stack([meas[:,0],meas[:,1],meas[:,3]],axis=1)
            else:
                z = meas
            dz = z - z_pred
            d = np.linalg.solve(np.linalg.cholesky(S), dz.T).T
            return np.sum(d*d,axis=1) 