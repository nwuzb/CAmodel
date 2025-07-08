import numpy as np

class Physical3DKF5D:
    """5 维状态向量的物理 KalmanFilter。

    状态 x = [u, v, h, Z, dZ]^T
        u, v   : 像素坐标系下检测框中心 (px)
        h      : 检测框高度 (px)
        Z      : 头顶深度 (m)
        dZ     : 深度方向速度 (m/s, 负值表示靠近相机)

    观测 z = [u, v, h]^T
    接口与 deep_sort.kalman_filter.KalmanFilter 基本一致，便于替换。
    """

    # 卡方检验阈值 (95%)，供 Deep SORT 进行 gating
    chi2inv95 = {1: 3.8415, 2: 5.9915, 3: 7.8147}

    def __init__(self, f: float = 795.0, cx: float = 268.0, cy: float = 137.0,
                 head: float = 0.3, dt: float = 1.0 / 30,
                 max_decel: float = 1.0):
        """参数
        f, cx, cy : 针孔摄像机内参 (像素)
        head      : 行人头部实际高度 H (m)
        dt        : 两帧时间间隔 Δt (s)
        max_decel : 深度方向最大减速度 │a_max│ (m/s²)
        """
        self.f, self.cx, self.cy, self.H = f, cx, cy, head
        self.DT = float(dt)
        self.max_decel = float(max_decel)

        # 过程噪声系数（放宽以增加 gating 容忍）
        self._std_pos = 1 / 2   # 进一步放宽位置噪声
        self._std_vel = 1 / 15  # 放宽速度噪声

        # 记录最近一次观测到的宽高比 a（默认为 0.5）
        self._a_last = 0.5

        # EMA 速度平滑控制
        self._warmup = 1  # 前几帧完全依赖观测速度

    # --------------------------- helpers ---------------------------
    def _uvh_to_xyz(self, u, v, h):
        """像素 → 相机坐标系 (X,Y,Z)。"""
        Z = self.f * self.H / max(h, 1e-6)
        X = (u - self.cx) * Z / self.f
        Y = (v - self.cy) * Z / self.f
        return X, Y, Z

    def _xyz_to_uvh(self, X, Y, Z):
        """相机坐标 → 像素坐标."""
        u = self.f * X / Z + self.cx
        v = self.f * Y / Z + self.cy
        h = self.f * self.H / Z
        return u, v, h

    # ------------------------- KF interface ------------------------
    def initiate(self, meas):
        """创建新轨迹的初始状态。

        参数
        ------
        meas : ndarray
            (u, v, a, h) or (u, v, h)。若含 a(宽高比) 将被忽略。
        返回
        ------
        mean, cov : ndarray
        """
        # 兼容 3 或 4 维输入
        if len(meas) == 4:
            u, v, a, h = meas
            self._a_last = float(a)
        else:
            u, v, h = meas
            a = self._a_last
        _, _, Z = self._uvh_to_xyz(u, v, h)
        dZ_init = -0.7  # 经验初速度 (向相机)

        mean = np.array([u, v, h, Z, dZ_init], dtype=float)

        # 协方差：像素位置、h 用较小方差；Z、dZ 用稍大方差
        std_u_v = 2 * self._std_pos * h
        std_h = 2 * self._std_pos * h
        std_Z = 5.0  # 米
        std_dZ = 5.0  # m/s

        std = np.array([std_u_v, std_u_v, std_h, std_Z, std_dZ], dtype=float)
        cov = np.diag(std ** 2)

        return mean, cov

    def predict(self, mean, cov):
        """状态预测."""
        # 解包
        u, v, h, Z, dZ = mean
        # 根据上一步 u,v,h,Z 推得 X,Y
        X, Y, _ = self._uvh_to_xyz(u, v, h)

        # 计算自适应减速度
        if Z > 0.05:  # 避免除零
            accel_mag = min(self.max_decel, (dZ ** 2) / (2 * Z))
        else:
            accel_mag = self.max_decel
        acceleration = -np.sign(dZ) * accel_mag

        # 深度 & 速度更新
        Z2 = Z + dZ * self.DT + 0.5 * acceleration * self.DT ** 2
        dZ2 = dZ + acceleration * self.DT

        # 约束 1: 不允许远离相机 (Z 变大)
        if Z2 > Z:
            Z2 = Z
            dZ2 = min(dZ2, 0.0)

        # 约束 2: 速度范围
        dZ2 = np.clip(dZ2, -20.0, -0.1) if dZ2 < 0 else 0.0

        # 重新投影得到新的 u,v,h（允许横向随透视微调）
        u2, v2, h2 = self._xyz_to_uvh(X, Y, Z2)

        mean[:5] = [u2, v2, h2, Z2, dZ2]

        # 过程噪声（与当前高 h2 成线性关系）
        pos_std = self._std_pos * h2
        vel_std = self._std_vel * h2
        process_noise = np.square([
            pos_std,    # u
            pos_std,    # v
            pos_std,    # h
            vel_std,    # Z
            vel_std,    # dZ
        ])
        cov += np.diag(process_noise)

        # warmup 计数递减
        if self._warmup > 0:
            self._warmup -= 1

        return mean, cov

    def project(self, mean, cov):
        """投影到测量空间 (u,v,a,h)，其中 a 取最近观测值."""
        u, v, h = mean[:3]
        a = self._a_last
        pos_std = self._std_pos * h
        R = np.diag([pos_std ** 2, pos_std ** 2, (0.05) ** 2, pos_std ** 2])

        # 构造 4×5 投影矩阵 H
        H = np.zeros((4,5))
        H[0,0] = 1
        H[1,1] = 1
        H[3,2] = 1  # h from state h
        # a 不在状态，直接补

        proj = np.array([u, v, a, h], dtype=float)
        # 协方差先取前 3 维对应，再手动加 a 方差
        S = np.zeros((4,4))
        S[:2,:2] = cov[:2,:2] + np.eye(2)*pos_std**2
        S[3,3] = cov[2,2] + pos_std**2
        S[2,2] = (0.05)**2
        return proj, S

    # -------- 内部: 仅 (u,v,h) 投影, 用于 Kalman update --------
    def _project3(self, mean, cov):
        u, v, h = mean[:3]
        pos_std = self._std_pos * h
        R = np.diag([pos_std ** 2, pos_std ** 2, pos_std ** 2])
        proj = np.array([u, v, h], dtype=float)
        S = cov[:3, :3] + R
        return proj, S

    def update(self, mean, cov, meas):
        """卡尔曼更新。

        meas 可以是 (u,v,a,h) 或 (u,v,h)。这里统一取 u,v,h 三维进行更新。
        """
        # 处理测量维度
        if len(meas) == 4:
            u_m, v_m, a_m, h_m = meas
            meas3 = np.array([u_m, v_m, h_m])
            self._a_last = float(a_m)
        else:
            u_m, v_m, h_m = meas
            meas3 = meas

        proj, S = self._project3(mean, cov)
        K = cov[:, :3] @ np.linalg.inv(S)  # (5,3)

        innovation = meas3 - proj
        mean = mean + K @ innovation
        cov = cov - K @ S @ K.T

        # 根据新的 u,v,h，重新计算 Z 与速度
        u, v, h = mean[:3]
        Z_prev = mean[3]
        _, _, Z_new = self._uvh_to_xyz(u, v, h)

        # 不允许 Z 增大
        if Z_new > Z_prev:
            Z_new = Z_prev  # 保持原深度
            # 同步 h
            h = self.f * self.H / Z_new
            mean[2] = h

        # 更新速度 with EMA
        raw_dZ = (Z_new - Z_prev) / self.DT if self.DT else 0.0
        if abs(raw_dZ - mean[4]) > 0.5:
            alpha = 0.7
        else:
            alpha = 0.9
        mean[4] = alpha * mean[4] + (1 - alpha) * raw_dZ
        mean[3] = Z_new  # 更新深度

        return mean, cov

    def gating_distance(self, mean, cov, meas, only_position=False):
        """Mahalanobis distance for gating.

        meas: ndarray (...,4) or (...,3)
        若为 4 维则忽略第 3 维(a)。
        """
        proj, S = self.project(mean, cov)

        # 构造匹配维度 2 (u,v) 或 3 (u,v,h)
        if only_position:
            proj_use = proj[:2]
            S_use = S[:2, :2]
            meas_use = meas[:, :2] if meas.ndim == 2 else meas[:2]
        else:
            # 使用 4 维完全一致的 (u,v,a,h)
            if meas.ndim == 1:
                meas_use = meas if len(meas) == 4 else np.insert(meas, 2, self._a_last)
            else:
                if meas.shape[1] == 4:
                    meas_use = meas
                else:
                    a_col = np.full((meas.shape[0],1), self._a_last)
                    meas_use = np.concatenate([meas[:, :2], a_col, meas[:,2:3]], axis=1)
            # 拼接 proj 的 a
            proj_use = np.insert(proj[:3], 2, self._a_last)
            S_use = np.zeros((4,4))
            S_use[:2,:2] = S[:2,:2]
            # 对 a 给一个较小方差
            S_use[2,2] = (0.05)**2
            S_use[3,3] = S[2,2]
            # 近似忽略协方差项

        chol = np.linalg.cholesky(S_use)
        if meas_use.ndim == 1:
            d = np.linalg.solve(chol, (meas_use - proj_use).T)
            return np.sum(d * d)
        else:
            d = np.linalg.solve(chol, (meas_use - proj_use).T).T
            return np.sum(d * d, axis=1) 