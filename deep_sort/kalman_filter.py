"""
加速度卡尔曼滤波器 (AccelerationKalmanFilter) 说明
===================================================

相比标准卡尔曼滤波器的优势：
1. 能够处理加速度状态，使得跟踪更加准确，尤其是对于加速和减速目标
2. 对于高机动性目标（如快速变向的行人）跟踪性能更好
3. 能够利用速度测量信息，进一步提高跟踪精度
4. 能够更好地处理目标暂时遮挡和离开视野情况

如何调优参数：
- self._std_weight_position: 位置过程噪声权重
- self._std_weight_velocity: 速度过程噪声权重
- self._std_weight_acceleration: 加速度过程噪声权重
  较小的值会使跟踪更平滑但响应更慢，较大的值会使跟踪更敏感但可能更不稳定

- self._velocity_smooth_factor: 速度平滑因子
- self._acceleration_smooth_factor: 加速度平滑因子
  控制状态更新时的平滑程度，较大的值会更相信历史数据，较小的值会更相信测量

- 对于缺帧处理，使用conservative_predict方法，提供了更快的状态衰减
- 对于边界条件，增加了更精细的处理以减少ID切换和错误匹配

根据应用场景选择合适的值，通常通过参数校准.py脚本进行调优验证。
"""

# vim: expandtab:ts=4:sw=4
import numpy as np
import scipy.linalg


"""
Table for the 0.95 quantile of the chi-square distribution with N degrees of
freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
function and used as Mahalanobis gating threshold.
"""
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919}


class KalmanFilter(object):
    """
    A simple Kalman filter for tracking bounding boxes in image space.

    The 8-dimensional state space

        x, y, a, h, vx, vy, va, vh

    contains the bounding box center position (x, y), aspect ratio a, height h,
    and their respective velocities.

    Object motion follows a constant velocity model. The bounding box location
    (x, y, a, h) is taken as direct observation of the state space (linear
    observation model).

    """

    def __init__(self):
        ndim, dt = 4, 1.

        # Create Kalman filter model matrices.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

    def initiate(self, measurement):
        """Create track from unassociated measurement.

        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y, a, h) with center position (x, y),
            aspect ratio a, and height h.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector (8 dimensional) and covariance matrix (8x8
            dimensional) of the new track. Unobserved velocities are initialized
            to 0 mean.

        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        """Run Kalman filter prediction step.

        Parameters
        ----------
        mean : ndarray
            The 8 dimensional mean vector of the object state at the previous
            time step.
        covariance : ndarray
            The 8x8 dimensional covariance matrix of the object state at the
            previous time step.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state.

        """
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance

    def project(self, mean, covariance):
        """Project state distribution to measurement space.

        Parameters
        ----------
        mean : ndarray
            The state's mean vector (8 dimensional array).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).

        Returns
        -------
        (ndarray, ndarray)
            Returns the projected mean and covariance matrix of the given state
            estimate.

        """
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def update(self, mean, covariance, measurement):
        """Run Kalman filter correction step.

        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (8 dimensional).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        measurement : ndarray
            The 4 dimensional measurement vector (x, y, a, h), where (x, y)
            is the center position, a the aspect ratio, and h the height of the
            bounding box.

        Returns
        -------
        (ndarray, ndarray)
            Returns the measurement-corrected state distribution.

        """
        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
            check_finite=False).T
        innovation = measurement - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements,
                        only_position=False):
        """Compute gating distance between state distribution and measurements.

        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 4 degrees of
        freedom, otherwise 2.

        Parameters
        ----------
        mean : ndarray
            Mean vector over the state distribution (8 dimensional).
        covariance : ndarray
            Covariance of the state distribution (8x8 dimensional).
        measurements : ndarray
            An Nx4 dimensional matrix of N measurements, each in
            format (x, y, a, h) where (x, y) is the bounding box center
            position, a the aspect ratio, and h the height.
        only_position : Optional[bool]
            If True, distance computation is done with respect to the bounding
            box center position only.

        Returns
        -------
        ndarray
            Returns an array of length N, where the i-th element contains the
            squared Mahalanobis distance between (mean, covariance) and
            `measurements[i]`.

        """
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        cholesky_factor = np.linalg.cholesky(covariance)
        d = measurements - mean
        z = scipy.linalg.solve_triangular(
            cholesky_factor, d.T, lower=True, check_finite=False,
            overwrite_b=True)
        squared_maha = np.sum(z * z, axis=0)
        return squared_maha


class AccelerationKalmanFilter(KalmanFilter):
    """
    带有加速度的卡尔曼滤波器，用于处理具有加速度变化的跟踪场景。
    在状态向量中添加了加速度分量。
    
    状态向量结构: [x, y, a, h, vx, vy, va, vh, ax, ay, aa, ah]
    其中 (x,y) 是中心坐标，a 是宽高比，h 是高度
    vx, vy, va, vh 是相应的速度
    ax, ay, aa, ah 是相应的加速度
    """
    
    def __init__(self):
        """初始化带加速度的卡尔曼滤波器，添加加速度状态量"""
        ndim, dt = 4, 1.
        
        # 扩展了状态向量，包含位置、速度和加速度
        # x = [cx, cy, a, h, vx, vy, va, vh, ax, ay, aa, ah]^T
        self._motion_mat = np.eye(3 * ndim, 3 * ndim)
        
        # 更新位置 = 位置 + 速度*dt + 0.5*加速度*dt^2
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
            self._motion_mat[i, 2*ndim + i] = 0.5 * dt**2
        
        # 更新速度 = 速度 + 加速度*dt
        for i in range(ndim):
            self._motion_mat[ndim + i, 2*ndim + i] = dt
            
        # 加速度保持不变
        
        # 过程噪声协方差矩阵
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160
        self._std_weight_acceleration = 1. / 10
        
        # 平滑系数 - 控制预测的平滑程度
        self._velocity_smooth_factor = 0.85
        self._acceleration_smooth_factor = 0.7
        
        # 测量矩阵: 只测量位置 [x, y, a, h]
        self._update_mat = np.eye(ndim, 3 * ndim)

    def initiate(self, measurement):
        """从边界框创建跟踪状态

        Parameters
        ----------
        measurement : ndarray
            边界框坐标 (cx, cy, a, h) 其中 a 是宽高比，h 是高度.

        Returns
        -------
        (ndarray, ndarray)
            状态向量和协方差矩阵的均值。初始状态向量中的速度和加速度为0。
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean_acc = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel, mean_acc]

        std = [
            2 * self._std_weight_position * measurement[3],   # 位置标准差
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],  # 速度标准差
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_acceleration * measurement[3],  # 加速度标准差
            10 * self._std_weight_acceleration * measurement[3],
            1e-5,
            10 * self._std_weight_acceleration * measurement[3]
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        """执行卡尔曼预测步骤。

        Parameters
        ----------
        mean : ndarray
            上一时刻的状态向量均值。
        covariance : ndarray
            上一时刻的状态协方差矩阵。

        Returns
        -------
        (ndarray, ndarray)
            预测后的状态向量和协方差矩阵。
        """
        # 应用运动模型来更新均值向量
        predicted_mean = np.dot(self._motion_mat, mean)
        
        # 平滑速度和加速度
        # 应用平滑因子，避免速度和加速度预测过度波动
        predicted_mean[4:8] = predicted_mean[4:8] * self._velocity_smooth_factor + mean[4:8] * (1 - self._velocity_smooth_factor)
        predicted_mean[8:12] = predicted_mean[8:12] * self._acceleration_smooth_factor + mean[8:12] * (1 - self._acceleration_smooth_factor)
        
        # 为计算过程噪声协方差矩阵设置参数
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]
        ]
        std_acc = [
            self._std_weight_acceleration * mean[3],
            self._std_weight_acceleration * mean[3],
            1e-5,
            self._std_weight_acceleration * mean[3]
        ]
        
        # 创建过程噪声协方差矩阵
        motion_cov = np.diag(np.concatenate([np.square(std_pos), np.square(std_vel), np.square(std_acc)]))
        
        # 应用动力学模型和添加过程噪声
        predicted_covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return predicted_mean, predicted_covariance

    def project(self, mean, covariance):
        """将状态分布投影到测量空间。

        Parameters
        ----------
        mean : ndarray
            状态向量均值。
        covariance : ndarray
            状态协方差矩阵。

        Returns
        -------
        (ndarray, ndarray)
            投影后的均值和协方差矩阵。
        """
        # 应用测量矩阵
        projected_mean = np.dot(self._update_mat, mean)
        
        # 为测量噪声协方差设置参数
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]
        ]
        
        # 创建测量噪声协方差矩阵
        innovation_cov = np.diag(np.square(std))
        
        # 计算投影协方差
        projected_cov = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
        
        # 添加测量噪声
        projected_cov += innovation_cov

        return projected_mean, projected_cov

    def update(self, mean, covariance, measurement):
        """执行卡尔曼更新步骤。

        Parameters
        ----------
        mean : ndarray
            预测的状态向量均值。
        covariance : ndarray
            预测的状态协方差矩阵。
        measurement : ndarray
            边界框坐标 (cx, cy, a, h).

        Returns
        -------
        (ndarray, ndarray)
            更新后的状态向量和协方差矩阵。
        """
        # 投影状态到测量空间
        projected_mean, projected_cov = self.project(mean, covariance)

        # 计算卡尔曼增益
        kalman_gain = np.linalg.multi_dot((
            covariance, self._update_mat.T, np.linalg.inv(projected_cov)))
        
        # 计算新息向量
        innovation = measurement - projected_mean
        
        # 更新状态估计
        new_mean = mean + np.dot(kalman_gain, innovation)
        
        # 更新状态协方差
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))

        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements,
                      only_position=False):
        """计算测量值与投影状态间的闸门距离。

        Parameters
        ----------
        mean : ndarray
            状态向量均值。
        covariance : ndarray
            状态协方差矩阵。
        measurements : ndarray
            一组测量值。
        only_position : Optional[bool]
            如果为True，则仅使用位置部分，忽略尺寸信息。

        Returns
        -------
        ndarray
            返回一个长度为测量值数量的数组，包含每个测量值的马氏距离平方。
        """
        # 投影状态到测量空间
        mean, covariance = self.project(mean, covariance)
        
        # 如果仅考虑位置部分（忽略尺寸），则提取相关分量
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        # 计算马氏距离
        d = measurements - mean
        
        if only_position:  # 只使用2x2矩阵
            cholesky_factor = np.linalg.cholesky(covariance)
            z = np.linalg.solve(cholesky_factor, d.T).T
            squared_maha = np.sum(z * z, axis=1)
        else:
            # 完整测量空间的马氏距离
            cholesky_factor = np.linalg.cholesky(covariance)
            z = np.linalg.solve(cholesky_factor, d.T).T
            squared_maha = np.sum(z * z, axis=1)

        return squared_maha
        
    def smooth_prediction(self, mean, covariance, history):
        """使用历史轨迹进行平滑预测
        
        Parameters
        ----------
        mean : ndarray
            当前估计的状态向量均值
        covariance : ndarray
            当前估计的状态协方差矩阵
        history : list
            过去几帧的状态向量
            
        Returns
        -------
        (ndarray, ndarray)
            平滑预测后的状态向量和协方差矩阵
        """
        if len(history) < 2:
            # 如果历史不足，使用默认预测
            return self.predict(mean, covariance)
        
        # 根据历史状态计算平均速度和加速度
        latest_pos = history[-1][:4]  # 最近的位置
        second_latest_pos = history[-2][:4]  # 第二近的位置
        
        # 计算最近的速度
        current_vel = (latest_pos - second_latest_pos)
        
        # 如果历史足够长，计算加速度
        if len(history) >= 3:
            third_latest_pos = history[-3][:4]
            prev_vel = (second_latest_pos - third_latest_pos)
            current_acc = (current_vel - prev_vel)
        else:
            # 没有足够历史，使用现有加速度
            current_acc = mean[8:12]
        
        # 创建平滑状态向量
        smooth_mean = mean.copy()
        
        # 更新速度 - 在当前速度和历史计算的速度之间取加权平均
        alpha_v = 0.7  # 速度平滑系数
        smooth_mean[4:8] = alpha_v * mean[4:8] + (1 - alpha_v) * current_vel
        
        # 更新加速度 - 平滑处理
        alpha_a = 0.5  # 加速度平滑系数
        smooth_mean[8:12] = alpha_a * mean[8:12] + (1 - alpha_a) * current_acc
        
        # 基于平滑后的速度和加速度预测下一个位置
        dt = 1.0  # 时间步长
        # 位置 = 位置 + 速度*dt + 0.5*加速度*dt^2
        smooth_mean[:4] = latest_pos + smooth_mean[4:8] * dt + 0.5 * smooth_mean[8:12] * dt * dt
        
        # 协方差矩阵保持不变
        smooth_covariance = covariance.copy()
        
        return smooth_mean, smooth_covariance
    
    def conservative_predict(self, mean, covariance, time_since_update=1):
        """在丢失检测时使用的保守预测
        
        参数
        ----------
        mean : ndarray
            上一时刻的状态向量均值
        covariance : ndarray
            上一时刻的状态协方差矩阵
        time_since_update : int
            自上次成功更新以来的帧数
            
        返回
        -------
        (ndarray, ndarray)
            保守预测的状态向量和协方差矩阵
        """
        # 创建一个逐渐衰减的时间系数，使得速度和加速度随时间衰减
        time_factor = np.exp(-0.25 * time_since_update)
        
        # 应用运动模型
        predicted_mean = mean.copy()
        
        # 位置更新 - 只使用速度更新，忽略加速度影响
        dt = 1.0  # 时间步长
        predicted_mean[:4] += predicted_mean[4:8] * dt * time_factor
        
        # 速度随时间衰减 - 目标在没有检测时应该减速
        predicted_mean[4:8] *= time_factor
        
        # 加速度比速度衰减更快
        predicted_mean[8:12] *= time_factor * 0.7
        
        # 增加协方差 - 不确定性随时间增加
        # 为计算过程噪声协方差矩阵设置参数，比正常预测增加不确定性
        uncertainty_factor = 1.0 + 0.5 * time_since_update
        
        std_pos = [
            self._std_weight_position * mean[3] * uncertainty_factor,
            self._std_weight_position * mean[3] * uncertainty_factor,
            1e-2 * uncertainty_factor,
            self._std_weight_position * mean[3] * uncertainty_factor
        ]
        std_vel = [
            self._std_weight_velocity * mean[3] * uncertainty_factor,
            self._std_weight_velocity * mean[3] * uncertainty_factor,
            1e-5 * uncertainty_factor,
            self._std_weight_velocity * mean[3] * uncertainty_factor
        ]
        std_acc = [
            self._std_weight_acceleration * mean[3] * uncertainty_factor,
            self._std_weight_acceleration * mean[3] * uncertainty_factor,
            1e-5 * uncertainty_factor,
            self._std_weight_acceleration * mean[3] * uncertainty_factor
        ]
        
        # 创建保守的过程噪声协方差矩阵
        motion_cov = np.diag(np.concatenate([np.square(std_pos), np.square(std_vel), np.square(std_acc)]))
        
        # 更新协方差矩阵
        predicted_covariance = covariance.copy() + motion_cov
        
        return predicted_mean, predicted_covariance
