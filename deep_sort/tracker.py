# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track, TrackState


class Tracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """
    #在Tracker类的初始化中，n_init参数被设置为3：
    #新检测需要连续3帧都成功匹配才会被标记为"Confirmed"
    #如果在这3帧内任何一帧没有匹配成功，轨迹会被标记为"Deleted"
    def __init__(self, metric, max_iou_distance=0.7, max_age=30, n_init=2):   #在Tracker类的初始化中，n_init参数被默认设置为3
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        self.kf = kalman_filter.KalmanFilter()
        self.tracks = []
        self._next_id = 1

    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)

        # Update track set.
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(
                self.kf, detections[detection_idx])
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)

    def _match(self, detections):

        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices)

            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # Associate confirmed tracks using appearance features.
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks)

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        """创建新轨迹"""
        mean, covariance = self.kf.initiate(detection.to_xyah())
        
        # 针对边缘场景特殊处理：检查新检测是否位于边缘区域
        bbox = detection.to_tlbr()
        x1, y1, x2, y2 = map(int, bbox)
        
        # 边缘区域创建的轨迹使用较短的生存周期
        max_age = self.max_age
        
        # 检查是否在边缘区域
        strict_boundary_margin = self.boundary_margin * 0.7
        is_edge_detection = (x1 < strict_boundary_margin or x2 > self.frame_width - strict_boundary_margin or
                            y1 < strict_boundary_margin or y2 > self.frame_height - strict_boundary_margin)
                            
        # 边缘区域的新轨迹生存期较短，避免过度保留
        if is_edge_detection:
            max_age = min(max_age, 5)  # 边缘区域最大生存期较短
        
        self.tracks.append(Track(
            mean, covariance, self._next_id, self.n_init, max_age,
            detection.feature))
        self._next_id += 1


class AccelerationTracker(Tracker):
    """
    Multi-target tracker that uses a Kalman filter with acceleration.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.AccelerationKalmanFilter
        A Kalman filter with acceleration to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.
    """

    def __init__(self, metric, max_iou_distance=0.7, max_age=30, n_init=3, frame_size=(1536, 864)):
        """Initialize tracker with a custom Kalman filter that includes acceleration."""
        # 不直接调用父类的__init__方法，而是复制其内容并修改
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        # 使用加速度卡尔曼滤波器
        self.kf = kalman_filter.AccelerationKalmanFilter()
        self.tracks = []
        self._next_id = 1
        
        # 用于跨帧信息融合
        self.track_history = {}  # 存储每个轨迹的历史状态
        self.track_features = {}  # 存储每个轨迹的历史特征
        self.max_history_length = 10  # 最多保存10帧历史
        
        # 边界检测参数
        self.frame_width, self.frame_height = frame_size
        self.boundary_margin = 60  # 增加边界检测余量
        self.leaving_tracks = set()  # 保存正在离开画面的轨迹ID
        
        # 添加跨帧匹配质量跟踪
        self.match_quality = {}  # 跟踪ID -> 匹配质量历史
        self.match_quality_threshold = 0.8  # 匹配质量阈值
        
        # 轨迹消失状态跟踪 - 更快速地处理离开画面的轨迹
        self.disappearing_tracks = {}  # 轨迹ID -> 消失计数
        self.max_disappear_count = 3  # 消失多次后降低max_age

    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.kf)
            
            # 检查轨迹是否触及图像边缘
            if self._check_at_boundary(track):
                # 触及边缘的轨迹直接标记为删除状态
                track.state = TrackState.Deleted  # 直接设置状态为删除，而不是调用不存在的方法
                # 记录已删除的轨迹ID
                self.leaving_tracks.add(track.track_id)
                
    def _check_at_boundary(self, track):
        """检查轨迹是否触及图像边缘"""
        # 仅检查已确认的轨迹
        if not track.is_confirmed():
            return False
            
        # 获取边界框
        bbox = track.to_tlbr()
        x1, y1, x2, y2 = map(int, bbox)
        
        # 边界余量
        margin = 20
        
        # 检查是否触及或超出图像边缘
        if (x1 <= margin or x2 >= self.frame_width - margin or
            y1 <= margin or y2 >= self.frame_height - margin):
            return True
            
        return False
        
    def update(self, detections):
        """Perform measurement update and track management."""
        # 执行匹配
        matches, unmatched_tracks, unmatched_detections = self._match(detections)
        
        # 过滤掉已经标记为删除的轨迹的匹配
        filtered_matches = []
        for track_idx, detection_idx in matches:
            if self.tracks[track_idx].track_id not in self.leaving_tracks:
                filtered_matches.append((track_idx, detection_idx))
            else:
                # 将已删除轨迹对应的检测添加到未匹配检测中
                unmatched_detections.append(detection_idx)
        
        # 使用过滤后的匹配
        matches = filtered_matches
        
        # 更新轨迹
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(
                self.kf, detections[detection_idx])
                
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
            
        for detection_idx in unmatched_detections:
            # 不为已删除的轨迹创建新轨迹
            self._initiate_track(detections[detection_idx])
            
        # 清理已删除的轨迹
        self.tracks = [t for t in self.tracks if not t.is_deleted()]
        
        # 清理已删除轨迹的特征历史
        active_track_ids = [t.track_id for t in self.tracks]
        for track_id in list(self.track_history.keys()):
            if track_id not in active_track_ids:
                if track_id in self.track_history:
                    del self.track_history[track_id]
                if track_id in self.track_features:
                    del self.track_features[track_id]
                if track_id in self.match_quality:
                    del self.match_quality[track_id]
                if track_id in self.disappearing_tracks:
                    del self.disappearing_tracks[track_id]
        
        # 更新距离度量
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)

    def _match(self, detections):

        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices)

            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # Associate confirmed tracks using appearance features.
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks)

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        """创建新轨迹"""
        mean, covariance = self.kf.initiate(detection.to_xyah())
        
        # 针对边缘场景特殊处理：检查新检测是否位于边缘区域
        bbox = detection.to_tlbr()
        x1, y1, x2, y2 = map(int, bbox)
        
        # 边缘区域创建的轨迹使用较短的生存周期
        max_age = self.max_age
        
        # 检查是否在边缘区域
        strict_boundary_margin = self.boundary_margin * 0.7
        is_edge_detection = (x1 < strict_boundary_margin or x2 > self.frame_width - strict_boundary_margin or
                            y1 < strict_boundary_margin or y2 > self.frame_height - strict_boundary_margin)
                            
        # 边缘区域的新轨迹生存期较短，避免过度保留
        if is_edge_detection:
            max_age = min(max_age, 5)  # 边缘区域最大生存期较短
        
        self.tracks.append(Track(
            mean, covariance, self._next_id, self.n_init, max_age,
            detection.feature))
        self._next_id += 1
