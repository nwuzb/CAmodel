# vim: expandtab:ts=4:sw=4
from __future__ import division, print_function, absolute_import

import argparse
import os

import cv2
import numpy as np

from application_util import preprocessing
from application_util import visualization
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker, AccelerationTracker


def gather_sequence_info(sequence_dir, detection_file):
    """Gather sequence information, such as image filenames, detections,
    groundtruth (if available).

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_file : str
        Path to the detection file.

    Returns
    -------
    Dict
        A dictionary of the following sequence information:

        * sequence_name: Name of the sequence
        * image_filenames: A dictionary that maps frame indices to image
          filenames.
        * detections: A numpy array of detections in MOTChallenge format.
        * groundtruth: A numpy array of ground truth in MOTChallenge format.
        * image_size: Image size (height, width).
        * min_frame_idx: Index of the first frame.
        * max_frame_idx: Index of the last frame.

    """
    image_dir = os.path.join(sequence_dir, "img1")
    image_filenames = {
        int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
        for f in os.listdir(image_dir)}
    groundtruth_file = os.path.join(sequence_dir, "gt/gt.txt")

    detections = None
    if detection_file is not None:
        detections = np.load(detection_file)
    groundtruth = None
    if os.path.exists(groundtruth_file):
        groundtruth = np.loadtxt(groundtruth_file, delimiter=',')

    if len(image_filenames) > 0:
        image = cv2.imread(next(iter(image_filenames.values())),
                           cv2.IMREAD_GRAYSCALE)
        image_size = image.shape
    else:
        image_size = None

    if len(image_filenames) > 0:
        min_frame_idx = min(image_filenames.keys())
        max_frame_idx = max(image_filenames.keys())
    else:
        min_frame_idx = int(detections[:, 0].min())
        max_frame_idx = int(detections[:, 0].max())

    info_filename = os.path.join(sequence_dir, "seqinfo.ini")
    if os.path.exists(info_filename):
        with open(info_filename, "r") as f:
            line_splits = [l.split('=') for l in f.read().splitlines()[1:]]
            info_dict = dict(
                s for s in line_splits if isinstance(s, list) and len(s) == 2)

        update_ms = 1000 / int(info_dict["frameRate"])
    else:
        update_ms = None

    feature_dim = detections.shape[1] - 10 if detections is not None else 0
    seq_info = {
        "sequence_name": os.path.basename(sequence_dir),
        "image_filenames": image_filenames,
        "detections": detections,
        "groundtruth": groundtruth,
        "image_size": image_size,
        "min_frame_idx": min_frame_idx,
        "max_frame_idx": max_frame_idx,
        "feature_dim": feature_dim,
        "update_ms": update_ms
    }
    return seq_info


def create_detections(detection_mat, frame_idx, min_height=0):
    """Create detections for given frame index from the raw detection matrix.

    Parameters
    ----------
    detection_mat : ndarray
        Matrix of detections. The first 10 columns of the detection matrix are
        in the standard MOTChallenge detection format. In the remaining columns
        store the feature vector associated with each detection.
    frame_idx : int
        The frame index.
    min_height : Optional[int]
        A minimum detection bounding box height. Detections that are smaller
        than this value are disregarded.

    Returns
    -------
    List[tracker.Detection]
        Returns detection responses at given frame index.

    """
    frame_indices = detection_mat[:, 0].astype(np.int64)
    mask = frame_indices == frame_idx

    detection_list = []
    for row in detection_mat[mask]:
        bbox, confidence, feature = row[2:6], row[6], row[10:]
        if bbox[3] < min_height:
            continue
        detection_list.append(Detection(bbox, confidence, feature))
    return detection_list


def run(sequence_dir, detection_file, output_file, min_confidence,
        nms_max_overlap, min_detection_height, max_cosine_distance,
        nn_budget, display, use_acceleration=False, max_age=30, n_init=3, max_iou_distance=0.7,
        std_weight_position=1.0/20, std_weight_velocity=1.0/160, std_weight_acceleration=1.0/10,
        velocity_smooth_factor=0.85, acceleration_smooth_factor=0.7,
        debug_video_path=None):
    """Run multi-target tracker on a particular sequence.

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_file : str
        Path to the detections file.
    output_file : str
        Path to the tracking output file. This file will contain the tracking
        results on completion.
    min_confidence : float
        Detection confidence threshold. Disregard all detections that have
        a confidence lower than this value.
    nms_max_overlap: float
        Maximum detection overlap (non-maximum suppression threshold).
    min_detection_height : int
        Detection height threshold. Disregard all detections that have
        a height lower than this value.
    max_cosine_distance : float
        Gating threshold for cosine distance metric (object appearance).
    nn_budget : Optional[int]
        Maximum size of the appearance descriptor gallery. If None, no budget
        is enforced.
    display : bool
        If True, show visualization of intermediate tracking results.
    use_acceleration : bool
        If True, use Kalman filter with acceleration model.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed.
    max_iou_distance : float
        Maximum IOU distance for matching by IOU.
    std_weight_position : float
        卡尔曼滤波器位置过程噪声权重
    std_weight_velocity : float
        卡尔曼滤波器速度过程噪声权重
    std_weight_acceleration : float
        卡尔曼滤波器加速度过程噪声权重
    velocity_smooth_factor : float
        速度平滑因子
    acceleration_smooth_factor : float
        加速度平滑因子
    debug_video_path : str
        Path to save debug video. If None, no video is saved.
    """
    seq_info = gather_sequence_info(sequence_dir, detection_file)
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget)
    
    # 根据参数选择使用哪种跟踪器
    if use_acceleration:
        # 获取图像尺寸用于加速度跟踪器
        image_size = seq_info["image_size"]
        image_width = image_size[1] if image_size and len(image_size) >= 2 else 1920
        image_height = image_size[0] if image_size and len(image_size) >= 2 else 1080
        frame_size = (image_width, image_height)
        
        # 创建加速度跟踪器
        tracker = AccelerationTracker(
            metric, 
            max_iou_distance=max_iou_distance,
            max_age=max_age, 
            n_init=n_init,
            frame_size=frame_size
        )
        
        # 配置卡尔曼滤波器参数
        if hasattr(tracker.kf, '_std_weight_position'):
            tracker.kf._std_weight_position = std_weight_position
            tracker.kf._std_weight_velocity = std_weight_velocity
            tracker.kf._std_weight_acceleration = std_weight_acceleration
            
            # 如果存在平滑因子参数，也进行配置
            if hasattr(tracker.kf, '_velocity_smooth_factor'):
                tracker.kf._velocity_smooth_factor = velocity_smooth_factor
                tracker.kf._acceleration_smooth_factor = acceleration_smooth_factor
                
            # print(f"已配置卡尔曼滤波器参数:")
            # print(f"  位置噪声权重: {tracker.kf._std_weight_position}")
            # print(f"  速度噪声权重: {tracker.kf._std_weight_velocity}")
            # print(f"  加速度噪声权重: {tracker.kf._std_weight_acceleration}")
            # print(f"  速度平滑因子: {tracker.kf._velocity_smooth_factor if hasattr(tracker.kf, '_velocity_smooth_factor') else 'N/A'}")
            # print(f"  加速度平滑因子: {tracker.kf._acceleration_smooth_factor if hasattr(tracker.kf, '_acceleration_smooth_factor') else 'N/A'}")

        # 如果外部通过 kwargs 传入了方向 gating 相关配置，则覆写
        tracker.lateral_threshold = globals().get('DIRECTION_LATERAL_THRESHOLD', tracker.lateral_threshold)
        tracker.backward_tol = globals().get('DIRECTION_BACKWARD_TOL', tracker.backward_tol)
        tracker.direction_min_age = globals().get('DIRECTION_MIN_AGE', tracker.direction_min_age)
        tracker.debug_direction = globals().get('DEBUG_DIRECTION', tracker.debug_direction)
    else:
        # 创建普通跟踪器
        tracker = Tracker(metric, max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init)

    results = []

    def frame_callback(vis, frame_idx):
        if frame_idx % 250 == 0: # 每200帧打印一次
            print("Processing frame %05d" % frame_idx)

        # Load image and generate detections.
        detections = create_detections(
            seq_info["detections"], frame_idx, min_detection_height)
        detections = [d for d in detections if d.confidence >= min_confidence]

        # Run non-maximum suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(
            boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Update tracker.
        tracker.predict()
        tracker.update(detections)

        # Update visualization.
        if display or debug_video_path is not None:
            image = cv2.imread(
                seq_info["image_filenames"][frame_idx], cv2.IMREAD_COLOR)
            vis.set_image(image.copy())
            vis.draw_detections(detections)
            vis.draw_trackers(tracker.tracks)

        # Store results.
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlwh()
            results.append([
                frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])

    # Run tracker.
    if display:
        visualizer = visualization.Visualization(seq_info, update_ms=5)
        # 如需保存调试视频
        if debug_video_path is not None:
            # 将窗口形状调整为原始分辨率，确保保存视频分辨率一致
            import numpy as _np
            orig_shape = seq_info["image_size"][::-1]  # (width,height)
            visualizer.viewer._window_shape = orig_shape
            visualizer.viewer.image = _np.zeros(orig_shape + (3,), dtype=_np.uint8)
            visualizer.viewer.enable_videowriter(debug_video_path, fourcc_string="mp4v", fps=2)
    else:
        # 即使不显示窗口，如果需要保存debug视频，也要创建带视频保存功能的可视化器
        if debug_video_path is not None:
            visualizer = visualization.VideoOnlyVisualization(seq_info, debug_video_path)
        else:
            visualizer = visualization.NoVisualization(seq_info)
    visualizer.run(frame_callback)

    # Store results.
    f = open(output_file, 'w')
    for row in results:
        print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
            row[0], row[1], row[2], row[3], row[4], row[5]),file=f)

    # 关闭 videowriter
    if debug_video_path is not None:
        if hasattr(visualizer, 'viewer') and visualizer.viewer is not None:
            visualizer.viewer.disable_videowriter()
        elif hasattr(visualizer, 'close_video'):
            visualizer.close_video()


def bool_string(input_string):
    if input_string not in {"True","False"}:
        raise ValueError("Please Enter a valid Ture/False choice")
    else:
        return (input_string == "True")

def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Deep SORT")
    parser.add_argument(
        "--sequence_dir", help="Path to MOTChallenge sequence directory",
        default=None, required=True)
    parser.add_argument(
        "--detection_file", help="Path to custom detections.", default=None,
        required=True)
    parser.add_argument(
        "--output_file", help="Path to the tracking output file. This file will"
        " contain the tracking results on completion.",
        default="/tmp/hypotheses.txt")
    parser.add_argument(
        "--min_confidence", help="Detection confidence threshold. Disregard "
        "all detections that have a confidence lower than this value.",
        default=0.8, type=float)
    parser.add_argument(
        "--min_detection_height", help="Threshold on the detection bounding "
        "box height. Detections with height smaller than this value are "
        "disregarded", default=0, type=int)
    parser.add_argument(
        "--nms_max_overlap",  help="Non-maximum suppression threshold: Maximum "
        "detection overlap.", default=1.0, type=float)
    parser.add_argument(
        "--max_cosine_distance", help="Gating threshold for cosine distance "
        "metric (object appearance).", type=float, default=0.2)
    parser.add_argument(
        "--nn_budget", help="Maximum size of the appearance descriptors "
        "gallery. If None, no budget is enforced.", type=int, default=None)
    parser.add_argument(
        "--display", help="Show intermediate tracking results",
        default=True, type=bool_string)
    parser.add_argument(
        "--use_acceleration", help="Use Kalman filter with acceleration model",
        default=False, type=bool_string)
    parser.add_argument(
        "--max_age", help="Maximum number of missed misses before a track is deleted",
        default=30, type=int)
    parser.add_argument(
        "--n_init", help="Number of consecutive detections before the track is confirmed",
        default=3, type=int)
    parser.add_argument(
        "--max_iou_distance", help="Maximum IOU distance for matching by IOU",
        default=0.7, type=float)
    parser.add_argument(
        "--std_weight_position", help="卡尔曼滤波器位置过程噪声权重",
        default=1.0/20, type=float)
    parser.add_argument(
        "--std_weight_velocity", help="卡尔曼滤波器速度过程噪声权重",
        default=1.0/160, type=float)
    parser.add_argument(
        "--std_weight_acceleration", help="卡尔曼滤波器加速度过程噪声权重",
        default=1.0/10, type=float)
    parser.add_argument(
        "--velocity_smooth_factor", help="速度平滑因子",
        default=0.85, type=float)
    parser.add_argument(
        "--acceleration_smooth_factor", help="加速度平滑因子",
        default=0.7, type=float)
    parser.add_argument(
        "--debug_video_path", help="Path to save debug video",
        default=None, type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        args.sequence_dir, args.detection_file, args.output_file,
        args.min_confidence, args.nms_max_overlap, args.min_detection_height,
        args.max_cosine_distance, args.nn_budget, args.display, args.use_acceleration,
        args.max_age, args.n_init, args.max_iou_distance,
        args.std_weight_position, args.std_weight_velocity, args.std_weight_acceleration,
        args.velocity_smooth_factor, args.acceleration_smooth_factor,
        args.debug_video_path)
