# vim: expandtab:ts=4:sw=4
import numpy as np
import colorsys
from .image_viewer import ImageViewer
import cv2
from deep_sort.tracker import DEFAULT_LATERAL_THRESHOLD as _DEF_LAT


def create_unique_color_float(tag, hue_step=0.41):
    """Create a unique RGB color code for a given track id (tag).

    The color code is generated in HSV color space by moving along the
    hue angle and gradually changing the saturation.

    Parameters
    ----------
    tag : int
        The unique target identifying tag.
    hue_step : float
        Difference between two neighboring color codes in HSV space (more
        specifically, the distance in hue channel).

    Returns
    -------
    (float, float, float)
        RGB color code in range [0, 1]

    """
    h, v = (tag * hue_step) % 1, 1. - (int(tag * hue_step) % 4) / 5.
    r, g, b = colorsys.hsv_to_rgb(h, 1., v)
    return r, g, b


def create_unique_color_uchar(tag, hue_step=0.41):
    """Create a unique RGB color code for a given track id (tag).

    The color code is generated in HSV color space by moving along the
    hue angle and gradually changing the saturation.

    Parameters
    ----------
    tag : int
        The unique target identifying tag.
    hue_step : float
        Difference between two neighboring color codes in HSV space (more
        specifically, the distance in hue channel).

    Returns
    -------
    (int, int, int)
        RGB color code in range [0, 255]

    """
    r, g, b = create_unique_color_float(tag, hue_step)
    return int(255*r), int(255*g), int(255*b)


class NoVisualization(object):
    """
    A dummy visualization object that loops through all frames in a given
    sequence to update the tracker without performing any visualization.
    """

    def __init__(self, seq_info):
        self.frame_idx = seq_info["min_frame_idx"]
        self.last_idx = seq_info["max_frame_idx"]

    def set_image(self, image):
        pass

    def draw_groundtruth(self, track_ids, boxes):
        pass

    def draw_detections(self, detections):
        pass

    def draw_trackers(self, trackers):
        pass

    def run(self, frame_callback):
        while self.frame_idx <= self.last_idx:
            frame_callback(self, self.frame_idx)
            self.frame_idx += 1


class VideoOnlyVisualization(object):
    """
    A visualization object that saves video without displaying window.
    Only performs video recording of tracking results.
    """

    def __init__(self, seq_info, video_path):
        self.frame_idx = seq_info["min_frame_idx"]
        self.last_idx = seq_info["max_frame_idx"]
        self.video_path = video_path
        self.image_size = seq_info["image_size"]
        
        # 初始化视频写入器
        self.video_writer = None
        self.current_image = None
        
        # 确保输出目录存在
        import os
        output_dir = os.path.dirname(video_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def set_image(self, image):
        """设置当前帧图像"""
        self.current_image = image.copy()
        
        # 初始化视频写入器（在第一帧时）
        if self.video_writer is None:
            height, width = image.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(self.video_path, fourcc, 2.0, (width, height))

    def draw_groundtruth(self, track_ids, boxes):
        """绘制真值标注"""
        if self.current_image is None:
            return
        
        for track_id, box in zip(track_ids, boxes):
            color = create_unique_color_uchar(track_id)
            # 绘制边界框
            box = box.astype(np.int64)
            cv2.rectangle(self.current_image, (box[0], box[1]), 
                         (box[0] + box[2], box[1] + box[3]), color, 2)
            # 绘制标签
            cv2.putText(self.current_image, str(track_id), 
                       (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def draw_detections(self, detections):
        """绘制检测结果"""
        if self.current_image is None:
            return
            
        for detection in detections:
            bbox = detection.tlwh.astype(np.int64)
            cv2.rectangle(self.current_image, (bbox[0], bbox[1]), 
                         (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 0, 255), 2)

    PRED_VIS_WINDOW = 5  # 连续预测≤5帧仍可显示

    def draw_trackers(self, tracks):
        """绘制跟踪结果"""
        if self.current_image is None:
            return
            
        for track in tracks:
            # 忽略未确认或过久未更新的轨迹
            if not track.is_confirmed() or track.time_since_update > self.PRED_VIS_WINDOW:
                continue
            
            bbox = track.to_tlwh().astype(np.int64)
            
            # 如果是预测帧，使用橙色并加Pred标签
            if track.time_since_update > 0:
                color = (0, 165, 255)  # BGR 橙色
                label_text = f"Pred {track.track_id}"
            else:
                color = create_unique_color_uchar(track.track_id)
                label_text = str(track.track_id)
            
            # 绘制边界框
            cv2.rectangle(self.current_image, (bbox[0], bbox[1]), 
                         (bbox[0] + bbox[2], bbox[1] + bbox[3]), color, 2)
            
            # 绘制标签
            cv2.putText(self.current_image, label_text, 
                       (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # =========== 方向 ROI 调试绘制 ===========
            if hasattr(track, 'direction') and track.direction is not None:
                dir_vec = track.direction
                if np.linalg.norm(dir_vec) < 1e-3:
                    continue
                    
                # 方向线起点: 当前中心
                tlwh = track.to_tlwh()
                cx = int(tlwh[0] + tlwh[2] / 2)
                cy = int(tlwh[1] + tlwh[3] / 2)

                # 绘制中心到前方射线 (长度 100px)
                length = 100
                end_pt = (int(cx + dir_vec[0] * length), int(cy + dir_vec[1] * length))
                cv2.line(self.current_image, (cx, cy), end_pt, color, 1)

                # 绘制平行带状 ROI 边界 (± lateral_threshold)
                lateral_th = getattr(track, 'lateral_threshold', _DEF_LAT)
                # 计算垂直方向向量
                perp = np.array([-dir_vec[1], dir_vec[0]])
                perp = perp / (np.linalg.norm(perp) + 1e-6)

                pt1 = (int(cx + perp[0]*lateral_th), int(cy + perp[1]*lateral_th))
                pt2 = (int(end_pt[0] + perp[0]*lateral_th), int(end_pt[1] + perp[1]*lateral_th))
                pt3 = (int(cx - perp[0]*lateral_th), int(cy - perp[1]*lateral_th))
                pt4 = (int(end_pt[0] - perp[0]*lateral_th), int(end_pt[1] - perp[1]*lateral_th))

                # 画 ROI 边界两条平行线
                cv2.line(self.current_image, pt1, pt2, color, 1)
                cv2.line(self.current_image, pt3, pt4, color, 1)

    def run(self, frame_callback):
        """运行可视化循环"""
        while self.frame_idx <= self.last_idx:
            frame_callback(self, self.frame_idx)
            
            # 写入当前帧到视频
            if self.video_writer is not None and self.current_image is not None:
                self.video_writer.write(self.current_image)
            
            self.frame_idx += 1

    def close_video(self):
        """关闭视频写入器"""
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            # print(f"Debug视频已保存到: {self.video_path}")


class Visualization(object):
    """
    This class shows tracking output in an OpenCV image viewer.
    """

    def __init__(self, seq_info, update_ms):
        image_shape = seq_info["image_size"][::-1]
        aspect_ratio = float(image_shape[1]) / image_shape[0]
        image_shape = 1024, int(aspect_ratio * 1024)
        self.viewer = ImageViewer(
            update_ms, image_shape, "Figure %s" % seq_info["sequence_name"])
        self.viewer.thickness = 2
        self.frame_idx = seq_info["min_frame_idx"]
        self.last_idx = seq_info["max_frame_idx"]

    def run(self, frame_callback):
        self.viewer.run(lambda: self._update_fun(frame_callback))

    def _update_fun(self, frame_callback):
        if self.frame_idx > self.last_idx:
            return False  # Terminate
        frame_callback(self, self.frame_idx)
        self.frame_idx += 1
        return True

    def set_image(self, image):
        self.viewer.image = image

    def draw_groundtruth(self, track_ids, boxes):
        self.viewer.thickness = 2
        for track_id, box in zip(track_ids, boxes):
            self.viewer.color = create_unique_color_uchar(track_id)
            self.viewer.rectangle(*box.astype(np.int64), label=str(track_id))

    def draw_detections(self, detections):
        self.viewer.thickness = 2
        self.viewer.color = 0, 0, 255
        for i, detection in enumerate(detections):
            self.viewer.rectangle(*detection.tlwh)

    PRED_VIS_WINDOW = 5  # 连续预测≤5帧仍可显示

    def draw_trackers(self, tracks):
        self.viewer.thickness = 2
        for track in tracks:
            # 忽略未确认或过久未更新的轨迹
            if not track.is_confirmed() or track.time_since_update > self.PRED_VIS_WINDOW:
                continue
            tlwh_arr = track.to_tlwh().astype(np.int64)
            # 如果是预测帧（time_since_update>0），使用橙色并加Pred标签
            if track.time_since_update > 0:
                self.viewer.color = (0,165,255)  # BGR 橙色
                label_text = f"Pred {track.track_id}"
            else:
                label_text = str(track.track_id)

            self.viewer.rectangle(*tlwh_arr, label=label_text)

            # =========== 方向 ROI 调试绘制 ===========
            if hasattr(track, 'direction') and track.direction is not None:
                dir_vec = track.direction
                if np.linalg.norm(dir_vec) < 1e-3:
                    continue
                # 方向线起点: 当前中心
                tlwh = track.to_tlwh()
                cx = int(tlwh[0] + tlwh[2] / 2)
                cy = int(tlwh[1] + tlwh[3] / 2)

                # 绘制中心到前方射线 (长度 100px)
                length = 100
                end_pt = (int(cx + dir_vec[0] * length), int(cy + dir_vec[1] * length))
                cv2.line(self.viewer.image, (cx, cy), end_pt, self.viewer._color, 1)

                # 绘制平行带状 ROI 边界 (± lateral_threshold)
                lateral_th = getattr(track, 'lateral_threshold', _DEF_LAT)
                # 计算垂直方向向量
                perp = np.array([-dir_vec[1], dir_vec[0]])
                perp = perp / (np.linalg.norm(perp) + 1e-6)

                pt1 = (int(cx + perp[0]*lateral_th), int(cy + perp[1]*lateral_th))
                pt2 = (int(end_pt[0] + perp[0]*lateral_th), int(end_pt[1] + perp[1]*lateral_th))
                pt3 = (int(cx - perp[0]*lateral_th), int(cy - perp[1]*lateral_th))
                pt4 = (int(end_pt[0] - perp[0]*lateral_th), int(end_pt[1] - perp[1]*lateral_th))

                # 画 ROI 边界两条平行线
                cv2.line(self.viewer.image, pt1, pt2, self.viewer._color, 1)
                cv2.line(self.viewer.image, pt3, pt4, self.viewer._color, 1)
