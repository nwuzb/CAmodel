#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import cv2
import subprocess
import torch
from pathlib import Path
import shutil
import tensorflow as tf
from tqdm import tqdm
import motmetrics as mm
import json
import sys
import time
import datetime

# 添加日志记录功能
class TeeLogger:
    """同时将输出写入到控制台和文件"""
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log_file = open(log_file, 'a', encoding='utf-8')
        self.count = 0
        self.write_separator()
        
    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()
        
    def write_separator(self):
        """写入分隔线和序号"""
        self.count += 1
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        separator = f"\n\n{'='*80}\n运行序号: {self.count} - 时间: {timestamp}\n{'='*80}\n\n"
        self.terminal.write(separator)
        self.log_file.write(separator)
        
    def close(self):
        self.log_file.close()

# 初始化日志记录器
def init_logger(output_dir):
    log_file = "/Users/binzeng/MA/GT_videos/评估追踪器性能yolo_run_log.txt"
    sys.stdout = TeeLogger(log_file)
    return sys.stdout

# 导入DeepSORT相关模块
from deep_sort.tracker import Tracker, AccelerationTracker
from deep_sort.nn_matching import NearestNeighborDistanceMetric
from deep_sort.detection import Detection
from tools.generate_detections import create_box_encoder
from ultralytics import YOLO

# ================ 配置参数（可修改） ================
# 输入输出文件路径
VIDEO_PATH = "/Users/binzeng/MA/EvaluateVideos/2_clip_02_rightpart_48.mp4"  # 视频文件路径
LABEL_SOURCE = "/Users/binzeng/MA/GT_videos/gt_60_5_clip_01_left_28/gt/labels.txt"  # 标签文件源路径

# 自动生成输出路径
VIDEO_NAME = os.path.splitext(os.path.basename(VIDEO_PATH))[0]  # 获取视频名称（不含扩展名）
VIDEO_DIR = os.path.dirname(VIDEO_PATH)  # 获取视频所在目录
OUTPUT_DIR = os.path.join(VIDEO_DIR, f"{VIDEO_NAME}_pregt")  # 输出主目录
DETECTION_DIR = os.path.join(OUTPUT_DIR, "detection")  # 检测结果目录
GT_DIR = os.path.join(OUTPUT_DIR, "gt")  # gt目录
OUTPUT_GT = os.path.join(GT_DIR, "gt.txt")  # gt文件路径
OUTPUT_LABEL = os.path.join(GT_DIR, "labels.txt")  # 标签文件路径，改为labels.txt
OUTPUT_ZIP = os.path.join(OUTPUT_DIR, f"{VIDEO_NAME}_pretraingt.zip")  # 压缩文件路径

# 跟踪结果过滤参数
MIN_TRACK_LEN = 3  # 最小跟踪长度（帧数）
MIN_BOX_WIDTH = 15  # 最小框宽度

# YOLO模型参数
YOLO_PARAMS = {
    'model_path': "/Users/binzeng/MA/EvaluateVideos/finetune.pt",  # YOLO模型权重路径
    'conf_thres': 0.7,           # 置信度阈值
    'iou_thres': 0.5,           # NMS IOU阈值
    'img_size': 736,             # 输入图像大小
    'device': 'mps' if torch.backends.mps.is_available() else 'cpu',  # 设备选择
}

# DeepSORT 跟踪参数
TRACKING_PARAMS = { 
    'min_confidence': 0.5,         # 检测置信度阈值
    'nms_max_overlap': 0.5,        # 非极大值抑制阈值
    'min_detection_height': 20,    # 最小检测框高度
    'max_cosine_distance': 0.4,    # 特征匹配距离阈值
    'nn_budget': 60,               # 特征库大小
    'max_age': 30,                 # 目标消失后保持跟踪的最大帧数
    'n_init': 1,                   # 确认为稳定跟踪目标所需的最小检测帧数
    'max_iou_distance': 0.95,      # 最大IoU距离
    'use_acceleration': True,      # 是否使用加速度卡尔曼滤波器
    'std_weight_position': 1.0 / 10,    # 位置过程噪声权重
    'std_weight_velocity': 1.0 / 40,    # 速度过程噪声权重
    'std_weight_acceleration': 1.0 / 100, # 加速度过程噪声权重
    'velocity_smooth_factor': 0.8,      # 速度平滑因子
    'acceleration_smooth_factor': 0.6,  # 加速度平滑因子
}

# 特征提取模型路径
MODEL_PATH = "/Users/binzeng/MA/CAmodel/head_feature_encoder_best"

# 可视化参数
VISUALIZATION_PARAMS = {
    'track_color': (0, 255, 0),        # 绿色跟踪框
    'crossed_track_color': (255, 128, 0),  # 亮蓝色过线框
    'text_color': (255, 255, 255),     # 白色文字
    'track_thickness': 2,              # 跟踪框线宽
    'remap_ids': True,                 # 是否重映射ID
    'show_original_id': False,         # 是否显示原始ID
    'line_margin': 0.2,                # 计数线位置（相对于图像宽度的比例）
}

# ===================================================

def load_yolo_model():
    """加载YOLO模型"""
    try:
        # 使用ultralytics的YOLO API加载模型
        model = YOLO(YOLO_PARAMS['model_path'])
        if torch.backends.mps.is_available():
            print("使用MPS设备进行加速")
            model.to('mps')
        else:
            print("使用CPU设备")
            model.to('cpu')
        return model
    except Exception as e:
        print(f"加载YOLO模型时出错: {e}")
        return None

def prepare_sequence_dir(video_path, gt_path, output_dir):
    """准备MOT序列格式的目录结构"""
    # 创建序列名称（基于视频文件名）
    sequence_name = Path(video_path).stem
    sequence_dir = os.path.join(output_dir, "mot_data", sequence_name)
    
    # 创建MOT格式目录结构
    os.makedirs(os.path.join(sequence_dir, "img1"), exist_ok=True)
    os.makedirs(os.path.join(sequence_dir, "det"), exist_ok=True)
    os.makedirs(os.path.join(sequence_dir, "gt"), exist_ok=True)
    
    # 复制GT文件
    gt_dest = os.path.join(sequence_dir, "gt", "gt.txt")
    shutil.copy(gt_path, gt_dest)
    print(f"复制GT文件到: {gt_dest}")
    
    # 先统计实际帧数
    cap = cv2.VideoCapture(video_path)
    actual_total_frames = 0
    while True:
        ret, _ = cap.read()
        if not ret:
            break
        actual_total_frames += 1
    cap.release()
    
    # 再重新打开视频，正式处理
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # 创建seqinfo.ini文件
    with open(os.path.join(sequence_dir, "seqinfo.ini"), "w") as f:
        f.write("[Sequence]\n")
        f.write(f"name={sequence_name}\n")
        f.write(f"imDir=img1\n")
        f.write(f"frameRate={fps}\n")
        f.write(f"seqLength={actual_total_frames}\n")
        f.write(f"imWidth={width}\n")
        f.write(f"imHeight={height}\n")
        f.write(f"imExt=.jpg\n")
    
    # 处理每一帧
    frame_count = 0
    progress_bar = tqdm(total=actual_total_frames, desc=f"提取视频 {sequence_name} 的帧")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        frame_id = frame_count  # MOT格式从1开始计数
        
        # 保存图片
        img_path = os.path.join(sequence_dir, "img1", f"{frame_id:06d}.jpg")
        cv2.imwrite(img_path, frame)
        
        progress_bar.update(1)
    
    progress_bar.close()
    cap.release()
    
    print(f"共处理 {frame_count} 帧")
    print(f"序列保存为 {sequence_dir}")
    return sequence_dir

def generate_detections_with_yolo(sequence_dir, model):
    """使用YOLO模型生成检测结果"""
    # 检测结果输出目录
    detection_output_dir = os.path.join(os.path.dirname(sequence_dir), "..", "detections")
    os.makedirs(detection_output_dir, exist_ok=True)
    
    # 获取序列名称
    sequence_name = os.path.basename(sequence_dir)
    
    # 读取图像目录
    image_dir = os.path.join(sequence_dir, "img1")
    image_filenames = {
        int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
        for f in os.listdir(image_dir) if f.endswith('.jpg')
    }
    
    # 创建检测文件
    detections_with_features = []
    
    # 创建特征提取器
    try:
        # 如果未指定模型路径，尝试自动查找
        if not MODEL_PATH:
            # 根据freeze_model.py中的信息，查找可能的模型位置
            possible_paths = [
                # 相对于当前文件的路径
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                           "resources/networks/mars-small128.pb"),
                # 相对于项目根目录的路径
                "./deep_sort/resources/networks/mars-small128.pb",
                "./resources/networks/mars-small128.pb",
                "../resources/networks/mars-small128.pb",
                # 绝对路径
                "/Users/binzeng/MA/CAmodel/deep_sort/resources/networks/mars-small128.pb",
                "/Users/binzeng/MA/CAmodel/resources/networks/mars-small128.pb",
                "/Users/binzeng/MA/EvaluateVideos/mars-small128.pb"
            ]
            
            # 遍历所有可能的路径
            model_filename = None
            for path in possible_paths:
                if os.path.exists(path):
                    model_filename = path
                    break
            
            if model_filename is None:
                print("警告: 无法找到特征提取模型文件，将使用简单的颜色直方图特征提取器")
                raise FileNotFoundError("无法找到特征提取模型文件")
        else:
            model_filename = MODEL_PATH
            if not os.path.exists(model_filename):
                raise FileNotFoundError(f"指定的模型路径不存在: {model_filename}")
        
        print(f"加载特征提取器模型: {model_filename}")
        encoder = create_box_encoder(model_filename, batch_size=32) # 创建特征提取器
        print(f"特征提取器已加载: {model_filename}")
    except Exception as e:
        print(f"创建特征提取器时出错: {e}")
        print("使用简单的颜色直方图特征提取器作为后备方案")
        
        # 创建一个简单的特征提取器（基于颜色直方图）作为后备
        def simple_encoder(image, boxes):
            features = []
            for box in boxes:
                x, y, w, h = map(int, box)
                if x < 0 or y < 0 or w <= 0 or h <= 0 or x+w > image.shape[1] or y+h > image.shape[0]:
                    features.append(np.zeros(512))
                    continue
                
                roi = image[y:y+h, x:x+w]
                if roi.size > 0:
                    # 计算颜色直方图作为特征
                    hist = cv2.calcHist([roi], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
                    hist = cv2.normalize(hist, hist).flatten()
                    # 将特征扩展到512维
                    feature = np.zeros(512)
                    feature[:min(len(hist), 512)] = hist[:min(len(hist), 512)]
                    features.append(feature)
                else:
                    features.append(np.zeros(512))
            return np.array(features)
        
        encoder = simple_encoder
    
    # 处理每一帧
    frame_indices = sorted(image_filenames.keys())
    progress_bar = tqdm(frame_indices, desc=f"使用YOLO处理序列 {sequence_name}")
    
    for frame_idx in progress_bar:
        img_path = image_filenames[frame_idx]
        image = cv2.imread(img_path)
        
        if image is None:
            print(f"警告: 无法读取图像 {img_path}")
            continue
        
        # 将BGR转换为RGB
        frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # YOLO检测
        results = model.predict(
            source=frame_rgb,
            conf=YOLO_PARAMS['conf_thres'],
            iou=YOLO_PARAMS['iou_thres'],
            verbose=False
        )
        
        # 处理检测结果
        boxes = []
        if len(results) > 0:
            for box in results[0].boxes:
                if box.cls == 0:  # 只处理类别0（人）
                    xyxy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = xyxy
                    w = x2 - x1
                    h = y2 - y1
                    boxes.append([x1, y1, w, h])
        
        if boxes:
            boxes = np.array(boxes)
            try:
                features = encoder(image, boxes)
            except Exception as e:
                print(f"特征提取错误: {e}")
                features = np.zeros((len(boxes), 512))
            
            # 合并检测和特征
            for i, (box, feature) in enumerate(zip(boxes, features)):
                # MOTChallenge格式前10列: frame,id,x,y,w,h,score,...
                detection = np.zeros(10 + len(feature))
                detection[0] = frame_idx  # frame
                detection[1] = -1         # id (在检测阶段设为-1)
                detection[2:6] = box      # bbox (x,y,w,h)
                detection[6] = 1.0        # confidence
                detection[10:] = feature
                detections_with_features.append(detection)
    
    # 保存带有特征的检测结果
    output_filename = os.path.join(detection_output_dir, f"{sequence_name}.npy")
    np.save(output_filename, np.asarray(detections_with_features), allow_pickle=False)
    print(f"检测特征已保存到: {output_filename}")
    
    return detection_output_dir

def run_tracking_evaluation(sequence_dir, detection_dir, params):
    """运行跟踪评估并计算MOT指标"""
    sequence_name = os.path.basename(sequence_dir)
    
    # 创建结果目录
    results_dir = os.path.join(params['detection_dir'], "tracking_results")
    os.makedirs(results_dir, exist_ok=True)
    
    # 设置输出文件路径
    output_file = os.path.join(results_dir, f"{sequence_name}.txt")
    
    print(f"\n=== 运行跟踪评估: {sequence_name} ===")
    print(f"使用加速度卡尔曼滤波器: {params['use_acceleration']}")
    print(f"最大生命周期: {params.get('max_age', 30)}")
    print(f"初始化帧数: {params.get('n_init', 3)}")
    print(f"最大IoU距离: {params.get('max_iou_distance', 0.7)}")
    
    # 获取运动模型参数
    std_weight_position = params.get('std_weight_position', 1.0/20)
    std_weight_velocity = params.get('std_weight_velocity', 1.0/160)
    std_weight_acceleration = params.get('std_weight_acceleration', 1.0/10)
    velocity_smooth_factor = params.get('velocity_smooth_factor', 0.85)
    acceleration_smooth_factor = params.get('acceleration_smooth_factor', 0.7)
    
    print(f"运动模型参数:")
    print(f"  位置噪声权重: {std_weight_position}")
    print(f"  速度噪声权重: {std_weight_velocity}")
    print(f"  加速度噪声权重: {std_weight_acceleration}")
    print(f"  速度平滑因子: {velocity_smooth_factor}")
    print(f"  加速度平滑因子: {acceleration_smooth_factor}")
    
    # 导入deep_sort_app模块进行跟踪
    import deep_sort_app
    
    # 运行跟踪
    detection_file = os.path.join(detection_dir, f"{sequence_name}.npy")
    
    if not os.path.exists(detection_file):
        print(f"错误: 检测文件不存在 {detection_file}")
        return None
    
    try:
        deep_sort_app.run(
            sequence_dir=sequence_dir,
            detection_file=detection_file,
            output_file=output_file,
            min_confidence=params['min_confidence'],
            nms_max_overlap=params['nms_max_overlap'],
            min_detection_height=params['min_detection_height'],
            max_cosine_distance=params['max_cosine_distance'],
            nn_budget=params['nn_budget'],
            max_age=params.get('max_age', 30),
            n_init=params.get('n_init', 3),
            max_iou_distance=params.get('max_iou_distance', 0.7),
            display=False,
            use_acceleration=params['use_acceleration'],
            # 添加运动模型参数
            std_weight_position=std_weight_position,
            std_weight_velocity=std_weight_velocity,
            std_weight_acceleration=std_weight_acceleration,
            velocity_smooth_factor=velocity_smooth_factor,
            acceleration_smooth_factor=acceleration_smooth_factor
        )
        
        print(f"跟踪结果已保存到: {output_file}")
        return output_file
    except Exception as e:
        print(f"运行跟踪时出错: {e}")
        return None

def compute_mot_metrics(gt_file, result_file):
    """计算MOT评估指标"""
    print("\n=== 计算MOT指标 ===")
    
    # 读取Ground Truth和结果文件
    gt_data = np.loadtxt(gt_file, delimiter=',')
    if not os.path.exists(result_file):
        print(f"错误: 找不到结果文件 {result_file}")
        return None
        
    result_data = np.loadtxt(result_file, delimiter=',')
    
    # 创建accumulator
    acc = mm.MOTAccumulator(auto_id=True)
    
    # 处理每一帧
    frame_ids = np.unique(gt_data[:, 0]).astype(int)
    
    for frame_id in frame_ids:
        # 获取当前帧的GT数据
        gt_mask = gt_data[:, 0] == frame_id
        gt_boxes = gt_data[gt_mask]
        
        # 获取当前帧的结果数据
        res_mask = result_data[:, 0] == frame_id if len(result_data) > 0 else []
        res_boxes = result_data[res_mask] if len(result_data) > 0 else []
        
        # 获取GT对象ID和边界框
        gt_ids = gt_boxes[:, 1].astype(int)
        gt_bboxes = gt_boxes[:, 2:6]  # (x, y, w, h)
        
        # 获取结果对象ID和边界框
        res_ids = res_boxes[:, 1].astype(int) if len(res_boxes) > 0 else []
        res_bboxes = res_boxes[:, 2:6] if len(res_boxes) > 0 else []
        
        # 计算IoU距离
        if len(gt_ids) > 0 and len(res_ids) > 0:
            distances = mm.distances.iou_matrix(gt_bboxes, res_bboxes, max_iou=0.5)
            acc.update(gt_ids, res_ids, distances)
        else:
            acc.update(gt_ids, res_ids, [])
    
    # 计算指标
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=[
        'num_frames', 'num_objects', 'num_matches', 'num_switches',
        'num_false_positives', 'num_misses', 'mota', 'motp',
        'mostly_tracked', 'partially_tracked', 'mostly_lost',
        'precision', 'recall'
    ], name='acc')
    
    # 输出指标
    print("\n=============== MOT评估指标 ===============")
    print(f"MOTA: {summary['mota'].values[0]:.2%}")
    print(f"MOTP: {summary['motp'].values[0]:.2%}")
    print(f"ID Switches: {summary['num_switches'].values[0]}")
    print(f"Matches: {summary['num_matches'].values[0]}")
    print(f"False Positives: {summary['num_false_positives'].values[0]}")
    print(f"Misses: {summary['num_misses'].values[0]}")
    print(f"Precision: {summary['precision'].values[0]:.2%}")
    print(f"Recall: {summary['recall'].values[0]:.2%}")
    print(f"Mostly Tracked: {summary['mostly_tracked'].values[0]}")
    print(f"Partially Tracked: {summary['partially_tracked'].values[0]}")
    print(f"Mostly Lost: {summary['mostly_lost'].values[0]}")
    print("============================================")
    
    # 将指标保存到文件
    results_path = os.path.dirname(result_file)
    metrics_file = os.path.join(results_path, f"{os.path.basename(result_file).split('.')[0]}_metrics.txt")
    
    with open(metrics_file, 'w') as f:
        f.write("=============== MOT评估指标 ===============\n")
        f.write(f"MOTA: {summary['mota'].values[0]:.2%}\n")
        f.write(f"MOTP: {summary['motp'].values[0]:.2%}\n")
        f.write(f"ID Switches: {summary['num_switches'].values[0]}\n")
        f.write(f"Matches: {summary['num_matches'].values[0]}\n")
        f.write(f"False Positives: {summary['num_false_positives'].values[0]}\n")
        f.write(f"Misses: {summary['num_misses'].values[0]}\n")
        f.write(f"Precision: {summary['precision'].values[0]:.2%}\n")
        f.write(f"Recall: {summary['recall'].values[0]:.2%}\n")
        f.write(f"Mostly Tracked: {summary['mostly_tracked'].values[0]}\n")
        f.write(f"Partially Tracked: {summary['partially_tracked'].values[0]}\n")
        f.write(f"Mostly Lost: {summary['mostly_lost'].values[0]}\n")
        f.write("============================================\n")
    
    print(f"指标已保存到: {metrics_file}")
    return summary

def filter_short_tracks(tracking_data, min_length=5):
    """过滤掉短轨迹
    
    Args:
        tracking_data: 跟踪结果数据，MOT格式 (frame,id,x,y,w,h,...)
        min_length: 最小轨迹长度，小于此长度的轨迹将被过滤
        
    Returns:
        过滤后的跟踪结果
    """
    from collections import defaultdict
    
    # 如果输入数据为空或无效，直接返回空数组
    if tracking_data is None or len(tracking_data) == 0:
        return np.empty((0, 10))  # MOT格式默认至少有10列
    
    # 确保tracking_data是numpy数组
    if not isinstance(tracking_data, np.ndarray):
        try:
            tracking_data = np.array(tracking_data)
        except Exception as e:
            print(f"将跟踪数据转换为numpy数组时出错: {e}")
            return np.empty((0, 10))
    
    # 检查数据维度
    if len(tracking_data.shape) != 2 or tracking_data.shape[1] < 6:
        print(f"警告: 跟踪数据格式错误，维度: {tracking_data.shape if hasattr(tracking_data, 'shape') else '未知'}")
        return np.empty((0, 10))
    
    # 为每个ID统计出现的帧
    id_frames = defaultdict(list)
    for row in tracking_data:
        try:
            if len(row) >= 2:  # 确保行至少有frame和id两列
                frame_id = int(row[0])
                track_id = int(row[1])
                id_frames[track_id].append(frame_id)
        except (ValueError, IndexError, TypeError) as e:
            print(f"过滤轨迹时出错: {e}, 跳过此行数据: {row}")
            continue
    
    # 过滤短轨迹
    valid_ids = {tid for tid, frames in id_frames.items() if len(frames) >= min_length}
    
    # 创建过滤后的数据
    try:
        filtered = np.array([row for row in tracking_data if int(row[1]) in valid_ids])
        
        # 确保输出数组不为空
        if len(filtered) == 0:
            print("警告: 过滤后没有跟踪结果，请检查min_length参数或跟踪结果")
            return np.empty((0, tracking_data.shape[1]))
            
        # 检查过滤后数据格式是否正确
        if filtered.shape[1] < 6:  # 至少需要frame,id,x,y,w,h
            print(f"警告: 过滤后的数据维度不正确: {filtered.shape}")
            return np.empty((0, tracking_data.shape[1]))
            
        return filtered
    except Exception as e:
        print(f"创建过滤后的跟踪结果时出错: {e}")
        # 确保返回值具有正确的列数
        return np.empty((0, tracking_data.shape[1] if hasattr(tracking_data, 'shape') and len(tracking_data.shape) > 1 else 10))

def visualize_results(video_path, tracking_result, output_dir):
    """可视化跟踪结果"""
    if not os.path.exists(tracking_result):
        print(f"错误: 找不到结果文件 {tracking_result}")
        return

    try:
        tracking_data = np.loadtxt(tracking_result, delimiter=',')
        tracking_data = filter_short_tracks(tracking_data, min_length=5)
    except Exception as e:
        print(f"加载或过滤跟踪结果时出错: {e}")
        return
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频 {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    vis_output_path = os.path.join(output_dir, "visualized_tracking.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(vis_output_path, fourcc, fps, (width, height))

    left_line = int(width * VISUALIZATION_PARAMS['line_margin'])
    right_line = int(width * (1 - VISUALIZATION_PARAMS['line_margin']))

    counted_ids = set()      # 所有计数过的ID
    left_counted_ids = set() # 穿过左线的ID
    right_counted_ids = set()# 穿过右线的ID
    next_mapped_id = 1
    id_mapping = {}
    confirmed_tracks = set()
    last_frame_dict = {}     # 记录每个ID上一次出现的帧号
    
    # 记录已过线的ID
    crossed_ids = set()

    # 加载YOLO检测结果
    yolo_det_dir = os.path.join(os.path.dirname(tracking_result), "..", "detections")
    sequence_name = os.path.splitext(os.path.basename(video_path))[0]
    yolo_det_file = os.path.join(yolo_det_dir, f"{sequence_name}.npy")
    yolo_dets = None
    if os.path.exists(yolo_det_file):
        try:
            yolo_dets = np.load(yolo_det_file)
        except Exception as e:
            print(f"加载YOLO检测结果时出错: {e}")

    with tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), desc="可视化跟踪结果") as pbar:
        while cap.isOpened():
            try:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_count = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                
                # 确保tracking_data不为空且格式正确
                if tracking_data is None or len(tracking_data) == 0 or tracking_data.shape[1] < 6:
                    # 只绘制基本帧信息，没有跟踪结果
                    cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                    out.write(frame)
                    pbar.update(1)
                    continue
                
                mask = tracking_data[:, 0] == frame_count
                results = tracking_data[mask]

                # 画计数线
                cv2.line(frame, (left_line, 0), (left_line, height), (0, 255, 255), 2)
                cv2.line(frame, (right_line, 0), (right_line, height), (0, 255, 255), 2)

                # 画YOLO检测框（红色）
                if yolo_dets is not None:
                    try:
                        dets_this_frame = yolo_dets[yolo_dets[:, 0] == frame_count]
                        for det in dets_this_frame:
                            try:
                                x, y, w, h = map(int, det[2:6])
                                
                                # 添加安全检查，确保x、y、w、h是有效的坐标值
                                if (np.isnan(x) or np.isnan(y) or np.isnan(w) or np.isnan(h) or 
                                    w <= 0 or h <= 0 or x < 0 or y < 0 or 
                                    x + w >= width or y + h >= height):
                                    # 跳过无效的边界框
                                    continue
                                    
                                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 红色
                            except Exception as e:
                                print(f"绘制YOLO检测框时出错: {e}")
                                continue
                    except Exception as e:
                        print(f"处理YOLO检测结果时出错: {e}")

                for row in results:
                    try:
                        original_id = int(row[1])
                        x, y, w, h = map(int, row[2:6])
                        
                        # 添加安全检查，确保x、y、w、h是有效的坐标值和合理范围内
                        if (np.isnan(x) or np.isnan(y) or np.isnan(w) or np.isnan(h) or 
                            w <= 0 or h <= 0 or x < 0 or y < 0 or 
                            x + w >= width or y + h >= height):
                            # 跳过无效的边界框
                            continue
                            
                        center_x = x + w // 2
                        center_y = y + h // 2
                        confirmed_tracks.add(original_id)

                        # ID映射
                        if VISUALIZATION_PARAMS['remap_ids']:
                            if original_id not in id_mapping:
                                id_mapping[original_id] = next_mapped_id
                                next_mapped_id += 1
                            display_id = id_mapping[original_id]
                        else:
                            display_id = original_id

                        # 计数逻辑
                        if display_id not in counted_ids:
                            if center_x <= left_line:
                                left_counted_ids.add(display_id)
                                counted_ids.add(display_id)
                                crossed_ids.add(display_id)  # 标记为过线ID
                            elif center_x >= right_line:
                                right_counted_ids.add(display_id)
                                counted_ids.add(display_id)
                                crossed_ids.add(display_id)  # 标记为过线ID

                        # 判断是否为预测帧（ID在当前帧不是连续出现）
                        is_predicted = False
                        if display_id in last_frame_dict:
                            if frame_count - last_frame_dict[display_id] > 1:
                                is_predicted = True
                        last_frame_dict[display_id] = frame_count

                        # 框颜色
                        if display_id in crossed_ids:
                            color = VISUALIZATION_PARAMS['crossed_track_color']  # 过线ID使用亮蓝色
                        elif is_predicted:
                            color = (0, 165, 255)  # 橙色，预测框
                            cv2.putText(frame, "Pred", (x, y - 25),  # 添加Pred标签
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        else:
                            color = VISUALIZATION_PARAMS['track_color']  # 绿色，所有正常跟踪框

                        # 绘制边界框和ID 
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, VISUALIZATION_PARAMS['track_thickness'])
                        id_text = f"ID:{display_id}"
                        if VISUALIZATION_PARAMS['show_original_id'] and VISUALIZATION_PARAMS['remap_ids']:
                            id_text += f" ({original_id})"
                        cv2.putText(frame, id_text, (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, VISUALIZATION_PARAMS['text_color'], 2)
                    except Exception as e:
                        print(f"处理跟踪ID {row[1] if len(row) > 1 else '未知'}时出错: {e}")
                        continue

                # 显示计数信息
                cv2.putText(frame, f"Left: {len(left_counted_ids)}", (10, height-90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                cv2.putText(frame, f"Right: {len(right_counted_ids)}", (10, height-60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                cv2.putText(frame, f"Total: {len(counted_ids)}", (10, height-30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

                out.write(frame)
                pbar.update(1)
            except Exception as e:
                print(f"处理帧 {frame_count if 'frame_count' in locals() else '未知'} 时发生错误: {e}")
                # 尝试继续处理下一帧
                continue

    cap.release()
    out.release()
    print(f"可视化视频已保存到: {vis_output_path}")
    print(f"左向计数: {len(left_counted_ids)}")
    print(f"右向计数: {len(right_counted_ids)}")
    print(f"总计数: {len(counted_ids)}")

    # 保存计数结果
    count_results = {
        'left_count': len(left_counted_ids),
        'right_count': len(right_counted_ids),
        'total_ids': len(confirmed_tracks),
        'mapped_ids': len(id_mapping)
    }
    count_file = os.path.join(output_dir, "count_results.json")
    with open(count_file, 'w') as f:
        json.dump(count_results, f, indent=2)
    print(f"计数结果已保存到: {count_file}")

def process_frame(model, frame):
    """处理单帧图像
    Args:
        model: YOLO模型
        frame: BGR格式的numpy数组
    Returns:
        YOLO结果对象
    """
    if frame is None:
        return None
    
    try:
        # 将BGR转换为RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 使用新的YOLO API进行预测
        results = model.predict(
            source=frame_rgb,
            conf=YOLO_PARAMS['conf_thres'],
            iou=YOLO_PARAMS['iou_thres'],
            verbose=False
        )
        
        if len(results) > 0:
            return results[0]
        return None
            
    except Exception as e:
        print(f"Error in process_frame: {str(e)}")
        return None

def generate_pre_annotations(tracking_data, output_file, min_track_len=3, min_box_width=15):
    """生成预标注文件
    
    Args:
        tracking_data: 跟踪结果数据，MOT格式 (frame,id,x,y,w,h,...)
        output_file: 输出文件路径
        min_track_len: 最小跟踪长度（帧数）
        min_box_width: 最小框宽度
    """
    # 过滤短轨迹
    filtered_data = filter_short_tracks(tracking_data, min_length=min_track_len)
    
    if filtered_data is None or len(filtered_data) == 0:
        print("警告: 过滤后没有跟踪结果")
        return
    
    # 过滤小框
    mask = filtered_data[:, 4] >= min_box_width  # 第4列是宽度w
    filtered_data = filtered_data[mask]
    
    if len(filtered_data) == 0:
        print("警告: 过滤小框后没有跟踪结果")
        return
    
    # 准备输出数据
    output_data = []
    for row in filtered_data:
        frame_id = int(row[0])
        track_id = int(row[1])
        x, y, w, h = map(float, row[2:6])  # 确保是浮点数
        
        # MOT格式：frame,id,x,y,w,h,1,1,1.0
        output_row = [frame_id, track_id, x, y, w, h, 1, 1, 1.0]
        output_data.append(output_row)
    
    # 按帧号和ID排序
    output_data.sort(key=lambda x: (x[0], x[1]))
    
    # 保存到文件
    try:
        with open(output_file, 'w') as f:
            for row in output_data:
                line = ','.join(map(str, row))
                f.write(line + '\n')
        print(f"预标注文件已保存到: {output_file}")
        print(f"共生成 {len(output_data)} 个标注框")
    except Exception as e:
        print(f"保存预标注文件时出错: {e}")

def main():
    """主函数"""
    # 将配置参数打包为字典
    params = {
        'video_path': VIDEO_PATH,
        'output_dir': OUTPUT_DIR,
        'detection_dir': DETECTION_DIR,
        'gt_dir': GT_DIR,
        'output_gt': OUTPUT_GT,
        'output_label': OUTPUT_LABEL,
        'output_zip': OUTPUT_ZIP,
        'min_track_len': MIN_TRACK_LEN,
        'min_box_width': MIN_BOX_WIDTH,
        'model_path': MODEL_PATH,
        'min_confidence': TRACKING_PARAMS['min_confidence'],
        'nms_max_overlap': TRACKING_PARAMS['nms_max_overlap'],
        'min_detection_height': TRACKING_PARAMS['min_detection_height'],
        'max_cosine_distance': TRACKING_PARAMS['max_cosine_distance'],
        'nn_budget': TRACKING_PARAMS['nn_budget'],
        'use_acceleration': TRACKING_PARAMS['use_acceleration'],
        'max_age': TRACKING_PARAMS['max_age'],
        'n_init': TRACKING_PARAMS['n_init'],
        'max_iou_distance': TRACKING_PARAMS['max_iou_distance'],
        'std_weight_position': TRACKING_PARAMS['std_weight_position'],
        'std_weight_velocity': TRACKING_PARAMS['std_weight_velocity'],
        'std_weight_acceleration': TRACKING_PARAMS['std_weight_acceleration'],
        'velocity_smooth_factor': TRACKING_PARAMS['velocity_smooth_factor'],
        'acceleration_smooth_factor': TRACKING_PARAMS['acceleration_smooth_factor'],
    }
    
    # 初始化日志记录器
    logger = init_logger(OUTPUT_DIR)
    
    print("=== 生成预标注文件 ===")
    print(f"视频路径: {params['video_path']}")
    print(f"输出目录: {params['output_dir']}")
    print(f"检测结果目录: {params['detection_dir']}")
    print(f"GT目录: {params['gt_dir']}")
    
    print("\n=== 跟踪结果过滤参数 ===")
    print(f"最小跟踪长度: {params['min_track_len']}")
    print(f"最小框宽度: {params['min_box_width']}")
    
    print("\n=== YOLO参数 ===")
    print(f"模型路径: {YOLO_PARAMS['model_path']}")
    print(f"置信度阈值: {YOLO_PARAMS['conf_thres']}")
    print(f"IOU阈值: {YOLO_PARAMS['iou_thres']}")
    print(f"图像大小: {YOLO_PARAMS['img_size']}")
    print(f"设备: {YOLO_PARAMS['device']}")
    
    print("\n=== 跟踪参数 ===")
    print(f"最小置信度阈值: {params['min_confidence']}")
    print(f"非极大值抑制阈值: {params['nms_max_overlap']}")
    print(f"最小检测框高度: {params['min_detection_height']}")
    print(f"最大余弦距离: {params['max_cosine_distance']}")
    print(f"特征库大小: {params['nn_budget']}")
    print(f"最大生命周期: {params['max_age']}")
    print(f"初始化帧数: {params['n_init']}")
    print(f"最大IoU距离: {params['max_iou_distance']}")
    print(f"使用加速度模型: {params['use_acceleration']}")
    
    try:
        # 清理并创建目录结构
        if os.path.exists(OUTPUT_DIR):
            print(f"\n清空输出目录: {OUTPUT_DIR}")
            shutil.rmtree(OUTPUT_DIR)
        
        print(f"创建目录结构...")
        os.makedirs(DETECTION_DIR)
        os.makedirs(GT_DIR)
        
        # 复制标签文件
        print(f"复制标签文件...")
        shutil.copy2(LABEL_SOURCE, OUTPUT_LABEL)
        
        # 第1步: 加载YOLO模型
        print("\n步骤1: 加载YOLO模型...")
        yolo_model = load_yolo_model()
        if yolo_model is None:
            print("错误: 无法加载YOLO模型")
            return
        
        # 第2步: 从视频中提取帧
        print("\n步骤2: 从视频中提取帧...")
        cap = cv2.VideoCapture(VIDEO_PATH)
        if not cap.isOpened():
            print("错误: 无法打开视频")
            return
            
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # 创建临时序列目录
        sequence_dir = os.path.join(DETECTION_DIR, "sequence")
        img_dir = os.path.join(sequence_dir, "img1")
        os.makedirs(img_dir, exist_ok=True)
        
        # 创建seqinfo.ini文件
        with open(os.path.join(sequence_dir, "seqinfo.ini"), "w") as f:
            f.write("[Sequence]\n")
            f.write(f"name=sequence\n")
            f.write(f"imDir=img1\n")
            f.write(f"frameRate={fps}\n")
            f.write(f"seqLength={total_frames}\n")
            f.write(f"imWidth={width}\n")
            f.write(f"imHeight={height}\n")
            f.write(f"imExt=.jpg\n")
        
        # 提取帧
        frame_count = 0
        progress_bar = tqdm(total=total_frames, desc="提取视频帧")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            frame_path = os.path.join(img_dir, f"{frame_count:06d}.jpg")
            cv2.imwrite(frame_path, frame)
            progress_bar.update(1)
        
        progress_bar.close()
        cap.release()
        print(f"共提取 {frame_count} 帧")
        
        # 第3步: 使用YOLO生成检测结果
        print("\n步骤3: 使用YOLO生成检测结果...")
        detection_dir = generate_detections_with_yolo(sequence_dir, yolo_model)
        if not detection_dir:
            print("错误: 无法生成检测结果")
            return
        
        # 第4步: 运行跟踪
        print("\n步骤4: 运行跟踪...")
        tracking_result = run_tracking_evaluation(sequence_dir, detection_dir, params)
        if not tracking_result:
            print("错误: 无法运行跟踪")
            return
        
        # 第5步: 生成预标注文件
        print("\n步骤5: 生成预标注文件...")
        tracking_data = np.loadtxt(tracking_result, delimiter=',')
        generate_pre_annotations(tracking_data, OUTPUT_GT, 
                               min_track_len=MIN_TRACK_LEN,
                               min_box_width=MIN_BOX_WIDTH)
        
        # 第6步: 压缩gt文件夹
        print("\n步骤6: 压缩gt文件夹...")
        if os.path.exists(OUTPUT_ZIP):
            os.remove(OUTPUT_ZIP)
        shutil.make_archive(os.path.splitext(OUTPUT_ZIP)[0], 'zip', OUTPUT_DIR, 'gt')
        
        print("\n=== 完成 ===")
        print(f"预标注文件已保存到: {OUTPUT_GT}")
        print(f"压缩文件已保存到: {OUTPUT_ZIP}")
        
    except Exception as e:
        print(f"运行过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 关闭日志记录器
        if isinstance(sys.stdout, TeeLogger):
            sys.stdout.close()
            sys.stdout = sys.stdout.terminal

if __name__ == "__main__":
    main() 