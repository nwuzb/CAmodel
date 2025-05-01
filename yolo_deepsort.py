import cv2
import numpy as np
from deep_sort.tracker import Tracker
from deep_sort.nn_matching import NearestNeighborDistanceMetric
from deep_sort.detection import Detection
from ultralytics import YOLO
import os
import torch
from PIL import Image

# YOLO检测参数
DETECTION_PARAMS = {
    'conf': 0.7,      # 置信度阈值：用于过滤检测结果，值越高要求越严格
    'iou': 0.4,       # IoU阈值：用于非极大值抑制，值越高允许的框重叠程度越高
    'max_det': 300,   # 每帧最大检测数：限制每帧检测的最大目标数量
    'classes': [0],   # 只检测人类（class 0）
}

# DeepSORT跟踪参数
TRACKING_PARAMS = {
    'max_cosine_distance': 0.4,    # 控制外观特征匹配容忍度（越小越严格）
    'nn_budget': 100,              # 控制历史外观特征数量
    'max_age': 30,                 # 目标消失后保持跟踪的最大帧数
    'n_init': 2,                   # 确认为稳定跟踪目标所需的最小检测帧数
    'max_iou_distance': 1.0,       # 最大IoU距离：用于跟踪匹配，值越大允许的运动越大
}

# 可视化参数
VISUALIZATION_PARAMS = {
    'detection_color': (0, 0, 255),    # 红色检测框
    'track_color': (0, 255, 0),        # 绿色跟踪框
    'text_color': (255, 255, 255),     # 白色文字
    'detection_thickness': 1,           # 检测框线宽
    'track_thickness': 2,              # 跟踪框线宽
    'line_margin': 0.125,              # 计数线位置（相对于图像宽度的比例）
    'remap_ids': True,                 # 是否重映射ID
    'show_original_id': False          # 是否显示原始ID
}

def create_tracker():
    metric = NearestNeighborDistanceMetric(
        "cosine", 
        TRACKING_PARAMS['max_cosine_distance'], 
        TRACKING_PARAMS['nn_budget']
    )
    tracker = Tracker(
        metric,
        max_iou_distance=TRACKING_PARAMS['max_iou_distance'],
        max_age=TRACKING_PARAMS['max_age'],
        n_init=TRACKING_PARAMS['n_init']
    )
    return tracker

def xyxy_to_xywh(xyxy):
    """将(x1,y1,x2,y2)格式转换为(x,y,w,h)格式"""
    x1, y1, x2, y2 = xyxy
    w = x2 - x1
    h = y2 - y1
    x = x1  # 使用左上角x坐标
    y = y1  # 使用左上角y坐标
    return np.array([x, y, w, h])

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
        
        # 将numpy数组转换为PIL Image
        pil_image = Image.fromarray(frame_rgb)
        
        # 创建临时文件来保存当前帧
        temp_frame_path = 'temp_frame.jpg'
        pil_image.save(temp_frame_path)
        
        try:
            # 使用文件路径作为输入
            results = model.predict(
                source=temp_frame_path,
                conf=DETECTION_PARAMS['conf'],
                iou=DETECTION_PARAMS['iou'],
                max_det=DETECTION_PARAMS['max_det'],
                classes=DETECTION_PARAMS['classes'],
                verbose=False,
                device='mps'
            )
            
            if len(results) > 0:
                return results[0]
            return None
            
        finally:
            # 清理临时文件
            if os.path.exists(temp_frame_path):
                os.remove(temp_frame_path)
                
    except Exception as e:
        print(f"Error in process_frame: {str(e)}")
        return None

def process_video(video_path, yolo_weights, output_path):
    print("\n=== 设备信息 ===")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"MPS是否可用: {torch.backends.mps.is_available()}")
    print(f"MPS是否内置: {torch.backends.mps.is_built()}")
    
    # 初始化YOLO模型
    model = YOLO(yolo_weights)
    if torch.backends.mps.is_available():
        print("使用MPS设备进行加速")
        model.to('mps')
    else:
        print("使用CPU设备")
        model.to('cpu')
    
    print("\n=== 检测参数 ===")
    print("YOLO参数:")
    for k, v in DETECTION_PARAMS.items():
        print(f"  {k}: {v}")
    print("\nDeepSORT参数:")
    for k, v in TRACKING_PARAMS.items():
        print(f"  {k}: {v}")
    
    tracker = create_tracker()
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误：无法打开视频文件 {video_path}")
        return
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\n=== 视频信息 ===")
    print(f"分辨率: {width}x{height}")
    print(f"帧率: {fps}")
    print(f"总帧数: {total_frames}")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 初始化计数线位置
    left_line = int(width * VISUALIZATION_PARAMS['line_margin'])
    right_line = int(width * (1 - VISUALIZATION_PARAMS['line_margin']))
    
    # 初始化计数器和跟踪历史
    left_count = 0
    right_count = 0
    track_history = {}
    id_mapping = {}
    next_mapped_id = 1
    confirmed_tracks = set()
    
    frame_count = 0
    detection_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"正在处理第 {frame_count}/{total_frames} 帧 ({frame_count/total_frames*100:.1f}%)")
            
        results = process_frame(model, frame)
        if results is None:
            continue
            
        boxes = []
        confidences = []
        features = []
        
        if len(results.boxes) > 0:
            for box in results.boxes:
                if box.cls.cpu().numpy()[0] == 0:  # 只处理人
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    
                    # 绘制检测框
                    x1, y1, x2, y2 = map(int, xyxy)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), 
                                VISUALIZATION_PARAMS['detection_color'], 
                                VISUALIZATION_PARAMS['detection_thickness'])
                    cv2.putText(frame, f'conf: {conf:.2f}', (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                               VISUALIZATION_PARAMS['detection_color'], 2)
                    
                    # 转换为DeepSORT所需的格式
                    bbox_xywh = xyxy_to_xywh(xyxy)
                    boxes.append(bbox_xywh)
                    confidences.append(conf)
                    
                    # 提取特征
                    person_img = frame[y1:y2, x1:x2]
                    if person_img.size > 0:
                        hist = cv2.calcHist([person_img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
                        hist = cv2.normalize(hist, hist).flatten()
                        features.append(hist)
                    else:
                        features.append(np.zeros(512))
        
        if len(boxes) > 0:
            detection_count += 1
            boxes = np.array(boxes)
            confidences = np.array(confidences)
            features = np.array(features)
            
            detections = []
            for bbox, conf, feature in zip(boxes, confidences, features):
                detection = Detection(bbox, conf, feature)
                detections.append(detection)
            
            tracker.predict()
            tracker.update(detections)
            
            # 处理每个跟踪目标
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                
                confirmed_tracks.add(track.track_id)
                
                # ID映射处理
                original_id = track.track_id
                if VISUALIZATION_PARAMS['remap_ids']:
                    if original_id not in id_mapping:
                        id_mapping[original_id] = next_mapped_id
                        next_mapped_id += 1
                    display_id = id_mapping[original_id]
                else:
                    display_id = original_id
                
                bbox = track.to_tlbr()
                x1, y1, x2, y2 = map(int, bbox)
                current_x = (x1 + x2) / 2
                
                # 绘制跟踪框和ID
                cv2.rectangle(frame, (x1, y1), (x2, y2), 
                            VISUALIZATION_PARAMS['track_color'],
                            VISUALIZATION_PARAMS['track_thickness'])
                
                # 显示ID
                id_text = f"ID:{display_id}"
                if VISUALIZATION_PARAMS['show_original_id'] and VISUALIZATION_PARAMS['remap_ids']:
                    id_text += f" ({original_id})"
                
                cv2.putText(frame, id_text, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                           VISUALIZATION_PARAMS['text_color'], 2)
                
                # 处理计数
                if original_id not in track_history:
                    track_history[original_id] = {
                        'prev_x': current_x,
                        'crossed_left': False,
                        'crossed_right': False
                    }
                    continue
                
                prev_x = track_history[original_id]['prev_x']
                
                # 判断左线穿越
                if not track_history[original_id]['crossed_left']:
                    if (prev_x < left_line and current_x >= left_line) or \
                       (prev_x > left_line and current_x <= left_line):
                        left_count += 1
                        track_history[original_id]['crossed_left'] = True
                
                # 判断右线穿越
                if not track_history[original_id]['crossed_right']:
                    if (prev_x < right_line and current_x >= right_line) or \
                       (prev_x > right_line and current_x <= right_line):
                        right_count += 1
                        track_history[original_id]['crossed_right'] = True
                
                track_history[original_id]['prev_x'] = current_x
            
            # 清理无效轨迹
            active_ids = {t.track_id for t in tracker.tracks if t.is_confirmed()}
            for tid in list(track_history.keys()):
                if tid not in active_ids:
                    del track_history[tid]
        
        # 绘制计数线和统计信息
        cv2.line(frame, (left_line, 0), (left_line, height), (0,255,0), 2)
        cv2.line(frame, (right_line, 0), (right_line, height), (0,0,255), 2)
        
        # 显示统计信息
        stats_y = 30
        cv2.putText(frame, f"Left: {left_count}", (left_line+10, stats_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.putText(frame, f"Right: {right_count}", (right_line-150, stats_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        cv2.putText(frame, f"Total: {left_count + right_count}",
                   (width//2-100, height-30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,0), 3)
        
        # 显示ID映射信息
        if VISUALIZATION_PARAMS['remap_ids']:
            cv2.putText(frame, f"Original IDs: {len(confirmed_tracks)}",
                       (10, height-60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(frame, f"Mapped IDs: {len(id_mapping)}",
                       (10, height-30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        
        out.write(frame)
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"\n=== 处理完成 ===")
    print(f"总帧数: {frame_count}")
    print(f"检测到目标的帧数: {detection_count}")
    print(f"检测率: {detection_count/frame_count*100:.1f}%")
    print(f"左线计数: {left_count}")
    print(f"右线计数: {right_count}")
    print(f"总计数: {left_count + right_count}")
    print(f"原始ID总数: {len(confirmed_tracks)}")
    print(f"映射后ID总数: {len(id_mapping)}")
    print(f"输出视频保存为: {output_path}")

if __name__ == '__main__':
    video_path = '/Users/binzeng/MA/EvaluateVideos/2_clip_03_rightpart_122.mp4'
    yolo_weights = '/Users/binzeng/MA/EvaluateVideos/finetune.pt'
    # 构造输出路径：将文件名放入 results 文件夹下
    output_path = os.path.join('/Users/binzeng/MA/Basemodel/results', os.path.basename(video_path))

    
    process_video(video_path, yolo_weights, output_path) 