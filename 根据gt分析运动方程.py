import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import cv2
from sklearn.linear_model import LinearRegression
from collections import defaultdict
import seaborn as sns
import matplotlib as mpl
from matplotlib.font_manager import FontProperties

# 设置中文字体 - 修改为适用于macOS的字体
# 尝试多种可能的中文字体
plt.rcParams['axes.unicode_minus'] = False

# 尝试多种可能的macOS中文字体
chinese_fonts = ['Arial Unicode MS', 'PingFang SC', 'Heiti SC', 'Hiragino Sans GB', 'STHeiti']

# 检查字体是否可用
font_found = False
for font_name in chinese_fonts:
    try:
        font_prop = FontProperties(fname=mpl.font_manager.findfont(font_name))
        plt.rcParams['font.sans-serif'] = [font_name] + plt.rcParams['font.sans-serif']
        font_found = True
        print(f"使用字体: {font_name}")
        break
    except:
        continue

# 如果没有找到中文字体，则使用英文标签
if not font_found:
    print("未找到中文字体，将使用英文标签")
    use_chinese = False
else:
    use_chinese = True

# 文件路径
video_path = "/Users/binzeng/MA/GT_videos/gt_30_2_clip_03_right_122/30fps_2_clip_03_122.mp4"
gt_path = "/Users/binzeng/MA/GT_videos/gt_30_2_clip_03_right_122/gt.txt"
result_dir = "/Users/binzeng/MA/GT_videos/gt_30_2_clip_03_right_122/result"

# 创建结果文件夹
os.makedirs(result_dir, exist_ok=True)

# 读取视频信息
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()
print(f"视频帧率: {fps} FPS, 分辨率: {width}x{height}")

# 读取GT数据
# MOT格式: frame_id, track_id, x, y, width, height, confidence, -1, -1, -1
columns = ['frame', 'id', 'x', 'y', 'width', 'height', 'conf', 'x3d', 'y3d', 'z3d']
gt_data = pd.read_csv(gt_path, header=None, names=columns)

# 计算中心点坐标
gt_data['center_x'] = gt_data['x'] + gt_data['width'] / 2
gt_data['center_y'] = gt_data['y'] + gt_data['height'] / 2

# 统计基本信息
frame_count = gt_data['frame'].max()
track_count = gt_data['id'].nunique()
print(f"总帧数: {frame_count}")
print(f"总目标数: {track_count}")

# 按ID分组
tracks = {}
for track_id, group in gt_data.groupby('id'):
    tracks[track_id] = group.sort_values('frame')

# 计算每个轨迹的速度和加速度
for track_id, track in tracks.items():
    # 计算帧间速度
    track_data = []
    for i in range(1, len(track)):
        prev = track.iloc[i-1]
        curr = track.iloc[i]
        frame_diff = curr['frame'] - prev['frame']
        time_diff = frame_diff / fps  # 转换为秒
        
        dx = curr['center_x'] - prev['center_x']
        dy = curr['center_y'] - prev['center_y']
        
        speed_x = dx / time_diff if time_diff > 0 else 0
        speed_y = dy / time_diff if time_diff > 0 else 0
        speed = np.sqrt(speed_x**2 + speed_y**2)
        
        direction = np.arctan2(dy, dx) * 180 / np.pi  # 方向角度
        
        track_data.append({
            'frame': curr['frame'],
            'time': curr['frame'] / fps,
            'x': curr['center_x'],
            'y': curr['center_y'],
            'speed_x': speed_x,
            'speed_y': speed_y,
            'speed': speed,
            'direction': direction
        })
    
    # 计算加速度
    for i in range(1, len(track_data)):
        prev = track_data[i-1]
        curr = track_data[i]
        time_diff = (curr['frame'] - prev['frame']) / fps
        
        if time_diff > 0:
            acc_x = (curr['speed_x'] - prev['speed_x']) / time_diff
            acc_y = (curr['speed_y'] - prev['speed_y']) / time_diff
            acc = np.sqrt(acc_x**2 + acc_y**2)
        else:
            acc_x = acc_y = acc = 0
            
        track_data[i]['acc_x'] = acc_x
        track_data[i]['acc_y'] = acc_y
        track_data[i]['acc'] = acc
    
    # 为第一帧添加加速度数据
    if track_data:
        track_data[0]['acc_x'] = track_data[0]['acc_y'] = track_data[0]['acc'] = 0
    
    tracks[track_id] = pd.DataFrame(track_data)

# 绘制轨迹图
plt.figure(figsize=(12, 10))
for track_id, track in tracks.items():
    plt.plot(track['x'], track['y'], '-o', linewidth=1, markersize=2, label=f'ID {track_id}')
    # 起点标绿，终点标红
    plt.plot(track['x'].iloc[0], track['y'].iloc[0], 'go', markersize=6)
    plt.plot(track['x'].iloc[-1], track['y'].iloc[-1], 'ro', markersize=6)

plt.xlim(0, width)
plt.ylim(height, 0)  # 反转Y轴以匹配图像坐标
plt.xlabel('X Coordinate' if not use_chinese else 'X坐标')
plt.ylabel('Y Coordinate' if not use_chinese else 'Y坐标')
plt.title('Target Trajectories' if not use_chinese else '目标轨迹')
plt.grid(True)
if track_count < 20:  # 当目标数量合适时显示图例
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig(os.path.join(result_dir, 'trajectories.png'), dpi=300, bbox_inches='tight')
plt.close()

# 分析速度分布
all_speeds = np.concatenate([track['speed'].values for track in tracks.values()])
plt.figure(figsize=(10, 6))
sns.histplot(all_speeds, kde=True)
plt.xlabel('Speed (pixels/s)' if not use_chinese else '速度 (像素/秒)')
plt.ylabel('Frequency' if not use_chinese else '频次')
plt.title('Speed Distribution' if not use_chinese else '目标速度分布')
plt.grid(True)
plt.savefig(os.path.join(result_dir, 'speed_distribution.png'), dpi=300)
plt.close()

# 分析加速度分布
all_accs = np.concatenate([track['acc'].values for track in tracks.values() if 'acc' in track.columns])
plt.figure(figsize=(10, 6))
sns.histplot(all_accs, kde=True)
plt.xlabel('Acceleration (pixels/s²)' if not use_chinese else '加速度 (像素/秒²)')
plt.ylabel('Frequency' if not use_chinese else '频次')
plt.title('Acceleration Distribution' if not use_chinese else '目标加速度分布')
plt.grid(True)
plt.savefig(os.path.join(result_dir, 'acceleration_distribution.png'), dpi=300)
plt.close()

# 分析方向变化
direction_changes = []
for track_id, track in tracks.items():
    if len(track) > 1:
        direction = track['direction'].values
        # 计算相邻方向之间的差异（注意角度的循环性）
        diff = np.diff(direction)
        # 调整差异以处理角度跨越+/-180度的情况
        diff = (diff + 180) % 360 - 180
        direction_changes.extend(np.abs(diff))

plt.figure(figsize=(10, 6))
sns.histplot(direction_changes, kde=True)
plt.xlabel('Direction Change (degrees)' if not use_chinese else '方向变化 (度)')
plt.ylabel('Frequency' if not use_chinese else '频次')
plt.title('Direction Changes Distribution' if not use_chinese else '目标方向变化分布')
plt.grid(True)
plt.savefig(os.path.join(result_dir, 'direction_changes.png'), dpi=300)
plt.close()

# 分析运动模式（线性/非线性）
linearity_scores = {}
for track_id, track in tracks.items():
    if len(track) > 5:  # 只分析有足够数据点的轨迹
        X = track['x'].values.reshape(-1, 1)
        y = track['y'].values
        
        model = LinearRegression()
        model.fit(X, y)
        r2_score = model.score(X, y)
        linearity_scores[track_id] = r2_score

if linearity_scores:
    plt.figure(figsize=(10, 6))
    scores = list(linearity_scores.values())
    plt.bar(range(len(scores)), scores)
    plt.xlabel('Target ID' if not use_chinese else '目标ID')
    plt.ylabel('Linearity (R²)' if not use_chinese else '线性度 (R²)')
    plt.title('Trajectory Linearity Analysis' if not use_chinese else '轨迹线性度分析')
    plt.xticks(range(len(scores)), list(linearity_scores.keys()))
    plt.grid(True)
    plt.savefig(os.path.join(result_dir, 'linearity_analysis.png'), dpi=300)
    plt.close()

# 时间序列分析
# 选取几个有代表性的目标进行分析
representative_ids = list(tracks.keys())[:min(5, len(tracks))]
for metric in ['speed', 'acc']:
    plt.figure(figsize=(12, 6))
    for track_id in representative_ids:
        track = tracks[track_id]
        if metric in track.columns:
            plt.plot(track['time'], track[metric], '-', label=f'ID {track_id}')
    
    plt.xlabel('Time (s)' if not use_chinese else '时间 (秒)')
    if metric == 'speed':
        plt.ylabel('Speed (pixels/s)' if not use_chinese else '速度 (像素/秒)')
        plt.title('Speed vs Time' if not use_chinese else '目标速度随时间变化')
    else:
        plt.ylabel('Acceleration (pixels/s²)' if not use_chinese else '加速度 (像素/秒²)')
        plt.title('Acceleration vs Time' if not use_chinese else '目标加速度随时间变化')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(result_dir, f'{metric}_time_series.png'), dpi=300)
    plt.close()

# 分析速度与位置的关系
plt.figure(figsize=(12, 10))
for track_id in representative_ids:
    track = tracks[track_id]
    plt.scatter(track['x'], track['y'], c=track['speed'], cmap='viridis', 
                s=30, label=f'ID {track_id}')

plt.colorbar(label='Speed (pixels/s)' if not use_chinese else '速度 (像素/秒)')
plt.xlim(0, width)
plt.ylim(height, 0)  # 反转Y轴以匹配图像坐标
plt.xlabel('X Coordinate' if not use_chinese else 'X坐标')
plt.ylabel('Y Coordinate' if not use_chinese else 'Y坐标')
plt.title('Speed vs Position' if not use_chinese else '速度与位置关系')
plt.grid(True)
plt.savefig(os.path.join(result_dir, 'speed_vs_position.png'), dpi=300)
plt.close()

# 统计并导出关键指标
stats = {
    '平均速度': np.mean(all_speeds),
    '速度标准差': np.std(all_speeds),
    '最大速度': np.max(all_speeds),
    '平均加速度': np.mean(all_accs),
    '加速度标准差': np.std(all_accs),
    '最大加速度': np.max(all_accs),
    '平均方向变化': np.mean(direction_changes) if direction_changes else 0,
    '非线性程度': 1 - np.mean(list(linearity_scores.values())) if linearity_scores else 0
}

# 创建分析报告
report_filename = os.path.join(result_dir, 'motion_analysis.txt')

# 使用英文和中文双语报告
with open(report_filename, 'w', encoding='utf-8') as f:
    f.write("# 运动特性分析报告 (Motion Analysis Report)\n\n")
    f.write(f"视频帧率 (Video FPS): {fps} FPS, 分辨率 (Resolution): {width}x{height}\n")
    f.write(f"总帧数 (Total Frames): {frame_count}\n")
    f.write(f"总目标数 (Total Objects): {track_count}\n\n")
    
    f.write("## 统计指标 (Statistics)\n\n")
    stat_names_en = {
        '平均速度': 'Average Speed',
        '速度标准差': 'Speed Standard Deviation', 
        '最大速度': 'Maximum Speed',
        '平均加速度': 'Average Acceleration', 
        '加速度标准差': 'Acceleration Standard Deviation',
        '最大加速度': 'Maximum Acceleration', 
        '平均方向变化': 'Average Direction Change',
        '非线性程度': 'Non-linearity Degree'
    }
    
    for key, value in stats.items():
        f.write(f"{key} ({stat_names_en[key]}): {value:.4f}\n")
    
    f.write("\n## DeepSORT运动方程建议 (DeepSORT Motion Model Suggestions)\n\n")
    
    # 根据分析结果提出建议，同时提供英文
    if stats['平均速度'] > 50:
        f.write("- 目标平均速度较高，建议增大状态转移矩阵中速度相关的方差\n")
        f.write("  (High average speed, consider increasing velocity-related variance in state transition matrix)\n")
    else:
        f.write("- 目标平均速度适中，可使用默认的状态转移矩阵方差\n")
        f.write("  (Moderate average speed, default state transition matrix variance should work)\n")
        
    if stats['速度标准差'] > 30:
        f.write("- 速度变化较大，建议增大过程噪声协方差矩阵中速度维度的值\n")
        f.write("  (Large speed variation, consider increasing velocity dimension values in process noise covariance matrix)\n")
    else:
        f.write("- 速度变化较小，可使用默认的过程噪声协方差矩阵\n")
        f.write("  (Small speed variation, default process noise covariance matrix should work)\n")
        
    if stats['平均加速度'] > 100:
        f.write("- 加速度较大，建议考虑使用恒加速度模型替代恒速度模型\n")
        f.write("  (High acceleration, consider using constant acceleration model instead of constant velocity model)\n")
    else:
        f.write("- 加速度适中，可继续使用恒速度模型\n")
        f.write("  (Moderate acceleration, constant velocity model should work)\n")
        
    if stats['平均方向变化'] > 30:
        f.write("- 目标方向变化频繁，建议减小卡尔曼滤波器的运动权重，增大外观权重\n")
        f.write("  (Frequent direction changes, consider reducing motion weight and increasing appearance weight in Kalman filter)\n")
    else:
        f.write("- 目标方向变化不大，可使用默认的运动权重和外观权重\n")
        f.write("  (Small direction changes, default motion and appearance weights should work)\n")
        
    if stats['非线性程度'] > 0.5:
        f.write("- 轨迹非线性程度高，建议考虑使用更复杂的非线性卡尔曼滤波器\n")
        f.write("  (High non-linearity in trajectories, consider using more complex non-linear Kalman filter)\n")
    else:
        f.write("- 轨迹基本符合线性假设，线性卡尔曼滤波器应该能够处理\n")
        f.write("  (Trajectories mostly follow linear assumption, linear Kalman filter should work)\n")

print(f"分析完成，结果已保存到: {result_dir}")

# 显示几个关键图像
print("主要统计指标 (Key Statistics):")
for key, value in stats.items():
    if key in stat_names_en:
        print(f"{key} ({stat_names_en[key]}): {value:.4f}")
    else:
        print(f"{key}: {value:.4f}")
