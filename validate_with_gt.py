#!/usr/bin/env python
# coding: utf-8
"""validate_with_gt.py
====================================
使用 Ground Truth (gt.txt) 生成伪检测文件 (.npy)，
绕过 YOLO 检测器，以评估 DeepSORT/H5D 等跟踪模型
在 *理想检测* 场景下的上限性能。

运行示例：
    python validate_with_gt.py --folder "/path/to/gt_and_video_dir" \
                               --feature-dim 128 \
                               --use-acceleration

文件夹结构需至少包含：
    video.mp4      // 与 gt.txt 对应的视频
    gt.txt         // MOTChallenge 格式标注

输出目录默认 <folder>-results-GTDet，可通过 --output-dir 指定。
"""
from __future__ import annotations
import os
# argparse 已移除，用户采用常量配置
import shutil
from pathlib import Path
import numpy as np
import random
import importlib.util

# 复用现有评估流程中的工具函数
from 评估追踪器性能yolo import (
    prepare_sequence_dir,
    run_tracking_evaluation,
    compute_mot_metrics,
    TRACKING_PARAMS,
    visualize_results,
)

# 动态导入 Run_3DModel_H5D 以注入 H5D EKF 和相关 Monkey-Patch
def _import_h5d_patch():
    this_dir = Path(__file__).resolve().parent
    h5d_path = this_dir / "Run_3DModel_H5D.py"
    if not h5d_path.exists():
        print("WARNING: 未找到 Run_3DModel_H5D.py，无法注入 H5D EKF")
        return
    spec = importlib.util.spec_from_file_location("Run_3DModel_H5D", str(h5d_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # 不触发其 __main__ 部分，只获取 Monkey-Patch 副作用

# ---------------- 工具函数 ----------------

def _find_video_file(folder: Path) -> Path:
    """在给定文件夹中搜索第一个视频文件 (mp4/avi/mov)。"""
    for ext in (".mp4", ".avi", ".mov", ".mkv"):
        for f in folder.iterdir():
            if f.suffix.lower() == ext:
                return f
    raise FileNotFoundError(f"未在 {folder} 中找到视频文件 (支持 mp4/avi/mov/mkv)。")


def generate_detections_from_gt(
    gt_path: Path,
    sequence_dir: Path,
    output_dir: Path,
    feature_dim: int = 128,
    confidence: float = 1.0,
) -> Path:
    """根据 GT 生成带外观特征的检测 .npy 文件并返回 detection_dir。"""
    # 输出目录：<output_dir>/detections
    detection_dir = output_dir / "detections"
    detection_dir.mkdir(parents=True, exist_ok=True)

    sequence_name = sequence_dir.name

    # 读取 GT
    gt = np.loadtxt(gt_path, delimiter=",")
    if gt.ndim == 1:
        gt = gt.reshape(1, -1)

    # 为每个 gt-id 分配一个固定随机特征向量（单位向量）
    rng = np.random.default_rng(0)
    id2feat: dict[int, np.ndarray] = {}

    detections = []
    for row in gt:
        frame, obj_id, x, y, w, h = row[:6]
        frame = int(frame)
        obj_id = int(obj_id)

        if obj_id not in id2feat:
            vec = rng.standard_normal(feature_dim)
            vec = vec / (np.linalg.norm(vec) + 1e-12)
            id2feat[obj_id] = vec
        feat = id2feat[obj_id]

        det_row = np.zeros(10 + feature_dim, dtype=float)
        det_row[0] = frame                # frame idx (1-based)
        det_row[1] = -1                   # detection stage id placeholder
        det_row[2:6] = [x, y, w, h]       # bbox tlwh
        det_row[6] = confidence           # conf
        # det_row[7:10] 默认为 0，即 MOT 的 3 个可选字段
        det_row[10:] = feat               # appearance feature
        detections.append(det_row)

    detections = np.asarray(detections, dtype=float)
    detections = detections[np.argsort(detections[:, 0])]  # 按帧排序

    output_file = detection_dir / f"{sequence_name}.npy"
    np.save(output_file, detections, allow_pickle=False)
    return detection_dir

# ---------------- 用户配置 ----------------

# ！！！请在此处填写文件夹路径！！！
FOLDER_PATH = "/Users/binzeng/MA/GT_videos/gt_60_videos/test_gt_60_2_clip_02_right_47_test"  # 必填，包含 video.mp4 + gt.txt
# 可选：指定结果输出目录；设为 None 则自动 <folder>-results-GTDet
OUTPUT_DIR = None
# 生成的外观特征维度
FEATURE_DIM = 128
# 是否启用加速度 Kalman 过滤器
USE_ACCELERATION = True

# ---------------- 主流程 & 对外函数 ----------------

def run_validate(folder: Path | str,
                 output_dir: Path | str | None = None,
                 feature_dim: int = 128,
                 use_acceleration: bool = True):
    """执行基于 GT 的跟踪评估，可供外部调用。"""
    folder = Path(folder).expanduser().resolve()
    if not folder.is_dir():
        raise NotADirectoryError(folder)

    video_path = _find_video_file(folder)
    gt_path = folder / "gt.txt"
    if not gt_path.exists():
        raise FileNotFoundError(f"未找到 {gt_path}")

    output_dir = Path(output_dir) if output_dir else folder.with_name(folder.name + "-results-GTDet")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("步骤1: 生成 MOT 数据目录 …")
    sequence_dir = Path(prepare_sequence_dir(str(video_path), str(gt_path), str(output_dir)))

    print("步骤2: 由 GT 生成伪检测 …")
    detection_dir = generate_detections_from_gt(gt_path, sequence_dir, output_dir, feature_dim)

    print("步骤3: 运行跟踪评估 …")
    # 先导入 H5D Monkey-Patch
    _import_h5d_patch()
    params = dict(TRACKING_PARAMS)
    params.update({
        "output_dir": str(output_dir),
        "use_acceleration": use_acceleration,
    })
    tracking_result = run_tracking_evaluation(str(sequence_dir), str(detection_dir), params)

    print("步骤4: 计算 MOT 指标 …")
    if tracking_result:
        compute_mot_metrics(str(gt_path), tracking_result)
    else:
        print("跟踪结果不存在，无法计算指标。")

    # 步骤 5: 可视化追踪结果
    print("步骤5: 可视化结果 …")
    if tracking_result:
        visualize_results(str(video_path), tracking_result, str(output_dir))

    print(f"全部完成。结果保存至 {output_dir}")


def main():
    if FOLDER_PATH == "/absolute/path/to/gt_folder":
        raise ValueError("请先在 validate_with_gt.py 顶部填写 FOLDER_PATH！")
    run_validate(FOLDER_PATH, OUTPUT_DIR, FEATURE_DIM, USE_ACCELERATION)


if __name__ == "__main__":
    main() 