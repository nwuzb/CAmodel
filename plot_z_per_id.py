#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""plot_z_per_id.py
读取由 Run_3DModel_H5D_GT.py 生成的 state_log.csv，
为每个轨迹 ID 绘制 Z 随 frame 的折线图并保存 PNG 文件。

用法：
    python plot_z_per_id.py /path/to/state_log.csv [--out OUT_DIR]
若 OUT_DIR 省略，则保存到与 CSV 同级目录下的 plots/ 目录。
"""
from __future__ import annotations
import csv
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt

# ---------------- 用户可在此修改路径 ----------------
CSV_PATH = "/Users/binzeng/MA/GT_videos/gt_60_videos/test_gt_60_2_clip_02_right_47_test-results-H5D-GT/state_log.csv"
# 若留空或 None, 默认与 CSV 同目录下创建 plots/
OUT_DIR = None  # "/path/to/plots"
# --------------------------------------------------

csv_path = Path(CSV_PATH).expanduser().resolve()
if not csv_path.exists():
    raise FileNotFoundError(csv_path)

out_dir = Path(OUT_DIR) if OUT_DIR else csv_path.parent / "plots"
out_dir.mkdir(parents=True, exist_ok=True)

# 读取数据
id2data: dict[int, list[tuple[int,float]]] = defaultdict(list)
with open(csv_path, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        tid = int(row["id"])
        frame = int(row["frame"])
        Z = float(row["Z"])
        id2data[tid].append((frame, Z))

# 按 frame 排序
for tid in id2data:
    id2data[tid].sort(key=lambda x: x[0])

print(f"读取到 {len(id2data)} 条轨迹，开始绘图 ...")

for tid, points in id2data.items():
    frames, Zs = zip(*points)
    plt.figure(figsize=(6,4))
    plt.plot(frames, Zs, marker="o", markersize=2, linewidth=1)
    plt.title(f"Track ID {tid}: Z vs Frame")
    plt.xlabel("Frame")
    plt.ylabel("Z (m)")
    plt.grid(True, linestyle="--", alpha=0.5)
    out_file = out_dir / f"Z_curve_ID{tid}.png"
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"保存 {out_file}")

print("全部完成！") 