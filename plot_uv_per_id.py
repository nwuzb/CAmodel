#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""plot_uv_per_id.py
读取 state_log.csv，为每个轨迹 ID 绘制中心像素 (u,v) 随 Frame 的变化曲线。

用法：
    python plot_uv_per_id.py /path/to/state_log.csv [--out OUT_DIR]
输出：在 OUT_DIR（默认 csv 同级 plots_uv/）生成 U_V_curve_IDxx.png，
      其中一张图包含两条线：u-Frame（蓝）、v-Frame（橙）。
"""
from __future__ import annotations
import csv
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt

# ---------------- 用户可在此修改路径 ----------------
CSV_PATH = "/Users/binzeng/MA/GT_videos/gt_60_videos/test_gt_60_2_clip_02_right_47_test-results-H5D-GT/state_log.csv"
OUT_DIR = None  # "/path/to/plots_uv"
# --------------------------------------------------

csv_path = Path(CSV_PATH).expanduser().resolve()
if not csv_path.exists():
    raise FileNotFoundError(csv_path)

out_dir = Path(OUT_DIR) if OUT_DIR else csv_path.parent / "plots_uv"
out_dir.mkdir(parents=True, exist_ok=True)

# 读取数据到 id -> [(frame,u,v)]
id2data: dict[int, list[tuple[int,float,float]]] = defaultdict(list)
with open(csv_path, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        tid = int(row["id"])
        frame = int(row["frame"])
        u = float(row["u"])
        v = float(row["v"])
        id2data[tid].append((frame, u, v))

for tid, points in id2data.items():
    # 按 frame 排序
    points.sort(key=lambda x: x[0])
    frames = [p[0] for p in points]
    us = [p[1] for p in points]
    vs = [p[2] for p in points]

    plt.figure(figsize=(8,5))
    plt.plot(frames, us, label="u (px)", color="tab:blue")
    plt.plot(frames, vs, label="v (px)", color="tab:orange")
    plt.title(f"Track ID {tid}: u & v vs Frame")
    plt.xlabel("Frame")
    plt.ylabel("Pixel")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    out_file = out_dir / f"UV_curve_ID{tid}.png"
    plt.savefig(out_file, dpi=150)
    plt.close()
    print(f"已保存 {out_file}")

print("全部完成！") 