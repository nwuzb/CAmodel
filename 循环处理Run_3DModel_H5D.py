#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""循环处理Run_3DModel_H5D.py
根据 Excel 列表批量执行 deep_sort/Run_3DModel_H5D.py。
要求 Excel 四列依次为：F_PIXELS, CX, CY, FOLDER。

使用方式：直接运行本脚本即可，修改下面 EXCEL_PATH 常量。

原理：
1. 逐行读取表格；
2. 临时复制一份 Run_3DModel_H5D.py 并替换参数;
3. subprocess 调用该临时脚本;
4. 收集结果文件夹到统一目录;
5. 自动统计分析所有结果;
6. 日志输出到同目录。"""
from __future__ import annotations
import subprocess, tempfile, shutil, re, sys
from pathlib import Path
import pandas as pd

# ---------------- 用户可在此修改 ----------------
EXCEL_PATH = "/Users/binzeng/MA/GT_videos/gt_所有视频_循环参数新min_length=3.xlsx"
RUN_SCRIPT = Path(__file__).parent / "Run_3DModel_H5D.py"
PYTHON_BIN = sys.executable  # 当前 python
# ------------------------------------------------

# ============ 整合统计terminallog.py功能 ============
# 指标映射：列名 → 正则 / 解析回调
PATTERNS: dict[str, str | tuple[str, callable[[str], object]]] = {
    "Folder"          : r"提取视频\s+([\w\-\.]+)",
    "yolo"           : r"模型[:：]\s*([^\s]+)",
    "MOTA"           : r"MOTA[:：]\s*([\d\.]+)%",
    "MOTP"           : r"MOTP[:：]\s*([\d\.]+)%",
    "IDF1"           : r"IDF1[:：]\s*([\d\.]+)%",
    "IDP"            : r"ID Precision.*?[:：]\s*([\d\.]+)%",
    "IDR"            : r"ID Recall.*?[:：]\s*([\d\.]+)%",
    "ID Switches"    : r"ID Switches[:：]\s*(\d+)",
    "Matches"        : r"Matches[:：]\s*(\d+)",
    "False Positives": r"False Positives[:：]\s*(\d+)",
    "Misses"         : r"Misses[:：]\s*(\d+)",
    "FAF"            : r"FAF.*?[:：]\s*([\d\.]+)",
    "Precision"      : r"^Precision[:：]\s*([\d\.]+)%",
    "Recall"         : r"^Recall[:：]\s*([\d\.]+)%",
    "Mostly Tracked" : r"Mostly Tracked[:：]\s*(\d+)",
    "Partially Tracked": r"Partially Tracked[:：]\s*(\d+)",
    "Mostly Lost"    : r"Mostly Lost[:：]\s*(\d+)",
    "left_Counting"           : r"左向计数[:：]\s*(\d+)",
    "right_Counting"           : r"右向计数[:：]\s*(\d+)",
    "total_Counting"           : r"总计数[:：]\s*(\d+)",
    "actual_Counting"         : r"实际人数[:：]?\s*(\d+)",
}

# 列表顺序
COLUMN_ORDER = list(PATTERNS.keys()) + ["Relative Counting Error Rate"]

def parse_terminallog(file_path: Path) -> dict[str, object]:
    """解析单个 terminallog.txt，返回指标字典。如果缺失返回空 dict。"""
    if not file_path.is_file():
        return {}

    metrics: dict[str, object] = {}
    with file_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            for key, regex in PATTERNS.items():
                if key in metrics:
                    continue  # 已提取
                m = re.search(regex, line)
                if m:
                    value = m.group(1)
                    # 类型转换
                    if key in ("MOTA", "MOTP", "IDF1", "IDP", "IDR", "Precision", "Recall"):
                        metrics[key] = float(value)
                    elif key in ("FAF",):
                        metrics[key] = float(value)
                    elif value.isdigit():
                        metrics[key] = int(value)
                    else:
                        metrics[key] = value
    
    # 如果没有提取到总计数，但有左计数和右计数，则计算总计数
    if "总计数" not in metrics and "左计数" in metrics and "右计数" in metrics:
        metrics["总计数"] = metrics["左计数"] + metrics["右计数"]
    
    return metrics

def analyze_results(collection_dir: Path) -> pd.DataFrame:
    """分析收集目录中的所有结果文件夹，生成汇总表"""
    rows = []
    for path in collection_dir.glob("*results*"):
        if not path.is_dir():
            continue
        log = path / "terminallog.txt"
        data = parse_terminallog(log)
        if data:
            # 若文件夹列丢失，则用目录名（统一使用英文列名）
            data.setdefault("Folder", path.name)
            
            # ---- 从文件夹名解析实际人数 actual_Counting ----
            if "actual_Counting" not in data:
                # 文件夹名示例: 001_gt_25_clip_right_4_results-xxx 或 _11_results
                m = re.search(r"_(\d+)_results", path.name)
                if m:
                    data["actual_Counting"] = int(m.group(1))
                    print(f"从文件夹名 {path.name} 提取实际人数: {data['actual_Counting']}")

            # ---- 确保 total_Counting 正确 ----
            if "total_Counting" not in data and "left_Counting" in data and "right_Counting" in data:
                data["total_Counting"] = data["left_Counting"] + data["right_Counting"]
                print(f"计算总计数: {data['left_Counting']} + {data['right_Counting']} = {data['total_Counting']}")

            # ---- 计算 Relative Counting Error Rate ----
            if "total_Counting" in data and "actual_Counting" in data and data["actual_Counting"] and data["actual_Counting"] > 0:
                err = round(abs(data["total_Counting"] - data["actual_Counting"]) / data["actual_Counting"] * 100, 2)
                data["Relative Counting Error Rate"] = err
                print(f"计算计数误差率: |{data['total_Counting']} - {data['actual_Counting']}| / {data['actual_Counting']} * 100% = {err}%")
            
            rows.append(data)
    
    df = pd.DataFrame(rows)
    if not df.empty:
        # 按预设列顺序排列
        df = df.reindex(columns=COLUMN_ORDER)
    return df

def save_analysis_results(df: pd.DataFrame, collection_dir: Path):
    """保存分析结果到CSV和Excel文件"""
    if df.empty:
        print("⚠️ 未找到任何有效的 terminallog.txt 文件")
        return
    
    # 计算平均值（仅数值列）
    numeric_cols = df.select_dtypes("number").columns
    avg: pd.Series | None = None
    if len(numeric_cols):
        avg = df[numeric_cols].mean(numeric_only=True).round(2)
    
    # 准备保存的数据框（包含平均值行）
    df_to_save = df.copy()
    # 如果存在旧的 "文件夹" 列，删除之
    if "文件夹" in df_to_save.columns:
        df_to_save = df_to_save.drop(columns=["文件夹"])

    df_to_save = df_to_save.reindex(columns=COLUMN_ORDER)  # 确保列顺序一致

    # ----- 追加平均值行 -----
    if avg is not None:
        avg_row = {col: (avg[col] if col in numeric_cols else None) for col in df_to_save.columns}
        avg_row["Folder"] = "Average Value"
        df_to_save = pd.concat([df_to_save, pd.DataFrame([avg_row])], ignore_index=True)
    
    # 导出前将所有数值列四舍五入
    numeric_cols_to_save = df_to_save.select_dtypes("number").columns
    if len(numeric_cols_to_save):
        df_to_save[numeric_cols_to_save] = df_to_save[numeric_cols_to_save].round(2)
    
    # 保存CSV文件
    # csv_path = collection_dir / "结果统计汇总.csv"
    # df_to_save.to_csv(csv_path, index=False, encoding="utf-8-sig")
    
    # 保存Excel文件
    excel_path = collection_dir / "新的2_MOT评估结果统计汇总.xlsx"
    df_to_save.to_excel(excel_path, index=False)
    
    print(f"\n=== 结果统计汇总 ===")
    try:
        print(df.to_markdown(index=False, floatfmt=".2f"))
    except ImportError:
        # tabulate 未安装时的备用方案
        print(df)
    
    if avg is not None:
        print("\n=== 平均值 ===")
        for k, v in avg.items():
            print(f"{k}: {v:.2f}")
    
    print(f"\n统计汇总已保存:")
    # print(f"  CSV: {csv_path}")
    print(f"  Excel: {excel_path}")

# ============ 原有的循环处理逻辑 ============

excel = Path(EXCEL_PATH).expanduser().resolve()
if not excel.exists():
    raise FileNotFoundError(excel)

# 创建统一的结果收集目录
excel_stem = excel.stem  # 获取文件名（不含扩展名）
collection_dir = excel.parent / f"{excel_stem}_循环处理结果"

# 如果目录已存在，先清空
if collection_dir.exists():
    print(f"检测到结果目录已存在，正在清空: {collection_dir}")
    shutil.rmtree(collection_dir)
    print("  ✓ 已清空旧的结果目录")

collection_dir.mkdir(exist_ok=True)
print(f"结果收集目录: {collection_dir}")

df = pd.read_excel(excel, header=0, usecols=[0,1,2,3])  # 首行作为列名
df = df.iloc[:, :4]
df.columns = ["F_PIXELS","CX","CY","FOLDER"]

# 将数值列转 float，忽略非数字
for col in ["F_PIXELS","CX","CY"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

print(f"读取到 {len(df)} 条任务，开始批量处理 ...")

pattern_map = {
    "F_PIXELS": re.compile(r"^F_PIXELS\s*[,=]"),
    "CX":       re.compile(r"^CX\s*[,=]"),
    "CY":       re.compile(r"^CY\s*[,=]"),
    "FOLDER":   re.compile(r"^FOLDER\s*="),
}

# 记录处理结果
processing_results = []

for idx, row in df.iterrows():
    f_pix, cx, cy, folder = row
    folder = str(folder).strip()
    print(f"[{idx+1}/{len(df)}] FOLDER={folder}")

    # 读原脚本
    src_text = RUN_SCRIPT.read_text(encoding="utf-8")
    new_lines = []
    orig_dir_str = str(RUN_SCRIPT.parent)
    for line in src_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("SCRIPT_DIR ") and "Path(" in stripped:
            new_lines.append(f"SCRIPT_DIR = Path(r'{orig_dir_str}')  # 固定原始路径")
            new_lines.append(f"PROJECT_ROOT = SCRIPT_DIR.parent  # 保持不变")
            new_lines.append(f"INNER_PKG_DIR = SCRIPT_DIR / 'deep_sort'")
            continue
        if stripped.startswith("F_PIXELS, CX, CY") and "=" in stripped:
            new_lines.append(f"F_PIXELS, CX, CY = {f_pix}, {cx}, {cy}  # 参数自动注入")
            continue
        if stripped.startswith("FOLDER") and "=" in stripped:
            new_lines.append(f"FOLDER = \"{folder}\"  # 参数自动注入")
            continue
        new_lines.append(line)
    temp_dir = Path(tempfile.mkdtemp(prefix="h5d_batch_"))
    temp_script = temp_dir / "Run_3DModel_H5D_temp.py"
    temp_script.write_text("\n".join(new_lines), encoding="utf-8")

    # 调用
    log_file = temp_dir / "run.log"
    with log_file.open("w", encoding="utf-8") as lf:
        proc = subprocess.run([PYTHON_BIN, str(temp_script)], stdout=lf, stderr=lf)
    
    # 处理结果
    success = proc.returncode == 0
    if success:
        print("  ✓ 完成")
        
        # 查找并移动结果文件夹
        # 1) 在与 FOLDER 同级目录下搜索匹配 "<folder>-results*" 的文件夹
        parent_dir = Path(folder).parent
        pattern = Path(folder).name + "-results*"
        candidate_dirs = [d for d in parent_dir.glob(pattern) if d.is_dir()]

        # 2) 按修改时间倒序排序，选择最新的一个
        candidate_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        if candidate_dirs:
            expected_result_dir = candidate_dirs[0]
            # 生成唯一的目标文件夹名（保持原批次序号＋源文件夹名前缀）
            folder_basename = Path(folder).name
            target_dir = collection_dir / f"{idx+1:03d}_{folder_basename}_results-H5D_latest"

            # 如果目标目录已存在，添加序号避免冲突
            counter = 1
            original_target = target_dir
            while target_dir.exists():
                target_dir = Path(str(original_target) + f"_{counter}")
                counter += 1

            try:
                shutil.move(str(expected_result_dir), str(target_dir))
                print(f"    → 结果已移动到: {target_dir.name}")
                processing_results.append({
                    "序号": idx+1,
                    "FOLDER": folder,
                    "状态": "完成",
                    "结果目录": target_dir.name
                })
            except Exception as e:
                print(f"    ✗ 移动结果文件夹失败: {e}")
                processing_results.append({
                    "序号": idx+1,
                    "FOLDER": folder,
                    "状态": "移动失败",
                    "结果目录": f"原位置: {expected_result_dir}"
                })
        else:
            print("    ⚠ 未找到任何匹配的结果文件夹 (pattern: {} )".format(pattern))
            processing_results.append({
                "序号": idx+1,
                "FOLDER": folder,
                "状态": "结果缺失",
                "结果目录": "未找到"
            })
    else:
        print("  ✗ 失败，详见", log_file)
        processing_results.append({
            "序号": idx+1,
            "FOLDER": folder,
            "状态": "处理失败",
            "结果目录": f"日志: {log_file}"
        })

print("全部任务处理完毕！")

# # 生成详细的处理结果汇总
# summary_df = pd.DataFrame(processing_results)
# summary_path = collection_dir / "处理结果汇总.xlsx"
# summary_df.to_excel(summary_path, index=False)
# print(f"处理结果汇总表已保存到: {summary_path}")

# 输出统计信息
total_tasks = len(df)
completed_tasks = len([r for r in processing_results if r["状态"] == "完成"])
failed_tasks = total_tasks - completed_tasks

print(f"\n=== 处理统计 ===")
print(f"总任务数: {total_tasks}")
print(f"成功完成: {completed_tasks}")
print(f"失败任务: {failed_tasks}")
print(f"成功率: {completed_tasks/total_tasks*100:.1f}%")
print(f"所有结果文件夹已收集到: {collection_dir}")

# ============ 自动进行结果统计分析 ============
print(f"\n开始分析收集到的结果文件夹...")
analysis_df = analyze_results(collection_dir)
save_analysis_results(analysis_df, collection_dir)

print(f"\n🎉 全部流程完成！所有文件都保存在: {collection_dir}") 