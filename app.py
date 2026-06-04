import os
import sys
import subprocess
import time
import threading
import signal
import csv
import json
import random
import math
import re
from pathlib import Path

import gradio as gr
import numpy as np
import cv2
from ultralytics import YOLO
import plotly.graph_objects as go

# 导入自定义模块
from sam_config import sam_status
from sam_leaf_segment import segment_leaf, segment_from_boxes, analyze_disease_extent
from multimodal_api import call_vision_api, MODELS, DEFAULT_API_KEY, run_llm_leaf_diagnosis

# --------------------------------------------------------------------------------
# 全局配置与状态
# --------------------------------------------------------------------------------
ROOT = Path(__file__).parent
DATA_YAML = str(ROOT / "datasets" / "data.yaml")
# Ultralytics 默认训练输出根；误写成 runs/detect/runs/detect/... 时也会被下方 rglob 扫到
RUNS_DETECT_DIR = ROOT / "runs" / "detect"
# 诊断页 Plotly 图默认高度参考（条形图/空图等）
DIAGNOSIS_PLOTLY_HEIGHT = 480
# 与 datasets/data.yaml 中 names 一致；未命中时界面仍显示英文类名
DISEASE_NAME_ZH = {
    "Rust": "锈病",
    "Mosaic": "花叶病",
    "Grey_spot": "灰斑病",
    "Brown_Spot": "褐斑病",
    "Alternaria_Boltch": "链格孢叶斑病",
}


def _disease_display_name(en_key: str) -> str:
    """用于图上文字：中文（英文键）。"""
    zh = DISEASE_NAME_ZH.get(en_key)
    if zh:
        return f"{zh} ({en_key})"
    return str(en_key)


def _disease_cn_only(en_key: str) -> str:
    return DISEASE_NAME_ZH.get(en_key, en_key)


def _build_llm_detector_context(
    detections: dict,
    disease_conf_sum: dict,
    disease_conf_count: dict,
    sam_profile: dict | None,
    sam_quant_attempted: bool,
) -> str:
    """供 LLM 第二轮 / 文本模型使用的 YOLO + SAM 结构化文字摘要。"""
    lines: list[str] = []
    lines.append("【YOLO】")
    if not detections:
        lines.append("无检出框。")
    else:
        for name, cnt in sorted(detections.items(), key=lambda x: (-x[1], x[0])):
            n = disease_conf_count.get(name, 0)
            mean_c = disease_conf_sum.get(name, 0.0) / max(n, 1)
            zh = _disease_cn_only(name)
            lines.append(
                f"- 类名(英): {name} | 中文: {zh} | 框数: {cnt} | 置信度均值: {mean_c:.4f}"
            )
    lines.append("")
    lines.append("【SAM 量化】")
    if not sam_quant_attempted:
        lines.append("本轮未执行 SAM（用户关闭或未进入 SAM 流程）。叶内病斑比等可能为占位或缺失。")
    elif not sam_profile:
        lines.append("已尝试 SAM 量化但未得到有效 profile（可能失败）。")
    else:
        lf = sam_profile.get("leaf_frame_confidence")
        lines.append(f"- leaf_frame_confidence（叶片在画面中占比相关，0~1）: {lf}")
        byd = sam_profile.get("by_disease") or {}
        for dname, dm in byd.items():
            lines.append(
                f"- 类 {dname}: 叶内病斑比={float(dm.get('leaf_lesion_ratio', 0)):.4f}, "
                f"框掩膜一致性={float(dm.get('box_mask_consistency', 0)):.4f}, "
                f"病斑像素={int(dm.get('lesion_area_px', 0))}, 叶片像素参考={int(dm.get('leaf_area_px', 0))}"
            )
    return "\n".join(lines)


def _llm_diagnosis_to_markdown(data: object) -> str:
    """将 smart_diagnosis 第 5 路 dict 转成易读 Markdown（与 gr.JSON 同源）。"""
    if not isinstance(data, dict):
        return "（暂无结构化输出）"

    def _line(label: str, val: object) -> str:
        if val is None or val == "":
            return ""
        s = str(val).strip()
        if not s:
            return ""
        return f"- **{label}**：{s}\n"

    mode = data.get("mode")
    tip = data.get("说明")
    err = data.get("error")
    lines: list[str] = []

    mode_zh = {
        "no_image": "未上传图像",
        "no_weights": "无检测权重",
        "pending": "等待推理",
        "pending_no_yolo": "等待 LLM（无检测框）",
        "no_detector_no_llm": "未检出且未开 LLM",
        "loading": "LLM 请求中",
        "llm_disabled": "未启用 LLM",
        "vision_only_no_detection": "单轮视觉（无 YOLO 框）",
        "two_round_vision": "两轮视觉（先图后融合）",
        "text_only_single": "单轮文本融合（模型不可看图）",
        "error": "异常",
    }.get(str(mode), str(mode) if mode is not None else "")

    if mode_zh:
        lines.append(f"##### 流程状态\n{mode_zh}")
    if tip and str(tip).strip():
        lines.append("\n" + _line("界面提示", tip).rstrip("\n"))
    if err:
        lines.append("\n" + _line("错误", err).rstrip("\n"))
        det = data.get("detail")
        if det:
            lines.append(_line("详情", det).rstrip("\n"))

    warns = data.get("warnings")
    if isinstance(warns, list) and warns:
        lines.append("\n##### 系统提示")
        for w in warns:
            if str(w).strip():
                lines.append(f"\n- {w}")

    r1 = data.get("round1")
    r2 = data.get("round2")

    if isinstance(r1, dict) and r1:
        lines.append("\n##### 第一轮：仅依据图像（模型尚不知道 YOLO/SAM）")
        cands = r1.get("visual_candidates")
        if isinstance(cands, list) and cands:
            lines.append("\n| 可能病害 | 可能性 | 简要依据 |\n| --- | --- | --- |")
            for c in cands[:6]:
                if not isinstance(c, dict):
                    continue
                nz = str(c.get("name_zh", "")).replace("|", "｜")
                lk = str(c.get("likelihood", "")).replace("|", "｜")
                bb = str(c.get("brief_basis", "")).replace("|", "｜")
                lines.append(f"\n| {nz} | {lk} | {bb} |")
        lines.append("\n" + _line("可见症状", r1.get("symptoms_observed")).rstrip("\n"))
        lines.append(_line("不确定性", r1.get("uncertainty")).rstrip("\n"))
        if r1.get("_parse_error") or r1.get("_raw"):
            lines.append("\n" + _line("解析备注", r1.get("_parse_error") or "见 JSON 中 _raw").rstrip("\n"))

    if isinstance(r2, dict) and r2:
        if mode == "vision_only_no_detection" or r2.get("mode") == "no_yolo_boxes":
            lines.append("\n##### 单轮结论（无检测框时的视觉 JSON）")
        elif mode == "text_only_single" or r2.get("mode") == "text_only_model":
            lines.append("\n##### 单轮结论（纯文本模型，仅算法摘要）")
        else:
            lines.append("\n##### 第二轮：图像 + YOLO/SAM 交叉验证结论")
        lines.append("\n" + _line("症状综合", r2.get("symptoms")).rstrip("\n"))
        lines.append(_line("严重程度", r2.get("severity")).rstrip("\n"))
        lines.append(_line("产量/品质影响", r2.get("loss_estimate")).rstrip("\n"))
        lines.append(_line("防治建议", r2.get("treatment")).rstrip("\n"))
        lines.append(_line("预防", r2.get("prevention")).rstrip("\n"))
        lines.append(_line("交叉验证说明", r2.get("cross_validation")).rstrip("\n"))
        lines.append(_line("与 YOLO 一致性", r2.get("agreement_with_yolo")).rstrip("\n"))
        lines.append(_line("与 SAM 量化一致性", r2.get("agreement_with_sam_quant")).rstrip("\n"))
        lines.append(_line("备注", r2.get("note")).rstrip("\n"))
        if r2.get("_parse_error") or r2.get("_raw"):
            lines.append("\n" + _line("解析备注", r2.get("_parse_error") or "见 JSON 中 _raw").rstrip("\n"))

    out = "".join(lines).strip()
    return out if out else "（空摘要）"


def _smart_diag_row(yolo, sam, mh, mf, ai: dict, log: str):
    """诊断流式输出的一行：与界面 outputs 顺序一致（含易读 Markdown）。"""
    return (yolo, sam, mh, mf, _llm_diagnosis_to_markdown(ai), ai, log)


def _extract_base_model_tag(base_weight):
    raw = Path(str(base_weight)).name.lower()
    m = re.search(r"(yolov\d+[a-z]|yolo\d+[a-z])", raw)
    if not m:
        return None
    token = m.group(1)
    if token.startswith("yolov"):
        return f"v{token[len('yolov'):]}"
    if token.startswith("yolo"):
        return f"v{token[len('yolo'):]}"
    return None


def _display_run_name(run_name, base_weight):
    tag = _extract_base_model_tag(base_weight)
    if not tag:
        return run_name

    normalized_name = str(run_name).lower()
    if tag in normalized_name:
        return run_name

    if normalized_name == "train" or normalized_name.startswith("train_"):
        return f"train_{tag}"
    return run_name

# 扫描已训练模型（递归 runs/detect，兼容误嵌套的 runs/detect/runs/detect/...）
def _scan_trained_models():
    options = []
    base = RUNS_DETECT_DIR
    if not base.is_dir():
        return options
    seen = set()
    for weights in sorted(base.rglob("weights/best.pt")):
        try:
            key = str(weights.resolve())
        except Exception:
            key = str(weights)
        if key in seen:
            continue
        seen.add(key)
        run_dir = weights.parent.parent
        run_name = run_dir.name
        base_weight = ""
        args_path = run_dir / "args.yaml"
        if args_path.exists():
            try:
                with open(args_path, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        s = line.strip()
                        if s.startswith("model:"):
                            base_weight = s.split(":", 1)[1].strip().strip("'").strip('"')
                            break
            except Exception:
                base_weight = ""
        display_name = _display_run_name(run_name, base_weight)
        options.append((display_name, str(weights)))

    # 新训练在上：按权重文件修改时间倒序
    options.sort(key=lambda t: Path(t[1]).stat().st_mtime, reverse=True)
    return options


MODEL_OPTIONS = _scan_trained_models()

# --------------------------------------------------------------------------------
# 模型评估数据读取与可视化
# --------------------------------------------------------------------------------
def _parse_simple_yaml(yaml_path: Path):
    parsed = {}
    if not yaml_path.exists():
        return parsed
    with open(yaml_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            key, value = line.split(":", 1)
            parsed[key.strip()] = value.strip().strip("'").strip('"')
    return parsed


def _to_float(value, default=0.0):
    try:
        v = float(value)
        if not math.isfinite(v):
            return default
        return v
    except Exception:
        return default


def _to_int(value, default=0):
    try:
        return int(float(value))
    except Exception:
        return default


_MODEL_PARAM_CACHE = {}


def _estimate_params_million(local_weight_path):
    if not local_weight_path:
        return None
    key = str(local_weight_path)
    if key in _MODEL_PARAM_CACHE:
        return _MODEL_PARAM_CACHE[key]
    try:
        weight = Path(local_weight_path)
        if not weight.exists():
            return None
        yolo = YOLO(str(weight))
        params = sum(p.numel() for p in yolo.model.parameters())
        val = float(params) / 1e6
        _MODEL_PARAM_CACHE[key] = val
        return val
    except Exception:
        return None


def _model_fixed_color(model_name, base_weight, fallback_palette, fallback_idx):
    text = f"{model_name} {base_weight}".lower()
    if "v8x" in text or "yolov8x" in text:
        return "#22C55E"  # green
    if "11n" in text or "yolo11n" in text:
        return "#3B82F6"  # blue
    if "11m" in text or "yolo11m" in text:
        return "#F59E0B"  # yellow/orange
    if "v8m" in text or "yolov8m" in text:
        return "#A855F7"  # purple
    return fallback_palette[fallback_idx % len(fallback_palette)]


def _collect_training_run_dirs():
    """在 runs/detect 下递归查找 results.csv，返回对应训练目录列表（按修改时间倒序）。"""
    if not RUNS_DETECT_DIR.is_dir():
        return []
    seen = set()       # 已见过的目录，防重复
    rows = []
    for csv_path in RUNS_DETECT_DIR.rglob("results.csv"):  # rglob：递归搜索所有子文件夹
        run_dir = csv_path.parent                          # results.csv 的上一级即一次训练 run 目录
        if not run_dir.is_dir():
            continue
        try:
            key = str(run_dir.resolve())
        except Exception:
            key = str(run_dir)
        if key in seen:
            continue
        seen.add(key)
        try:
            mtime = run_dir.stat().st_mtime                # 目录最后修改时间，用于排序
        except OSError:
            mtime = 0.0
        rows.append((mtime, run_dir))
    rows.sort(key=lambda x: x[0], reverse=True)             # 新的训练排在前面
    return [r[1] for r in rows]                            # 只返回目录路径，不要时间戳


def _resolve_val_image_paths(sample_count=100):
    """
    收集测速用的验证集图片路径列表。
    sample_count: 最多用多少张图；验证集更多时会随机抽取。
    """
    # 读取 data.yaml（数据集配置文件），得到 val、path 等字段
    data_cfg = _parse_simple_yaml(Path(DATA_YAML))
    val_entry = data_cfg.get("val", "")      # 验证集路径，可能是文件夹或 .txt 列表
    data_root = data_cfg.get("path", "")     # 数据集根目录
    yaml_parent = Path(DATA_YAML).parent     # data.yaml 所在文件夹

    def _candidate_bases():
        """列出可能的数据集根路径，兼容相对/绝对路径写法。"""
        candidates = []
        if data_root:
            raw = Path(data_root)
            if raw.is_absolute():
                candidates.append(raw)  # 已是绝对路径，直接用
            else:
                # 相对路径：分别尝试「相对 yaml 目录」「相对项目根」「当前目录」
                candidates.append((yaml_parent / raw).resolve())
                candidates.append((ROOT / raw).resolve())
                candidates.append(raw.resolve())
        candidates.append(yaml_parent.resolve())
        candidates.append(ROOT.resolve())

        # 去重，避免同一目录重复加入
        seen = set()
        uniq = []
        for c in candidates:
            cs = str(c)
            if cs not in seen:
                seen.add(cs)
                uniq.append(c)
        return uniq

    bases = _candidate_bases()

    # 根据 val 字段拼出所有可能的验证集路径
    val_candidates = []
    if val_entry:
        v = Path(val_entry)
        if v.is_absolute():
            val_candidates.append(v)
        else:
            for b in bases:
                val_candidates.append((b / v).resolve())
    else:
        # data.yaml 没写 val 时，默认找 images/val
        for b in bases:
            val_candidates.append((b / "images" / "val").resolve())

    # 取第一个真实存在的路径作为验证集位置
    val_path = None
    for c in val_candidates:
        if c.exists():
            val_path = c
            break
    if val_path is None:
        val_path = val_candidates[0] if val_candidates else (yaml_parent / "images" / "val")

    image_paths = []
    if val_path.is_file() and val_path.suffix.lower() == ".txt":
        # val 是文本文件：每行一张图片的路径
        with open(val_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                p = line.strip()           # 去掉首尾空白
                if not p:
                    continue               # 空行跳过
                img = Path(p)
                if not img.is_absolute():
                    img = (val_path.parent / img).resolve()  # 相对路径转绝对路径
                if img.exists():
                    image_paths.append(img)
    elif val_path.is_dir():
        # val 是文件夹：递归找所有常见图片后缀
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
        image_paths = [p for p in val_path.rglob("*") if p.suffix.lower() in exts]

    image_paths = sorted(set(image_paths))   # 去重并排序
    if not image_paths:
        return []                            # 没找到任何图片

    sample_count = max(1, int(sample_count)) # 至少 1 张
    if len(image_paths) > sample_count:
        rng = random.Random(42)              # 固定随机种子，每次抽到同一批图
        image_paths = sorted(rng.sample(image_paths, sample_count))
    return image_paths


def benchmark_model_inference(device, imgsz, conf, sample_count, warmup_count):
    """
    对 runs/detect 下每个训练好的模型做推理测速。
    参数来自界面：device 设备、imgsz 输入尺寸、conf 置信度、
    sample_count 测速图片数、warmup_count 预热次数。
    每个模型结果写入其目录下的 benchmark.json。
    """
    # 扫描 runs/detect，收集所有含 results.csv 的训练目录
    run_dirs = _collect_training_run_dirs()
    if not run_dirs:
        return "⚠️ 未找到可测速模型（需在 runs/detect 下任意层级的 results.csv）"

    # 从验证集取测速图片；失败则无法继续
    image_paths = _resolve_val_image_paths(sample_count=sample_count)
    if not image_paths:
        return "⚠️ 未找到验证集图像，请检查 data.yaml 的 val 路径"

    sample_count = len(image_paths)                              # 实际使用的图片数量
    warmup_count = max(0, min(int(warmup_count), sample_count))  # 预热次数限制在 0~样本数
    device_str = str(device).strip() if device is not None else "0"  # "cuda:0" 或 "cpu"
    conf_val = float(conf)       # 检测置信度阈值，如 0.25
    imgsz_val = int(imgsz)       # 输入边长，如 640

    updated, failed = 0, 0       # 成功/失败计数
    for run_dir in run_dirs:     # 逐个训练目录测速
        # 优先用 best.pt（验证集最优），没有则用 last.pt（最后一轮）
        weight_path = run_dir / "weights" / "best.pt"
        if not weight_path.exists():
            weight_path = run_dir / "weights" / "last.pt"
        if not weight_path.exists():
            failed += 1
            continue               # 没有权重文件，跳过该模型

        try:
            model = YOLO(str(weight_path))  # 加载 YOLO 模型

            # 预热：跑 warmup_count 次 predict，不计入下面的耗时统计
            # 作用是让 GPU 完成显存分配、内核编译等，避免首帧偏慢
            for i in range(warmup_count):
                model.predict(
                    source=str(image_paths[i]),  # 第 i 张测速图
                    imgsz=imgsz_val,
                    conf=conf_val,
                    device=device_str,
                    verbose=False,               # 不打印日志
                )

            # 四个列表：分别存每张图的预处理、推理、后处理、总耗时（毫秒）
            pre_list, inf_list, post_list, total_list = [], [], [], []
            wall_start = time.perf_counter()     # 记录整段测速开始时刻（秒）
            for img_path in image_paths:
                results = model.predict(
                    source=str(img_path),
                    imgsz=imgsz_val,
                    conf=conf_val,
                    device=device_str,
                    verbose=False,
                )
                # predict 返回列表，取第一张图的结果；speed 是耗时字典（单位 ms）
                speed = results[0].speed if results else {}
                pre = float(speed.get("preprocess", 0.0))    # 读图、缩放、归一化
                inf = float(speed.get("inference", 0.0))     # 神经网络前向计算
                post = float(speed.get("postprocess", 0.0))   # 解码框、NMS 等
                total = pre + inf + post                     # 单张图总耗时
                pre_list.append(pre)
                inf_list.append(inf)
                post_list.append(post)
                total_list.append(total)
            wall_ms = (time.perf_counter() - wall_start) * 1000.0  # 整段墙钟耗时转毫秒

            if total_list:
                total_ms = float(np.mean(total_list))              # 所有样本总耗时的平均值
                p95_total_ms = float(np.percentile(total_list, 95))  # 95% 样本不超过的耗时
            else:
                total_ms, p95_total_ms = 0.0, 0.0

            # FPS = 每秒处理张数；1e-9 防止除以 0
            fps = (1000.0 / total_ms) if total_ms > 1e-9 else 0.0

            # 组装测速结果，写入 JSON 供界面表格读取
            benchmark_data = {
                "version": 1,
                "device": device_str,
                "imgsz": imgsz_val,
                "conf": conf_val,
                "sample_count": sample_count,
                "warmup_count": warmup_count,
                "wall_ms": round(wall_ms, 3),                    # 整批测速实际耗时
                "mean_preprocess_ms": round(float(np.mean(pre_list)) if pre_list else 0.0, 4),
                "mean_inference_ms": round(float(np.mean(inf_list)) if inf_list else 0.0, 4),
                "mean_postprocess_ms": round(float(np.mean(post_list)) if post_list else 0.0, 4),
                "mean_total_ms": round(total_ms, 4),             # 界面「推理总耗时」列
                "p95_total_ms": round(p95_total_ms, 4),          # 界面「P95推理耗时」列
                "fps": round(fps, 3),                            # 界面「FPS」列
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            bench_path = run_dir / "benchmark.json"
            with open(bench_path, "w", encoding="utf-8") as f:
                json.dump(benchmark_data, f, ensure_ascii=False, indent=2)
            updated += 1
        except Exception:
            failed += 1              # 加载或推理出错，记失败

    return f"✅ 识别测速完成：成功 {updated} 个模型，失败 {failed} 个模型（样本 {sample_count} 张，设备 {device_str}）"


def _apply_3d_scene(fig, title_text, x_title, y_title, z_title, x_range=None, y_range=None, z_range=None):
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                title=dict(text=x_title, font=dict(color="#0f172a")),
                backgroundcolor="rgba(10,12,16,0.0)",
                gridcolor="rgba(240,240,240,0.16)",
                zerolinecolor="rgba(240,240,240,0.2)",
                showspikes=False,
                range=x_range,
                tickfont=dict(color="#1e293b")
            ),
            yaxis=dict(
                title=dict(text=y_title, font=dict(color="#0f172a")),
                backgroundcolor="rgba(10,12,16,0.0)",
                gridcolor="rgba(240,240,240,0.16)",
                zerolinecolor="rgba(240,240,240,0.2)",
                showspikes=False,
                range=y_range,
                tickfont=dict(color="#1e293b")
            ),
            zaxis=dict(
                title=dict(text=z_title, font=dict(color="#0f172a")),
                backgroundcolor="rgba(10,12,16,0.0)",
                gridcolor="rgba(240,240,240,0.16)",
                zerolinecolor="rgba(240,240,240,0.2)",
                showspikes=False,
                range=z_range,
                tickfont=dict(color="#1e293b")
            ),
            camera=dict(eye=dict(x=1.7, y=1.25, z=1.2)),
            aspectmode="manual",
            aspectratio=dict(x=1.5, y=1.0, z=1.15),
        ),
        margin=dict(l=0, r=0, b=0, t=36),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        title=dict(text=f"<b>{title_text}</b>", x=0.5, font=dict(size=18, color="#0f172a")),
        font=dict(color="#0f172a")
    )


def _hide_3d_scene_axes(fig):
    """与模型多维画像一致：不显示 Plotly 默认的 XYZ 轴刻度与网格，仅保留自定义几何与文字。"""
    fig.update_scenes(
        xaxis_showticklabels=False,
        xaxis_ticks="",
        xaxis_showgrid=False,
        xaxis_zeroline=False,
        yaxis_showticklabels=False,
        yaxis_ticks="",
        yaxis_showgrid=False,
        yaxis_zeroline=False,
        zaxis_showticklabels=False,
        zaxis_ticks="",
        zaxis_showgrid=False,
        zaxis_zeroline=False,
    )


# 未启用 SAM 时与旧版 3D 雷达一致的占位辐条刻度（条形图横轴 0～1）
_SAM_METRIC_PLACEHOLDER = 0.38


def _collect_disease_metric_rows(
    detections: dict,
    disease_conf_sum: dict,
    disease_conf_count: dict,
    sam_profile: dict | None,
) -> list[dict]:
    if not detections:
        return []
    max_count = max(detections.values()) if detections else 1
    leaf_frame = float((sam_profile or {}).get("leaf_frame_confidence", 0.42))
    by_dis = (sam_profile or {}).get("by_disease") or {}
    active = bool(sam_profile)
    rows: list[dict] = []
    ordered = sorted(detections.items(), key=lambda x: x[1], reverse=True)
    ph = _SAM_METRIC_PLACEHOLDER
    for disease, count in ordered:
        conf_mean = 0.0
        if disease_conf_count.get(disease, 0) > 0:
            conf_mean = disease_conf_sum.get(disease, 0.0) / max(disease_conf_count[disease], 1)
        conf_mean = min(max(conf_mean, 0.0), 1.0)
        if active:
            dm = by_dis.get(disease, {}) or {}
            raw_lr = float(dm.get("leaf_lesion_ratio", 0.0))
            raw_lr = min(1.0, max(0.0, raw_lr))
            box_cons = float(dm.get("box_mask_consistency", 0.0))
            box_cons = min(1.0, max(0.0, box_cons))
            lf = min(1.0, max(0.0, leaf_frame))
            bar_lr, bar_lf, bar_bc = raw_lr, lf, box_cons
            lesion_px = int(dm.get("lesion_area_px", 0))
            leaf_px = int(dm.get("leaf_area_px", 0))
        else:
            raw_lr = 0.0
            box_cons = 0.0
            lf = 0.0
            bar_lr, bar_lf, bar_bc = ph, ph, ph
            lesion_px = 0
            leaf_px = 0
        rows.append(
            {
                "disease": disease,
                "display": _disease_display_name(disease),
                "count": int(count),
                "count_norm": min(1.0, count / max(max_count, 1)),
                "conf_mean": conf_mean,
                "raw_leaf_ratio": raw_lr,
                "leaf_frame": lf,
                "box_cons": box_cons,
                "bar_leaf_ratio": bar_lr,
                "bar_leaf_frame": bar_lf,
                "bar_box_cons": bar_bc,
                "lesion_px": lesion_px,
                "leaf_px": leaf_px,
                "sam_active": active,
            }
        )
    return rows


# 诊断页条形图：白底 + 黑字
_METRICS_CHART_TEXT = "#0f172a"
_METRICS_CHART_BG = "#ffffff"
# 量化指标 HTML 区
_METRICS_HTML_TEXT = "#0f172a"
_METRICS_HTML_MUTED = "#334155"
_METRICS_HTML_CARD_BG = "#f8fafc"


def _wrap_metrics_html(inner: str) -> str:
    """量化指标 HTML：统一白底黑字容器。"""
    return (
        f'<div class="metrics-html-zone" style="background:#ffffff;color:{_METRICS_HTML_TEXT};'
        "border:1px solid #e2e8f0;border-radius:8px;padding:12px 14px;"
        'margin-top:8px;line-height:1.55;font-size:0.95rem;">'
        f"{inner}</div>"
    )


def _metrics_html_placeholder(message: str) -> str:
    return _wrap_metrics_html(f'<p style="margin:0;color:{_METRICS_HTML_TEXT};">{message}</p>')


def _empty_disease_metrics_fig(message: str = "暂无数据") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        text=message,
        showarrow=False,
        font=dict(size=15, color=_METRICS_CHART_TEXT),
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(
        height=max(220, int(DIAGNOSIS_PLOTLY_HEIGHT * 0.45)),
        margin=dict(l=16, r=16, t=28, b=16),
        paper_bgcolor="#ffffff",
        plot_bgcolor=_METRICS_CHART_BG,
        font=dict(color=_METRICS_CHART_TEXT),
    )
    return fig


def _build_disease_metrics_card_html(rows: list[dict]) -> str:
    if not rows:
        return _metrics_html_placeholder("尚无检出或指标。")
    parts = [
        f'<div style="display:flex;flex-wrap:wrap;gap:10px;color:{_METRICS_HTML_TEXT};">'
    ]
    for r in rows:
        if r["sam_active"]:
            sam_metrics_html = (
                f'<div style="font-size:0.88rem;color:{_METRICS_HTML_MUTED};margin-top:6px;line-height:1.45;">'
                f'叶内病斑 <b style="color:{_METRICS_HTML_TEXT}">{r["raw_leaf_ratio"]*100:.1f}%</b> ({r["lesion_px"]:,}px / {r["leaf_px"]:,}px)<br>'
                f'叶片取景 <b style="color:{_METRICS_HTML_TEXT}">{r["leaf_frame"]*100:.0f}/100</b> · '
                f'框-掩膜一致 <b style="color:{_METRICS_HTML_TEXT}">{r["box_cons"]*100:.1f}%</b>'
                f"</div>"
            )
        else:
            sam_metrics_html = ""

        parts.append(
            f'<div style="flex:1;min-width:180px;border:1px solid #e2e8f0;border-radius:8px;'
            f'padding:10px 12px;background:{_METRICS_HTML_CARD_BG};">'
            f'<div style="font-weight:700;color:{_METRICS_HTML_TEXT};margin-bottom:4px;font-size:1rem;">{r["display"]}</div>'
            f'<div style="font-size:0.9rem;color:{_METRICS_HTML_MUTED};">框数 <b style="color:{_METRICS_HTML_TEXT}">{r["count"]}</b> · '
            f'置信均值 <b style="color:{_METRICS_HTML_TEXT}">{r["conf_mean"]*100:.1f}%</b></div>'
            f"{sam_metrics_html}"
            "</div>"
        )
    parts.append("</div>")
    return _wrap_metrics_html("".join(parts))


def _build_disease_metrics_bar_figure(rows: list[dict]) -> go.Figure:
    from plotly.subplots import make_subplots

    if not rows:
        return _empty_disease_metrics_fig()
        
    sam_active = any(r.get("sam_active") for r in rows)
    if sam_active:
        metric_labels = ["框数(归一化)", "置信度均值", "叶内病斑比", "叶片取景", "框-掩膜一致"]
        colors = ["#34D399", "#60A5FA", "#F472B6", "#F59E0B", "#A78BFA"]
    else:
        metric_labels = ["框数(归一化)", "置信度均值"]
        colors = ["#34D399", "#60A5FA"]
        
    n = len(rows)
    titles = [r["display"] for r in rows]
    fig = make_subplots(
        rows=n,
        cols=1,
        subplot_titles=titles,
        vertical_spacing=min(0.11, 0.06 + 0.02 * max(0, 4 - n)),
    )
    for i, r in enumerate(rows, start=1):
        if sam_active:
            xs = [
                r["count_norm"],
                r["conf_mean"],
                r["bar_leaf_ratio"],
                r["bar_leaf_frame"],
                r["bar_box_cons"],
            ]
            raw_lr = r["raw_leaf_ratio"]
            hover = [
                f"框数: {r['count']}（相对本图最大类归一化）",
                f"置信度均值: {r['conf_mean']*100:.1f}%",
                f"叶内病斑面积比: {raw_lr*100:.1f}% ({r.get('lesion_px', 0)}px / {r.get('leaf_px', 0)}px)",
                f"叶片取景分: {r['leaf_frame']*100:.0f}/100",
                f"框掩膜一致性: {r['box_cons']*100:.1f}%",
            ]
        else:
            xs = [
                r["count_norm"],
                r["conf_mean"],
            ]
            hover = [
                f"框数: {r['count']}（相对本图最大类归一化）",
                f"置信度均值: {r['conf_mean']*100:.1f}%",
            ]
            
        texts = []
        for j, v in enumerate(xs):
            if sam_active and j == 3:
                texts.append(f"{v*100:.0f}/100")
            else:
                texts.append(f"{v*100:.0f}%")
        fig.add_trace(
            go.Bar(
                x=xs,
                y=metric_labels,
                orientation="h",
                marker_color=colors,
                text=texts,
                textposition="outside",
                textfont=dict(size=11, color=_METRICS_CHART_TEXT),
                hovertext=hover,
                hoverinfo="text",
            ),
            row=i,
            col=1,
        )
        fig.update_xaxes(
            range=[0, 1.08],
            tickformat=".0%",
            row=i,
            col=1,
            gridcolor="rgba(15,23,42,0.12)",
            zeroline=False,
            tickfont=dict(color=_METRICS_CHART_TEXT, size=11),
        )
        fig.update_yaxes(
            row=i,
            col=1,
            tickfont=dict(color=_METRICS_CHART_TEXT, size=12),
        )
    h = int(min(max(240, 72 + 128 * n), 560))
    fig.update_layout(
        height=h,
        showlegend=False,
        paper_bgcolor="#ffffff",
        plot_bgcolor=_METRICS_CHART_BG,
        font=dict(color=_METRICS_CHART_TEXT, size=12),
        margin=dict(l=6, r=48, t=40 + 12 * n, b=20),
        title=dict(
            text="<b>各类病害 · 指标条形</b>",
            font=dict(size=15, color=_METRICS_CHART_TEXT),
        ),
        hoverlabel=dict(
            font=dict(size=12, color=_METRICS_CHART_TEXT),
            bgcolor="#ffffff",
            bordercolor="rgba(15,23,42,0.2)",
        ),
    )
    fig.update_annotations(font=dict(color=_METRICS_CHART_TEXT, size=13))
    return fig


def _build_model_profile_3d(records):
    fig = go.Figure()
    metric_names = [
        "mAP50",
        "mAP50-95",
        "精确率 (Precision)",
        "召回率 (Recall)",
        "识别速度得分",
        "部署成本得分",
        "综合效率",
    ]
    # 明确起点：从左上方向开始，顺时针排布
    start_angle = np.deg2rad(125.0)
    angles = np.linspace(start_angle, start_angle - 2 * np.pi, len(metric_names), endpoint=False)
    max_radius = 200.0
    # 轴名单独画在轮辐末端之外：系数越大，轴名离多边形（100% 环）越远；与 scene range 配合避免裁切
    axis_label_radius_factor = 1.32
    layer_gap = 24.0
    fallback_palette = ["#EC4899", "#14B8A6", "#8B5CF6", "#EF4444", "#06B6D4", "#A3E635", "#F97316"]

    if not records:
        fig.add_trace(go.Scatter3d(
            x=[0], y=[0], z=[0],
            mode="markers+text",
            marker=dict(size=9, color="#A1A1AA"),
            text=["暂无模型数据"],
            textposition="top center",
            hoverinfo="skip",
            showlegend=False,
        ))
        _apply_3d_scene(
            fig,
            "模型多维画像",
            "维度 X",
            "维度 Y",
            "模型层",
            x_range=[-68, 68],
            y_range=[-68, 68],
            z_range=[0, 30],
        )
        return fig

    for ring_pct in [25, 50, 75, 100]:
        rr = (ring_pct / 100.0) * max_radius
        rx = [rr * np.cos(a) for a in angles] + [rr * np.cos(angles[0])]
        ry = [rr * np.sin(a) for a in angles] + [rr * np.sin(angles[0])]
        rz = [0.0] * (len(angles) + 1)
        fig.add_trace(go.Scatter3d(
            x=rx, y=ry, z=rz,
            mode="lines",
            line=dict(color="rgba(180,190,205,0.20)", width=2),
            hoverinfo="skip",
            showlegend=False,
        ))
        # 环形刻度标签（替代 XYZ 坐标刻度）
        fig.add_trace(go.Scatter3d(
            x=[rr + max_radius * 0.04],
            y=[0.0],
            z=[0.0],
            mode="text",
            text=[f"{ring_pct}%"],
            textfont=dict(size=11, color="#334155"),
            hoverinfo="skip",
            showlegend=False,
        ))

    for axis_idx, axis_name in enumerate(metric_names):
        c, s = np.cos(angles[axis_idx]), np.sin(angles[axis_idx])
        ax = max_radius * c
        ay = max_radius * s
        lx = max_radius * axis_label_radius_factor * c
        ly = max_radius * axis_label_radius_factor * s
        fig.add_trace(go.Scatter3d(
            x=[0, ax], y=[0, ay], z=[0, 0],
            mode="lines",
            line=dict(color="rgba(210,220,230,0.34)", width=3),
            hoverinfo="skip",
            showlegend=False,
        ))
        fig.add_trace(go.Scatter3d(
            x=[lx], y=[ly], z=[0.0],
            mode="text",
            text=[axis_name],
            textposition="top center",
            textfont=dict(size=12, color="#0f172a"),
            hoverinfo="skip",
            showlegend=False,
        ))

    ordered = sorted(records, key=lambda r: r.get("map50_95", 0.0), reverse=True)

    # 重新规划刻度：
    # 按维度做鲁棒归一化（P10~P90），并给视觉半径设置保底，避免“贴地看起来像 0”
    metric_getters = [
        lambda r: min(max(r.get("map50", 0.0), 0.0), 1.0),
        lambda r: min(max(r.get("map50_95", 0.0), 0.0), 1.0),
        lambda r: min(max(r.get("precision", 0.0), 0.0), 1.0),
        lambda r: min(max(r.get("recall", 0.0), 0.0), 1.0),
        lambda r: min(max(r.get("speed_score", 0.0), 0.0), 1.0),
        lambda r: min(max(r.get("deploy_cost_score", 0.0), 0.0), 1.0),
        lambda r: min(max(r.get("efficiency_score", 0.0), 0.0), 1.0),
    ]
    raw_metric_rows = [[getter(rec) for getter in metric_getters] for rec in ordered]
    dim_stats = []
    for dim_idx in range(len(metric_names)):
        vals = [row[dim_idx] for row in raw_metric_rows]
        if not vals:
            dim_stats.append((0.0, 1.0))
            continue
        lo = float(np.percentile(vals, 10))
        hi = float(np.percentile(vals, 90))
        if hi - lo < 1e-6:
            lo = min(vals)
            hi = max(vals)
        dim_stats.append((lo, hi))

    visual_floor = 0.24  # 最小视觉半径占比
    gamma = 0.78         # >0 且 <1 时，小值会被“抬起来”

    for idx, rec in enumerate(ordered):
        raw_metrics = raw_metric_rows[idx]
        metrics_pct = [m * 100.0 for m in raw_metrics]

        metrics = []
        for dim_idx, raw_v in enumerate(raw_metrics):
            lo, hi = dim_stats[dim_idx]
            if hi - lo < 1e-6:
                norm = 0.62
            else:
                clipped = min(max(raw_v, lo), hi)
                norm = (clipped - lo) / (hi - lo)
            norm = min(max(norm, 0.0), 1.0)
            visual_v = visual_floor + (1.0 - visual_floor) * (norm ** gamma)
            metrics.append(visual_v)

        # 第一层抬高，避免与底部参考环遮挡
        z_layer = (idx + 1) * layer_gap
        color = _model_fixed_color(rec["model_name"], rec.get("base_weight", ""), fallback_palette, idx)

        px = [metrics[i] * max_radius * np.cos(angles[i]) for i in range(len(metric_names))]
        py = [metrics[i] * max_radius * np.sin(angles[i]) for i in range(len(metric_names))]
        pz = [z_layer] * len(metric_names)

        vx = [0.0] + px
        vy = [0.0] + py
        vz = [z_layer] + pz
        i_idx, j_idx, k_idx = [], [], []
        for p in range(1, len(metric_names) + 1):
            q = p + 1 if p < len(metric_names) else 1
            i_idx.append(0)
            j_idx.append(p)
            k_idx.append(q)

        hover_text = (
            f"<b>{rec['model_name']}</b><br>"
            f"基线: {rec['base_weight']}<br>"
            f"mAP50: {metrics_pct[0]:.1f}%<br>"
            f"mAP50-95: {metrics_pct[1]:.1f}%<br>"
            f"精确率 (Precision): {metrics_pct[2]:.1f}%<br>"
            f"召回率 (Recall): {metrics_pct[3]:.1f}%<br>"
            f"推理总耗时: {rec.get('infer_total_ms', 0.0):.2f} ms<br>"
            f"FPS: {rec.get('fps', 0.0):.2f}<br>"
            f"识别速度得分: {metrics_pct[4]:.1f}%<br>"
            f"部署成本得分: {metrics_pct[5]:.1f}%<br>"
            f"综合效率: {metrics_pct[6]:.1f}%"
        )
        fig.add_trace(go.Mesh3d(
            x=vx, y=vy, z=vz,
            i=i_idx, j=j_idx, k=k_idx,
            color=color,
            opacity=0.30,
            flatshading=True,
            hovertext=hover_text,
            hoverinfo="text",
            showlegend=False,
            lighting=dict(ambient=0.42, diffuse=0.82, specular=0.45, roughness=0.35),
            lightposition=dict(x=90, y=-110, z=140),
        ))

        px_closed = px + [px[0]]
        py_closed = py + [py[0]]
        pz_closed = pz + [pz[0]]
        fig.add_trace(go.Scatter3d(
            x=px_closed,
            y=py_closed,
            z=pz_closed,
            mode="lines+markers",
            marker=dict(size=4.2, color=color, opacity=0.95),
            line=dict(color=color, width=6),
            hovertext=[f"{rec['model_name']} - {metric_names[i % len(metric_names)]}: {metrics_pct[i % len(metric_names)]:.1f}%" for i in range(len(px_closed))],
            hoverinfo="text",
            showlegend=False,
        ))

        fig.add_trace(go.Scatter3d(
            x=[max_radius + max_radius * 0.08], y=[0], z=[z_layer],
            mode="text",
            text=[rec["model_name"]],
            textfont=dict(size=12, color="#0f172a"),
            hoverinfo="skip",
            showlegend=False,
        ))

    _apply_3d_scene(
        fig,
        "模型多维画像",
        "",
        "",
        "",
        x_range=[-(max_radius + 100.0), (max_radius + 100.0)],
        y_range=[-(max_radius + 92.0), (max_radius + 92.0)],
        z_range=[0, max(30.0, len(ordered) * layer_gap + 34.0)],
    )
    fig.update_layout(
        scene_camera=dict(eye=dict(x=2.45, y=1.95, z=1.70)),
        margin=dict(l=20, r=20, b=22, t=48),
    )
    # 去掉 XYZ 刻度与网格，避免干扰多维环刻度阅读
    _hide_3d_scene_axes(fig)
    fig.update_layout(font=dict(color="#0f172a"))
    return fig


def refresh_model_metrics():
    records = []
    for run_dir in _collect_training_run_dirs():
        results_csv = run_dir / "results.csv"
        args_yaml = run_dir / "args.yaml"
        args = _parse_simple_yaml(args_yaml)
        last_row = None
        try:
            with open(results_csv, "r", encoding="utf-8", errors="ignore", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    last_row = row
        except Exception:
            continue

        if not last_row:
            continue

        # 若已执行过测速，读取 benchmark.json；否则 benchmark 为空字典
        benchmark = {}
        bench_path = run_dir / "benchmark.json"
        if bench_path.exists():
            try:
                with open(bench_path, "r", encoding="utf-8", errors="ignore") as f:
                    benchmark = json.load(f) or {}
            except Exception:
                benchmark = {}

        model_name = _display_run_name(run_dir.name, args.get("model", ""))
        base_weight = args.get("model", "unknown")
        epochs = _to_int(args.get("epochs", 0), 0)
        batch = _to_int(args.get("batch", 0), 0)
        imgsz = _to_int(args.get("imgsz", 0), 0)
        map50 = _to_float(last_row.get("metrics/mAP50(B)"), 0.0)
        map50_95 = _to_float(last_row.get("metrics/mAP50-95(B)"), 0.0)
        precision = _to_float(last_row.get("metrics/precision(B)"), 0.0)
        recall = _to_float(last_row.get("metrics/recall(B)"), 0.0)
        train_time = _to_float(last_row.get("time"), 0.0)
        avg_epoch_time = train_time / max(epochs, 1) if train_time > 0 else 0.0

        local_weight = run_dir / "weights" / "best.pt"
        if not local_weight.exists():
            local_weight = run_dir / "weights" / "last.pt"
        params_m = _estimate_params_million(local_weight) if local_weight.exists() else None

        # 从 benchmark.json 映射到表格列
        infer_total_ms = _to_float(benchmark.get("mean_total_ms"), 0.0)   # 平均推理总耗时(ms)
        p95_total_ms = _to_float(benchmark.get("p95_total_ms"), 0.0)    # P95 耗时(ms)
        fps = _to_float(benchmark.get("fps"), 0.0)                      # 每秒帧数
        bench_device = benchmark.get("device", "-")                       # 测速设备
        bench_samples = _to_int(benchmark.get("sample_count"), 0)         # 测速样本数

        records.append({
            "model_name": model_name,
            "base_weight": base_weight,
            "epochs": epochs,
            "batch": batch,
            "imgsz": imgsz,
            "map50": map50,
            "map50_95": map50_95,
            "precision": precision,
            "recall": recall,
            "avg_epoch_time": avg_epoch_time,
            "params_m": params_m,
            "infer_total_ms": infer_total_ms,
            "p95_total_ms": p95_total_ms,
            "fps": fps,
            "bench_device": bench_device,
            "bench_samples": bench_samples,
        })

    # 把各模型的绝对指标换算成 0~1 相对得分，用于横向对比和综合效率
    if records:
        # 收集所有已测速模型的 infer_total_ms，求最快、最慢
        speed_values = [r["infer_total_ms"] for r in records if r["infer_total_ms"] > 0]
        if speed_values:
            t_min, t_max = min(speed_values), max(speed_values)
        else:
            t_min, t_max = 0.0, 0.0

        # 参数量取 log10，避免 2M 与 68M 差距过大导致中间模型得分挤在一起
        logp_vals = []
        for r in records:
            pm = r["params_m"]
            if pm is not None and pm > 0:
                logp_vals.append(math.log10(pm))
        if logp_vals:
            lp_min, lp_max = min(logp_vals), max(logp_vals)
        else:
            lp_min, lp_max = 0.0, 0.0

        for rec in records:
            t = rec["infer_total_ms"]
            if t > 0 and t_max > t_min:
                # 耗时越短得分越高：(最慢-当前)/(最慢-最快)，最快=1，最慢=0
                speed_score = (t_max - t) / (t_max - t_min)
            elif t > 0:
                speed_score = 1.0   # 只有一个模型有测速数据
            else:
                speed_score = 0.5   # 未测速，给中性分

            p = rec["params_m"]
            if p is not None and p > 0 and lp_max > lp_min:
                lp = math.log10(p)
                # 参数量越小得分越高，公式同 speed_score
                deploy_cost_score = (lp_max - lp) / (lp_max - lp_min)
            elif p is not None and p > 0:
                deploy_cost_score = 1.0
            else:
                deploy_cost_score = 0.5

            # 加权求和：mAP50-95×0.45 + 精确率×0.20 + 召回率×0.15 + 速度分×0.12 + 部署分×0.08
            efficiency_score = (
                rec["map50_95"] * 0.45
                + rec["precision"] * 0.20
                + rec["recall"] * 0.15
                + speed_score * 0.12
                + deploy_cost_score * 0.08
            )
            rec["speed_score"] = float(min(max(speed_score, 0.0), 1.0))       # 限制在 0~1
            rec["deploy_cost_score"] = float(min(max(deploy_cost_score, 0.0), 1.0))
            rec["efficiency_score"] = float(min(max(efficiency_score, 0.0), 1.0))

    table_rows = [
        [
            rec["model_name"],
            rec["base_weight"],
            rec["epochs"],
            rec["batch"],
            rec["imgsz"],
            round(rec["map50"], 4),
            round(rec["map50_95"], 4),
            round(rec["precision"], 4),
            round(rec["recall"], 4),
            round(rec["avg_epoch_time"], 3) if rec["avg_epoch_time"] > 0 else "-",
            round(rec["infer_total_ms"], 3) if rec["infer_total_ms"] > 0 else "-",
            round(rec["fps"], 2) if rec["fps"] > 0 else "-",
            round(rec["p95_total_ms"], 3) if rec["p95_total_ms"] > 0 else "-",
            rec["bench_device"],
            rec["bench_samples"] if rec["bench_samples"] > 0 else "-",
            round(rec["params_m"], 2) if rec["params_m"] is not None else "-",
            round(rec.get("speed_score", 0.5), 4),
            round(rec.get("deploy_cost_score", 0.5), 4),
            round(rec.get("efficiency_score", 0.0), 4),
        ]
        for rec in records
    ]

    fig = _build_model_profile_3d(records)
    return table_rows, fig


def refresh_training_dashboard():
    """刷新检测页「模型」下拉 + 评估表与 3D 多维画像（会递归扫描 runs/detect）。"""
    global MODEL_OPTIONS
    MODEL_OPTIONS = _scan_trained_models()
    rows, fig = refresh_model_metrics()
    names = [o[0] for o in MODEL_OPTIONS]
    return gr.update(choices=names, value=names[0] if names else None), rows, fig


# --------------------------------------------------------------------------------
# 核心预测逻辑
# --------------------------------------------------------------------------------
def smart_diagnosis(image, model_choice, conf_threshold, use_sam, sam_model, sam_device, use_ai, api_key, ai_model):
    """
    一站式诊断：流式输出 (YOLO -> SAM -> 病害量化指标卡/条形图 -> AI)
    """
    empty_metrics_html = _metrics_html_placeholder("请先上传图片")
    if image is None:
        _ai0 = {"说明": "请先上传叶片图片后再运行诊断。", "mode": "no_image"}
        yield (
            None,
            None,
            empty_metrics_html,
            _empty_disease_metrics_fig("请先上传图片"),
            _llm_diagnosis_to_markdown(_ai0),
            _ai0,
            "请上传一张叶片图片",
        )
        return

    # 初始化所有返回值，作为初始流输出
    yolo_plot_rgb = None
    analysis_img = None
    stats_info = "⌛ 正在启动推理引擎..."
    ai_suggestion: dict = {"说明": "等待中…", "mode": "pending"}
    metrics_html = _metrics_html_placeholder("等待中…")
    metrics_fig = _empty_disease_metrics_fig("等待诊断")

    try:
        # 1. 释放显存
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except: pass

        opts = _scan_trained_models()
        if not opts:
            _nw = {"error": "请完成训练或检查权重路径", "mode": "no_weights"}
            yield (
                None,
                None,
                _metrics_html_placeholder("当前无可用模型权重。"),
                _empty_disease_metrics_fig("无权重"),
                _llm_diagnosis_to_markdown(_nw),
                _nw,
                "⚠️ 未找到训练权重。请将训练输出放在 runs/detect/<保存名称>/weights/best.pt（避免 runs/detect/runs/detect 重复嵌套）。",
            )
            return
        if isinstance(model_choice, int):
            idx = max(0, min(int(model_choice), len(opts) - 1))
            model_path = opts[idx][1]
        else:
            model_path = None
            if model_choice is not None:
                choice_str = str(model_choice).strip()
                for name, wpath in opts:
                    if name == choice_str:
                        model_path = wpath
                        break
                if model_path is None and choice_str.endswith(".pt"):
                    model_path = choice_str
            if model_path is None and opts:
                model_path = opts[0][1]
        model = YOLO(model_path)
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        results = model.predict(source=img_bgr, imgsz=512, conf=conf_threshold, verbose=False)
        yolo_plot_rgb = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)
        analysis_img = yolo_plot_rgb # 暂时作为占位
        
        boxes = results[0].boxes
        detections = {}
        disease_conf_sum = {}
        disease_conf_count = {}
        box_class_names_list: list[str] = []

        if boxes is not None and len(boxes) > 0:
            # 关键修复：一次性将所有检测结果转为 CPU 上的 Python 类型，避免后续混入 Tensor
            cls_ids = boxes.cls.cpu().tolist()
            confs = boxes.conf.cpu().tolist()
            xyxy_list = boxes.xyxy.cpu().tolist()

            for i, cls_id in enumerate(cls_ids):
                name = model.names[int(cls_id)]
                box_class_names_list.append(name)
                detections[name] = detections.get(name, 0) + 1

                conf = float(confs[i])
                disease_conf_sum[name] = disease_conf_sum.get(name, 0.0) + conf
                disease_conf_count[name] = disease_conf_count.get(name, 0) + 1
            
            stats_info = "✅ YOLO 已完成定位\n" + "\n".join(
                [f"  • {_disease_display_name(k)}: {v} 处" for k, v in detections.items()]
            )
        else:
            stats_info = "⚪ 未检出病斑"
            if use_ai:
                ai_suggestion = {"说明": "即将使用 LLM（未检出框，仅依据图像）", "mode": "pending_no_yolo"}
            else:
                ai_suggestion = {"说明": "未检出病斑。建议定期观察。", "mode": "no_detector_no_llm"}

        sam_profile = None
        if detections:
            m_rows = _collect_disease_metric_rows(
                detections, disease_conf_sum, disease_conf_count, sam_profile
            )
            metrics_html = _build_disease_metrics_card_html(m_rows)
            metrics_fig = _build_disease_metrics_bar_figure(m_rows)
        else:
            metrics_html = _build_disease_metrics_card_html([])
            metrics_fig = _build_disease_metrics_bar_figure([])

        # 【第一次输出】：用户能立刻看到带框的图、文字统计与指标（SAM 前为占位）
        yield _smart_diag_row(yolo_plot_rgb, analysis_img, metrics_html, metrics_fig, ai_suggestion, stats_info)

        has_boxes = boxes is not None and len(boxes) > 0

        if not has_boxes:
            if use_ai:
                if not (api_key or "").strip():
                    ai_suggestion = {"error": "请填写 API Key 以开启 LLM 分析"}
                else:
                    ai_suggestion = {"说明": "⌛ LLM 分析中（未检出框，仅视觉 JSON）…", "mode": "loading"}
                    yield _smart_diag_row(yolo_plot_rgb, analysis_img, metrics_html, metrics_fig, ai_suggestion, stats_info)
                    ctx = _build_llm_detector_context({}, {}, {}, None, sam_quant_attempted=False)
                    print("--- [LLM] 无 YOLO 框，单轮视觉 ---")
                    ai_suggestion = run_llm_leaf_diagnosis(
                        image,
                        api_key,
                        ai_model,
                        has_detector_boxes=False,
                        detector_context=ctx,
                    )
                    yield _smart_diag_row(yolo_plot_rgb, analysis_img, metrics_html, metrics_fig, ai_suggestion, stats_info)
            return

        if use_sam:
            sam_pending = "\n\n⌛ 正在执行 SAM 像素级量化..."
            stats_info += sam_pending
            yield _smart_diag_row(yolo_plot_rgb, analysis_img, metrics_html, metrics_fig, ai_suggestion, stats_info)

            boxes_xyxy = boxes.xyxy.detach().cpu().numpy()
            overlay, sam_info, sam_profile = analyze_disease_extent(
                image,
                boxes_xyxy,
                model_type=sam_model,
                device=sam_device,
                box_class_names=box_class_names_list,
            )
            if overlay is not None:
                analysis_img = overlay
                sam_done = "\n\n" + sam_info
            else:
                sam_done = "\n\n⚠️ SAM 像素级量化未成功：" + (sam_info or "无详情")
                sam_profile = None
            stats_info = stats_info.replace(sam_pending, sam_done)
            m_rows = _collect_disease_metric_rows(
                detections, disease_conf_sum, disease_conf_count, sam_profile
            )
            metrics_html = _build_disease_metrics_card_html(m_rows)
            metrics_fig = _build_disease_metrics_bar_figure(m_rows)
            yield _smart_diag_row(yolo_plot_rgb, analysis_img, metrics_html, metrics_fig, ai_suggestion, stats_info)

            if use_ai:
                if not (api_key or "").strip():
                    ai_suggestion = {"error": "请填写 API Key 以开启 LLM 分析"}
                else:
                    ai_suggestion = {"说明": "⌛ LLM 交叉验证中（先视觉后融合算法）…", "mode": "loading"}
                    yield _smart_diag_row(yolo_plot_rgb, analysis_img, metrics_html, metrics_fig, ai_suggestion, stats_info)
                    det_ctx = _build_llm_detector_context(
                        detections,
                        disease_conf_sum,
                        disease_conf_count,
                        sam_profile,
                        sam_quant_attempted=True,
                    )
                    print("--- [LLM] 有框 + SAM 量化上下文 ---")
                    ai_suggestion = run_llm_leaf_diagnosis(
                        image,
                        api_key,
                        ai_model,
                        has_detector_boxes=True,
                        detector_context=det_ctx,
                    )
            else:
                ai_suggestion = {"说明": "未启用 LLM。可勾选「启用LLM分析」。", "mode": "llm_disabled"}

            yield _smart_diag_row(yolo_plot_rgb, analysis_img, metrics_html, metrics_fig, ai_suggestion, stats_info)

        else:
            stats_info += (
                "\n\nℹ️ 未启用 SAM：条形图中「叶内病斑比、叶片取景、框-掩膜一致」为占位刻度（"
                f"{int(_SAM_METRIC_PLACEHOLDER * 100)}%）。"
            )
            yield _smart_diag_row(yolo_plot_rgb, analysis_img, metrics_html, metrics_fig, ai_suggestion, stats_info)

            if use_ai:
                if not (api_key or "").strip():
                    ai_suggestion = {"error": "请填写 API Key 以开启 LLM 分析"}
                else:
                    ai_suggestion = {"说明": "⌛ LLM 交叉验证中（未跑 SAM，量化项为占位）…", "mode": "loading"}
                    yield _smart_diag_row(yolo_plot_rgb, analysis_img, metrics_html, metrics_fig, ai_suggestion, stats_info)
                    det_ctx = _build_llm_detector_context(
                        detections,
                        disease_conf_sum,
                        disease_conf_count,
                        sam_profile,
                        sam_quant_attempted=False,
                    )
                    print("--- [LLM] 有框 / 未启用 SAM ---")
                    ai_suggestion = run_llm_leaf_diagnosis(
                        image,
                        api_key,
                        ai_model,
                        has_detector_boxes=True,
                        detector_context=det_ctx,
                    )
            else:
                ai_suggestion = {"说明": "未启用 LLM。可勾选「启用LLM分析」。", "mode": "llm_disabled"}

            yield _smart_diag_row(yolo_plot_rgb, analysis_img, metrics_html, metrics_fig, ai_suggestion, stats_info)

    except Exception as e:
        import html as html_lib
        import traceback
        traceback.print_exc()
        err_html = _metrics_html_placeholder(f"诊断异常：{html_lib.escape(str(e))}")
        _err_ai = {"error": "诊断过程异常", "detail": str(e), "mode": "error"}
        yield (
            yolo_plot_rgb,
            analysis_img,
            err_html,
            _empty_disease_metrics_fig("诊断失败"),
            _llm_diagnosis_to_markdown(_err_ai),
            _err_ai,
            f"❌ 系统异常: {e}",
        )


# --------------------------------------------------------------------------------
# 训练相关逻辑
# --------------------------------------------------------------------------------
TRAIN_PROC = None
TRAIN_LOG_PATH = ROOT / "train_realtime.log"


def _build_yolo_train_cmd(model_name, epochs, batch, imgsz, device, exp_name):
    """
    训练命令：
    - 不用 yolo.exe：在 Windows 上多为非控制台入口，PIPE 常捕不到输出，且易秒退 exit=1 却无日志。
    - 不用 python -m ultralytics：8.4+ 无 ultralytics.__main__。
    - 使用 Python 调用 ultralytics.cfg.entrypoint(debug=...)；其中 debug 须以「yolo」开头，
      因为 entrypoint 会对参数字符串做 [1:] 切片（模拟去掉 argv0）。
    """
    try:
        import ultralytics  # noqa: F401
    except ImportError:
        pass
    else:
        parts = [
            "train",
            f"model={model_name}",
            f"data={DATA_YAML}",
            f"epochs={int(epochs)}",
            f"batch={int(batch)}",
            f"imgsz={int(imgsz)}",
            f"device={device}",
            "project=runs/detect",
            f"name={exp_name}",
            "exist_ok=True",
        ]
        debug = "yolo " + " ".join(parts)
        code = "from ultralytics.cfg import entrypoint; entrypoint(%r)" % (debug,)
        return [str(sys.executable), "-u", "-c", code]

    return [
        str(sys.executable),
        str(ROOT / "train.py"),
        "--model",
        str(model_name),
        "--data",
        str(DATA_YAML),
        "--epochs",
        str(int(epochs)),
        "--batch",
        str(int(batch)),
        "--imgsz",
        str(int(imgsz)),
        "--device",
        str(device),
        "--project",
        "runs/detect",
        "--name",
        str(exp_name),
    ]


def start_training(model_name, epochs, batch, imgsz, device, exp_name):
    global TRAIN_PROC
    if TRAIN_PROC and TRAIN_PROC.poll() is None:
        return "⚠️ 训练已经在运行中！"

    if TRAIN_LOG_PATH.exists():
        TRAIN_LOG_PATH.unlink()

    cmd = _build_yolo_train_cmd(model_name, epochs, batch, imgsz, device, exp_name)

    header = (
        f"=== 训练启动 {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n"
        f"命令: {' '.join(cmd)}\n"
        f"工作目录: {ROOT}\n\n"
    )
    TRAIN_LOG_PATH.write_text(header, encoding="utf-8")

    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "PYTHONIOENCODING": "utf-8",
        "NO_PROXY": "127.0.0.1,localhost",
    }

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(ROOT),
            env=env,
            text=True,
            bufsize=1,
            encoding="utf-8",
            errors="replace",
        )
        TRAIN_PROC = proc

        def _pump_training_log():
            """子进程 stdout 泵入文件；避免父进程过早关闭 stdout 文件导致 Windows 下日志全空。"""
            try:
                with open(TRAIN_LOG_PATH, "a", encoding="utf-8", errors="replace", newline="") as lf:
                    if proc.stdout is not None:
                        for line in proc.stdout:
                            lf.write(line)
                            lf.flush()
                    code = proc.wait()
                    lf.write(f"\n=== 进程结束 exit={code} ===\n")
                    lf.flush()
            except Exception as ex:
                try:
                    with open(TRAIN_LOG_PATH, "a", encoding="utf-8", errors="replace") as lf:
                        lf.write(f"\n[日志泵异常] {type(ex).__name__}: {ex}\n")
                        lf.flush()
                except Exception:
                    pass

        threading.Thread(target=_pump_training_log, daemon=True).start()
        return f"🚀 训练已启动 (PID: {TRAIN_PROC.pid})"
    except Exception as e:
        TRAIN_PROC = None
        try:
            with open(TRAIN_LOG_PATH, "a", encoding="utf-8", errors="replace") as lf:
                lf.write(f"\n❌ 启动失败: {type(e).__name__}: {e}\n")
        except Exception:
            pass
        return f"❌ 启动失败: {e}"

def stop_training():
    global TRAIN_PROC
    if TRAIN_PROC and TRAIN_PROC.poll() is None:
        TRAIN_PROC.terminate()
        return "🛑 训练进程已终止"
    return "⚪ 当前无运行中的训练"

def get_train_log():
    if not TRAIN_LOG_PATH.exists():
        return "等待日志生成...", "⚪ 未开始"
    
    with open(TRAIN_LOG_PATH, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    
    status = "🟢 正在训练" if (TRAIN_PROC and TRAIN_PROC.poll() is None) else "🔴 已停止/完成"
    return content[-15000:], status

# --------------------------------------------------------------------------------
# CSS 样式 (Console Dashboard)
# --------------------------------------------------------------------------------
custom_css = """
body {
    background: radial-gradient(1200px 700px at 0% 0%, #1e2a1f 0%, rgba(30,42,31,0) 60%),
                radial-gradient(1000px 600px at 100% 0%, #3a2918 0%, rgba(58,41,24,0) 55%),
                linear-gradient(165deg, #0f1113 0%, #171a1f 45%, #111417 100%) !important;
    background-attachment: fixed !important;
    font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif !important;
}

.gradio-container {
    max-width: 1500px !important;
}

.app-header {
    padding: 14px 18px;
    border-radius: 8px;
    border: 1px solid rgba(255, 255, 255, 0.15);
    background: linear-gradient(120deg, rgba(19,26,33,0.82), rgba(36,31,24,0.75));
    box-shadow: 0 16px 34px rgba(0, 0, 0, 0.30);
}

.app-header h1 {
    margin: 0;
    font-size: 2.1rem;
    color: #f8fafc;
    letter-spacing: 0;
}

.panel {
    border-radius: 8px !important;
    border: 1px solid rgba(255, 255, 255, 0.14) !important;
    background: linear-gradient(130deg, rgba(19,22,30,0.86), rgba(27,32,24,0.82)) !important;
    box-shadow: 0 10px 26px rgba(0, 0, 0, 0.26) !important;
    padding: 14px !important;
}

.panel h3,
.panel h4 {
    margin: 0 0 10px 0 !important;
    color: #f1f5f9 !important;
    letter-spacing: 0 !important;
}

.section-note {
    color: #334155;
    font-size: 0.92rem;
    margin-bottom: 10px;
    line-height: 1.55;
}

/* 面板内主文字：避免 Soft 主题在深色背景下仍用浅灰导致看不清 */
.panel textarea,
.panel input[type="text"],
.panel input[type="password"],
.panel input[type="number"] {
    color: #f8fafc !important;
    background: rgba(15, 23, 42, 0.72) !important;
    border: 1px solid rgba(148, 163, 184, 0.4) !important;
    -webkit-text-fill-color: #f8fafc !important;
}

.panel .markdown,
.panel .prose,
.panel .prose p,
.panel .prose li {
    color: #e2e8f0 !important;
}

.panel label,
.panel .label-wrap span,
.panel .info {
    color: #cbd5e1 !important;
}

/* 针对 Gradio 默认组件的浅灰背景，听取建议直接把它们内部的文字改为深色，解决灰底白字问题 */
.panel table,
.panel table th,
.panel table td,
.panel table span {
    color: #1e293b !important; /* 深灰色文字 */
}

.panel .options,
.panel .options li {
    color: #1e293b !important; /* 深灰色文字 */
}

/* 诊断：SAM 与推理设备同一行，三个选项横向不换行、略压缩 */
.sam-model-device-row {
    flex-wrap: nowrap !important;
    align-items: flex-end !important;
}
.sam-model-device-row > div {
    min-width: 0 !important;
}
.panel .sam-infer-device-radio .wrap,
.panel .sam-infer-device-radio .radio-group,
.panel .sam-infer-device-radio fieldset {
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: nowrap !important;
    align-items: center !important;
    gap: 0.2rem !important;
}
.panel .sam-infer-device-radio label,
.panel .sam-infer-device-radio span[data-testid] {
    font-size: 0.78rem !important;
    padding: 0.15rem 0.4rem !important;
    white-space: nowrap !important;
}

/* 诊断页：检测摘要 + 量化指标合并为一块白底黑字 */
.metrics-merged-panel.panel {
    background: #ffffff !important;
    border: 1px solid rgba(15, 23, 42, 0.12) !important;
    box-shadow: 0 6px 22px rgba(0, 0, 0, 0.1) !important;
}
.metrics-merged-panel h4 {
    color: #0f172a !important;
}
.metrics-merged-panel .markdown,
.metrics-merged-panel .prose,
.metrics-merged-panel .prose p {
    color: #0f172a !important;
}
.metrics-merged-panel textarea {
    color: #0f172a !important;
    background: #f8fafc !important;
    border: 1px solid #e2e8f0 !important;
    -webkit-text-fill-color: #0f172a !important;
}
.metrics-merged-panel .metrics-inline-note {
    color: #334155 !important;
    background: #f1f5f9 !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px;
    padding: 8px 12px;
    font-size: 0.92rem;
    line-height: 1.5;
    margin-top: 8px;
    margin-bottom: 0;
}

/* LLM 建议：纵向更紧凑 */
.llm-suggest-panel.panel {
    padding: 10px 12px !important;
}
.llm-suggest-panel.panel h4 {
    margin: 0 0 6px 0 !important;
    font-size: 1.02rem !important;
}
.llm-suggest-panel .markdown,
.llm-suggest-panel .prose {
    font-size: 0.9rem !important;
    line-height: 1.36 !important;
}
.llm-suggest-panel .prose h5,
.llm-suggest-panel .markdown h5 {
    margin: 0.35em 0 0.18em 0 !important;
    font-size: 0.88rem !important;
    line-height: 1.25 !important;
}
.llm-suggest-panel .prose p {
    margin: 0.12em 0 !important;
}
.llm-suggest-panel .prose ul,
.llm-suggest-panel .prose ol {
    margin: 0.15em 0 0.1em 0 !important;
    padding-left: 1.15em !important;
}
.llm-suggest-panel .prose li {
    margin: 0.08em 0 !important;
}
.llm-suggest-panel table {
    margin: 0.2em 0 0.15em 0 !important;
    font-size: 0.86rem !important;
}

/* 诊断页右下：微型折叠（JSON / 日志），默认弱化 */
.diag-debug-corner-row {
    justify-content: flex-end !important;
    align-items: flex-start !important;
    flex-wrap: nowrap !important;
    margin: 2px 0 0 0 !important;
    gap: 0.25rem !important;
    min-height: 0 !important;
}
.diag-mini-acc-wrap {
    flex: 0 0 auto !important;
    min-width: 0 !important;
    max-width: 4.8rem !important;
    opacity: 0.45 !important;
    transition: opacity 0.15s ease, max-width 0.15s ease !important;
}
.diag-mini-acc-wrap:has(details[open]) {
    max-width: min(92vw, 540px) !important;
    z-index: 6 !important;
    opacity: 0.98 !important;
}
.diag-debug-corner-row:hover .diag-mini-acc-wrap:not(:has(details[open])) {
    opacity: 0.92 !important;
}
.diag-mini-acc details > summary {
    list-style: none !important;
    cursor: pointer !important;
    padding: 0 2px !important;
    font-size: 0.62rem !important;
    letter-spacing: 0 !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    max-width: 4.2rem !important;
    line-height: 1.15 !important;
}
.diag-mini-acc details[open] > summary {
    max-width: 100% !important;
}
.diag-mini-acc .wrap,
.diag-mini-acc .form {
    min-height: 0 !important;
}

.primary-btn {
    border-radius: 8px !important;
    border: none !important;
    background: linear-gradient(120deg, #14b8a6, #f97316) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    letter-spacing: 0 !important;
}

.tab-nav {
    border-radius: 8px !important;
    background: rgba(15, 18, 22, 0.6) !important;
    border: 1px solid rgba(255, 255, 255, 0.13) !important;
}

.tab-nav button {
    color: #e2e8f0 !important;
}

.tab-nav button.selected {
    background: rgba(56, 189, 248, 0.2) !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
}

.gradio-image,
.gradio-textbox,
textarea,
input[type="text"],
input[type="password"] {
    border-radius: 8px !important;
}
"""

# --------------------------------------------------------------------------------
# UI 构建 (Refactored Console)
# --------------------------------------------------------------------------------
app_theme = gr.themes.Soft()
demo = gr.Blocks(title="植物病害检测终端")

with demo:
    gr.HTML("""
        <div class="app-header">
            <h1>植物病害检测终端</h1>
        </div>
    """)

    with gr.Tabs():
        with gr.TabItem("诊断"):
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Group(elem_classes=["panel"]):
                        gr.Markdown("### 输入")
                        input_img = gr.Image(label="叶片图像", type="numpy", height=320)

                        with gr.Row():
                            model_dd = gr.Dropdown(
                                choices=[opt[0] for opt in MODEL_OPTIONS],
                                value=MODEL_OPTIONS[0][0] if MODEL_OPTIONS else None,
                                label="检测模型",
                                scale=5,  # 缩小一点
                            )
                            conf_sld = gr.Slider(
                                minimum=0.01, maximum=0.9, value=0.25, step=0.01, label="置信度阈值", scale=4  # 扩大约 30%
                            )

                        use_sam_cb = gr.Checkbox(label="启用SAM分割", value=True)
                        with gr.Row(elem_classes=["sam-model-device-row"]):
                            sam_type = gr.Dropdown(
                                choices=["SAM 2.1", "SAM 1"],
                                value="SAM 2.1",
                                label="SAM模型",
                                scale=1,
                            )
                            sam_dev = gr.Radio(
                                choices=["auto", "cuda", "cpu"],
                                value="auto",
                                label="推理设备",
                                scale=5,
                                elem_classes=["sam-infer-device-radio"],
                            )

                        use_ai_cb = gr.Checkbox(label="启用LLM分析", value=False)
                        with gr.Row():
                            ai_api_key = gr.Textbox(label="API Key", type="password", value=DEFAULT_API_KEY, placeholder="sk-...")
                            ai_model_dd = gr.Dropdown(
                                choices=list(MODELS.keys()),
                                value=list(MODELS.keys())[0],
                                label="LLM模型",
                            )

                        run_btn = gr.Button("运行诊断", elem_classes=["primary-btn"], size="lg")

                with gr.Column(scale=2):
                    with gr.Row():
                        with gr.Group(elem_classes=["panel"]):
                            gr.Markdown("#### 检测结果")
                            yolo_res = gr.Image(label="YOLO 输出", height=300)
                        with gr.Group(elem_classes=["panel"]):
                            gr.Markdown("#### 分割结果")
                            sam_res = gr.Image(label="SAM 输出（黄：叶片掩膜；红：叶内病变）", height=300)

                    with gr.Row():
                        with gr.Column(scale=2):
                            with gr.Group(elem_classes=["panel", "metrics-merged-panel"]):
                                gr.Markdown("#### 检测摘要与量化指标")
                                disease_metrics_html = gr.HTML()
                                disease_metrics_plot = gr.Plot(label="", show_label=False)
                        with gr.Column(scale=1):
                            with gr.Group(elem_classes=["panel", "llm-suggest-panel"]):
                                gr.Markdown("#### 处理建议（LLM）")
                                ai_readable = gr.Markdown(value="")

                    with gr.Row(elem_classes=["diag-debug-corner-row"]):
                        with gr.Column(elem_classes=["diag-mini-acc-wrap"], min_width=52):
                            with gr.Accordion("JSON", open=False, elem_classes=["diag-mini-acc"]):
                                ai_out = gr.JSON(label="", value={}, show_label=False)
                        with gr.Column(elem_classes=["diag-mini-acc-wrap"], min_width=52):
                            with gr.Accordion("日志", open=False, elem_classes=["diag-mini-acc"]):
                                debug_log = gr.Textbox(label="", lines=8, show_label=False)

            run_btn.click(
                fn=smart_diagnosis,
                inputs=[input_img, model_dd, conf_sld, use_sam_cb, sam_type, sam_dev, use_ai_cb, ai_api_key, ai_model_dd],
                outputs=[yolo_res, sam_res, disease_metrics_html, disease_metrics_plot, ai_readable, ai_out, debug_log],
            )

        with gr.TabItem("训练与评估"):
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Group(elem_classes=["panel"]):
                        gr.Markdown("### 训练参数")
                        tr_base = gr.Textbox(value="yolov8x.pt", label="基础权重")
                        tr_ep = gr.Slider(10, 300, 100, step=10, label="Epochs")
                        tr_bs = gr.Slider(4, 64, 16, step=4, label="Batch")
                        tr_sz = gr.Dropdown(["512", "640", "800"], value="512", label="图像尺寸")
                        tr_name = gr.Textbox(
                            value="train_v8x_new",
                            label="保存名称",
                            info="保存至 runs/detect/<名称>/；勿与已有目录重名以免覆盖。",
                        )

                        with gr.Row():
                            tr_btn = gr.Button("启动训练", variant="primary", elem_classes=["primary-btn"])
                            tr_stop = gr.Button("停止训练", variant="stop")
                        tr_fb = gr.Textbox(
                            label="操作反馈",
                            lines=2,
                            info="启动/停止后立即回显的状态摘要。",
                        )

                    with gr.Group(elem_classes=["panel"]):
                        gr.Markdown("### 训练日志")
                        tr_status = gr.Textbox(
                            label="进程状态",
                            value="空闲",
                            info="定时刷新；详情见下方「实时输出」。",
                        )
                        tr_log = gr.Textbox(label="实时输出", lines=10, autoscroll=True)
                        tr_timer = gr.Timer(value=3)

                with gr.Column(scale=2):
                    with gr.Group(elem_classes=["panel"]):
                        with gr.Row():
                            gr.Markdown("### 模型评估")
                            refresh_btn = gr.Button("刷新", size="sm")
                            bench_btn = gr.Button("开始识别测速", size="sm", elem_classes=["primary-btn"])

                        with gr.Row():
                            bench_device = gr.Dropdown(choices=["cuda:0", "cpu"], value="cuda:0", label="测速设备", scale=2)
                            bench_imgsz = gr.Dropdown(choices=["512", "640", "800"], value="640", label="测速尺寸", scale=1)
                            bench_conf = gr.Slider(minimum=0.01, maximum=0.9, value=0.25, step=0.01, label="测速置信度", scale=2)
                        with gr.Row():
                            bench_samples = gr.Slider(minimum=20, maximum=300, value=100, step=10, label="测速样本数", scale=2)
                            bench_warmup = gr.Slider(
                                minimum=0,
                                maximum=50,
                                value=20,
                                step=5,
                                label="测速预热轮次",
                                scale=1,
                                info="正式计时前先丢弃若干次推理，减轻 GPU 冷启动与首次编译影响。",
                            )
                        bench_fb = gr.Textbox(label="测速状态", lines=2, value="尚未执行识别测速")

                        metrics_df = gr.Dataframe(
                            headers=[
                                "模型名称", "基础权重", "轮数 (Epochs)", "批次 (Batch)", "尺寸 (ImgSz)",
                                "mAP50", "mAP50-95", "精确率 (Precision)", "召回率 (Recall)",
                                "平均每轮训练耗时", "推理总耗时(ms)", "FPS", "P95推理耗时(ms)", "测速设备", "测速样本数",
                                "参数量(M)", "识别速度得分", "部署成本得分", "综合效率"
                            ],
                            interactive=False,
                        )
                        gr.HTML('<div class="section-note">每个模型一层多边形：精度、召回、识别速度、部署成本（参数量越小得分越高，按 log 参数量对比）与综合效率同时可见。图中环形刻度是相对对比刻度（已做鲁棒缩放），用于避免强模型把其余模型压扁成贴地形态。</div>')
                        metrics_plot = gr.Plot(label="模型多维画像")

            tr_btn.click(fn=start_training, inputs=[tr_base, tr_ep, tr_bs, tr_sz, gr.State("0"), tr_name], outputs=[tr_fb])
            tr_stop.click(fn=stop_training, outputs=[tr_fb])
            tr_timer.tick(fn=get_train_log, outputs=[tr_log, tr_status])
            demo.load(fn=refresh_training_dashboard, outputs=[model_dd, metrics_df, metrics_plot])
            refresh_btn.click(fn=refresh_training_dashboard, outputs=[model_dd, metrics_df, metrics_plot])
            bench_evt = bench_btn.click(
                fn=benchmark_model_inference,
                inputs=[bench_device, bench_imgsz, bench_conf, bench_samples, bench_warmup],
                outputs=[bench_fb],
            )
            bench_evt.then(fn=refresh_model_metrics, outputs=[metrics_df, metrics_plot])

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        theme=app_theme,
        css=custom_css,
    )
