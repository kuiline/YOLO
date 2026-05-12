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
from multimodal_api import ask_multimodal, call_vision_api, MODELS, DEFAULT_API_KEY
from mosaic_analyzer import analyze_mosaic

# --------------------------------------------------------------------------------
# 全局配置与状态
# --------------------------------------------------------------------------------
ROOT = Path(__file__).parent
DATA_YAML = str(ROOT / "datasets" / "data.yaml")


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

# 扫描已训练模型
def _scan_trained_models():
    options = []
    search_roots = [
        ROOT / "runs" / "detect",
    ]
    for base in search_roots:
        if not base.exists(): continue
        # 搜索所有 weights/best.pt
        for weights in sorted(base.glob("*/weights/best.pt")):
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
    return fallback_palette[fallback_idx % len(fallback_palette)]


def _collect_training_run_dirs():
    train_root = ROOT / "runs" / "detect"
    if not train_root.exists():
        return []
    runs = []
    for run_dir in sorted(train_root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if run_dir.is_dir() and (run_dir / "results.csv").exists():
            runs.append(run_dir)
    return runs


def _resolve_val_image_paths(sample_count=100):
    data_cfg = _parse_simple_yaml(Path(DATA_YAML))
    val_entry = data_cfg.get("val", "")
    data_root = data_cfg.get("path", "")
    yaml_parent = Path(DATA_YAML).parent

    def _candidate_bases():
        candidates = []
        if data_root:
            raw = Path(data_root)
            if raw.is_absolute():
                candidates.append(raw)
            else:
                # 常见两种写法都兼容：
                # 1) path 相对 data.yaml 所在目录
                # 2) path 相对项目根目录
                candidates.append((yaml_parent / raw).resolve())
                candidates.append((ROOT / raw).resolve())
                candidates.append(raw.resolve())
        candidates.append(yaml_parent.resolve())
        candidates.append(ROOT.resolve())

        seen = set()
        uniq = []
        for c in candidates:
            cs = str(c)
            if cs not in seen:
                seen.add(cs)
                uniq.append(c)
        return uniq

    bases = _candidate_bases()

    val_candidates = []
    if val_entry:
        v = Path(val_entry)
        if v.is_absolute():
            val_candidates.append(v)
        else:
            for b in bases:
                val_candidates.append((b / v).resolve())
    else:
        for b in bases:
            val_candidates.append((b / "images" / "val").resolve())

    val_path = None
    for c in val_candidates:
        if c.exists():
            val_path = c
            break
    if val_path is None:
        # 最后兜底，避免空值导致异常
        val_path = val_candidates[0] if val_candidates else (yaml_parent / "images" / "val")

    image_paths = []
    if val_path.is_file() and val_path.suffix.lower() == ".txt":
        with open(val_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                p = line.strip()
                if not p:
                    continue
                img = Path(p)
                if not img.is_absolute():
                    img = (val_path.parent / img).resolve()
                if img.exists():
                    image_paths.append(img)
    elif val_path.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
        image_paths = [p for p in val_path.rglob("*") if p.suffix.lower() in exts]

    image_paths = sorted(set(image_paths))
    if not image_paths:
        return []

    sample_count = max(1, int(sample_count))
    if len(image_paths) > sample_count:
        rng = random.Random(42)
        image_paths = sorted(rng.sample(image_paths, sample_count))
    return image_paths


def benchmark_model_inference(device, imgsz, conf, sample_count, warmup_count):
    run_dirs = _collect_training_run_dirs()
    if not run_dirs:
        return "⚠️ 未找到可测速模型（runs/detect/*/results.csv）"

    image_paths = _resolve_val_image_paths(sample_count=sample_count)
    if not image_paths:
        return "⚠️ 未找到验证集图像，请检查 data.yaml 的 val 路径"

    sample_count = len(image_paths)
    warmup_count = max(0, min(int(warmup_count), sample_count))
    device_str = str(device).strip() if device is not None else "0"
    conf_val = float(conf)
    imgsz_val = int(imgsz)

    updated, failed = 0, 0
    for run_dir in run_dirs:
        weight_path = run_dir / "weights" / "best.pt"
        if not weight_path.exists():
            weight_path = run_dir / "weights" / "last.pt"
        if not weight_path.exists():
            failed += 1
            continue

        try:
            model = YOLO(str(weight_path))

            # warmup
            for i in range(warmup_count):
                model.predict(
                    source=str(image_paths[i]),
                    imgsz=imgsz_val,
                    conf=conf_val,
                    device=device_str,
                    verbose=False,
                )

            pre_list, inf_list, post_list, total_list = [], [], [], []
            wall_start = time.perf_counter()
            for img_path in image_paths:
                results = model.predict(
                    source=str(img_path),
                    imgsz=imgsz_val,
                    conf=conf_val,
                    device=device_str,
                    verbose=False,
                )
                speed = results[0].speed if results else {}
                pre = float(speed.get("preprocess", 0.0))
                inf = float(speed.get("inference", 0.0))
                post = float(speed.get("postprocess", 0.0))
                total = pre + inf + post
                pre_list.append(pre)
                inf_list.append(inf)
                post_list.append(post)
                total_list.append(total)
            wall_ms = (time.perf_counter() - wall_start) * 1000.0

            if total_list:
                total_ms = float(np.mean(total_list))
                p95_total_ms = float(np.percentile(total_list, 95))
            else:
                total_ms, p95_total_ms = 0.0, 0.0

            fps = (1000.0 / total_ms) if total_ms > 1e-9 else 0.0
            benchmark_data = {
                "version": 1,
                "device": device_str,
                "imgsz": imgsz_val,
                "conf": conf_val,
                "sample_count": sample_count,
                "warmup_count": warmup_count,
                "wall_ms": round(wall_ms, 3),
                "mean_preprocess_ms": round(float(np.mean(pre_list)) if pre_list else 0.0, 4),
                "mean_inference_ms": round(float(np.mean(inf_list)) if inf_list else 0.0, 4),
                "mean_postprocess_ms": round(float(np.mean(post_list)) if post_list else 0.0, 4),
                "mean_total_ms": round(total_ms, 4),
                "p95_total_ms": round(p95_total_ms, 4),
                "fps": round(fps, 3),
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            bench_path = run_dir / "benchmark.json"
            with open(bench_path, "w", encoding="utf-8") as f:
                json.dump(benchmark_data, f, ensure_ascii=False, indent=2)
            updated += 1
        except Exception:
            failed += 1

    return f"✅ 识别测速完成：成功 {updated} 个模型，失败 {failed} 个模型（样本 {sample_count} 张，设备 {device_str}）"


_PRISM_I = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2]
_PRISM_J = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3]
_PRISM_K = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6]


def _add_prism(fig, x_center, y_center, width, depth, height, color, hover_text, name):
    x0, x1 = x_center - width / 2.0, x_center + width / 2.0
    y0, y1 = y_center - depth / 2.0, y_center + depth / 2.0
    z0, z1 = 0.0, max(height, 0.0)

    vx = [x0, x0, x1, x1, x0, x0, x1, x1]
    vy = [y0, y1, y1, y0, y0, y1, y1, y0]
    vz = [z0, z0, z0, z0, z1, z1, z1, z1]

    fig.add_trace(go.Mesh3d(
        x=vx,
        y=vy,
        z=vz,
        i=_PRISM_I,
        j=_PRISM_J,
        k=_PRISM_K,
        color=color,
        opacity=0.94,
        flatshading=True,
        name=name,
        showlegend=False,
        hovertext=hover_text,
        hoverinfo="text",
        lighting=dict(ambient=0.35, diffuse=0.88, specular=0.7, roughness=0.25, fresnel=0.2),
        lightposition=dict(x=110, y=-140, z=180),
    ))

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    xe, ye, ze = [], [], []
    for a, b in edges:
        xe.extend([vx[a], vx[b], None])
        ye.extend([vy[a], vy[b], None])
        ze.extend([vz[a], vz[b], None])
    fig.add_trace(go.Scatter3d(
        x=xe,
        y=ye,
        z=ze,
        mode="lines",
        line=dict(color="rgba(255,255,255,0.35)", width=3),
        hoverinfo="skip",
        showlegend=False,
    ))


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

def _build_disease_profile_3d(detections, disease_areas, disease_conf_sum, disease_conf_count, disease_centers, image_shape):
    fig = go.Figure()
    metric_names = ["框数", "面积占比", "均值置信", "空间离散", "中心偏移"]
    angles = np.linspace(0, 2 * np.pi, len(metric_names), endpoint=False)
    max_radius = 58.0
    palette = ["#34D399", "#F59E0B", "#60A5FA", "#F472B6", "#22D3EE", "#A3E635", "#FB7185"]

    if not detections:
        fig.add_trace(go.Scatter3d(
            x=[0], y=[0], z=[0],
            mode="markers+text",
            marker=dict(size=10, color="#A1A1AA"),
            text=["暂无病害数据"],
            textposition="top center",
            hoverinfo="skip",
            showlegend=False,
        ))
        _apply_3d_scene(
            fig,
            "病害多维画像",
            "维度 X",
            "维度 Y",
            "病害层",
            x_range=[-65, 65],
            y_range=[-65, 65],
            z_range=[0, 40],
        )
        return fig

    h, w = image_shape[:2]
    image_area = max(1.0, float(h * w))
    max_count = max(detections.values()) if detections else 1
    layer_gap = 28.0

    # 仅绘制一套基准轴/基准环，减少杂讯
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

    for axis_idx, axis_name in enumerate(metric_names):
        ax = max_radius * np.cos(angles[axis_idx])
        ay = max_radius * np.sin(angles[axis_idx])
        fig.add_trace(go.Scatter3d(
            x=[0, ax], y=[0, ay], z=[0, 0],
            mode="lines+text",
            line=dict(color="rgba(210,220,230,0.35)", width=4),
            text=["", axis_name],
            textposition="top center",
            textfont=dict(size=14, color="#0f172a"),
            hoverinfo="skip",
            showlegend=False,
        ))

    ordered = sorted(detections.items(), key=lambda x: x[1], reverse=True)
    for idx, (disease, count) in enumerate(ordered):
        area = disease_areas.get(disease, 0.0)
        area_ratio = min(1.0, area / image_area)
        conf_mean = 0.0
        if disease_conf_count.get(disease, 0) > 0:
            conf_mean = disease_conf_sum.get(disease, 0.0) / max(disease_conf_count[disease], 1)
        conf_mean = min(max(conf_mean, 0.0), 1.0)

        centers = disease_centers.get(disease, [])
        if centers:
            c = np.array(centers, dtype=float)
            center_mean = c.mean(axis=0)
            d = np.sqrt(np.sum((c - center_mean) ** 2, axis=1))
            dispersion = min(1.0, float(d.mean()) / 0.35)
            center_bias = min(1.0, float(np.sqrt(((center_mean - np.array([0.5, 0.5])) ** 2).sum()) / 0.70710678))
        else:
            dispersion = 0.0
            center_bias = 0.0

        metrics = [
            min(1.0, count / max(max_count, 1)),
            area_ratio,
            conf_mean,
            dispersion,
            center_bias,
        ]
        metrics_pct = [m * 100.0 for m in metrics]

        z_layer = idx * layer_gap
        color = palette[idx % len(palette)]

        px = [metrics[i] * max_radius * np.cos(angles[i]) for i in range(len(metric_names))]
        py = [metrics[i] * max_radius * np.sin(angles[i]) for i in range(len(metric_names))]
        pz = [z_layer] * len(metric_names)

        # 面填充：中心点 + 各顶点，使用三角扇方式
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
            f"<b>{disease}</b><br>"
            f"框数: {count}<br>"
            f"面积占比: {metrics_pct[1]:.1f}%<br>"
            f"均值置信: {metrics_pct[2]:.1f}%<br>"
            f"空间离散: {metrics_pct[3]:.1f}%<br>"
            f"中心偏移: {metrics_pct[4]:.1f}%"
        )
        fig.add_trace(go.Mesh3d(
            x=vx, y=vy, z=vz,
            i=i_idx, j=j_idx, k=k_idx,
            color=color,
            opacity=0.32,
            flatshading=True,
            hovertext=hover_text,
            hoverinfo="text",
            showlegend=False,
            lighting=dict(ambient=0.42, diffuse=0.8, specular=0.45, roughness=0.35),
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
            marker=dict(size=4.5, color=color, opacity=0.95),
            line=dict(color=color, width=6),
            hovertext=[f"{disease} - {metric_names[i % len(metric_names)]}: {metrics_pct[i % len(metric_names)]:.1f}%" for i in range(len(px_closed))],
            hoverinfo="text",
            showlegend=False,
        ))

        fig.add_trace(go.Scatter3d(
            x=[0], y=[0], z=[z_layer],
            mode="text",
            text=[disease],
            textfont=dict(size=13, color="#0f172a"),
            hoverinfo="skip",
            showlegend=False,
        ))

    _apply_3d_scene(
        fig,
        "病害多维画像",
        "多维投影 X",
        "多维投影 Y",
        "病害层",
        x_range=[-65, 65],
        y_range=[-65, 65],
        z_range=[-5, max(10.0, (len(ordered) - 1) * layer_gap + 16)],
    )
    fig.update_layout(font=dict(color="#0f172a"), height=640)
    return fig


def _build_model_profile_3d(records):
    fig = go.Figure()
    metric_names = ["mAP50", "mAP50-95", "Precision", "Recall", "识别速度得分", "参数规模得分", "综合效率"]
    # 明确起点：从左上方向开始，顺时针排布
    start_angle = np.deg2rad(125.0)
    angles = np.linspace(start_angle, start_angle - 2 * np.pi, len(metric_names), endpoint=False)
    max_radius = 200.0
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
        ax = max_radius * np.cos(angles[axis_idx])
        ay = max_radius * np.sin(angles[axis_idx])
        fig.add_trace(go.Scatter3d(
            x=[0, ax], y=[0, ay], z=[0, 0],
            mode="lines+text",
            line=dict(color="rgba(210,220,230,0.34)", width=3),
            text=["", axis_name],
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
        lambda r: min(max(r.get("param_scale_score", 0.0), 0.0), 1.0),
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
            f"Precision: {metrics_pct[2]:.1f}%<br>"
            f"Recall: {metrics_pct[3]:.1f}%<br>"
            f"推理总耗时: {rec.get('infer_total_ms', 0.0):.2f} ms<br>"
            f"FPS: {rec.get('fps', 0.0):.2f}<br>"
            f"识别速度得分: {metrics_pct[4]:.1f}%<br>"
            f"参数规模得分: {metrics_pct[5]:.1f}%<br>"
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
        x_range=[-(max_radius + 70.0), (max_radius + 70.0)],
        y_range=[-(max_radius + 60.0), (max_radius + 60.0)],
        z_range=[0, max(30.0, len(ordered) * layer_gap + 34.0)],
    )
    fig.update_layout(scene_camera=dict(eye=dict(x=2.45, y=1.95, z=1.70)))
    # 去掉 XYZ 刻度与网格，避免干扰多维环刻度阅读
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

        infer_total_ms = _to_float(benchmark.get("mean_total_ms"), 0.0)
        p95_total_ms = _to_float(benchmark.get("p95_total_ms"), 0.0)
        fps = _to_float(benchmark.get("fps"), 0.0)
        bench_device = benchmark.get("device", "-")
        bench_samples = _to_int(benchmark.get("sample_count"), 0)

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

    # 归一化分数：
    # 1) 识别速度得分：mean_total_ms 越低得分越高（来自 benchmark）
    # 2) 参数规模得分：参数量越大得分越高
    if records:
        speed_values = [r["infer_total_ms"] for r in records if r["infer_total_ms"] > 0]
        if speed_values:
            t_min, t_max = min(speed_values), max(speed_values)
        else:
            t_min, t_max = 0.0, 0.0

        param_values = [r["params_m"] for r in records if r["params_m"] is not None]
        if param_values:
            p_min, p_max = min(param_values), max(param_values)
        else:
            p_min, p_max = 0.0, 0.0

        for rec in records:
            t = rec["infer_total_ms"]
            if t > 0 and t_max > t_min:
                speed_score = (t_max - t) / (t_max - t_min)
            elif t > 0:
                speed_score = 1.0
            else:
                speed_score = 0.5

            p = rec["params_m"]
            if p is not None and p_max > p_min:
                param_scale_score = (p - p_min) / (p_max - p_min)
            elif p is not None:
                param_scale_score = 1.0
            else:
                param_scale_score = 0.5

            # 综合效率：准确率主导 + 训练速度 + 参数规模
            efficiency_score = (
                rec["map50_95"] * 0.45
                + rec["precision"] * 0.20
                + rec["recall"] * 0.15
                + speed_score * 0.12
                + param_scale_score * 0.08
            )
            rec["speed_score"] = float(min(max(speed_score, 0.0), 1.0))
            rec["param_scale_score"] = float(min(max(param_scale_score, 0.0), 1.0))
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
            round(rec.get("param_scale_score", 0.5), 4),
            round(rec.get("efficiency_score", 0.0), 4),
        ]
        for rec in records
    ]

    fig = _build_model_profile_3d(records)
    return table_rows, fig

# --------------------------------------------------------------------------------
# 核心预测逻辑
# --------------------------------------------------------------------------------
def smart_diagnosis(image, model_choice, conf_threshold, use_sam, sam_model, sam_device, use_ai, api_key, ai_model):
    """
    一站式诊断：流式输出版 (YOLO -> SAM -> 3D -> AI)
    """
    if image is None:
        fig = go.Figure()
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        profile_fig = go.Figure()
        profile_fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        yield None, None, "请上传一张叶片图片", "请上传图片后重试", fig, profile_fig
        return

    # 初始化所有返回值，作为初始流输出
    yolo_plot_rgb = None
    analysis_img = None
    stats_info = "⌛ 正在启动推理引擎..."
    ai_suggestion = "等待中..."
    fig = go.Figure()
    profile_fig = go.Figure()

    try:
        # 1. 释放显存
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except: pass

        # 2. YOLO 检测 (秒开)
        model_path = MODEL_OPTIONS[model_choice][1] if isinstance(model_choice, int) else model_choice
        model = YOLO(model_path)
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        results = model.predict(source=img_bgr, imgsz=512, conf=conf_threshold, verbose=False)
        yolo_plot_rgb = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)
        analysis_img = yolo_plot_rgb # 暂时作为占位
        
        boxes = results[0].boxes
        detections = {}
        disease_areas = {}
        disease_conf_sum = {}
        disease_conf_count = {}
        disease_centers = {}
        
        if boxes is not None and len(boxes) > 0:
            # 关键修复：一次性将所有检测结果转为 CPU 上的 Python 类型，避免后续混入 Tensor
            cls_ids = boxes.cls.cpu().tolist()
            confs = boxes.conf.cpu().tolist()
            xyxy_list = boxes.xyxy.cpu().tolist()
            
            for i, cls_id in enumerate(cls_ids):
                name = model.names[int(cls_id)]
                detections[name] = detections.get(name, 0) + 1
                
                box = xyxy_list[i] # 这是一个包含 4 个 float 的列表
                area = (box[2] - box[0]) * (box[3] - box[1])
                disease_areas[name] = disease_areas.get(name, 0) + float(area)
                
                conf = float(confs[i])
                disease_conf_sum[name] = disease_conf_sum.get(name, 0.0) + conf
                disease_conf_count[name] = disease_conf_count.get(name, 0) + 1
                
                ih, iw = image.shape[:2]
                cx = ((box[0] + box[2]) * 0.5) / iw
                cy = ((box[1] + box[3]) * 0.5) / ih
                disease_centers.setdefault(name, []).append((cx, cy))
            
            stats_info = "✅ YOLO 已完成定位\n" + "\n".join([f"  • {k}: {v} 处" for k, v in detections.items()])
        else:
            stats_info = "⚪ 未检出病斑"
            ai_suggestion = "叶片表现健康，建议定期观察。"

        # 【第一次输出】：用户能立刻看到带框的图和文字统计
        yield yolo_plot_rgb, analysis_img, stats_info, ai_suggestion, fig, profile_fig

        # 3. 如果开启 SAM 面积分析 (稍慢)
        if boxes is not None and len(boxes) > 0 and use_sam:
            stats_info += "\n\n⌛ 正在执行 SAM 像素级量化..."
            yield yolo_plot_rgb, analysis_img, stats_info, ai_suggestion, fig, profile_fig
            
            boxes_xyxy = boxes.xyxy.detach().cpu().numpy()
            overlay, sam_info = analyze_disease_extent(image, boxes_xyxy, model_type=sam_model, device=sam_device)
            if overlay is not None:
                analysis_img = overlay
                stats_info = stats_info.replace("\n\n⌛ 正在执行 SAM 像素级量化...", "\n\n" + sam_info)
                # 【第二次输出】：更新分割图和面积百分比
                yield yolo_plot_rgb, analysis_img, stats_info, ai_suggestion, fig, profile_fig
            
            # 4. 生成 3D 图表 (瞬时)
            stats_info += "\n\n⌛ 正在生成 3D 空间画像..."
            yield yolo_plot_rgb, analysis_img, stats_info, ai_suggestion, fig, profile_fig

            fig = _build_disease_prism_fig(detections, disease_areas) # 内部封装逻辑
            profile_fig = _build_disease_profile_3d(detections, disease_areas, disease_conf_sum, disease_conf_count, disease_centers, image.shape)
            
            stats_info = stats_info.replace("\n\n⌛ 正在生成 3D 空间画像...", "\n\n✅ 3D 量化看板已就绪")
            # 【第三次输出】：更新 3D 图表
            yield yolo_plot_rgb, analysis_img, stats_info, ai_suggestion, fig, profile_fig

            # 5. 如果开启 AI 智能分析 (最慢)
            if use_ai:
                if not api_key:
                    ai_suggestion = "⚠️ 请填写 API Key 以开启 AI 分析"
                else:
                    ai_suggestion = "⌛ AI 专家正在深度诊断，请稍候..."
                    yield yolo_plot_rgb, analysis_img, stats_info, ai_suggestion, fig, profile_fig
                    
                    disease_str = ", ".join(detections.keys())
                    prompt = f"我的植物叶片检测到了以下病害：{disease_str}。请结合这些病害给出专业的诊断报告和防治建议。"
                    real_model_id = MODELS.get(ai_model, ai_model)
                    print(f"--- 正在调用 AI 模型: {real_model_id} ---")
                    ai_suggestion = ask_multimodal(image, prompt, api_key, real_model_id)
            else:
                ai_suggestion = "AI 诊断已关闭，请勾选左侧“启用文本建议”"
            
            # 【最终输出】：所有数据全量更新
            yield yolo_plot_rgb, analysis_img, stats_info, ai_suggestion, fig, profile_fig

    except Exception as e:
        import traceback
        traceback.print_exc()
        # 发生错误时，确保返回 6 个值，避免前端崩溃
        err_fig = go.Figure()
        err_fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        yield yolo_plot_rgb, analysis_img, f"❌ 系统异常: {e}", "诊断失败", err_fig, err_fig

def _build_disease_prism_fig(detections, disease_areas):
    """
    内部辅助：构建 3D 柱体分布图 (恢复完整视觉特效)
    """
    fig = go.Figure()
    palette = ["#34D399", "#F59E0B", "#60A5FA", "#F472B6", "#22D3EE", "#A3E635", "#FB7185"]
    
    if not detections:
        fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode="markers+text", text=["未检出病斑"]))
        _apply_3d_scene(fig, "病害分布", "序列", "数量", "面积", x_range=[-1, 1], y_range=[-1, 1], z_range=[0, 20])
        return fig

    diseases = list(detections.keys())
    counts = [detections[d] for d in diseases]
    areas = [disease_areas.get(d, 0.0) for d in diseases]
    max_area = max(areas) if areas else 1.0
    max_count = max(counts) if counts else 1
    x_positions = [i * 1.25 for i in range(len(diseases))]

    # 1. 恢复地面网格
    floor_x = np.linspace(-0.8, max(x_positions) + 0.8, 14)
    floor_y = np.linspace(0.0, max_count + 1.8, 10)
    gx, gy = np.meshgrid(floor_x, floor_y)
    gz = np.zeros_like(gx)
    fig.add_trace(go.Surface(
        x=gx, y=gy, z=gz, showscale=False,
        colorscale=[[0, "#171b22"], [1, "#2b3340"]],
        opacity=0.72, hoverinfo="skip"
    ))

    top_x, top_y, top_z, top_text, top_color = [], [], [], [], []
    
    # 2. 恢复动态缩放的柱体
    for idx, disease in enumerate(diseases):
        count = counts[idx]
        area = areas[idx]
        height = max(10.0, (area / max_area) * 100.0)
        width = 0.48 + min(0.85, count * 0.11)
        depth = 0.48 + min(0.85, count * 0.11)
        y_pos = count * 0.95 + 0.5
        color = palette[idx % len(palette)]

        hover_text = f"<b>{disease}</b><br>数量: {count}<br>面积: {area:.0f}px"
        _add_prism(fig, x_positions[idx], y_pos, width, depth, height, color, hover_text, disease)

        # 准备顶部菱形数据
        top_x.append(x_positions[idx])
        top_y.append(y_pos)
        top_z.append(height + 2.4)
        top_text.append(disease)
        top_color.append(color)

    # 3. 恢复顶部菱形快速标记
    fig.add_trace(go.Scatter3d(
        x=top_x, y=top_y, z=top_z,
        mode="markers+text",
        marker=dict(symbol="diamond", size=10, color=top_color, opacity=0.9),
        text=top_text, textposition="top center", showlegend=False
    ))

    _apply_3d_scene(fig, "病害多维空间分布量化图", 
                    "病害类别索引 (Disease Category)", 
                    "病斑分布丰度 (Lesion Abundance)", 
                    "相对受损面积 (Relative Lesion Area)", 
                    x_range=[-0.8, max(x_positions) + 0.8], 
                    y_range=[0, max_count + 2.0], 
                    z_range=[0, 130])
    return fig


# --------------------------------------------------------------------------------
# 训练相关逻辑
# --------------------------------------------------------------------------------
TRAIN_PROC = None
TRAIN_LOG_PATH = ROOT / "train_realtime.log"

def start_training(model_name, epochs, batch, imgsz, device, exp_name):
    global TRAIN_PROC
    if TRAIN_PROC and TRAIN_PROC.poll() is None:
        return "⚠️ 训练已经在运行中！"
    
    if TRAIN_LOG_PATH.exists():
        TRAIN_LOG_PATH.unlink()
        
    cmd = [
        str(sys.executable), "-m", "ultralytics", "train",
        f"model={model_name}",
        f"data={DATA_YAML}",
        f"epochs={epochs}",
        f"batch={batch}",
        f"imgsz={imgsz}",
        f"device={device}",
        f"name={exp_name}",
        "exist_ok=True"
    ]
    
    try:
        with open(TRAIN_LOG_PATH, "w") as f:
            TRAIN_PROC = subprocess.Popen(
                cmd, stdout=f, stderr=subprocess.STDOUT,
                cwd=str(ROOT),
                env={**os.environ, "PYTHONUNBUFFERED": "1", "NO_PROXY": "127.0.0.1,localhost"}
            )
        return f"🚀 训练已启动 (PID: {TRAIN_PROC.pid})"
    except Exception as e:
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

.app-header .sub {
    margin-top: 4px;
    font-size: 0.95rem;
    color: #cbd5e1;
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
    color: #94a3b8;
    font-size: 0.86rem;
    margin-bottom: 10px;
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
demo = gr.Blocks(title="智慧农业：植物病害终端")

with demo:
    gr.HTML("""
        <div class="app-header">
            <h1>植物病害监测终端</h1>
            <div class="sub">检测、分割、训练和模型对比在一个工作台内完成</div>
        </div>
    """)

    with gr.Tabs():
        with gr.TabItem("诊断"):
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Group(elem_classes=["panel"]):
                        gr.Markdown("### 输入与参数")
                        gr.HTML('<div class="section-note">上传叶片图像后可直接运行，右侧将同步更新定位、分割和 3D 统计。</div>')
                        input_img = gr.Image(label="叶片图像", type="numpy", height=320)

                        with gr.Row():
                            model_dd = gr.Dropdown(
                                choices=[opt[0] for opt in MODEL_OPTIONS],
                                value=MODEL_OPTIONS[0][0] if MODEL_OPTIONS else None,
                                label="检测模型识别引擎",
                                type="index",
                                scale=5,  # 缩小一点
                            )
                            conf_sld = gr.Slider(
                                minimum=0.01, maximum=0.9, value=0.25, step=0.01, label="置信度阈值", scale=4  # 扩大约 30%
                            )

                        use_sam_cb = gr.Checkbox(label="启用 SAM 分割", value=True)
                        with gr.Row():
                            sam_type = gr.Dropdown(choices=["vit_b", "vit_l", "vit_h"], value="vit_b", label="SAM 型号")
                            sam_dev = gr.Radio(choices=["auto", "cuda", "cpu"], value="auto", label="设备")

                        use_ai_cb = gr.Checkbox(label="启用文本建议", value=False)
                        with gr.Row():
                            ai_api_key = gr.Textbox(label="API Key", type="password", value=DEFAULT_API_KEY, placeholder="sk-...")
                            ai_model_dd = gr.Dropdown(
                                choices=list(MODELS.keys()),
                                value=list(MODELS.keys())[0],
                                label="建议模型",
                            )

                        run_btn = gr.Button("运行诊断", elem_classes=["primary-btn"], size="lg")

                with gr.Column(scale=2):
                    with gr.Row():
                        with gr.Group(elem_classes=["panel"]):
                            gr.Markdown("#### 检测框")
                            yolo_res = gr.Image(label="YOLO 输出", height=300)
                        with gr.Group(elem_classes=["panel"]):
                            gr.Markdown("#### 分割结果")
                            sam_res = gr.Image(label="SAM 输出", height=300)

                    with gr.Row():
                        with gr.Group(elem_classes=["panel"]):
                            gr.Markdown("#### 3D 病害空间图")
                            gr.HTML('<div class="section-note">柱体高度表示受损面积，柱体尺度表示病斑数量，顶端菱形用于快速定位病害类别。</div>')
                            plotly_chart = gr.Plot(label="病害空间图")
                        with gr.Group(elem_classes=["panel"]):
                            gr.Markdown("#### 3D 多维多边形图")
                            gr.HTML('<div class="section-note">每种病害一层：框数、面积占比、均值置信、空间离散、中心偏移。</div>')
                            disease_profile_chart = gr.Plot(label="病害多维画像")

                    with gr.Row():
                        with gr.Group(elem_classes=["panel"]):
                            gr.Markdown("#### 检测摘要")
                            stats_out = gr.Textbox(label="", lines=7, show_label=False)
                        with gr.Group(elem_classes=["panel"]):
                            gr.Markdown("#### 处理建议")
                            ai_out = gr.Textbox(label="", lines=7, show_label=False)

            run_btn.click(
                fn=smart_diagnosis,
                inputs=[input_img, model_dd, conf_sld, use_sam_cb, sam_type, sam_dev, use_ai_cb, ai_api_key, ai_model_dd],
                outputs=[yolo_res, sam_res, stats_out, ai_out, plotly_chart, disease_profile_chart],
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
                        tr_name = gr.Textbox(value="train_v8x_new", label="实验名")

                        with gr.Row():
                            tr_btn = gr.Button("启动训练", variant="primary", elem_classes=["primary-btn"])
                            tr_stop = gr.Button("停止训练", variant="stop")
                        tr_fb = gr.Textbox(label="执行状态", lines=2)

                    with gr.Group(elem_classes=["panel"]):
                        gr.Markdown("### 训练日志")
                        tr_status = gr.Textbox(label="运行状态", value="空闲")
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
                            bench_warmup = gr.Slider(minimum=0, maximum=50, value=20, step=5, label="Warmup", scale=1)
                        bench_fb = gr.Textbox(label="测速状态", lines=2, value="尚未执行识别测速")

                        metrics_df = gr.Dataframe(
                            headers=[
                                "模型名称", "基础权重", "轮数 (Epochs)", "批次 (Batch)", "尺寸 (ImgSz)",
                                "mAP50", "mAP50-95", "Precision", "Recall",
                                "平均每轮训练耗时", "推理总耗时(ms)", "FPS", "P95推理耗时(ms)", "测速设备", "测速样本数",
                                "参数量(M)", "识别速度得分", "参数规模得分", "综合效率"
                            ],
                            interactive=False,
                        )
                        gr.HTML('<div class="section-note">每个模型一层多边形：精度、召回、识别速度、参数规模与综合效率同时可见。图中环形刻度是相对对比刻度（已做鲁棒缩放），用于避免强模型把其余模型压扁成贴地形态。</div>')
                        metrics_plot = gr.Plot(label="模型多维画像")

            tr_btn.click(fn=start_training, inputs=[tr_base, tr_ep, tr_bs, tr_sz, gr.State("0"), tr_name], outputs=[tr_fb])
            tr_stop.click(fn=stop_training, outputs=[tr_fb])
            tr_timer.tick(fn=get_train_log, outputs=[tr_log, tr_status])
            demo.load(fn=refresh_model_metrics, outputs=[metrics_df, metrics_plot])
            refresh_btn.click(fn=refresh_model_metrics, outputs=[metrics_df, metrics_plot])
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
