import os
import sys
import subprocess
import time
import threading
import signal
import csv
from pathlib import Path

import gradio as gr
import numpy as np
import cv2
from ultralytics import YOLO
import plotly.graph_objects as go

# 导入自定义模块
from sam_config import sam_status
from sam_leaf_segment import segment_leaf, segment_from_boxes, analyze_disease_extent
from multimodal_api import ask_multimodal, call_vision_api
from mosaic_analyzer import analyze_mosaic

# --------------------------------------------------------------------------------
# 全局配置与状态
# --------------------------------------------------------------------------------
ROOT = Path(__file__).parent
DATA_YAML = str(ROOT / "datasets" / "data.yaml")

# 扫描已训练模型
def _scan_trained_models():
    options = []
    search_roots = [
        ROOT / "runs" / "detect" / "runs" / "detect",
        ROOT / "runs" / "detect",
    ]
    for base in search_roots:
        if not base.exists(): continue
        # 搜索所有 weights/best.pt
        for weights in sorted(base.glob("*/weights/best.pt")):
            name = weights.parent.parent.name
            options.append((f"自训练 [{name}]", str(weights)))
    
    # 加入基础模型
    for m in ["yolov8x.pt", "yolo11n.pt"]:
        options.append((f"预训练 [{m}]", m))
    
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
        return float(value)
    except Exception:
        return default


def _to_int(value, default=0):
    try:
        return int(float(value))
    except Exception:
        return default


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
                title=x_title,
                backgroundcolor="rgba(10,12,16,0.0)",
                gridcolor="rgba(240,240,240,0.16)",
                zerolinecolor="rgba(240,240,240,0.2)",
                showspikes=False,
                range=x_range,
            ),
            yaxis=dict(
                title=y_title,
                backgroundcolor="rgba(10,12,16,0.0)",
                gridcolor="rgba(240,240,240,0.16)",
                zerolinecolor="rgba(240,240,240,0.2)",
                showspikes=False,
                range=y_range,
            ),
            zaxis=dict(
                title=z_title,
                backgroundcolor="rgba(10,12,16,0.0)",
                gridcolor="rgba(240,240,240,0.2)",
                zerolinecolor="rgba(240,240,240,0.25)",
                showspikes=False,
                range=z_range,
            ),
            camera=dict(eye=dict(x=1.7, y=1.25, z=1.2)),
            aspectmode="manual",
            aspectratio=dict(x=1.5, y=1.0, z=1.15),
        ),
        margin=dict(l=0, r=0, b=0, t=36),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        title=dict(text=title_text, x=0.5, font=dict(size=16, color="#f4f6f8")),
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
            textfont=dict(size=12, color="#dbe4ee"),
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
            textfont=dict(size=13, color="#f8fafc"),
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
    fig.update_layout(font=dict(color="#e5e7eb"))
    return fig


def refresh_model_metrics():
    records = []
    train_root = ROOT / "runs" / "detect"
    if train_root.exists():
        for run_dir in sorted(train_root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if not run_dir.is_dir():
                continue
            results_csv = run_dir / "results.csv"
            args_yaml = run_dir / "args.yaml"
            if not results_csv.exists():
                continue

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

            model_name = run_dir.name
            base_weight = args.get("model", "unknown")
            epochs = _to_int(args.get("epochs", 0), 0)
            batch = _to_int(args.get("batch", 0), 0)
            imgsz = _to_int(args.get("imgsz", 0), 0)
            map50 = _to_float(last_row.get("metrics/mAP50(B)"), 0.0)
            map50_95 = _to_float(last_row.get("metrics/mAP50-95(B)"), 0.0)
            precision = _to_float(last_row.get("metrics/precision(B)"), 0.0)
            recall = _to_float(last_row.get("metrics/recall(B)"), 0.0)

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
                "cost": max(epochs, 1) * max(batch, 1) * ((max(imgsz, 1) / 512.0) ** 2),
            })

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
        ]
        for rec in records
    ]

    fig = go.Figure()
    if records:
        palette = ["#34D399", "#F59E0B", "#60A5FA", "#F472B6", "#22D3EE", "#A3E635"]
        max_cost = max(r["cost"] for r in records) if records else 1.0
        step = 1.35
        x_positions = [i * step for i in range(len(records))]

        floor_x = np.linspace(-0.8, max(x_positions) + 0.8, 16) if x_positions else np.linspace(-0.8, 0.8, 16)
        floor_y = np.linspace(0.0, 2.2, 8)
        gx, gy = np.meshgrid(floor_x, floor_y)
        gz = np.zeros_like(gx)
        fig.add_trace(go.Surface(
            x=gx,
            y=gy,
            z=gz,
            showscale=False,
            colorscale=[[0, "#151a20"], [1, "#222933"]],
            opacity=0.75,
            hoverinfo="skip",
        ))

        top_x, top_y, top_z, top_text, top_size, top_color = [], [], [], [], [], []
        for idx, rec in enumerate(records):
            height = max(8.0, rec["map50_95"] * 120.0)
            depth = 0.45 + min(0.9, rec["recall"] * 0.9)
            width = 0.45 + min(0.9, rec["precision"] * 0.9)
            y_pos = 0.55 + min(1.5, (rec["cost"] / max(max_cost, 1e-9)) * 1.5)
            color = palette[idx % len(palette)]

            hover_text = (
                f"<b>{rec['model_name']}</b><br>"
                f"基线: {rec['base_weight']}<br>"
                f"mAP50: {rec['map50']:.4f}<br>"
                f"mAP50-95: {rec['map50_95']:.4f}<br>"
                f"Precision: {rec['precision']:.4f}<br>"
                f"Recall: {rec['recall']:.4f}<br>"
                f"训练代价指数: {rec['cost']:.0f}"
            )
            _add_prism(fig, x_positions[idx], y_pos, width, depth, height, color, hover_text, rec["model_name"])

            top_x.append(x_positions[idx])
            top_y.append(y_pos)
            top_z.append(height + 2.2)
            top_text.append(rec["model_name"])
            top_size.append(6 + rec["precision"] * 12)
            top_color.append(color)

        fig.add_trace(go.Scatter3d(
            x=top_x,
            y=top_y,
            z=top_z,
            mode="markers+text",
            text=top_text,
            textposition="top center",
            marker=dict(
                size=top_size,
                color=top_color,
                symbol="diamond",
                opacity=0.95,
            ),
            hoverinfo="skip",
            showlegend=False,
        ))

        _apply_3d_scene(
            fig,
            "模型对比空间",
            "模型序列",
            "训练代价归一位",
            "mAP50-95 × 120",
            x_range=[-0.8, max(x_positions) + 0.8],
            y_range=[0.0, 2.3],
            z_range=[0, max(top_z) + 8.0] if top_z else None,
        )
    else:
        fig.add_trace(go.Scatter3d(
            x=[0.0], y=[0.0], z=[0.0],
            mode="markers+text",
            marker=dict(size=10, color="#A1A1AA"),
            text=["暂无训练数据"],
            textposition="top center",
            hoverinfo="skip",
            showlegend=False,
        ))
        _apply_3d_scene(
            fig,
            "模型对比空间",
            "模型序列",
            "训练代价归一位",
            "mAP50-95 × 120",
            x_range=[-1, 1],
            y_range=[-1, 1],
            z_range=[0, 20],
        )

    fig.update_layout(
        font=dict(color="#e5e7eb"),
    )
    return table_rows, fig

# --------------------------------------------------------------------------------
# 核心预测逻辑
# --------------------------------------------------------------------------------
def smart_diagnosis(image, model_choice, conf_threshold, use_sam, sam_model, sam_device, use_ai, api_key, ai_model):
    """
    一站式诊断：YOLO -> SAM (面积占比) -> AI 分析 -> Plotly 3D 量化图表
    """
    if image is None:
        fig = go.Figure()
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        profile_fig = go.Figure()
        profile_fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        return None, None, "请上传一张叶片图片", "请上传图片后重试", fig, profile_fig
    
    try:
        # 释放显存，防止 OOM
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass
        # 1. 加载 YOLO 模型
        model_path = MODEL_OPTIONS[model_choice][1] if isinstance(model_choice, int) else model_choice
        print(f"--- 正在调用模型路径: {model_path} ---")
        model = YOLO(model_path)
        
        # 2. 图像转换 (Gradio RGB -> OpenCV BGR)
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # 3. YOLO 预测
        results = model.predict(source=img_bgr, imgsz=512, conf=conf_threshold, verbose=False)
        yolo_plot_bgr = results[0].plot()
        yolo_plot_rgb = cv2.cvtColor(yolo_plot_bgr, cv2.COLOR_BGR2RGB)
        
        boxes = results[0].boxes
        detections = {}
        disease_areas = {}
        disease_conf_sum = {}
        disease_conf_count = {}
        disease_centers = {}
        
        # 4. 初始化结果
        analysis_img = yolo_plot_rgb
        stats_info = "⚪ 未检测到病害"
        ai_suggestion = "等待分析..."
        
        if boxes is not None and len(boxes) > 0:
            for i, cls_id in enumerate(boxes.cls):
                name = model.names[int(cls_id)]
                detections[name] = detections.get(name, 0) + 1
                
                # 估算病斑面积 (YOLO box 面积)
                box = boxes.xyxy[i]
                x1 = float(box[0])
                y1 = float(box[1])
                x2 = float(box[2])
                y2 = float(box[3])
                w = max(0.0, x2 - x1)
                h = max(0.0, y2 - y1)
                disease_areas[name] = disease_areas.get(name, 0) + (w * h)

                conf = float(boxes.conf[i]) if boxes.conf is not None else 0.0
                disease_conf_sum[name] = disease_conf_sum.get(name, 0.0) + conf
                disease_conf_count[name] = disease_conf_count.get(name, 0) + 1

                ih, iw = image.shape[:2]
                if iw > 0 and ih > 0:
                    cx = ((x1 + x2) * 0.5) / float(iw)
                    cy = ((y1 + y2) * 0.5) / float(ih)
                    disease_centers.setdefault(name, []).append((cx, cy))
            
            stats_info = "✅ YOLO 检测到：\n" + "\n".join([f"  • {k}: {v} 处" for k, v in detections.items()])
            
            # 5. 如果开启 SAM 面积分析
            if use_sam:
                boxes_xyxy = boxes.xyxy.detach().cpu().numpy()
                overlay, sam_info = analyze_disease_extent(image, boxes_xyxy, model_type=sam_model, device=sam_device)
                if overlay is not None:
                    analysis_img = overlay
                    stats_info += "\n\n" + sam_info
            
            # 6. 如果开启 AI 智能分析
            if use_ai:
                if not api_key:
                    ai_suggestion = "⚠️ 请填写 API Key 以开启 AI 分析"
                else:
                    disease_str = ", ".join(detections.keys())
                    prompt = f"我的植物叶片检测到了以下病害：{disease_str}。请结合这些病害给出专业的诊断报告和防治建议。"
                    ai_suggestion = ask_multimodal(image, prompt, api_key, ai_model)
                    
        # 7. 生成 3D 图表 (立体柱体：高度=面积、深度=数量)
        fig = go.Figure()
        palette = ["#34D399", "#F59E0B", "#60A5FA", "#F472B6", "#22D3EE", "#A3E635", "#FB7185"]

        if detections:
            diseases = list(detections.keys())
            counts = [detections[d] for d in diseases]
            areas = [disease_areas.get(d, 0.0) for d in diseases]
            max_area = max(areas) if areas else 1.0
            max_count = max(counts) if counts else 1

            x_positions = [i * 1.25 for i in range(len(diseases))]
            floor_x = np.linspace(-0.8, max(x_positions) + 0.8, 14)
            floor_y = np.linspace(0.0, max_count + 1.8, 10)
            gx, gy = np.meshgrid(floor_x, floor_y)
            gz = np.zeros_like(gx)
            fig.add_trace(go.Surface(
                x=gx,
                y=gy,
                z=gz,
                showscale=False,
                colorscale=[[0, "#171b22"], [1, "#2b3340"]],
                opacity=0.72,
                hoverinfo="skip",
            ))

            top_x, top_y, top_z, top_text, top_size, top_color = [], [], [], [], [], []
            for idx, disease in enumerate(diseases):
                count = counts[idx]
                area = areas[idx]
                area_height = (area / max(max_area, 1e-9)) * 100.0
                height = max(10.0, area_height)
                width = 0.48 + min(0.85, count * 0.11)
                depth = 0.48 + min(0.85, count * 0.11)
                y_pos = count * 0.95 + 0.5
                color = palette[idx % len(palette)]

                hover_text = (
                    f"<b>{disease}</b><br>"
                    f"病斑数量: {count}<br>"
                    f"受损面积估算: {area:.0f} px<br>"
                    f"面积归一高度: {height:.1f}"
                )
                _add_prism(fig, x_positions[idx], y_pos, width, depth, height, color, hover_text, disease)

                top_x.append(x_positions[idx])
                top_y.append(y_pos)
                top_z.append(height + 2.4)
                top_text.append(disease)
                top_size.append(8 + min(10, count * 1.2))
                top_color.append(color)

            fig.add_trace(go.Scatter3d(
                x=top_x,
                y=top_y,
                z=top_z,
                mode="markers+text",
                marker=dict(
                    symbol="diamond",
                    size=top_size,
                    color=top_color,
                    opacity=0.96,
                ),
                text=top_text,
                textposition="top center",
                hoverinfo="skip",
                showlegend=False,
            ))

            _apply_3d_scene(
                fig,
                "病害立体分布",
                "病害序列",
                "病斑数量位",
                "面积归一高度",
                x_range=[-0.8, max(x_positions) + 0.8],
                y_range=[0, max_count + 2.0],
                z_range=[0, max(top_z) + 8.0],
            )
        else:
            fig.add_trace(go.Scatter3d(
                x=[0], y=[0], z=[0], mode="markers+text",
                marker=dict(size=10, color="#86EFAC"),
                text=["未检出病斑"],
                textposition="top center",
                hoverinfo="skip",
                showlegend=False,
            ))
            _apply_3d_scene(
                fig,
                "病害立体分布",
                "病害序列",
                "病斑数量位",
                "面积归一高度",
                x_range=[-1, 1],
                y_range=[-1, 1],
                z_range=[0, 20],
            )

        profile_fig = _build_disease_profile_3d(
            detections,
            disease_areas,
            disease_conf_sum,
            disease_conf_count,
            disease_centers,
            image.shape,
        )

        return yolo_plot_rgb, analysis_img, stats_info, ai_suggestion, fig, profile_fig

    except Exception as e:
        import traceback
        traceback.print_exc()
        fig = go.Figure()
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        profile_fig = go.Figure()
        profile_fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        return None, None, f"诊断过程出错: {e}", "分析失败", fig, profile_fig

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
                                value=MODEL_OPTIONS[0][0] if MODEL_OPTIONS else "yolov8x.pt",
                                label="检测模型",
                                type="index",
                                scale=2,
                            )
                            conf_sld = gr.Slider(
                                minimum=0.01, maximum=0.9, value=0.25, step=0.01, label="置信度", scale=1
                            )

                        use_sam_cb = gr.Checkbox(label="启用 SAM 分割", value=True)
                        with gr.Row():
                            sam_type = gr.Dropdown(choices=["vit_b", "vit_l", "vit_h"], value="vit_b", label="SAM 型号")
                            sam_dev = gr.Radio(choices=["auto", "cuda", "cpu"], value="auto", label="设备")

                        use_ai_cb = gr.Checkbox(label="启用文本建议", value=False)
                        with gr.Row():
                            ai_api_key = gr.Textbox(label="API Key", type="password", placeholder="sk-...")
                            ai_model_dd = gr.Dropdown(
                                choices=["Kimi-K2.5 (Pro)", "GLM-4.6V"],
                                value="Kimi-K2.5 (Pro)",
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

                        metrics_df = gr.Dataframe(
                            headers=["模型名称", "基础权重", "轮数 (Epochs)", "批次 (Batch)", "尺寸 (ImgSz)", "mAP50", "mAP50-95", "Precision", "Recall"],
                            interactive=False,
                        )
                        gr.HTML('<div class="section-note">立体柱高度 = mAP50-95，柱体体积与训练代价相关，顶部菱形用于快速识别模型。</div>')
                        metrics_plot = gr.Plot(label="模型 3D 对比")

            tr_btn.click(fn=start_training, inputs=[tr_base, tr_ep, tr_bs, tr_sz, gr.State("0"), tr_name], outputs=[tr_fb])
            tr_stop.click(fn=stop_training, outputs=[tr_fb])
            tr_timer.tick(fn=get_train_log, outputs=[tr_log, tr_status])
            demo.load(fn=refresh_model_metrics, outputs=[metrics_df, metrics_plot])
            refresh_btn.click(fn=refresh_model_metrics, outputs=[metrics_df, metrics_plot])

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        theme=app_theme,
        css=custom_css,
    )
