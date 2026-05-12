import os
import sys
import subprocess
import time
import threading
import signal
from pathlib import Path

import gradio as gr
import numpy as np
import cv2
from ultralytics import YOLO

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
# 核心预测逻辑
# --------------------------------------------------------------------------------
def smart_diagnosis(image, model_choice, conf_threshold, use_sam, sam_model, sam_device, use_ai, api_key, ai_model):
    """
    一站式诊断：YOLO -> SAM (面积占比) -> AI 分析
    """
    if image is None:
        return None, None, "请上传一张叶片图片", "请上传图片后重试"
    
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
        
        # 4. 初始化结果
        analysis_img = yolo_plot_rgb
        stats_info = "⚪ 未检测到病害"
        ai_suggestion = "等待分析..."
        
        if boxes is not None and len(boxes) > 0:
            for cls_id in boxes.cls:
                name = model.names[int(cls_id)]
                detections[name] = detections.get(name, 0) + 1
            
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
        
        return yolo_plot_rgb, analysis_img, stats_info, ai_suggestion

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, None, f"诊断过程出错: {e}", "分析失败"

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
# UI 构建
# --------------------------------------------------------------------------------
demo = gr.Blocks(theme="soft", title="植物叶片病害智能分析系统")

with demo:
    gr.HTML("""
        <div style="text-align: center; padding: 20px;">
            <h1 style="color: #2D5A27;">🌿 植物叶片病害智能分析系统</h1>
            <p style="color: #666;">基于 YOLOv8x + SAM + 多模态大模型</p>
        </div>
    """)

    with gr.Tabs():
        # 选项卡 1：智能诊断中心
        with gr.TabItem("🔬 智能诊断中心"):
            with gr.Row():
                with gr.Column(scale=1):
                    input_img = gr.Image(label="上传叶片图片", type="numpy", height=400)
                    with gr.Group():
                        gr.Markdown("### 🛠️ 诊断配置")
                        model_dd = gr.Dropdown(
                            choices=[opt[0] for opt in MODEL_OPTIONS],
                            value=MODEL_OPTIONS[0][0] if MODEL_OPTIONS else "yolov8x.pt",
                            label="核心检测模型",
                            type="index"
                        )
                        conf_sld = gr.Slider(minimum=0.01, maximum=0.9, value=0.25, step=0.01, label="检测置信度")
                        use_sam_cb = gr.Checkbox(label="启用 SAM 精细分割与占比计算", value=True)
                        with gr.Row():
                            sam_type = gr.Dropdown(choices=["vit_b", "vit_l", "vit_h"], value="vit_b", label="SAM 规模")
                            sam_dev = gr.Radio(choices=["auto", "cuda", "cpu"], value="auto", label="设备")
                        gr.Markdown("---")
                        use_ai_cb = gr.Checkbox(label="启用 AI 专家深度诊断", value=False)
                        ai_api_key = gr.Textbox(label="SiliconFlow API Key", type="password", placeholder="sk-...")
                        ai_model_dd = gr.Dropdown(choices=["Kimi-K2.5 (Pro)", "GLM-4.6V"], value="Kimi-K2.5 (Pro)", label="AI 模型")
                    run_btn = gr.Button("🔍 开始一站式诊断", variant="primary", size="lg")

                with gr.Column(scale=2):
                    with gr.Row():
                        yolo_res = gr.Image(label="YOLO 定位结果", height=350)
                        sam_res = gr.Image(label="SAM 面积分析结果", height=350)
                    with gr.Row():
                        with gr.Column(scale=1):
                            stats_out = gr.Textbox(label="📊 量化分析报告", lines=10)
                        with gr.Column(scale=2):
                            ai_out = gr.Textbox(label="🤖 AI 专家防治建议", lines=10)

            run_btn.click(
                fn=smart_diagnosis,
                inputs=[input_img, model_dd, conf_sld, use_sam_cb, sam_type, sam_dev, use_ai_cb, ai_api_key, ai_model_dd],
                outputs=[yolo_res, sam_res, stats_out, ai_out]
            )

        # 选项卡 2：模型训练
        with gr.TabItem("🏋️ 模型训练"):
            gr.Markdown("## 模型自主训练控制台")
            with gr.Row():
                with gr.Column(scale=1):
                    tr_base = gr.Textbox(value="yolov8x.pt", label="基础权重")
                    tr_ep = gr.Slider(10, 300, 100, step=10, label="轮数")
                    tr_bs = gr.Slider(4, 64, 16, step=4, label="批量大小")
                    tr_sz = gr.Dropdown(["512", "640", "800"], value="512", label="尺寸")
                    tr_name = gr.Textbox(value="train_v8x_new", label="实验名称")
                    with gr.Row():
                        tr_btn = gr.Button("🚀 启动训练", variant="primary")
                        tr_stop = gr.Button("🛑 停止", variant="stop")
                    tr_fb = gr.Textbox(label="状态反馈")
                
                with gr.Column(scale=2):
                    tr_status = gr.Textbox(label="运行状态", value="⚪ 等待中")
                    tr_log = gr.Textbox(label="实时日志", lines=20, autoscroll=True)
                    tr_timer = gr.Timer(value=3)

            tr_btn.click(fn=start_training, inputs=[tr_base, tr_ep, tr_bs, tr_sz, gr.State("0"), tr_name], outputs=[tr_fb])
            tr_stop.click(fn=stop_training, outputs=[tr_fb])
            tr_timer.tick(fn=get_train_log, outputs=[tr_log, tr_status])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
