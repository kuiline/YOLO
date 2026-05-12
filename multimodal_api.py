"""
植物病害多模态分析核心 API 模块 (增强版)
支持 SiliconFlow API，具备自动重试、详尽错误处理和多模态适配功能。
"""

import base64
import json
import io
import time
import requests
import numpy as np
from PIL import Image
from typing import Optional, Dict, Any, List

# --- 全局配置 ---
API_URL = "https://api.siliconflow.cn/v1/chat/completions"
DEFAULT_API_KEY = "sk-tjiwupnjibyvegdommtafazxynchqxykdedlwrwenjhfizty"

MODELS = {
    "Kimi-K2.6 (视觉)": "Pro/moonshotai/Kimi-K2.6",
    "DeepSeek-V4 (纯文本)": "deepseek-ai/DeepSeek-V4-Flash",
    "GLM-5.1 (纯文本)": "Pro/zai-org/GLM-5.1",
    "MiniMax-M2.5 (纯文本)": "Pro/MiniMaxAI/MiniMax-M2.5",
    "Qwen-3.5 (纯文本/视觉)": "Qwen/Qwen3.5-397B-A17B",
}

DEFAULT_MODEL = MODELS["Kimi-K2.6 (视觉)"]

DEFAULT_PROMPT = """你是一个专业的植物病理学家。
请仔细观察这张图片中的植物叶片，结合 YOLO 检测到的病害结果（如果有），给出：
1. 病害症状的详细描述。
2. 科学的防治建议（物理防治、生物防治、化学防治）。
3. 预防措施，防止病害再次发生。"""

class SiliconFlowError(Exception):
    """自定义 API 异常类"""
    def __init__(self, message, error_type=None, status_code=None):
        super().__init__(message)
        self.error_type = error_type
        self.status_code = status_code

def image_to_base64(image: np.ndarray) -> str:
    """将 numpy 图片转为 base64 字符串，包含异常处理"""
    try:
        if image is None:
            return ""
        pil_img = Image.fromarray(image.astype(np.uint8))
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"--- [图片转换失败]: {e} ---")
        return ""

def call_vision_api(
    image: np.ndarray,
    prompt: str = DEFAULT_PROMPT,
    api_key: str = "",
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    max_tokens: int = 1500,
    max_retries: int = 3,
) -> str:
    """
    调用 SiliconFlow API (增强型)。
    具备自动重试机制、超时管理和详尽的错误解析。
    """
    key = api_key or DEFAULT_API_KEY
    if not key:
        return "❌ 错误: 未配置 API Key。"

    # 识别视觉模型
    vision_keywords = ["Kimi", "GLM-4V", "vlm", "vision", "Qwen"]
    is_vision_model = any(k in model for k in vision_keywords)

    # 准备 Payload
    try:
        if is_vision_model:
            b64 = image_to_base64(image)
            if not b64:
                content = prompt + "\n(注：图片转换失败，已按纯文本模式诊断)"
            else:
                content = [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                ]
        else:
            content = prompt

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False # 暂不支持流式返回到 UI
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key.strip()}",
        }
    except Exception as e:
        return f"❌ 构造请求失败: {e}"

    # 带重试机制的请求循环
    last_error = ""
    for attempt in range(max_retries):
        try:
            print(f"--- [AI 请求] 尝试 {attempt+1}/{max_retries}, 模型: {model} ---")
            
            # 使用较长的读取超时，因为 VLM 推理较慢
            response = requests.post(
                API_URL, 
                json=payload, 
                headers=headers, 
                timeout=(10, 150) # (连接超时, 读取超时)
            )
            
            # 状态码处理
            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"].strip()
            
            # 错误解析
            error_msg = f"HTTP {response.status_code}"
            try:
                err_json = response.json()
                error_msg = err_json.get("error", {}).get("message", error_msg)
            except:
                error_msg = response.text or error_msg
            
            # 如果是限速或服务器忙，触发重试
            if response.status_code in [429, 500, 502, 503, 504]:
                wait_time = (attempt + 1) * 3
                print(f"--- [API 繁忙] {error_msg}, {wait_time}秒后重试... ---")
                time.sleep(wait_time)
                last_error = error_msg
                continue
            else:
                # 业务错误（如 Key 错、余额不足等）直接返回
                return f"❌ AI 服务商报错: {error_msg}"

        except requests.exceptions.Timeout:
            last_error = "请求响应超时 (Wait Timeout)"
            print(f"--- [超时] 尝试 {attempt+1} 失败 ---")
        except requests.exceptions.RequestException as e:
            last_error = f"网络连接异常: {str(e)}"
            print(f"--- [网络错误] {e} ---")
            
        time.sleep(2) # 基础重试间隔

    return f"❌ AI 诊断失败 (已尝试{max_retries}次): {last_error}"

# 别名兼容旧代码
ask_multimodal = call_vision_api
