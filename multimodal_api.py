"""
多模态大模型 API 接入模块
SiliconFlow API - 支持 Kimi-K2.5 等视觉模型
"""

import base64
import json
import io
from typing import Optional

import numpy as np
import requests

API_URL = "https://api.siliconflow.cn/v1/chat/completions"
# 默认使用 Kimi-K2.5，支持视觉
DEFAULT_MODEL = "Pro/moonshotai/Kimi-K2.5"

# 预设提示词 - 叶片病害分析
DEFAULT_PROMPT = """请仔细观察这张植物叶片图片，分析其中可能存在的病害情况。
如果发现异常，请说明：
1. 病害类型或可能原因
2. 病害的严重程度（轻/中/重）
3. 简要的防治建议

如果没有明显病害，请说明叶片健康状况。"""


def image_to_base64(image: np.ndarray, format: str = "jpeg") -> str:
    """将 numpy 图片转为 base64 字符串"""
    from PIL import Image
    if image is None:
        raise ValueError("图片为空")
    pil_img = Image.fromarray(image.astype(np.uint8) if image.dtype != np.uint8 else image)
    buf = io.BytesIO()
    pil_img.save(buf, format=format.upper(), quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def call_vision_api(
    image: np.ndarray,
    prompt: str = DEFAULT_PROMPT,
    api_key: str = "",
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    max_tokens: int = 1000,
) -> str:
    """
    调用 SiliconFlow 多模态 API，传入图片和文本提示。
    
    Args:
        image: numpy 格式图片 (H, W, C)
        prompt: 用户提示词
        api_key: API Key，为空时从环境变量 SILICONFLOW_API_KEY 读取
        model: 模型名称
        temperature: 温度参数
        max_tokens: 最大输出 token
    
    Returns:
        模型返回的文本内容
    """
    import os
    key = api_key or os.environ.get("SILICONFLOW_API_KEY", "")
    if not key:
        return "❌ 请先设置 API Key：在输入框中填写，或设置环境变量 SILICONFLOW_API_KEY"
    
    try:
        b64 = image_to_base64(image)
        data_url = f"data:image/jpeg;base64,{b64}"
        
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": data_url,
                                "detail": "auto",
                            },
                        },
                    ],
                }
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        }
        
        resp = requests.post(API_URL, json=payload, headers=headers, timeout=60)
        resp.raise_for_status()
        
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        return content.strip()
    
    except requests.exceptions.Timeout:
        return "❌ 请求超时，请稍后重试"
    except requests.exceptions.RequestException as e:
        if hasattr(e, "response") and e.response is not None:
            try:
                err_data = e.response.json()
                msg = err_data.get("error", {}).get("message", str(e))
            except Exception:
                msg = e.response.text or str(e)
        else:
            msg = str(e)
        return f"❌ API 请求失败：{msg}"
    except (KeyError, IndexError) as e:
        return f"❌ 解析响应失败：{e}"
    except Exception as e:
        return f"❌ 错误：{type(e).__name__}: {e}"
# 别名，兼容旧版代码引用
ask_multimodal = call_vision_api
