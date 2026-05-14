"""
植物病害多模态分析核心 API 模块
支持 SiliconFlow OpenAI 兼容接口：视觉单轮、文本单轮、两轮视觉交叉验证 + JSON 输出解析。
"""

import base64
import io
import json
import re
import time
from typing import Any, Dict, List

import numpy as np
import requests
from PIL import Image

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


def is_vision_capable(model_id: str) -> bool:
    """与 call_vision_api 一致：用于判断是否可传图。"""
    vision_keywords = ("Kimi", "GLM-4V", "vlm", "vision", "Qwen")
    return any(k in model_id for k in vision_keywords)


def extract_json_object(text: str) -> Any:
    """从模型回复中截取第一个 JSON 对象并 json.loads。"""
    s = text.strip()
    if "```" in s:
        m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s, re.I)
        if m:
            s = m.group(1).strip()
    i0 = s.find("{")
    i1 = s.rfind("}")
    if i0 == -1 or i1 <= i0:
        raise ValueError("模型输出中未找到 JSON 对象")
    return json.loads(s[i0 : i1 + 1])


def image_to_base64(image: np.ndarray) -> str:
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


def chat_completion(
    messages: List[Dict[str, Any]],
    api_key: str,
    model: str,
    *,
    temperature: float = 0.7,
    max_tokens: int = 1500,
    max_retries: int = 3,
) -> str:
    key = (api_key or DEFAULT_API_KEY or "").strip()
    if not key:
        return "❌ 错误: 未配置 API Key。"

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}",
    }

    last_error = ""
    for attempt in range(max_retries):
        try:
            print(f"--- [AI 请求] 尝试 {attempt + 1}/{max_retries}, 模型: {model} ---")
            response = requests.post(
                API_URL,
                json=payload,
                headers=headers,
                timeout=(10, 150),
            )
            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"].strip()

            error_msg = f"HTTP {response.status_code}"
            try:
                err_json = response.json()
                error_msg = err_json.get("error", {}).get("message", error_msg)
            except Exception:
                error_msg = response.text or error_msg

            if response.status_code in (429, 500, 502, 503, 504):
                wait_time = (attempt + 1) * 3
                print(f"--- [API 繁忙] {error_msg}, {wait_time}秒后重试... ---")
                time.sleep(wait_time)
                last_error = error_msg
                continue
            return f"❌ AI 服务商报错: {error_msg}"

        except requests.exceptions.Timeout:
            last_error = "请求响应超时 (Wait Timeout)"
            print(f"--- [超时] 尝试 {attempt + 1} 失败 ---")
        except requests.exceptions.RequestException as e:
            last_error = f"网络连接异常: {str(e)}"
            print(f"--- [网络错误] {e} ---")

        time.sleep(2)

    return f"❌ AI 诊断失败 (已尝试{max_retries}次): {last_error}"


def call_vision_api(
    image: np.ndarray | None,
    prompt: str = DEFAULT_PROMPT,
    api_key: str = "",
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    max_tokens: int = 1500,
    max_retries: int = 3,
) -> str:
    """单轮：可选图像 + 文本 user 消息。"""
    is_vision = is_vision_capable(model)
    try:
        if is_vision and image is not None:
            b64 = image_to_base64(image)
            if not b64:
                content: str | List[Dict[str, Any]] = prompt + "\n(注：图片转换失败，已按纯文本模式诊断)"
            else:
                content = [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ]
        else:
            content = prompt

        messages = [{"role": "user", "content": content}]
        return chat_completion(
            messages,
            api_key,
            model,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
        )
    except Exception as e:
        return f"❌ 构造请求失败: {e}"


# --- 两轮诊断提示词（仅 JSON，控制长度）---

PROMPT_ROUND1_VISION_ONLY = """你是植物叶片病害分析助手。下面会给你一张叶片图像。
重要：当前**尚未**向你提供任何 YOLO/SAM 算法输出，请你**仅凭图像**判断。

请只输出一个 JSON 对象（不要使用 Markdown 代码围栏，不要输出任何 JSON 以外的文字）。字段如下：
{
  "round": 1,
  "visual_candidates": [
    {"name_zh": "可能病害中文名或暂无法确定", "likelihood": "高|中|低", "brief_basis": "不超过35字"}
  ],
  "symptoms_observed": "可见症状，不超过100字",
  "uncertainty": "不确定性说明，不超过60字"
}
visual_candidates 最多 4 条。"""


PROMPT_ROUND2_FUSION = """你是植物叶片病害分析助手。下面再次提供**同一张**叶片图像供你核对。

【第一轮你仅依据图像给出的 JSON】
{round1_json}

【算法侧客观结果（YOLO 检测 + SAM 量化，可能与第一轮不一致）】
{detector_block}

请交叉验证：对比第一轮可能病害与 YOLO 类别、SAM 叶内病斑比等；给出综合结论。

只输出一个 JSON 对象（不要使用 Markdown 代码围栏，不要输出 JSON 以外文字）。字段如下：
{
  "round": 2,
  "symptoms": "综合症状描述，不超过120字",
  "severity": "必须是以下之一：轻|中|重|不确定",
  "loss_estimate": "对产量或品质影响，不超过50字",
  "treatment": "防治建议，不超过100字",
  "prevention": "预防措施，不超过100字",
  "cross_validation": "与第一轮及算法结果的对照说明，不超过120字",
  "agreement_with_yolo": "必须是以下之一：一致|部分一致|不一致|不确定",
  "agreement_with_sam_quant": "必须是以下之一：一致|部分一致|不适用|不确定"
}"""


PROMPT_VISION_NO_DETECTOR = """你是植物叶片病害分析助手。自动检测器 YOLO **在本图未检出病斑框**（可能存在漏检或病斑过小）。
请**仅凭图像**判断叶片是否异常，并给出建议。

只输出一个 JSON 对象（不要使用 Markdown 代码围栏）。字段如下：
{
  "mode": "no_yolo_boxes",
  "symptoms": "症状描述，不超过120字",
  "severity": "轻|中|重|不确定",
  "loss_estimate": "粗略估计，不超过50字",
  "treatment": "防治建议，不超过100字",
  "prevention": "预防措施，不超过100字",
  "note": "说明不确定性及是否建议人工复查，不超过80字"
}"""


PROMPT_TEXT_ONLY_FUSION = """你是植物叶片病害分析助手。当前模型**无法读取图像**，以下为算法给出的检测与量化文字摘要，请据此做综合判断（结论仍可能不完整）。

【算法摘要】
{ctx}

只输出一个 JSON 对象（不要使用 Markdown 代码围栏）。字段如下：
{
  "mode": "text_only_model",
  "symptoms": "不超过120字",
  "severity": "轻|中|重|不确定",
  "loss_estimate": "不超过50字",
  "treatment": "不超过100字",
  "prevention": "不超过100字",
  "cross_validation": "对摘要可靠性的说明，不超过100字",
  "agreement_with_yolo": "一致|部分一致|不一致|不确定",
  "agreement_with_sam_quant": "一致|部分一致|不适用|不确定"
}"""


def run_llm_leaf_diagnosis(
    image: np.ndarray,
    api_key: str,
    ui_model_key: str,
    *,
    has_detector_boxes: bool,
    detector_context: str,
) -> Dict[str, Any]:
    """
    LLM 辅助诊断编排：
    - 无检出框：单轮视觉 JSON。
    - 有检出框 + 视觉模型：两轮（先纯视觉 JSON，再融合 YOLO+SAM 文本 + 同图）。
    - 有检出框 + 非视觉模型：单轮文本融合 JSON。
    """
    model = MODELS.get(ui_model_key, ui_model_key)
    out: Dict[str, Any] = {"mode": None, "round1": None, "round2": None, "warnings": []}

    if not (api_key or DEFAULT_API_KEY or "").strip():
        out["error"] = "未配置 API Key"
        return out

    try:
        if not has_detector_boxes:
            out["mode"] = "vision_only_no_detection"
            raw = call_vision_api(
                image,
                PROMPT_VISION_NO_DETECTOR,
                api_key,
                model,
                temperature=0.35,
                max_tokens=900,
            )
            if raw.startswith("❌"):
                out["error"] = raw
                return out
            try:
                out["round2"] = extract_json_object(raw)
            except Exception as e:
                out["round2"] = {"_raw": raw, "_parse_error": str(e)}
            return out

        if not is_vision_capable(model):
            out["mode"] = "text_only_single"
            out["warnings"].append("当前所选模型不支持图像输入，已改为单轮文本融合（含检测与量化摘要）。")
            prompt = PROMPT_TEXT_ONLY_FUSION.replace("{ctx}", detector_context.strip() or "（无摘要）")
            raw = chat_completion(
                [{"role": "user", "content": prompt}],
                api_key,
                model,
                temperature=0.35,
                max_tokens=1000,
            )
            if raw.startswith("❌"):
                out["error"] = raw
                return out
            try:
                out["round2"] = extract_json_object(raw)
            except Exception as e:
                out["round2"] = {"_raw": raw, "_parse_error": str(e)}
            return out

        out["mode"] = "two_round_vision"
        r1_raw = call_vision_api(
            image,
            PROMPT_ROUND1_VISION_ONLY,
            api_key,
            model,
            temperature=0.35,
            max_tokens=700,
        )
        if r1_raw.startswith("❌"):
            out["error"] = r1_raw
            return out
        try:
            out["round1"] = extract_json_object(r1_raw)
        except Exception:
            out["round1"] = {"_raw": r1_raw, "_parse_error": "round1 JSON 解析失败"}

        r1_compact = json.dumps(out["round1"], ensure_ascii=False)[:3800]
        block = detector_context.strip() or "（无结构化检测摘要）"
        p2 = PROMPT_ROUND2_FUSION.replace("{round1_json}", r1_compact).replace("{detector_block}", block)
        r2_raw = call_vision_api(
            image,
            p2,
            api_key,
            model,
            temperature=0.28,
            max_tokens=900,
        )
        if r2_raw.startswith("❌"):
            out["error"] = r2_raw
            return out
        try:
            out["round2"] = extract_json_object(r2_raw)
        except Exception as e:
            out["round2"] = {"_raw": r2_raw, "_parse_error": str(e)}
        return out

    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
        out["mode"] = "error"
        return out


# 别名兼容旧代码
ask_multimodal = call_vision_api
