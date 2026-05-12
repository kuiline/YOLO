"""
花叶病增强分析
基于 SAM 叶片主体分割，在叶片区域内部提取黄绿斑驳异常区域，
用于辅助识别 YOLO 难以稳定检测的大面积 Mosaic / 花叶病。
"""

from __future__ import annotations

import cv2
import numpy as np

from sam_leaf_segment import _overlay_mask, _pick_device
from sam_config import load_sam_model, sam_status


def _largest_leaf_mask(image: np.ndarray, model_type: str = "vit_b", device: str = "auto") -> tuple[np.ndarray | None, str]:
    ok, msg = sam_status(model_type)
    if not ok:
        return None, msg

    try:
        from segment_anything import SamAutomaticMaskGenerator

        dev = _pick_device(device)
        sam = load_sam_model(model_type=model_type, device=dev)
        generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=24,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            min_mask_region_area=500,
        )
        masks = generator.generate(image)
        if not masks:
            return None, "SAM 未找到叶片主体"
        best = max(masks, key=lambda x: x.get("area", 0))
        return best["segmentation"].astype(bool), f"叶片区域提取完成，设备: {dev}"
    except Exception as e:
        return None, f"叶片区域提取失败: {type(e).__name__}: {e}"


def analyze_mosaic(image: np.ndarray, model_type: str = "vit_b", device: str = "auto"):
    if image is None:
        return None, None, "请先上传一张叶片图片"

    leaf_mask, leaf_msg = _largest_leaf_mask(image, model_type=model_type, device=device)
    if leaf_mask is None:
        return None, None, leaf_msg

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    h = hsv[:, :, 0].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32)
    v = hsv[:, :, 2].astype(np.float32)

    leaf_h = h[leaf_mask]
    leaf_s = s[leaf_mask]
    leaf_v = v[leaf_mask]

    h_mean = float(np.mean(leaf_h))
    s_mean = float(np.mean(leaf_s))
    v_mean = float(np.mean(leaf_v))

    abnormal = leaf_mask & (
        ((np.abs(h - h_mean) > 12) & (s < s_mean + 10)) |
        ((v > v_mean + 18) & (s < s_mean + 5)) |
        ((v < v_mean - 20) & (s < s_mean + 15))
    )

    abnormal = abnormal.astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    abnormal = cv2.morphologyEx(abnormal, cv2.MORPH_OPEN, kernel)
    abnormal = cv2.morphologyEx(abnormal, cv2.MORPH_CLOSE, kernel)
    abnormal_mask = abnormal > 0

    leaf_area = int(leaf_mask.sum())
    abnormal_area = int(abnormal_mask.sum())
    ratio = abnormal_area / leaf_area if leaf_area > 0 else 0.0

    overlay = image.copy()
    overlay = _overlay_mask(overlay, leaf_mask, color=(60, 180, 255), alpha=0.18)
    overlay = _overlay_mask(overlay, abnormal_mask, color=(255, 60, 140), alpha=0.45)

    mask_img = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    mask_img[abnormal_mask] = [255, 255, 255]

    if ratio >= 0.22:
        level = "重度疑似花叶"
    elif ratio >= 0.10:
        level = "中度疑似花叶"
    elif ratio >= 0.04:
        level = "轻度疑似花叶"
    else:
        level = "未见明显大面积花叶特征"

    info = (
        "花叶增强分析完成\n"
        f"- {leaf_msg}\n"
        f"- 叶片面积: {leaf_area} px\n"
        f"- 异常区域面积: {abnormal_area} px\n"
        f"- 异常覆盖率: {ratio:.2%}\n"
        f"- 辅助结论: {level}"
    )
    return overlay, mask_img, info
