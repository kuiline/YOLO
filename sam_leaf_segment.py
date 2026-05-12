"""
SAM 叶片分割与 YOLO+SAM 精细分割
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from sam_config import load_sam_model, sam_status


def _pick_device(device: str = "auto") -> str:
    if device != "auto":
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _overlay_mask(image: np.ndarray, mask: np.ndarray, color=(64, 220, 120), alpha: float = 0.45) -> np.ndarray:
    out = image.copy().astype(np.float32)
    color_arr = np.array(color, dtype=np.float32)
    out[mask] = out[mask] * (1.0 - alpha) + color_arr * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def segment_leaf(image: np.ndarray, model_type: str = "vit_b", device: str = "auto") -> Tuple[np.ndarray | None, np.ndarray | None, str]:
    if image is None:
        return None, None, "请先上传一张叶片图片"

    ok, msg = sam_status(model_type)
    if not ok:
        return None, None, msg

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
            return None, None, "SAM 未找到可用分割区域"

        best = max(masks, key=lambda x: x.get("area", 0))
        seg = best["segmentation"].astype(bool)

        mask_img = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
        mask_img[seg] = [255, 255, 255]
        overlay = _overlay_mask(image, seg)

        area = int(best.get("area", int(seg.sum())))
        ratio = area / float(image.shape[0] * image.shape[1])
        info = f"叶片主体分割完成\n- 模型: {model_type}\n- 设备: {dev}\n- mask 面积: {area} px\n- 画面占比: {ratio:.2%}"
        return overlay, mask_img, info
    except Exception as e:
        return None, None, f"SAM 分割失败: {type(e).__name__}: {e}"


def segment_from_boxes(image: np.ndarray, boxes_xyxy: np.ndarray, model_type: str = "vit_b", device: str = "auto") -> tuple[np.ndarray | None, np.ndarray | None, str]:
    """根据 YOLO 检测框调用 SAM 精细分割。"""
    if image is None:
        return None, None, "请先上传一张图片"
    if boxes_xyxy is None or len(boxes_xyxy) == 0:
        return image, None, "未检测到可用于 SAM 分割的目标框"

    ok, msg = sam_status(model_type)
    if not ok:
        return None, None, msg

    try:
        from segment_anything import SamPredictor

        dev = _pick_device(device)
        sam = load_sam_model(model_type=model_type, device=dev)
        predictor = SamPredictor(sam)
        predictor.set_image(image)

        merged = np.zeros(image.shape[:2], dtype=bool)
        overlay = image.copy()
        total_masks = 0

        for box in boxes_xyxy:
            masks, scores, _ = predictor.predict(
                box=np.asarray(box, dtype=np.float32),
                multimask_output=False,
            )
            if masks is None or len(masks) == 0:
                continue
            seg = masks[0].astype(bool)
            merged |= seg
            overlay = _overlay_mask(overlay, seg, color=(255, 80, 80), alpha=0.35)
            total_masks += 1

        if total_masks == 0:
            return image, None, "SAM 未能根据检测框生成有效分割结果"

        mask_img = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
        mask_img[merged] = [255, 255, 255]
        info = f"YOLO + SAM 分割完成\n- 模型: {model_type}\n- 设备: {dev}\n- 检测框数量: {len(boxes_xyxy)}\n- 有效 mask 数量: {total_masks}"
        return overlay, mask_img, info
    except Exception as e:
        return None, None, f"YOLO + SAM 分割失败: {type(e).__name__}: {e}"


def analyze_disease_extent(image: np.ndarray, boxes_xyxy: np.ndarray, model_type: str = "vit_b", device: str = "auto") -> tuple[np.ndarray | None, str]:
    """
    综合分析：提取叶片主体面积 vs 病斑分割面积，计算受损比例。
    """
    if image is None:
        return None, "未提供图片"
    
    dev = _pick_device(device)
    try:
        sam = load_sam_model(model_type=model_type, device=dev)
        
        # 1. 提取叶片主体 (使用自动掩码生成器)
        from segment_anything import SamAutomaticMaskGenerator, SamPredictor
        
        generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=8, # 大幅提速
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            min_mask_region_area=1000,
        )
        
        masks = generator.generate(image)
        if not masks:
            return None, "SAM 未能提取到叶片主体"
        
        # 取面积最大的作为叶片
        leaf_mask_data = max(masks, key=lambda x: x.get("area", 0))
        leaf_mask = leaf_mask_data["segmentation"].astype(bool)
        leaf_area = int(leaf_mask.sum())
        
        # 2. 提取病斑区域 (基于 YOLO 框)
        predictor = SamPredictor(sam)
        predictor.set_image(image)
        
        lesion_mask = np.zeros(image.shape[:2], dtype=bool)
        if boxes_xyxy is not None and len(boxes_xyxy) > 0:
            for box in boxes_xyxy:
                m, _, _ = predictor.predict(
                    box=np.asarray(box, dtype=np.float32),
                    multimask_output=False,
                )
                if m is not None and len(m) > 0:
                    lesion_mask |= m[0].astype(bool)
        
        # 3. 计算占比
        effective_lesion_mask = lesion_mask & leaf_mask
        lesion_area = int(effective_lesion_mask.sum())
        ratio = lesion_area / leaf_area if leaf_area > 0 else 0.0
        
        # 4. 可视化
        overlay = _overlay_mask(image, leaf_mask, color=(60, 180, 255), alpha=0.15)
        overlay = _overlay_mask(overlay, effective_lesion_mask, color=(255, 50, 50), alpha=0.45)
        
        info = (
            f"📊 精细化量化分析结果：\n"
            f"- 叶片总面积: {leaf_area:,} 像素\n"
            f"- 病变覆盖面积: {lesion_area:,} 像素\n"
            f"- **受损面积占比: {ratio:.2%}**"
        )
        return overlay, info

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"量化分析失败: {e}"
