"""
SAM 叶片分割与 YOLO+SAM 精细分割
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from sam_config import load_sam_model, sam_status, get_sam_predictor, get_sam_generator


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


def _find_leaf_mask(masks: list, w: int, h: int) -> dict:
    """找出真正的叶片掩膜，避免把背景当成叶片（背景通常面积最大且贴满图像边缘）。"""
    best_mask = None
    best_score = -1
    
    for m in masks:
        area = m.get("area", 0)
        bx, by, bw, bh = m.get("bbox", [0, 0, w, h])
        
        # 判断是否为全局背景：占据极大面积，且几乎贴合图像的四个边界
        is_background = (
            bx <= int(w * 0.05) and 
            by <= int(h * 0.05) and 
            (bx + bw) >= int(w * 0.95) and 
            (by + bh) >= int(h * 0.95) and
            area > w * h * 0.4
        )
        
        if is_background:
            continue
            
        if area > best_score:
            best_score = area
            best_mask = m
            
    # 如果全被过滤掉了（比如整张图真的是一片巨大无比的叶子），兜底取最大面积
    if best_mask is None and masks:
        best_mask = max(masks, key=lambda x: x.get("area", 0))
        
    return best_mask

def segment_leaf(image: np.ndarray, model_type: str = "SAM 2.1", device: str = "auto") -> Tuple[np.ndarray | None, np.ndarray | None, str]:
    if image is None:
        return None, None, "请先上传一张叶片图片"

    ok, msg = sam_status(model_type)
    if not ok:
        return None, None, msg

    try:
        dev = _pick_device(device)
        sam = load_sam_model(model_type=model_type, device=dev)
        generator = get_sam_generator(
            sam, 
            model_type=model_type,
            points_per_side=24,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            min_mask_region_area=500,
        )

        masks = generator.generate(image)
        if not masks:
            return None, None, "SAM 未找到可用分割区域"

        best = _find_leaf_mask(masks, image.shape[1], image.shape[0])
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


def segment_from_boxes(image: np.ndarray, boxes_xyxy: np.ndarray, model_type: str = "SAM 2.1", device: str = "auto") -> tuple[np.ndarray | None, np.ndarray | None, str]:
    """根据 YOLO 检测框调用 SAM 精细分割。"""
    if image is None:
        return None, None, "请先上传一张图片"
    if boxes_xyxy is None or len(boxes_xyxy) == 0:
        return image, None, "未检测到可用于 SAM 分割的目标框"

    ok, msg = sam_status(model_type)
    if not ok:
        return None, None, msg

    try:
        dev = _pick_device(device)
        sam = load_sam_model(model_type=model_type, device=dev)
        predictor = get_sam_predictor(sam, model_type=model_type)
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


def analyze_disease_extent(
    image: np.ndarray, 
    boxes_xyxy: np.ndarray, 
    model_type: str = "SAM 2.1", 
    device: str = "auto",
    box_class_names: list[str] | None = None
) -> tuple[np.ndarray | None, str, dict | None]:
    """
    综合分析：提取叶片主体面积 vs 病斑分割面积，计算受损比例。
    """
    if image is None:
        return None, "未提供图片", None
    
    dev = _pick_device(device)
    try:
        sam = load_sam_model(model_type=model_type, device=dev)
        
        # 1. 提取叶片主体 (使用自动掩码生成器)
        generator = get_sam_generator(
            sam, 
            model_type=model_type,
            points_per_side=8, # 大幅提速
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            min_mask_region_area=1000,
        )
        
        masks = generator.generate(image)
        if not masks:
            return None, "SAM 未能提取到叶片主体", None
        
        # 找出真正的叶片掩膜，避免把背景当成叶片
        leaf_mask_data = _find_leaf_mask(masks, image.shape[1], image.shape[0])
        leaf_mask = leaf_mask_data["segmentation"].astype(bool)
        leaf_area = int(leaf_mask.sum())
        img_area = image.shape[0] * image.shape[1]
        leaf_frame_conf = leaf_area / img_area if img_area > 0 else 0.0
        
        # 2. 提取病斑区域 (基于 YOLO 框)
        predictor = get_sam_predictor(sam, model_type=model_type)
        predictor.set_image(image)
        
        lesion_mask = np.zeros(image.shape[:2], dtype=bool)
        by_disease = {}
        disease_masks = {}
        disease_box_areas = {}
        
        if boxes_xyxy is not None and len(boxes_xyxy) > 0:
            for idx, box in enumerate(boxes_xyxy):
                cls_name = box_class_names[idx] if box_class_names and idx < len(box_class_names) else "unknown"
                
                m, _, _ = predictor.predict(
                    box=np.asarray(box, dtype=np.float32),
                    multimask_output=False,
                )
                if m is not None and len(m) > 0:
                    seg = m[0].astype(bool)
                    lesion_mask |= seg
                    
                    if cls_name not in disease_masks:
                        disease_masks[cls_name] = np.zeros(image.shape[:2], dtype=bool)
                        disease_box_areas[cls_name] = 0.0
                        
                    disease_masks[cls_name] |= seg
                    
                    bx1, by1, bx2, by2 = map(int, box)
                    bx1, by1 = max(0, bx1), max(0, by1)
                    bx2, by2 = min(image.shape[1], bx2), min(image.shape[0], by2)
                    box_area = max(0, (bx2 - bx1) * (by2 - by1))
                    disease_box_areas[cls_name] += box_area
        
        # 3. 计算占比
        effective_lesion_mask = lesion_mask & leaf_mask
        lesion_area = int(effective_lesion_mask.sum())
        ratio = lesion_area / leaf_area if leaf_area > 0 else 0.0
        
        for cls_name, d_mask in disease_masks.items():
            d_effective = d_mask & leaf_mask
            d_lesion_area = int(d_effective.sum())
            d_ratio = d_lesion_area / leaf_area if leaf_area > 0 else 0.0
            
            box_area = disease_box_areas[cls_name]
            box_cons = d_lesion_area / box_area if box_area > 0 else 0.0
            
            by_disease[cls_name] = {
                "leaf_lesion_ratio": d_ratio,
                "box_mask_consistency": min(1.0, box_cons),
                "lesion_area_px": d_lesion_area,
                "leaf_area_px": leaf_area
            }
            
        sam_profile = {
            "leaf_frame_confidence": leaf_frame_conf,
            "by_disease": by_disease
        }
        
        # 4. 可视化
        overlay = _overlay_mask(image, leaf_mask, color=(255, 255, 0), alpha=0.45)
        overlay = _overlay_mask(overlay, lesion_mask, color=(255, 50, 50), alpha=0.45)
        
        info = (
            f"📊 精细化量化分析结果：\n"
            f"- 叶片总面积: {leaf_area:,} 像素\n"
            f"- 病变覆盖面积: {lesion_area:,} 像素\n"
            f"- **受损面积占比: {ratio:.2%}**"
        )
        return overlay, info, sam_profile

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"量化分析失败: {e}", None
