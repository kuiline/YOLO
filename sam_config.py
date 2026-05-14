"""
SAM 基础配置与模型加载
用于后续接入叶片分割 / 病斑辅助分析
"""

import os
from pathlib import Path
from functools import lru_cache

ROOT = Path(__file__).parent
SAM_DIR = ROOT / "sam_models"
SAM_DIR.mkdir(exist_ok=True)

# 确保 sam2 的配置文件夹在搜索路径中
SAM2_BASE = ROOT / "sam2-main"
SAM2_CONFIG_DIR = SAM2_BASE / "sam2" / "configs" / "sam2.1"

SAM_CHECKPOINTS = {
    "SAM 1": SAM_DIR / "sam_vit_b_01ec64.pth",
    "SAM 2.1": SAM_DIR / "sam2.1_hiera_large.pt",
    # 兼容旧代码
    "vit_b": SAM_DIR / "sam_vit_b_01ec64.pth",
    "vit_l": SAM_DIR / "sam_vit_l_0b3195.pth",
    "vit_h": SAM_DIR / "sam_vit_h_4b8939.pth",
}

# SAM 2.1 配置文件映射
SAM2_CONFIGS = {
    "SAM 2.1": "sam2.1_hiera_l.yaml"
}


def get_sam_checkpoint(model_type: str = "SAM 2.1") -> Path:
    if model_type not in SAM_CHECKPOINTS:
        raise ValueError(f"不支持的 SAM 模型类型: {model_type}")
    return SAM_CHECKPOINTS[model_type]


def sam_status(model_type: str = "SAM 2.1") -> tuple[bool, str]:
    try:
        ckpt = get_sam_checkpoint(model_type)
        if ckpt.exists():
            return True, f"SAM 权重已就绪: {ckpt}"
        return False, f"SAM 权重未找到，请下载到: {ckpt}"
    except Exception as e:
        return False, str(e)


@lru_cache(maxsize=3)
def load_sam_model(model_type: str = "SAM 2.1", device: str = "cuda"):
    """延迟加载 SAM 模型。支持 SAM 1 (segment_anything) 和 SAM 2.1 (sam2)。"""
    ckpt = get_sam_checkpoint(model_type)
    if not ckpt.exists():
        raise FileNotFoundError(f"SAM 权重不存在: {ckpt}")

    if "SAM 2.1" in model_type:
        # 修正 Hydra 路径问题
        from hydra import compose, initialize_config_dir
        from sam2.build_sam import build_sam2
        
        config_name = SAM2_CONFIGS.get(model_type, "sam2.1_hiera_l.yaml")
        # 移除 .yaml 后缀，因为 hydra.compose 不需要它
        config_base = config_name.replace(".yaml", "")
        
        # 绝对路径
        abs_config_dir = str(SAM2_CONFIG_DIR.resolve())
        
        # 注意：hydra 可能会因为多次 initialize 报错，这里用 try 保护
        try:
            from hydra.core.global_hydra import GlobalHydra
            if GlobalHydra.instance().is_initialized():
                GlobalHydra.instance().clear()
            initialize_config_dir(config_dir=abs_config_dir, version_base="1.2")
        except Exception as e:
            print(f"Hydra init warning: {e}")

        model = build_sam2(config_name, str(ckpt), device=device)
        return model
    else:
        from segment_anything import sam_model_registry
        # 如果是 SAM 1，model_type 映射回 vit_b 等
        m_key = model_type
        if m_key == "SAM 1": m_key = "vit_b"
        
        sam = sam_model_registry[m_key](checkpoint=str(ckpt))
        sam.to(device=device)
        return sam


def get_sam_predictor(model, model_type: str):
    """根据模型类型获取对应的 Predictor。"""
    if "SAM 2.1" in model_type:
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        return SAM2ImagePredictor(model)
    else:
        from segment_anything import SamPredictor
        return SamPredictor(model)

def get_sam_generator(model, model_type: str, **kwargs):
    """根据模型类型获取对应的 AutomaticMaskGenerator。"""
    if "SAM 2.1" in model_type:
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        return SAM2AutomaticMaskGenerator(model, **kwargs)
    else:
        from segment_anything import SamAutomaticMaskGenerator
        return SamAutomaticMaskGenerator(model, **kwargs)
