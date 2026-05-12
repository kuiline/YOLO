"""
SAM 基础配置与模型加载
用于后续接入叶片分割 / 病斑辅助分析
"""

from pathlib import Path
from functools import lru_cache

ROOT = Path(__file__).parent
SAM_DIR = ROOT / "sam_models"
SAM_DIR.mkdir(exist_ok=True)

SAM_CHECKPOINTS = {
    "vit_b": SAM_DIR / "sam_vit_b_01ec64.pth",
    "vit_l": SAM_DIR / "sam_vit_l_0b3195.pth",
    "vit_h": SAM_DIR / "sam_vit_h_4b8939.pth",
}


def get_sam_checkpoint(model_type: str = "vit_b") -> Path:
    if model_type not in SAM_CHECKPOINTS:
        raise ValueError(f"不支持的 SAM 模型类型: {model_type}")
    return SAM_CHECKPOINTS[model_type]


def sam_status(model_type: str = "vit_b") -> tuple[bool, str]:
    ckpt = get_sam_checkpoint(model_type)
    if ckpt.exists():
        return True, f"SAM 权重已就绪: {ckpt}"
    return False, f"SAM 权重未找到，请下载到: {ckpt}"


@lru_cache(maxsize=3)
def load_sam_model(model_type: str = "vit_b", device: str = "cuda"):
    """延迟加载 SAM 模型，后续真正接入 UI 时直接复用。"""
    from segment_anything import sam_model_registry

    ckpt = get_sam_checkpoint(model_type)
    if not ckpt.exists():
        raise FileNotFoundError(f"SAM 权重不存在: {ckpt}")

    sam = sam_model_registry[model_type](checkpoint=str(ckpt))
    sam.to(device=device)
    return sam
