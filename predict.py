"""
YOLO 目标检测推理/预测脚本
对单张图片、文件夹或摄像头进行目标检测
"""

from ultralytics import YOLO
import argparse


def predict_image(model_path: str, source: str, save: bool = True):
    """
    对图片或文件夹进行预测
    
    Args:
        model_path: 训练好的模型权重路径 (如 runs/detect/train/weights/best.pt)
        source: 预测来源 - 图片路径、文件夹路径、或 0(摄像头)
        save: 是否保存预测结果
    """
    model = YOLO(model_path)
    
    results = model.predict(
        source=source,
        imgsz=512,           # 与训练时保持一致（你的图片是512x512）
        save=save,           # 保存带标注的图片
        save_txt=True,       # 保存检测结果为txt
        save_conf=True,      # 保存置信度
        conf=0.25,           # 置信度阈值
        iou=0.7,             # NMS IoU阈值
        show=False,          # 设为True可实时显示（需要GUI）
    )
    
    return results


def main():
    parser = argparse.ArgumentParser(description="YOLO 目标检测推理")
    parser.add_argument(
        "--model",
        type=str,
        default="runs/detect/train/weights/best.pt",
        help="模型权重路径",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="datasets/images/test",  # 默认预测test集
        help="预测来源：图片路径、文件夹、或0(摄像头)",
    )
    parser.add_argument(
        "--nosave",
        action="store_true",
        help="不保存结果",
    )
    
    args = parser.parse_args()
    
    predict_image(
        model_path=args.model,
        source=args.source,
        save=not args.nosave,
    )


if __name__ == "__main__":
    main()
