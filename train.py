"""
YOLO 目标检测模型训练脚本
支持通过命令行参数选择底模和基础训练参数
"""

from __future__ import annotations

import argparse

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO 训练脚本")
    parser.add_argument("--model", type=str, default="yolov8x.pt", help="模型权重名或路径")
    parser.add_argument("--data", type=str, default="datasets/data.yaml", help="数据集配置文件")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--imgsz", type=int, default=512, help="输入图像尺寸")
    parser.add_argument("--batch", type=int, default=8, help="批大小")
    parser.add_argument("--device", type=str, default="0", help="训练设备，如 0 / cpu")
    parser.add_argument("--workers", type=int, default=8, help="数据加载线程")
    parser.add_argument("--project", type=str, default="runs/detect", help="输出目录")
    parser.add_argument("--name", type=str, default="train_v8x", help="实验名称")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--exist-ok", action="store_true", default=True, help="允许覆盖同名实验")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        cache=False,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
        pretrained=True,
        optimizer="auto",
        verbose=True,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
