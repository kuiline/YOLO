# YOLO 叶片病害检测项目

基于 Ultralytics YOLO 的植物叶片病害目标检测项目。

## 项目结构

```
biyesheji/
├── datasets/                 # 数据集目录
│   ├── data.yaml            # 数据集配置文件
│   ├── images/              # 图片目录
│   │   ├── train/           # 训练集图片
│   │   ├── val/             # 验证集图片
│   │   └── test/            # 测试集图片
│   └── labels/              # 标注文件（YOLO格式）
│       ├── train/
│       ├── val/
│       └── test/
├── runs/                     # 训练输出（自动生成）
├── train.py                  # 训练脚本
├── predict.py                # 推理/预测脚本
├── requirements.txt          # 依赖
└── README.md
```

## 数据集说明

- **类别**：5 种病害 - Rust（锈病）、Mosaic（花叶病）、Grey_spot（灰斑病）、Brown_Spot（褐斑病）、Alternaria_Boltch（ Alternaria 叶斑病）
- **标注格式**：YOLO 格式（每行：`class_id x_center y_center width height`，归一化坐标 0-1）
- **图片与标签对应**：图片文件名需与标签文件名一致（扩展名不同），如 `000048.jpg` 对应 `000048.txt`

## 快速开始

### 1. 环境准备

```bash
# 创建虚拟环境（推荐）
python -m venv venv
venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 准备图片数据

确保图片已放入对应目录，并与 labels 中的标注文件一一对应：

- `datasets/images/train/` - 训练图片（.jpg / .png）
- `datasets/images/val/` - 验证图片
- `datasets/images/test/` - 测试图片

### 3. 训练模型

```bash
python train.py
```

可选参数（在 train.py 中修改或通过命令行）：

- `--epochs 100`：训练轮数
- `--batch 16`：批次大小（根据显存调整）
- `--imgsz 640`：输入图像尺寸
- `--model yolo11n.pt`：预训练模型（n/s/m/l/x 表示从小到大）

### 4. 推理预测

训练完成后，使用最佳权重进行预测：

```bash
python predict.py --source 图片路径或文件夹路径 --weights runs/detect/train/weights/best.pt
```

## 常用命令示例

```bash
# 使用小模型快速验证
python train.py --model yolo11n.pt --epochs 10

# 使用中等模型获得更好效果
python train.py --model yolo11s.pt --epochs 100 --batch 8

# 摄像头实时检测
python predict.py --source 0 --weights runs/detect/train/weights/best.pt
```

## 注意事项

- 首次运行会自动下载预训练权重
- 若显存不足，可减小 `--batch`（如 4 或 8）
- 训练结果保存在 `runs/detect/train/` 下
