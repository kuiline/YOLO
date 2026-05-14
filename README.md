# YOLO 叶片病害检测项目

基于 Ultralytics YOLO 的植物叶片病害目标检测；集成 **Gradio Web 界面**（推理、训练监控、模型对比、SAM 分割等）与 **SAM 2** 辅助分割脚本。

---

## 项目结构说明

下面按「**从入口到数据**」的顺序说明：先看怎么跑，再看代码与资源各放哪里。

### 1. 总览（目录树）

大文件目录（图片、权重、训练产物）在树里用注释代替逐文件列出。

```
biyesheji/
├── app.py                      # Gradio 主程序：检测、训练面板、模型对比、智能诊断等
├── train.py                    # 命令行训练入口（Ultralytics）
├── predict.py                  # 命令行推理入口
├── sam_config.py               # SAM2 路径、权重目录、加载状态文案
├── sam_leaf_segment.py         # 叶片 SAM 分割 / 病害区域分析（供 app 或脚本调用）
├── multimodal_api.py           # 多模态 / 外部 API 相关逻辑（由 app 引用）
├── requirements.txt
├── README.md
├── .gitignore
│
├── run_app.bat                 # Windows：一键启动 Gradio（调用 venv 中的 python）
├── run_app.ps1
├── run_train.ps1               # 可选：命令行训练快捷脚本
│
├── datasets/                   # 数据集（训练/验证/测试）
│   ├── data.yaml               # 类别数、路径、类别名（YOLO 数据集配置）
│   ├── images/                 # train / val / test 子目录，放 jpg/png
│   └── labels/                 # 与 images 同名的 .txt，YOLO 检测框格式
│
├── runs/                       # 训练与实验输出（Ultralytics 默认 + 自定义子目录）
│   ├── detect/                 # 各次 train 结果：weights/best.pt、results.csv、args.yaml 等
│   └── …                       # 其他对比实验目录（如 leaf_compare_*，含 summary、图表等）
│
├── sam2-main/                  # Meta SAM 2 官方源码（本地子模块/拷贝，供 pip -e 或 sys.path）
├── sam_models/                 # SAM2 权重文件放置目录（需自行下载，见 app 内说明）
│
├── venv/                       # Python 虚拟环境（本地创建，通常不入库）
│
└── *.pt                        # 根目录预训练权重（YOLOv8/v9/v10、YOLO11 等，训练或演示用）
```

**阅读提示**：日常改界面逻辑看 `app.py`；只训模型看 `train.py` + `datasets/` + `runs/detect/`；接 SAM 看 `sam_config.py` 与 `sam_leaf_segment.py`。

---

### 2. 根目录主要文件（做什么用）

| 文件 | 作用（一句话） |
|------|----------------|
| `app.py` | 项目主入口：上传图片检测、训练启动与日志、多模型对比表与 3D 图、SAM 分割与量化诊断等。 |
| `train.py` | 封装 `yolo train`，指定 `data.yaml`、epochs、model 路径等，结果写入 `runs/detect/…`。 |
| `predict.py` | 封装 `yolo predict`，对单张图或目录批量出框。 |
| `sam_config.py` | 把 `sam2-main` 加入 `sys.path`、拼接 `sam_models` 下权重路径、返回加载失败时的提示文案。 |
| `sam_leaf_segment.py` | 叶片分割、病害区域占比等图像逻辑，供 Gradio 或离线脚本复用。 |
| `multimodal_api.py` | 与「多模态 / 问答 / 外部接口」相关的后端封装。 |
| `requirements.txt` | pip 依赖列表；含 `gradio`、`ultralytics` 及 SAM2 所需项等。 |
| `run_app.bat` / `run_app.ps1` | 在已创建 `venv` 的前提下启动 `app.py`。 |
| `run_train.ps1` | 可选：统一训练参数、工作目录的 PowerShell 示例。 |

---

### 3. `datasets/`（数据从哪里来、和谁对应）

| 路径 | 说明 |
|------|------|
| `datasets/data.yaml` | 声明 `path`、`train`/`val`/`test` 相对路径、`nc` 类别数、`names` 类别名；`train.py` 默认读它。 |
| `datasets/images/{train,val,test}/` | 原始图像；文件名（不含扩展名）决定对应标签名。 |
| `datasets/labels/{train,val,test}/` | 每张图一个同主文件名的 `.txt`；每行：`class_id x_center y_center width height`（0–1 归一化）。 |

图片与标签必须**成对出现**，例如 `000048.jpg` ↔ `000048.txt`。

---

### 4. `runs/`（训练结果与对比实验）

| 路径 | 说明 |
|------|------|
| `runs/detect/<实验名>/` | Ultralytics 一次训练的完整目录：`weights/best.pt`、`results.csv`、`args.yaml`、曲线图等；**Gradio「训练与评估」会递归扫描**其中的 `best.pt` 与 `benchmark.json`（若有）。 |
| `runs/<自定义实验名>/` | 非 Ultralytics 标准结构时，可放自生成的 `summary.json`、`per_image.csv`、预览图等，用于论文图表或界面展示（视 `app.py` 是否引用而定）。 |

删除某次训练目录会导致界面中**不再出现该模型**；删前请确认不再需要该权重。

---

### 5. SAM 2 相关目录

| 路径 | 说明 |
|------|------|
| `sam2-main/` | SAM 2 官方 Python 包源码；安装方式二选一：`pip install -e ./sam2-main` 或依赖 `sam_config.py` 动态加路径。 |
| `sam_models/` | 放置如 `sam2.1_hiera_large.pt` 等权重；**需自行下载**到该目录，否则界面中 SAM 功能会提示缺文件。 |

---

### 6. 环境与权重（根目录 `venv/`、`*.pt`）

| 路径 | 说明 |
|------|------|
| `venv/` | 本地虚拟环境；`.gitignore` 通常忽略；换机器后需重新 `python -m venv venv` 并 `pip install -r requirements.txt`。 |
| 根目录 `*.pt` | Ultralytics 官方预训练权重（如 `yolov8n.pt`）；训练时作 `--model` 起点，或推理时作 `--weights`。体积大，**Git 常忽略**，拷贝项目时记得一并带上。 |

---

## 数据集说明

- **类别**：5 种病害 — Rust（锈病）、Mosaic（花叶病）、Grey_spot（灰斑病）、Brown_Spot（褐斑病）、Alternaria_Boltch（链格孢叶斑病）
- **标注格式**：YOLO 格式（每行：`class_id x_center y_center width height`，归一化坐标 0–1）
- **图片与标签对应**：图片文件名需与标签文件名一致（扩展名不同），如 `000048.jpg` 对应 `000048.txt`

---

## 快速开始

### 1. 环境准备

```bash
# 创建虚拟环境（推荐）
python -m venv venv
venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

若使用 SAM 2 源码目录，可在虚拟环境中执行（可选）：

```bash
pip install -e ./sam2-main
```

### 2. 准备图片数据

确保图片已放入对应目录，并与 `labels` 中的标注文件一一对应：

- `datasets/images/train/` — 训练图片（.jpg / .png）
- `datasets/images/val/` — 验证图片
- `datasets/images/test/` — 测试图片

### 3. 启动 Web 界面（推荐演示方式）

在项目根目录执行：

```bash
venv\Scripts\activate
python app.py
```

或使用 `run_app.bat` / `run_app.ps1`（需已配置好 `venv`）。

### 4. 命令行训练

```bash
python train.py
```

可选参数（在 `train.py` 中修改或通过命令行）：

- `--epochs 100`：训练轮数
- `--batch 16`：批次大小（根据显存调整）
- `--imgsz 640`：输入图像尺寸
- `--model yolo11n.pt`：预训练模型（n/s/m/l/x 表示从小到大）

### 5. 命令行推理预测

训练完成后，使用最佳权重进行预测：

```bash
python predict.py --source 图片路径或文件夹路径 --weights runs/detect/train/weights/best.pt
```

---

## 常用命令示例

```bash
# 使用小模型快速验证
python train.py --model yolo11n.pt --epochs 10

# 使用中等模型获得更好效果
python train.py --model yolo11s.pt --epochs 100 --batch 8

# 摄像头实时检测
python predict.py --source 0 --weights runs/detect/train/weights/best.pt
```

---

## 注意事项

- 首次训练若本地无对应 `*.pt`，Ultralytics 可能自动下载预训练权重（需网络）。
- 若显存不足，可减小 `--batch`（如 4 或 8）。
- 默认训练结果在 `runs/detect/<任务名>/` 下；Gradio 会从 `runs/detect` **递归**收集模型用于对比与诊断。
- `train_realtime.log` 若在训练时生成，仅为实时日志，可删除，不影响功能。
