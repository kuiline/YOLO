from __future__ import annotations

import argparse
from pathlib import Path

plt = None
FancyArrowPatch = None
FancyBboxPatch = None


def load_matplotlib() -> None:
    global plt, FancyArrowPatch, FancyBboxPatch
    if plt is not None:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot
    from matplotlib.patches import FancyArrowPatch as ArrowPatch
    from matplotlib.patches import FancyBboxPatch as BboxPatch

    plt = pyplot
    FancyArrowPatch = ArrowPatch
    FancyBboxPatch = BboxPatch


def setup_font() -> None:
    load_matplotlib()
    plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def box(ax, xy, width, height, label, fill="#F3F4F6", fontsize=9):
    rect = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.04",
        facecolor=fill,
        edgecolor="#222222",
        linewidth=1.1,
    )
    ax.add_patch(rect)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, label, ha="center", va="center", fontsize=fontsize)


def arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=12, linewidth=1.1, color="#222222"))


def draw_architecture(output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")

    layers = [
        ("Web 交互层", "图像上传 / 模型选择 / 结果展示", 5.7),
        ("诊断解释层", "多模态诊断 / 文本建议生成", 4.45),
        ("算法处理层", "YOLO 检测 / SAM 分割 / 指标计算", 3.2),
        ("模型资源层", "检测权重 / 分割权重 / 调用配置", 1.95),
        ("数据存储层", "数据集 / 训练结果 / 本地文件", 0.7),
    ]
    colors = ["#EAF2FF", "#EEF7ED", "#FFF5DB", "#F4EDFF", "#F2F2F2"]
    for (title, detail, y), color in zip(layers, colors):
        box(ax, (0.6, y), 8.8, 0.85, f"{title}\n{detail}", color, fontsize=10)
    for y in [5.7, 4.45, 3.2, 1.95]:
        arrow(ax, (5, y), (5, y - 0.37))

    fig.tight_layout()
    fig.savefig(output_dir / "system_architecture.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def draw_diagnosis_flow(output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    items = [
        ((0.5, 2.4), "输入图像"),
        ((2.6, 3.6), "YOLO\n病害检测"),
        ((2.6, 1.2), "SAM/SAM2\n叶片分割"),
        ((5.2, 3.6), "检测框统计\n类别与置信度"),
        ((5.2, 1.2), "面积占比\n空间分布"),
        ((8.0, 2.4), "结果融合\n量化分析"),
        ((10.0, 2.4), "界面展示\n诊断建议"),
    ]
    for xy, label in items:
        box(ax, xy, 1.55, 0.9, label, "#F8FAFC")

    arrows = [
        ((2.05, 2.85), (2.6, 4.05)),
        ((2.05, 2.85), (2.6, 1.65)),
        ((4.15, 4.05), (5.2, 4.05)),
        ((4.15, 1.65), (5.2, 1.65)),
        ((6.75, 4.05), (8.0, 3.1)),
        ((6.75, 1.65), (8.0, 2.65)),
        ((9.55, 2.85), (10.0, 2.85)),
    ]
    for start, end in arrows:
        arrow(ax, start, end)

    fig.tight_layout()
    fig.savefig(output_dir / "diagnosis_flow.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate basic thesis diagrams.")
    parser.add_argument("--output-dir", default="thesis_figures", help="Output directory.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_font()
    draw_architecture(output_dir)
    draw_diagnosis_flow(output_dir)
    print(f"Figures written to {output_dir}")


if __name__ == "__main__":
    main()
