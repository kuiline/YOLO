# Thesis Tools

这些脚本用于论文写作、格式检查和图件生成。它们不依赖本机固定路径，适合跟随代码仓库同步到另一台电脑。

## render_docx.py

把 Word 文档渲染成 PDF 和逐页 PNG，用于检查页眉、页脚、目录、图表和整体版式。

```powershell
python tools/thesis/render_docx.py thesis.docx --output-dir rendered/thesis --emit-pdf
```

可选参数：

- `--soffice`：指定 LibreOffice `soffice.com` 路径。
- `--dpi`：PNG 渲染分辨率，默认 120。
- `--pdfium-path`：指定本地 `pypdfium2` 依赖目录。默认优先尝试项目根目录 `.pydeps_pdfium`，否则使用当前 Python 环境中的 `pypdfium2`。

## audit_docx_format.py

对比参考论文和目标论文的 OOXML 格式信息，输出页面设置、样式、标题段落和正文样本的差异。

```powershell
python tools/thesis/audit_docx_format.py --reference reference.docx --target thesis.docx
```

## extract_docx_text.py

导出 Word 正文文本，便于检查章节内容、引用顺序和重复表达。

```powershell
python tools/thesis/extract_docx_text.py thesis.docx --output thesis.txt
```

## make_thesis_figures.py

生成论文中可继续修改的基础架构图和诊断流程图。

```powershell
python tools/thesis/make_thesis_figures.py --output-dir thesis_figures
```

该脚本需要当前 Python 环境安装 `matplotlib`。
