from __future__ import annotations

import argparse
import json
import zipfile
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET


W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": W}


def qn(name: str) -> str:
    return f"{{{W}}}{name}"


def read_xml(docx: Path, member: str) -> ET.Element | None:
    try:
        with zipfile.ZipFile(docx) as package:
            return ET.fromstring(package.read(member))
    except KeyError:
        return None


def attr(element: ET.Element | None, name: str) -> str | None:
    return None if element is None else element.attrib.get(qn(name))


def text_of(paragraph: ET.Element) -> str:
    parts: list[str] = []
    for text_node in paragraph.iter(qn("t")):
        if text_node.text:
            parts.append(text_node.text)
    return "".join(parts).strip()


def section_info(document: ET.Element | None) -> dict[str, str | None]:
    if document is None:
        return {}
    sect_pr = document.find(".//w:sectPr", NS)
    if sect_pr is None:
        return {}
    page_size = sect_pr.find("w:pgSz", NS)
    margins = sect_pr.find("w:pgMar", NS)
    info: dict[str, str | None] = {
        "page_width": attr(page_size, "w"),
        "page_height": attr(page_size, "h"),
    }
    for key in ["top", "bottom", "left", "right", "header", "footer", "gutter"]:
        info[f"margin_{key}"] = attr(margins, key)
    return info


def paragraph_format(paragraph: ET.Element) -> dict[str, str | None]:
    ppr = paragraph.find("w:pPr", NS)
    rpr = paragraph.find(".//w:rPr", NS)
    spacing = ppr.find("w:spacing", NS) if ppr is not None else None
    indent = ppr.find("w:ind", NS) if ppr is not None else None
    style = ppr.find("w:pStyle", NS) if ppr is not None else None
    justify = ppr.find("w:jc", NS) if ppr is not None else None
    fonts = rpr.find("w:rFonts", NS) if rpr is not None else None
    size = rpr.find("w:sz", NS) if rpr is not None else None
    bold = rpr.find("w:b", NS) if rpr is not None else None

    return {
        "style": attr(style, "val"),
        "justify": attr(justify, "val"),
        "line": attr(spacing, "line"),
        "line_rule": attr(spacing, "lineRule"),
        "before": attr(spacing, "before"),
        "after": attr(spacing, "after"),
        "first_line": attr(indent, "firstLine"),
        "first_line_chars": attr(indent, "firstLineChars"),
        "font_east_asia": attr(fonts, "eastAsia"),
        "font_ascii": attr(fonts, "ascii"),
        "size": attr(size, "val"),
        "bold": "1" if bold is not None and attr(bold, "val") != "0" else "0",
    }


def collect_paragraphs(document: ET.Element | None) -> list[dict[str, object]]:
    if document is None:
        return []
    rows: list[dict[str, object]] = []
    for paragraph in document.findall(".//w:body/w:p", NS):
        text = text_of(paragraph)
        if not text:
            continue
        fmt = paragraph_format(paragraph)
        rows.append({"text": text, "format": fmt})
    return rows


def summarize(docx: Path) -> dict[str, object]:
    document = read_xml(docx, "word/document.xml")
    paragraphs = collect_paragraphs(document)
    style_counts = Counter(str(row["format"].get("style")) for row in paragraphs)
    heading_samples = [
        row for row in paragraphs
        if str(row["format"].get("style") or "").lower().startswith("heading") or str(row["text"])[:1].isdigit()
    ][:20]
    body_samples = [
        row for row in paragraphs
        if len(str(row["text"])) >= 40 and not str(row["text"])[:1].isdigit()
    ][:20]
    return {
        "file": str(docx),
        "section": section_info(document),
        "paragraph_count": len(paragraphs),
        "style_counts": dict(style_counts.most_common(20)),
        "heading_samples": heading_samples,
        "body_samples": body_samples,
    }


def compare_values(label: str, reference: dict[str, object], target: dict[str, object]) -> list[str]:
    lines = [f"\n## {label}"]
    keys = sorted(set(reference) | set(target))
    for key in keys:
        ref_value = reference.get(key)
        target_value = target.get(key)
        marker = "OK" if ref_value == target_value else "DIFF"
        lines.append(f"- {key}: reference={ref_value!r}, target={target_value!r} [{marker}]")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare basic DOCX format information against a reference file.")
    parser.add_argument("--reference", required=True, help="Reference DOCX path.")
    parser.add_argument("--target", required=True, help="Target DOCX path.")
    parser.add_argument("--json", dest="json_output", help="Optional JSON report path.")
    args = parser.parse_args()

    reference = summarize(Path(args.reference).resolve())
    target = summarize(Path(args.target).resolve())

    lines: list[str] = []
    lines.append(f"Reference: {reference['file']}")
    lines.append(f"Target:    {target['file']}")
    lines.extend(compare_values("Page Setup", reference["section"], target["section"]))
    lines.append("\n## Paragraph Count")
    lines.append(f"- reference={reference['paragraph_count']}, target={target['paragraph_count']}")
    lines.append("\n## Top Styles")
    lines.append(f"- reference={reference['style_counts']}")
    lines.append(f"- target={target['style_counts']}")
    lines.append("\n## Target Heading Samples")
    for row in target["heading_samples"][:10]:
        lines.append(f"- {row['text'][:80]} | {row['format']}")
    lines.append("\n## Target Body Samples")
    for row in target["body_samples"][:5]:
        lines.append(f"- {row['text'][:80]} | {row['format']}")

    print("\n".join(lines))

    if args.json_output:
        output = Path(args.json_output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps({"reference": reference, "target": target}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nJSON report: {output}")


if __name__ == "__main__":
    main()
