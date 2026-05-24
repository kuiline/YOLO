from __future__ import annotations

import argparse
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


def paragraph_text(paragraph: ET.Element) -> str:
    parts: list[str] = []
    for node in paragraph.iter():
        if node.tag == f"{{{NS['w']}}}t" and node.text:
            parts.append(node.text)
        elif node.tag == f"{{{NS['w']}}}tab":
            parts.append("\t")
    return "".join(parts).strip()


def extract_text(docx: Path) -> list[str]:
    with zipfile.ZipFile(docx) as package:
        xml = package.read("word/document.xml")
    root = ET.fromstring(xml)
    body = root.find("w:body", NS)
    if body is None:
        return []

    paragraphs: list[str] = []
    for paragraph in body.findall("w:p", NS):
        text = paragraph_text(paragraph)
        if text:
            paragraphs.append(text)
    return paragraphs


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract plain text paragraphs from a DOCX file.")
    parser.add_argument("input_docx")
    parser.add_argument("--output", "-o", help="Optional output .txt path. Prints to stdout when omitted.")
    args = parser.parse_args()

    docx = Path(args.input_docx).resolve()
    paragraphs = extract_text(docx)
    text = "\n\n".join(paragraphs)

    if args.output:
        output = Path(args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
        print(f"Wrote {len(paragraphs)} paragraphs to {output}")
    else:
        print(text)


if __name__ == "__main__":
    main()
