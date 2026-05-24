from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_SOFFICE = Path(r"C:\Program Files\LibreOffice\program\soffice.com")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_pdfium_path(explicit: str | None) -> Path | None:
    if explicit:
        return Path(explicit).resolve()
    local = repo_root() / ".pydeps_pdfium"
    return local if local.exists() else None


def convert_docx_to_pdf(docx: Path, out_dir: Path, soffice: Path, timeout: int) -> Path:
    if not soffice.exists():
        raise FileNotFoundError(f"LibreOffice executable not found: {soffice}")

    out_dir.mkdir(parents=True, exist_ok=True)
    profile = out_dir / "_lo_profile"
    profile.mkdir(parents=True, exist_ok=True)

    expected = out_dir / f"{docx.stem}.pdf"
    if expected.exists():
        expected.unlink()

    cmd = [
        str(soffice),
        "--headless",
        "--nologo",
        "--nofirststartwizard",
        "--nodefault",
        "--nolockcheck",
        f"-env:UserInstallation=file:///{profile.as_posix()}",
        "--convert-to",
        "pdf",
        "--outdir",
        str(out_dir),
        str(docx),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(
            f"LibreOffice failed ({proc.returncode})\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )

    pdfs = sorted(out_dir.glob("*.pdf"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not pdfs:
        raise RuntimeError(f"LibreOffice did not create a PDF.\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    return pdfs[0]


def render_pdf(pdf: Path, out_dir: Path, dpi: int, pdfium_path: Path | None) -> int:
    if pdfium_path:
        sys.path.insert(0, str(pdfium_path))
    import pypdfium2 as pdfium

    pdf_doc = pdfium.PdfDocument(str(pdf))
    scale = dpi / 72.0
    for index, page in enumerate(pdf_doc, start=1):
        bitmap = page.render(scale=scale)
        image = bitmap.to_pil()
        image.save(out_dir / f"page-{index:03d}.png")
    return len(pdf_doc)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a DOCX file to PDF and page PNGs.")
    parser.add_argument("input_docx", help="Path to the .docx file.")
    parser.add_argument("--output-dir", required=True, help="Directory for generated PDF/PNG files.")
    parser.add_argument("--dpi", type=int, default=120, help="PNG render DPI.")
    parser.add_argument("--emit-pdf", action="store_true", help="Keep a stable copy named after the input DOCX.")
    parser.add_argument("--soffice", default=str(DEFAULT_SOFFICE), help="Path to LibreOffice soffice.com.")
    parser.add_argument("--pdfium-path", default=None, help="Optional path containing pypdfium2.")
    parser.add_argument("--timeout", type=int, default=180, help="LibreOffice conversion timeout in seconds.")
    args = parser.parse_args()

    docx = Path(args.input_docx).resolve()
    if not docx.exists():
        raise FileNotFoundError(docx)

    out_dir = Path(args.output_dir).resolve()
    pdf = convert_docx_to_pdf(docx, out_dir, Path(args.soffice).resolve(), args.timeout)
    page_count = render_pdf(pdf, out_dir, args.dpi, resolve_pdfium_path(args.pdfium_path))

    if args.emit_pdf:
        keep = out_dir / f"{docx.stem}.pdf"
        if pdf != keep:
            shutil.copy2(pdf, keep)

    print(f"Rendered {page_count} pages to {out_dir}")
    print(f"PDF: {pdf}")


if __name__ == "__main__":
    main()
