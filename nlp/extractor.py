"""
HemaLens — NLP extraction of blood parameters from text, PDF, and images.
"""
import re
from pathlib import Path
from typing import Optional

try:
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

from nlp.aliases import resolve

# ── Regex patterns ──────────────────────────────────────────────
_COLON  = re.compile(
    r"([A-Za-z][A-Za-z0-9\s\(\)/\-\.]+?)\s*[:\-\|]\s*"
    r"([<>≤≥]?\s*\d{1,3}(?:[,]\d{3})*(?:\.\d+)?)",
    re.IGNORECASE,
)
_INLINE = re.compile(
    r"([A-Za-z][A-Za-z0-9\s\(\)/\-\.]{2,40}?)\s+"
    r"([<>≤≥]?\s*\d{1,3}(?:[,]\d{3})*(?:\.\d+)?)",
    re.IGNORECASE,
)
_REF_STRIP = re.compile(r"\(?\d+\.?\d*\s*[-–]\s*\d+\.?\d*\)?")
_TARGETED  = {
    "Hemoglobin":  r"(?:hb|hgb|hemoglobin)\s*[:\-]?\s*(\d+\.?\d*)",
    "WBC":         r"(?:wbc|white\s*blood\s*cell)[^\d]*?(\d+\.?\d*)",
    "RBC":         r"(?:rbc|red\s*blood)[^\d]*?(\d+\.?\d*)",
    "Platelets":   r"(?:plt|platelet)[^\d]*?(\d{2,6}\.?\d*)",
    "Glucose":     r"(?:glucose|blood\s*sugar|fbs)[^\d]*?(\d+\.?\d*)",
    "Creatinine":  r"(?:creatinine|creat)[^\d]*?(\d+\.?\d*)",
    "HbA1c":       r"(?:hba1c|a1c|glycated)[^\d]*?(\d+\.?\d*)",
    "TSH":         r"tsh[^\d]*?(\d+\.?\d*)",
    "Cholesterol": r"(?:total\s+cholesterol|cholesterol)[^\d]*?(\d+\.?\d*)",
}


def _parse(s: str) -> Optional[float]:
    try:
        return float(s.replace(",", "").strip().lstrip("<>≤≥ "))
    except ValueError:
        return None


def _apply_pattern(pattern: re.Pattern, text: str, out: dict):
    for m in pattern.finditer(text):
        label = _REF_STRIP.sub("", m.group(1).strip())
        canon = resolve(label)
        if canon and canon not in out:
            v = _parse(m.group(2))
            if v is not None:
                out[canon] = v


def extract_from_text(text: str) -> dict:
    out: dict = {}
    _apply_pattern(_COLON,  text, out)
    _apply_pattern(_INLINE, text, out)

    if SPACY_AVAILABLE:
        try:
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(text)
            for ent in doc.ents:
                if ent.label_ in ("QUANTITY", "CARDINAL", "PERCENT"):
                    ctx   = text[max(0, ent.start_char - 60): ent.start_char]
                    canon = resolve(ctx[-40:].strip())
                    if canon and canon not in out:
                        v = _parse(ent.text)
                        if v is not None:
                            out[canon] = v
        except Exception:
            pass

    for canon, pat in _TARGETED.items():
        if canon not in out:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                v = _parse(m.group(1))
                if v is not None:
                    out[canon] = v

    return out


def extract_from_pdf(path: str) -> dict:
    if not PDF_AVAILABLE:
        raise RuntimeError("pdfplumber not installed. Run: pip install pdfplumber")
    lines = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables():
                for row in table:
                    if row:
                        lines.append("  |  ".join(str(c) for c in row if c))
            t = page.extract_text()
            if t:
                lines.append(t)
    return extract_from_text("\n".join(lines))


def extract_from_image(path: str) -> dict:
    if not OCR_AVAILABLE:
        raise RuntimeError("pytesseract/Pillow not installed.")
    text = pytesseract.image_to_string(
        Image.open(path).convert("L"), config="--oem 3 --psm 6"
    )
    return extract_from_text(text)


def extract_from_file(file_path: str) -> dict:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    ext = path.suffix.lower()
    if ext == ".txt":
        return extract_from_text(path.read_text(encoding="utf-8", errors="ignore"))
    if ext == ".pdf":
        return extract_from_pdf(file_path)
    if ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"):
        return extract_from_image(file_path)
    raise ValueError(f"Unsupported file type: {ext}")


if __name__ == "__main__":
    import json, argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--file")
    parser.add_argument("--text")
    args = parser.parse_args()
    result = extract_from_file(args.file) if args.file else extract_from_text(args.text or "")
    print(json.dumps(result, indent=2))