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

try:
    from ml.config import REFERENCE_RANGES as _REF_RANGES
except Exception:
    try:
        import importlib
        cfg = importlib.import_module("config")
        _REF_RANGES = getattr(cfg, "REFERENCE_RANGES", {}) or {}
    except Exception:
        _REF_RANGES = {}

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
    "Cholesterol": r"(?:total\s+cholesterol|cholesterol)(?!\s*[\w\s]*ldl)[^\d]*?(\d+\.?\d*)",
    "LDL":         r"(?:ldl|low[\s\-]density)[^\d]*?(\d+\.?\d*)",
    "Triglycerides": r"(?:triglyceride|trigs?)[^\d]*?(\d+\.?\d*)",
    "Alkaline_Phosphatase": r"(?:alkaline\s*phosphatase|alk\s*phos)[^\d]*?(\d+\.?\d*)",
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

    # Sanitise extracted numeric values against reference ranges to
    # correct common OCR/formatting errors (missing decimals, extra zeroes).
    _sanitise_against_reference_ranges(out)

    return out


def _sanitise_against_reference_ranges(out: dict) -> None:
    """Adjust extracted numeric values when they are implausible for the
    parameter's reference range. Attempts simple scale corrections like
    dividing or multiplying by 10/100 where appropriate.

    This is conservative — only applies corrections to CBC parameters where
    OCR errors (missing decimals, extra zeroes) are most common. Metabolic
    and lipid panel values are left unchanged to avoid false rescaling.
    """
    # CBC parameters where decimal/scale errors are common
    SANITISABLE_PARAMS = {
        "Hemoglobin", "RBC", "Hematocrit", "MCV", "MCH", "MCHC", "RDW",
        "WBC", "Platelets",
    }
    
    for canon, value in list(out.items()):
        if canon not in _REF_RANGES or canon not in SANITISABLE_PARAMS:
            continue
        # determine a numeric (low, high) pair to check against
        rr = _REF_RANGES[canon]
        if "both" in rr:
            low, high = rr["both"]
        elif "male" in rr or "female" in rr:
            # gender unknown here; build a permissive combined range
            male_pair = rr.get("male")
            female_pair = rr.get("female")

            # validate pairs are sequences of two numbers and coerce safely
            def _valid_pair(p):
                return isinstance(p, (list, tuple)) and len(p) == 2

            def _as_pair(p):
                if _valid_pair(p):
                    return (p[0], p[1])
                return None

            mp = _as_pair(male_pair)
            fp = _as_pair(female_pair)

            if mp is None and fp is None:
                continue

            if mp is None:
                mp = fp
            if fp is None:
                fp = mp

            # At this point at least one of mp/fp is non-None; assert
            # so static analysis understands indexing is safe.
            assert mp is not None and fp is not None
            male_low, male_high = mp[0], mp[1]
            female_low, female_high = fp[0], fp[1]

            low = min(male_low, female_low)
            high = max(male_high, female_high)
        else:
            continue

        # skip non-numeric or already-plausible values
        try:
            v = float(value)
        except Exception:
            continue
        if low <= v <= high:
            continue

        # Conservative heuristics: only apply scaling corrections when the
        # value is clearly an order-of-magnitude away from the reference range.
        # Avoid changing slightly out-of-range values (e.g., 215 vs 200).
        try:
            scale_up_needed = (v < low) and (low > 0 and (low / v) >= 3.0)
        except Exception:
            scale_up_needed = False
        scale_down_needed = (v > high) and (high > 0 and (v / high) >= 3.0)

        # Try divisors first (only when value is much larger than expected)
        if scale_down_needed:
            for d in (10.0, 100.0, 1000.0):
                v2 = v / d
                if low <= v2 <= high:
                    out[canon] = v2
                    break

        # Try multipliers (only when value is much smaller than expected)
        if scale_up_needed:
            for m in (10.0, 100.0):
                v2 = v * m
                if low <= v2 <= high:
                    out[canon] = v2
                    break


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