"""
Parameter alias registry.
Maps every common lab label variation → canonical parameter name used in config.py.

Used by extractor.py for NLP matching.
Each entry: (compiled_regex, canonical_name)
"""
import re

# ─────────────────────────────────────────────────────────────
# RAW ALIAS MAP  { regex_pattern : canonical_name }
# ─────────────────────────────────────────────────────────────
_RAW = {
    # ── HEMOGLOBIN ───────────────────────────────────────────
    r"\bhb\b|\bhgb\b|hemoglobin|haemoglobin": "Hemoglobin",

    # ── RBC ──────────────────────────────────────────────────
    r"\brbc\b|red\s*blood\s*cell|red\s*blood\s*count|erythrocyte\s*count": "RBC",

    # ── WBC ──────────────────────────────────────────────────
    r"\bwbc\b|white\s*blood\s*cell|white\s*blood\s*count|"
    r"total\s*leukocyte|leukocyte\s*count|tlc\b": "WBC",

    # ── PLATELETS ────────────────────────────────────────────
    r"\bplt\b|\bplatelet|thrombocyte|platelet\s*count": "Platelets",

    # ── HEMATOCRIT ───────────────────────────────────────────
    r"\bhct\b|hematocrit|haematocrit|packed\s*cell\s*volume|\bpcv\b": "Hematocrit",

    # ── MCV ──────────────────────────────────────────────────
    r"\bmcv\b|mean\s*corp\S*\s*volume|mean\s*cell\s*volume": "MCV",

    # ── MCH ──────────────────────────────────────────────────
    r"\bmch\b(?!c)|mean\s*corp\S*\s*hemo(?!\S*\s*conc)|mean\s*cell\s*hemo(?!\S*\s*conc)": "MCH",

    # ── MCHC ─────────────────────────────────────────────────
    r"\bmchc\b|mean\s*corp\S*\s*hemo\S*\s*conc|mean\s*cell\s*hemo\S*\s*conc": "MCHC",

    # ── RDW ──────────────────────────────────────────────────
    r"\brdw\b|red\s*cell\s*dist\S*\s*width|red\s*blood\s*cell\s*dist": "RDW",

    # ── DIFFERENTIAL ─────────────────────────────────────────
    r"\bneutrophil|\bneut\b|\bpoly\b|\bpmn\b|polymorphonuclear|segmented\s*neut": "Neutrophils",
    r"\blymphocyte|\blymp\b|\blymph\b": "Lymphocytes",
    r"\bmonocyte|\bmono\b": "Monocytes",
    r"\beosinophil|\beos\b": "Eosinophils",
    r"\bbasophil|\bbaso\b": "Basophils",

    # ── GLUCOSE ──────────────────────────────────────────────
    r"\bglucose|\bblood\s*sugar|\bfbs\b|\brbs\b|\bppbs\b|"
    r"fasting\s*(?:blood\s*)?glucose|random\s*(?:blood\s*)?glucose|"
    r"fasting\s*plasma\s*glucose|\bfpg\b": "Glucose",

    # ── HbA1c ────────────────────────────────────────────────
    r"\bhba1c\b|glycated\s*hemo|glycosylated\s*hemo|\ba1c\b|"
    r"glyco\S*\s*hb|hb\s*a1c": "HbA1c",

    # ── INSULIN ──────────────────────────────────────────────
    r"\binsulin\b|fasting\s*insulin|serum\s*insulin": "Insulin",

    # ── CREATININE ───────────────────────────────────────────
    r"\bcreatinine|\bcreat\b|\bscr\b|serum\s*creatinine": "Creatinine",

    # ── BUN ──────────────────────────────────────────────────
    r"\bbun\b|blood\s*urea\s*nitrogen|urea\s*nitrogen|\burea\b(?!\s*acid)": "Blood_Urea_Nitrogen",

    # ── URIC ACID ────────────────────────────────────────────
    r"\buric\s*acid|\bua\b(?!\s*\d)|serum\s*uric": "Uric_Acid",

    # ── eGFR ─────────────────────────────────────────────────
    r"\begfr\b|estimated\s*gfr|glomerular\s*filtration": "eGFR",

    # ── ELECTROLYTES ─────────────────────────────────────────
    r"\bsodium\b|\bna\+?\b": "Sodium",
    r"\bpotassium\b|\bk\+?\b(?!\s*\d{4})": "Potassium",
    r"\bcalcium\b|\bca\b(?!\s*\d{4})|serum\s*calcium": "Calcium",
    r"\bmagnesium\b|\bmg\b(?!\s*/\s*dl)|\bmg\+?\b": "Magnesium",
    r"\bphosphorus\b|\bphosphate\b|\bphos\b|\bp\b(?=\s*:\s*\d)": "Phosphorus",
    r"\bchloride\b|\bcl\b(?=\s*[:\|])": "Chloride",
    r"\bbicarbonate\b|\bhco3\b|\btotal\s*co2\b|\bco2\b(?=\s*[:\|])": "Bicarbonate",

    # ── LIVER PANEL ──────────────────────────────────────────
    r"\btotal\s*protein|\btp\b(?=\s*[:\|])": "Total_Protein",
    r"\balbumin|\balb\b": "Albumin",
    r"\bglobulin\b": "Globulin",
    r"\btotal\s*bilirubin|\bt[\.\s]?bil|tbil\b": "Total_Bilirubin",
    r"\bdirect\s*bilirubin|\bd[\.\s]?bil|dbil\b|conjugated\s*bil": "Direct_Bilirubin",
    r"\bindirect\s*bilirubin|\bi[\.\s]?bil|ibil\b|unconjugated\s*bil": "Indirect_Bilirubin",
    r"\balt\b|\bsgpt\b|alanine\s*(?:amino)?transferase|alanine\s*transaminase": "ALT",
    r"\bast\b|\bsgot\b|aspartate\s*(?:amino)?transferase|aspartate\s*transaminase": "AST",
    r"\balkaline\s*phosphatase|\balp\b|\balk\s*phos\b": "Alkaline_Phosphatase",
    r"\bggt\b|gamma\s*(?:glutamyl|gt)|γ[\s\-]?gt": "GGT",

    # ── LIPID PANEL ──────────────────────────────────────────
    r"\btotal\s*cholesterol|\bcholesterol\b|\bchol\b": "Cholesterol",
    r"\bhdl\b|high[\s\-]density\s*lip|hdl[\s\-]c": "HDL",
    r"\bldl\b|low[\s\-]density\s*lip|ldl[\s\-]c": "LDL",
    r"\btriglyceride|\btgl\b|\btg\b(?=\s*[:\|])|\btrigs?\b": "Triglycerides",
    r"\bvldl\b|very\s*low[\s\-]density": "VLDL",

    # ── THYROID ──────────────────────────────────────────────
    r"\btsh\b|thyroid[\s\-]stimulating\s*hormone|thyrotropin": "TSH",
    r"\bt3\b(?!\s*uptake)|triiodothyronine|total\s*t3": "T3",
    r"\bt4\b|thyroxine|total\s*t4": "T4",
    r"\bfree\s*t3\b|\bft3\b|free\s*triiodothyronine": "Free_T3",
    r"\bfree\s*t4\b|\bft4\b|free\s*thyroxine": "Free_T4",

    # ── IRON STUDIES ─────────────────────────────────────────
    r"\bserum\s*iron\b|\biron\b(?!\s*(?:binding|deficiency|study|panel))": "Iron",
    r"\bferritin\b": "Ferritin",
    r"\btibc\b|total\s*iron[\s\-]binding|iron\s*binding\s*capacity": "TIBC",
    r"\btransferrin\s*saturation|\bsat\s*%\b|transferrin\s*sat": "Transferrin_Saturation",

    # ── INFLAMMATION ─────────────────────────────────────────
    r"\bcrp\b|c[\s\-]reactive\s*protein|high[\s\-]?sensitivity\s*crp|hscrp": "CRP",
    r"\besr\b|erythrocyte\s*sedimentation|sed\s*rate|westergren": "ESR",

    # ── VITAMINS ─────────────────────────────────────────────
    r"\bvitamin\s*d\b|\b25[\s\-]oh\s*d\b|25[\s\-]hydroxyvitamin|calcidiol|vit\s*d": "Vitamin_D",
    r"\bvitamin\s*b[\s\-]?12\b|cyanocobalamin|cobalamin|vit\s*b12": "Vitamin_B12",
    r"\bfolate\b|\bfolic\s*acid\b|\bserum\s*folate\b": "Folate",
}

# ─────────────────────────────────────────────────────────────
# COMPILED LOOKUP  [ (compiled_re, canonical), ... ]
# ─────────────────────────────────────────────────────────────
COMPILED_ALIASES = [
    (re.compile(pat, re.IGNORECASE), canon)
    for pat, canon in _RAW.items()
]


def resolve(label: str) -> str | None:
    """
    Map a raw label string → canonical parameter name.
    Returns None if no alias matches.

    Examples
    --------
    >>> resolve("SGPT")
    'ALT'
    >>> resolve("Fasting Blood Sugar")
    'Glucose'
    >>> resolve("Hb")
    'Hemoglobin'
    """
    label = label.strip()
    for pattern, canonical in COMPILED_ALIASES:
        if pattern.search(label):
            return canonical
    return None


# ─────────────────────────────────────────────────────────────
# UNIT NORMALISATION  (optional helper)
# ─────────────────────────────────────────────────────────────
UNIT_ALIASES = {
    "g/dl":   "g/dL",
    "mg/dl":  "mg/dL",
    "meq/l":  "mEq/L",
    "iu/l":   "U/L",
    "u/l":    "U/L",
    "mmol/l": "mmol/L",
    "nmol/l": "nmol/L",
    "pmol/l": "pmol/L",
    "µg/dl":  "µg/dL",
    "ug/dl":  "µg/dL",
    "ng/ml":  "ng/mL",
    "pg/ml":  "pg/mL",
    "miU/l":  "mIU/L",
    "miu/l":  "mIU/L",
    "fl":     "fL",
    "pg":     "pg",
    "k/ul":   "K/µL",
    "k/µl":   "K/µL",
    "10^3/µl":"K/µL",
    "m/ul":   "M/µL",
    "m/µl":   "M/µL",
    "10^6/µl":"M/µL",
    "million/µl": "M/µL",
    "mm/hr":  "mm/hr",
    "mm/h":   "mm/hr",
}

def normalise_unit(raw: str) -> str:
    """Normalise a unit string to a consistent display form."""
    return UNIT_ALIASES.get(raw.strip().lower(), raw.strip())


# ─────────────────────────────────────────────────────────────
# QUICK SELF-TEST
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tests = [
        ("Hb",                    "Hemoglobin"),
        ("HGB",                   "Hemoglobin"),
        ("SGPT",                  "ALT"),
        ("SGOT",                  "AST"),
        ("FBS",                   "Glucose"),
        ("Fasting Blood Sugar",   "Glucose"),
        ("PCV",                   "Hematocrit"),
        ("TLC",                   "WBC"),
        ("T. Bilirubin",          "Total_Bilirubin"),
        ("Free T4",               "Free_T4"),
        ("25-OH Vitamin D",       "Vitamin_D"),
        ("HbA1c",                 "HbA1c"),
        ("eGFR",                  "eGFR"),
        ("Platelet Count",        "Platelets"),
        ("Serum Iron",            "Iron"),
        ("Transferrin Saturation","Transferrin_Saturation"),
        ("CRP",                   "CRP"),
        ("Unknown Param",         None),
    ]
    passed = 0
    for label, expected in tests:
        got = resolve(label)
        status = "✅" if got == expected else "❌"
        if got != expected:
            print(f"{status} resolve('{label}') → {got!r}  (expected {expected!r})")
        else:
            passed += 1
    print(f"\n{passed}/{len(tests)} alias tests passed.")