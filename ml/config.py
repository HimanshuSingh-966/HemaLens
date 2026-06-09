"""
Central configuration for HemaLens Blood Report Analysis.
All training, inference, and API settings live here.
Override any value via environment variables or .env file.
"""
import os

# Load .env if present (dev only — Render injects env vars directly in production)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed — fall back to OS env vars

# ── PATHS ─────────────────────────────────────────────────
DATA_PATH   = os.getenv("DATA_PATH",   "data/diagnostic_pathology.csv")
MODEL_DIR   = os.getenv("MODEL_DIR",   "models")
RESULTS_DIR = os.getenv("RESULTS_DIR", "results")

# ── TRAINING ──────────────────────────────────────────────
TARGET_COL   = os.getenv("TARGET_COL",   "Diagnosis")
GENDER_COL   = os.getenv("GENDER_COL",   "Gender")
TEST_SIZE    = float(os.getenv("TEST_SIZE",    "0.2"))
RANDOM_STATE = int(os.getenv("RANDOM_STATE",   "42"))
CV_FOLDS     = int(os.getenv("CV_FOLDS",       "5"))

# ── API ───────────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# ── REFERENCE RANGES ──────────────────────────────────────
# Format: {"both": (low, high)} or {"male": (...), "female": (...)}
# "unit" key is for display only.
REFERENCE_RANGES = {
    # ── CBC ──────────────────────────────────────────────
    "Hemoglobin": {
        "male": (13.5, 17.5), "female": (12.0, 15.5), "unit": "g/dL"
    },
    "RBC": {
        "male": (4.5, 5.9), "female": (4.1, 5.1), "unit": "million/µL"
    },
    "WBC":        {"both": (4.5,  11.0),  "unit": "K/µL"},
    "Platelets":  {"both": (150,  400),   "unit": "K/µL"},
    "Hematocrit": {
        "male": (41, 53), "female": (36, 46), "unit": "%"
    },
    "MCV":        {"both": (80,   100),   "unit": "fL"},
    "MCH":        {"both": (27,   33),    "unit": "pg"},
    "MCHC":       {"both": (32,   36),    "unit": "g/dL"},
    "RDW":        {"both": (11.5, 14.5),  "unit": "%"},

    # ── DIFFERENTIAL ─────────────────────────────────────
    "Neutrophils":  {"both": (40, 70),  "unit": "%"},
    "Lymphocytes":  {"both": (20, 40),  "unit": "%"},
    "Monocytes":    {"both": (2,  10),  "unit": "%"},
    "Eosinophils":  {"both": (1,  6),   "unit": "%"},
    "Basophils":    {"both": (0,  1),   "unit": "%"},

    # ── METABOLIC ─────────────────────────────────────────
    "Glucose":    {"both": (70, 100),   "unit": "mg/dL"},
    "HbA1c":      {"both": (4.0, 5.6),  "unit": "%"},
    "Insulin":    {"both": (2.6, 24.9), "unit": "µIU/mL"},

    # ── KIDNEY ───────────────────────────────────────────
    "Creatinine": {
        "male": (0.7, 1.2), "female": (0.5, 1.0), "unit": "mg/dL"
    },
    "Blood_Urea_Nitrogen": {"both": (7,  25),   "unit": "mg/dL"},
    "Uric_Acid":           {
        "male": (3.5, 7.2), "female": (2.6, 6.0), "unit": "mg/dL"
    },
    "eGFR":                {"both": (60, 120),  "unit": "mL/min/1.73m²"},

    # ── ELECTROLYTES ─────────────────────────────────────
    "Sodium":    {"both": (136, 145), "unit": "mEq/L"},
    "Potassium": {"both": (3.5, 5.0), "unit": "mEq/L"},
    "Calcium":   {"both": (8.5, 10.5),"unit": "mg/dL"},
    "Magnesium": {"both": (1.7, 2.2), "unit": "mg/dL"},
    "Phosphorus":{"both": (2.5, 4.5), "unit": "mg/dL"},
    "Chloride":  {"both": (98, 106),  "unit": "mEq/L"},
    "Bicarbonate":{"both": (22, 29),  "unit": "mEq/L"},

    # ── LIVER ─────────────────────────────────────────────
    "Total_Protein":        {"both": (6.0,  8.3),  "unit": "g/dL"},
    "Albumin":              {"both": (3.5,  5.0),  "unit": "g/dL"},
    "Globulin":             {"both": (2.0,  3.5),  "unit": "g/dL"},
    "Total_Bilirubin":      {"both": (0.2,  1.2),  "unit": "mg/dL"},
    "Direct_Bilirubin":     {"both": (0.0,  0.3),  "unit": "mg/dL"},
    "Indirect_Bilirubin":   {"both": (0.2,  0.9),  "unit": "mg/dL"},
    "ALT":  {"male": (7, 56),  "female": (7, 45),  "unit": "U/L"},
    "AST":                  {"both": (10,   40),   "unit": "U/L"},
    "Alkaline_Phosphatase": {"both": (44,   147),  "unit": "U/L"},
    "GGT":  {"male": (8, 61),  "female": (5, 36),  "unit": "U/L"},

    # ── LIPIDS ───────────────────────────────────────────
    "Cholesterol":   {"both": (0,   200), "unit": "mg/dL"},
    "HDL":  {"male": (40, 60), "female": (50, 60), "unit": "mg/dL"},
    "LDL":           {"both": (0,   130), "unit": "mg/dL"},
    "Triglycerides": {"both": (0,   150), "unit": "mg/dL"},
    "VLDL":          {"both": (2,   30),  "unit": "mg/dL"},

    # ── THYROID ──────────────────────────────────────────
    "TSH": {"both": (0.4, 4.0),  "unit": "mIU/L"},
    "T3":  {"both": (80,  200),  "unit": "ng/dL"},
    "T4":  {"both": (5.0, 12.0), "unit": "µg/dL"},
    "Free_T3": {"both": (2.3, 4.2), "unit": "pg/mL"},
    "Free_T4": {"both": (0.8, 1.8), "unit": "ng/dL"},

    # ── IRON STUDIES ─────────────────────────────────────
    "Iron":    {"male": (65, 175), "female": (50, 170), "unit": "µg/dL"},
    "Ferritin":{"male": (20, 500), "female": (20, 200), "unit": "ng/mL"},
    "TIBC":    {"both": (250, 370), "unit": "µg/dL"},
    "Transferrin_Saturation": {"both": (20, 50), "unit": "%"},

    # ── INFLAMMATION ─────────────────────────────────────
    "CRP":  {"both": (0, 1.0),  "unit": "mg/L"},
    "ESR":  {"male": (0, 15),   "female": (0, 20), "unit": "mm/hr"},

    # ── VITAMINS ─────────────────────────────────────────
    "Vitamin_D":   {"both": (30, 100), "unit": "ng/mL"},
    "Vitamin_B12": {"both": (200, 900),"unit": "pg/mL"},
    "Folate":      {"both": (2.7, 17), "unit": "ng/mL"},
}

# ── DIAGNOSIS RULES (for rule-based fallback) ─────────────
DIAGNOSIS_RULES = {
    "Anemia": [
        ("Hemoglobin", "low"), ("RBC", "low"), ("MCV", "low"),
    ],
    "Diabetes Mellitus": [
        ("Glucose", "high"), ("HbA1c", "high"),
    ],
    "Pre-Diabetes": [
        ("Glucose", "borderline_high"), ("HbA1c", "borderline_high"),
    ],
    "Infection/Inflammation": [
        ("WBC", "high"), ("CRP", "high"), ("ESR", "high"),
    ],
    "Hepatic Dysfunction": [
        ("ALT", "high"), ("AST", "high"), ("Total_Bilirubin", "high"),
    ],
    "Renal Impairment": [
        ("Creatinine", "high"), ("Blood_Urea_Nitrogen", "high"),
    ],
    "Dyslipidemia": [
        ("Cholesterol", "high"), ("LDL", "high"), ("Triglycerides", "high"),
    ],
    "Hypothyroidism": [
        ("TSH", "high"), ("Free_T4", "low"),
    ],
    "Hyperthyroidism": [
        ("TSH", "low"), ("Free_T4", "high"),
    ],
    "Iron Deficiency": [
        ("Iron", "low"), ("Ferritin", "low"), ("TIBC", "high"),
    ],
    "Polycythemia": [
        ("Hemoglobin", "high"), ("RBC", "high"), ("Hematocrit", "high"),
    ],
    "Thrombocytopenia": [
        ("Platelets", "low"),
    ],
    "Leukopenia": [
        ("WBC", "low"),
    ],
}