"""
HemaLens — Reference range checking and legacy single-model fallback.
"""
import os
import json
import joblib
import numpy as np
from typing import Optional

MODEL_DIR = "models"

try:
    from ml.config import REFERENCE_RANGES as _REF_RANGES
except ImportError:
    from config import REFERENCE_RANGES as _REF_RANGES


def load_model_artifacts():
    model_path = os.path.join(MODEL_DIR, "best_model.pkl")
    meta_path  = os.path.join(MODEL_DIR, "metadata.json")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}.")
    pipeline = joblib.load(model_path)
    with open(meta_path) as f:
        metadata = json.load(f)
    return pipeline, metadata


def check_reference_ranges(params: dict, ref_ranges: Optional[dict] = None, gender: str = "unknown") -> list:
    if ref_ranges is None:
        ref_ranges = _REF_RANGES
    flags = []
    for param, value in params.items():
        if param not in ref_ranges or value is None:
            continue
        rr = ref_ranges[param]
        if "both" in rr:
            low, high = rr["both"]
        elif gender.lower() in ("male", "m") and "male" in rr:
            low, high = rr["male"]
        elif gender.lower() in ("female", "f") and "female" in rr:
            low, high = rr["female"]
        else:
            continue
        unit = rr.get("unit", "")
        if value < low:
            flags.append({"parameter": param, "value": value, "status": "LOW",
                          "normal_range": f"{low}–{high} {unit}",
                          "severity": _severity(value, low, high)})
        elif value > high:
            flags.append({"parameter": param, "value": value, "status": "HIGH",
                          "normal_range": f"{low}–{high} {unit}",
                          "severity": _severity(value, low, high)})
    return flags


def _severity(value: float, low: float, high: float) -> str:
    dev = max((low - value) / (low + 1e-9), (value - high) / (high + 1e-9))
    if dev < 0.1:  return "MILD"
    if dev < 0.3:  return "MODERATE"
    return "SEVERE"


def classify_severity(value: float, low: float, high: float) -> str:
    """Public wrapper used by tests and callers that need range severity only."""
    return _severity(value, low, high)


def _risk_level(flags: list) -> str:
    if not flags:
        return "NORMAL"
    severities = {f["severity"] for f in flags}
    if "SEVERE" in severities:
        return "HIGH"
    if "MODERATE" in severities:
        return "MODERATE"
    return "LOW"


def _recommendations(diagnosis: str, flags: list) -> list[str]:
    diagnosis_text = (diagnosis or "").lower()
    recs = []

    if "anemia" in diagnosis_text:
        recs.append("Consult a hematologist. Consider iron, B12, and folate testing.")
    if "diabetes" in diagnosis_text:
        recs.append("Consult an endocrinologist. Monitor glucose and HbA1c regularly.")
    if "kidney" in diagnosis_text or "renal" in diagnosis_text:
        recs.append("Consult a nephrologist and repeat kidney function tests.")
    if "liver" in diagnosis_text or "hepatic" in diagnosis_text:
        recs.append("Consult a hepatologist and repeat liver function tests.")
    if "thyroid" in diagnosis_text:
        recs.append("Consult an endocrinologist and repeat thyroid function tests.")

    severe_params = [f["parameter"] for f in flags if f.get("severity") == "SEVERE"]
    if severe_params:
        recs.append(
            f"Urgent clinical review recommended for severely abnormal parameters: "
            f"{', '.join(severe_params)}."
        )

    if not recs:
        recs.append("Routine follow-up recommended. Discuss results with a licensed clinician.")

    return recs


def predict_single(params: dict, gender: str = "unknown") -> dict:
    """Legacy single-model fallback — used by routes.py if no specialists are trained."""
    import pandas as pd
    pipeline, metadata = load_model_artifacts()
    df = pd.DataFrame([params])
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].replace({"male": 1, "female": 0, "Male": 1, "Female": 0})
    for col in metadata["feature_names"]:
        if col not in df.columns:
            df[col] = np.nan
    df = df[metadata["feature_names"]]
    pred      = pipeline.predict(df)[0]
    diagnosis = metadata["target_classes"][pred]
    try:
        proba       = pipeline.predict_proba(df)[0]
        confidence  = float(np.max(proba))
        class_proba = {metadata["target_classes"][i]: round(float(p), 4) for i, p in enumerate(proba)}
    except Exception:
        confidence, class_proba = None, {}
    flags = check_reference_ranges(params, gender=gender)
    risk = _risk_level(flags)
    return {
        "diagnosis":       diagnosis,
        "confidence":      round(confidence, 4) if confidence else None,
        "probabilities":   class_proba,
        "abnormal_flags":  flags,
        "total_abnormal":  len(flags),
        "risk_level":      risk,
        "recommendations": _recommendations(diagnosis, flags),
    }
