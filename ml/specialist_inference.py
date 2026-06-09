"""
HemaLens — Multi-Specialist Inference Engine
=============================================
Loads all trained specialist models and runs them in parallel
on a set of extracted blood parameters.

Each specialist only activates if enough of its required
features are present in the input.

Usage:
    from ml.specialist_inference import predict_all_specialists

    params = {"Hemoglobin": 10.2, "WBC": 11.8, "Glucose": 126, "TSH": 0.1}
    results = predict_all_specialists(params, gender="male")
"""

import os
import json
import warnings
import joblib
import numpy as np
from typing import Optional

# Suppress XGBoost device-change warnings that fire during pickle deserialization
# These are harmless — XGBoost correctly falls back to CPU automatically
warnings.filterwarnings("ignore", message=".*grow_gpu_hist.*")
warnings.filterwarnings("ignore", message=".*No visible GPU.*")
warnings.filterwarnings("ignore", message=".*Device is changed from GPU.*")
warnings.filterwarnings("ignore", message=".*Changing updater.*")

DEFAULT_SPECIALIST_DIR = "models/specialists"
SPECIALIST_DIR = DEFAULT_SPECIALIST_DIR


def _use_hf_models() -> bool:
    return os.getenv("USE_HF_MODELS", "false").lower() in {"1", "true", "yes", "on"}


def _specialist_dir() -> str:
    """Return local specialist directory, downloading from Hugging Face if enabled."""
    if not _use_hf_models():
        return os.getenv("SPECIALIST_DIR", DEFAULT_SPECIALIST_DIR)

    repo_id = os.getenv("HF_MODEL_REPO")
    if not repo_id:
        raise RuntimeError("USE_HF_MODELS=true requires HF_MODEL_REPO.")

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "USE_HF_MODELS=true requires huggingface_hub. "
            "Install dependencies with `pip install -r requirements.txt`."
        ) from exc

    return snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        revision=os.getenv("HF_MODEL_REVISION") or None,
        token=os.getenv("HF_TOKEN") or None,
    )

# Cached loaded models  { specialist_name: (pipeline, metadata, label_encoder) }
_CACHE: dict = {}


# ─────────────────────────────────────────────────────────────
# GPU / CPU DETECTION
# ─────────────────────────────────────────────────────────────
def _detect_device() -> str:
    """Return 'cuda' if a GPU is available, else 'cpu'."""
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, timeout=3
        )
        return "cuda" if result.returncode == 0 else "cpu"
    except (FileNotFoundError, Exception):
        return "cpu"

_DEVICE = _detect_device()
print(f"  🖥️  Inference device: {_DEVICE.upper()}")


# ─────────────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────────────
def load_specialist(name: str) -> tuple | None:
    """Load a specialist model from disk. Returns (pipeline, metadata, le) or None."""
    if name in _CACHE:
        return _CACHE[name]

    specialist_path = os.path.join(_specialist_dir(), name)
    model_path    = os.path.join(specialist_path, "model.pkl")
    meta_path     = os.path.join(specialist_path, "metadata.json")
    le_path       = os.path.join(specialist_path, "label_encoder.pkl")

    if not all(os.path.exists(p) for p in [model_path, meta_path]):
        return None

    pipeline = joblib.load(model_path)
    with open(meta_path) as f:
        metadata = json.load(f)
    le = joblib.load(le_path) if os.path.exists(le_path) else None

    # Set XGBoost classifier device to match current hardware
    # This prevents "mismatched devices" warning when model was trained on GPU
    # but inference is running on CPU (or vice versa)
    for step_name, step in pipeline.steps:
        if hasattr(step, "set_params") and hasattr(step, "device"):
            step.set_params(device=_DEVICE)
        elif hasattr(step, "set_params") and hasattr(step, "predictor"):
            # older XGBoost API
            step.set_params(
                tree_method="hist",
                device=_DEVICE,
                predictor="auto",
            )
        # Clear stored feature names — only name validation is skipped,
        # imputer median fill logic still works correctly
        if hasattr(step, "feature_names_in_"):
            step.feature_names_in_ = None

    _CACHE[name] = (pipeline, metadata, le)
    return _CACHE[name]


def load_all_specialists() -> dict:
    """Load every specialist that has been trained. Returns dict of loaded ones."""
    loaded = {}
    specialist_dir = _specialist_dir()
    if not os.path.isdir(specialist_dir):
        return loaded
    for name in os.listdir(specialist_dir):
        result = load_specialist(name)
        if result:
            loaded[name] = result
    return loaded


def available_specialists() -> list[str]:
    """Return names of all trained specialists on disk."""
    specialist_dir = _specialist_dir()
    if not os.path.isdir(specialist_dir):
        return []
    return [
        d for d in os.listdir(specialist_dir)
        if os.path.isfile(os.path.join(specialist_dir, d, "model.pkl"))
    ]


# ─────────────────────────────────────────────────────────────
# REFERENCE RANGES (for abnormality flags)
# ─────────────────────────────────────────────────────────────
def _get_ref_ranges() -> dict:
    try:
        from ml.config import REFERENCE_RANGES
        return REFERENCE_RANGES
    except ImportError:
        try:
            from config import REFERENCE_RANGES  # when run as script inside ml/
            return REFERENCE_RANGES
        except ImportError:
            return {}


def _check_flags(params: dict, gender: str = "male") -> list:
    try:
        from ml.inference import check_reference_ranges
    except ImportError:
        from inference import check_reference_ranges  # script mode fallback
    return check_reference_ranges(params, gender=gender)


def _risk_level(flags: list) -> str:
    if not flags:
        return "NORMAL"
    sevs = {f["severity"] for f in flags}
    if "SEVERE"   in sevs: return "HIGH"
    if "MODERATE" in sevs: return "MODERATE"
    return "LOW"


# ─────────────────────────────────────────────────────────────
# SINGLE SPECIALIST PREDICTION
# ─────────────────────────────────────────────────────────────
def _predict_one(name: str, pipeline, metadata: dict, le, params: dict) -> dict | None:
    """
    Run one specialist on the given params.
    Returns None if not enough features are present to make a valid prediction.

    We extract imputer statistics and scaler parameters from the fitted pipeline
    and apply them manually — completely bypassing sklearn validation checks.
    """
    feature_names = metadata["feature_names"]
    min_required  = metadata.get("min_features_required", 2)
    n_features    = len(feature_names)

    # Count how many required features are present and non-null
    present = [f for f in feature_names if f in params and params[f] is not None]
    if len(present) < min_required:
        return None  # specialist stays silent

    # Extract fitted steps
    steps = {sname: step for sname, step in pipeline.steps}
    clf   = steps.get("clf") or pipeline.steps[-1][1]

    # Get imputer medians — length = exact number of features model was trained on
    imputer  = steps.get("imputer")
    scaler   = steps.get("scaler")
    medians  = np.array(imputer.statistics_, dtype=np.float64) if imputer else None
    mean_    = np.array(scaler.mean_,        dtype=np.float64) if scaler  else None
    scale_   = np.array(scaler.scale_,       dtype=np.float64) if scaler  else None

    # n_fitted = number of features the pipeline was actually trained on
    # Priority: scaler > imputer > metadata (most to least reliable)
    if mean_ is not None:
        n_fitted = len(mean_)
    elif medians is not None:
        n_fitted = len(medians)
    else:
        n_fitted = n_features

    # Build X with exactly n_fitted values in training order
    # feature_names may be shorter than n_fitted — pad with NaN in that case
    row = []
    for i in range(n_fitted):
        if i < len(feature_names):
            f = feature_names[i]
            row.append(float(params[f]) if f in params and params[f] is not None else np.nan)
        else:
            row.append(np.nan)  # padding for features beyond metadata list
    X = np.array(row, dtype=np.float64).reshape(1, n_fitted)

    # Fill NaNs with training medians
    if medians is not None:
        nan_mask = np.isnan(X[0])
        if nan_mask.any():
            fill = medians[:n_fitted]
            X[0, nan_mask] = fill[nan_mask]

    # Apply scaling — both mean_ and scale_ are always set together
    if mean_ is not None and scale_ is not None:
        X = (X - mean_[:n_fitted]) / scale_[:n_fitted]

    # Predict directly on classifier — bypasses sklearn pipeline validation
    pred_idx  = int(clf.predict(X)[0])
    classes   = metadata["target_classes"]
    diagnosis = classes[pred_idx]

    try:
        proba       = clf.predict_proba(X)[0]
        confidence  = float(np.max(proba))
        class_proba = {classes[i]: round(float(p), 4) for i, p in enumerate(proba)}
    except Exception:
        confidence  = None
        class_proba = {}

    return {
        "specialist":        name,
        "description":       metadata.get("description", name),
        "diagnosis":         diagnosis,
        "confidence":        round(confidence, 4) if confidence else None,
        "probabilities":     class_proba,
        "features_used":     present,
        "features_missing":  [f for f in feature_names if f not in present],
        "model_f1":          metadata.get("test_f1_weighted"),
        "active":            True,
    }


# ─────────────────────────────────────────────────────────────
# MAIN ENTRY POINT: RUN ALL SPECIALISTS
# ─────────────────────────────────────────────────────────────
def predict_all_specialists(params: dict, gender: str = "male") -> dict:
    """
    Run every trained specialist on the extracted parameters.

    Returns a unified result dict with:
      - specialist_results: one entry per active specialist
      - skipped_specialists: specialists that lacked enough features
      - abnormal_flags: reference-range violations
      - risk_level: overall risk
      - summary: combined plain-English summary
    """
    all_specialists = load_all_specialists()

    if not all_specialists:
        raise RuntimeError(
            f"No trained specialist models found in {SPECIALIST_DIR}/. "
            "Run ml/train_specialists.py first."
        )

    specialist_results = []
    skipped            = []

    for name, (pipeline, metadata, le) in sorted(all_specialists.items()):
        result = _predict_one(name, pipeline, metadata, le, params)
        if result:
            specialist_results.append(result)
        else:
            skipped.append({
                "specialist":  name,
                "description": metadata.get("description", name),
                "reason":      f"Only {sum(1 for f in metadata['feature_names'] if f in params)} / "
                               f"{metadata.get('min_features_required', 2)} required features present",
                "active":      False,
            })

    # Abnormality flags (reference-range based, independent of ML)
    flags      = _check_flags(params, gender)
    risk       = _risk_level(flags)
    summary    = _build_summary(specialist_results, flags, risk)
    conditions = _extract_conditions(specialist_results)

    return {
        "specialist_results":  specialist_results,
        "skipped_specialists": skipped,
        "n_active":            len(specialist_results),
        "n_skipped":           len(skipped),
        "detected_conditions": conditions,
        "abnormal_flags":      flags,
        "total_abnormal":      len(flags),
        "risk_level":          risk,
        "summary":             summary,
        "recommendations":     _recommendations(conditions, flags),
    }


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def _extract_conditions(results: list) -> list[str]:
    """Collect non-Normal diagnoses across all active specialists."""
    conditions = []
    for r in results:
        diag = r["diagnosis"]
        conf = r["confidence"] or 0
        # Use configurable threshold so deployments can tune sensitivity
        try:
            threshold = float(os.getenv("SPECIALIST_CONF_THRESHOLD", "0.5"))
        except Exception:
            threshold = 0.5
        if "normal" not in diag.lower() and conf >= threshold:
            conditions.append(diag)
    return list(dict.fromkeys(conditions))  # deduplicate preserving order


def _build_summary(results: list, flags: list, risk: str) -> str:
    if not results:
        return "No specialist models were activated. Upload a report with more lab values."

    active_domains = [r["specialist"].capitalize() for r in results]
    conditions     = _extract_conditions(results)

    if not conditions:
        return (
            f"All active specialists ({', '.join(active_domains)}) "
            f"returned normal results. "
            f"{len(flags)} parameter(s) flagged outside reference range."
        )

    return (
        f"{len(results)} specialist model(s) activated "
        f"({', '.join(active_domains)}). "
        f"Detected: {', '.join(conditions)}. "
        f"Overall risk: {risk}."
    )


def _recommendations(conditions: list, flags: list) -> list[str]:
    recs = []
    condition_text = " ".join(conditions).lower()

    rule_map = {
        "anemia":           "Hematology consult recommended. Consider iron/B12/folate panel.",
        "diabetes":         "Endocrinology referral. Daily glucose monitoring advised.",
        "pre-diabet":       "Lifestyle modification: diet and exercise. Repeat HbA1c in 3 months.",
        "liver disease":    "Hepatology review. Avoid hepatotoxic agents. Repeat LFTs in 4-6 weeks.",
        "liver":            "Hepatology review. Avoid hepatotoxic agents. Repeat LFTs in 4-6 weeks.",
        "kidney":           "Nephrology referral. Monitor fluid balance and electrolytes.",
        "thyroid":          "Thyroid function panel (TSH, Free T3, Free T4). Endocrinology consult.",
        "thalassemia":      "Haematology referral. Genetic counselling may be appropriate.",
        "hypothyroid":      "Thyroid hormone replacement therapy evaluation required.",
        "hyperthyroid":     "Antithyroid therapy evaluation. Radioactive iodine assessment.",
        "leukemia":         "Urgent haematology referral. Full bone marrow evaluation may be required.",
        "thrombocytopenia": "Platelet count repeat advised. Haematology consult for cause evaluation.",
        "normocytic":       "Further investigation needed to determine underlying cause of anaemia.",
        "microcytic":       "Iron studies and haemoglobin electrophoresis recommended.",
        "macrocytic":       "Vitamin B12 and folate levels should be checked.",
        "chronic kidney":   "Nephrology referral. Dietary protein restriction and fluid monitoring advised.",
    }
    for keyword, rec in rule_map.items():
        if keyword in condition_text:
            recs.append(rec)

    severe_params = [f["parameter"] for f in flags if f["severity"] == "SEVERE"]
    if severe_params:
        recs.append(
            f"⚠️ Severely abnormal parameters: {', '.join(severe_params)}. "
            "Urgent clinical review recommended."
        )

    if not recs:
        recs.append("All findings within acceptable range. Routine follow-up in 6 months.")

    return recs


# ─────────────────────────────────────────────────────────────
# CLI — quick test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, json as _json
    # Ensure project root is on path so 'ml' package is importable when run as script
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _root not in sys.path:
        sys.path.insert(0, _root)

    sample = {
        # CBC — anemia specialist
        "Hemoglobin": 10.2, "RBC": 3.8, "WBC": 11.8, "Platelets": 210,
        "Hematocrit": 31.5, "MCV": 78.4, "MCH": 26.8, "MCHC": 30.5,
        # Diabetes specialist
        "Glucose": 126.0, "HbA1c": 7.2, "BMI": 27.5, "Age": 45,
        # Thyroid specialist
        "TSH": 0.1, "T3": 210.0, "T4": 13.0,
        # Liver specialist
        "Total_Bilirubin": 0.9, "Direct_Bilirubin": 0.3,
        "ALT": 38.0, "AST": 35.0, "Alkaline_Phosphatase": 95.0,
        "Total_Protein": 7.2, "Albumin": 4.0, "Albumin_Globulin_Ratio": 1.5,
        # Kidney specialist
        "Creatinine": 1.1, "Blood_Urea_Nitrogen": 18.0,
        "Sodium": 138.0, "Potassium": 4.2,
    }

    print("🩸 Running all specialist models on sample parameters...")
    result = predict_all_specialists(sample, gender="male")

    print(f"\n{'='*55}")
    print(f"  SPECIALIST INFERENCE RESULTS")
    print(f"{'='*55}")
    print(f"  Active specialists : {result['n_active']}")
    print(f"  Skipped            : {result['n_skipped']}")
    print(f"  Risk level         : {result['risk_level']}")
    print(f"  Conditions detected: {result['detected_conditions']}")
    print(f"\n  Summary: {result['summary']}")
    print(f"\n  Recommendations:")
    for rec in result["recommendations"]:
        print(f"    → {rec}")

    print(f"\n  Per-Specialist Results:")
    for r in result["specialist_results"]:
        conf = f"{r['confidence']:.0%}" if r['confidence'] else "N/A"
        print(f"    [{r['specialist']:10}] {r['diagnosis']:<30} confidence={conf}")

    if result["skipped_specialists"]:
        print(f"\n  Skipped (insufficient features):")
        for s in result["skipped_specialists"]:
            print(f"    [{s['specialist']:10}] {s['reason']}")
