"""
HemaLens — Specialist Model Trainer
====================================
Trains one independent ML model per disease domain.
Each model only uses the features relevant to its domain.

Usage:
    python ml/train_specialists.py

    # Train only specific specialists:
    python ml/train_specialists.py --only anemia liver

Datasets (place in data/raw/):
    data/raw/anemia.csv      → kaggle: ehababoelnaga/anemia-types-classification
    data/raw/liver.csv       → kaggle: uciml/indian-liver-patient-records
    data/raw/diabetes.csv    → kaggle: uciml/pima-indians-diabetes-database
    data/raw/kidney.csv      → kaggle: mansoordaku/ckdisease
    data/raw/thyroid.csv     → kaggle: emmanuelfwerr/thyroid-disease-data
    data/raw/diagnostic_pathology.csv → kaggle: pareshbadnore (fallback)

Output (one folder per specialist):
    models/specialists/anemia/   → model.pkl, metadata.json
    models/specialists/liver/    → model.pkl, metadata.json
    models/specialists/diabetes/ → model.pkl, metadata.json
    models/specialists/kidney/   → model.pkl, metadata.json
    models/specialists/thyroid/  → model.pkl, metadata.json
"""

import os
import json
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, f1_score, accuracy_score
import xgboost as xgb

RAW_DIR       = "data/raw"
SPECIALIST_DIR = "models/specialists"
RESULTS_DIR   = "results/specialists"
RANDOM_STATE  = 42
TEST_SIZE     = 0.2
CV_FOLDS      = 5


# ─────────────────────────────────────────────────────────────
# SPECIALIST DEFINITIONS
# Each entry defines:
#   file        → CSV filename in data/raw/
#   features    → columns to use as features (canonical names)
#   target      → target column in that CSV
#   label_map   → raw target value → human-readable diagnosis
#   rename      → dataset-specific column renames to canonical names
# ─────────────────────────────────────────────────────────────
SPECIALISTS = {

    "anemia": {
        "file":     "anemia.csv",
        "features": ["Hemoglobin", "RBC", "WBC", "Platelets", "Hematocrit",
                     "MCV", "MCH", "MCHC", "RDW"],
        "target":   "Diagnosis",
        "rename": {
            "HGB": "Hemoglobin", "Hgb": "Hemoglobin", "HB": "Hemoglobin",
            "HCT": "Hematocrit", "Hct": "Hematocrit", "PCV": "Hematocrit",
            "PLT": "Platelets",  "Sex": "Gender",
        },
        "label_map": {
            "0": "Normal",
            "1": "Iron Deficiency Anemia",
            "2": "Megaloblastic Anemia",
            "3": "Hemolytic Anemia",
            "4": "Aplastic Anemia",
            "5": "Thalassemia",
            "6": "Sickle Cell Anemia",
            "7": "Microcytic Anemia",
            "8": "Macrocytic Anemia",
        },
        "description": "CBC-based anemia classification",
        "min_features_required": 3,   # need at least this many features present to activate
    },

    "liver": {
        "file":     "liver.csv",
        "features": ["Total_Bilirubin", "Direct_Bilirubin", "ALT", "AST",
                     "Alkaline_Phosphatase", "Total_Protein", "Albumin",
                     "Albumin_Globulin_Ratio", "Age"],
        "target":   "Diagnosis",
        "encoding": "latin-1",
        "rename": {
            # abhi8923shriv dataset — spaces + non-breaking spaces in col names
            "Age of the patient":                    "Age",
            "Gender of the patient":                 "Gender",
            "Total Bilirubin":                       "Total_Bilirubin",
            "Direct Bilirubin":                      "Direct_Bilirubin",
            " Alkphos Alkaline Phosphotase":      "Alkaline_Phosphatase",
            " Sgpt Alamine Aminotransferase":     "ALT",
            "Sgot Aspartate Aminotransferase":       "AST",
            "Total Protiens":                        "Total_Protein",
            " ALB Albumin":                       "Albumin",
            "A/G Ratio Albumin and Globulin Ratio":  "Albumin_Globulin_Ratio",
            "Result":                                "_raw_target",
            # old UCI dataset fallback
            "Alkaline_Phosphotase":                  "Alkaline_Phosphatase",
            "Alamine_Aminotransferase":              "ALT",
            "Aspartate_Aminotransferase":            "AST",
            "Total_Protiens":                        "Total_Protein",
            "Dataset":                               "_raw_target",
            "Selector":                              "_raw_target",
        },
        "label_map": {
            "1": "Liver Disease", "2": "Normal",
            1:   "Liver Disease", 2:   "Normal",
        },
        "raw_target_col": "_raw_target",
        "description": "Liver function panel classification (30k rows)",
        "min_features_required": 3,
    },

    "diabetes": {
        "file":     "diabetes.csv",
        "features": ["Glucose", "HbA1c", "BMI", "Age"],
        "target":   "Diagnosis",
        "rename": {
            "blood_glucose_level": "Glucose",
            "HbA1c_level":         "HbA1c",
            "bmi":                 "BMI",
            "age":                 "Age",
            "gender":              "Gender",
            "diabetes":            "_raw_target",
            "Outcome":             "_raw_target",
        },
        "label_map": {
            "0": "Normal", "1": "Diabetes Mellitus",
            0:   "Normal", 1:   "Diabetes Mellitus",
        },
        "raw_target_col": "_raw_target",
        "description": "Glucose/HbA1c-based diabetes detection (100k rows)",
        "min_features_required": 1,
    },

    "kidney": {
        "file":     "kidney.csv",
        "features": ["Creatinine", "Blood_Urea_Nitrogen", "Sodium", "Potassium",
                     "Hemoglobin", "WBC", "RBC", "Hematocrit"],
        "target":   "Diagnosis",
        "rename": {
            "sc":   "Creatinine",
            "bu":   "Blood_Urea_Nitrogen",
            "sod":  "Sodium",
            "pot":  "Potassium",
            "hemo": "Hemoglobin",
            "wc":   "WBC",
            "wbc":  "WBC",
            "rc":   "RBC",
            "pcv":  "Hematocrit",
            "age":  "Age",
            "classification": "_raw_target",
            "class":          "_raw_target",
        },
        "label_map": {
            "ckd":    "Chronic Kidney Disease",
            "notckd": "Normal",
            "CKD":    "Chronic Kidney Disease",
            "1":      "Chronic Kidney Disease",
            "0":      "Normal",
        },
        "raw_target_col": "_raw_target",
        "na_values": ["?", "\t?", " ?"],
        "description": "Renal function panel classification",
        "min_features_required": 2,
    },

    "thyroid": {
        "file":     "thyroid.csv",
        "features": ["TSH", "T3", "T4"],
        "target":   "Diagnosis",
        "rename": {
            # Only rename TT4 → T4, leave FTI/T4U as-is so they don't create duplicates
            "TT4":          "T4",
            "target":       "_raw_target",
            "ThyroidClass": "_raw_target",
            "binaryClass":  "_raw_target",
        },
        "label_map": {
            # Standard string labels
            "negative":                "Normal",
            "compensated_hypothyroid": "Compensated Hypothyroidism",
            "primary_hypothyroid":     "Hypothyroidism",
            "secondary_hypothyroid":   "Hypothyroidism",
            "hyperthyroid":            "Hyperthyroidism",
            # Single-letter codes from emmanuelfwerr dataset
            # "-" = negative/normal, letters = various thyroid conditions
            "-":   "Normal",
            "A":   "Hyperthyroidism",
            "AK":  "Hyperthyroidism",
            "B":   "Hyperthyroidism",
            "C":   "Hypothyroidism",
            "C|I": "Hypothyroidism",
            "D":   "Hypothyroidism",
            "D|R": "Hypothyroidism",
            "E":   "Hypothyroidism",
            "F":   "Hyperthyroidism",
            "FK":  "Hyperthyroidism",
            "G":   "Hyperthyroidism",
            "GI":  "Hyperthyroidism",
            "GK":  "Hyperthyroidism",
            "GKJ": "Hyperthyroidism",
            "H|K": "Hyperthyroidism",
            "I":   "Hypothyroidism",
            "J":   "Hypothyroidism",
            "K":   "Hyperthyroidism",
            "KJ":  "Hyperthyroidism",
            "L":   "Hypothyroidism",
            "LJ":  "Hypothyroidism",
            "M":   "Hypothyroidism",
            "MI":  "Hypothyroidism",
            "MK":  "Hypothyroidism",
            "N":   "Normal",
            "Normal": "Normal",
            "Hypothyroidism": "Hypothyroidism",
            "O":   "Hyperthyroidism",
            "OI":  "Hyperthyroidism",
            "P":   "Hypothyroidism",
            "Q":   "Hypothyroidism",
            "R":   "Hyperthyroidism",
            "S":   "Hypothyroidism",
            "T":   "Hyperthyroidism",
        },
        "raw_target_col": "_raw_target",
        "description": "Thyroid function panel classification",
        "min_features_required": 1,
    },

    # general specialist removed — conditions covered by dedicated specialists
    # anemia→anemia, diabetes→diabetes, cholesterol→liver, hypertension→not a blood test
}


# ─────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────
def load_specialist_data(name: str, spec: dict) -> pd.DataFrame | None:
    path = os.path.join(RAW_DIR, spec["file"])
    if not os.path.exists(path):
        print(f"  ⚠️  {spec['file']} not found — skipping {name}")
        return None

    na_vals = spec.get("na_values", ["NA", "na", ""])
    encoding = spec.get("encoding", "utf-8")
    try:
        df = pd.read_csv(path, na_values=na_vals, encoding=encoding)
    except UnicodeDecodeError:
        print(f"  ⚠️  UTF-8 failed — retrying with latin-1 encoding")
        df = pd.read_csv(path, na_values=na_vals, encoding="latin-1")
    print(f"  Raw: {df.shape[0]} rows × {df.shape[1]} cols")

    # Rename to canonical
    df = df.rename(columns=spec.get("rename", {}))

    # Replace biologically impossible 0s
    for col in spec.get("zero_invalid", []):
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)

    # Map raw target → Diagnosis label
    raw_target_col = spec.get("raw_target_col")
    if raw_target_col and raw_target_col in df.columns:
        label_map = spec.get("label_map", {})
        df["Diagnosis"] = df[raw_target_col].map(label_map)
        df["Diagnosis"] = df["Diagnosis"].fillna(df[raw_target_col].astype(str))
        df = df.drop(columns=[raw_target_col])
    elif spec["target"] in df.columns:
        # Target already named correctly; apply label_map if needed
        label_map = spec.get("label_map", {})
        if label_map:
            df[spec["target"]] = df[spec["target"]].map(label_map).fillna(df[spec["target"]].astype(str))
        if spec["target"] != "Diagnosis":
            df = df.rename(columns={spec["target"]: "Diagnosis"})
    else:
        print(f"  ❌ Target column not found in {spec['file']}")
        return None

    # Drop rows with missing diagnosis
    df = df.dropna(subset=["Diagnosis"])
    df["Diagnosis"] = df["Diagnosis"].astype(str).str.strip()
    df = df[df["Diagnosis"] != ""]

    # explicit Series cast before value_counts to satisfy type checker
    diagnosis_series = pd.Series(df["Diagnosis"])
    print(f"  Classes: {diagnosis_series.value_counts().to_dict()}")
    return pd.DataFrame(df)  # explicit DataFrame cast — avoids Series | DataFrame ambiguity


# ─────────────────────────────────────────────────────────────
# FEATURE PREP
# ─────────────────────────────────────────────────────────────
def prepare_features(df: pd.DataFrame, feature_cols: list) -> tuple:
    available = [c for c in feature_cols if c in df.columns]
    if not available:
        raise ValueError("No feature columns found in dataset")

    # Strictly select ONLY the feature columns — no extras
    # This prevents duplicate columns (e.g. TT4+FTI both → T4) or
    # extra renamed columns (Age, Gender) from leaking into the pipeline
    X = df[available].copy()
    # Drop duplicate columns — keep first occurrence only
    X = X.loc[:, ~X.columns.duplicated(keep="first")]
    # Re-filter available to match deduplicated columns
    available = [c for c in available if c in X.columns]
    X = X[available].apply(pd.to_numeric, errors="coerce")
    y_raw = df["Diagnosis"]

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    print(f"  Features used ({len(available)}): {available}")
    # np.asarray normalises the union type (ndarray|tuple|None) to a plain ndarray
    # then list() converts safely — works regardless of what the type checker infers
    classes: list = list(np.asarray(le.classes_)) if le.classes_ is not None else []
    print(f"  Label classes: {classes}")
    return X, y, le, available


# ─────────────────────────────────────────────────────────────
# MODEL SELECTION
# ─────────────────────────────────────────────────────────────
def get_pipeline(n_classes: int) -> Pipeline:
    """XGBoost pipeline — uses GPU if available, falls back to CPU."""
    import subprocess, sys
    try:
        # Check if CUDA is available
        result = subprocess.run(["nvidia-smi"], capture_output=True)
        use_gpu = result.returncode == 0
    except FileNotFoundError:
        use_gpu = False

    clf = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        tree_method="hist",           # unified tree method (XGBoost 2.x)
        device="cuda" if use_gpu else "cpu",
        n_jobs=-1 if not use_gpu else 1,  # n_jobs ignored on GPU
    )
    if use_gpu:
        print("  ⚡ GPU detected — training on CUDA")
    else:
        print("  💻 No GPU detected — training on CPU")
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("clf",     clf),
    ])


# ─────────────────────────────────────────────────────────────
# TRAIN ONE SPECIALIST
# ─────────────────────────────────────────────────────────────
def train_specialist(name: str, spec: dict) -> bool:
    print(f"\n{'='*55}")
    print(f"  🔬 Training specialist: {name.upper()}")
    print(f"  {spec['description']}")
    print(f"{'='*55}")

    df = load_specialist_data(name, spec)
    if df is None:
        return False

    try:
        X, y, le, features_used = prepare_features(df, spec["features"])
    except ValueError as e:
        print(f"  ❌ Feature prep failed: {e}")
        return False

    # Need at least 2 classes and enough samples
    n_classes = len(le.classes_)
    if n_classes < 2:
        print(f"  ❌ Only {n_classes} class found — skipping")
        return False
    if len(X) < 50:
        print(f"  ❌ Too few samples ({len(X)}) — skipping")
        return False

    # Stratified split — fall back to non-stratified if any class has < 2 members
    from collections import Counter
    min_class_count = min(Counter(y).values())
    try:
        if min_class_count < 2:
            raise ValueError("too few")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
    except ValueError:
        print(f"  ⚠️  Some classes have < 2 members — using non-stratified split")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )

    pipeline = get_pipeline(n_classes)

    # Cross-validation
    cv = StratifiedKFold(n_splits=min(CV_FOLDS, n_classes), shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(pipeline, X_train, y_train,
                                cv=cv, scoring="f1_weighted", n_jobs=-1)
    print(f"  CV F1 (weighted): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Final fit
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="weighted")
    print(f"  Test Accuracy: {acc:.4f} | F1: {f1:.4f}")
    # Guard against test split missing some classes
    classes_list = list(np.asarray(le.classes_)) if le.classes_ is not None else []
    labels_present = sorted(set(int(i) for i in set(y_test) | set(y_pred)))
    target_names_safe = [classes_list[i] for i in labels_present]
    print(f"\n{classification_report(y_test, y_pred, labels=labels_present, target_names=target_names_safe)}")

    # Save
    out_dir = os.path.join(SPECIALIST_DIR, name)
    os.makedirs(out_dir, exist_ok=True)

    joblib.dump(pipeline, os.path.join(out_dir, "model.pkl"))

    metadata = {
        "specialist":        name,
        "description":       spec["description"],
        "feature_names":     features_used,
        "target_classes":    list(le.classes_),
        "n_classes":         n_classes,
        "test_accuracy":     round(acc, 4),
        "test_f1_weighted":  round(f1, 4),
        "cv_f1_mean":        round(float(cv_scores.mean()), 4),
        "cv_f1_std":         round(float(cv_scores.std()), 4),
        "n_train_samples":   len(X_train),
        "min_features_required": spec.get("min_features_required", 2),
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    joblib.dump(le, os.path.join(out_dir, "label_encoder.pkl"))

    # Confusion matrix plot
    _plot_confusion(y_test, y_pred, le.classes_, name)

    print(f"  💾 Saved to {out_dir}/")
    return True


def _plot_confusion(y_test, y_pred, classes, name):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(max(6, len(classes)), max(5, len(classes)-1)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {name.capitalize()} Specialist")
    plt.tight_layout()
    res_dir = os.path.join(RESULTS_DIR, name)
    os.makedirs(res_dir, exist_ok=True)
    plt.savefig(os.path.join(res_dir, "confusion_matrix.png"), dpi=150)
    plt.close()


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train specialist models")
    parser.add_argument("--only", nargs="+", choices=list(SPECIALISTS.keys()),
                        help="Train only specific specialists")
    args = parser.parse_args()

    os.makedirs(SPECIALIST_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    targets = args.only if args.only else list(SPECIALISTS.keys())

    print(f"\n🩸 HemaLens Specialist Training Pipeline")
    print(f"   Training: {', '.join(targets)}")
    print(f"   Raw data: {RAW_DIR}/")

    results = {}
    for name in targets:
        success = train_specialist(name, SPECIALISTS[name])
        results[name] = "✅ trained" if success else "⚠️  skipped"

    print(f"\n{'='*55}")
    print("  TRAINING SUMMARY")
    print(f"{'='*55}")
    for name, status in results.items():
        print(f"  {name:<12} {status}")
    print(f"\n  Models saved to: {SPECIALIST_DIR}/")


if __name__ == "__main__":
    main()