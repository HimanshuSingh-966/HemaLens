"""
Blood Report Analysis - Training Script
Dataset: Diagnostic Pathology Test Results (Kaggle - pareshbadnore)
Compatible with similar CBC/blood panel datasets.
"""

import pandas as pd
import numpy as np
import os
import joblib
import json
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, accuracy_score, f1_score)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_PATH      = "data/diagnostic_pathology.csv"
MODEL_DIR      = "models"
RESULTS_DIR    = "results"
TARGET_COL     = "Diagnosis"          # adjust to actual target column
TEST_SIZE      = 0.2
RANDOM_STATE   = 42
CV_FOLDS       = 5

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# BLOOD PARAMETER REFERENCE RANGES
# Used for feature engineering & validation
# ─────────────────────────────────────────────
REFERENCE_RANGES = {
    "Hemoglobin":       {"male": (13.5, 17.5), "female": (12.0, 15.5), "unit": "g/dL"},
    "RBC":              {"male": (4.5, 5.9),   "female": (4.1, 5.1),   "unit": "million/µL"},
    "WBC":              {"both": (4.5, 11.0),  "unit": "thousand/µL"},
    "Platelets":        {"both": (150, 400),   "unit": "thousand/µL"},
    "Hematocrit":       {"male": (41, 53),     "female": (36, 46),     "unit": "%"},
    "MCV":              {"both": (80, 100),    "unit": "fL"},
    "MCH":              {"both": (27, 33),     "unit": "pg"},
    "MCHC":             {"both": (32, 36),     "unit": "g/dL"},
    "Neutrophils":      {"both": (40, 70),     "unit": "%"},
    "Lymphocytes":      {"both": (20, 40),     "unit": "%"},
    "Monocytes":        {"both": (2, 10),      "unit": "%"},
    "Eosinophils":      {"both": (1, 6),       "unit": "%"},
    "Basophils":        {"both": (0, 1),       "unit": "%"},
    "Glucose":          {"both": (70, 100),    "unit": "mg/dL"},
    "Creatinine":       {"male": (0.7, 1.2),   "female": (0.5, 1.0),   "unit": "mg/dL"},
    "Blood_Urea_Nitrogen": {"both": (7, 25),   "unit": "mg/dL"},
    "Sodium":           {"both": (136, 145),   "unit": "mEq/L"},
    "Potassium":        {"both": (3.5, 5.0),   "unit": "mEq/L"},
    "Calcium":          {"both": (8.5, 10.5),  "unit": "mg/dL"},
    "Total_Protein":    {"both": (6.0, 8.3),   "unit": "g/dL"},
    "Albumin":          {"both": (3.5, 5.0),   "unit": "g/dL"},
    "Total_Bilirubin":  {"both": (0.2, 1.2),   "unit": "mg/dL"},
    "ALT":              {"male": (7, 56),       "female": (7, 45),      "unit": "U/L"},
    "AST":              {"both": (10, 40),      "unit": "U/L"},
    "Alkaline_Phosphatase": {"both": (44, 147),"unit": "U/L"},
    "TSH":              {"both": (0.4, 4.0),   "unit": "mIU/L"},
    "Iron":             {"male": (65, 175),    "female": (50, 170),    "unit": "µg/dL"},
    "Ferritin":         {"male": (20, 500),    "female": (20, 200),    "unit": "ng/mL"},
    "Cholesterol":      {"both": (0, 200),     "unit": "mg/dL"},
    "HDL":              {"male": (40, 60),     "female": (50, 60),     "unit": "mg/dL"},
    "LDL":              {"both": (0, 130),     "unit": "mg/dL"},
    "Triglycerides":    {"both": (0, 150),     "unit": "mg/dL"},
    "HbA1c":            {"both": (4.0, 5.6),   "unit": "%"},
}


# ─────────────────────────────────────────────
# DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[-1].lower()
    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")
    print(f"✅ Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


def preprocess(df: pd.DataFrame, target_col: str, gender_col: str = "Gender"):
    print("\n📊 Preprocessing...")

    # Drop duplicates & fully-empty rows
    df = df.drop_duplicates().dropna(how="all")

    # Encode gender if present
    if gender_col in df.columns:
        df[gender_col] = df[gender_col].str.lower().map({"male": 1, "female": 0, "m": 1, "f": 0})

    # Encode target
    le = LabelEncoder()
    df[target_col] = le.fit_transform(df[target_col].astype(str))
    print(f"   Target classes: {list(le.classes_)}")

    # Feature engineering: flag abnormal values
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != target_col]

    for param, ranges in REFERENCE_RANGES.items():
        if param in df.columns:
            low, high = (ranges.get("both") or ranges.get("male", (None, None)))
            if low is not None:
                df[f"{param}_abnormal"] = ((df[param] < low) | (df[param] > high)).astype(int)
                df[f"{param}_deviation"] = df[param].apply(
                    lambda x: (x - low) / (high - low + 1e-9) if pd.notnull(x) else np.nan
                )

    # Count total abnormal parameters
    abnormal_cols = [c for c in df.columns if c.endswith("_abnormal")]
    if abnormal_cols:
        df["total_abnormal_count"] = df[abnormal_cols].sum(axis=1)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Drop non-numeric
    X = X.select_dtypes(include=[np.number])

    print(f"   Features after engineering: {X.shape[1]}")
    return X, y, le


# ─────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────
def get_models():
    return {
        "RandomForest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
            ("clf",     RandomForestClassifier(n_estimators=300, max_depth=10,
                                               min_samples_split=5, class_weight="balanced",
                                               random_state=RANDOM_STATE, n_jobs=-1)),
        ]),
        "GradientBoosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
            ("clf",     GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                                   max_depth=5, random_state=RANDOM_STATE)),
        ]),
        "XGBoost": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf",     xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                                           use_label_encoder=False, eval_metric="mlogloss",
                                           random_state=RANDOM_STATE, n_jobs=-1)),
        ]),
        "LightGBM": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf",     lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05,
                                           num_leaves=63, class_weight="balanced",
                                           random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)),
        ]),
    }


# ─────────────────────────────────────────────
# TRAINING & EVALUATION
# ─────────────────────────────────────────────
def train_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    models    = get_models()
    results   = {}
    best_score = -1
    best_name  = None

    for name, pipeline in models.items():
        print(f"\n🔧 Training {name}...")
        cv  = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        cvs = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1_weighted", n_jobs=-1)
        print(f"   CV F1 (weighted): {cvs.mean():.4f} ± {cvs.std():.4f}")

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, average="weighted")
        try:
            y_prob = pipeline.predict_proba(X_test)
            auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")
        except Exception:
            auc = None

        results[name] = {"accuracy": acc, "f1_weighted": f1, "roc_auc": auc, "cv_f1_mean": cvs.mean()}
        print(f"   Test Accuracy: {acc:.4f} | F1: {f1:.4f}" + (f" | AUC: {auc:.4f}" if auc else ""))

        if f1 > best_score:
            best_score = f1
            best_name  = name

        # Save model
        joblib.dump(pipeline, os.path.join(MODEL_DIR, f"{name}.pkl"))

    print(f"\n🏆 Best model: {best_name} (F1={best_score:.4f})")

    # Save best model as primary
    import shutil
    shutil.copy(os.path.join(MODEL_DIR, f"{best_name}.pkl"),
                os.path.join(MODEL_DIR, "best_model.pkl"))

    # Save results
    with open(os.path.join(RESULTS_DIR, "model_comparison.json"), "w") as f:
        json.dump(results, f, indent=2)

    return models[best_name], X_test, y_test, results, best_name


# ─────────────────────────────────────────────
# FEATURE IMPORTANCE PLOT
# ─────────────────────────────────────────────
def plot_feature_importance(pipeline, feature_names, model_name):
    try:
        clf = pipeline.named_steps["clf"]
        if hasattr(clf, "feature_importances_"):
            imp = pd.Series(clf.feature_importances_, index=feature_names).sort_values(ascending=False)[:20]
            plt.figure(figsize=(10, 6))
            sns.barplot(x=imp.values, y=imp.index, palette="viridis")
            plt.title(f"Top 20 Feature Importances — {model_name}", fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS_DIR, "feature_importance.png"), dpi=150)
            plt.close()
            print("   📈 Feature importance plot saved.")
    except Exception as e:
        print(f"   ⚠️ Could not plot feature importance: {e}")


# ─────────────────────────────────────────────
# CONFUSION MATRIX PLOT
# ─────────────────────────────────────────────
def plot_confusion_matrix(y_test, y_pred, label_encoder):
    cm = confusion_matrix(y_test, y_pred)
    labels = label_encoder.classes_
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()


# ─────────────────────────────────────────────
# SAVE FEATURE METADATA
# ─────────────────────────────────────────────
def save_metadata(X, label_encoder, feature_names):
    meta = {
        "feature_names": feature_names,
        "num_features": len(feature_names),
        "target_classes": list(label_encoder.classes_),
        "reference_ranges": {k: {kk: list(vv) if isinstance(vv, tuple) else vv
                                 for kk, vv in v.items()} for k, v in REFERENCE_RANGES.items()},
    }
    with open(os.path.join(MODEL_DIR, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("💾 Metadata saved.")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  BLOOD REPORT ANALYSIS — TRAINING PIPELINE")
    print("=" * 60)

    df      = load_data(DATA_PATH)
    X, y, le = preprocess(df, TARGET_COL)
    feature_names = list(X.columns)

    best_pipeline, X_test, y_test, results, best_name = train_evaluate(X, y)

    y_pred = best_pipeline.predict(X_test)
    print(f"\n📋 Classification Report ({best_name}):")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    plot_feature_importance(best_pipeline, feature_names, best_name)
    plot_confusion_matrix(y_test, y_pred, le)
    save_metadata(X, le, feature_names)

    print("\n✅ Training complete! Artifacts saved to:")
    print(f"   Models  → {MODEL_DIR}/")
    print(f"   Results → {RESULTS_DIR}/")


if __name__ == "__main__":
    main()