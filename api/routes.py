"""
API route handlers for HemaLens.
All routes are mounted under /api/v1 by main.py.

Two analysis modes:
  - /analyze/file   → upload a report file (NLP extract → specialist inference)
  - /analyze/params → submit params directly → specialist inference
  - /extract        → NLP extraction only, no inference
  - /extract/text   → same but from pasted text
  - /specialists    → list available trained specialists
  - /reference-ranges → normal ranges lookup
  - /health         → health check
"""
import os
import tempfile
import time
from typing import Optional

from fastapi import APIRouter, File, UploadFile, Query, HTTPException, Body
from fastapi.responses import JSONResponse

from api.schemas import (
    ParamsRequest, SpecialistAnalyzeResponse, SpecialistResult,
    ExtractResponse, ReferenceRangesResponse, HealthResponse,
    AbnormalFlag, SpecialistsListResponse,
)
from ml.config import REFERENCE_RANGES

router = APIRouter()

ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".txt"}


def _save_upload(file: UploadFile) -> str:
    ext = os.path.splitext(file.filename or "")[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )
    tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    tmp.write(file.file.read())
    tmp.close()
    return tmp.name


def _run_specialists(params: dict, gender: str) -> dict:
    """Run specialist inference, fall back to single model if specialists not trained."""
    try:
        from ml.specialist_inference import predict_all_specialists
        return predict_all_specialists(params, gender)
    except RuntimeError:
        # No specialist models — fall back to legacy single model
        from ml.inference import predict_single
        r = predict_single(params, gender)
        return {
            "specialist_results": [{
                "specialist":    "general",
                "description":   "General diagnostic model",
                "diagnosis":     r["diagnosis"],
                "confidence":    r["confidence"],
                "probabilities": r.get("probabilities", {}),
                "features_used": list(params.keys()),
                "features_missing": [],
                "model_f1":      None,
                "active":        True,
            }],
            "skipped_specialists":  [],
            "n_active":             1,
            "n_skipped":            0,
            "detected_conditions":  [r["diagnosis"]] if "normal" not in r["diagnosis"].lower() else [],
            "abnormal_flags":       r["abnormal_flags"],
            "total_abnormal":       r["total_abnormal"],
            "risk_level":           r["risk_level"],
            "summary":              r["diagnosis"],
            "recommendations":      r["recommendations"],
        }


# ─────────────────────────────────────────────────────────────
# POST /analyze/file
# ─────────────────────────────────────────────────────────────
@router.post(
    "/analyze/file",
    response_model=SpecialistAnalyzeResponse,
    summary="Upload a lab report → NLP extraction → all specialist models",
)
async def analyze_file(
    file: UploadFile = File(..., description="PDF, image, or plain-text lab report"),
    gender: str = Query("male", enum=["male", "female"]),
):
    t0 = time.perf_counter()
    tmp_path = _save_upload(file)

    try:
        from nlp.extractor import extract_from_file
        params = extract_from_file(tmp_path)

        if not params:
            raise HTTPException(
                status_code=422,
                detail="No blood parameters could be extracted. "
                       "Ensure the report has labeled numeric values.",
            )

        result = _run_specialists(params, gender)
        elapsed = round(time.perf_counter() - t0, 3)

        return SpecialistAnalyzeResponse(
            source=file.filename or "unknown",
            gender=gender,
            params_extracted=len(params),
            params=params,
            specialist_results=result["specialist_results"],
            skipped_specialists=result["skipped_specialists"],
            n_active=result["n_active"],
            n_skipped=result["n_skipped"],
            detected_conditions=result["detected_conditions"],
            abnormal_flags=result["abnormal_flags"],
            total_abnormal=result["total_abnormal"],
            risk_level=result["risk_level"],
            summary=result["summary"],
            recommendations=result["recommendations"],
            processing_time_s=elapsed,
        )
    finally:
        os.unlink(tmp_path)


# ─────────────────────────────────────────────────────────────
# POST /analyze/params
# ─────────────────────────────────────────────────────────────
@router.post(
    "/analyze/params",
    response_model=SpecialistAnalyzeResponse,
    summary="Submit parameter values as JSON → all specialist models",
)
async def analyze_params(body: ParamsRequest):
    if not body.params:
        raise HTTPException(status_code=422, detail="params must not be empty.")

    t0 = time.perf_counter()
    result  = _run_specialists(body.params, body.gender)
    elapsed = round(time.perf_counter() - t0, 3)

    return SpecialistAnalyzeResponse(
        source="manual_input",
        gender=body.gender,
        params_extracted=len(body.params),
        params=body.params,
        specialist_results=result["specialist_results"],
        skipped_specialists=result["skipped_specialists"],
        n_active=result["n_active"],
        n_skipped=result["n_skipped"],
        detected_conditions=result["detected_conditions"],
        abnormal_flags=result["abnormal_flags"],
        total_abnormal=result["total_abnormal"],
        risk_level=result["risk_level"],
        summary=result["summary"],
        recommendations=result["recommendations"],
        processing_time_s=elapsed,
    )


# ─────────────────────────────────────────────────────────────
# POST /extract
# ─────────────────────────────────────────────────────────────
@router.post("/extract", response_model=ExtractResponse,
             summary="NLP extraction only — no diagnosis")
async def extract_only(file: UploadFile = File(...)):
    tmp_path = _save_upload(file)
    try:
        from nlp.extractor import extract_from_file
        params = extract_from_file(tmp_path)
        return ExtractResponse(source=file.filename or "unknown",
                               params=params, count=len(params))
    finally:
        os.unlink(tmp_path)


# ─────────────────────────────────────────────────────────────
# POST /extract/text
# ─────────────────────────────────────────────────────────────
@router.post("/extract/text", response_model=ExtractResponse,
             summary="Extract parameters from pasted text")
async def extract_text(text: str = Body(..., embed=True)):
    from nlp.extractor import extract_from_text
    params = extract_from_text(text)
    return ExtractResponse(source="pasted_text", params=params, count=len(params))


# ─────────────────────────────────────────────────────────────
# GET /specialists
# ─────────────────────────────────────────────────────────────
@router.get("/specialists", response_model=SpecialistsListResponse,
            summary="List available trained specialist models")
def list_specialists():
    from ml.specialist_inference import load_all_specialists
    loaded = load_all_specialists()
    specialists = []
    for name, (_, metadata, _) in sorted(loaded.items()):
        specialists.append({
            "name":          name,
            "description":   metadata.get("description", name),
            "features":      metadata.get("feature_names", []),
            "classes":       metadata.get("target_classes", []),
            "f1_score":      metadata.get("test_f1_weighted"),
            "n_train":       metadata.get("n_train_samples"),
            "min_features":  metadata.get("min_features_required", 2),
        })
    return SpecialistsListResponse(specialists=specialists, count=len(specialists))


# ─────────────────────────────────────────────────────────────
# GET /reference-ranges
# ─────────────────────────────────────────────────────────────
@router.get("/reference-ranges", response_model=ReferenceRangesResponse,
            summary="Normal reference ranges for all parameters")
def get_reference_ranges(gender: str = Query("male", enum=["male", "female"])):
    ranges = {}
    for param, rr in REFERENCE_RANGES.items():
        if "both" in rr:
            low, high = rr["both"]
        elif gender in rr:
            low, high = rr[gender]
        else:
            low, high = list(v for k, v in rr.items() if k != "unit")[0]
        ranges[param] = {"low": low, "high": high, "unit": rr.get("unit", "")}
    return ReferenceRangesResponse(gender=gender, ranges=ranges, count=len(ranges))


# ─────────────────────────────────────────────────────────────
# GET /health
# ─────────────────────────────────────────────────────────────
@router.get("/health", response_model=HealthResponse, summary="Health check")
def health():
    from ml.specialist_inference import available_specialists
    trained = available_specialists()
    return HealthResponse(
        status="ok",
        model_loaded=len(trained) > 0,
        specialists_trained=trained,
        version="2.0.0",
    )