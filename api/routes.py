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
from collections import defaultdict, deque
from typing import Optional

from fastapi import APIRouter, File, UploadFile, Query, HTTPException, Body, Request
from fastapi.responses import JSONResponse

from api.schemas import (
    TTSRequest,
    ParamsRequest, SpecialistAnalyzeResponse, SpecialistResult,
    ExtractResponse, ReferenceRangesResponse, HealthResponse,
    AbnormalFlag, SpecialistsListResponse,
)
from ml.config import REFERENCE_RANGES

router = APIRouter()

ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".txt"}
ALLOWED_CONTENT_TYPES = {
    "application/pdf",
    "image/png",
    "image/jpeg",
    "image/tiff",
    "image/bmp",
    "text/plain",
    "application/octet-stream",
}
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
UPLOAD_CHUNK_BYTES = 1024 * 1024
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "30"))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))
RATE_LIMIT_PATHS = {
    "/api/v1/analyze/file",
    "/api/v1/analyze/params",
    "/api/v1/tts",
    "/api/v1/narrate",
}
_RATE_LIMIT_BUCKETS: dict[tuple[str, str], deque[float]] = defaultdict(deque)


def _client_ip(request: Request) -> str:
    forwarded_for = request.headers.get("x-forwarded-for", "")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _enforce_rate_limit(request: Request) -> None:
    if request.url.path not in RATE_LIMIT_PATHS:
        return

    now = time.monotonic()
    bucket = _RATE_LIMIT_BUCKETS[(_client_ip(request), request.url.path)]
    cutoff = now - RATE_LIMIT_WINDOW_SECONDS

    while bucket and bucket[0] < cutoff:
        bucket.popleft()

    if len(bucket) >= RATE_LIMIT_REQUESTS:
        retry_after = max(1, int(RATE_LIMIT_WINDOW_SECONDS - (now - bucket[0])))
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please try again shortly.",
            headers={"Retry-After": str(retry_after)},
        )

    bucket.append(now)


def _save_upload(file: UploadFile) -> str:
    ext = os.path.splitext(file.filename or "")[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )
    if file.content_type and file.content_type.lower() not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported content type '{file.content_type}'.",
        )

    tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    total = 0
    try:
        while True:
            chunk = file.file.read(UPLOAD_CHUNK_BYTES)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large. Maximum allowed size is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
                )
            tmp.write(chunk)
        tmp.close()
        return tmp.name
    except Exception:
        tmp.close()
        os.unlink(tmp.name)
        raise


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
    request: Request,
    file: UploadFile = File(..., description="PDF, image, or plain-text lab report"),
    gender: str = Query("male", enum=["male", "female"]),
):
    _enforce_rate_limit(request)
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
async def analyze_params(request: Request, body: ParamsRequest):
    _enforce_rate_limit(request)
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


# ─────────────────────────────────────────────────────────────
# POST /tts
# ─────────────────────────────────────────────────────────────
@router.post(
    "/tts",
    summary="Convert diagnosis result to spoken audio via Sarvam AI TTS",
    response_class=JSONResponse,
)
async def text_to_speech(request: Request, body: "TTSRequest"):
    _enforce_rate_limit(request)
    from fastapi.responses import Response
    from api.narrator import narrate
    from api.sarvam import translate_text, text_to_speech as sarvam_tts, LANGUAGE_CODES

    # Validate language
    lang = body.language.lower()
    if lang not in LANGUAGE_CODES:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported language '{lang}'. Supported: {', '.join(LANGUAGE_CODES.keys())}"
        )

    # Step 1 — Generate natural English sentences from result JSON
    english_text = narrate(body.result, mode=body.mode)

    # Step 2 — Translate to target language (skip if English)
    spoken_text = translate_text(english_text, lang)

    # Step 3 — Convert to speech
    audio_bytes = sarvam_tts(spoken_text, lang, body.mode)

    if audio_bytes is None:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=503,
            detail="Sarvam TTS unavailable. Check SARVAM_API_KEY environment variable."
        )

    return Response(
        content=audio_bytes,
        media_type="audio/wav",
        headers={"Content-Disposition": "inline; filename=hemalens_result.wav"}
    )


# ─────────────────────────────────────────────────────────────
# POST /narrate
# Returns the text that would be spoken — useful for debugging
# and for displaying on-screen alongside audio
# ─────────────────────────────────────────────────────────────
@router.post(
    "/narrate",
    summary="Get the spoken text for a diagnosis result (no audio — text only)",
)
async def narrate_text(request: Request, body: "TTSRequest"):
    _enforce_rate_limit(request)
    from api.narrator import narrate
    from api.sarvam import translate_text, LANGUAGE_CODES

    lang = body.language.lower()
    english_text = narrate(body.result, mode=body.mode)

    if lang != "english" and lang in LANGUAGE_CODES:
        translated = translate_text(english_text, lang)
    else:
        translated = english_text

    return {
        "mode":          body.mode,
        "language":      lang,
        "english_text":  english_text,
        "spoken_text":   translated,
        "char_count":    len(translated),
    }
