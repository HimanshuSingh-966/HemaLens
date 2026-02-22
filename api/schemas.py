"""
Pydantic v2 schemas — HemaLens API.
All Field() calls use keyword-only arguments (Pydantic v2 compliant).
"""
from __future__ import annotations
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, field_validator


# ─────────────────────────────────────────────────────────────
# REQUEST
# ─────────────────────────────────────────────────────────────

class ParamsRequest(BaseModel):
    gender: str = Field(
        default="male",
        pattern="^(male|female)$",
        description="Patient gender — affects gender-specific reference ranges",
    )
    params: Dict[str, float] = Field(
        ...,
        description="Parameter name to numeric value mapping",
    )

    @field_validator("params")
    @classmethod
    def params_not_empty(cls, v: dict) -> dict:
        if not v:
            raise ValueError("params must contain at least one entry.")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "gender": "male",
                "params": {"Hemoglobin": 10.2, "WBC": 11.8, "Glucose": 126.0, "TSH": 0.1},
            }
        }
    }


# ─────────────────────────────────────────────────────────────
# SHARED PIECES
# ─────────────────────────────────────────────────────────────

class AbnormalFlag(BaseModel):
    parameter:    str   = Field(..., description="Parameter name")
    value:        float = Field(..., description="Measured value")
    status:       str   = Field(..., description="LOW or HIGH")
    normal_range: str   = Field(..., description="Normal range string")
    severity:     str   = Field(..., description="MILD, MODERATE, or SEVERE")


class SpecialistResult(BaseModel):
    """Result from one specialist model."""
    specialist:       str            = Field(..., description="Specialist name, e.g. anemia")
    description:      str            = Field(..., description="Human-readable domain description")
    diagnosis:        str            = Field(..., description="Predicted diagnosis")
    confidence:       Optional[float] = Field(None, ge=0.0, le=1.0, description="Prediction confidence")
    probabilities:    Dict[str, float] = Field(default_factory=dict, description="Per-class probabilities")
    features_used:    List[str]      = Field(default_factory=list, description="Features present in input")
    features_missing: List[str]      = Field(default_factory=list, description="Features absent from input")
    model_f1:         Optional[float] = Field(None, description="Model F1 score from training")
    active:           bool           = Field(default=True, description="Whether this specialist was activated")


class SkippedSpecialist(BaseModel):
    """A specialist that could not activate due to missing features."""
    specialist:  str  = Field(..., description="Specialist name")
    description: str  = Field(..., description="Domain description")
    reason:      str  = Field(..., description="Why it was skipped")
    active:      bool = Field(default=False)


# ─────────────────────────────────────────────────────────────
# RESPONSE — SPECIALIST ANALYSIS
# ─────────────────────────────────────────────────────────────

class SpecialistAnalyzeResponse(BaseModel):
    """
    Full multi-specialist analysis result.
    Returned by both /analyze/file and /analyze/params.
    """
    source:               str                    = Field(..., description="Filename or manual_input")
    gender:               str                    = Field(..., description="male or female")
    params_extracted:     int                    = Field(..., description="Number of parameters found")
    params:               Dict[str, float]        = Field(..., description="All extracted parameter values")

    # Specialist results
    specialist_results:   List[SpecialistResult]  = Field(default_factory=list)
    skipped_specialists:  List[SkippedSpecialist] = Field(default_factory=list)
    n_active:             int                    = Field(..., description="Number of activated specialists")
    n_skipped:            int                    = Field(..., description="Number of skipped specialists")

    # Aggregated findings
    detected_conditions:  List[str]              = Field(default_factory=list, description="Non-normal diagnoses from active specialists")
    abnormal_flags:       List[AbnormalFlag]      = Field(default_factory=list, description="Reference-range violations")
    total_abnormal:       int                    = Field(..., description="Total out-of-range parameters")
    risk_level:           str                    = Field(..., description="NORMAL | LOW | MODERATE | HIGH")
    summary:              str                    = Field(..., description="Plain-English combined summary")
    recommendations:      List[str]              = Field(default_factory=list)
    processing_time_s:    Optional[float]        = Field(None, description="Server processing time in seconds")

    model_config = {
        "json_schema_extra": {
            "example": {
                "source": "report.pdf",
                "gender": "male",
                "params_extracted": 14,
                "params": {"Hemoglobin": 10.2, "Glucose": 126.0},
                "specialist_results": [
                    {"specialist": "anemia", "diagnosis": "Iron Deficiency Anemia",
                     "confidence": 0.91, "active": True,
                     "description": "CBC-based anemia classification",
                     "probabilities": {}, "features_used": ["Hemoglobin"],
                     "features_missing": [], "model_f1": 0.94},
                ],
                "skipped_specialists": [],
                "n_active": 1, "n_skipped": 3,
                "detected_conditions": ["Iron Deficiency Anemia"],
                "abnormal_flags": [],
                "total_abnormal": 2,
                "risk_level": "MODERATE",
                "summary": "1 specialist activated. Detected: Iron Deficiency Anemia.",
                "recommendations": ["Hematology consult recommended."],
                "processing_time_s": 0.21,
            }
        }
    }


# ─────────────────────────────────────────────────────────────
# RESPONSE — EXTRACT ONLY
# ─────────────────────────────────────────────────────────────

class ExtractResponse(BaseModel):
    source: str              = Field(..., description="Source identifier")
    params: Dict[str, float] = Field(..., description="Extracted parameter values")
    count:  int              = Field(..., description="Number of parameters extracted")


# ─────────────────────────────────────────────────────────────
# RESPONSE — REFERENCE RANGES
# ─────────────────────────────────────────────────────────────

class RangeEntry(BaseModel):
    low:  float = Field(..., description="Lower bound")
    high: float = Field(..., description="Upper bound")
    unit: str   = Field(..., description="Unit of measurement")


class ReferenceRangesResponse(BaseModel):
    gender: str                   = Field(..., description="Gender these ranges apply to")
    ranges: Dict[str, RangeEntry] = Field(..., description="Parameter to range mapping")
    count:  int                   = Field(..., description="Total number of parameters")


# ─────────────────────────────────────────────────────────────
# RESPONSE — SPECIALISTS LIST
# ─────────────────────────────────────────────────────────────

class SpecialistInfo(BaseModel):
    name:         str        = Field(..., description="Specialist name")
    description:  str        = Field(..., description="Domain description")
    features:     List[str]  = Field(default_factory=list, description="Feature columns this specialist uses")
    classes:      List[str]  = Field(default_factory=list, description="Possible diagnosis classes")
    f1_score:     Optional[float] = Field(None, description="Test F1 score")
    n_train:      Optional[int]   = Field(None, description="Training sample count")
    min_features: int        = Field(default=2, description="Minimum features needed to activate")


class SpecialistsListResponse(BaseModel):
    specialists: List[SpecialistInfo] = Field(default_factory=list)
    count:       int                  = Field(..., description="Number of trained specialists")


# ─────────────────────────────────────────────────────────────
# RESPONSE — HEALTH
# ─────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status:              str       = Field(..., description="ok when running")
    model_loaded:        bool      = Field(..., description="True if at least one specialist is ready")
    specialists_trained: List[str] = Field(default_factory=list, description="Names of trained specialists")
    version:             str       = Field(..., description="API version")