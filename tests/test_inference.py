"""
Unit tests for ml/inference.py (reference-range logic, severity, risk)
These tests run WITHOUT requiring a trained model — they test the
rule-based helpers that always run alongside the ML prediction.

Run:  pytest tests/test_inference.py -v
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from ml.inference import (
    check_reference_ranges,
    classify_severity,
    _risk_level,
    _recommendations,
)


# ── classify_severity ────────────────────────────────────────

class TestClassifySeverity:
    def test_mild_low(self):
        # 5% below lower bound
        assert classify_severity(9.5, 10.0, 20.0) == "MILD"

    def test_moderate_low(self):
        # 20% below lower bound
        assert classify_severity(8.0, 10.0, 20.0) == "MODERATE"

    def test_severe_low(self):
        # 50% below lower bound
        assert classify_severity(5.0, 10.0, 20.0) == "SEVERE"

    def test_mild_high(self):
        assert classify_severity(21.0, 10.0, 20.0) == "MILD"

    def test_severe_high(self):
        assert classify_severity(35.0, 10.0, 20.0) == "SEVERE"


# ── check_reference_ranges ───────────────────────────────────

class TestCheckReferenceRanges:
    def test_normal_returns_empty(self):
        flags = check_reference_ranges({"Hemoglobin": 15.0}, gender="male")
        assert flags == []

    def test_low_hemoglobin_flagged(self):
        flags = check_reference_ranges({"Hemoglobin": 9.0}, gender="male")
        assert len(flags) == 1
        assert flags[0]["parameter"] == "Hemoglobin"
        assert flags[0]["status"] == "LOW"

    def test_high_wbc_flagged(self):
        flags = check_reference_ranges({"WBC": 15.0})
        assert any(f["parameter"] == "WBC" and f["status"] == "HIGH" for f in flags)

    def test_high_glucose_flagged(self):
        flags = check_reference_ranges({"Glucose": 200.0})
        assert any(f["parameter"] == "Glucose" for f in flags)

    def test_gender_specific_female_hemoglobin(self):
        # 12.5 is normal for female (12.0–15.5) but low for male (13.5–17.5)
        female_flags = check_reference_ranges({"Hemoglobin": 12.5}, gender="female")
        male_flags   = check_reference_ranges({"Hemoglobin": 12.5}, gender="male")
        assert female_flags == []
        assert len(male_flags) == 1 and male_flags[0]["status"] == "LOW"

    def test_unknown_param_skipped(self):
        flags = check_reference_ranges({"NonExistentParam": 999.9})
        assert flags == []

    def test_multiple_abnormal(self):
        params = {"Hemoglobin": 8.0, "WBC": 15.0, "Glucose": 180.0}
        flags  = check_reference_ranges(params, gender="male")
        assert len(flags) == 3

    def test_severity_in_flag(self):
        flags = check_reference_ranges({"Hemoglobin": 5.0}, gender="male")
        assert "severity" in flags[0]
        assert flags[0]["severity"] in ("MILD", "MODERATE", "SEVERE")

    def test_normal_range_string_in_flag(self):
        flags = check_reference_ranges({"Glucose": 200.0})
        assert "normal_range" in flags[0]
        assert "–" in flags[0]["normal_range"]


# ── _risk_level ──────────────────────────────────────────────

class TestRiskLevel:
    def test_no_flags_is_normal(self):
        assert _risk_level([]) == "NORMAL"

    def test_mild_flag_is_low(self):
        flags = [{"severity": "MILD"}]
        assert _risk_level(flags) == "LOW"

    def test_moderate_flag_is_moderate(self):
        flags = [{"severity": "MILD"}, {"severity": "MODERATE"}]
        assert _risk_level(flags) == "MODERATE"

    def test_severe_flag_is_high(self):
        flags = [{"severity": "MILD"}, {"severity": "SEVERE"}]
        assert _risk_level(flags) == "HIGH"


# ── _recommendations ─────────────────────────────────────────

class TestRecommendations:
    def test_anemia_keyword(self):
        recs = _recommendations("Anemia", [])
        assert any("hematologist" in r.lower() or "iron" in r.lower() for r in recs)

    def test_diabetes_keyword(self):
        recs = _recommendations("Diabetes Mellitus", [])
        assert any("endocrinol" in r.lower() or "glucose" in r.lower() for r in recs)

    def test_normal_gives_followup(self):
        recs = _recommendations("Normal Blood Profile", [])
        assert len(recs) >= 1

    def test_severe_params_mentioned(self):
        flags = [
            {"parameter": "Hemoglobin", "severity": "SEVERE"},
            {"parameter": "WBC",        "severity": "MILD"},
        ]
        recs = _recommendations("Anemia", flags)
        severe_rec = " ".join(recs)
        assert "Hemoglobin" in severe_rec or "urgent" in severe_rec.lower()

    def test_returns_list(self):
        recs = _recommendations("Unknown Condition", [])
        assert isinstance(recs, list)