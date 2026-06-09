"""
Unit tests for nlp/extractor.py
Run:  pytest tests/test_extractor.py -v
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from nlp.extractor import extract_from_text


# ── FIXTURES ─────────────────────────────────────────────────

COLON_REPORT = """
COMPLETE BLOOD COUNT (CBC)
Hemoglobin     : 10.2  g/dL      (13.5 - 17.5)
RBC Count      : 3.8   million/µL (4.5 - 5.9)
WBC Count      : 11.8  K/µL       (4.5 - 11.0)
Platelet Count : 210   K/µL       (150 - 400)
Hematocrit     : 31.5  %          (41 - 53)
MCV            : 78.4  fL         (80 - 100)
MCH            : 26.8  pg         (27 - 33)
Neutrophils    : 72    %          (40 - 70)
Lymphocytes    : 21    %          (20 - 40)
"""

BIOCHEMISTRY_REPORT = """
BIOCHEMISTRY
Fasting Glucose  : 126   mg/dL     (70 - 100)
HbA1c            : 7.2   %         (4.0 - 5.6)
Creatinine       : 1.1   mg/dL     (0.7 - 1.2)
Total Bilirubin  : 0.8   mg/dL     (0.2 - 1.2)
SGPT (ALT)       : 38    U/L       (7 - 56)
SGOT (AST)       : 35    U/L       (10 - 40)
TSH              : 2.3   mIU/L     (0.4 - 4.0)
Total Cholesterol: 215   mg/dL     (<200)
HDL Cholesterol  : 45    mg/dL     (40 - 60)
LDL Cholesterol  : 138   mg/dL     (<130)
Triglycerides    : 162   mg/dL     (<150)
"""

ALIAS_REPORT = """
Hb        11.5 g/dL
TLC       9800 /µL
PCV       36%
FBS       98 mg/dL
T. Bil    1.0 mg/dL
Alk Phos  88 U/L
"""

INLINE_REPORT = """
Hemoglobin 10.8 g/dL
WBC 7.2 K/µL
Platelets 185 K/µL
Glucose 95 mg/dL
"""


# ── TESTS: COLON FORMAT ──────────────────────────────────────

class TestColonFormat:
    def setup_method(self):
        self.result = extract_from_text(COLON_REPORT)

    def test_hemoglobin(self):
        assert "Hemoglobin" in self.result
        assert self.result["Hemoglobin"] == pytest.approx(10.2)

    def test_rbc(self):
        assert "RBC" in self.result
        assert self.result["RBC"] == pytest.approx(3.8)

    def test_wbc(self):
        assert "WBC" in self.result
        assert self.result["WBC"] == pytest.approx(11.8)

    def test_platelets(self):
        assert "Platelets" in self.result
        assert self.result["Platelets"] == pytest.approx(210.0)

    def test_hematocrit(self):
        assert "Hematocrit" in self.result
        assert self.result["Hematocrit"] == pytest.approx(31.5)

    def test_mcv(self):
        assert "MCV" in self.result
        assert self.result["MCV"] == pytest.approx(78.4)

    def test_neutrophils(self):
        assert "Neutrophils" in self.result
        assert self.result["Neutrophils"] == pytest.approx(72.0)


# ── TESTS: BIOCHEMISTRY REPORT ───────────────────────────────

class TestBiochemistry:
    def setup_method(self):
        self.result = extract_from_text(BIOCHEMISTRY_REPORT)

    def test_glucose(self):
        assert "Glucose" in self.result
        assert self.result["Glucose"] == pytest.approx(126.0)

    def test_hba1c(self):
        assert "HbA1c" in self.result
        assert self.result["HbA1c"] == pytest.approx(7.2)

    def test_alt_via_sgpt(self):
        assert "ALT" in self.result
        assert self.result["ALT"] == pytest.approx(38.0)

    def test_ast_via_sgot(self):
        assert "AST" in self.result
        assert self.result["AST"] == pytest.approx(35.0)

    def test_cholesterol(self):
        assert "Cholesterol" in self.result
        assert self.result["Cholesterol"] == pytest.approx(215.0)

    def test_ldl(self):
        assert "LDL" in self.result
        assert self.result["LDL"] == pytest.approx(138.0)

    def test_triglycerides(self):
        assert "Triglycerides" in self.result
        assert self.result["Triglycerides"] == pytest.approx(162.0)


# ── TESTS: ALIAS RECOGNITION ─────────────────────────────────

class TestAliases:
    def setup_method(self):
        self.result = extract_from_text(ALIAS_REPORT)

    def test_hb_alias(self):
        assert "Hemoglobin" in self.result
        assert self.result["Hemoglobin"] == pytest.approx(11.5)

    def test_tlc_alias(self):
        assert "WBC" in self.result

    def test_pcv_alias(self):
        assert "Hematocrit" in self.result

    def test_fbs_alias(self):
        assert "Glucose" in self.result

    def test_alk_phos_alias(self):
        assert "Alkaline_Phosphatase" in self.result


# ── TESTS: INLINE FORMAT ─────────────────────────────────────

class TestInlineFormat:
    def setup_method(self):
        self.result = extract_from_text(INLINE_REPORT)

    def test_hemoglobin(self):
        assert "Hemoglobin" in self.result
        assert self.result["Hemoglobin"] == pytest.approx(10.8)

    def test_wbc(self):
        assert "WBC" in self.result

    def test_glucose(self):
        assert "Glucose" in self.result


# ── TESTS: EDGE CASES ────────────────────────────────────────

class TestEdgeCases:
    def test_empty_string(self):
        result = extract_from_text("")
        assert isinstance(result, dict)

    def test_no_params(self):
        result = extract_from_text("This report has no parameters at all.")
        assert isinstance(result, dict)

    def test_returns_floats(self):
        result = extract_from_text(COLON_REPORT)
        for v in result.values():
            assert isinstance(v, float), f"Expected float, got {type(v)}"

    def test_no_negative_values(self):
        result = extract_from_text(COLON_REPORT)
        for k, v in result.items():
            assert v >= 0, f"Negative value for {k}: {v}"