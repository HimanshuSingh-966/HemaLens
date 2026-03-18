"""
HemaLens — Result Narrator
===========================
Converts the specialist ML diagnosis JSON into natural, speakable
English sentences in two modes:

  patient  — simple, calm language for non-medical users
  clinical — concise, technical language for clinicians

These sentences are then passed to Sarvam TTS for voice output.

Note: Recommendation templates here are intentionally generic until
the HemaLens Doctor Questionnaire responses are collected and
validated clinical language is incorporated.
"""
from typing import Optional


# ─────────────────────────────────────────────────────────────
# SEVERITY LABELS
# ─────────────────────────────────────────────────────────────
_SEVERITY_PATIENT = {
    "MILD":     "slightly",
    "MODERATE": "moderately",
    "SEVERE":   "significantly abnormal (requires immediate attention)",
}

_SEVERITY_CLINICAL = {
    "MILD":     "mild",
    "MODERATE": "moderate",
    "SEVERE":   "severe",
}


# ─────────────────────────────────────────────────────────────
# CONDITION DESCRIPTIONS
# Patient-friendly plain language per diagnosis keyword
# ─────────────────────────────────────────────────────────────
_PATIENT_DESCRIPTIONS = {
    "iron deficiency anemia":  "low iron in your blood, which means your body is not making enough red blood cells",
    "anemia":                  "a low red blood cell count",
    "thalassemia":             "an inherited blood condition affecting your red blood cells",
    "aplastic anemia":         "a serious condition where your bone marrow is not making enough blood cells",
    "sickle cell":             "an inherited condition affecting the shape of your red blood cells",
    "normocytic anemia":       "a type of anemia where your red blood cells are normal in size but low in number",
    "macrocytic anemia":       "a type of anemia where your red blood cells are larger than normal",
    "microcytic anemia":       "a type of anemia where your red blood cells are smaller than normal",
    "diabetes mellitus":       "high blood sugar levels, which means your body is not managing sugar properly",
    "pre-diabetes":            "blood sugar levels that are higher than normal but not yet diabetes",
    "chronic kidney disease":  "a gradual loss of kidney function that needs careful management",
    "kidney disease":          "a condition affecting how well your kidneys are working",
    "liver disease":           "a condition affecting your liver function",
    "hypothyroidism":          "an underactive thyroid gland, meaning your thyroid is not making enough hormones",
    "hyperthyroidism":         "an overactive thyroid gland, meaning your thyroid is making too many hormones",
    "thyroid":                 "a thyroid gland abnormality",
}

# ─────────────────────────────────────────────────────────────
# RECOMMENDATION TEMPLATES
# These will be enriched after doctor questionnaire responses
# are incorporated. Current versions are clinically reviewed
# generic guidance.
# ─────────────────────────────────────────────────────────────
_PATIENT_RECOMMENDATIONS = {
    "iron deficiency anemia":  "Your blood's iron store is low. This is treatable. Please take iron supplements as prescribed by your doctor. Eat iron-rich foods like dal, spinach, jaggery, and meat. Avoid tea and coffee with your meals. Please consult your doctor.",
    "anemia":                  "Please visit your doctor for further tests to find the cause of your low blood count. Do not self-medicate based on this AI output.",
    "thalassemia":             "Please consult a blood specialist. Genetic counselling may also be helpful for your family.",
    "aplastic anemia":         "Please see a blood specialist urgently. This condition needs immediate medical attention.",
    "diabetes mellitus":       "Please monitor your blood sugar daily. Follow a low-sugar diet, exercise regularly, and take your medications as prescribed. Check your HbA1c every 3 months. Please consult your doctor.",
    "pre-diabetes":            "Your blood sugar is higher than normal. Please reduce sugar and refined carbohydrates in your diet, exercise for at least 30 minutes daily, and repeat your HbA1c test in 3 months. Please consult your doctor.",
    "chronic kidney disease":  "Please reduce protein and salt in your diet, drink the right amount of water as advised, and see a kidney specialist. Monitor your lab values regularly. Please consult your doctor.",
    "kidney disease":          "Please see a kidney specialist and monitor your kidney function regularly. Please consult your doctor and do not self-medicate.",
    "liver disease":           "This is treatable. Please avoid alcohol completely, take only doctor-approved medications, eat a healthy low-fat diet, and repeat your liver tests in 4 to 6 weeks. Please consult your doctor.",
    "hypothyroidism":          "Your thyroid hormone levels indicate an underactive thyroid. This is treatable. Your doctor may recommend thyroid hormone replacement. Please follow up with your doctor and avoid starting medications on your own.",
    "hyperthyroidism":         "Your thyroid hormone levels are high. Please consult your doctor promptly. Avoid caffeine and get adequate rest.",
    "thyroid":                 "Please consult your doctor for a complete thyroid evaluation. Do not self-medicate based on AI output.",
    "macrocytic anemia":       "Please check your Vitamin B12 and folate levels with your doctor, as deficiencies are very common, especially if you follow a vegetarian diet.",
    "microcytic anemia":       "Please check your iron levels and haemoglobin type with your doctor.",
}

_CLINICAL_RECOMMENDATIONS = {
    "iron deficiency anemia":  "Oral iron supplementation indicated. Confirm with serum ferritin, TIBC/transferrin saturation, and peripheral smear. Low MCV (<80 fL) + Low MCH (<27 pg) + High RDW (>14.5%) pattern. CBC repeat at 6 weeks. High prevalence in Indian reproductive-age women; check dietary and menstrual history.",
    "anemia":                  "Further workup required. Consider full iron studies, serum B12, folate, reticulocyte count, peripheral blood smear, LFT, TFT, and Hb electrophoresis if thalassemia suspected. Note: B12/folate deficiency extremely common in Indian vegetarian population.",
    "thalassemia":             "Haematology referral. Haemoglobin electrophoresis and genetic counselling recommended. Note: IDA and thalassemia trait can coexist — treating only one leads to incomplete response.",
    "aplastic anemia":         "Urgent haematology referral. Bone marrow biopsy indicated. In Indian context: rule out visceral leishmaniasis (kala-azar) and post-viral hepatitis aplastic anemia.",
    "diabetes mellitus":       "Initiate or review antidiabetic therapy. HbA1c target < 7%. Annual screening for nephropathy, retinopathy, neuropathy. Note: metformin causes B12 depletion — common in Indian diabetics; monitor B12 levels.",
    "pre-diabetes":            "Lifestyle modification programme. Repeat HbA1c in 3 months. Consider metformin if high-risk.",
    "chronic kidney disease":  "Nephrology referral. eGFR-based CKD staging (use CKD-EPI with Indian adjustment — creatinine may underestimate GFR due to lower muscle mass). Dietary protein restriction. ACE inhibitor or ARB if proteinuria. Monitor electrolytes for hyperkalemia, hyponatremia, metabolic acidosis.",
    "kidney disease":          "Nephrology referral. Monitor creatinine, eGFR, electrolytes. Urine albumin:creatinine ratio. Renal ultrasound. HbA1c (diabetes is leading cause of CKD in India).",
    "liver disease":           "Hepatology review. Avoid hepatotoxic agents. Repeat LFTs in 4-6 weeks. Mandatory viral hepatitis serology (HBsAg, Anti-HCV) given high carrier rate in India. Consider NAFLD (highly prevalent in urban India). USG abdomen recommended.",
    "hypothyroidism":          "Levothyroxine initiation or dose adjustment. Target TSH 0.5-2.5 mIU/L. Repeat TFT in 6-8 weeks. Anti-TPO antibody (Hashimoto's — most common cause in urban India, especially women 25-45 yrs). Trimester-specific TSH ranges for pregnant patients.",
    "hyperthyroidism":         "Antithyroid therapy evaluation. Consider propylthiouracil or carbimazole. Radioiodine assessment as indicated. In India: consider autonomous thyroid nodule or Graves' in young women.",
    "thyroid":                 "Full thyroid panel: TSH, Free T3, Free T4. Anti-TPO antibodies if autoimmune suspected. Endocrinology consult. Thyroid USG for nodule evaluation.",
    "macrocytic anemia":       "Vitamin B12 and folate assay. B12 IM injections or high-dose oral supplementation if deficient. Extremely common in Indian vegetarian population.",
    "microcytic anemia":       "Iron studies and haemoglobin electrophoresis. Oral iron if IDA confirmed. Rule out coexisting thalassemia trait.",
}


# ─────────────────────────────────────────────────────────────
# MAIN NARRATOR FUNCTION
# ─────────────────────────────────────────────────────────────
def narrate(result: dict, mode: str = "patient") -> str:
    """
    Convert a specialist analysis result dict into a natural spoken sentence block.

    Args:
        result: the full JSON response from /analyze/file or /analyze/params
        mode:   'patient' or 'clinical'

    Returns:
        A single string ready to be sent to Sarvam TTS.
    """
    if mode == "clinical":
        return _narrate_clinical(result)
    return _narrate_patient(result)


def _narrate_patient(result: dict) -> str:
    """Simple, friendly language for the general patient."""
    lines = []

    # Opening
    lines.append("Hello. Here is a summary of your blood test results.")

    # Conditions found
    conditions = result.get("detected_conditions", [])
    if not conditions or all("normal" in c.lower() for c in conditions):
        lines.append("Your results are within the normal range. No significant abnormalities were detected.")
    else:
        lines.append(f"Your blood test has found {len(conditions)} condition{'s' if len(conditions) > 1 else ''} that need your attention.")
        for cond in conditions:
            desc = _match_description(cond, _PATIENT_DESCRIPTIONS)
            if desc:
                lines.append(f"You have been found to have {desc}.")
            else:
                lines.append(f"Your results suggest {cond}.")

    # Abnormal parameters — mention only if any
    flags = result.get("abnormal_flags", [])
    if flags:
        severe = [f for f in flags if f.get("severity") == "SEVERE"]
        if severe:
            params_str = ", ".join(f["parameter"] for f in severe)
            lines.append(f"Some of your values are significantly out of range and need urgent attention: {params_str}.")
        else:
            lines.append(f"A total of {len(flags)} of your blood values were outside the normal range.")

    # Recommendations
    recs = _get_patient_recommendations(conditions)
    if recs:
        lines.append("Here is what you should do next.")
        for rec in recs:
            lines.append(rec)

    # Risk level
    risk = result.get("risk_level", "NORMAL")
    if risk == "HIGH":
        lines.append("Your overall risk level is high. Please see your doctor as soon as possible.")
    elif risk == "MODERATE":
        lines.append("Your overall risk level is moderate. Please schedule an appointment with your doctor.")
    else:
        lines.append("Please continue with routine follow-up as advised by your doctor.")

    # Disclaimer
    lines.append("This report is AI-generated and intended as clinical decision support only. It does not replace physician evaluation. Abnormal findings must always be correlated with clinical history and examination by a qualified doctor. Please consult your doctor and do not self-medicate.")

    return " ".join(lines)


def _narrate_clinical(result: dict) -> str:
    """Concise, technical summary for clinicians."""
    lines = []

    gender  = result.get("gender", "unknown")
    n_params = result.get("params_extracted", 0)
    risk    = result.get("risk_level", "NORMAL")

    lines.append(f"HemaLens clinical summary. Patient gender: {gender}. Parameters analysed: {n_params}. Overall risk: {risk}.")

    # Active specialists
    specialist_results = result.get("specialist_results", [])
    active = [s for s in specialist_results if s.get("active")]
    for s in active:
        diag = s.get("diagnosis", "")
        conf = s.get("confidence")
        f1   = s.get("model_f1")
        conf_str = f"{round(conf*100)}% confidence" if conf else ""
        f1_str   = f"Model F1: {round(f1, 3)}" if f1 else ""
        meta = ", ".join(filter(None, [conf_str, f1_str]))
        lines.append(f"{s['specialist'].capitalize()} specialist: {diag}. {meta}.")

    # Abnormal flags
    flags = result.get("abnormal_flags", [])
    if flags:
        flag_parts = []
        for f in flags:
            sev = _SEVERITY_CLINICAL.get(f.get("severity",""), "")
            flag_parts.append(f"{f['parameter']} {f.get('value','')} ({sev} {f.get('status','').lower()}, ref {f.get('normal_range','')})")
        lines.append("Abnormal parameters: " + "; ".join(flag_parts) + ".")

    # Recommendations
    conditions = result.get("detected_conditions", [])
    recs = _get_clinical_recommendations(conditions)
    if recs:
        lines.append("Recommendations: " + " ".join(recs))

    lines.append("This summary is AI-generated and intended as clinical decision support only. It does not replace physician evaluation. Abnormal findings must be correlated with clinical history and examination.")

    return " ".join(lines)


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def _match_description(condition: str, lookup: dict) -> Optional[str]:
    cond_lower = condition.lower()
    for key, desc in lookup.items():
        if key in cond_lower:
            return desc
    return None


def _get_patient_recommendations(conditions: list) -> list[str]:
    recs, seen = [], set()
    for cond in conditions:
        cond_lower = cond.lower()
        for key, rec in _PATIENT_RECOMMENDATIONS.items():
            if key in cond_lower and key not in seen:
                recs.append(rec)
                seen.add(key)
    if not recs:
        recs.append("Please follow up with your doctor for a full review of your results.")
    return recs


def _get_clinical_recommendations(conditions: list) -> list[str]:
    recs, seen = [], set()
    for cond in conditions:
        cond_lower = cond.lower()
        for key, rec in _CLINICAL_RECOMMENDATIONS.items():
            if key in cond_lower and key not in seen:
                recs.append(rec)
                seen.add(key)
    return recs