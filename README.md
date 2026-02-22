# Blood Report Analysis System — Setup & Usage

## Project Structure
```
blood_report_project/
├── ui/
│   └── index.html          # Full UI (open in browser directly)
├── ml/
│   ├── train.py            # Training pipeline
│   └── inference.py        # Inference & prediction
├── nlp/
│   └── extractor.py        # NLP parameter extraction
├── data/                   # Place your dataset CSV here
│   └── diagnostic_pathology.csv
├── models/                 # Auto-created during training
├── results/                # Auto-created during training
└── requirements.txt
```

---

## Installation

```bash
pip install pandas numpy scikit-learn xgboost lightgbm \
            matplotlib seaborn joblib \
            pdfplumber pytesseract Pillow \
            spacy
python -m spacy download en_core_web_sm
```

---

## Usage

### 1. Training
```bash
cd ml
# Edit DATA_PATH and TARGET_COL in train.py to match your dataset
python train.py
```
Outputs:
- `models/best_model.pkl` — Best performing model
- `models/metadata.json` — Feature names & class labels
- `results/feature_importance.png`
- `results/confusion_matrix.png`
- `results/model_comparison.json`

### 2. Inference — Single Sample
```bash
python inference.py single \
  --params '{"Hemoglobin":10.2,"WBC":11.8,"Glucose":126,"HbA1c":7.2}' \
  --gender male
```

### 3. Inference — Batch CSV
```bash
python inference.py batch --input data/test.csv --output results/predictions.csv
```

### 4. NLP Extraction
```bash
# From text file
python nlp/extractor.py --file report.txt

# From PDF
python nlp/extractor.py --file report.pdf --output extracted.json

# From raw text
python nlp/extractor.py --text "Hemoglobin: 10.2 g/dL, WBC: 11.8, HbA1c: 7.2"

# Demo mode (no args)
python nlp/extractor.py
```

### 5. UI
Open `ui/index.html` in any browser. No server required.
For full PDF/image support, connect to the Python FastAPI backend (see below).

---

## Recommended Datasets

| Dataset | Source | Best For |
|---------|--------|----------|
| Diagnostic Pathology Test Results | Kaggle: pareshbadnore | Primary (CBC + metabolic) |
| Laboratory Test Results – Anonymized | Kaggle: pinuto | Complementary real-world labs |
| Disease Symptoms & Patient Profile | Kaggle: uom190346a | Symptom + lab fusion |
| MIMIC-III Clinical Data | PhysioNet | ICU/clinical (requires credentialing) |
| UK Biobank Blood Panel | UK Biobank | Large population study |
| OpenMRS Demo Data | OpenMRS | EHR integration testing |

---

## FastAPI Backend Integration (Optional)

```python
from fastapi import FastAPI, File, UploadFile
from nlp.extractor import extract_from_file
from ml.inference import predict_single
import tempfile, os

app = FastAPI()

@app.post("/analyze")
async def analyze(file: UploadFile = File(...), gender: str = "male"):
    with tempfile.NamedTemporaryFile(suffix=os.path.splitext(file.filename)[1], delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    params    = extract_from_file(tmp_path)
    result    = predict_single(params, gender)
    os.unlink(tmp_path)
    return {"params": params, "result": result}
```

Run: `uvicorn main:app --reload`

---

## Notes on Dataset Column Mapping

If the Kaggle dataset uses different column names, edit `train.py`:
```python
TARGET_COL = "Diagnosis"  # Change to your actual target column name
```

The extractor and inference pipeline are column-name agnostic and rely on
the parameter aliases defined in `PARAMETER_ALIASES` (nlp/extractor.py).