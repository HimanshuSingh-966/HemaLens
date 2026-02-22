# HemaLens — AI Blood Report Analysis

> Upload a lab report. Get instant diagnostic insights from 5 specialist ML models.

**Live Demo:** [https://hemalens.netlify.app](https://hemalens.netlify.app)  
**API:** [https://hemalens-api.onrender.com/docs](https://hemalens.onrender.com) 

---

## What It Does

HemaLens extracts blood parameters from lab reports (PDF, image, or text) using an NLP pipeline, then runs them through 5 independently trained specialist ML models — one per disease domain.

| Specialist | F1 Score | Dataset Size | Detects |
|---|---|---|---|
| Anemia | 98.4% | 1,500 rows | Iron deficiency, thalassemia, aplastic, sickle cell |
| Diabetes | 96.9% | 100,000 rows | Diabetes mellitus, pre-diabetes |
| Kidney | 95.0% | 400 rows | Chronic kidney disease |
| Liver | 99.6% | 30,691 rows | Liver disease |
| Thyroid | 89.1% | 9,172 rows | Hyperthyroidism, hypothyroidism |

---

## Project Structure

```
HemaLens/
├── api/
│   ├── main.py                  # FastAPI app + CORS + static serving
│   ├── routes.py                # All API endpoints
│   └── schemas.py               # Pydantic request/response models
├── ml/
│   ├── config.py                # Reference ranges + app config
│   ├── inference.py             # Reference range checker + legacy fallback
│   ├── specialist_inference.py  # Multi-specialist inference engine
│   └── train_specialists.py    # Training pipeline (run locally only)
├── nlp/
│   ├── extractor.py             # PDF / image / text parameter extraction
│   └── aliases.py               # Lab label → canonical name mapping
├── ui/
│   └── index.html               # Frontend (single file, no build step)
├── models/
│   └── specialists/             # Trained .pkl models (committed to git)
│       ├── anemia/
│       ├── diabetes/
│       ├── kidney/
│       ├── liver/
│       └── thyroid/
├── data/
│   └── raw/                     # Training CSVs — local only, not committed
├── .env.example                 # Environment variable template
├── render.yaml                  # Render deployment config
└── requirements.txt
```

---

## Local Development

### 1. Install dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Configure environment

```bash
cp .env.example .env
# defaults work out of the box for local dev
```

### 3. Run the API

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

API docs: `http://localhost:8000/docs`  
UI: `http://localhost:8000/`

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/v1/analyze/file` | Upload PDF/image/txt → NLP extract → all specialists |
| POST | `/api/v1/analyze/params` | Submit params as JSON → all specialists |
| POST | `/api/v1/extract/text` | NLP extraction only, no diagnosis |
| GET | `/api/v1/specialists` | List trained specialist models |
| GET | `/api/v1/reference-ranges` | Normal ranges for all parameters |
| GET | `/api/v1/health` | Health check |

### Quick test

```bash
curl -X POST https://your-render-url.onrender.com/api/v1/analyze/params \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "male",
    "params": {
      "Hemoglobin": 10.2, "WBC": 11.8, "Glucose": 126,
      "HbA1c": 7.2, "TSH": 0.1, "Creatinine": 1.8
    }
  }'
```

---

## Retraining

Datasets are not committed to the repo. Download from Kaggle and place in `data/raw/`:

| File | Kaggle Dataset |
|---|---|
| `anemia.csv` | ehababoelnaga/anemia-types-classification |
| `diabetes.csv` | iammustafatz/diabetes-prediction-dataset |
| `liver.csv` | abhi8923shriv/liver-disease-patient-dataset |
| `kidney.csv` | mansoordaku/ckdisease |
| `thyroid.csv` | emmanuelfwerr/thyroid-disease-data |

```bash
# Retrain all
python ml/train_specialists.py

# Retrain one
python ml/train_specialists.py --only thyroid
```

---

## Deployment

### Backend → Render

1. Push repo to GitHub — models are committed (~7.5MB total, no LFS needed)
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect your GitHub repo — Render auto-detects `render.yaml`
4. Deploy — API live at `https://your-app.onrender.com`

### Frontend → Netlify

1. Go to [app.netlify.com/drop](https://app.netlify.com/drop) and drag `ui/index.html`
2. Update the `API_BASE` in `ui/index.html` to point to your Render URL:

```js
// ui/index.html — top of <script> block
const API_BASE = window.location.hostname === "localhost"
  ? "http://localhost:8000/api/v1"
  : "https://your-render-url.onrender.com/api/v1";  // ← replace this
```

---

## Tech Stack

- **ML:** XGBoost, scikit-learn (pipeline, imputer, scaler)
- **NLP:** Regex, spaCy NER, pdfplumber, pytesseract
- **API:** FastAPI, Pydantic v2, uvicorn
- **Frontend:** Vanilla HTML/CSS/JS — no framework, no build step

---

## Medical Disclaimer

This system is for informational and research purposes only. It is **not** a substitute for professional medical diagnosis, advice, or treatment. Always consult a licensed healthcare provider.
