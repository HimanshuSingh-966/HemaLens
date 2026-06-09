.PHONY: help install install-api install-ui test lint format train serve clean dev

# ── DEFAULT HELP ──────────────────────────────────────────
help:
	@echo "HemaLens — Blood Report Analysis | Available Commands"
	@echo ""
	@echo "Setup & Install:"
	@echo "  make install       Install all dependencies (Python + Node)"
	@echo "  make install-api   Install Python dependencies only"
	@echo "  make install-ui    Install Node.js dependencies only"
	@echo ""
	@echo "Testing & Quality:"
	@echo "  make test          Run all tests (extractor, inference)"
	@echo "  make test-extract  Run extractor tests only"
	@echo "  make test-infer    Run inference tests only"
	@echo "  make lint          Run linter (pylint/flake8)"
	@echo "  make format        Auto-format code (black, isort)"
	@echo ""
	@echo "Training & Models:"
	@echo "  make train         Train all specialist models"
	@echo "  make train-anemia  Train anemia specialist only"
	@echo "  make train-diabetes Train diabetes specialist only"
	@echo ""
	@echo "Running:"
	@echo "  make serve         Start FastAPI server (port 8000)"
	@echo "  make dev           Start both API and UI (concurrent)"
	@echo "  make serve-ui      Start React dev server (port 5173)"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean         Remove caches, artifacts, __pycache__"
	@echo "  make clean-models  Remove trained model artifacts"
	@echo "  make clean-all     Clean everything (cache + models)"
	@echo ""

# ── INSTALLATION ──────────────────────────────────────────
install: install-api install-ui
	@echo "✅ All dependencies installed."

install-api:
	@echo "📦 Installing Python dependencies..."
	pip install -r requirements.txt
	@echo "✅ Python dependencies installed."

install-ui:
	@echo "📦 Installing Node.js dependencies..."
	cd ui && npm install
	@echo "✅ Node.js dependencies installed."

# ── TESTING ───────────────────────────────────────────────
test: test-extract test-infer
	@echo "✅ All tests completed."

test-extract:
	@echo "🧪 Running extractor tests..."
	python -m pytest tests/test_extractor.py -v

test-infer:
	@echo "🧪 Running inference tests..."
	python -m pytest tests/test_inference.py -v

# ── LINTING & FORMATTING ──────────────────────────────────
lint:
	@echo "🔍 Running linter..."
	python -m pylint nlp/ api/ ml/ --disable=all --enable=E,F 2>/dev/null || true
	@echo "✅ Linting complete."

format:
	@echo "🎨 Auto-formatting code..."
	python -m black nlp/ api/ ml/ tests/ --line-length=100 2>/dev/null || echo "Black not installed; skipping."
	python -m isort nlp/ api/ ml/ tests/ 2>/dev/null || echo "isort not installed; skipping."
	@echo "✅ Formatting complete."

# ── TRAINING ──────────────────────────────────────────────
train:
	@echo "🧠 Training all specialist models..."
	python ml/train_specialists.py
	@echo "✅ Training complete. Models saved to models/specialists/"

train-anemia:
	@echo "🧠 Training anemia specialist..."
	python ml/train_specialists.py --specialist anemia

train-diabetes:
	@echo "🧠 Training diabetes specialist..."
	python ml/train_specialists.py --specialist diabetes

# ── RUNNING SERVICES ──────────────────────────────────────
serve:
	@echo "🚀 Starting FastAPI server on http://localhost:8000"
	uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

serve-ui:
	@echo "🚀 Starting React dev server on http://localhost:5173"
	cd ui && npm run dev

dev:
	@echo "🚀 Starting API and UI in parallel..."
	@echo "   API:  http://localhost:8000"
	@echo "   UI:   http://localhost:5173"
	@echo "   Press Ctrl+C to stop both."
	@(trap 'kill %1 %2 2>/dev/null' INT; \
	  make serve & \
	  make serve-ui & \
	  wait)

# ── CLEANUP ───────────────────────────────────────────────
clean:
	@echo "🗑️  Cleaning caches and artifacts..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name ".DS_Store" -delete
	rm -rf .egg-info dist build 2>/dev/null || true
	@echo "✅ Cleanup complete."

clean-models:
	@echo "🗑️  Removing trained models..."
	rm -rf models/specialists/* 2>/dev/null || true
	@echo "✅ Models removed."

clean-all: clean clean-models
	@echo "✅ Full cleanup complete."

# ── QUICK UTILITIES ───────────────────────────────────────
extract-text:
	@echo "Usage: make extract-text TEXT='<your lab report text>'"
	python -c "from nlp.extractor import extract_from_text; import json; print(json.dumps(extract_from_text('$(TEXT)'), indent=2))"

check-deps:
	@echo "🔍 Checking Python dependencies..."
	python -c "import sys; print(f'Python: {sys.version}')"
	pip list | grep -E "fastapi|uvicorn|joblib|scikit-learn|pandas"

.PHONY: help install install-api install-ui test test-extract test-infer \
        lint format train train-anemia train-diabetes serve serve-ui dev \
        clean clean-models clean-all extract-text check-deps
