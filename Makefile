.PHONY: help install install-train data baseline train-phobert train-xlmr train-visobert \
        export-onnx benchmark api app test lint format typecheck docker-up clean

PY ?= python
MODEL ?= models/baseline

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-16s %s\n", $$1, $$2}'

install:  ## Install the package with serving, UI and dev tools
	$(PY) -m pip install -e ".[dev,serve,app,onnx]"

install-train:  ## Add PyTorch, MLflow, pyvi (needed for transformer training)
	$(PY) -m pip install -e ".[train]"

data:  ## Raw CSVs (data/raw) -> clean parquet splits + statistics
	$(PY) -m viabsa.data.prepare --config configs/data.yaml

baseline:  ## TF-IDF + logistic regression baseline
	$(PY) -m viabsa.training.train_baseline --config configs/baseline.yaml

train-phobert:  ## Fine-tune PhoBERT (GPU recommended)
	$(PY) -m viabsa.training.train_transformer --config configs/transformer_phobert.yaml

train-xlmr:  ## Fine-tune XLM-RoBERTa
	$(PY) -m viabsa.training.train_transformer --config configs/transformer_xlmr.yaml

train-visobert:  ## Fine-tune ViSoBERT
	$(PY) -m viabsa.training.train_transformer --config configs/transformer_visobert.yaml

export-onnx:  ## Export MODEL=models/phobert to int8 ONNX in MODEL_onnx
	$(PY) -m viabsa.export_onnx --model $(MODEL) --out $(MODEL)_onnx --quantize

benchmark:  ## Compare models, e.g. make benchmark MODELS="models/baseline models/phobert"
	$(PY) -m viabsa.benchmark --models $(MODELS)

api:  ## Run the FastAPI service on :8000
	VIABSA_MODEL_PATH=$(MODEL) uvicorn viabsa.serving.api:app --reload --port 8000

app:  ## Run the Streamlit demo on :8501
	VIABSA_MODEL_PATH=$(MODEL) streamlit run app/streamlit_app.py

test:  ## Run the test suite with coverage
	$(PY) -m pytest --cov=viabsa --cov-report=term-missing

lint:  ## Ruff lint + format check
	ruff check . && ruff format --check .

format:  ## Auto-format
	ruff check . --fix && ruff format .

typecheck:  ## mypy
	mypy src app

docker-up:  ## Build and start API + Streamlit (models/ is mounted read-only)
	docker compose up --build

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage mlruns
