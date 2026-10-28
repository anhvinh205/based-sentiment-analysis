"""Unified predictors (baseline / PyTorch / ONNX) behind one interface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from viabsa.constants import ASPECTS, ID2LABEL, LABELS
from viabsa.data.preprocess import Preprocessor


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


class Predictor:
    """Base class. Subclasses implement ``_predict_proba`` on already-preprocessed text."""

    kind = "base"
    supports_explain = False

    def __init__(self, meta: dict[str, Any]) -> None:
        self.meta = meta
        self.pre = Preprocessor.from_meta(meta)

    def _predict_proba(self, texts: list[str]) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Return ``(n, n_aspects, n_labels)`` probabilities for raw review strings."""
        return self._predict_proba(self.pre(texts))

    def predict(self, texts: list[str]) -> list[dict[str, dict[str, Any]]]:
        """Per review: ``{aspect: {label, confidence, probs}}``."""
        proba = self.predict_proba(texts)
        results = []
        for row in proba:
            item = {}
            for j, aspect in enumerate(ASPECTS):
                k = int(row[j].argmax())
                item[aspect] = {
                    "label": ID2LABEL[k],
                    "confidence": float(row[j, k]),
                    "probs": {name: float(row[j, i]) for i, name in enumerate(LABELS)},
                }
            results.append(item)
        return results

    def explain(self, text: str, aspect: str) -> dict[str, Any]:
        raise NotImplementedError(f"{self.kind} predictor does not support explanations")


class BaselinePredictor(Predictor):
    kind = "baseline"

    def __init__(self, path: Path, meta: dict[str, Any]) -> None:
        super().__init__(meta)
        from viabsa.models.baseline import BaselineModel

        self.model = BaselineModel.load(path)

    def _predict_proba(self, texts: list[str]) -> np.ndarray:
        return self.model.predict_proba(texts)


class TransformerPredictor(Predictor):
    kind = "transformer"
    supports_explain = True

    def __init__(self, path: Path, meta: dict[str, Any], device: str = "cpu") -> None:
        super().__init__(meta)
        from viabsa.models.multihead import load_model

        self.device = device
        self.model, self.tokenizer, _ = load_model(path, device)
        self.max_length = int(meta.get("max_length", 128))

    def _predict_proba(self, texts: list[str]) -> np.ndarray:
        from viabsa.models.multihead import predict_proba_texts

        return predict_proba_texts(
            self.model, self.tokenizer, texts, self.max_length, device=self.device
        )

    def explain(self, text: str, aspect: str, steps: int = 16) -> dict[str, Any]:
        from viabsa.explain import integrated_gradients

        prepared = self.pre([text])[0]
        out = integrated_gradients(
            self.model,
            self.tokenizer,
            prepared,
            ASPECTS.index(aspect),
            steps=steps,
            max_length=self.max_length,
            device=self.device,
        )
        out["target_label"] = ID2LABEL[out["target_class"]]
        return out


class OnnxPredictor(Predictor):
    kind = "onnx"

    def __init__(self, path: Path, meta: dict[str, Any]) -> None:
        super().__init__(meta)
        import onnxruntime as ort
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(path / "tokenizer")
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(path / meta["onnx_file"]), opts, providers=["CPUExecutionProvider"]
        )
        self.max_length = int(meta.get("max_length", 128))

    def _predict_proba(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        chunks = []
        for i in range(0, len(texts), batch_size):
            enc = self.tokenizer(
                texts[i : i + batch_size],
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors="np",
            )
            logits = self.session.run(
                ["logits"],
                {
                    "input_ids": enc["input_ids"].astype("int64"),
                    "attention_mask": enc["attention_mask"].astype("int64"),
                },
            )[0]
            chunks.append(softmax(logits))
        return np.concatenate(chunks) if chunks else np.zeros((0, len(ASPECTS), len(LABELS)))


def resolve_path(path: str | Path) -> Path:
    """Local directory, or a Hugging Face Hub repo id such as ``user/viabsa-phobert``."""
    p = Path(path)
    if p.exists():
        return p
    if p.is_absolute() or p.parent.is_dir():
        raise FileNotFoundError(f"Model path not found: {path}")
    if "/" in str(path) and not str(path).startswith((".", "/")):
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(str(path)))
    raise FileNotFoundError(f"Model path not found: {path}")


def load_predictor(path: str | Path, device: str = "cpu") -> Predictor:
    path = resolve_path(path)
    meta = json.loads((path / "meta.json").read_text())
    kind = meta.get("type")
    if kind == "baseline":
        return BaselinePredictor(path, meta)
    if kind == "transformer":
        return TransformerPredictor(path, meta, device)
    if kind == "onnx":
        return OnnxPredictor(path, meta)
    raise ValueError(f"Unknown model type {kind!r} in {path / 'meta.json'}")
