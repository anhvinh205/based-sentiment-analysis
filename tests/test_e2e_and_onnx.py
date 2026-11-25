"""End-to-end training run and ONNX export, using a tiny random backbone saved to disk."""

from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from viabsa.constants import ASPECTS, LABELS  # noqa: E402
from viabsa.inference import load_predictor  # noqa: E402
from viabsa.training.train_transformer import run  # noqa: E402

from .test_torch_model import make_encoder, make_tokenizer  # noqa: E402

pytestmark = pytest.mark.torch


@pytest.fixture(scope="module")
def trained(workspace, tmp_path_factory):
    import pandas as pd

    root = tmp_path_factory.mktemp("e2e")
    corpus = pd.concat(
        [
            pd.read_parquet(workspace / "processed" / f"{s}.parquet")
            for s in ("train", "val", "test")
        ]
    )["text_clean"].tolist()
    tok = make_tokenizer(corpus)
    backbone = root / "tiny-backbone"
    make_encoder(len(tok)).save_pretrained(backbone)
    tok.save_pretrained(backbone)

    cfg = {
        "name": "tiny",
        "data_dir": str(workspace / "processed"),
        "out_dir": str(root / "model"),
        "reports_dir": str(root / "reports"),
        "seed": 0,
        "model": {
            "backbone": str(backbone),
            "pooling": "cls",
            "dropout": 0.0,
            "max_length": 32,
            "word_segment": False,
        },
        "training": {
            "epochs": 4,
            "batch_size": 32,
            "lr": 2e-3,
            "head_lr_multiplier": 5,
            "weight_decay": 0.0,
            "warmup_ratio": 0.1,
            "max_grad_norm": 1.0,
            "fp16": False,
            "patience": 4,
        },
        "loss": {"class_weight_power": 0.5, "class_weight_max": 10.0, "focal_gamma": 0.0},
        "tracking": {"mlflow": False, "experiment": "test"},
    }
    metrics = run(cfg)
    return root, metrics


def test_run_writes_all_artifacts(trained):
    root, metrics = trained
    assert (root / "reports" / "metrics_tiny.json").exists()
    assert (root / "reports" / "figures" / "confusion_tiny.png").exists()
    assert (root / "reports" / "errors_tiny.csv").exists()
    assert (root / "model" / "classifier.pt").exists()
    assert json.loads((root / "model" / "meta.json").read_text())["type"] == "transformer"
    assert 0.0 <= metrics["mean_macro_f1"] <= 1.0 and "mean_ece" in metrics
    assert "template_reviews" in metrics["slices"] or "free_text_reviews" in metrics["slices"]


def test_trained_model_serves_through_predictor(trained):
    root, _ = trained
    out = load_predictor(root / "model").predict(["son mịn giá rẻ"])[0]
    assert set(out) == set(ASPECTS) and out["price"]["label"] in LABELS


def test_onnx_export_feed_ignores_extra_tokenizer_inputs():
    from viabsa.export_onnx import _onnx_feed

    from .test_torch_model import make_tokenizer

    tokenizer = make_tokenizer(["son đẹp giao hàng nhanh"])
    sample = tokenizer(["son đẹp giao hàng nhanh"], return_tensors="pt", return_token_type_ids=True)

    assert "token_type_ids" in sample
    feed = _onnx_feed(sample)
    assert set(feed) == {"input_ids", "attention_mask"}
    assert all(value.dtype == np.int64 for value in feed.values())


def test_export_rejects_output_overlapping_source_before_loading(tmp_path, monkeypatch):
    from viabsa import export_onnx

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    sentinel = model_dir / "classifier.pt"
    sentinel.write_text("model")
    monkeypatch.setattr(
        export_onnx,
        "load_model",
        lambda _: pytest.fail("must reject overlapping paths before loading"),
    )

    with pytest.raises(ValueError, match="must not overlap"):
        export_onnx.export(model_dir, model_dir)

    assert sentinel.read_text() == "model"


def test_export_rejects_nonempty_output_directory(tmp_path, monkeypatch):
    from viabsa import export_onnx

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    existing = out_dir / "keep.txt"
    existing.write_text("keep")
    monkeypatch.setattr(
        export_onnx,
        "load_model",
        lambda _: pytest.fail("must reject existing output before loading"),
    )

    with pytest.raises(FileExistsError, match="must be empty"):
        export_onnx.export(model_dir, out_dir)

    assert existing.read_text() == "keep"


def test_onnx_export_matches_pytorch(trained, tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from viabsa.export_onnx import export

    root, _ = trained
    info = export(root / "model", tmp_path / "fp32", quantize=False)
    assert info["max_logit_diff"] < 1e-3

    texts = ["son mịn giá rẻ", "giao hàng chậm quá, màu xấu", "thơm lắm"]
    ref = load_predictor(root / "model").predict_proba(texts)
    onnx_pred = load_predictor(tmp_path / "fp32")
    assert onnx_pred.kind == "onnx"
    assert np.allclose(onnx_pred.predict_proba(texts), ref, atol=1e-4)


def test_quantised_onnx_is_smaller_and_valid(trained, tmp_path):
    pytest.importorskip("onnx")
    from viabsa.export_onnx import export

    root, _ = trained
    info = export(root / "model", tmp_path / "q", quantize=True)
    assert info["sizes_mb"]["model.quant.onnx"] < info["sizes_mb"]["model.onnx"]
    proba = load_predictor(tmp_path / "q").predict_proba(["son mịn giá rẻ"])
    assert np.allclose(proba.sum(-1), 1.0, atol=1e-4)
