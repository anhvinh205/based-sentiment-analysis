from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from viabsa.constants import ASPECTS, N_LABELS
from viabsa.data.load import label_matrix
from viabsa.inference import BaselinePredictor, load_predictor, resolve_path
from viabsa.models.baseline import BaselineModel


def test_missing_relative_model_path_is_not_treated_as_hub_repo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "models").mkdir()
    hub = ModuleType("huggingface_hub")
    hub.snapshot_download = lambda _: pytest.fail("local paths must not call the Hub")

    with (
        patch.dict("sys.modules", {"huggingface_hub": hub}),
        pytest.raises(FileNotFoundError, match="models/phobert"),
    ):
        resolve_path("models/phobert")


def test_missing_absolute_model_path_is_not_treated_as_hub_repo(tmp_path):
    path = tmp_path / "models" / "phobert"
    hub = ModuleType("huggingface_hub")
    hub.snapshot_download = lambda _: pytest.fail("local paths must not call the Hub")

    with (
        patch.dict("sys.modules", {"huggingface_hub": hub}),
        pytest.raises(FileNotFoundError, match="phobert"),
    ):
        resolve_path(path)


def test_hugging_face_repo_id_is_downloaded():
    hub = ModuleType("huggingface_hub")
    download = Mock(return_value="C:/cache/model")
    hub.snapshot_download = download
    with patch.dict("sys.modules", {"huggingface_hub": hub}):
        assert resolve_path("owner/model") == Path("C:/cache/model")
    download.assert_called_once_with("owner/model")


def test_probabilities_are_valid(workspace):
    model = BaselineModel.load(workspace / "baseline")
    proba = model.predict_proba(["son mịn giá rẻ", "giao hàng chậm"])
    assert proba.shape == (2, len(ASPECTS), N_LABELS)
    assert np.allclose(proba.sum(-1), 1.0)


def test_baseline_learns_the_synthetic_task(workspace):
    test = pd.read_parquet(workspace / "processed" / "test.parquet")
    model = BaselineModel.load(workspace / "baseline")
    pred = model.predict(test["text_clean"].tolist())
    accuracy = (pred == label_matrix(test)).mean()
    assert accuracy > 0.9


def test_predictor_uses_same_preprocessing_as_training(baseline_dir):
    predictor = load_predictor(baseline_dir)
    assert isinstance(predictor, BaselinePredictor)
    messy = predictor.predict(["SON  mịnnnn,  ko khô"])[0]
    clean = predictor.predict(["son mịn, không khô"])[0]
    assert messy["texture"]["label"] == clean["texture"]["label"]


def test_predict_output_schema(baseline_dir):
    out = load_predictor(baseline_dir).predict(["màu rất đẹp"])[0]
    assert set(out) == set(ASPECTS)
    item = out["colour"]
    assert item["label"] in {"none", "negative", "neutral", "positive"}
    assert abs(sum(item["probs"].values()) - 1.0) < 1e-6
    assert out["colour"]["label"] == "positive"
