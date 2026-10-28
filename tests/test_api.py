import pytest
from fastapi.testclient import TestClient

from viabsa.constants import ASPECTS
from viabsa.inference import load_predictor
from viabsa.serving.api import ModelInfo, PredictResponse, create_app


@pytest.fixture()
def client(baseline_dir):
    with TestClient(create_app(load_predictor(baseline_dir))) as c:
        yield c


def test_model_type_schemas_disable_conflicting_protected_namespace():
    assert PredictResponse.model_config["protected_namespaces"] == ()
    assert ModelInfo.model_config["protected_namespaces"] == ()


def test_health_and_model_info(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}
    info = client.get("/model-info").json()
    assert info["model_type"] == "baseline" and info["aspects"] == ASPECTS
    assert info["supports_explain"] is False


def test_predict_returns_all_aspects(client):
    r = client.post("/predict", json={"texts": ["son mịn", "giao hàng chậm quá"]})
    assert r.status_code == 200
    body = r.json()
    assert len(body["results"]) == 2 and set(body["results"][0]) == set(ASPECTS)
    assert body["latency_ms"] >= 0


@pytest.mark.parametrize(
    "payload", [{"texts": []}, {"texts": [""]}, {"texts": ["x"] * 65}, {"texts": ["x" * 2001]}, {}]
)
def test_predict_validates_input(client, payload):
    assert client.post("/predict", json=payload).status_code == 422


def test_explain_unsupported_and_invalid_aspect(client):
    assert client.post("/explain", json={"text": "son đẹp", "aspect": "colour"}).status_code == 501
    assert client.post("/explain", json={"text": "son đẹp", "aspect": "nope"}).status_code == 422
