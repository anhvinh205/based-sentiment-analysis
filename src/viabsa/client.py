"""Thin HTTP client exposing the same interface as a local ``Predictor`` (used by Streamlit)."""

from __future__ import annotations

from typing import Any

import httpx


class ApiClient:
    def __init__(self, base_url: str, timeout: float = 60.0) -> None:
        self.http = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)
        info = self.http.get("/model-info")
        info.raise_for_status()
        self.info: dict[str, Any] = info.json()
        self.kind: str = self.info["model_type"]
        self.supports_explain: bool = self.info["supports_explain"]
        self.meta: dict[str, Any] = self.info["meta"]

    def predict(self, texts: list[str], batch_size: int = 64) -> list[dict[str, dict[str, Any]]]:
        results: list[dict[str, dict[str, Any]]] = []
        for i in range(0, len(texts), batch_size):
            resp = self.http.post("/predict", json={"texts": texts[i : i + batch_size]})
            resp.raise_for_status()
            results.extend(resp.json()["results"])
        return results

    def explain(self, text: str, aspect: str) -> dict[str, Any]:
        resp = self.http.post("/explain", json={"text": text, "aspect": aspect})
        resp.raise_for_status()
        data = resp.json()
        return {
            "target_label": data["target_label"],
            "target_prob": data["target_prob"],
            "words": [(w["word"], w["score"]) for w in data["words"]],
        }
