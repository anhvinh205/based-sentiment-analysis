"""Shared transformer encoder + one 4-way classification head per aspect."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from viabsa.constants import N_ASPECTS, N_LABELS


class MultiHeadAbsa(nn.Module):
    """Encoder -> pooled vector -> linear layer producing ``n_aspects x n_labels`` logits."""

    class_weights: torch.Tensor

    def __init__(
        self,
        encoder: Any,
        n_aspects: int = N_ASPECTS,
        n_labels: int = N_LABELS,
        pooling: str = "cls",
        dropout: float = 0.1,
        class_weights: np.ndarray | torch.Tensor | None = None,
        focal_gamma: float = 0.0,
    ) -> None:
        super().__init__()
        if pooling not in {"cls", "mean"}:
            raise ValueError("pooling must be 'cls' or 'mean'")
        self.encoder = encoder
        self.n_aspects, self.n_labels = n_aspects, n_labels
        self.pooling, self.focal_gamma = pooling, focal_gamma
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(encoder.config.hidden_size, n_aspects * n_labels)
        weights = torch.ones(n_aspects, n_labels)
        if class_weights is not None:
            weights = torch.as_tensor(class_weights, dtype=torch.float32)
        self.register_buffer("class_weights", weights, persistent=False)

    def pool(self, hidden: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        if self.pooling == "cls" or attention_mask is None:
            return hidden[:, 0]
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1.0)

    def loss_fn(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Class-weighted (optionally focal) cross-entropy averaged over reviews and aspects."""
        log_p = F.log_softmax(logits.float(), dim=-1)
        log_p_true = log_p.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # (B, A)
        loss = -log_p_true
        if self.focal_gamma > 0:
            loss = loss * (1.0 - log_p_true.exp()).pow(self.focal_gamma)
        aspect_idx = torch.arange(self.n_aspects, device=labels.device)
        loss = loss * self.class_weights.to(labels.device)[aspect_idx, labels]
        return loss.mean()

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        out = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds
        )
        pooled = self.pool(out.last_hidden_state, attention_mask)
        logits = self.classifier(self.dropout(pooled)).view(-1, self.n_aspects, self.n_labels)
        loss = self.loss_fn(logits, labels) if labels is not None else None
        return {"logits": logits, "loss": loss}


@torch.no_grad()
def predict_proba_texts(
    model: MultiHeadAbsa,
    tokenizer: Any,
    texts: list[str],
    max_length: int = 128,
    batch_size: int = 64,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """Softmax probabilities, shape ``(n, n_aspects, n_labels)``; batches are length-sorted."""
    model.eval()
    order = np.argsort([len(t) for t in texts])
    out = np.zeros((len(texts), model.n_aspects, model.n_labels), dtype="float32")
    for start in range(0, len(texts), batch_size):
        idx = order[start : start + batch_size]
        enc = tokenizer(
            [texts[i] for i in idx],
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        ).to(device)
        logits = model(enc["input_ids"], enc["attention_mask"])["logits"]
        out[idx] = torch.softmax(logits.float(), dim=-1).cpu().numpy()
    return out


def save_model(model: MultiHeadAbsa, tokenizer: Any, out_dir: str | Path, meta: dict) -> None:
    """Persist encoder, classification layer, tokenizer and metadata in one directory."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(out / "encoder")
    tokenizer.save_pretrained(out / "tokenizer")
    torch.save(model.classifier.state_dict(), out / "classifier.pt")
    meta = {**meta, "type": "transformer", "pooling": model.pooling}
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))


def load_model(path: str | Path, device: str = "cpu") -> tuple[MultiHeadAbsa, Any, dict]:
    from transformers import AutoModel, AutoTokenizer

    path = Path(path)
    meta = json.loads((path / "meta.json").read_text())
    encoder = AutoModel.from_pretrained(path / "encoder")
    tokenizer = AutoTokenizer.from_pretrained(path / "tokenizer")
    model = MultiHeadAbsa(encoder, pooling=meta.get("pooling", "cls"))
    model.classifier.load_state_dict(torch.load(path / "classifier.pt", map_location="cpu"))
    return model.to(device).eval(), tokenizer, meta
