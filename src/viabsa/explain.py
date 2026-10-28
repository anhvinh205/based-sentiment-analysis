"""Integrated Gradients over input embeddings: which words drive an aspect's prediction?"""

from __future__ import annotations

from typing import Any

import torch

from viabsa.explain_utils import merge_subword_scores
from viabsa.models.multihead import MultiHeadAbsa


def integrated_gradients(
    model: MultiHeadAbsa,
    tokenizer: Any,
    text: str,
    aspect_idx: int,
    target_class: int | None = None,
    steps: int = 16,
    max_length: int = 128,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Attribute the probability of ``target_class`` for one aspect to the input words.

    Uses an all-zero embedding baseline and a Riemann sum with ``steps`` points, evaluated as one
    batch. ``text`` must already be preprocessed (cleaned + segmented) exactly as at inference.
    Positive scores push towards the target class, negative scores away from it.
    """
    model.eval().to(device)
    enc = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt").to(device)
    ids, mask = enc["input_ids"], enc["attention_mask"]
    embeddings = model.encoder.get_input_embeddings()

    with torch.no_grad():
        base_logits = model(ids, mask)["logits"][0, aspect_idx]
        probs = torch.softmax(base_logits, dim=-1)
        target = int(probs.argmax()) if target_class is None else target_class
        emb = embeddings(ids)  # (1, T, H)

    alphas = torch.linspace(1.0 / steps, 1.0, steps, device=device).view(-1, 1, 1)
    scaled = (alphas * emb).detach().requires_grad_(True)  # baseline is zero
    with torch.enable_grad():
        logits = model(inputs_embeds=scaled, attention_mask=mask.expand(steps, -1))["logits"]
        score = torch.softmax(logits[:, aspect_idx], dim=-1)[:, target].sum()
        grads = torch.autograd.grad(score, scaled)[0]

    token_scores = (emb[0] * grads.mean(dim=0)).sum(dim=-1)  # (T,)
    tokens = tokenizer.convert_ids_to_tokens(ids[0].tolist())
    words = merge_subword_scores(tokens, token_scores.tolist())
    return {
        "target_class": target,
        "target_prob": float(probs[target]),
        "words": words,
        "tokens": tokens,
        "token_scores": token_scores.tolist(),
    }
