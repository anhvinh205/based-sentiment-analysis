"""Torch dataset + dynamic-padding collator for the multi-aspect task."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class ReviewDataset(Dataset):
    def __init__(self, texts: list[str], labels: np.ndarray, tokenizer: Any, max_length: int):
        enc = tokenizer(list(texts), truncation=True, max_length=max_length)
        self.input_ids: list[list[int]] = enc["input_ids"]
        self.labels = np.asarray(labels, dtype="int64")

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, i: int) -> dict[str, Any]:
        return {"input_ids": self.input_ids[i], "labels": self.labels[i]}


def make_collate_fn(tokenizer: Any):
    """Pad each batch to its own longest sequence (much faster than padding to max_length)."""

    def collate(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        padded = tokenizer.pad([{"input_ids": b["input_ids"]} for b in batch], return_tensors="pt")
        padded["labels"] = torch.as_tensor(np.stack([b["labels"] for b in batch]))
        return dict(padded)

    return collate
