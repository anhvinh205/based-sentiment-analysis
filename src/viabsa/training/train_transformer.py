"""Fine-tune a multi-head transformer (PhoBERT / XLM-R / ViSoBERT) for aspect sentiment.

Usage::

    python -m viabsa.training.train_transformer --config configs/transformer_phobert.yaml
    python -m viabsa.training.train_transformer --config configs/transformer_xlmr.yaml \
        --set training.epochs=3 training.batch_size=16
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from viabsa.config import load_config
from viabsa.data.dataset import ReviewDataset, make_collate_fn
from viabsa.data.load import label_matrix
from viabsa.data.preprocess import Preprocessor
from viabsa.evaluation.metrics import compute_metrics, metrics_table
from viabsa.evaluation.report import write_report
from viabsa.models.multihead import MultiHeadAbsa, predict_proba_texts, save_model
from viabsa.tracking import Tracker
from viabsa.training.class_weights import compute_class_weights

logger = logging.getLogger(__name__)
TEXT = "text_clean"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_optimizer(model: MultiHeadAbsa, lr: float, head_mult: float, weight_decay: float):
    """AdamW with a higher LR for the new heads and no weight decay on biases / LayerNorm."""
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    groups: list[dict[str, Any]] = []
    for part, part_lr in ((model.encoder, lr), (model.classifier, lr * head_mult)):
        named = list(part.named_parameters())
        groups.append(
            {
                "params": [p for n, p in named if not any(k in n for k in no_decay)],
                "lr": part_lr,
                "weight_decay": weight_decay,
            }
        )
        groups.append(
            {
                "params": [p for n, p in named if any(k in n for k in no_decay)],
                "lr": part_lr,
                "weight_decay": 0.0,
            }
        )
    return torch.optim.AdamW([g for g in groups if g["params"]])


def fit(
    model: MultiHeadAbsa,
    tokenizer: Any,
    train_texts: list[str],
    y_train: np.ndarray,
    val_texts: list[str],
    y_val: np.ndarray,
    cfg: dict[str, Any],
    device: torch.device,
    tracker: Tracker | None = None,
) -> dict[str, Any]:
    """Train with warm-up + linear decay, AMP, grad clipping and early stopping on val macro-F1."""
    from transformers import get_linear_schedule_with_warmup

    tcfg, max_len = cfg["training"], cfg["model"]["max_length"]
    train_ds = ReviewDataset(train_texts, y_train, tokenizer, max_len)
    loader = DataLoader(
        train_ds, batch_size=tcfg["batch_size"], shuffle=True, collate_fn=make_collate_fn(tokenizer)
    )
    optimizer = build_optimizer(model, tcfg["lr"], tcfg["head_lr_multiplier"], tcfg["weight_decay"])
    total_steps = len(loader) * tcfg["epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * tcfg["warmup_ratio"]), total_steps
    )
    use_amp = bool(tcfg.get("fp16")) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_score, best_state, bad_epochs, history = -1.0, None, 0, []
    model.to(device)
    for epoch in range(1, tcfg["epochs"] + 1):
        model.train()
        running, start = 0.0, time.time()
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss = model(**batch)["loss"]
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["max_grad_norm"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.detach().item()

        proba = predict_proba_texts(model, tokenizer, val_texts, max_len, device=device)
        val = compute_metrics(y_val, proba.argmax(-1), proba)
        row = {
            "epoch": epoch,
            "train_loss": running / len(loader),
            "val_mean_macro_f1": val["mean_macro_f1"],
            "val_mean_ece": val["mean_ece"],
            "seconds": round(time.time() - start, 1),
        }
        history.append(row)
        logger.info("epoch %(epoch)d | loss %(train_loss).4f | val F1 %(val_mean_macro_f1).4f", row)
        if tracker:
            tracker.metrics({k: v for k, v in row.items() if k != "epoch"}, step=epoch)

        if val["mean_macro_f1"] > best_score:
            best_score, bad_epochs = val["mean_macro_f1"], 0
            best_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
        else:
            bad_epochs += 1
            if bad_epochs >= tcfg["patience"]:
                logger.info("early stopping at epoch %d", epoch)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {"best_val_mean_macro_f1": best_score, "history": history}


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    from transformers import AutoModel, AutoTokenizer

    set_seed(cfg.get("seed", 42))
    device = pick_device()
    data_dir = Path(cfg["data_dir"])
    train, val, test = (
        pd.read_parquet(data_dir / f"{s}.parquet") for s in ("train", "val", "test")
    )
    stats = json.loads((data_dir / "data_stats.json").read_text())

    mcfg, lcfg = cfg["model"], cfg["loss"]
    pre = Preprocessor(stats["cleaning"], word_segment=mcfg["word_segment"])
    texts = {
        n: pre(df[TEXT].tolist(), already_clean=True)
        for n, df in (("train", train), ("val", val), ("test", test))
    }

    tokenizer = AutoTokenizer.from_pretrained(mcfg["backbone"])
    encoder = AutoModel.from_pretrained(mcfg["backbone"])
    weights = compute_class_weights(
        label_matrix(train), lcfg["class_weight_power"], lcfg["class_weight_max"]
    )
    model = MultiHeadAbsa(
        encoder,
        pooling=mcfg["pooling"],
        dropout=mcfg["dropout"],
        class_weights=weights,
        focal_gamma=lcfg["focal_gamma"],
    )

    tracker = Tracker(cfg["tracking"]["mlflow"], cfg["tracking"]["experiment"], cfg["name"])
    tracker.params(
        {
            **{f"model.{k}": v for k, v in mcfg.items()},
            **{f"train.{k}": v for k, v in cfg["training"].items()},
            **{f"loss.{k}": v for k, v in lcfg.items()},
            "device": device.type,
        }
    )

    result = fit(
        model,
        tokenizer,
        texts["train"],
        label_matrix(train),
        texts["val"],
        label_matrix(val),
        cfg,
        device,
        tracker,
    )

    proba = predict_proba_texts(model, tokenizer, texts["test"], mcfg["max_length"], device=device)
    metrics = write_report(
        cfg["name"],
        test,
        proba,
        cfg["reports_dir"],
        {
            "best_val_mean_macro_f1": result["best_val_mean_macro_f1"],
            "history": result["history"],
            "backbone": mcfg["backbone"],
        },
    )
    tracker.metrics(
        {"test_mean_macro_f1": metrics["mean_macro_f1"], "test_mean_ece": metrics["mean_ece"]}
    )
    save_model(
        model.cpu(),
        tokenizer,
        cfg["out_dir"],
        {
            "backbone": mcfg["backbone"],
            "max_length": mcfg["max_length"],
            "word_segment": mcfg["word_segment"],
            "cleaning": stats["cleaning"],
            "name": cfg["name"],
        },
    )
    tracker.artifact(Path(cfg["reports_dir"]) / f"metrics_{cfg['name']}.json")
    tracker.end()
    logger.info("test mean macro-F1 = %.4f", metrics["mean_macro_f1"])
    print(metrics_table(metrics))
    return metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", nargs="*", default=[], dest="overrides")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run(load_config(args.config, args.overrides))


if __name__ == "__main__":
    main()
