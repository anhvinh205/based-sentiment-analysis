"""Export a trained transformer to ONNX, optionally quantise to int8, and verify parity.

Usage::

    python -m viabsa.export_onnx --model models/phobert --out models/phobert_onnx --quantize
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch

from viabsa.models.multihead import MultiHeadAbsa, load_model

logger = logging.getLogger(__name__)
ONNX_INPUT_NAMES = ("input_ids", "attention_mask")


def _onnx_feed(sample: Mapping[str, torch.Tensor]) -> dict[str, np.ndarray]:
    return {name: sample[name].numpy().astype("int64") for name in ONNX_INPUT_NAMES}


def _validate_export_paths(model_dir: str | Path, out_dir: str | Path) -> tuple[Path, Path]:
    model_path = Path(model_dir).resolve()
    out_path = Path(out_dir).resolve()
    if model_path == out_path or model_path in out_path.parents or out_path in model_path.parents:
        raise ValueError("ONNX output directory must not overlap the source model directory")
    if out_path.exists() and (not out_path.is_dir() or any(out_path.iterdir())):
        raise FileExistsError(f"ONNX output directory must be empty or not exist: {out_path}")
    return model_path, out_path


class _LogitsOnly(torch.nn.Module):
    def __init__(self, model: MultiHeadAbsa) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids, attention_mask)["logits"]


def export(model_dir: str | Path, out_dir: str | Path, quantize: bool = True, opset: int = 17):
    model_path, out = _validate_export_paths(model_dir, out_dir)
    model, tokenizer, meta = load_model(model_path)
    out.mkdir(parents=True, exist_ok=True)
    fp32_path = out / "model.onnx"

    sample = tokenizer(
        ["son đẹp giao hàng nhanh", "màu lên chuẩn"], padding=True, return_tensors="pt"
    )
    wrapper = _LogitsOnly(model).eval()
    torch.onnx.export(
        wrapper,
        (sample["input_ids"], sample["attention_mask"]),
        str(fp32_path),
        input_names=list(ONNX_INPUT_NAMES),
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits": {0: "batch"},
        },
        opset_version=opset,
        dynamo=False,
    )
    onnx_file = "model.onnx"
    if quantize:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        quantize_dynamic(str(fp32_path), str(out / "model.quant.onnx"), weight_type=QuantType.QInt8)
        onnx_file = "model.quant.onnx"

    tokenizer.save_pretrained(out / "tokenizer")
    new_meta = {**meta, "type": "onnx", "onnx_file": onnx_file, "source_model": str(model_path)}
    (out / "meta.json").write_text(json.dumps(new_meta, indent=2, ensure_ascii=False))

    # Parity check against PyTorch on the sample batch.
    import onnxruntime as ort

    with torch.no_grad():
        ref = wrapper(sample["input_ids"], sample["attention_mask"]).numpy()
    sess = ort.InferenceSession(str(out / onnx_file), providers=["CPUExecutionProvider"])
    got = sess.run(["logits"], _onnx_feed(sample))[0]
    diff = float(np.abs(ref - got).max())
    sizes = {p.name: round(p.stat().st_size / 1e6, 3) for p in out.glob("*.onnx")}
    logger.info("max |logit diff| vs PyTorch: %.4f; sizes (MB): %s", diff, sizes)
    return {"max_logit_diff": diff, "sizes_mb": sizes, "onnx_file": onnx_file}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--quantize", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    export(args.model, args.out, args.quantize)


if __name__ == "__main__":
    main()
