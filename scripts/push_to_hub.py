"""Upload a trained model directory (and MODEL_CARD.md) to the Hugging Face Hub.

Usage::

    huggingface-cli login
    python scripts/push_to_hub.py --model models/phobert_onnx --repo YOUR_USER/viabsa-phobert-onnx

Then point the demo at it with ``VIABSA_MODEL_PATH=YOUR_USER/viabsa-phobert-onnx``.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--repo", required=True, help="e.g. your-username/viabsa-phobert")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    api = HfApi()
    api.create_repo(args.repo, exist_ok=True, private=args.private)
    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp) / "model"
        shutil.copytree(args.model, staging)
        card = Path(__file__).resolve().parent.parent / "MODEL_CARD.md"
        if card.exists():
            shutil.copy(card, staging / "README.md")
        api.upload_folder(folder_path=str(staging), repo_id=args.repo)
    print(f"Uploaded to https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
