---
language: vi
license: mit
tags: [sentiment-analysis, aspect-based-sentiment, vietnamese, text-classification]
---

# ViABSA model card

## Model
Shared transformer encoder (PhoBERT-base-v2, XLM-R base or ViSoBERT) with one linear layer producing
8 aspects × 4 classes (`none`, `negative`, `neutral`, `positive`). Trained with class-weighted
cross-entropy (optionally focal). A TF-IDF + logistic-regression baseline is provided for comparison.

## Intended use
Research and portfolio demonstration: summarising which aspects of Vietnamese lipstick reviews are
praised or criticised. **Not** intended for decisions about individuals, moderation or ranking sellers.

## Training data
See `DATA_STATEMENT.md` (Vietnamese lipstick reviews, ~12.4k training reviews after cleaning).

## Evaluation
Test set of 1,534 reviews; metrics and per-aspect tables are in `README.md` and
`reports/metrics_<model>.json`. Fill in the transformer results after training: `TODO`.

## Limitations and risks
* Trained on one product category; expect degradation on other products or platforms.
* Negative price and neutral packing reviews are almost absent from the data: predictions for those
  classes are unreliable and their test metrics rest on a handful of examples.
* Sarcasm, code-mixing and heavy abbreviation are error-prone.
* Word-level explanations (Integrated Gradients) are approximations, not causal proofs.

## How to cite / contact
`TODO: your name and contact`
