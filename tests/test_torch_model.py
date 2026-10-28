"""Tests that need PyTorch + transformers. Everything runs offline on a tiny random BERT."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from tokenizers import Tokenizer, models, pre_tokenizers, processors  # noqa: E402
from transformers import BertConfig, BertModel, PreTrainedTokenizerFast  # noqa: E402

from viabsa.constants import ASPECTS, N_ASPECTS, N_LABELS  # noqa: E402
from viabsa.data.load import label_matrix  # noqa: E402
from viabsa.explain import integrated_gradients  # noqa: E402
from viabsa.inference import load_predictor  # noqa: E402
from viabsa.models.multihead import (  # noqa: E402
    MultiHeadAbsa,
    load_model,
    predict_proba_texts,
    save_model,
)
from viabsa.training.class_weights import compute_class_weights  # noqa: E402
from viabsa.training.train_transformer import build_optimizer, fit  # noqa: E402

pytestmark = pytest.mark.torch


def make_tokenizer(corpus: list[str]) -> PreTrainedTokenizerFast:
    specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"]
    words = sorted({w for text in corpus for w in text.lower().replace(",", " ").split()})
    vocab = {tok: i for i, tok in enumerate(specials + words)}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=[("[CLS]", vocab["[CLS]"]), ("[SEP]", vocab["[SEP]"])],
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
    )


def make_encoder(vocab_size: int) -> BertModel:
    torch.manual_seed(0)
    cfg = BertConfig(
        vocab_size=vocab_size,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
    )
    return BertModel(cfg)


@pytest.fixture(scope="module")
def data(workspace_module):
    import pandas as pd

    train = pd.read_parquet(workspace_module / "processed" / "train.parquet")
    val = pd.read_parquet(workspace_module / "processed" / "val.parquet")
    return train, val


@pytest.fixture(scope="module")
def workspace_module(workspace):
    return workspace


@pytest.fixture(scope="module")
def tokenizer(data):
    train, val = data
    return make_tokenizer(train["text_clean"].tolist() + val["text_clean"].tolist())


def test_forward_shapes_and_loss(tokenizer):
    model = MultiHeadAbsa(make_encoder(len(tokenizer)))
    enc = tokenizer(["son đẹp", "giao hàng chậm quá"], padding=True, return_tensors="pt")
    labels = torch.randint(0, N_LABELS, (2, N_ASPECTS))
    out = model(enc["input_ids"], enc["attention_mask"], labels)
    assert out["logits"].shape == (2, N_ASPECTS, N_LABELS)
    assert torch.isfinite(out["loss"]) and out["loss"].item() > 0


def test_class_weights_and_focal_change_the_loss(tokenizer):
    enc = tokenizer(["son đẹp", "giao hàng chậm"], padding=True, return_tensors="pt")
    labels = torch.tensor([[1] * N_ASPECTS, [3] * N_ASPECTS])
    enc_model = make_encoder(len(tokenizer))
    plain = MultiHeadAbsa(enc_model).eval()
    weighted = MultiHeadAbsa(enc_model, class_weights=np.full((N_ASPECTS, N_LABELS), 2.0)).eval()
    focal = MultiHeadAbsa(enc_model, focal_gamma=2.0).eval()
    for other in (weighted, focal):  # same classifier weights, so only the loss differs
        other.classifier.load_state_dict(plain.classifier.state_dict())
    args = (enc["input_ids"], enc["attention_mask"], labels)
    base = plain(*args)["loss"].item()
    assert weighted(*args)["loss"].item() == pytest.approx(2 * base, rel=1e-4)
    assert focal(*args)["loss"].item() < base


def test_mean_pooling_ignores_padding(tokenizer):
    model = MultiHeadAbsa(make_encoder(len(tokenizer)), pooling="mean").eval()
    single = tokenizer(["son đẹp"], return_tensors="pt")
    padded = tokenizer(["son đẹp", "giao hàng chậm quá nhanh"], padding=True, return_tensors="pt")
    a = model(single["input_ids"], single["attention_mask"])["logits"][0]
    b = model(padded["input_ids"], padded["attention_mask"])["logits"][0]
    assert torch.allclose(a, b, atol=1e-4)


def test_optimizer_has_head_lr_and_no_decay_groups(tokenizer):
    model = MultiHeadAbsa(make_encoder(len(tokenizer)))
    opt = build_optimizer(model, lr=1e-4, head_mult=10, weight_decay=0.01)
    lrs = {g["lr"] for g in opt.param_groups}
    assert lrs == {1e-4, 1e-3} and any(g["weight_decay"] == 0.0 for g in opt.param_groups)


def test_fit_learns_save_load_and_explain(data, tokenizer, tmp_path):
    train, val = data
    cfg = {
        "model": {"max_length": 32},
        "training": {
            "epochs": 8,
            "batch_size": 32,
            "lr": 2e-3,
            "head_lr_multiplier": 5,
            "weight_decay": 0.0,
            "warmup_ratio": 0.1,
            "max_grad_norm": 1.0,
            "fp16": False,
            "patience": 8,
        },
    }
    weights = compute_class_weights(label_matrix(train), 0.5, 10.0)
    model = MultiHeadAbsa(make_encoder(len(tokenizer)), class_weights=weights)
    texts_tr, texts_va = train["text_clean"].tolist(), val["text_clean"].tolist()
    result = fit(
        model,
        tokenizer,
        texts_tr,
        label_matrix(train),
        texts_va,
        label_matrix(val),
        cfg,
        torch.device("cpu"),
    )
    losses = [h["train_loss"] for h in result["history"]]
    assert losses[-1] < losses[0] * 0.8
    assert result["best_val_mean_macro_f1"] > 0.3

    # save -> load -> identical probabilities
    proba = predict_proba_texts(model, tokenizer, texts_va[:20], 32)
    meta = {
        "backbone": "tiny-random-bert",
        "max_length": 32,
        "word_segment": False,
        "cleaning": {},
        "name": "tiny",
    }
    save_model(model, tokenizer, tmp_path / "m", meta)
    loaded, tok2, _ = load_model(tmp_path / "m")
    proba2 = predict_proba_texts(loaded, tok2, texts_va[:20], 32)
    assert np.allclose(proba, proba2, atol=1e-5)

    # explain through the public predictor interface
    predictor = load_predictor(tmp_path / "m")
    assert predictor.supports_explain
    exp = predictor.explain("son mịn giá rẻ", "price")
    assert exp["target_label"] in {"none", "negative", "neutral", "positive"}
    assert len(exp["words"]) == 4 and all(np.isfinite(s) for _, s in exp["words"])


def test_integrated_gradients_completeness(tokenizer):
    """Sum of attributions ~ f(x) - f(baseline) (the completeness axiom), up to Riemann error."""
    model = MultiHeadAbsa(make_encoder(len(tokenizer))).eval()
    text = "son mịn giá rẻ"
    aspect = ASPECTS.index("price")
    out = integrated_gradients(model, tokenizer, text, aspect, steps=200, max_length=32)
    enc = tokenizer(text, return_tensors="pt")
    emb = model.encoder.get_input_embeddings()(enc["input_ids"])
    with torch.no_grad():
        p_x = torch.softmax(model(enc["input_ids"], enc["attention_mask"])["logits"][0, aspect], -1)
        p_0 = torch.softmax(
            model(inputs_embeds=torch.zeros_like(emb), attention_mask=enc["attention_mask"])[
                "logits"
            ][0, aspect],
            -1,
        )
    expected = float(p_x[out["target_class"]] - p_0[out["target_class"]])
    assert sum(out["token_scores"]) == pytest.approx(expected, abs=0.02)
    assert len(out["tokens"]) == len(out["token_scores"])
