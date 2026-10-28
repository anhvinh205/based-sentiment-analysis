"""Streamlit demo: ``streamlit run app/streamlit_app.py``.

Backends (first match wins):
    VIABSA_API_URL     talk to a running FastAPI service
    VIABSA_MODEL_PATH  model directory or Hugging Face repo id, loaded in-process
                       (default: models/baseline)
"""

from __future__ import annotations

import html
import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from viabsa.batch import select_review_rows
from viabsa.constants import ASPECT_DISPLAY, ASPECTS, LABELS

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = [
    "Son lên màu chuẩn, mịn nhưng hơi khô môi, giá rẻ nên mình ok. "
    "Shop giao nhanh, đóng gói cẩn thận.",
    "Màu không giống hình, son trôi nhanh, giao hàng chậm quá 😡",
    "Thơm nhẹ, lì mà không bị khô, lên môi rất xinh. Sẽ ủng hộ shop tiếp",
    "Son bình thường, không có gì đặc biệt.",
]
CHIP = {
    "positive": ("#1b5e20", "#e8f5e9"),
    "negative": ("#b71c1c", "#ffebee"),
    "neutral": ("#5d4037", "#fff8e1"),
}

st.set_page_config(page_title="ViABSA - Vietnamese ABSA", page_icon="💄", layout="wide")


@st.cache_resource(show_spinner="Loading model...")
def get_backend():
    api_url = os.environ.get("VIABSA_API_URL")
    if api_url:
        from viabsa.client import ApiClient

        return ApiClient(api_url)
    from viabsa.inference import load_predictor

    path = os.environ.get("VIABSA_MODEL_PATH", str(ROOT / "models" / "baseline"))
    return load_predictor(path, os.environ.get("VIABSA_DEVICE", "cpu"))


def chip(aspect: str, label: str, confidence: float) -> str:
    fg, bg = CHIP[label]
    name = ASPECT_DISPLAY[aspect]["vi"]
    return (
        f"<span style='background:{bg};color:{fg};padding:4px 10px;border-radius:14px;"
        f"margin:2px 6px 2px 0;display:inline-block;font-size:0.9rem'>"
        f"<b>{html.escape(name)}</b>: {label} <small>({confidence:.0%})</small></span>"
    )


def highlight(words: list[tuple[str, float]]) -> str:
    """Orange = pushes towards the predicted label, blue = pushes against it."""
    peak = max((abs(s) for _, s in words), default=0.0) or 1.0
    parts = []
    for word, score in words:
        alpha = min(abs(score) / peak, 1.0) * 0.75
        rgb = "255,152,0" if score >= 0 else "66,133,244"
        parts.append(
            f"<span style='background:rgba({rgb},{alpha:.2f});padding:2px 3px;border-radius:4px'>"
            f"{html.escape(word)}</span>"
        )
    return " ".join(parts)


def flatten(results: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {a: r[a]["label"] for a in ASPECTS} | {f"{a}_conf": r[a]["confidence"] for a in ASPECTS}
            for r in results
        ]
    )


def tab_analyze(backend) -> None:
    choice = st.selectbox("Try an example or write your own", ["(custom)", *EXAMPLES])
    text = st.text_area(
        "Review",
        value="" if choice == "(custom)" else choice,
        height=110,
        placeholder="Nhập review son môi bằng tiếng Việt...",
    )
    if st.button("Analyze", type="primary", disabled=not (text or "").strip()):
        st.session_state["analyzed"] = text
        st.session_state["result"] = backend.predict([text])[0]
    result, analyzed = st.session_state.get("result"), st.session_state.get("analyzed")
    if not result:
        return

    mentioned = {a: r for a, r in result.items() if r["label"] != "none"}
    st.subheader("Detected aspects")
    if mentioned:
        st.markdown(
            "".join(chip(a, r["label"], r["confidence"]) for a, r in mentioned.items()),
            unsafe_allow_html=True,
        )
    else:
        st.info("No aspect detected in this review.")

    with st.expander("Class probabilities for all aspects"):
        table = pd.DataFrame({a: r["probs"] for a, r in result.items()}).T[LABELS]
        st.dataframe(table.style.format("{:.1%}").background_gradient(cmap="Blues", axis=None))

    if getattr(backend, "supports_explain", False):
        st.subheader("Why? (Integrated Gradients)")
        aspect = st.selectbox(
            "Explain aspect",
            ASPECTS,
            format_func=lambda a: ASPECT_DISPLAY[a]["en"],
            key="exp_aspect",
        )
        if st.button("Explain prediction"):
            with st.spinner("Computing attributions..."):
                exp = backend.explain(analyzed, aspect)
            st.caption(
                f"Target class: **{exp['target_label']}** "
                "- orange words support it, blue words oppose it"
            )
            st.markdown(highlight(exp["words"]), unsafe_allow_html=True)
    else:
        st.caption(
            "Word-level explanations need a transformer model; the loaded model "
            "is a TF-IDF baseline."
        )


def tab_batch(backend) -> None:
    upload = st.file_uploader("Upload a CSV with a review column", type="csv")
    if upload is None:
        st.caption("Tip: use data/raw/data_test.csv from the repo's dataset folder.")
        return
    try:
        df = pd.read_csv(upload)
    except pd.errors.EmptyDataError:
        st.error("The uploaded CSV is empty or has no header row.")
        return
    if df.empty:
        st.warning("The uploaded CSV has a header but no review rows.")
        return
    column = st.selectbox(
        "Text column",
        df.columns,
        index=list(df.columns).index("data") if "data" in df.columns else 0,
    )
    limit = st.slider("Rows to analyze", 1, min(5000, len(df)), min(500, len(df)))
    if st.button("Run batch analysis", type="primary"):
        selected, skipped = select_review_rows(df, column, limit)
        if skipped:
            st.info(f"Skipped {skipped} blank review row(s).")
        if selected.empty:
            st.warning("No non-blank reviews found in the selected rows.")
        else:
            texts = selected[column].astype(str).tolist()
            with st.spinner(f"Analyzing {len(texts)} reviews..."):
                out = flatten(backend.predict(texts))
            st.session_state["batch"] = pd.concat(
                [selected[[column]].reset_index(drop=True), out], axis=1
            )
    batch = st.session_state.get("batch")
    if batch is None:
        return
    counts = (
        batch[ASPECTS]
        .apply(lambda s: s.value_counts())
        .T.reindex(columns=LABELS)
        .fillna(0)
        .drop(columns="none")
    )
    st.subheader("Sentiment per aspect")
    st.bar_chart(counts, color=["#ef5350", "#ffca28", "#66bb6a"])
    st.dataframe(batch)
    st.download_button(
        "Download results (CSV)",
        batch.to_csv(index=False).encode("utf-8"),
        "viabsa_predictions.csv",
        "text/csv",
    )


def tab_insights() -> None:
    reports = ROOT / "reports"
    metrics = {
        p.stem.removeprefix("metrics_"): json.loads(p.read_text())
        for p in sorted(reports.glob("metrics_*.json"))
    }
    stats_path = reports / "data_stats.json"

    st.subheader("Model comparison (test set)")
    if metrics:
        summary = pd.DataFrame(
            {
                name: {
                    "mean macro-F1": m["mean_macro_f1"],
                    "detection F1": m["mean_detection_f1"],
                    "sentiment F1": m["mean_sentiment_f1"],
                    "ECE": m.get("mean_ece", float("nan")),
                }
                for name, m in metrics.items()
            }
        ).T
        st.dataframe(summary.style.format("{:.3f}"))
        pick = str(st.selectbox("Per-aspect detail", list(metrics)))
        detail = pd.DataFrame(
            {
                a: {
                    k: v
                    for k, v in m.items()
                    if k in ("macro_f1", "detection_f1", "sentiment_f1", "accuracy")
                }
                for a, m in metrics[pick]["per_aspect"].items()
            }
        ).T
        st.dataframe(detail.style.format("{:.3f}"))
        img = reports / "figures" / f"confusion_{pick}.png"
        if img.exists():
            st.image(str(img), caption="Confusion matrices (test)")
    else:
        st.info("No reports yet - run `make baseline` or `make train-phobert`.")

    st.subheader("Label distribution (train split)")
    if stats_path.exists():
        dist = json.loads(stats_path.read_text())["per_split"]["train"]["label_distribution"]
        st.bar_chart(
            pd.DataFrame(dist).T.reindex(columns=LABELS).fillna(0).drop(columns="none"),
            color=["#ef5350", "#ffca28", "#66bb6a"],
        )
        st.caption(
            "Negative and neutral classes are rare for price and packing, "
            "so those F1 scores rest on only a handful of test reviews."
        )
    else:
        st.info("Run `make data` to generate dataset statistics.")


def tab_about() -> None:
    st.markdown(
        """
**ViABSA** predicts, for each of 8 aspects of a Vietnamese lipstick review
(staying power, texture, smell, price, colour, shipping, packing, others),
whether it is *not mentioned*, *negative*, *neutral* or *positive*.

* **Models:** TF-IDF + logistic regression baseline; multi-head PhoBERT / XLM-R / ViSoBERT
* **Explainability:** Integrated Gradients highlight the words behind each prediction
* **Serving:** FastAPI + ONNX (int8) for CPU inference
* **Code & docs:** see the GitHub repository README and `MODEL_CARD.md`

*Limitations:* trained on lipstick reviews only; rare classes (negative price, neutral packing)
are poorly measured; sarcasm and mixed-language reviews are error-prone.
"""
    )


def main() -> None:
    st.title("💄 ViABSA - Vietnamese Aspect-Based Sentiment Analysis")
    try:
        backend = get_backend()
    except Exception as exc:
        st.error(f"Could not load a model: {exc}")
        st.code("make data && make baseline   # then reload this page")
        st.stop()
    with st.sidebar:
        st.markdown(f"**Model:** `{backend.kind}`")
        name = getattr(backend, "meta", {}).get("backbone")
        if name:
            st.caption(name)
    t1, t2, t3, t4 = st.tabs(["Analyze a review", "Batch (CSV)", "Data & model insights", "About"])
    with t1:
        tab_analyze(backend)
    with t2:
        tab_batch(backend)
    with t3:
        tab_insights()
    with t4:
        tab_about()


main()
