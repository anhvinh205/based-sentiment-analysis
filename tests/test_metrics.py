import numpy as np
import pandas as pd
import pytest

from viabsa.constants import ASPECTS, N_ASPECTS, N_LABELS
from viabsa.evaluation.metrics import compute_metrics, expected_calibration_error
from viabsa.evaluation.report import SLICES, slice_metrics
from viabsa.training.class_weights import compute_class_weights


def test_negated_slice_uses_the_named_column(monkeypatch):
    n = 30
    df = pd.DataFrame(
        {
            "is_template": [i < 4 for i in range(n)],
            "had_gibberish": [4 <= i < 12 for i in range(n)],
            **{aspect: np.zeros(n, dtype=int) for aspect in ASPECTS},
        }
    )
    proba = np.zeros((n, N_ASPECTS, N_LABELS))
    proba[:, :, 0] = 1.0
    monkeypatch.setitem(SLICES, "clean_reviews", "~had_gibberish")

    slices = slice_metrics(df, proba)

    assert slices["clean_reviews"]["n"] == 22


def test_perfect_predictions_score_one():
    rng = np.random.default_rng(0)
    y = rng.integers(0, N_LABELS, size=(50, N_ASPECTS))
    m = compute_metrics(y, y.copy())
    assert m["mean_macro_f1"] == pytest.approx(1.0)
    assert m["exact_match"] == 1.0


def test_known_macro_f1_value():
    y_true = np.zeros((4, N_ASPECTS), dtype=int)
    y_pred = np.zeros((4, N_ASPECTS), dtype=int)
    y_true[:, 0] = [1, 1, 3, 3]
    y_pred[:, 0] = [1, 3, 3, 3]
    m = compute_metrics(y_true, y_pred)["per_aspect"]["stayingpower"]
    # class 1: P=1, R=.5 -> F1=2/3 ; class 3: P=2/3, R=1 -> F1=.8 ; macro = (2/3 + .8) / 2
    assert m["macro_f1"] == pytest.approx((2 / 3 + 0.8) / 2)
    assert m["detection_f1"] == pytest.approx(1.0)
    assert m["sentiment_f1"] == pytest.approx((2 / 3 + 0.8) / 2)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        compute_metrics(np.zeros((3, N_ASPECTS), dtype=int), np.zeros((2, N_ASPECTS), dtype=int))


def test_ece_extremes():
    assert expected_calibration_error(np.array([1.0, 1.0]), np.array([1.0, 1.0])) == 0.0
    assert expected_calibration_error(np.array([1.0, 1.0]), np.array([1.0, 0.0])) == pytest.approx(
        0.5
    )


def test_class_weights_inverse_frequency_and_cap():
    y = np.zeros((100, N_ASPECTS), dtype=int)
    y[:10, 0] = 1  # 90 none / 10 negative
    w = compute_class_weights(y, power=1.0, max_weight=100.0)
    assert w[0, 0] == pytest.approx(100 / (2 * 90)) and w[0, 1] == pytest.approx(5.0)
    assert w[0, 2] == 1.0 and w[0, 3] == 1.0  # absent classes keep weight 1
    assert np.allclose(compute_class_weights(y, power=0.0)[0, :2], 1.0)
    y[0, 1], y[1:, 1] = 1, 0  # 99 none / 1 negative
    assert compute_class_weights(y, power=1.0, max_weight=10.0)[1, 1] == 10.0
