"""Classical baseline: TF-IDF (word + char n-grams) and one logistic regression per aspect."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from viabsa.constants import N_ASPECTS, N_LABELS


class BaselineModel:
    """Multi-aspect classifier with a scikit-learn interface (``fit`` / ``predict_proba``)."""

    kind = "baseline"

    def __init__(
        self,
        word_ngram: tuple[int, int] = (1, 2),
        word_max_features: int | None = 50_000,
        char_ngram: tuple[int, int] = (2, 5),
        char_max_features: int | None = 150_000,
        min_df: int = 2,
        c_values: list[float] | None = None,
        class_weight: str | None = "balanced",
        max_iter: int = 400,
        seed: int = 42,
    ) -> None:
        self.word_vec = TfidfVectorizer(
            ngram_range=tuple(word_ngram),
            max_features=word_max_features,
            min_df=min_df,
            sublinear_tf=True,
        )
        self.char_vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=tuple(char_ngram),
            max_features=char_max_features,
            min_df=min_df,
            sublinear_tf=True,
        )
        self.c_values = c_values or [1.0]
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.seed = seed
        self.classifiers: list[LogisticRegression] = []
        self.best_c: list[float] = []

    # ------------------------------------------------------------------ features
    def _features(self, texts: list[str], fit: bool = False):
        if fit:
            return hstack([self.word_vec.fit_transform(texts), self.char_vec.fit_transform(texts)])
        return hstack([self.word_vec.transform(texts), self.char_vec.transform(texts)]).tocsr()

    def _new_classifier(self, c: float) -> LogisticRegression:
        return LogisticRegression(
            C=c,
            class_weight=self.class_weight,
            max_iter=self.max_iter,
            random_state=self.seed,
        )

    # ----------------------------------------------------------------- training
    def fit(
        self,
        texts: list[str],
        y: np.ndarray,
        val_texts: list[str] | None = None,
        y_val: np.ndarray | None = None,
    ) -> BaselineModel:
        """Fit vectorizers and one classifier per aspect.

        When validation data is given, ``C`` is selected independently per aspect by macro-F1.
        """
        from viabsa.evaluation.metrics import _macro_f1_present

        x = self._features(texts, fit=True).tocsr()
        x_val = self._features(val_texts) if val_texts is not None else None
        self.classifiers, self.best_c = [], []
        for j in range(N_ASPECTS):
            best_clf, best_score, best_c = None, -1.0, self.c_values[0]
            for c in self.c_values:
                clf = self._new_classifier(c).fit(x, y[:, j])
                if x_val is None or y_val is None or len(self.c_values) == 1:
                    best_clf, best_c = clf, c
                    break
                score = _macro_f1_present(y_val[:, j], clf.predict(x_val), list(range(N_LABELS)))
                if score > best_score:
                    best_clf, best_score, best_c = clf, score, c
            self.classifiers.append(best_clf)
            self.best_c.append(best_c)
        return self

    # ---------------------------------------------------------------- inference
    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Return ``(n, n_aspects, n_labels)`` probabilities (missing classes get 0)."""
        x = self._features(texts)
        out = np.zeros((len(texts), N_ASPECTS, N_LABELS), dtype="float64")
        for j, clf in enumerate(self.classifiers):
            out[:, j, clf.classes_] = clf.predict_proba(x)
        return out

    def predict(self, texts: list[str]) -> np.ndarray:
        return self.predict_proba(texts).argmax(axis=-1)

    # -------------------------------------------------------------- persistence
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path / "baseline.joblib", compress=3)

    @classmethod
    def load(cls, path: str | Path) -> BaselineModel:
        model: Any = joblib.load(Path(path) / "baseline.joblib")
        return model
