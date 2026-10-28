"""Optional MLflow tracking: every call is a no-op when MLflow is disabled or not installed."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class Tracker:
    def __init__(self, enabled: bool, experiment: str = "viabsa", run_name: str | None = None):
        self.mlflow: Any = None
        if not enabled:
            return
        try:
            import mlflow

            mlflow.set_experiment(experiment)
            mlflow.start_run(run_name=run_name)
            self.mlflow = mlflow
        except ImportError:
            logger.warning("mlflow not installed; tracking disabled")

    def params(self, params: dict[str, Any]) -> None:
        if self.mlflow:
            self.mlflow.log_params({k: str(v)[:250] for k, v in params.items()})

    def metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        if self.mlflow:
            self.mlflow.log_metrics(metrics, step=step)

    def artifact(self, path: str | Path) -> None:
        if self.mlflow and Path(path).exists():
            self.mlflow.log_artifact(str(path))

    def end(self) -> None:
        if self.mlflow:
            self.mlflow.end_run()
