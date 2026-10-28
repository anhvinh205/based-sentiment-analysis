"""Tiny YAML config loader with ``--set a.b=c`` style overrides."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    """Load a YAML file and apply dotted-key overrides such as ``training.lr=3e-5``."""
    with open(path, encoding="utf-8") as fh:
        cfg: dict[str, Any] = yaml.safe_load(fh) or {}
    for item in overrides or []:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"Override must look like key=value, got {item!r}")
        _set_nested(cfg, key.strip().split("."), _parse_scalar(raw.strip()))
    return cfg


def _parse_scalar(raw: str) -> Any:
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return {"true": True, "false": False}.get(raw.lower(), raw)


def _set_nested(cfg: dict[str, Any], keys: list[str], value: Any) -> None:
    node = cfg
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = value
