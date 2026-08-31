"""Configuration loading and path resolution."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Repo root = two levels up from this file (src/minesub/config.py).
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"


@dataclass
class Config:
    raw: dict[str, Any]
    path: Path

    # -- convenience accessors -------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    def resolve(self, *parts: str) -> Path:
        """Resolve a repo-relative path from the ``paths`` block."""
        # config.resolve("data", "raw", "dataset.csv")
        # would become data/raw/dataset.csv
        p = Path(*parts)
        return p if p.is_absolute() else REPO_ROOT / p

    @property
    def paths(self) -> dict[str, Path]:
        return {k: self.resolve(v) for k, v in self.raw["paths"].items()}

    def ensure_dirs(self) -> None:
        for p in self.paths.values():
            p.mkdir(parents=True, exist_ok=True)


def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(cfg_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return Config(raw=raw, path=cfg_path)
