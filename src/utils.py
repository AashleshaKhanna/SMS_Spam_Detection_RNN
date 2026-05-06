"""Utility functions for training and persistence."""

from __future__ import annotations

from pathlib import Path
import json

import torch


def ensure_dirs() -> None:
    Path("results").mkdir(exist_ok=True)
    Path("checkpoints").mkdir(exist_ok=True)


def save_vocab(path: str, stoi: dict[str, int], itos: dict[int, str]) -> None:
    """Save vocabulary mappings as JSON."""
    payload = {"stoi": stoi, "itos": {str(k): v for k, v in itos.items()}}
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_vocab(path: str) -> tuple[dict[str, int], dict[int, str]]:
    """Load vocabulary mappings from JSON."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    stoi = {k: int(v) for k, v in payload["stoi"].items()}
    itos = {int(k): v for k, v in payload["itos"].items()}
    return stoi, itos


def save_checkpoint(path: str, model: torch.nn.Module, metadata: dict) -> None:
    """Save model checkpoint."""
    ensure_dirs()
    torch.save({"model_state": model.state_dict(), "metadata": metadata}, path)
