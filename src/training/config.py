"""Shared training configuration objects."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from src.maze_dataset import MazeDataset, MazeRecord

DEFAULT_MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
TORCH_DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

MODEL_ALIASES = {
    DEFAULT_MODEL: {
        "mlx": DEFAULT_MODEL,
        "torch": TORCH_DEFAULT_MODEL,
    },
    TORCH_DEFAULT_MODEL: {
        "mlx": DEFAULT_MODEL,
        "torch": TORCH_DEFAULT_MODEL,
    },
}


@dataclass
class SFTConfig:
    model: str | None = None
    dataset: str | Path | None = None
    records: Sequence[MazeRecord] | None = None
    val_dataset: str | Path | None = None
    val_records: Sequence[MazeRecord] | None = None
    iters: int = 200
    epochs: float | None = None
    batch_size: int = 4
    lr: float = 1e-4
    lora_rank: int = 16
    output_dir: str | Path = "checkpoints/sft"
    backend: str = "auto"


@dataclass
class RLConfig:
    model: str | None = None
    adapters: str | Path | None = None
    dataset: str | Path | None = None
    records: Sequence[MazeRecord] | None = None
    max_steps: int = 200
    num_generations: int = 8
    max_tokens: int = 32
    temperature: float = 0.7
    lr: float = 5e-6
    beta: float = 0.04
    lora_rank: int = 16
    log_interval: int = 10
    save_interval: int = 50
    output_dir: str | Path = "checkpoints/grpo"
    backend: str = "auto"


def resolve_records(
    dataset: str | Path | None = None,
    records: Sequence[MazeRecord] | None = None,
) -> list[MazeRecord]:
    """Load records from either a path or an in-memory collection."""
    if records is not None:
        return list(records)
    if dataset is None:
        raise ValueError("Specify either dataset or records")
    loaded = MazeDataset.load(dataset)
    return loaded.records


def resolve_model_for_backend(model: str | None, backend_name: str) -> str:
    """Translate a model id into the backend's preferred default."""
    selected = model or DEFAULT_MODEL
    aliases = MODEL_ALIASES.get(selected)
    if aliases and backend_name in aliases:
        return aliases[backend_name]
    return selected
