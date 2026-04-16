"""Backend selection and common interfaces."""

from __future__ import annotations

import importlib.util
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from src.maze_dataset import MazeRecord
from src.training.config import RLConfig, SFTConfig
from src.training.rewards import RewardFn


class MazePolicy(Protocol):
    """Minimal generation interface shared across backends."""

    backend_name: str

    def generate_completion(
        self,
        record: MazeRecord,
        max_tokens: int = 32,
        temperature: float = 0.0,
    ) -> str: ...


class TrainingBackend(Protocol):
    """Common backend operations used by notebooks and wrappers."""

    name: str

    def train_sft(self, config: SFTConfig) -> Path: ...

    def train_rl(self, config: RLConfig, reward_fn: RewardFn) -> Path: ...

    def load_policy(
        self,
        model: str | None = None,
        adapter_path: str | Path | None = None,
        lora_rank: int = 16,
    ) -> MazePolicy: ...


@dataclass(frozen=True)
class BackendChoice:
    name: str
    reason: str


def backend_available(name: str) -> bool:
    """Check whether the given backend's imports appear to be available."""
    required = {
        "mlx": ("mlx", "mlx_lm"),
        "torch": ("torch", "transformers", "peft"),
    }.get(name)
    if required is None:
        raise ValueError(f"Unknown backend: {name}")
    return all(importlib.util.find_spec(module) is not None for module in required)


def choose_backend(preferred: str = "auto") -> BackendChoice:
    """Choose a backend based on availability and platform."""
    if preferred != "auto":
        if not backend_available(preferred):
            raise RuntimeError(f"Requested backend {preferred!r} is not available")
        return BackendChoice(preferred, "explicit")

    if platform.system() == "Darwin" and backend_available("mlx"):
        return BackendChoice("mlx", "darwin+mlx")
    if backend_available("torch"):
        return BackendChoice("torch", "torch_available")
    if backend_available("mlx"):
        return BackendChoice("mlx", "mlx_fallback")
    raise RuntimeError(
        "No supported training backend is available. "
        "Install MLX on Apple Silicon or torch/transformers/peft for cloud GPUs."
    )


def load_backend(name: str) -> TrainingBackend:
    """Instantiate a backend by name."""
    if name == "mlx":
        from src.training.mlx_backend import MLXBackend

        return MLXBackend()
    if name == "torch":
        from src.training.torch_backend import TorchBackend

        return TorchBackend()
    raise ValueError(f"Unknown backend: {name}")
