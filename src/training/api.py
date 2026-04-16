"""High-level backend-agnostic training entry points."""

from __future__ import annotations

from pathlib import Path

from src.evaluate import EvalSummary, evaluate_policy_records
from src.training.backend import MazePolicy, choose_backend, load_backend
from src.training.config import RLConfig, SFTConfig
from src.training.rewards import RewardSpec, resolve_reward_fn


def get_backend(preferred: str = "auto"):
    """Select and instantiate a backend."""
    choice = choose_backend(preferred)
    return load_backend(choice.name)


def train_sft(config: SFTConfig) -> Path:
    """Run SFT using the chosen backend."""
    backend = get_backend(config.backend)
    return backend.train_sft(config)


def train_rl(config: RLConfig, reward: RewardSpec = None) -> Path:
    """Run RL using the chosen backend and reward function."""
    backend = get_backend(config.backend)
    reward_fn = resolve_reward_fn(reward)
    return backend.train_rl(config, reward_fn)


def load_policy(
    backend: str = "auto",
    model: str | None = None,
    adapter_path: str | Path | None = None,
    lora_rank: int = 16,
) -> MazePolicy:
    """Load an inference policy for generation and evaluation."""
    selected = get_backend(backend)
    return selected.load_policy(model=model, adapter_path=adapter_path, lora_rank=lora_rank)


def evaluate_policy(
    policy: MazePolicy,
    records,
    max_tokens: int = 32,
    temperature: float = 0.0,
    num_samples: int = 1,
    verbose: bool = False,
) -> tuple[list, EvalSummary]:
    """Evaluate a backend policy using the shared evaluation logic."""
    return evaluate_policy_records(
        policy,
        records,
        max_tokens=max_tokens,
        temperature=temperature,
        num_samples=num_samples,
        verbose=verbose,
    )
