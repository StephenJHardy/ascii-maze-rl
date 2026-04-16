"""Backend-agnostic training API for maze experiments."""

from src.training.api import (
    evaluate_policy,
    get_backend,
    load_policy,
    train_rl,
    train_sft,
)
from src.training.config import DEFAULT_MODEL, RLConfig, SFTConfig
from src.training.rewards import RewardSpec, resolve_reward_fn

__all__ = [
    "DEFAULT_MODEL",
    "RLConfig",
    "RewardSpec",
    "SFTConfig",
    "evaluate_policy",
    "get_backend",
    "load_policy",
    "resolve_reward_fn",
    "train_rl",
    "train_sft",
]
