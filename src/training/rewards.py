"""Reward function resolution helpers."""

from __future__ import annotations

import importlib
from typing import Callable

from src.maze_gen import Maze
from src.reward import compute_reward

RewardFn = Callable[[str, Maze], float]
RewardSpec = RewardFn | str | None


def resolve_reward_fn(reward: RewardSpec = None) -> RewardFn:
    """
    Resolve a reward specification into a callable.

    Supported forms:
      - ``None`` -> default ``src.reward.compute_reward``
      - callable -> returned as-is
      - ``"module.submodule:function_name"`` -> imported callable
    """
    if reward is None:
        return compute_reward
    if callable(reward):
        return reward
    if ":" not in reward:
        raise ValueError(
            "Reward string must use 'module.path:function_name', "
            f"got {reward!r}"
        )
    module_name, attr_name = reward.split(":", 1)
    module = importlib.import_module(module_name)
    fn = getattr(module, attr_name)
    if not callable(fn):
        raise TypeError(f"Resolved reward object is not callable: {reward!r}")
    return fn
