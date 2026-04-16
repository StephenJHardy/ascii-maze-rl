"""Tests for backend selection and reward resolution."""

from __future__ import annotations

import types

import pytest

from src.evaluate import evaluate_policy_records
from src.maze_dataset import MazeRecord
from src.maze_gen import generate
from src.training.backend import BackendChoice, choose_backend
from src.training.rewards import resolve_reward_fn


def test_resolve_reward_fn_none_uses_default():
    reward_fn = resolve_reward_fn()
    assert reward_fn.__name__ == "compute_reward"


def test_resolve_reward_fn_callable_passthrough():
    def custom_reward(completion, maze):
        return 0.123

    assert resolve_reward_fn(custom_reward) is custom_reward


def test_resolve_reward_fn_import_path():
    reward_fn = resolve_reward_fn("src.reward:compute_reward")
    assert reward_fn.__name__ == "compute_reward"


def test_resolve_reward_fn_bad_string():
    with pytest.raises(ValueError):
        resolve_reward_fn("src.reward.compute_reward")


def test_choose_backend_prefers_explicit_available(monkeypatch):
    monkeypatch.setattr(
        "src.training.backend.backend_available",
        lambda name: name == "torch",
    )
    choice = choose_backend("torch")
    assert choice == BackendChoice(name="torch", reason="explicit")


def test_choose_backend_auto_prefers_mlx_on_darwin(monkeypatch):
    monkeypatch.setattr("platform.system", lambda: "Darwin")
    monkeypatch.setattr(
        "src.training.backend.backend_available",
        lambda name: name == "mlx",
    )
    choice = choose_backend()
    assert choice == BackendChoice(name="mlx", reason="darwin+mlx")


def test_choose_backend_auto_falls_back_to_torch(monkeypatch):
    monkeypatch.setattr("platform.system", lambda: "Linux")
    monkeypatch.setattr(
        "src.training.backend.backend_available",
        lambda name: name == "torch",
    )
    choice = choose_backend()
    assert choice == BackendChoice(name="torch", reason="torch_available")


class DummyPolicy:
    backend_name = "dummy"

    def __init__(self, completion: str):
        self.completion = completion

    def generate_completion(self, record, max_tokens: int = 32, temperature: float = 0.0) -> str:
        del record, max_tokens, temperature
        return self.completion


def test_evaluate_policy_records_with_generic_policy():
    maze = generate(3, 3, seed=42)
    record = MazeRecord.from_maze(maze)
    policy = DummyPolicy(record.solution_moves)
    results, summary = evaluate_policy_records(policy, [record])
    assert len(results) == 1
    assert summary.total == 1
    assert summary.solved == 1
    assert summary.solve_rate == pytest.approx(1.0)
