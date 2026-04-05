"""Tests for the evaluation module."""

import pytest

from src.evaluate import EvalResult, EvalSummary, summarize_results
from src.maze_dataset import DatasetConfig, MazeDataset, SizeConfig


@pytest.fixture
def sample_dataset():
    config = DatasetConfig(
        name="test",
        algorithm="wilson",
        sizes=[
            SizeConfig(width=3, height=3, count=5, start_seed=0),
            SizeConfig(width=4, height=4, count=5, start_seed=100),
        ],
    )
    return MazeDataset.generate(config, progress=False)


_SENTINEL = object()


def make_eval_result(
    maze_id="3x3_0",
    width=3,
    height=3,
    solution_length=4,
    completion="d r r d",
    moves_parsed=_SENTINEL,
    reward=1.0,
    solved=True,
    valid_steps=4,
    progress=1.0,
) -> EvalResult:
    if moves_parsed is _SENTINEL:
        moves_parsed = ["d", "r", "r", "d"]
    return EvalResult(
        maze_id=maze_id,
        width=width,
        height=height,
        solution_length=solution_length,
        completion=completion,
        moves_parsed=moves_parsed,
        reward=reward,
        solved=solved,
        valid_steps=valid_steps,
        progress=progress,
    )


class TestEvalResult:
    def test_fields(self):
        r = make_eval_result()
        assert r.maze_id == "3x3_0"
        assert r.solved is True
        assert r.reward == 1.0

    def test_failed_result(self):
        r = make_eval_result(
            solved=False, reward=-1.0, moves_parsed=[], valid_steps=0, progress=0.0,
        )
        assert r.solved is False
        assert r.moves_parsed == []


class TestSummarizeResults:
    def test_all_solved(self):
        results = [make_eval_result(maze_id=f"3x3_{i}") for i in range(5)]
        summary = summarize_results(results)
        assert summary.total == 5
        assert summary.solved == 5
        assert summary.solve_rate == pytest.approx(1.0)
        assert summary.parseable == 5

    def test_none_solved(self):
        results = [
            make_eval_result(
                maze_id=f"3x3_{i}", solved=False, reward=-1.0,
                moves_parsed=None, completion="I can't solve this",
                valid_steps=0, progress=0.0,
            )
            for i in range(5)
        ]
        summary = summarize_results(results)
        assert summary.solved == 0
        assert summary.solve_rate == pytest.approx(0.0)
        assert summary.parseable == 0

    def test_mixed_results(self):
        results = [
            make_eval_result(maze_id="a", solved=True, reward=1.0),
            make_eval_result(maze_id="b", solved=False, reward=0.2, valid_steps=2, progress=0.5),
            make_eval_result(maze_id="c", solved=False, reward=-1.0, moves_parsed=None,
                             completion="gibberish", valid_steps=0, progress=0.0),
        ]
        summary = summarize_results(results)
        assert summary.total == 3
        assert summary.solved == 1
        assert summary.parseable == 2

    def test_by_size(self):
        results = [
            make_eval_result(maze_id="3x3_0", width=3, height=3, solved=True, reward=1.0),
            make_eval_result(maze_id="3x3_1", width=3, height=3, solved=True, reward=1.0),
            make_eval_result(maze_id="4x4_0", width=4, height=4, solved=False, reward=0.2,
                             solution_length=8, valid_steps=3, progress=0.4),
        ]
        summary = summarize_results(results)
        assert "3x3" in summary.by_size
        assert "4x4" in summary.by_size
        assert summary.by_size["3x3"]["solve_rate"] == pytest.approx(1.0)
        assert summary.by_size["4x4"]["solve_rate"] == pytest.approx(0.0)

    def test_by_difficulty(self):
        results = [
            make_eval_result(maze_id="easy", solution_length=4, solved=True),
            make_eval_result(maze_id="medium", solution_length=10, solved=False,
                             reward=0.2, valid_steps=3, progress=0.5),
            make_eval_result(maze_id="hard", solution_length=20, solved=False,
                             reward=0.1, valid_steps=2, progress=0.2),
        ]
        summary = summarize_results(results)
        assert "easy" in summary.by_difficulty
        assert "medium" in summary.by_difficulty
        assert "hard" in summary.by_difficulty
        assert summary.by_difficulty["easy"]["solve_rate"] == pytest.approx(1.0)

    def test_empty_results(self):
        summary = summarize_results([])
        assert summary.total == 0
        assert summary.solve_rate == pytest.approx(0.0)

    def test_mean_reward(self):
        results = [
            make_eval_result(reward=1.0),
            make_eval_result(reward=0.5),
            make_eval_result(reward=0.0),
        ]
        summary = summarize_results(results)
        assert summary.mean_reward == pytest.approx(0.5)

    def test_mean_progress(self):
        results = [
            make_eval_result(progress=1.0),
            make_eval_result(progress=0.5),
            make_eval_result(progress=0.0),
        ]
        summary = summarize_results(results)
        assert summary.mean_progress == pytest.approx(0.5)


class TestEvalSummary:
    def test_solve_rate_property(self):
        s = EvalSummary(total=10, solved=7)
        assert s.solve_rate == pytest.approx(0.7)

    def test_parse_rate_property(self):
        s = EvalSummary(total=10, parseable=8)
        assert s.parse_rate == pytest.approx(0.8)

    def test_empty_summary(self):
        s = EvalSummary()
        assert s.solve_rate == pytest.approx(0.0)
        assert s.parse_rate == pytest.approx(0.0)
