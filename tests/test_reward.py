"""Tests for the reward function."""

import pytest

from src.maze_gen import generate
from src.maze_repr import solution_to_str
from src.reward import compute_reward


@pytest.fixture
def maze_3x3():
    return generate(3, 3, seed=42)


@pytest.fixture
def maze_5x5():
    return generate(5, 5, seed=42)


class TestComputeReward:
    def test_gibberish_returns_negative(self, maze_3x3):
        assert compute_reward("I cannot solve this maze", maze_3x3) == -1.0

    def test_empty_string_returns_negative(self, maze_3x3):
        assert compute_reward("", maze_3x3) == -1.0

    def test_perfect_solution_returns_one(self, maze_3x3):
        moves_str = solution_to_str(maze_3x3.solution_moves)
        reward = compute_reward(moves_str, maze_3x3)
        assert reward == pytest.approx(1.0)

    def test_perfect_solution_5x5(self, maze_5x5):
        moves_str = solution_to_str(maze_5x5.solution_moves)
        reward = compute_reward(moves_str, maze_5x5)
        assert reward == pytest.approx(1.0)

    def test_solved_but_longer_path(self, maze_3x3):
        optimal = solution_to_str(maze_3x3.solution_moves)
        padded = optimal + " " + optimal  # double the moves (won't work if it hits walls)
        reward = compute_reward(optimal, maze_3x3)
        assert reward == pytest.approx(1.0)

    def test_partial_path_gets_partial_credit(self, maze_5x5):
        moves = list(maze_5x5.solution_moves)
        half = moves[: len(moves) // 2]
        half_str = " ".join(half)
        reward = compute_reward(half_str, maze_5x5)
        assert 0.0 < reward < 0.5

    def test_more_moves_more_reward(self, maze_5x5):
        moves = list(maze_5x5.solution_moves)
        quarter_str = " ".join(moves[: len(moves) // 4])
        half_str = " ".join(moves[: len(moves) // 2])
        three_quarter_str = " ".join(moves[: 3 * len(moves) // 4])

        r_quarter = compute_reward(quarter_str, maze_5x5)
        r_half = compute_reward(half_str, maze_5x5)
        r_three_quarter = compute_reward(three_quarter_str, maze_5x5)

        assert r_quarter < r_half < r_three_quarter

    def test_wall_collision_gets_some_credit(self, maze_3x3):
        # A single valid move that's part of the solution
        first_move = maze_3x3.solution_moves[0]
        reward = compute_reward(first_move, maze_3x3)
        assert reward >= 0.0

    def test_reward_range(self, maze_5x5):
        assert compute_reward("", maze_5x5) == -1.0
        assert compute_reward("not a maze solution", maze_5x5) == -1.0

        optimal = solution_to_str(maze_5x5.solution_moves)
        assert compute_reward(optimal, maze_5x5) == pytest.approx(1.0)

    def test_solved_reward_above_partial(self, maze_3x3):
        moves = list(maze_3x3.solution_moves)
        if len(moves) > 1:
            partial = " ".join(moves[:-1])
            full = " ".join(moves)
            assert compute_reward(full, maze_3x3) > compute_reward(partial, maze_3x3)

    def test_negative_is_below_any_valid_move(self, maze_3x3):
        gibberish_reward = compute_reward("hello world", maze_3x3)
        first_move = maze_3x3.solution_moves[0]
        move_reward = compute_reward(first_move, maze_3x3)
        assert gibberish_reward < move_reward

    def test_monotonic_across_seeds(self):
        """Reward curve is monotonic for every maze seed tested."""
        for seed in range(10):
            maze = generate(4, 4, seed=seed)
            moves = list(maze.solution_moves)
            if len(moves) < 3:
                continue
            rewards = []
            for i in range(1, len(moves) + 1):
                partial = " ".join(moves[:i])
                rewards.append(compute_reward(partial, maze))
            for i in range(len(rewards) - 1):
                assert rewards[i] <= rewards[i + 1], (
                    f"Reward decreased at step {i+1} for seed={seed}: "
                    f"{rewards[i]} > {rewards[i+1]}"
                )
