"""
Reward function for GRPO training.

Scores model completions against a maze using verifiable simulation.
Designed to provide smooth, monotonically increasing signal as the
model gets more of the solution right.

Reward ranges:
  -1.0        : unparseable output (gibberish, conversational filler)
   0.0 – 0.5  : partial credit (valid moves, progress toward exit)
   0.6 – 1.0  : solved (bonus for efficiency vs optimal path)
"""

from __future__ import annotations

from src.maze_gen import Maze
from src.maze_verify import (
    extract_moves,
    manhattan_progress,
    reached_exit,
    simulate,
)


def compute_reward(completion: str, maze: Maze) -> float:
    """
    Score a model completion against a maze.

    Returns a float in [-1.0, 1.0].
    """
    moves = extract_moves(completion)

    if moves is None:
        return -1.0

    path = simulate(moves, maze)
    valid_steps = len(path) - 1
    optimal_len = len(maze.solution) - 1  # solution includes start cell

    if optimal_len == 0:
        return 1.0 if reached_exit(path, maze) else -1.0

    if reached_exit(path, maze):
        efficiency = optimal_len / max(len(moves), optimal_len)
        return 0.6 + 0.4 * efficiency

    coverage = min(valid_steps / optimal_len, 1.0)
    progress = manhattan_progress(path[-1], maze.exit, maze.entry)
    progress = max(progress, 0.0)  # clamp negative progress

    return 0.5 * (0.7 * coverage + 0.3 * progress)


def reward_fn_for_maze(maze: Maze):
    """
    Return a reward function closed over a specific maze.

    Compatible with MLX-Tune's GRPOTrainer reward_fn interface:
        reward_fn(response: str, ground_truth: str) -> float
    """
    def reward_fn(response: str, ground_truth: str) -> float:
        return compute_reward(response, maze)
    return reward_fn
