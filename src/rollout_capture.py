"""
Capture GRPO rollouts for visualization.

Generates multiple completions per maze, scores them, computes advantages,
and packages everything for the rollout explorer viewer.

Works with both MLX (Mac) and PyTorch/CUDA models.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from src.maze_dataset import MazeDataset, MazeRecord
from src.maze_repr import SYSTEM_PROMPT, solution_to_str, to_str
from src.maze_verify import (
    DIRECTIONS,
    extract_moves,
    manhattan_progress,
    reached_exit,
    simulate,
)
from src.reward import compute_reward


@dataclass
class RolloutResult:
    """A single rollout (one generation for one maze)."""

    completion: str
    moves_parsed: list[str] | None
    path: list[list[int]]
    reward: float
    solved: bool
    valid_steps: int
    progress: float


@dataclass
class MazeRollouts:
    """All rollouts for a single maze, with advantages."""

    maze_id: str
    width: int
    height: int
    maze_str: str
    entry: list[int]
    exit: list[int]
    correct_path: list[list[int]]
    correct_moves: list[str]
    solution_length: int
    rollouts: list[RolloutResult]
    advantages: list[float]
    reward_mean: float
    reward_std: float


def score_completion(completion: str, maze) -> RolloutResult:
    """Score a single completion against a maze."""
    moves = extract_moves(completion)
    path_tuples = simulate(moves or [], maze)
    path = [list(p) for p in path_tuples]
    valid_steps = len(path_tuples) - 1
    solved = reached_exit(path_tuples, maze)
    progress = manhattan_progress(path_tuples[-1], maze.exit, maze.entry)

    return RolloutResult(
        completion=completion,
        moves_parsed=moves,
        path=path,
        reward=compute_reward(completion, maze),
        solved=solved,
        valid_steps=valid_steps,
        progress=max(progress, 0.0),
    )


def compute_advantages(rewards: list[float]) -> tuple[list[float], float, float]:
    """Compute group-relative advantages (same as GRPO)."""
    n = len(rewards)
    if n == 0:
        return [], 0.0, 0.0
    mean = sum(rewards) / n
    variance = sum((r - mean) ** 2 for r in rewards) / n
    std = max(variance**0.5, 1e-8)
    advantages = [(r - mean) / std for r in rewards]
    return advantages, mean, std


def capture_rollouts_pytorch(
    model,
    tokenizer,
    records: list[MazeRecord],
    num_generations: int = 8,
    temperature: float = 1.0,
    max_new_tokens: int = 40,
    progress: bool = True,
) -> list[MazeRollouts]:
    """Capture rollouts using a PyTorch/HuggingFace model."""
    import torch

    results = []
    total = len(records)

    for idx, record in enumerate(records):
        maze = record.to_maze()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": to_str(maze)},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        rollouts = []
        for _ in range(num_generations):
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    pad_token_id=tokenizer.eos_token_id,
                )
            response = tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True,
            )
            rollouts.append(score_completion(response, maze))

        rewards = [r.reward for r in rollouts]
        advantages, mean_r, std_r = compute_advantages(rewards)

        results.append(MazeRollouts(
            maze_id=record.id,
            width=record.width,
            height=record.height,
            maze_str=record.maze_str,
            entry=list(maze.entry),
            exit=list(maze.exit),
            correct_path=[list(p) for p in maze.solution],
            correct_moves=list(maze.solution_moves),
            solution_length=len(maze.solution_moves),
            rollouts=rollouts,
            advantages=advantages,
            reward_mean=mean_r,
            reward_std=std_r,
        ))

        if progress and ((idx + 1) % 5 == 0 or idx == total - 1):
            print(f"  [{idx+1}/{total}] {record.id}: "
                  f"mean_reward={mean_r:.3f} best={max(rewards):.3f}")

    return results


def save_rollouts(rollouts: list[MazeRollouts], path: str | Path):
    """Save rollouts to JSON for the viewer."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [asdict(r) for r in rollouts]
    with open(path, "w") as f:
        json.dump(data, f)
    print(f"Saved {len(rollouts)} maze rollouts to {path}")
