"""
Evaluation script for maze-solving models.

Scores a model (base or with LoRA adapters) against a set of mazes and
reports solve rate by size and difficulty.

Usage:
    # Evaluate base model (no adapters)
    uv run python -m src.evaluate --dataset data/eval_3x3.jsonl

    # Evaluate a trained checkpoint
    uv run python -m src.evaluate --dataset data/eval_3x3.jsonl --adapters checkpoints/step-200

    # Evaluate with custom settings
    uv run python -m src.evaluate --dataset data/eval_3x3.jsonl --temperature 0.0 --samples 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

from src.maze_dataset import MazeDataset, MazeRecord
from src.maze_verify import extract_moves, manhattan_progress, reached_exit, simulate
from src.reward import compute_reward

DEFAULT_MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"


@dataclass
class EvalResult:
    """Result of evaluating a single maze."""

    maze_id: str
    width: int
    height: int
    solution_length: int
    completion: str
    moves_parsed: list[str] | None
    reward: float
    solved: bool
    valid_steps: int
    progress: float


@dataclass
class EvalSummary:
    """Aggregated evaluation results."""

    total: int = 0
    solved: int = 0
    parseable: int = 0
    mean_reward: float = 0.0
    mean_progress: float = 0.0
    by_size: dict[str, dict] = field(default_factory=dict)
    by_difficulty: dict[str, dict] = field(default_factory=dict)

    @property
    def solve_rate(self) -> float:
        return self.solved / max(self.total, 1)

    @property
    def parse_rate(self) -> float:
        return self.parseable / max(self.total, 1)


def load_model_for_eval(
    model_id: str,
    adapter_path: str | None = None,
    lora_rank: int = 16,
):
    """Load a model for evaluation, optionally with LoRA adapters."""
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.tuner.utils import linear_to_lora_layers

    from src.train_grpo import find_layers

    model, tokenizer = load(model_id)

    if adapter_path is not None:
        adapter_path = Path(adapter_path)

        adapter_file = None
        for name in ["adapters.safetensors", "adapters.npz"]:
            candidate = adapter_path / name
            if candidate.exists():
                adapter_file = candidate
                break
        if adapter_file is None:
            raise FileNotFoundError(f"No adapters found in {adapter_path}")

        num_layers = len(find_layers(model))
        linear_to_lora_layers(
            model,
            num_layers=num_layers,
            config={
                "rank": lora_rank,
                "alpha": float(lora_rank),
                "dropout": 0.0,
                "scale": 1.0,
                "keys": [
                    "self_attn.q_proj",
                    "self_attn.k_proj",
                    "self_attn.v_proj",
                    "self_attn.o_proj",
                ],
            },
        )

        weights = mx.load(str(adapter_file))
        model.load_weights(list(weights.items()), strict=False)
        print(f"  Loaded adapters from {adapter_file}")

    model.eval()
    return model, tokenizer


def evaluate_maze(
    model,
    tokenizer,
    record: MazeRecord,
    max_tokens: int = 32,
    temperature: float = 0.0,
    num_samples: int = 1,
) -> EvalResult:
    """
    Evaluate a model on a single maze.

    If num_samples > 1, generates multiple completions and takes the best.
    Uses temperature=0.0 (greedy) by default for deterministic evaluation.
    """
    from src.maze_repr import to_prompt
    from src.train_grpo import generate_completion

    maze = record.to_maze()
    prompt = to_prompt(maze, tokenizer=tokenizer)

    best_reward = -2.0
    best_completion = ""
    best_moves = None

    for _ in range(num_samples):
        completion, _ = generate_completion(model, tokenizer, prompt, max_tokens, temperature)
        reward = compute_reward(completion, maze)
        if reward > best_reward:
            best_reward = reward
            best_completion = completion
            best_moves = extract_moves(completion)

    path = simulate(best_moves or [], maze)
    valid_steps = len(path) - 1
    progress = manhattan_progress(path[-1], maze.exit, maze.entry)

    return EvalResult(
        maze_id=record.id,
        width=record.width,
        height=record.height,
        solution_length=record.solution_length,
        completion=best_completion,
        moves_parsed=best_moves,
        reward=best_reward,
        solved=reached_exit(path, maze),
        valid_steps=valid_steps,
        progress=max(progress, 0.0),
    )


def evaluate_policy_records(
    policy,
    records: list[MazeRecord],
    max_tokens: int = 32,
    temperature: float = 0.0,
    num_samples: int = 1,
    verbose: bool = False,
) -> tuple[list[EvalResult], EvalSummary]:
    """Evaluate any policy object that implements generate_completion()."""
    results = []
    for i, record in enumerate(records):
        maze = record.to_maze()
        best_reward = -2.0
        best_completion = ""
        best_moves = None

        for _ in range(num_samples):
            completion = policy.generate_completion(
                record,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            reward = compute_reward(completion, maze)
            if reward > best_reward:
                best_reward = reward
                best_completion = completion
                best_moves = extract_moves(completion)

        path = simulate(best_moves or [], maze)
        valid_steps = len(path) - 1
        progress = manhattan_progress(path[-1], maze.exit, maze.entry)

        results.append(
            EvalResult(
                maze_id=record.id,
                width=record.width,
                height=record.height,
                solution_length=record.solution_length,
                completion=best_completion,
                moves_parsed=best_moves,
                reward=best_reward,
                solved=reached_exit(path, maze),
                valid_steps=valid_steps,
                progress=max(progress, 0.0),
            )
        )

        if verbose and (i + 1) % 10 == 0:
            solved_so_far = sum(1 for r in results if r.solved)
            print(
                f"  [{i+1}/{len(records)}] "
                f"solved={solved_so_far}/{i+1} "
                f"({100*solved_so_far/(i+1):.0f}%)",
            )
            sys.stdout.flush()

    return results, summarize_results(results)


def summarize_results(results: list[EvalResult]) -> EvalSummary:
    """Aggregate individual results into a summary."""
    summary = EvalSummary(total=len(results))

    if not results:
        return summary

    summary.solved = sum(1 for r in results if r.solved)
    summary.parseable = sum(1 for r in results if r.moves_parsed is not None)
    summary.mean_reward = sum(r.reward for r in results) / len(results)
    summary.mean_progress = sum(r.progress for r in results) / len(results)

    size_groups: dict[str, list[EvalResult]] = defaultdict(list)
    for r in results:
        size_groups[f"{r.width}x{r.height}"].append(r)

    for size, group in sorted(size_groups.items()):
        n = len(group)
        solved = sum(1 for r in group if r.solved)
        summary.by_size[size] = {
            "count": n,
            "solved": solved,
            "solve_rate": solved / n,
            "mean_reward": sum(r.reward for r in group) / n,
            "mean_progress": sum(r.progress for r in group) / n,
        }

    diff_groups: dict[str, list[EvalResult]] = defaultdict(list)
    for r in results:
        if r.solution_length <= 6:
            diff_groups["easy"].append(r)
        elif r.solution_length <= 14:
            diff_groups["medium"].append(r)
        else:
            diff_groups["hard"].append(r)

    for diff, group in diff_groups.items():
        n = len(group)
        solved = sum(1 for r in group if r.solved)
        summary.by_difficulty[diff] = {
            "count": n,
            "solved": solved,
            "solve_rate": solved / n,
            "mean_reward": sum(r.reward for r in group) / n,
        }

    return summary


def print_summary(summary: EvalSummary):
    """Print a human-readable evaluation summary."""
    print(f"\n{'='*60}")
    print("Evaluation Results")
    print(f"{'='*60}")
    print(f"  Total mazes:  {summary.total}")
    print(f"  Solved:       {summary.solved}/{summary.total} ({100*summary.solve_rate:.1f}%)")
    print(f"  Parseable:    {summary.parseable}/{summary.total} ({100*summary.parse_rate:.1f}%)")
    print(f"  Mean reward:  {summary.mean_reward:.3f}")
    print(f"  Mean progress:{summary.mean_progress:.3f}")

    if summary.by_size:
        print("\n  By size:")
        for size, stats in sorted(summary.by_size.items()):
            print(
                f"    {size:5s}: {stats['solved']:3d}/{stats['count']:3d} "
                f"({100*stats['solve_rate']:5.1f}%)  "
                f"reward={stats['mean_reward']:.3f}  "
                f"progress={stats['mean_progress']:.3f}"
            )

    if summary.by_difficulty:
        print("\n  By difficulty:")
        for diff in ["easy", "medium", "hard"]:
            if diff in summary.by_difficulty:
                stats = summary.by_difficulty[diff]
                print(
                    f"    {diff:6s}: {stats['solved']:3d}/{stats['count']:3d} "
                    f"({100*stats['solve_rate']:5.1f}%)  "
                    f"reward={stats['mean_reward']:.3f}"
                )
    print()


def evaluate_dataset(
    model,
    tokenizer,
    dataset: MazeDataset,
    max_tokens: int = 32,
    temperature: float = 0.0,
    num_samples: int = 1,
    verbose: bool = False,
) -> tuple[list[EvalResult], EvalSummary]:
    """Evaluate a model on an entire dataset."""
    results = []
    for i, record in enumerate(dataset):
        result = evaluate_maze(
            model, tokenizer, record,
            max_tokens=max_tokens,
            temperature=temperature,
            num_samples=num_samples,
        )
        results.append(result)

        if verbose and (i + 1) % 10 == 0:
            solved_so_far = sum(1 for r in results if r.solved)
            print(
                f"  [{i+1}/{len(dataset)}] "
                f"solved={solved_so_far}/{i+1} "
                f"({100*solved_so_far/(i+1):.0f}%)",
            )
            sys.stdout.flush()

    summary = summarize_results(results)
    return results, summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate maze solver")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--adapters", type=str, default=None,
                        help="Path to LoRA adapter checkpoint")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to evaluation JSONL")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results JSON to this path")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0.0 = greedy (deterministic)")
    parser.add_argument("--samples", type=int, default=1,
                        help="Generations per maze (take best)")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only first N mazes")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("ASCII Maze RL — Evaluation")
    print("=" * 60)

    print(f"\n  Model: {args.model}")
    print(f"  Adapters: {args.adapters or 'none (base model)'}")
    model, tokenizer = load_model_for_eval(args.model, args.adapters, args.lora_rank)

    dataset = MazeDataset.load(args.dataset)
    if args.limit:
        dataset = MazeDataset(records=dataset.records[: args.limit])
    print(f"  Dataset: {len(dataset)} mazes")
    print(f"  Temperature: {args.temperature}")
    print(f"  Samples per maze: {args.samples}")
    sys.stdout.flush()

    t0 = time.perf_counter()
    results, summary = evaluate_dataset(
        model, tokenizer, dataset,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        num_samples=args.samples,
        verbose=args.verbose,
    )
    elapsed = time.perf_counter() - t0

    print_summary(summary)
    print(f"  Evaluation time: {elapsed:.1f}s ({elapsed/len(dataset):.2f}s per maze)")

    if args.verbose:
        print("\nSample outputs:")
        for r in results[:5]:
            status = "SOLVED" if r.solved else "FAILED"
            print(f"  [{status}] {r.maze_id} (sol={r.solution_length} moves): "
                  f"reward={r.reward:.3f} output={r.completion[:60]!r}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_data = {
            "summary": asdict(summary),
            "results": [asdict(r) for r in results],
            "config": {
                "model": args.model,
                "adapters": args.adapters,
                "temperature": args.temperature,
                "samples": args.samples,
                "max_tokens": args.max_tokens,
            },
        }
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"  Results saved to {output_path}")


if __name__ == "__main__":
    main()
