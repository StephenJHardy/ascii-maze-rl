"""
GRPO training loop for maze solving.

Custom implementation since MLX-Tune's GRPOTrainer doesn't actually compute
gradients (it logs loss but never updates weights). We use MLX primitives
directly: mlx_lm for generation, nn.value_and_grad for training.

The loop:
  1. Sample a prompt from the dataset
  2. Generate G completions with the current policy (no gradient)
  3. Score each completion with the reward function
  4. Re-score the fixed token sequences under the current policy (with gradient)
  5. Compute GRPO loss using group-normalized advantages
  6. Update LoRA parameters via gradient descent

Usage:
    uv run python -m src.train_grpo
    uv run python -m src.train_grpo --overfit  # single-maze overfit test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm import load
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tuner.utils import LoRALinear, linear_to_lora_layers

from src.maze_dataset import MazeDataset, MazeRecord
from src.maze_gen import generate
from src.maze_repr import solution_to_str, to_prompt, to_str
from src.reward import compute_reward

DEFAULT_MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"


def find_layers(model):
    """Find the transformer layer list regardless of model architecture."""
    if hasattr(model, "language_model"):
        return model.language_model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "layers"):
        return model.layers
    raise ValueError("Cannot find transformer layers in model")


def setup_model(model_id: str, lora_rank: int = 16):
    """Load model, apply LoRA, freeze base weights."""
    model, tokenizer = load(model_id)

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

    model.freeze()
    for _, m in model.named_modules():
        if isinstance(m, LoRALinear):
            m.unfreeze()
            m.linear.freeze()

    model.train()

    n_trainable = sum(p.size for _, p in nn.utils.tree_flatten(model.trainable_parameters()))
    n_total = sum(p.size for _, p in nn.utils.tree_flatten(model.parameters()))
    print(f"  Trainable: {n_trainable:,} / {n_total:,} ({100 * n_trainable / n_total:.2f}%)")

    return model, tokenizer


def generate_completion(model, tokenizer, prompt_text: str, max_tokens: int, temperature: float):
    """Generate a single completion and return (text, token_ids)."""
    from mlx_lm.generate import generate_step

    prompt_tokens = tokenizer.encode(prompt_text)
    prompt_arr = mx.array(prompt_tokens)
    sampler = make_sampler(temp=temperature)

    generated = []
    for token, _ in generate_step(prompt_arr, model, max_tokens=max_tokens, sampler=sampler):
        token_id = token if isinstance(token, int) else token.item()
        if token_id == tokenizer.eos_token_id:
            break
        generated.append(token_id)

    completion_text = tokenizer.decode(generated)
    return completion_text, generated


def compute_log_probs(model, prompt_tokens: list[int], completion_tokens: list[int]):
    """
    Compute log probabilities of completion tokens under the current policy.

    This is the differentiable scoring step — a single forward pass through
    the full (prompt + completion) sequence, then extract log probs at the
    completion positions.
    """
    full_tokens = mx.array([prompt_tokens + completion_tokens])
    logits = model(full_tokens)

    log_probs = nn.log_softmax(logits[:, :-1, :], axis=-1)

    prompt_len = len(prompt_tokens)
    completion_arr = mx.array([completion_tokens])
    completion_log_probs = log_probs[:, prompt_len - 1 : prompt_len - 1 + len(completion_tokens), :]
    per_token_lp = mx.take_along_axis(
        completion_log_probs, completion_arr[:, :, None], axis=-1
    ).squeeze(-1)

    return per_token_lp.sum()


def grpo_step(
    model,
    tokenizer,
    optimizer,
    prompt_text: str,
    maze_record: MazeRecord,
    num_generations: int,
    max_tokens: int,
    temperature: float,
):
    """
    One GRPO training step:
      1. Generate G completions (no gradient)
      2. Score with reward function
      3. Compute advantages
      4. Differentiable loss via re-scoring
      5. Gradient update
    """
    maze = maze_record.to_maze()
    prompt_tokens = tokenizer.encode(prompt_text)

    completions = []
    completion_token_lists = []
    rewards = []

    model.eval()
    for _ in range(num_generations):
        text, tokens = generate_completion(model, tokenizer, prompt_text, max_tokens, temperature)
        completions.append(text)
        completion_token_lists.append(tokens)
        reward = compute_reward(text, maze)
        rewards.append(reward)

    rewards_arr = mx.array(rewards)
    mean_reward = mx.mean(rewards_arr)
    std_reward = mx.maximum(mx.std(rewards_arr), mx.array(1e-8))
    advantages = (rewards_arr - mean_reward) / std_reward

    model.train()

    def loss_fn(model):
        total_loss = mx.array(0.0)
        for i, comp_tokens in enumerate(completion_token_lists):
            if len(comp_tokens) == 0:
                continue
            log_prob = compute_log_probs(model, prompt_tokens, comp_tokens)
            total_loss = total_loss - advantages[i] * log_prob
        return total_loss / max(len(completion_token_lists), 1)

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    loss, grads = loss_and_grad(model)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state, loss)

    return {
        "loss": loss.item(),
        "reward_mean": mean_reward.item(),
        "reward_std": std_reward.item(),
        "reward_max": max(rewards),
        "reward_min": min(rewards),
        "completions": completions,
        "rewards": rewards,
    }


def make_overfit_dataset(tokenizer):
    """Create a single-maze dataset for the overfit test."""
    maze = generate(3, 3, seed=42)
    record = MazeRecord.from_maze(maze)
    prompt = to_prompt(maze, tokenizer=tokenizer)
    solution = solution_to_str(maze.solution_moves)
    print(f"\n  Overfit maze (3×3, seed=42):")
    print(f"  Solution: {solution} ({len(maze.solution_moves)} moves)")
    print(f"  Maze:")
    for line in to_str(maze).split("\n"):
        print(f"    {line}")
    return [record], [prompt]


def main():
    parser = argparse.ArgumentParser(description="GRPO maze training")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--overfit", action="store_true", help="Single-maze overfit test")
    parser.add_argument("--dataset", type=str, default=None, help="Path to dataset JSONL")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    args = parser.parse_args()

    print("=" * 60)
    print("ASCII Maze RL — GRPO Training")
    print("=" * 60)

    print(f"\n  Model: {args.model}")
    print(f"  LoRA rank: {args.lora_rank}")
    model, tokenizer = setup_model(args.model, lora_rank=args.lora_rank)

    if args.overfit:
        records, prompts = make_overfit_dataset(tokenizer)
    elif args.dataset:
        dataset = MazeDataset.load(args.dataset)
        records = dataset.records
        prompts = [to_prompt(r.to_maze(), tokenizer=tokenizer) for r in records]
        print(f"\n  Dataset: {len(records)} mazes")
    else:
        parser.error("Specify --overfit or --dataset")

    optimizer = optim.Adam(learning_rate=args.lr)

    print(f"\n  Max steps: {args.max_steps}")
    print(f"  Generations per step: {args.num_generations}")
    print(f"  Max completion tokens: {args.max_tokens}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Learning rate: {args.lr}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("Training...")
    print(f"{'='*60}\n")
    sys.stdout.flush()

    log_entries = []
    t_start = time.perf_counter()

    for step in range(1, args.max_steps + 1):
        idx = (step - 1) % len(records)
        record = records[idx]
        prompt = prompts[idx]

        t0 = time.perf_counter()
        metrics = grpo_step(
            model,
            tokenizer,
            optimizer,
            prompt,
            record,
            num_generations=args.num_generations,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        step_time = time.perf_counter() - t0

        metrics["step"] = step
        metrics["step_time"] = step_time
        log_entries.append({k: v for k, v in metrics.items() if k != "completions"})

        if step % args.log_interval == 0 or step == 1:
            elapsed = time.perf_counter() - t_start
            print(
                f"  Step {step:4d}/{args.max_steps} | "
                f"loss={metrics['loss']:7.3f} | "
                f"reward={metrics['reward_mean']:6.3f} ± {metrics['reward_std']:.3f} | "
                f"max={metrics['reward_max']:6.3f} | "
                f"time={step_time:.1f}s | "
                f"elapsed={elapsed:.0f}s"
            )
            best = max(metrics["completions"], key=lambda c: metrics["rewards"][metrics["completions"].index(c)])
            best_idx = metrics["completions"].index(best)
            print(f"         best: [{metrics['rewards'][best_idx]:+.3f}] {best[:80]}")
            sys.stdout.flush()

        if step % args.save_interval == 0:
            ckpt_path = output_dir / f"step-{step}"
            ckpt_path.mkdir(parents=True, exist_ok=True)
            weights = dict(nn.utils.tree_flatten(model.trainable_parameters()))
            mx.savez(str(ckpt_path / "adapters.npz"), **weights)
            with open(ckpt_path / "log.json", "w") as f:
                json.dump(log_entries, f, indent=2)
            print(f"         checkpoint saved to {ckpt_path}")

    total_time = time.perf_counter() - t_start
    print(f"\n{'='*60}")
    print(f"Training complete: {args.max_steps} steps in {total_time:.0f}s")
    print(f"{'='*60}")

    with open(output_dir / "log.json", "w") as f:
        json.dump(log_entries, f, indent=2)


if __name__ == "__main__":
    main()
