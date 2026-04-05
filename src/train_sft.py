"""
SFT (Supervised Fine-Tuning) for maze solving.

Fine-tunes LoRA adapters on solved maze examples so the model learns to
read maze structure and output maze-specific solutions. This creates a
warm-start checkpoint for GRPO training.

Uses mlx-lm's native LoRA trainer for efficiency.

Usage:
    uv run python -m src.train_sft
    uv run python -m src.train_sft --dataset data/train_3x3.jsonl --iters 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm import load
from mlx_lm.tuner.datasets import CacheDataset, ChatDataset
from mlx_lm.tuner.trainer import TrainingArgs, train
from mlx_lm.tuner.utils import LoRALinear, linear_to_lora_layers

from src.maze_dataset import MazeDataset
from src.maze_repr import SYSTEM_PROMPT
from src.train_grpo import DEFAULT_MODEL, find_layers


def build_chat_data(dataset: MazeDataset) -> list[dict]:
    """Convert maze records to chat format for SFT training."""
    chat_data = []
    for record in dataset:
        maze_str = record.maze_str
        solution = record.solution_moves

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": maze_str},
            {"role": "assistant", "content": solution},
        ]
        chat_data.append({"messages": messages})

    return chat_data


def main():
    parser = argparse.ArgumentParser(description="SFT maze training")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--dataset", type=str, default="data/train_3x3.jsonl")
    parser.add_argument("--val-dataset", type=str, default=None)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--output-dir", type=str, default="checkpoints/sft")
    args = parser.parse_args()

    print("=" * 60)
    print("ASCII Maze RL — SFT Training")
    print("=" * 60)

    print(f"\n  Model: {args.model}")
    model, tokenizer = load(args.model)

    num_layers = len(find_layers(model))
    linear_to_lora_layers(
        model,
        num_layers=num_layers,
        config={
            "rank": args.lora_rank,
            "alpha": float(args.lora_rank),
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

    n_trainable = sum(p.size for _, p in nn.utils.tree_flatten(model.trainable_parameters()))
    n_total = sum(p.size for _, p in nn.utils.tree_flatten(model.parameters()))
    print(f"  LoRA rank: {args.lora_rank}")
    print(f"  Trainable: {n_trainable:,} / {n_total:,} ({100 * n_trainable / n_total:.2f}%)")

    print(f"\n  Loading training data: {args.dataset}")
    maze_ds = MazeDataset.load(args.dataset)
    chat_data = build_chat_data(maze_ds)
    train_ds = CacheDataset(ChatDataset(chat_data, tokenizer, mask_prompt=True))
    print(f"  Training examples: {len(train_ds)}")

    val_ds = None
    if args.val_dataset:
        print(f"  Loading val data: {args.val_dataset}")
        val_maze_ds = MazeDataset.load(args.val_dataset)
        val_chat_data = build_chat_data(val_maze_ds)
        val_ds = CacheDataset(ChatDataset(val_chat_data, tokenizer, mask_prompt=True))
        print(f"  Validation examples: {len(val_ds)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArgs(
        batch_size=args.batch_size,
        iters=args.iters,
        val_batches=25,
        steps_per_report=10,
        steps_per_eval=50,
        steps_per_save=50,
        max_seq_length=512,
        adapter_file=str(output_dir / "adapters.safetensors"),
        grad_checkpoint=False,
    )

    optimizer = optim.Adam(learning_rate=args.lr)

    print(f"\n  Iterations: {args.iters}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Output: {output_dir}")
    print()
    sys.stdout.flush()

    model.train()
    train(
        model=model,
        optimizer=optimizer,
        train_dataset=train_ds,
        val_dataset=val_ds,
        args=training_args,
    )

    print("\nSFT training complete!")
    print(f"Adapters saved to: {output_dir}")


if __name__ == "__main__":
    main()
