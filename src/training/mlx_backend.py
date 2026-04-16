"""MLX-backed training implementation."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm import load
from mlx_lm.tuner.datasets import CacheDataset, ChatDataset
from mlx_lm.tuner.trainer import TrainingArgs, train
from mlx_lm.tuner.utils import LoRALinear, linear_to_lora_layers

from src.maze_repr import SYSTEM_PROMPT, to_prompt
from src.train_grpo import (
    DEFAULT_MODEL as MLX_DEFAULT_MODEL,
    find_layers,
    generate_completion as mlx_generate_completion,
    grpo_step,
    save_ref_weights,
    setup_model,
)
from src.training.backend import MazePolicy
from src.training.config import RLConfig, SFTConfig, resolve_model_for_backend, resolve_records
from src.training.rewards import RewardFn


def build_chat_data(records) -> list[dict]:
    """Convert maze records to chat format for SFT training."""
    chat_data = []
    for record in records:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": record.maze_str},
            {"role": "assistant", "content": record.solution_moves},
        ]
        chat_data.append({"messages": messages})
    return chat_data


class MLXPolicy:
    """MLX-backed generation wrapper."""

    backend_name = "mlx"

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate_completion(
        self,
        record,
        max_tokens: int = 32,
        temperature: float = 0.0,
    ) -> str:
        prompt = to_prompt(record.to_maze(), tokenizer=self.tokenizer)
        completion, _ = mlx_generate_completion(
            self.model,
            self.tokenizer,
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return completion


class MLXBackend:
    """Wrapper around the existing MLX training code."""

    name = "mlx"

    def train_sft(self, config: SFTConfig) -> Path:
        model_id = resolve_model_for_backend(config.model or MLX_DEFAULT_MODEL, self.name)
        print("=" * 60)
        print("ASCII Maze RL — SFT Training (MLX backend)")
        print("=" * 60)
        print(f"\n  Model: {model_id}")

        model, tokenizer = load(model_id)

        num_layers = len(find_layers(model))
        linear_to_lora_layers(
            model,
            num_layers=num_layers,
            config={
                "rank": config.lora_rank,
                "alpha": float(config.lora_rank),
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
        for _, module in model.named_modules():
            if isinstance(module, LoRALinear):
                module.unfreeze()
                module.linear.freeze()

        n_trainable = sum(p.size for _, p in nn.utils.tree_flatten(model.trainable_parameters()))
        n_total = sum(p.size for _, p in nn.utils.tree_flatten(model.parameters()))
        print(f"  LoRA rank: {config.lora_rank}")
        print(f"  Trainable: {n_trainable:,} / {n_total:,} ({100 * n_trainable / n_total:.2f}%)")

        train_records = resolve_records(config.dataset, config.records)
        train_ds = CacheDataset(ChatDataset(build_chat_data(train_records), tokenizer, mask_prompt=True))
        print(f"  Training examples: {len(train_ds)}")

        val_ds = None
        if config.val_dataset is not None or config.val_records is not None:
            val_records = resolve_records(config.val_dataset, config.val_records)
            val_ds = CacheDataset(ChatDataset(build_chat_data(val_records), tokenizer, mask_prompt=True))
            print(f"  Validation examples: {len(val_ds)}")

        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        training_args = TrainingArgs(
            batch_size=config.batch_size,
            iters=config.iters,
            val_batches=25,
            steps_per_report=10,
            steps_per_eval=50,
            steps_per_save=50,
            max_seq_length=512,
            adapter_file=str(output_dir / "adapters.safetensors"),
            grad_checkpoint=False,
        )
        optimizer = optim.Adam(learning_rate=config.lr)

        print(f"\n  Iterations: {config.iters}")
        print(f"  Batch size: {config.batch_size}")
        print(f"  Learning rate: {config.lr}")
        print(f"  Output: {output_dir}")
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
        return output_dir

    def train_rl(self, config: RLConfig, reward_fn: RewardFn) -> Path:
        if reward_fn.__module__ != "src.reward" or reward_fn.__name__ != "compute_reward":
            raise NotImplementedError(
                "Custom reward functions are not yet wired into the MLX GRPO path. "
                "Use the torch backend for reward experimentation."
            )

        model_id = resolve_model_for_backend(config.model or MLX_DEFAULT_MODEL, self.name)
        print("=" * 60)
        print("ASCII Maze RL — GRPO Training (MLX backend)")
        print("=" * 60)
        print(f"\n  Model: {model_id}")
        print(f"  LoRA rank: {config.lora_rank}")
        if config.adapters:
            print(f"  Warm-start: {config.adapters}")

        model, tokenizer = setup_model(
            model_id,
            lora_rank=config.lora_rank,
            adapter_path=str(config.adapters) if config.adapters is not None else None,
        )

        records = resolve_records(config.dataset, config.records)
        prompts = [to_prompt(r.to_maze(), tokenizer=tokenizer) for r in records]
        print(f"\n  Dataset: {len(records)} mazes")

        ref_weights = save_ref_weights(model)
        optimizer = optim.Adam(learning_rate=config.lr)
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        log_entries = []
        t_start = time.perf_counter()
        for step in range(1, config.max_steps + 1):
            idx = (step - 1) % len(records)
            metrics = grpo_step(
                model,
                tokenizer,
                optimizer,
                prompts[idx],
                records[idx],
                ref_weights=ref_weights,
                beta=config.beta,
                num_generations=config.num_generations,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
            )
            metrics["step"] = step
            log_entries.append({k: v for k, v in metrics.items() if k != "completions"})

            if step % config.log_interval == 0 or step == 1:
                elapsed = time.perf_counter() - t_start
                print(
                    f"  Step {step:4d}/{config.max_steps} | "
                    f"loss={metrics['loss']:7.3f} | "
                    f"reward={metrics['reward_mean']:6.3f} ± {metrics['reward_std']:.3f} | "
                    f"max={metrics['reward_max']:6.3f} | elapsed={elapsed:.0f}s"
                )
                sys.stdout.flush()

            if step % config.save_interval == 0:
                ckpt_path = output_dir / f"step-{step}"
                ckpt_path.mkdir(parents=True, exist_ok=True)
                weights = dict(nn.utils.tree_flatten(model.trainable_parameters()))
                mx.savez(str(ckpt_path / "adapters.npz"), **weights)
                with open(ckpt_path / "log.json", "w") as handle:
                    json.dump(log_entries, handle, indent=2)

        with open(output_dir / "log.json", "w") as handle:
            json.dump(log_entries, handle, indent=2)
        print(f"\nTraining complete. Output: {output_dir}")
        return output_dir

    def load_policy(
        self,
        model: str | None = None,
        adapter_path: str | Path | None = None,
        lora_rank: int = 16,
    ) -> MazePolicy:
        model_id = resolve_model_for_backend(model or MLX_DEFAULT_MODEL, self.name)
        model_obj, tokenizer = load(model_id)

        if adapter_path is not None:
            linear_to_lora_layers(
                model_obj,
                num_layers=len(find_layers(model_obj)),
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
            adapter_dir = Path(adapter_path)
            for name in ["adapters.safetensors", "adapters.npz"]:
                candidate = adapter_dir / name
                if candidate.exists():
                    weights = mx.load(str(candidate))
                    model_obj.load_weights(list(weights.items()), strict=False)
                    break
            else:
                raise FileNotFoundError(f"No adapters found in {adapter_dir}")

        model_obj.eval()
        return MLXPolicy(model_obj, tokenizer)
