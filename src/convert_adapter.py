"""
Convert MLX-LM LoRA adapters to HuggingFace PEFT format.

MLX format:  model.layers.{i}.self_attn.{proj}.lora_a  shape (in, r)
PEFT format: base_model.model.model.layers.{i}.self_attn.{proj}.lora_A.weight  shape (r, in)

Usage:
    uv run python -m src.convert_adapter \
        --input checkpoints/sft_large_5k \
        --model Qwen/Qwen2.5-0.5B-Instruct \
        --output checkpoints/sft_large_5k_peft \
        --push StephenJHardy/maze-sft-qwen2.5-0.5b
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def convert_mlx_to_peft(input_dir: str, output_dir: str, model_id: str, lora_rank: int = 16):
    """Convert MLX LoRA adapter to PEFT-compatible format."""
    from safetensors.numpy import load_file, save_file

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    adapter_file = input_path / "adapters.safetensors"
    mlx_weights = load_file(str(adapter_file))

    peft_weights = {}
    for key, value in mlx_weights.items():
        if ".lora_a" in key:
            peft_key = key.replace("model.layers.", "base_model.model.model.layers.")
            peft_key = peft_key.replace(".lora_a", ".lora_A.weight")
            peft_weights[peft_key] = value.T
        elif ".lora_b" in key:
            peft_key = key.replace("model.layers.", "base_model.model.model.layers.")
            peft_key = peft_key.replace(".lora_b", ".lora_B.weight")
            peft_weights[peft_key] = value.T

    save_file(peft_weights, str(output_path / "adapter_model.safetensors"))

    target_modules = set()
    for key in mlx_weights:
        parts = key.split(".")
        for i, part in enumerate(parts):
            if part in ("q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"):
                target_modules.add(part)

    adapter_config = {
        "auto_mapping": None,
        "base_model_name_or_path": model_id,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "layers_to_transform": None,
        "layers_pattern": None,
        "lora_alpha": lora_rank,
        "lora_dropout": 0.0,
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": lora_rank,
        "revision": None,
        "target_modules": sorted(target_modules),
        "task_type": "CAUSAL_LM",
    }

    with open(output_path / "adapter_config.json", "w") as f:
        json.dump(adapter_config, f, indent=2)

    print(f"Converted {len(mlx_weights)} MLX weights -> {len(peft_weights)} PEFT weights")
    print(f"Target modules: {sorted(target_modules)}")
    print(f"Saved to {output_path}")

    return output_path


def push_to_hub(output_dir: str, repo_id: str):
    """Push the converted adapter to HuggingFace Hub."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, exist_ok=True, repo_type="model")
    api.upload_folder(
        folder_path=output_dir,
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"Pushed to https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Convert MLX adapter to PEFT")
    parser.add_argument("--input", required=True, help="MLX adapter directory")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="Base model ID")
    parser.add_argument("--output", required=True, help="Output PEFT directory")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--push", type=str, default=None,
                        help="HuggingFace repo ID to push to")
    args = parser.parse_args()

    convert_mlx_to_peft(args.input, args.output, args.model, args.lora_rank)

    if args.push:
        push_to_hub(args.output, args.push)


if __name__ == "__main__":
    main()
