# ASCII Maze RL

Training a small LLM to solve ASCII mazes using SFT and GRPO (Group Relative
Policy Optimization). Runs entirely on a Mac laptop with Apple Silicon.

See [doc/plan.md](doc/plan.md) for the full project plan, methodology, and
results.

## Colab Walkthrough

For a Google Colab-friendly end-to-end walkthrough of the SFT → RL sequence,
see [notebooks/maze_sft_grpo_colab.ipynb](notebooks/maze_sft_grpo_colab.ipynb).
It reuses the repo's maze generation, prompt, reward, and evaluation logic,
but swaps the Apple-Silicon-only MLX trainers for a Colab-compatible
PyTorch/LoRA flow.

The notebook now calls the shared Python training API in `src.training`
instead of embedding trainer internals in notebook cells. That API chooses
`MLX` on Apple Silicon when available and otherwise falls back to
`PyTorch + LoRA`.

For RL reward experimentation, `src.training.train_rl(...)` accepts either:
- a Python callable `reward(completion: str, maze: Maze) -> float`
- an import string like `"my_rewards:shaped_reward"`

This keeps reward design exposed for teaching notebooks without duplicating
training logic across notebook cells.

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Quick Start

### 1. Generate data

```bash
# Base dataset (50K mazes, 3×3–9×9, ~10s)
uv run python -m src.dataset_builder

# Evaluation splits (disjoint seeds)
uv run python -m src.make_eval_splits
```

### 2. SFT training

```bash
# Train on the large dataset (7.5K examples, 3×3–7×7)
uv run python -m src.train_sft \
    --dataset data/train_large.jsonl \
    --val-dataset data/eval_full.jsonl \
    --iters 2000 \
    --batch-size 8 \
    --lr 1e-4 \
    --output-dir checkpoints/sft
```

### 3. GRPO training

```bash
# GRPO on top of an SFT checkpoint
uv run python -m src.train_grpo \
    --adapters checkpoints/sft \
    --dataset data/train_grpo_45.jsonl \
    --max-steps 500 \
    --num-generations 8 \
    --temperature 1.0 \
    --lr 5e-6 \
    --beta 0.1 \
    --max-tokens 40 \
    --output-dir checkpoints/grpo

# Single-maze overfit test (quick pipeline validation)
uv run python -m src.train_grpo --overfit --max-steps 200
```

### 4. Evaluate

```bash
# Evaluate base model (no adapters)
uv run python -m src.evaluate \
    --dataset data/eval_full.jsonl

# Evaluate an SFT or GRPO checkpoint
uv run python -m src.evaluate \
    --dataset data/eval_full.jsonl \
    --adapters checkpoints/sft \
    --output results/eval.json \
    --verbose

# Options
#   --temperature 0.0    greedy decoding (default, deterministic)
#   --temperature 0.7    sample for diversity
#   --samples 3          generate N completions per maze, take best
#   --max-tokens 60      max completion length (increase for larger mazes)
#   --limit 50           evaluate only first N mazes
```

### 5. Visualize results

```bash
# Build an interactive HTML explorer from eval results
uv run python -m src.build_viewer \
    --results results/eval.json \
    --dataset data/eval_full.jsonl \
    --output results/viewer.html

# Open in browser
open results/viewer.html
```

The viewer shows each maze with:
- The correct solution path and the model's attempted path overlaid on the grid
- Color-coded moves (green = valid, red = wall collision)
- Reward breakdown, progress metrics, and raw model output
- Filtering by size, solved/failed, and sorting by reward

## Other Utilities

```bash
# Smoke test: verify MLX-Tune works on your Mac
uv run python src/smoke_test.py
uv run python src/smoke_test.py mlx-community/Qwen2.5-0.5B-Instruct-4bit

# Maze census: enumerate unique mazes and solution length distributions
uv run python -m src.maze_census

# Write a custom dataset config, then generate from it
uv run python -m src.dataset_builder --write-config data/my_config.json
# (edit data/my_config.json)
uv run python -m src.dataset_builder --config data/my_config.json
```

## Running Tests

```bash
uv run pytest tests/ -v
```

## Project Structure

```
src/
  maze_gen.py           Maze generation (Wilson's algorithm) + solving (BFS)
  maze_repr.py          Expanded grid rendering, prompt formatting
  maze_verify.py        Move parsing, path simulation
  reward.py             Reward function for GRPO
  maze_dataset.py       MazeRecord/MazeDataset with JSONL serialization
  maze_census.py        Maze enumeration and distribution analysis
  dataset_builder.py    Generate base maze dataset from config
  make_eval_splits.py   Generate eval splits with disjoint seeds
  train_sft.py          SFT LoRA fine-tuning (mlx-lm native trainer)
  train_grpo.py         Custom GRPO training loop (MLX)
  evaluate.py           Evaluation (solve rate by size/difficulty)
  build_viewer.py       Build interactive HTML result explorer
  smoke_test.py         Framework validation
```
