# ASCII Maze RL

Training a small LLM to solve ASCII mazes using SFT and GRPO (Group Relative
Policy Optimization).

**Workshop exercise:** See [EXERCISE.md](EXERCISE.md) for the guided exercise
(Mac/Apple Silicon). For Colab/CUDA, see `notebooks/maze_grpo_workshop.ipynb`.

See [doc/plan.md](doc/plan.md) for the full project plan, methodology, and
results.

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Workshop Exercise (Mac)

Follow [EXERCISE.md](EXERCISE.md) for the step-by-step guided exercise.
The core flow:

1. Download pre-trained SFT model
2. Evaluate baseline performance
3. Design a reward function (the main exercise)
4. Run GRPO training
5. Evaluate improvement
6. Explore rollouts in the interactive viewer

## Quick Start (Scripts)

### Generate data

```bash
# Base dataset (50K mazes, 3×3–9×9, ~10s)
uv run python -m src.dataset_builder

# Evaluation splits (disjoint seeds)
uv run python -m src.make_eval_splits
```

### SFT training

```bash
uv run python -m src.train_sft \
    --dataset data/base.jsonl \
    --val-dataset data/eval_full.jsonl \
    --iters 2000 \
    --batch-size 8 \
    --lr 1e-4 \
    --output-dir checkpoints/sft
```

### GRPO training

```bash
# GRPO on top of an SFT model
uv run python -m src.train_grpo \
    --model checkpoints/sft-mlx-bf16 \
    --dataset data/train_grpo_exercise.jsonl \
    --max-steps 500 \
    --num-generations 8 \
    --temperature 1.0 \
    --lr 1e-5 \
    --beta 0.04 \
    --max-tokens 30 \
    --output-dir checkpoints/grpo

# Single-maze overfit test (quick pipeline validation)
uv run python -m src.train_grpo --model checkpoints/sft-mlx-bf16 --overfit --max-steps 50
```

### Evaluate

```bash
# Evaluate a model
uv run python -m src.evaluate \
    --model checkpoints/sft-mlx-bf16 \
    --dataset data/eval_small.jsonl \
    --verbose

# Evaluate with GRPO adapters
uv run python -m src.evaluate \
    --model checkpoints/sft-mlx-bf16 \
    --adapters checkpoints/grpo/step-500 \
    --dataset data/eval_small.jsonl \
    --verbose
```

### Visualize

```bash
# Build evaluation result viewer
uv run python -m src.build_viewer \
    --results results/eval.json \
    --dataset data/eval_small.jsonl \
    --output results/viewer.html

# Build rollout explorer (shows GRPO rollouts + advantages)
uv run python -m src.build_rollout_viewer \
    --rollouts results/rollouts.json \
    --output results/rollout_viewer.html
```

## Other Utilities

```bash
# Smoke test: verify MLX works on your Mac
uv run python src/smoke_test.py

# Maze census: enumerate unique mazes and solution length distributions
uv run python -m src.maze_census

# Generate data from a custom config
uv run python -m src.dataset_builder --write-config data/my_config.json
uv run python -m src.dataset_builder --config data/my_config.json
```

## Running Tests

```bash
uv run pytest tests/ -v
```

## Project Structure

```
EXERCISE.md             Guided workshop exercise (Mac)
notebooks/
  maze_sft_training.ipynb     SFT training (Colab/CUDA)
  maze_grpo_h100.ipynb        H100 GRPO demo (projected)
  maze_grpo_workshop.ipynb    T4/Colab participant exercise
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
  rollout_capture.py    Capture GRPO rollouts for visualization
  build_viewer.py       Build static eval result explorer
  build_rollout_viewer.py  Build interactive rollout explorer
  smoke_test.py         Framework validation
```
