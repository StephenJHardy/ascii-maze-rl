# ASCII Maze RL

Training a small LLM to solve ASCII mazes using SFT and GRPO (Group Relative
Policy Optimization).

See [doc/plan.md](doc/plan.md) for the full project plan, methodology, and
results.

## Workshop Exercise

Design a reward function that teaches GRPO to improve a maze-solving LLM.
Choose the notebook that matches your hardware:

| Notebook | Hardware | What it does |
|----------|----------|-------------|
| [Reward Design](notebooks/maze_reward_design.ipynb) | **Any computer** | Reward function design only — iterate on pre-generated rollouts with no GPU. Pure Python + numpy. |
| [Colab Workshop](notebooks/maze_grpo_workshop.ipynb) | **Colab T4 GPU** | Full exercise: reward design + GRPO training + evaluation. Reward design sections work without GPU. |
| [Mac Workshop](notebooks/maze_grpo_mac.ipynb) | **Mac (Apple Silicon)** | Full exercise using MLX on Mac: reward design + GRPO training + evaluation. |
| [Mac CLI Guide](mac_workshop.md) | **Mac (Apple Silicon)** | Same exercise as the Mac notebook but using the command line. |

**Quick links to open in Colab:**
- [Reward Design (no GPU)](https://colab.research.google.com/github/StephenJHardy/ascii-maze-rl/blob/main/notebooks/maze_reward_design.ipynb)
- [Full Workshop (T4 GPU)](https://colab.research.google.com/github/StephenJHardy/ascii-maze-rl/blob/main/notebooks/maze_grpo_workshop.ipynb)

## Local Setup (Mac)

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/StephenJHardy/ascii-maze-rl.git
cd ascii-maze-rl
uv sync
```

Then open `notebooks/maze_grpo_mac.ipynb` in Cursor/VS Code or run:

```bash
uv run jupyter notebook notebooks/maze_grpo_mac.ipynb
```

## Full Pipeline (for development)

### 1. Generate data

```bash
# Base dataset (50K mazes, 3×3–9×9, ~10s)
uv run python -m src.dataset_builder

# Evaluation splits (disjoint seeds)
uv run python -m src.make_eval_splits
```

### 2. SFT training

```bash
uv run python -m src.train_sft \
    --dataset data/base.jsonl \
    --val-dataset data/eval_full.jsonl \
    --iters 5000 \
    --batch-size 8 \
    --lr 5e-5 \
    --output-dir checkpoints/sft
```

### 3. GRPO training

```bash
uv run python -m src.train_grpo \
    --adapters checkpoints/sft \
    --dataset data/train_grpo_45.jsonl \
    --max-steps 500 \
    --num-generations 8 \
    --temperature 1.0 \
    --lr 5e-6 \
    --beta 0.04 \
    --max-tokens 40 \
    --output-dir checkpoints/grpo

# Single-maze overfit test (quick pipeline validation)
uv run python -m src.train_grpo --overfit --max-steps 200
```

### 4. Evaluate

```bash
uv run python -m src.evaluate \
    --dataset data/eval_full.jsonl \
    --adapters checkpoints/sft \
    --output results/eval.json \
    --verbose
```

### 5. Visualize results

```bash
uv run python -m src.build_viewer \
    --results results/eval.json \
    --dataset data/eval_full.jsonl \
    --output results/viewer.html

open results/viewer.html
```

## Running Tests

```bash
uv run pytest tests/ -v
```

## Project Structure

```
notebooks/
  maze_reward_design.ipynb   Reward design exercise (no GPU)
  maze_grpo_workshop.ipynb   Colab/T4 full workshop
  maze_grpo_mac.ipynb        Mac/MLX full workshop
  maze_grpo_h100.ipynb       H100 workshop (8×8/9×9)
  maze_sft_training.ipynb    SFT training notebook
src/
  maze_gen.py                Maze generation (Wilson's algorithm) + solving (BFS)
  maze_repr.py               Expanded grid rendering, prompt formatting
  maze_verify.py             Move parsing, path simulation
  reward.py                  Reward function for GRPO
  maze_dataset.py            MazeRecord/MazeDataset with JSONL serialization
  dataset_builder.py         Generate base maze dataset from config
  make_eval_splits.py        Generate eval splits with disjoint seeds
  train_sft.py               SFT LoRA fine-tuning (mlx-lm native trainer)
  train_grpo.py              Custom GRPO training loop (MLX)
  evaluate.py                Evaluation (solve rate by size/difficulty)
  build_viewer.py            Build interactive HTML result explorer
  build_rollout_viewer.py    GRPO rollout explorer (HTML)
  rollout_capture.py         Capture rollout data for the viewer
  convert_adapter.py         Convert CUDA adapters to MLX format
configs/
  pregenerated_rollouts.json Pre-generated rollouts for reward exploration
```
