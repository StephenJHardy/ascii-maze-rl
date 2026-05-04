# Mac Workshop Guide (non-notebook)

Train a small LLM to solve ASCII mazes using GRPO on Apple Silicon.
This guide walks through the same exercise as the workshop notebooks,
using the command line and `src/reward.py` directly.

## Prerequisites

- Mac with Apple Silicon (M1+) and ≥16GB unified memory
- Python 3.12+, [uv](https://docs.astral.sh/uv/) installed

## Setup

```bash
git clone https://github.com/StephenJHardy/ascii-maze-rl.git
cd ascii-maze-rl
uv sync
```

## Step 1: Download the Pre-trained SFT Model

The base model is `mlx-community/Qwen2.5-0.5B-Instruct-4bit` (4-bit
quantized, fits in Apple Silicon unified memory). The SFT adapter was
trained on Mac with MLX and published to HuggingFace in PEFT format.

Download and convert the adapter to MLX LoRA format:

```bash
uv run python -c "
from pathlib import Path
from huggingface_hub import hf_hub_download
from safetensors.numpy import load_file, save_file

adapter_dir = Path('checkpoints/sft-adapter')
adapter_dir.mkdir(parents=True, exist_ok=True)

# Download PEFT adapter
peft_path = hf_hub_download('StephenJHardy/maze-sft-qwen2.5-0.5b', 'adapter_model.safetensors')
peft_weights = load_file(peft_path)

# Convert PEFT -> MLX: transpose weights, fix key names
mlx_weights = {}
for key, value in peft_weights.items():
    mlx_key = key.replace('base_model.model.model.layers.', 'model.layers.')
    mlx_key = mlx_key.replace('.lora_A.weight', '.lora_a')
    mlx_key = mlx_key.replace('.lora_B.weight', '.lora_b')
    mlx_weights[mlx_key] = value.T

save_file(mlx_weights, str(adapter_dir / 'adapters.safetensors'))
print(f'Converted {len(peft_weights)} weights -> {adapter_dir}')

# Cache the base model
from mlx_lm import load
load('mlx-community/Qwen2.5-0.5B-Instruct-4bit')
print('Base model cached.')
"
```

## Step 2: Explore the Maze Format

```bash
uv run python -c "
from src.maze_gen import generate
from src.maze_repr import to_str, solution_to_str
from src.maze_verify import extract_moves, simulate, reached_exit

maze = generate(4, 4, seed=42)
print('Maze (4x4):')
print(to_str(maze))
print(f'\nSolution: {solution_to_str(maze.solution_moves)}')
print(f'Moves: {len(maze.solution_moves)}')

# Simulate a move sequence
test_output = 'r r d d d r'
moves = extract_moves(test_output)
path = simulate(moves, maze)
print(f'\nTest output: {test_output!r}')
print(f'Parsed: {moves}')
print(f'Path: {path}')
print(f'Reached exit: {reached_exit(path, maze)}')
"
```

The model receives the maze grid and must output space-separated moves
(u/d/l/r). Entry is marked `>` on the left, exit is `>` on the right.

## Step 3: Evaluate the SFT Baseline

Generate eval data and check how the model performs before GRPO:

```bash
uv run python -m src.make_eval_splits
uv run python -m src.evaluate \
    --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
    --adapters checkpoints/sft-adapter \
    --dataset data/eval_small.jsonl \
    --temperature 0.0 \
    --max-tokens 40 \
    --verbose
```

Expected results: ~100% on 3×3, ~88% on 4×4, ~32% on 5×5.

## Step 4: Explore Reward Functions with Pre-generated Rollouts ⭐

This is the core exercise. Pre-generated rollouts (8 completions per maze
at multiple temperatures) let you iterate on reward functions instantly,
with no model inference needed.

### Score rollouts with the current reward function

```bash
uv run python -c "
import json, numpy as np
from src.maze_gen import Maze
from src.maze_verify import extract_moves, simulate, reached_exit, manhattan_progress
from src.reward import compute_reward

with open('configs/pregenerated_rollouts.json') as f:
    data = json.load(f)

mazes_meta = data['mazes']
completions = data['completions']

def reconstruct_maze(meta):
    walls = frozenset(frozenset(tuple(c) for c in w) for w in meta['walls'])
    return Maze(width=meta['width'], height=meta['height'], walls=walls,
                entry=tuple(meta['entry']), exit=tuple(meta['exit']),
                solution=tuple(tuple(p) for p in meta['solution_path']), seed=0)

print(f\"{'Temp':>4s} | {'Mean Reward':>11s} | {'Mean Std':>9s} | {'Solved':>8s} | {'Zero-var':>8s}\")
print('-' * 55)

for temp in data['temperatures']:
    stds, rewards_all = [], []
    solved, zero_var = 0, 0
    total = len(mazes_meta) * 8

    for idx, meta in enumerate(mazes_meta):
        maze = reconstruct_maze(meta)
        comps = completions[str(temp)][idx]
        rewards = [compute_reward(c, maze) for c in comps]
        std = np.std(rewards)
        stds.append(std)
        rewards_all.append(np.mean(rewards))
        if std < 0.005: zero_var += 1
        for c in comps:
            moves = extract_moves(c)
            if moves and reached_exit(simulate(moves, maze), maze):
                solved += 1

    print(f'{temp:4.1f} | {np.mean(rewards_all):>11.4f} | {np.mean(stds):>9.4f} | '
          f'{solved:>3d}/{total:<3d} | {zero_var:>3d}/{len(mazes_meta):<3d}')

print()
print('Good reward functions have:')
print('  - Mean Std > 0.15 at temp=1.0')
print('  - Few zero-variance mazes')
print('  - If ALL mazes are zero-var, GRPO learns nothing!')
"
```

### Edit the reward function

Open `src/reward.py` and edit `compute_reward()`. This is the function
the GRPO training loop calls.

**Start simple and iterate:**

**v1 — Naive binary (the default):**
```python
def compute_reward(completion: str, maze: Maze) -> float:
    moves = extract_moves(completion)
    if moves is None:
        return 0.0
    path = simulate(moves, maze)
    return 1.0 if reached_exit(path, maze) else 0.0
```
Run the scoring script above — you'll see many zero-variance groups
(all 0.0 or all 1.0). GRPO can't learn from these.

**v2 — Penalize gibberish:**
```python
if moves is None:
    return -1.0  # not 0.0
```

**v3 — Partial credit for progress:**
```python
coverage = min(valid_steps / optimal_len, 1.0)
progress = manhattan_progress(path[-1], maze.exit, maze.entry)
```

**v4 — BFS distance (best):**
Use `solve()` from `src.maze_gen` to compute actual maze distance
from the model's final position to the exit. See the full solution
in `src/reward.py` or `EXERCISE.md`.

Re-run the scoring script after each edit to check how your changes
affect variance and signal quality. **No training needed for this step.**

### Build the rollout viewer (optional)

Visualize how your reward function scores different rollouts:

```bash
uv run python -c "
import json
from src.maze_gen import Maze
from src.maze_verify import extract_moves, simulate, reached_exit, manhattan_progress
from src.reward import compute_reward
from src.rollout_capture import MazeRollouts, RolloutResult, compute_advantages, save_rollouts

with open('configs/pregenerated_rollouts.json') as f:
    data = json.load(f)

TEMP = '1.0'

def reconstruct_maze(meta):
    walls = frozenset(frozenset(tuple(c) for c in w) for w in meta['walls'])
    return Maze(width=meta['width'], height=meta['height'], walls=walls,
                entry=tuple(meta['entry']), exit=tuple(meta['exit']),
                solution=tuple(tuple(p) for p in meta['solution_path']), seed=0)

all_rollouts = []
for idx, meta in enumerate(data['mazes']):
    maze = reconstruct_maze(meta)
    comps = data['completions'][TEMP][idx]
    rollouts = []
    for c in comps:
        moves = extract_moves(c)
        path = simulate(moves or [], maze)
        rollouts.append(RolloutResult(
            completion=c, moves_parsed=moves,
            path=[list(p) for p in path],
            reward=compute_reward(c, maze),
            solved=reached_exit(path, maze),
            valid_steps=len(path)-1,
            progress=max(manhattan_progress(path[-1], maze.exit, maze.entry), 0.0),
        ))
    rewards = [r.reward for r in rollouts]
    advantages, mean_r, std_r = compute_advantages(rewards)
    sol_moves = meta['solution_moves']
    all_rollouts.append(MazeRollouts(
        maze_id=meta['id'], width=meta['width'], height=meta['height'],
        maze_str=meta['maze_str'], entry=meta['entry'], exit=meta['exit'],
        correct_path=meta['solution_path'],
        correct_moves=sol_moves.split() if isinstance(sol_moves, str) else sol_moves,
        solution_length=meta['solution_length'],
        rollouts=rollouts, advantages=advantages,
        reward_mean=mean_r, reward_std=std_r,
    ))
    print(f'  {meta[\"id\"]}: mean={mean_r:.3f} std={std_r:.3f}')

save_rollouts(all_rollouts, 'results/exercise_rollouts.json')
"

uv run python -m src.build_rollout_viewer \
    --rollouts results/exercise_rollouts.json \
    --output results/exercise_rollout_viewer.html

open results/exercise_rollout_viewer.html
```

The viewer shows all 8 rollouts per maze with paths on the grid,
step-by-step animation, and which rollouts GRPO would reinforce.

## Step 5: Generate GRPO Training Data

```bash
uv run python -c "
from src.maze_dataset import MazeDataset, DatasetConfig, SizeConfig

config = DatasetConfig(
    name='grpo_exercise',
    algorithm='wilson',
    sizes=[
        SizeConfig(width=4, height=4, count=400, start_seed=510_000),
        SizeConfig(width=5, height=5, count=400, start_seed=520_000),
    ],
)
ds = MazeDataset.generate(config, progress=False)
ds.save('data/train_grpo_exercise.jsonl')
print(ds.summary())
"
```

## Step 6: Run GRPO

```bash
uv run python -m src.train_grpo \
    --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
    --adapters checkpoints/sft-adapter \
    --dataset data/train_grpo_exercise.jsonl \
    --max-steps 500 \
    --num-generations 8 \
    --temperature 1.0 \
    --lr 1e-5 \
    --beta 0.04 \
    --max-tokens 30 \
    --output-dir checkpoints/grpo-exercise
```

This takes ~15-20 minutes on Apple Silicon. Watch the logs:
- `reward_mean` should increase over time
- `reward_max` hitting 1.0 means some rollouts solve the maze
- `reward_std > 0` means GRPO has variance to learn from

## Step 7: Evaluate After GRPO

```bash
uv run python -m src.evaluate \
    --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
    --adapters checkpoints/grpo-exercise/step-500 \
    --dataset data/eval_small.jsonl \
    --temperature 0.0 \
    --max-tokens 40 \
    --verbose
```

Compare with the baseline from Step 3. With a good reward function,
you should see improvements on 4×4 and 5×5.

## Step 8: Iterate

To try a different reward function:

1. Edit `src/reward.py`
2. Re-run the rollout scoring (Step 4) to check variance
3. Re-run GRPO (Step 6) — it starts fresh from the SFT checkpoint
4. Re-evaluate (Step 7)

You can also adjust GRPO hyperparameters:
- `--temperature` (higher = more exploration, noisier rollouts)
- `--num-generations` (more = better advantage estimates, slower)
- `--beta` (KL penalty — lower = more deviation from SFT allowed)

## Key Takeaways

1. **GRPO uses group comparison** — it doesn't need a value model, just
   relative ranking within rollout groups
2. **Reward design is critical** — binary rewards give sparse signal;
   partial credit enables smooth optimization
3. **The model needs to be "almost there"** — GRPO refines existing
   capability, it can't teach from scratch
4. **Credit assignment is per-sequence** — GRPO can't tell which move
   was wrong, only whether the whole attempt was better or worse
   (this motivates value models / process reward models)
