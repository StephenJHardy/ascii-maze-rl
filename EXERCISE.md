# GRPO Maze-Solving Exercise

Train a small LLM to solve ASCII mazes better using GRPO (Group Relative
Policy Optimization). The model has been pre-trained with SFT to understand
the task — your job is to design a reward function that helps GRPO push
it further.

## Prerequisites

- Mac with Apple Silicon (M1+) and 32GB unified memory
- Python 3.12+, [uv](https://docs.astral.sh/uv/) installed
- ~2GB disk space for the model

## Setup

```bash
git clone https://github.com/StephenJHardy/ascii-maze-rl.git
cd ascii-maze-rl
git checkout dev
uv sync
```

## Step 1: Download the Pre-trained Model

The SFT model has been trained to solve mazes but isn't perfect — it solves
~85% of 3×3, ~30% of 4×4, and ~4% of 5×5. GRPO will improve these.

```bash
uv run python -c "from mlx_lm import load; load('StephenJHardy/maze-cuda-sft-5000-qwen2.5-0.5b')"
```

Then convert to MLX format (one-time, ~30s):

```bash
uv run mlx_lm.convert \
    --hf-path StephenJHardy/maze-cuda-sft-5000-qwen2.5-0.5b \
    --mlx-path checkpoints/sft-mlx-bf16
```

## Step 2: Explore the Maze Format

```bash
uv run python -c "
from src.maze_gen import generate
from src.maze_repr import to_str, solution_to_str

maze = generate(4, 4, seed=42)
print('Maze:')
print(to_str(maze))
print(f'\nSolution: {solution_to_str(maze.solution_moves)}')
print(f'Moves: {len(maze.solution_moves)}')
"
```

The model receives the maze grid and must output space-separated moves
(u/d/l/r). Entry is top-left (>), exit is bottom-right (>).

## Step 3: Evaluate the Baseline

Generate eval data and check how the SFT model performs before GRPO:

```bash
uv run python -m src.make_eval_splits
uv run python -m src.evaluate \
    --model checkpoints/sft-mlx-bf16 \
    --dataset data/eval_small.jsonl \
    --temperature 0.0 \
    --max-tokens 40 \
    --verbose
```

You should see ~85% on 3×3, ~30% on 4×4.

## Step 4: Generate GRPO Training Data

```bash
uv run python -m src.dataset_builder \
    --config configs/train_grpo_exercise.config.json \
    --output data/train_grpo_exercise.jsonl
```

Or generate directly in Python:

```python
from src.maze_dataset import MazeDataset, DatasetConfig, SizeConfig

config = DatasetConfig(
    name="grpo_exercise",
    algorithm="wilson",
    sizes=[
        SizeConfig(width=3, height=3, count=200, start_seed=500_000),
        SizeConfig(width=4, height=4, count=400, start_seed=510_000),
        SizeConfig(width=5, height=5, count=200, start_seed=520_000),
    ],
)
ds = MazeDataset.generate(config, progress=False)
ds.save("data/train_grpo_exercise.jsonl")
print(ds.summary())
```

## Step 5: Design Your Reward Function ⭐

Edit `src/reward.py` — this is the core exercise. The reward function
scores model completions and provides the signal GRPO uses to improve.

**Start with the naive binary reward and observe it fails:**

```python
def compute_reward(completion: str, maze: Maze) -> float:
    moves = extract_moves(completion)
    if moves is None:
        return 0.0
    path = simulate(moves, maze)
    return 1.0 if reached_exit(path, maze) else 0.0
```

Run GRPO with this (Step 6) — you'll see it doesn't improve or even
regresses. Then iterate:

**v2: Add negative reward for gibberish**
```python
if moves is None:
    return -1.0  # not 0.0!
```

**v3: Add partial credit for valid moves**
```python
coverage = min(valid_steps / optimal_len, 1.0)
```

**v4: Add progress toward exit**
```python
progress = manhattan_progress(path[-1], maze.exit, maze.entry)
```

**v5: Full solution** (see `src/reward.py` for the complete version)

### Why Does Reward Design Matter?

GRPO generates multiple rollouts per maze and compares them:
- If all rollouts score the same (e.g., all 0.0) → advantages are zero → no learning
- If one rollout scores higher → it gets reinforced → policy improves
- Partial credit creates smooth gradients between "terrible" and "perfect"

## Step 6: Run GRPO

```bash
uv run python -m src.train_grpo \
    --model checkpoints/sft-mlx-bf16 \
    --dataset data/train_grpo_exercise.jsonl \
    --max-steps 500 \
    --num-generations 8 \
    --temperature 1.0 \
    --lr 1e-5 \
    --beta 0.04 \
    --max-tokens 30 \
    --output-dir checkpoints/grpo-exercise
```

This takes ~15-20 minutes on a Mac. Watch the reward metrics:
- `reward_mean` should increase over time
- `reward_max` hitting 1.0 means some rollouts solve the maze
- `reward_std > 0` means there's variance for GRPO to learn from

## Step 7: Evaluate After GRPO

```bash
uv run python -m src.evaluate \
    --model checkpoints/sft-mlx-bf16 \
    --adapters checkpoints/grpo-exercise/step-500 \
    --dataset data/eval_small.jsonl \
    --temperature 0.0 \
    --max-tokens 40 \
    --verbose
```

Compare with the baseline from Step 3. With a good reward function,
you should see improvements on 3×3 and 4×4.

## Step 8: Visualize Rollouts

Capture rollouts and build the interactive viewer:

```bash
uv run python -c "
from src.rollout_capture import capture_rollouts_pytorch, save_rollouts
from src.maze_dataset import MazeDataset, DatasetConfig, SizeConfig
from mlx_lm import load

# Load trained model (base + adapters would need MLX adapter loading)
# For visualization, use the base SFT model with temperature sampling
model, tokenizer = load('checkpoints/sft-mlx-bf16')

viz_config = DatasetConfig(
    name='viz',
    algorithm='wilson',
    sizes=[
        SizeConfig(width=3, height=3, count=5, start_seed=999_000),
        SizeConfig(width=4, height=4, count=10, start_seed=999_100),
        SizeConfig(width=5, height=5, count=5, start_seed=999_200),
    ],
)
viz_ds = MazeDataset.generate(viz_config, progress=False)

# Note: capture_rollouts_pytorch uses PyTorch — for MLX, use the
# generate_completion function from train_grpo.py directly
# This is a simplified version for the exercise
from src.maze_verify import extract_moves, simulate, reached_exit, manhattan_progress
from src.reward import compute_reward
from src.rollout_capture import MazeRollouts, RolloutResult, compute_advantages, score_completion, save_rollouts
from src.maze_repr import to_str, SYSTEM_PROMPT
from mlx_lm.sample_utils import make_sampler
import mlx_lm

sampler = make_sampler(temp=1.0)
all_rollouts = []
for record in viz_ds:
    maze = record.to_maze()
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': to_str(maze)},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    rollouts = []
    for _ in range(8):
        response = mlx_lm.generate(model, tokenizer, prompt=prompt, max_tokens=30, sampler=sampler)
        rollouts.append(score_completion(response, maze))
    rewards = [r.reward for r in rollouts]
    advantages, mean_r, std_r = compute_advantages(rewards)
    all_rollouts.append(MazeRollouts(
        maze_id=record.id, width=record.width, height=record.height,
        maze_str=record.maze_str, entry=list(maze.entry), exit=list(maze.exit),
        correct_path=[list(p) for p in maze.solution],
        correct_moves=list(maze.solution_moves),
        solution_length=len(maze.solution_moves),
        rollouts=rollouts, advantages=advantages,
        reward_mean=mean_r, reward_std=std_r,
    ))
    print(f'  {record.id}: mean={mean_r:.3f} std={std_r:.3f}')
save_rollouts(all_rollouts, 'results/exercise_rollouts.json')
"

uv run python -m src.build_rollout_viewer \
    --rollouts results/exercise_rollouts.json \
    --output results/exercise_rollout_viewer.html

open results/exercise_rollout_viewer.html
```

The viewer shows:
- All 8 rollouts per maze with paths on the grid
- Step-by-step animation of each rollout
- Reward and advantage for each (which ones GRPO reinforces)
- Why zero-variance groups produce no learning signal

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

## Bonus: SFT Training (Optional)

The SFT model was pre-trained for you, but if you want to understand
how it was created:

```bash
# Generate SFT training data (7500 solved examples)
uv run python -m src.dataset_builder

# Train SFT (takes ~90 min on Mac)
uv run python -m src.train_sft \
    --dataset data/base.jsonl \
    --iters 5000 \
    --batch-size 8 \
    --lr 5e-5 \
    --output-dir checkpoints/my-sft
```
