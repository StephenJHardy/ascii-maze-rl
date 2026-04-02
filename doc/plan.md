# Training a Small LLM to Solve ASCII Mazes via GRPO

## Overview

This project trains a small language model to solve ASCII mazes of variable
size using Group Relative Policy Optimization (GRPO). The base instruct model
is given a text maze and must output a sequence of moves. A verifiable reward
function scores the output, and GRPO optimizes the policy directly.

The entire training pipeline runs on a Mac laptop with 32GB unified memory
using MLX-Tune on Apple Silicon.

**Prior art:** [AlphaMaze](https://arxiv.org/abs/2502.14669) (Dao & Vu, 2025)
used SFT + GRPO with custom tokens on 5×5 mazes, reaching 93% accuracy with a
1.5B Qwen model. We skip custom tokens entirely — the maze is plain text. We
may optionally provide a lightweight SFT warm-start LoRA to bootstrap the
model into the correct output format before GRPO takes over (see Section 5.3).

**Development strategy:** Start with the smallest model (0.5B) and the
simplest mazes (3×3) to validate the pipeline end-to-end. Scale up model size
and maze complexity only after the core RL loop is proven to work.

---

## 1. Maze Representation

### 1.1 Design Constraints

The representation must:

1. **Tokenize cleanly** — each grid position should map to exactly one token,
   so the model can "see" individual cells without multi-character merges.
2. **Be visually parseable** — a human (or the model's pattern-matching) can
   read the maze structure directly from the text.
3. **Use only standard characters** — no special tokens added to the
   tokenizer.

### 1.2 Tokenizer Validation

We tested three candidate formats against the Qwen2.5 tokenizer:

| Format             | Tokens | Regularity                                       |
|--------------------|--------|--------------------------------------------------|
| Traditional ASCII  | 41     | Bad — `---`, `|       |` merge unpredictably      |
| Space-separated    | 49     | Excellent — every position is exactly 1 token     |
| Comma-separated    | 52     | Poor — commas merge inconsistently with chars     |

**Winner: space-separated character grid.** Each character is separated by a
space, producing a perfectly regular 1-token-per-position encoding.

### 1.3 The Expanded Grid Format

A W×H maze of logical cells maps to a (2W+1) × (2H+1) character grid where
walls and passages each occupy their own position:

- **Even row, even col** → corner (always `#`)
- **Even row, odd col** → horizontal wall (`#`) or passage (`.`)
- **Odd row, even col** → vertical wall (`#`) or passage (`.`)
- **Odd row, odd col** → cell interior (always `.`)

Characters:

| Char | Meaning                                  |
|------|------------------------------------------|
| `#`  | Wall or corner                           |
| `.`  | Open passage or cell interior            |
| `>`  | Entry point (left edge) or exit (right edge) |

Example — a 3×3 maze:

```
# # # # # # #
> . . # . . #
# # . # . # #
# . . . . # #
# . # # . . #
# . . . # . >
# # # # # # #
```

**Entry** is always top-left, entering from the left (the `>` on row 1).
**Exit** is always bottom-right, exiting to the right (the `>` on the last
data row). These positions are fixed for all mazes, so the model always knows
where to start and where to go.

### 1.4 Solution Format

The model outputs a sequence of single-character directions, space-separated:

```
r r d d l d r r
```

Where `u` = up, `d` = down, `l` = left, `r` = right. Each direction is one
logical cell move (not one grid position — the model thinks in terms of cells,
not the expanded grid).

This tokenizes perfectly: each direction is exactly 1 token.

### 1.5 Prompt Template

```
Solve this maze. Find a path from the entrance (>) on the
left side to the exit (>) on the right side.

Maze:
# # # # # # #
> . . # . . #
# # . # . # #
# . . . . # #
# . # # . . #
# . . . # . >
# # # # # # #

Output ONLY your sequence of moves (u/d/l/r), space-separated.
Moves:
```

The model generates everything after `Moves:`. The prompt deliberately asks
for **direct output only** — no chain-of-thought, no reasoning. This is
important for two reasons:

1. **Shorter rollouts** — CoT wastes generation budget on tokens that don't
   contribute to the solution, directly impacting training speed.
2. **Small model reliability** — sub-2B models tend to hallucinate coordinates
   when asked to reason spatially (e.g., "I am at (2,2), moving right to
   (3,4)..."). Forcing direct output lets GRPO optimize the model's internal
   latent reasoning rather than an unreliable token-level narration.

CoT prompting is a potential Phase 5 extension for larger models or harder
mazes, but the trial should validate the pipeline without it.

---

## 2. Maze Generation

### 2.1 Algorithm

We generate mazes using **Wilson's algorithm**, which produces a **uniform
random spanning tree** of the grid graph via loop-erased random walks. This
means every possible perfect maze is equally likely to be generated, giving
full coverage of the maze space (see Phase 1.1 census results).

Every generated maze is a perfect maze — exactly one path between any two
cells.

As an alternative, **randomized DFS** (recursive backtracker) is available
via the `Algorithm.DFS` option. DFS produces mazes with characteristically
long corridors but can only generate a small subset of all possible mazes
(e.g. 14 of 192 for 3×3).

### 2.2 Solving

We solve each maze with **BFS** from the entry cell to the exit cell to find
the shortest path. The solution is stored as a list of moves (`r`, `d`, `l`,
`u`) representing cell-to-cell transitions.

### 2.3 Maze Sizes

| Size | Grid dimensions | Approx. solution length | Difficulty |
|------|-----------------|------------------------|------------|
| 3×3  | 7×7 chars       | 4–6 moves              | Trivial    |
| 4×4  | 9×9 chars       | 6–10 moves             | Easy       |
| 5×5  | 11×11 chars     | 8–16 moves             | Medium     |
| 6×6  | 13×13 chars     | 10–22 moves            | Medium     |
| 7×7  | 15×15 chars     | 12–30 moves            | Hard       |
| 8×8  | 17×17 chars     | 14–40 moves            | Hard       |
| 9×9  | 19×19 chars     | 16–50 moves            | Very Hard  |

### 2.4 Entry and Exit

- **Entry:** always the left edge of cell (0, 0) — row 1, col 0 in the
  expanded grid.
- **Exit:** always the right edge of cell (H-1, W-1) — the last data row,
  last col in the expanded grid.

Both are marked with `>` in the grid.

### 2.5 Deterministic Generation

Each maze is generated from a `(size, seed)` pair, making the entire dataset
reproducible. Seeds are drawn from a master RNG initialized with a fixed
project seed.

---

## 3. Dataset Construction

### 3.1 Data Splits

| Split     | Size    | Content                  | Purpose                        |
|-----------|---------|--------------------------|--------------------------------|
| Train     | 50,000  | Prompts + metadata       | GRPO rollouts                  |
| Val       | 2,000   | Prompts + solutions      | Periodic evaluation            |
| Test      | 2,000   | Prompts + solutions      | Final held-out evaluation      |
| MazeBench | 300     | Curated by difficulty     | Comparable benchmark           |

No SFT data — we go straight to GRPO, so the training set contains only
prompts (the model generates its own completions during training). The
solutions are stored alongside for the reward function.

### 3.2 Size Distribution

Weighted toward smaller mazes for faster initial learning:

| Size | Train  | Val  | Test | MazeBench |
|------|--------|------|------|-----------|
| 3×3  | 5,000  | 200  | 200  | 20        |
| 4×4  | 7,000  | 300  | 300  | 30        |
| 5×5  | 10,000 | 400  | 400  | 60        |
| 6×6  | 10,000 | 400  | 400  | 60        |
| 7×7  | 8,000  | 300  | 300  | 50        |
| 8×8  | 6,000  | 200  | 200  | 40        |
| 9×9  | 4,000  | 200  | 200  | 40        |

MazeBench is further categorized:
- **Easy** (≤8 moves): 100 mazes
- **Medium** (9–18 moves): 100 mazes
- **Hard** (19+ moves): 100 mazes

### 3.3 Dataset Format

Each example is stored with the prompt and metadata needed by the reward
function:

```python
{
    "prompt": "Solve this maze. Find a path from ...\n\nMaze:\n# # # ...\n\nOutput ONLY ...\nMoves:",
    "maze_walls": [[...], ...],    # adjacency structure for reward fn
    "solution": "r r d d l d r r", # shortest path (for reference)
    "solution_length": 8,          # number of moves in shortest path
    "maze_size": 3,                # logical width/height
    "entry": [0, 0],               # entry cell coords
    "exit": [2, 2],                # exit cell coords
}
```

The dataset is saved as HuggingFace `datasets` Arrow format for direct
loading by the trainer.

### 3.4 Uniqueness

With a 3×3 maze there are only ~50 structurally distinct perfect mazes (not
counting rotations/reflections), so the 3×3 portion of the training set will
contain many duplicates. This is intentional — the model should memorize small
mazes easily, providing a foundation of reward signal before tackling larger
ones.

For 5×5 and above, the space of possible mazes is enormous (millions+), so
collisions are negligible.

---

## 4. Reward Function

The reward is **fully verifiable** — we simulate the model's moves through the
maze and check the result. No learned reward model is needed.

### 4.1 Credit Assignment in GRPO

GRPO assigns a **single scalar reward per complete generation**. All tokens in
that generation — including correct early moves — receive the same advantage
during the policy gradient update. This creates a credit assignment problem:
if a completion starts with 6 correct moves then makes 2 wrong ones, the
correct moves are "penalized" alongside the errors.

GRPO compensates through **inter-generation comparison**: for each prompt, G
completions are sampled and scored. The generation with 6/8 correct moves gets
a higher advantage than one with 2/8 correct moves. Over training, the policy
shifts toward trajectories with longer correct prefixes. This means the scalar
reward must be **smooth and monotonically increasing with partial correctness**
so that GRPO can discriminate between "almost right" and "completely wrong."

The reward function below is designed with this principle in mind: every
additional correct move increases the reward, so that within any group of G
generations, the one with the longest valid prefix always scores highest.

### 4.2 Reward Function

```python
def compute_reward(completion: str, maze_meta: dict) -> float:
    moves = extract_moves(completion)

    if moves is None:
        # No parseable moves at all — the model output conversational
        # filler, apologies, or gibberish. This MUST be negative, not
        # zero. If every generation in a group scores 0, the advantages
        # are flat and GRPO learns nothing. Negative rewards force the
        # policy to collapse away from chatty outputs and into the
        # desired action space (outputting move characters).
        return -1.0

    # Simulate: walk through the maze, stop at first wall collision
    path = simulate(moves, maze_meta)
    valid_steps = len(path) - 1  # moves that didn't hit a wall
    optimal_len = maze_meta["solution_length"]

    if reached_exit(path, maze_meta):
        # Solved! Reward in [0.6, 1.0] based on efficiency.
        # Optimal path gets 1.0; longer detours get less.
        efficiency = optimal_len / max(len(moves), optimal_len)
        return 0.6 + 0.4 * efficiency

    # --- Partial credit: reward the valid prefix ---

    # (a) Coverage: fraction of optimal path length traversed without
    #     hitting a wall. A model that takes 6 valid steps in a maze
    #     with optimal length 8 gets coverage = 0.75.
    #     Capped at 1.0 (valid detours beyond optimal length still count
    #     but don't over-reward).
    coverage = min(valid_steps / optimal_len, 1.0)

    # (b) Progress: how close the model's final valid position is to the
    #     exit, measured by manhattan distance improvement from entry.
    #     Rewards moves that head toward the goal, not just wander.
    progress = manhattan_progress(path[-1], maze_meta["exit"],
                                  maze_meta["entry"])

    # Combine: weight coverage higher than progress because coverage
    # directly measures "correct prefix length" while progress is a
    # softer heuristic (a model could gain progress via wrong paths).
    partial = 0.5 * (0.7 * coverage + 0.3 * progress)

    return partial  # range [0.0, 0.5)
```

### 4.3 Why Negative Rewards for Format Failures

An instruct model's default behavior is to be a helpful assistant — it will
output paragraphs of explanation, apologies, or conversational filler. Getting
the first positive reward signal requires the model to stumble into outputting
a valid sequence of `u d l r` characters purely through random exploration.

If unparseable outputs score 0.0 and all G generations in a group are
unparseable, every generation gets advantage ≈ 0 and GRPO makes no meaningful
update. The model stays stuck in "assistant mode" indefinitely.

Making format failures **strictly negative** (-1.0) breaks this deadlock:
even in a group where every generation is gibberish, any generation that
happens to contain a parseable move sequence (even one that immediately hits a
wall, scoring ~0.01) gets a large relative advantage. This rapidly collapses
the policy away from chatty outputs and into the correct action space.

### 4.4 Reward Curve

The reward is designed to be strictly increasing with the length of the valid
prefix, providing a smooth gradient for GRPO:

| Outcome                          | Score     | Example                    |
|----------------------------------|-----------|----------------------------|
| Unparseable / no moves           | -1.0      | "I can't solve this maze"  |
| All moves hit walls immediately  | ~0.01     | `l u u u` (all blocked)    |
| 2/8 valid moves, little progress | ~0.10     | Gets started, then stuck   |
| 5/8 valid moves, good progress   | ~0.25     | Halfway through the maze   |
| 7/8 valid moves, near the exit   | ~0.40     | Almost solved              |
| Solved, but twice optimal length | 0.80      | Correct but wandering      |
| Solved, optimal path             | 1.0       | Perfect                    |

The gap between -1.0 (gibberish) and 0.01 (at least tried valid moves) is
deliberately large. This creates a strong initial gradient that teaches the
model format compliance before it even needs to understand maze structure.

Within the positive range, the monotonic curve means that in any group of G
generations, GRPO can always rank them meaningfully. The generation with the
longest valid prefix gets the highest advantage, reinforcing correct early
moves.

### 4.4 Why Not Per-Token Rewards?

An alternative is to assign a reward to each token individually (a "process
reward model"). This would solve credit assignment perfectly — correct moves
get positive reward, incorrect moves get negative — but it requires:

- A value function (moving toward PPO, not GRPO)
- Per-step reward computation during the backward pass
- Significantly more implementation complexity

For this project, the scalar reward with coverage-based partial credit is the
right tradeoff. The key insight: **we don't need per-token rewards if the
scalar reward is granular enough for GRPO's group comparison to work.** As long
as "6/8 correct" always scores higher than "4/8 correct," GRPO's advantage
normalization handles the rest.

If we later find credit assignment is a bottleneck (the model learns correct
first moves but can't extend to full solutions), per-token rewards via PPO is
a natural extension — see Section 5.4 on swappable RL algorithms.

### 4.5 Progress Metric

`manhattan_progress` measures how far the model's final position is toward the
exit relative to the start, normalized to [0, 1]:

```python
def manhattan_progress(pos, target, origin):
    total = abs(target[0] - origin[0]) + abs(target[1] - origin[1])
    remaining = abs(target[0] - pos[0]) + abs(target[1] - pos[1])
    return 1.0 - remaining / max(total, 1)
```

This is a secondary signal — the primary partial credit comes from coverage
(valid prefix length). Progress biases toward moves that head toward the exit,
breaking ties between two paths of equal valid length where one makes more
directional progress.

### 4.6 Move Extraction

We search the model's output for the last sequence of valid direction
characters. This allows the model to "think" before answering:

```python
import re

def extract_moves(text: str) -> list[str] | None:
    # Find all sequences of space-separated u/d/l/r
    matches = re.findall(r'\b([udlr](?:\s+[udlr])+)\b', text)
    if not matches:
        # Try single move
        m = re.search(r'\b([udlr])\b', text)
        return [m.group(1)] if m else None
    # Take the last match (the final answer after any reasoning)
    return matches[-1].split()
```

### 4.7 Path Simulation

```python
DIRECTIONS = {"u": (-1, 0), "d": (1, 0), "l": (0, -1), "r": (0, 1)}

def simulate(moves: list[str], maze: dict) -> list[tuple[int, int]]:
    pos = tuple(maze["entry"])
    path = [pos]
    for move in moves:
        dr, dc = DIRECTIONS[move]
        next_pos = (pos[0] + dr, pos[1] + dc)
        if not can_move(pos, next_pos, maze):
            break  # hit a wall — stop
        pos = next_pos
        path.append(pos)
    return path
```

---

## 5. Training Framework

### 5.1 Framework: MLX-Tune

[MLX-Tune](https://github.com/ARahim3/mlx-tune) is the only viable choice for
iterative RL development on Mac. It runs natively on Apple Silicon via the MLX
framework, supports GRPO with LoRA/QLoRA, and has an Unsloth/TRL-compatible
API.

TRL (Hugging Face) is the most mature GRPO implementation, but its vLLM
backend requires CUDA (unavailable on Mac) and MPS has known LLVM bugs with
GRPO. CPU-only rollout generation is too slow for iterative debugging. If we
later move to a cloud GPU, TRL becomes the right choice — but for Mac
development, it's not a practical option.

**Memory estimate for Qwen3.5-0.8B with 4-bit quantization + LoRA:**

| Component                     | Estimate   |
|-------------------------------|------------|
| Model weights (4-bit)         | ~0.5 GB    |
| LoRA adapters (r=16)          | ~30 MB     |
| Optimizer states              | ~60 MB     |
| Activations + KV cache        | ~1–3 GB    |
| GRPO rollouts (8 generations) | ~2–4 GB    |
| **Total**                     | **~4–8 GB** |

Fits easily in 32GB unified memory. Rollout generation is the bottleneck
(~3.2s for 8 × 32-token generations), not memory.

```python
from mlx_tune import FastLanguageModel, GRPOTrainer, GRPOConfig

model, tokenizer = FastLanguageModel.from_pretrained(
    "mlx-community/Qwen3.5-0.8B-MLX-4bit",
    max_seq_length=512,
)

model = FastLanguageModel.get_peft_model(
    model, r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
)
```

**Escape hatch:** If MLX-Tune's GRPO path has bugs, we implement GRPO
directly in MLX. The algorithm is simple enough:

1. For each prompt batch, sample G completions from the current policy
2. Score each completion with the reward function
3. Compute group-relative advantages: `A_i = (r_i - mean(r)) / std(r)`
4. Compute clipped surrogate loss: `L = -min(ratio * A, clip(ratio, 1±ε) * A)`
5. Add KL penalty: `L += β * KL(π || π_ref)`
6. Update via gradient descent

This also enables swapping in REINFORCE (drop the clipping) or PPO (add a
value function head).

### 5.2 Base Model

The model choice is configurable — all training scripts accept a model ID
argument. Our primary choice is informed by smoke testing three candidates:

| Model | Throughput | Maze output behaviour |
|-------|-----------|----------------------|
| Qwen2.5-0.5B-Instruct | 107 tok/s | Outputs moves immediately (mixed case) |
| Qwen3-0.6B | 108 tok/s | Burns tokens on `<think>` block, never reaches moves |
| **Qwen3.5-0.8B** | **60 tok/s** | **Outputs lowercase moves directly, cleanest tokenization** |

**Primary: Qwen3.5-0.8B** (`mlx-community/Qwen3.5-0.8B-MLX-4bit`)
- Already outputs lowercase space-separated directions without training
- Cleanest tokenization: newlines are separate tokens (no merging)
- Larger vocab (248K) with dedicated single-char tokens
- ~60 tok/s throughput — 3.2s for 8 rollouts, viable for iteration

**Alternatives for comparison:**
- Qwen2.5-0.5B-Instruct — ~2× faster rollouts, good fallback if speed matters
- Qwen2.5-1.5B-Instruct — more capacity for harder mazes, use after pipeline validated

### 5.3 Optional SFT Warm-Start

If pure GRPO from the base instruct model proves too hard to bootstrap (the
model can't find its way to outputting valid move sequences), we prepare a
lightweight SFT LoRA on a small number of solved maze examples (~1,000). This
teaches the model the output format and basic maze awareness, giving GRPO a
head start.

This warm-start LoRA could also be useful as a teaching tool: provide
participants with a model that knows the format but solves mazes poorly
(~40% on 5×5), and let them use GRPO to push it further.

The SFT warm-start is **not part of the initial trial** — Phase 0 tests
whether pure GRPO works first.

### 5.4 Training Configuration

Two configuration profiles — a fast trial config for validating the pipeline,
and the full config for actual training runs.

**Trial config** (for Phase 2 overfit test and early debugging):

| Parameter              | Value                    |
|------------------------|--------------------------|
| Base model             | Qwen3.5-0.8B (4-bit)    |
| LoRA rank (r)          | 16                       |
| LoRA alpha             | 16                       |
| LoRA targets           | q/k/v/o/gate/up/down_proj|
| Learning rate          | 5e-6                     |
| Num generations (G)    | 8                        |
| Batch size             | 1 prompt                 |
| Gradient accumulation  | 1                        |
| Max completion length  | 32 tokens                |
| KL coefficient (β)    | 0.04                     |
| Clip range (ε)        | 0.2                      |
| Temperature            | 0.7                      |
| Max training steps     | 500                      |
| Checkpoint interval    | 50 steps                 |
| Eval interval          | 50 steps                 |

The 32-token completion length is sufficient for 3×3 and 4×4 solutions (≤10
moves). At ~60 tok/s, each GRPO step takes roughly 3–6s (dominated by rollout
generation). 500 steps ≈ 25–50 minutes.

**Full config** (once pipeline is validated):

| Parameter              | Value                    |
|------------------------|--------------------------|
| Base model             | Qwen3.5-0.8B (4-bit)    |
| LoRA rank (r)          | 16                       |
| LoRA alpha             | 16                       |
| LoRA targets           | q/k/v/o/gate/up/down_proj|
| Learning rate          | 5e-6                     |
| Num generations (G)    | 8                        |
| Batch size             | 2 prompts                |
| Gradient accumulation  | 4                        |
| Max completion length  | 128 tokens               |
| KL coefficient (β)    | 0.04                     |
| Clip range (ε)        | 0.2                      |
| Temperature            | 0.7                      |
| Max training steps     | 3,000                    |
| Checkpoint interval    | 200 steps                |
| Eval interval          | 200 steps                |

The 128-token completion length covers the longest expected solution (9×9
optimal path ~50 moves = ~50 tokens) with headroom. Rollout generation is
the primary bottleneck — the backward pass through LoRA adapters is
comparatively cheap.

### 5.4 Swappable RL Algorithms

The training code should make the RL algorithm pluggable:

```python
class RLAlgorithm(Protocol):
    def compute_loss(
        self,
        logprobs: Tensor,          # log π(a|s) for current policy
        ref_logprobs: Tensor,      # log π_ref(a|s) for reference policy
        advantages: Tensor,        # group-relative advantages
        old_logprobs: Tensor,      # log π_old(a|s) from rollout policy
    ) -> Tensor: ...

class GRPO(RLAlgorithm): ...       # clipped surrogate + KL
class REINFORCE(RLAlgorithm): ...  # vanilla policy gradient + baseline
class PPO(RLAlgorithm): ...        # needs value function head
```

GRPO and REINFORCE need no value function. PPO requires a value head (a small
MLP on the last hidden state) or a separate value model. This abstraction lets
us experiment without rewriting the rollout + training loop.

---

## 6. Monitoring and Evaluation

### 6.1 Training Metrics (every N steps)

| Metric                 | What it tells us                              |
|------------------------|-----------------------------------------------|
| `reward/mean`          | Overall quality of rollouts                   |
| `reward/std`           | Diversity — too low means mode collapse        |
| `reward/max`           | Best-case performance in batch                |
| `policy/kl`            | Divergence from base model                    |
| `policy/entropy`       | Exploration level                             |
| `policy/clipfrac`      | How often the clipping constraint activates   |
| `loss/policy`          | Training loss                                 |
| `train/lr`             | Learning rate schedule                        |
| `train/tokens_per_sec` | Throughput for pacing expectations             |

### 6.2 Evaluation Metrics (every 200 steps, on validation set)

| Metric               | Description                                     |
|-----------------------|-------------------------------------------------|
| `solve_rate`          | % of mazes solved (path reaches exit)           |
| `solve_rate/3x3`     | Solve rate for 3×3 mazes                        |
| `solve_rate/5x5`     | Solve rate for 5×5 mazes                        |
| `solve_rate/7x7`     | Solve rate for 7×7 mazes                        |
| `solve_rate/9x9`     | Solve rate for 9×9 mazes                        |
| `valid_move_rate`     | % of outputs containing parseable moves         |
| `wall_collision_rate` | % of valid-move outputs that hit a wall         |
| `avg_progress`        | Mean progress toward exit (0–1)                 |
| `optimal_ratio`       | Mean (model path length / optimal path length)  |

### 6.3 Logging Setup

**Weights & Biases** for real-time dashboards, rollout inspection, and
hyperparameter tracking:

```python
import wandb
wandb.init(project="ascii-maze-rl")

config = GRPOConfig(report_to="wandb", logging_steps=10, ...)
```

**TensorBoard** as a local-only alternative if W&B isn't desired.

### 6.4 Evaluation Callback

A callback runs evaluation at regular intervals during training:

```python
class MazeEvalCallback:
    def on_step_end(self, step, model, tokenizer):
        if step % 200 != 0:
            return
        metrics = evaluate(model, tokenizer, val_dataset)
        wandb.log(metrics, step=step)
        save_samples(model, tokenizer, sample_mazes, step=step)
```

### 6.5 Sample Output Logging

Every evaluation step, we save 10 model outputs for qualitative review:
- Inspect whether the model is generating valid move sequences vs. chatty
  refusals (critical early in training)
- Categorize failures: format error, wall collision, wrong direction,
  incomplete path
- Track how quickly the model transitions from gibberish (reward -1.0) to
  valid moves (reward >0)

### 6.6 Standalone Evaluation Script

```bash
uv run python src/evaluate.py \
    --model checkpoints/grpo-step-2000 \
    --dataset data/mazebench \
    --output results/grpo-2000.json \
    --verbose  # prints sample outputs
```

---

## 7. Project Structure

```
ascii-maze-rl/
├── doc/
│   └── plan.md                 # This document
├── src/
│   ├── maze_gen.py             # Maze generation (DFS) + solving (BFS)
│   ├── maze_repr.py            # Expanded grid rendering, prompt formatting
│   ├── maze_verify.py          # Move parsing, path simulation, scoring
│   ├── reward.py               # Reward function for GRPO
│   ├── dataset_builder.py      # Generate all data splits
│   ├── train_grpo.py           # GRPO training script
│   ├── evaluate.py             # Evaluation script
│   └── callbacks.py            # Training callbacks (eval, logging)
├── data/                       # Generated datasets (gitignored)
├── checkpoints/                # Model checkpoints (gitignored)
├── results/                    # Evaluation outputs
├── tests/
│   ├── test_maze_gen.py
│   ├── test_maze_repr.py
│   ├── test_maze_verify.py
│   └── test_reward.py
├── pyproject.toml
└── README.md
```

---

## 8. Implementation Milestones

### Phase 0: Smoke Test ✅

Validated MLX-Tune on target Mac with three model candidates.

- [x] Install MLX-Tune, verify it loads models on target Mac
- [x] Run basic text generation — confirmed ~60 tok/s (Qwen3.5-0.8B)
- [x] Verify LoRA adapter creation works
- [x] Estimate wall-clock time: 3.2s for 8 × 32-token rollouts

Selected Qwen3.5-0.8B as primary model (best format compliance, clean
tokenization). See `src/smoke_test.py` for the reusable test script.

### Phase 1: Maze Infrastructure ✅

- [x] Implement `maze_gen.py` — DFS generation, BFS solving
- [x] Implement `maze_repr.py` — expanded grid rendering, prompt formatting
- [x] Implement `maze_verify.py` — move parsing, path simulation
- [x] Implement `reward.py` — reward function (including negative rewards)
- [x] Write tests for all modules (72 tests, all passing)
- [x] Verify tokenization with Qwen3.5 tokenizer (1 token per grid position,
      1 token per move, all sizes 3×3 through 9×9)

### Phase 1.1: Maze Census & Generator Fix ✅

Investigated how many structurally unique perfect mazes exist at each size,
and discovered that the DFS backtracker only generates a tiny subset of them.

**Spanning tree counts** (verified via Kirchhoff's matrix-tree theorem):

| Size | Unique perfect mazes | DFS coverage (100K seeds) | Wilson coverage (100K seeds) |
|------|---------------------|--------------------------|------------------------------|
| 2×2  | 4                   | 4 (100%)                 | 4 (100%)                     |
| 3×3  | 192                 | 14 (7.3%)                | 192 (100%)                   |
| 4×4  | 100,352             | 322 (0.3%)               | 63,431 (63%)                 |
| 5×5  | 557,568,000         | 23,291 (sample)          | 99,993 (sample)              |
| 6×6  | 32.6 trillion       | —                        | —                            |

DFS backtracker can only produce DFS-trees rooted at (0,0) — a strict subset
of all spanning trees. Switched default generator to **Wilson's algorithm**
(loop-erased random walks), which samples spanning trees uniformly.

**Solution length distribution** (Wilson's / true distribution):

- Most mazes have short solutions: 88.5% of 3×3 mazes have the minimum
  4-move solution, 72.3% of 4×4 have minimum 6 moves.
- DFS backtracker was biased toward long corridors, making solutions appear
  more evenly distributed — this was misleading.
- The true distribution is good for training: the model encounters many easy
  wins for early reward signal, with hard mazes as rare challenges.

**Changes made:**
- [x] Added Wilson's algorithm to `maze_gen.py` (default generator)
- [x] Kept DFS as an option via `Algorithm.DFS` enum
- [x] Added `maze_census.py` for enumeration and distribution analysis
- [x] All 72 existing tests still pass

### Phase 1.2: Base Dataset ✅

Built the data layer: generation, serialization, and loading. Pulled
forward from Phase 3 since the dataset is needed before the overfit test.

- [x] Implement `maze_dataset.py` — `MazeRecord`, `MazeDataset`,
      `DatasetConfig` with JSONL serialization and filtering
- [x] Implement `dataset_builder.py` — CLI generation from config
- [x] Generate base dataset: 50K mazes (3×3–9×9) in 10s, 86.5 MB JSONL
- [x] 17 new tests (89 total, all passing)

**Storage decision:** regenerate, don't store. 86.5 MB is too large for
git, but 10s generation from a deterministic config is fast enough. The
config file goes in git, the materialized data stays gitignored.

### Phase 2: The Overfit Test (0.5–1 day)

The critical go/no-go gate: can GRPO force the model to memorize a single
3×3 maze?

- [ ] Create a minimal dataset of **1 maze** (one 3×3 maze, repeated)
- [ ] Wire up `train_grpo.py` with the trial config (Qwen3.5-0.8B,
      32 tokens, batch=1)
- [ ] Run GRPO for up to 500 steps on this single maze
- [ ] **Success criterion:** model reliably outputs the correct move sequence
      for this one maze within a few hundred steps
- [ ] If this fails: diagnose whether the problem is format (model doesn't
      output moves at all) or navigation (outputs moves but wrong ones)
- [ ] If format is the problem: build the SFT warm-start LoRA (Section 5.3)
      and retry

This test is fast (<30 min on the trial config) and tells us immediately
whether the approach is viable.

### Phase 3: Train/Val/Test Splits + Full Training (2–3 days)

- [ ] Generate train/val/test/mazebench splits from the base dataset
- [ ] Implement `callbacks.py` for periodic evaluation
- [ ] Run GRPO training with full config
- [ ] Start with 3×3 only, add 4×4 and 5×5 once 3×3 converges
- [ ] Target: >90% on 3×3, >60% on 5×5 by step 2000

### Phase 4: Evaluation (1 day)

- [ ] Implement `evaluate.py`
- [ ] Run MazeBench evaluation on best checkpoint
- [ ] Analyze results by size, difficulty, failure mode
- [ ] Save and inspect sample outputs

### Phase 5: Extensions (optional)

- [ ] Scale up to Qwen2.5-1.5B-Instruct
- [ ] Curriculum learning — progressively add larger maze sizes
- [ ] SFT warm-start LoRA for faster bootstrap
- [ ] Alternative RL algorithms (REINFORCE, PPO)
- [ ] Chain-of-thought prompting for harder mazes
- [ ] Cloud GPU run with TRL + vLLM for faster iteration

---

## 9. Dependencies

```toml
[project]
name = "ascii-maze-rl"
requires-python = ">=3.12"
dependencies = [
    "mlx-tune",
    "datasets",
    "transformers",
    "wandb",
    "numpy",
    "tqdm",
    "pytest",
]

[project.optional-dependencies]
# For cloud GPU training (not needed on Mac)
cloud = [
    "trl>=1.0",
    "accelerate",
    "peft",
    "bitsandbytes",
    "vllm",
]
```

---

## 10. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| GRPO can't bootstrap from base instruct model | High | Phase 2 overfit test catches this early; SFT warm-start LoRA as fallback |
| MLX-Tune GRPO has bugs | High | Custom GRPO loop in MLX (~100 lines); algorithm is simple |
| Model outputs gibberish, not moves | High | Negative reward (-1.0) for unparseable output creates strong format-learning signal |
| Larger mazes (7×7+) are too hard | Medium | Focus on 3×3–5×5 first; curriculum learning as extension |
| Training too slow on Mac for iteration | Medium | 0.5B model + 32-token completions for trial; scale up only after pipeline validated |
| 32GB insufficient for rollouts at scale | Low | Already using 0.5B + 4-bit; reduce G to 4 if needed |

---

## 11. Open Questions

1. **Can GRPO bootstrap from the base instruct model?** The Phase 2 overfit
   test answers this directly. If the model can't memorize one 3×3 maze
   within a few hundred steps, we need the SFT warm-start. The negative
   reward for gibberish should help collapse the model into the right
   action space quickly.

2. **How many GRPO steps are needed?** AlphaMaze needed 1,600 steps *after*
   full SFT. Without SFT (or with only a light warm-start), we likely need
   more — budget is 3,000 steps, evaluate frequently to track the learning
   curve.

3. **Does maze size require curriculum learning?** Training on all sizes
   simultaneously may dilute the reward signal from small mazes where the
   model can actually learn. Starting with 3×3 only and progressively adding
   larger sizes may be more effective.

4. **Where is the accuracy ceiling for 0.5B?** The 0.5B model has limited
   capacity — it may plateau at smaller mazes. This determines when to
   scale to 1.5B.

---

## 12. References

- **AlphaMaze**: Dao & Vu (2025). *AlphaMaze: Enhancing Large Language Models'
  Spatial Intelligence via GRPO.* [arXiv:2502.14669](https://arxiv.org/abs/2502.14669)
- **GRPO**: Shao et al. (2024). *DeepSeekMath: Pushing the Limits of
  Mathematical Reasoning.* [arXiv:2402.03300](https://arxiv.org/abs/2402.03300)
- **DeepSeek-R1**: Guo et al. (2025). *Incentivizing Reasoning Capability in
  LLMs via RL.* [arXiv:2501.12948](https://arxiv.org/abs/2501.12948)
- **TRL v1.0**: [huggingface.co/docs/trl](https://huggingface.co/docs/trl)
- **MLX-Tune**: [github.com/ARahim3/mlx-tune](https://github.com/ARahim3/mlx-tune)
