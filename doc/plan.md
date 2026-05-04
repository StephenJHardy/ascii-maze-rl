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

The prompt uses the model's **chat template** with a tight system prompt
containing an example output. This was discovered in Phase 2 to be critical
— without the chat template, the model produces chatty explanations instead
of move sequences.

**System prompt:**
```
You solve mazes. Output ONLY moves as space-separated letters.
Example output: d r r u d
```

**User message:** just the maze grid, nothing else.

**Full prompt (via chat template):**
```
<|im_start|>system
You solve mazes. Output ONLY moves as space-separated letters.
Example output: d r r u d<|im_end|>
<|im_start|>user
# # # # # # #
> . . # . . #
# # . # . # #
# . . . . # #
# . # # . . #
# . . . # . >
# # # # # # #<|im_end|>
<|im_start|>assistant
```

This design was validated in Phase 2: without the chat template and tight
system prompt, the model echoed instructions, gave explanations, or produced
comma/bracket-wrapped moves. With it, valid space-separated move sequences
appear from the very first generation, giving GRPO immediate reward signal.

CoT prompting is a potential Phase 5 extension for larger models or harder
mazes.

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

### 5.1 Framework: Custom GRPO on MLX

We use a **custom GRPO implementation** built directly on MLX primitives
(`mlx_lm` for generation, `nn.value_and_grad` for training). MLX-Tune's
GRPOTrainer was found to be non-functional — it computes the loss but never
performs a gradient update (see Phase 2 notes).

TRL (Hugging Face) is the most mature GRPO implementation, but its vLLM
backend requires CUDA (unavailable on Mac). If we move to a cloud GPU, TRL
becomes the right choice.

**Key implementation details discovered during Phase 2:**

1. **LoRA freeze/unfreeze for quantized models:** After `model.freeze()` and
   `lora_module.unfreeze()`, the quantized base weights inside each
   `LoRALinear` are also unfrozen. This causes `QuantizedMatmul::vjp` to
   fail. Fix: re-freeze `m.linear.freeze()` inside each LoRA module.

2. **Qwen3.5 has CustomKernel ops without VJP:** The hybrid attention in
   Qwen3.5 uses custom Metal kernels that don't support backpropagation.
   Qwen2.5 uses standard attention and works fine with 4-bit LoRA training.

3. **Chat template is essential:** Without it, the model treats the prompt
   as arbitrary text to continue rather than an instruction to follow,
   producing chatty explanations instead of move sequences.

The GRPO loop:

1. For each prompt, sample G completions from the current policy (no gradient)
2. Score each completion with the reward function
3. Compute group-relative advantages: `A_i = (r_i - mean(r)) / std(r)`
4. Re-score completions under current policy (differentiable forward pass)
5. Policy gradient loss: `-mean(advantage * log_prob)`
6. Update LoRA parameters via Adam

### 5.2 Base Model

The model choice is configurable — all training scripts accept a model ID
argument. Our primary choice is informed by smoke testing three candidates:

| Model | Throughput | Maze output behaviour |
|-------|-----------|----------------------|
| Qwen2.5-0.5B-Instruct | 107 tok/s | Outputs moves immediately (mixed case) |
| Qwen3-0.6B | 108 tok/s | Burns tokens on `<think>` block, never reaches moves |
| **Qwen3.5-0.8B** | **60 tok/s** | **Outputs lowercase moves directly, cleanest tokenization** |

**Primary: Qwen2.5-0.5B-Instruct** (`mlx-community/Qwen2.5-0.5B-Instruct-4bit`)
- 4-bit quantized LoRA training works (gradient step in 0.6s)
- ~107 tok/s generation, ~2s per GRPO step (8 × 20 tokens)
- Standard attention architecture — full VJP support for backpropagation
- Proven in Phase 2 overfit test: converged in 50 steps

**Not viable for training (but fine for inference):**
- Qwen3.5-0.8B — `CustomKernel` ops in hybrid attention lack VJP support,
  blocking gradient computation. Works for generation/inference only.
- Qwen3-0.6B — burns tokens on `<think>` blocks, never outputs moves

**Scale-up option:**
- Qwen2.5-1.5B-Instruct — more capacity for harder mazes, same architecture

### 5.3 SFT Warm-Start

Phase 2 showed SFT is not needed for **format compliance** — the chat
template handles that. But Phase 3b showed SFT IS needed for **maze
awareness** — the model must learn to condition its output on the maze
structure before GRPO can refine the policy.

Without SFT, GRPO collapses to a single fixed output regardless of the
maze because the policy gradient gives contradictory signals across
different maze structures.

The SFT stage uses the same LoRA configuration as GRPO (same rank, same
target modules) and trains on solved maze examples using the mlx-lm native
trainer. The resulting LoRA adapter is then loaded as the starting point
for GRPO training.

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
│   ├── maze_gen.py             # Maze generation (Wilson's + DFS) + solving (BFS)
│   ├── maze_repr.py            # Expanded grid rendering, prompt formatting
│   ├── maze_verify.py          # Move parsing, path simulation
│   ├── reward.py               # Reward function for GRPO
│   ├── maze_dataset.py         # MazeRecord/MazeDataset with JSONL serialization
│   ├── maze_census.py          # Maze enumeration and distribution analysis
│   ├── dataset_builder.py      # Generate base maze dataset from config
│   ├── make_eval_splits.py     # Generate eval splits (disjoint seeds)
│   ├── train_sft.py            # SFT LoRA fine-tuning (mlx-lm native trainer)
│   ├── train_grpo.py           # Custom GRPO training loop (MLX)
│   ├── evaluate.py             # Evaluation script (solve rate by size/difficulty)
│   ├── smoke_test.py           # Phase 0 framework validation
│   ├── build_rollout_viewer.py # Interactive GRPO rollout explorer (HTML)
│   ├── rollout_capture.py      # Capture rollout data for the viewer
│   └── convert_adapter.py      # Convert CUDA adapters to MLX format
├── notebooks/
│   ├── maze_grpo_workshop.ipynb # T4/Colab GRPO workshop (5×5/6×6)
│   ├── maze_grpo_mac.ipynb     # Mac/MLX GRPO workshop (3×3–5×5)
│   ├── maze_grpo_h100.ipynb    # H100 GRPO on 8×8/9×9
│   └── maze_sft_training.ipynb # SFT training (produces base models)
├── configs/
│   └── pregenerated_rollouts.json  # Pre-generated rollouts for reward exploration
├── data/                       # Generated datasets (gitignored, regenerable)
├── checkpoints/                # Model checkpoints (gitignored)
├── results/                    # Evaluation outputs
├── tests/
│   ├── test_maze_gen.py
│   ├── test_maze_repr.py
│   ├── test_maze_verify.py
│   ├── test_reward.py
│   ├── test_maze_dataset.py
│   └── test_evaluate.py
├── EXERCISE.md                 # Mac CLI exercise guide (legacy)
├── mac_workshop.md             # Mac CLI workshop guide
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

### Phase 2: The Overfit Test ✅

Can GRPO force the model to memorize a single 3×3 maze? **Yes — in 50
steps (~97 seconds).**

- [x] Implemented custom GRPO training loop (`train_grpo.py`) — MLX-Tune's
      GRPOTrainer was found to be non-functional (computes loss but never
      updates weights)
- [x] Discovered and fixed LoRA freeze bug for quantized models
      (`m.linear.freeze()` inside each LoRA module)
- [x] Discovered Qwen3.5 has non-differentiable CustomKernel ops — switched
      to Qwen2.5-0.5B-Instruct (4-bit) which has full VJP support
- [x] Discovered chat template + tight system prompt eliminates chattiness
      (the model outputs valid move sequences from step 1)
- [x] **Overfit test passed:** perfect reward (1.0) on all 8 generations
      from step 50 onward, stable through step 200+

**Training curve (single 3×3 maze, solution: `d r r d`):**

| Step | Mean Reward | Max Reward | Best Output |
|------|-----------|-----------|-------------|
| 1    | 0.015     | 0.745     | `d r r u d d r d r r u` (random valid moves) |
| 10   | 0.251     | 0.707     | `d r r u d d r r u d d r r u d` (too long) |
| 30   | 0.690     | 0.920     | `d r r d d` (almost right, 1 extra move) |
| 50   | 0.922     | **1.000** | `d r r d` (perfect!) |
| 60+  | **1.000** | **1.000** | `d r r d` (all 8 generations correct) |

**Key learnings:**

1. **Prompt engineering was more important than SFT warm-start.** The system
   prompt `"You solve mazes. Output ONLY moves as space-separated letters.
   Example output: d r r u d"` plus the chat template eliminated format
   compliance issues entirely. The model produced valid move sequences from
   step 1, giving GRPO meaningful reward signal immediately.

2. **MLX-Tune's GRPO is broken** but MLX itself works perfectly for custom
   RL. The core primitives (`mlx_lm.generate`, `nn.value_and_grad`,
   `optim.Adam`) are solid — we just needed ~150 lines of custom code.

3. **4-bit quantized training works** with the freeze fix. ~2s per GRPO step
   (8 generations × 20 tokens + gradient update). No need for bf16.

4. **Config that worked:** Qwen2.5-0.5B-Instruct-4bit, LoRA r=16 on
   attention projections, lr=1e-5, temperature=1.0, 8 generations,
   20 max tokens.

### Phase 3: Multi-Maze Training + Evaluation (2–3 days)

The overfit test proved the pipeline works on a single maze. The core
question now: **does learning generalize across mazes?** We train on
multiple mazes and evaluate on held-out ones.

At ~2s/step, 2,000 steps ≈ 67 minutes — fast enough to iterate on
hyperparameters and curriculum within a day.

**3a: Evaluation tooling** ✅
- [x] Implement `evaluate.py` — scores a checkpoint (base or LoRA) against
      a maze dataset, reports solve rate by size and difficulty
- [x] Implement `make_eval_splits.py` — generates eval sets with disjoint
      seeds (eval_3x3: 192 mazes, eval_small: 200, eval_full: 200)
- [x] 13 new tests (102 total, all passing)

**3b: Multi-maze 3×3 training — SFT then GRPO**

Pure GRPO on multiple mazes failed to converge (3 runs attempted). The
model collapses to a single fixed output sequence regardless of the maze.
Root cause: vanilla policy gradient gives contradictory gradients across
different mazes — `r r d` is reinforced on mazes where it happens to work,
then penalized on mazes where it doesn't, leading to oscillation.

Adding a KL penalty (with proper frozen reference model) prevents garbage
output but doesn't solve the conditioning problem — the model needs to learn
to *read the maze* before GRPO can refine its solving strategy.

**Approach: SFT warm-start then GRPO**

1. SFT on solved examples teaches maze-awareness (read grid → output path)
2. GRPO then refines the policy for accuracy and efficiency

This mirrors the AlphaMaze approach (SFT to 86%, GRPO to 93%).

- [x] Built `train_sft.py` — LoRA fine-tune using mlx-lm native trainer
- [x] SFT on 1,000 solved 3×3 examples: 200 iterations, 2 minutes,
      train loss 4.479 → 0.169, val loss 4.479 → 0.164
- [x] **SFT eval on eval_3x3: 55.7% solve rate** (107/192), 100%
      parseable, 0.76 mean progress. Model correctly conditions output
      on maze structure (different mazes → different solutions).
- [x] Applied GRPO on SFT checkpoint: 500 steps, ~23 min, lr=5e-6, beta=0.1
- [x] **SFT+GRPO eval on eval_3x3: 79.2% solve rate** (152/192), up from
      55.7% SFT-only. Mean reward 0.845, mean progress 0.913.

**Results summary for 3×3 mazes:**

| Stage | Solve Rate | Mean Reward | Mean Progress |
|-------|-----------|-------------|---------------|
| Base model (no training) | ~0% | — | — |
| GRPO only (no SFT) | collapsed | — | — |
| SFT only (200 iters) | 55.7% | 0.653 | 0.760 |
| **SFT + GRPO (500 steps)** | **79.2%** | **0.845** | **0.913** |

GRPO provided a +23.5pp improvement over SFT alone, confirming the
two-stage approach works. The remaining 20.8% failures are mostly on
longer-path mazes (6–8 moves) where the model confuses similar structures.

**3c: Scaling to larger mazes — initial results**

Trained on mixed 3×3/4×4/5×5 data (500 examples each, 1500 total).
SFT 400 iterations (~6 min), GRPO 500 steps (~28 min).

- [x] Generated mixed training data (train_mixed.jsonl)
- [x] SFT on mixed data: val loss 5.5 → 0.36
- [x] GRPO on mixed SFT checkpoint: 500 steps

**Results by size (eval_full, SFT only → SFT+GRPO):**

| Size | SFT only | SFT + GRPO | Progress |
|------|---------|-----------|----------|
| 3×3  | 22.0%   | 50.0%     | 0.805    |
| 4×4  | 3.0%    | 8.0%      | 0.613    |
| 5×5  | 2.0%    | 0.0%      | 0.330    |
| 6×6  | 0.0%    | 0.0%      | 0.277    |
| 7×7  | 0.0%    | 0.0%      | 0.212    |

**Observations:**

- GRPO again roughly doubles what SFT provides (22→50% on 3×3, 3→8% on 4×4)
- The mixed SFT baseline is weaker than 3×3-only SFT (22% vs 55.7%) because
  training data is diluted across sizes
- 5×5+ shows zero solve rate — the model gets partway (33% progress on 5×5)
  but can't complete solutions
- The bottleneck is SFT quality: GRPO can refine but can't teach maze-solving
  from scratch

**3c.2: Scaling SFT — more data, more iterations**

The bottleneck was SFT quality, not GRPO. Scaled up to 7,500 examples
(2K each for 3×3–5×5, 1K for 6×6, 500 for 7×7) and 2000 iterations
with batch_size=8 (~90 min on Mac).

- [x] Generated train_large.jsonl (7,500 examples, 3×3–7×7)
- [x] SFT 2000 iters: val loss 3.3 → 0.256

**Large SFT results (eval_full):**

| Size | SFT 400iter/1.5K | SFT 2000iter/7.5K | Improvement |
|------|-----------------|-------------------|-------------|
| 3×3  | 22.0%           | **99.5%** (191/192) | +77.5pp   |
| 4×4  | 3.0%            | **68.0%**           | +65pp     |
| 5×5  | 2.0%            | **28.0%**           | +26pp     |
| 6×6  | 0.0%            | **3.3%**            | +3.3pp    |
| 7×7  | 0.0%            | 0.0%               | —         |

3×3 is essentially solved by SFT alone. 4×4 at 68% is a strong GRPO
baseline. 5×5 at 28% shows learning but needs refinement. The
`checkpoints/sft_large` adapter is the base for GRPO experiments.

**3c.3: GRPO on large SFT checkpoint**

Ran GRPO (500 steps, ~37 min) on sft_large with 4×4/5×5 training data.

- [x] GRPO training: frequently hitting 1.0 reward on training mazes
- [x] Evaluation on eval_full:

| Size | SFT only | SFT + GRPO | Change |
|------|---------|-----------|--------|
| 3×3  | 99.5%   | **100.0%** | +0.5pp |
| 4×4  | 68.0%   | **74.0%**  | +6pp   |
| 5×5  | 28.0%   | 18.0%     | -10pp  |
| 6×6  | 3.3%    | 3.3%      | —      |
| 7×7  | 0.0%    | 0.0%      | —      |

GRPO improved 4×4 but hurt 5×5. The 5×5 regression suggests GRPO's reward
signal at that solve rate is too noisy — it reinforces patterns that work
for 4×4 at the expense of 5×5. Possible mitigations: train GRPO on each
size separately, or use curriculum (GRPO on 4×4 first, then 5×5).

**3c.4: Longer SFT (5000 iterations)**

Val loss was still declining at 2000 iters, so ran 5000 iters with lr=5e-5
(~3.3 hours on Mac). Val loss: 3.3 → 0.197.

| Size | SFT 2000i | SFT 5000i | Change |
|------|----------|----------|--------|
| 3×3  | 99.5%    | **100%**  | +0.5pp |
| 4×4  | 68.0%    | **88.0%** | +20pp  |
| 5×5  | 28.0%    | **32.0%** | +4pp   |
| 6×6  | 3.3%     | **13.3%** | +10pp  |
| 7×7  | 0.0%     | **5.0%**  | +5pp   |

The 5000-iter SFT beats all previous GRPO runs — more supervised training
was more effective than RL at this scale. 4×4 is nearly solved (88%).
Val loss still declining, suggesting even more training would help.

- [x] SFT 5000 iters, val loss 0.197
- [x] Eval: 57.5% overall, 88% on 4×4, first 7×7 solve
- [x] GRPO 500 steps on 4×4/5×5 training data (~34 min)

| Size | SFT 5000i | SFT 5000i + GRPO | Change |
|------|----------|-----------------|--------|
| 3×3  | 100%     | 98.0%           | -2pp   |
| 4×4  | 88.0%    | **90.0%**       | +2pp   |
| 5×5  | 32.0%    | **34.0%**       | +2pp   |
| 6×6  | 13.3%    | 6.7%            | -6.6pp |
| 7×7  | 5.0%     | 5.0%            | —      |

GRPO gives a small lift on 4×4 and 5×5 (+2pp each) but regresses 3×3
and 6×6. At this SFT quality, GRPO's marginal contribution is small —
the SFT baseline is already strong and RL noise can hurt sizes outside
the GRPO training distribution.

**3c.5: Longer GRPO on larger mazes (1000 steps, 5×5–7×7)**

Ran 1000 GRPO steps with 16 generations, temp 1.2, max_tokens 60 on
5×5/6×6/7×7 data (~3.6 hours on Mac).

| Size | SFT 5Ki | GRPO 500s (5-7) | GRPO 1000s (5-7) |
|------|---------|----------------|-----------------|
| 3×3  | 100%    | 100%           | 98%             |
| 4×4  | 88%     | 88%            | 88%             |
| 5×5  | 32%     | 30%            | 26%             |
| 6×6  | 13.3%   | 3.3%           | **20.0%**       |
| 7×7  | 5%      | 0%             | 0%              |

The 1000-step run pushed 6×6 from 13.3% to 20.0% — the best 6×6 result.
The improvement came between step 500 and 1000 (3.3% → 20%), showing that
longer GRPO runs help on harder mazes where the model needs more exploration.
5×5 regressed, consistent with the pattern of GRPO improving target sizes
while regressing others it's not trained on.

**3d: CUDA/H100 SFT Results**

Trained SFT on H100 with TRL's SFTTrainer — 50K+ examples, bf16 precision,
5000 steps with cosine LR schedule. Dramatically better than Mac 4-bit results.

| Size | Mac SFT (4-bit, 5Ki) | CUDA SFT (bf16, 5Ki/50K) |
|------|---------------------|--------------------------|
| 3×3  | 100%                | **100%**                 |
| 4×4  | 88%                 | **100%**                 |
| 5×5  | 32%                 | **100%**                 |
| 6×6  | 13.3%               | **98%**                  |
| 7×7  | 5%                  | **90%**                  |
| 8×8  | —                   | **0%**                   |
| 9×9  | —                   | **0%**                   |

Key findings:
- bf16 precision + more data effectively solves 3×3 through 7×7
- Sharp cliff at 8×8: model outputs valid moves but hits walls after
  4-8 steps. It understands the task but makes navigation errors at
  decision points in larger mazes.
- The 8×8 failure mode is ideal for GRPO: partial solutions with room
  for improvement through exploration.

**Separate Colab SFT model:** A less-trained checkpoint will be pushed
to HuggingFace for T4 participants, with SFT quality tuned so 5×5/6×6
have room for GRPO improvement (~60-80% on 5×5, ~30-50% on 6×6).

### Phase 4: Workshop Setup

Three-tier workshop structure, all running the same core exercise
(reward function design + GRPO rollout exploration) at different scales.

**Notebooks & guides:**
- `notebooks/maze_sft_training.ipynb` — SFT training (run once to
  produce base models for the workshop)
- `notebooks/maze_grpo_h100.ipynb` — H100 GRPO on 8×8/9×9 (projected
  at front of room)
- `notebooks/maze_grpo_workshop.ipynb` — T4/Colab GRPO on 5×5/6×6
  (participants on Windows/old Macs)
- `notebooks/maze_grpo_mac.ipynb` — Mac/MLX notebook mirroring the
  Colab workshop flow: reward exploration with pre-generated rollouts,
  MLX GRPO training, and evaluation
- `mac_workshop.md` — command-line guide for Mac users who prefer
  working outside notebooks (same exercise, same flow)

**Pre-trained models on HuggingFace:**
- `StephenJHardy/maze-cuda-sft-qwen2.5-0.5b` — strong SFT (100% to
  7×7, 0% on 8×8) for H100 GRPO
- A separate Colab-specific model tuned for T4 performance on 5×5/6×6

**Workshop flow:**
1. Intro: show the maze format, the SFT baseline, what the model can/can't do
2. Exercise: design a reward function (start with binary → iterate to partial credit)
3. Run GRPO with their reward function (~8-10 min per run)
4. Explore rollouts in the viewer — see which rollouts GRPO reinforces
5. Iterate on reward function, observe improvements
6. Discussion: GRPO's group-relative advantages vs value models, credit
   assignment challenges

**Visualization tools:**
- [x] `src/build_viewer.py` — static eval result explorer
- [x] `src/build_rollout_viewer.py` — GRPO rollout explorer with:
  - All G rollouts per maze, side by side
  - Step-by-step path animation through the maze
  - Advantage visualization (which rollouts GRPO reinforces)
  - Reward decomposition (coverage, progress, valid steps)
  - GRPO mechanism explanation per maze
- [x] `src/rollout_capture.py` — captures full rollout data for the viewer

### Phase 5: Extensions (optional)

- [ ] Scale to Qwen2.5-1.5B-Instruct (more capacity for 8×8/9×9)
- [ ] Per-size GRPO to avoid cross-size regression
- [ ] Value model comparison — implement PPO with a value head to
      demonstrate per-token credit assignment vs GRPO's sequence-level
- [ ] Curriculum learning — automated difficulty scheduling
- [ ] Longer GRPO runs on H100 for 8×8/9×9 convergence

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

## 9a. Results Summary

### SFT Results

SFT (supervised fine-tuning on solved maze examples) was the primary driver
of maze-solving ability. More data and more iterations consistently improved
results:

| Size | SFT 400i/1.5K | SFT 2Ki/7.5K | SFT 5Ki/7.5K |
|------|--------------|-------------|-------------|
| 3×3  | 22%          | 99.5%       | **100%**    |
| 4×4  | 3%           | 68%         | **88%**     |
| 5×5  | 2%           | 28%         | **32%**     |
| 6×6  | 0%           | 3.3%        | **13.3%**   |
| 7×7  | 0%           | 0%          | **5%**      |

**Key SFT findings:**
- More training is the biggest lever — 5× more iterations gave +20pp on 4×4
  and +10pp on 6×6
- Val loss was still declining at 5000 iterations (0.197), suggesting further
  gains are possible with more compute
- 3×3 is solved by SFT alone; 4×4 is nearly solved (88%)
- The model learns to condition output on maze structure — different mazes
  get different solutions, not a fixed output

### GRPO Results

GRPO (Group Relative Policy Optimization) provides a modest improvement on
top of SFT, with important caveats:

**Best results by configuration:**

| Size | SFT only | Best SFT+GRPO | GRPO config |
|------|---------|--------------|-------------|
| 3×3  | 100%    | **100%**     | 3×3-only, 500 steps |
| 4×4  | 88%     | **90%**      | 4×4/5×5, 500 steps |
| 5×5  | 32%     | **34%**      | 4×4/5×5, 500 steps |
| 6×6  | 13.3%   | **20%**      | 5×5/6×6/7×7, 1000 steps |
| 7×7  | 5%      | 5%           | — |

**Key GRPO findings:**
- GRPO adds +2–7pp on the sizes it specifically trains on
- Longer runs help on harder mazes: 6×6 improved from 3.3% to 20% between
  GRPO step 500 and 1000
- GRPO consistently regresses sizes outside its training distribution
  (e.g. training on 5-7 hurts 5×5 while helping 6×6)
- At low SFT baselines, GRPO's contribution is larger (55.7% → 79.2% on
  3×3 with weak SFT); at high baselines, the marginal gain shrinks
- GRPO cannot substitute for SFT — without SFT warm-start, the model
  collapses to a fixed output regardless of maze structure

### Overall Assessment

The two-stage SFT → GRPO pipeline works, mirroring the AlphaMaze approach.
However, at this model scale (0.5B, 4-bit quantized) and compute budget
(Mac laptop), **SFT is the dominant factor** and GRPO provides refinement.
The levers for further improvement are:

1. **More SFT data and iterations** — the model hasn't saturated yet
2. **Larger model** (1.5B+) — more capacity for complex spatial reasoning
3. **Better hardware** — enables both of the above in reasonable time
4. **Per-size GRPO** — training each size independently to avoid regression

---

## 10. Risks and Mitigations

**Resolved risks** (from Phase 0–3b):

| Risk | Resolution |
|------|------------|
| GRPO can't bootstrap format | ✅ Solved: chat template + system prompt gives immediate format compliance |
| MLX-Tune GRPO has bugs | ✅ Solved: custom GRPO loop (~150 lines) |
| Model outputs gibberish | ✅ Solved: tight system prompt + example; KL penalty prevents drift |
| Training too slow on Mac | ✅ Solved: 4-bit quantized, ~2s/step |
| Qwen3.5 CustomKernel issue | ✅ Worked around: using Qwen2.5-0.5B instead |
| GRPO alone can't learn multi-maze | ✅ Diagnosed: needs SFT warm-start for maze conditioning |
| Policy collapse without KL | ✅ Solved: proper frozen reference model KL penalty |

**Active risks** (for further scaling):

| Risk | Impact | Mitigation |
|------|--------|------------|
| GRPO regresses non-target sizes | Medium | Per-size GRPO, or include all sizes in training data |
| 0.5B model capacity ceiling | Medium | 5×5 at 34%, 7×7 at 5% — may need 1.5B for larger mazes |
| Diminishing returns from more SFT | Medium | Val loss still declining but gains per iteration shrinking |

---

## 11. Open Questions

**Answered:**

1. ~~Can GRPO bootstrap format compliance?~~ **Yes** — chat template +
   system prompt with example. No SFT needed for format.

2. ~~Can GRPO alone learn multi-maze solving?~~ **No** — without SFT, the
   model collapses to a fixed output regardless of maze structure. The
   policy gradient gives contradictory signals across mazes. SFT
   warm-start is needed to teach maze-conditioned generation.

3. ~~Does the simplified policy gradient need KL?~~ **Yes** — without KL
   penalty the model drifts into garbage. With per-step KL (wrong
   reference) it stays clean but doesn't learn. Proper frozen-reference
   KL is implemented and prevents collapse.

4. ~~Does SFT teach enough maze awareness?~~ **Yes** — 200 iterations on
   1,000 examples reaches 55.7% solve rate on held-out 3×3 mazes. The
   model conditions output on maze structure (different solutions for
   different mazes). Failures are mostly on longer-path mazes.

5. ~~Can GRPO improve on SFT?~~ **Yes, modestly.** +2–7pp on target sizes.
   Larger effect at lower SFT baselines (+24pp at 55.7%), smaller at
   higher baselines (+2pp at 88%). Longer GRPO runs help on harder mazes
   (6×6: 13% → 20% at 1000 steps).

6. ~~How does performance scale with maze size?~~ **Answered.** 3×3 solved
   (100%). 4×4 nearly solved (90%). 5×5 partial (34%). 6×6 emerging (20%).
   7×7 marginal (5%). More SFT data is the primary lever.

7. ~~How much does more SFT data help?~~ **A lot.** 5× more iterations
   gave +20pp on 4×4, +10pp on 6×6, first 7×7 solve. Val loss still
   declining at 5000 iters — more compute would help.

**Still open:**

8. **What's the capacity ceiling of 0.5B?** Current best: 34% on 5×5,
   20% on 6×6. Is this a data/compute limit or a model capacity limit?

9. **Would per-size GRPO avoid cross-size regression?** Current GRPO
   runs improve target sizes but regress others.

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
