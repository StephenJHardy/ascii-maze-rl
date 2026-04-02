"""
Generate evaluation splits from disjoint seed ranges.

The base training dataset uses seeds starting at:
  3×3: 0, 4×4: 100000, 5×5: 200000, etc.

Eval splits use a separate seed range (starting at 900000+) to guarantee
no overlap with training data.

Usage:
    uv run python -m src.make_eval_splits
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.maze_dataset import DatasetConfig, MazeDataset, SizeConfig

EVAL_SEED_BASE = 900_000

EVAL_3X3 = DatasetConfig(
    name="eval_3x3",
    algorithm="wilson",
    sizes=[SizeConfig(width=3, height=3, count=192, start_seed=EVAL_SEED_BASE)],
)

EVAL_SMALL = DatasetConfig(
    name="eval_small",
    algorithm="wilson",
    sizes=[
        SizeConfig(width=3, height=3, count=100, start_seed=EVAL_SEED_BASE + 10_000),
        SizeConfig(width=4, height=4, count=100, start_seed=EVAL_SEED_BASE + 20_000),
    ],
)

EVAL_FULL = DatasetConfig(
    name="eval_full",
    algorithm="wilson",
    sizes=[
        SizeConfig(width=3, height=3, count=50, start_seed=EVAL_SEED_BASE + 30_000),
        SizeConfig(width=4, height=4, count=50, start_seed=EVAL_SEED_BASE + 40_000),
        SizeConfig(width=5, height=5, count=50, start_seed=EVAL_SEED_BASE + 50_000),
        SizeConfig(width=6, height=6, count=30, start_seed=EVAL_SEED_BASE + 60_000),
        SizeConfig(width=7, height=7, count=20, start_seed=EVAL_SEED_BASE + 70_000),
    ],
)

ALL_SPLITS = {
    "eval_3x3": EVAL_3X3,
    "eval_small": EVAL_SMALL,
    "eval_full": EVAL_FULL,
}

DATA_DIR = Path("data")


def main():
    parser = argparse.ArgumentParser(description="Generate evaluation splits")
    parser.add_argument(
        "--splits",
        nargs="*",
        default=list(ALL_SPLITS.keys()),
        choices=list(ALL_SPLITS.keys()),
        help="Which splits to generate (default: all)",
    )
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for name in args.splits:
        config = ALL_SPLITS[name]
        print(f"Generating {name} ({config.total_count()} mazes)...")
        dataset = MazeDataset.generate(config, progress=False)
        path = DATA_DIR / f"{name}.jsonl"
        dataset.save(path)
        config.save(path.with_suffix(".config.json"))
        print(f"  Saved to {path} ({path.stat().st_size / 1000:.0f} KB)")
        print(f"  {dataset.summary()}")
        print()


if __name__ == "__main__":
    main()
