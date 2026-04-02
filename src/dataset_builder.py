"""
Dataset generation script.

Generates the base maze dataset from a config file (or a built-in default).
The config specifies how many mazes of each size to generate. The output is
a JSONL file in data/ containing all materialized maze records.

Usage:
    # Generate with default config
    uv run python -m src.dataset_builder

    # Generate from a custom config
    uv run python -m src.dataset_builder --config data/my_config.json

    # Write the default config to a file (for editing)
    uv run python -m src.dataset_builder --write-config data/config.json
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from src.maze_dataset import DatasetConfig, MazeDataset, SizeConfig

DEFAULT_CONFIG = DatasetConfig(
    name="base",
    algorithm="wilson",
    sizes=[
        SizeConfig(width=3, height=3, count=5_000, start_seed=0),
        SizeConfig(width=4, height=4, count=7_000, start_seed=100_000),
        SizeConfig(width=5, height=5, count=10_000, start_seed=200_000),
        SizeConfig(width=6, height=6, count=10_000, start_seed=300_000),
        SizeConfig(width=7, height=7, count=8_000, start_seed=400_000),
        SizeConfig(width=8, height=8, count=6_000, start_seed=500_000),
        SizeConfig(width=9, height=9, count=4_000, start_seed=600_000),
    ],
)

DATA_DIR = Path("data")


def main():
    parser = argparse.ArgumentParser(description="Generate maze dataset")
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config JSON file (uses default if not specified)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSONL path (default: data/{name}.jsonl)",
    )
    parser.add_argument(
        "--write-config", type=str, default=None,
        help="Write the default config to a file and exit",
    )
    args = parser.parse_args()

    if args.write_config:
        DEFAULT_CONFIG.save(args.write_config)
        print(f"Config written to {args.write_config}")
        return

    if args.config:
        config = DatasetConfig.load(args.config)
        print(f"Loaded config from {args.config}")
    else:
        config = DEFAULT_CONFIG
        print("Using default config")

    output_path = Path(args.output) if args.output else DATA_DIR / f"{config.name}.jsonl"

    print(f"Generating {config.total_count():,} mazes...")
    t0 = time.perf_counter()
    dataset = MazeDataset.generate(config)
    elapsed = time.perf_counter() - t0

    print(f"\nGenerated in {elapsed:.1f}s ({config.total_count() / elapsed:,.0f} mazes/sec)")
    print(dataset.summary())

    dataset.save(output_path)
    file_size = output_path.stat().st_size
    print(f"\nSaved to {output_path} ({file_size / 1_000_000:.1f} MB)")

    config_path = output_path.with_suffix(".config.json")
    config.save(config_path)
    print(f"Config saved to {config_path}")


if __name__ == "__main__":
    main()
