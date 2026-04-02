"""
Maze dataset: generation, storage, and loading.

The dataset is a JSONL file where each line is a self-contained maze record
with all fields needed by the trainer, reward function, viewer, or analyser.

Since mazes are deterministic (width + height + seed + algorithm → unique maze),
the full dataset can be regenerated from a compact manifest. We store the
materialized JSONL for convenience but keep it gitignored.

Usage:
    # Build
    ds = MazeDataset.generate(config)
    ds.save("data/mazes.jsonl")

    # Load
    ds = MazeDataset.load("data/mazes.jsonl")
    record = ds[0]
    maze = record.to_maze()  # reconstruct the Maze object
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from src.maze_gen import Algorithm, Maze, generate
from src.maze_repr import solution_to_str, to_prompt, to_str


@dataclass
class MazeRecord:
    """A single maze entry with all fields materialized."""

    id: str
    width: int
    height: int
    seed: int
    algorithm: str
    prompt: str
    maze_str: str
    solution_moves: str
    solution_length: int
    solution_path: list[list[int]]
    entry: list[int]
    exit: list[int]
    walls: list[list[list[int]]]

    @staticmethod
    def from_maze(maze: Maze, algorithm: str = "wilson") -> MazeRecord:
        return MazeRecord(
            id=f"{maze.width}x{maze.height}_{maze.seed}",
            width=maze.width,
            height=maze.height,
            seed=maze.seed,
            algorithm=algorithm,
            prompt=to_prompt(maze),
            maze_str=to_str(maze),
            solution_moves=solution_to_str(maze.solution_moves),
            solution_length=len(maze.solution_moves),
            solution_path=[list(p) for p in maze.solution],
            entry=list(maze.entry),
            exit=list(maze.exit),
            walls=sorted([sorted([list(c) for c in w]) for w in maze.walls]),
        )

    def to_maze(self) -> Maze:
        """Reconstruct a Maze object from this record."""
        walls = frozenset(
            frozenset(tuple(c) for c in w) for w in self.walls
        )
        return Maze(
            width=self.width,
            height=self.height,
            walls=walls,
            entry=tuple(self.entry),
            exit=tuple(self.exit),
            solution=tuple(tuple(p) for p in self.solution_path),
            seed=self.seed,
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> MazeRecord:
        return MazeRecord(**d)


@dataclass
class SizeConfig:
    """How many mazes to generate at a given size."""

    width: int
    height: int
    count: int
    start_seed: int = 0


@dataclass
class DatasetConfig:
    """Configuration for generating a maze dataset."""

    sizes: list[SizeConfig]
    algorithm: str = "wilson"
    name: str = "mazes"

    def total_count(self) -> int:
        return sum(s.count for s in self.sizes)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "algorithm": self.algorithm,
            "sizes": [asdict(s) for s in self.sizes],
        }

    @staticmethod
    def from_dict(d: dict) -> DatasetConfig:
        return DatasetConfig(
            name=d.get("name", "mazes"),
            algorithm=d.get("algorithm", "wilson"),
            sizes=[SizeConfig(**s) for s in d["sizes"]],
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @staticmethod
    def load(path: str | Path) -> DatasetConfig:
        with open(path) as f:
            return DatasetConfig.from_dict(json.load(f))


@dataclass
class MazeDataset:
    """A collection of maze records with load/save/filter capabilities."""

    records: list[MazeRecord] = field(default_factory=list)
    config: DatasetConfig | None = None

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> MazeRecord:
        return self.records[idx]

    def __iter__(self):
        return iter(self.records)

    def filter_by_size(self, width: int, height: int) -> MazeDataset:
        filtered = [r for r in self.records if r.width == width and r.height == height]
        return MazeDataset(records=filtered)

    def filter_by_solution_length(self, min_len: int = 0, max_len: int = 1000) -> MazeDataset:
        filtered = [r for r in self.records if min_len <= r.solution_length <= max_len]
        return MazeDataset(records=filtered)

    @property
    def size_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in self.records:
            key = f"{r.width}x{r.height}"
            counts[key] = counts.get(key, 0) + 1
        return counts

    @property
    def solution_length_distribution(self) -> dict[int, int]:
        dist: dict[int, int] = {}
        for r in self.records:
            dist[r.solution_length] = dist.get(r.solution_length, 0) + 1
        return dist

    def save(self, path: str | Path) -> None:
        """Save as JSONL (one JSON object per line)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for record in self.records:
                f.write(json.dumps(record.to_dict()) + "\n")

    @staticmethod
    def load(path: str | Path) -> MazeDataset:
        """Load from JSONL."""
        records = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(MazeRecord.from_dict(json.loads(line)))
        return MazeDataset(records=records)

    @staticmethod
    def generate(config: DatasetConfig, progress: bool = True) -> MazeDataset:
        """Generate a dataset from a config."""
        algo = Algorithm(config.algorithm)
        records: list[MazeRecord] = []

        if progress:
            from tqdm import tqdm
            total = config.total_count()
            pbar = tqdm(total=total, desc=f"Generating {config.name}")

        for size_cfg in config.sizes:
            for i in range(size_cfg.count):
                seed = size_cfg.start_seed + i
                maze = generate(
                    size_cfg.width, size_cfg.height,
                    seed=seed, algorithm=algo,
                )
                record = MazeRecord.from_maze(maze, algorithm=config.algorithm)
                records.append(record)
                if progress:
                    pbar.update(1)

        if progress:
            pbar.close()

        return MazeDataset(records=records, config=config)

    def summary(self) -> str:
        """Human-readable summary of the dataset."""
        lines = [f"MazeDataset: {len(self.records)} records"]
        lines.append(f"  Sizes: {self.size_counts}")
        dist = self.solution_length_distribution
        if dist:
            lengths = sorted(dist.keys())
            lines.append(f"  Solution lengths: {lengths[0]}–{lengths[-1]} moves")
            avg = sum(k * v for k, v in dist.items()) / sum(dist.values())
            lines.append(f"  Mean solution length: {avg:.1f} moves")
        return "\n".join(lines)
