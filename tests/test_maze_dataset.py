"""Tests for maze dataset generation, serialization, and loading."""

import json
import tempfile
from pathlib import Path

import pytest

from src.maze_dataset import DatasetConfig, MazeDataset, MazeRecord, SizeConfig
from src.maze_gen import generate


@pytest.fixture
def small_config():
    return DatasetConfig(
        name="test",
        algorithm="wilson",
        sizes=[
            SizeConfig(width=3, height=3, count=10, start_seed=0),
            SizeConfig(width=4, height=4, count=5, start_seed=100),
        ],
    )


@pytest.fixture
def small_dataset(small_config):
    return MazeDataset.generate(small_config, progress=False)


class TestMazeRecord:
    def test_from_maze_roundtrip(self):
        maze = generate(3, 3, seed=42)
        record = MazeRecord.from_maze(maze)
        reconstructed = record.to_maze()
        assert reconstructed.width == maze.width
        assert reconstructed.height == maze.height
        assert reconstructed.walls == maze.walls
        assert reconstructed.entry == maze.entry
        assert reconstructed.exit == maze.exit
        assert reconstructed.solution == maze.solution

    def test_dict_roundtrip(self):
        maze = generate(5, 5, seed=7)
        record = MazeRecord.from_maze(maze)
        d = record.to_dict()
        restored = MazeRecord.from_dict(d)
        assert restored.id == record.id
        assert restored.width == record.width
        assert restored.solution_moves == record.solution_moves
        assert restored.walls == record.walls

    def test_json_roundtrip(self):
        maze = generate(4, 4, seed=99)
        record = MazeRecord.from_maze(maze)
        json_str = json.dumps(record.to_dict())
        restored = MazeRecord.from_dict(json.loads(json_str))
        assert restored.to_maze().walls == maze.walls

    def test_fields_populated(self):
        maze = generate(3, 3, seed=0)
        record = MazeRecord.from_maze(maze)
        assert record.id == "3x3_0"
        assert record.width == 3
        assert record.height == 3
        assert record.seed == 0
        assert record.algorithm == "wilson"
        assert len(record.prompt) > 0
        assert len(record.maze_str) > 0
        assert len(record.solution_moves) > 0
        assert record.solution_length > 0
        assert len(record.solution_path) > 1
        assert record.entry == [0, 0]
        assert record.exit == [2, 2]
        assert len(record.walls) > 0


class TestDatasetConfig:
    def test_total_count(self, small_config):
        assert small_config.total_count() == 15

    def test_save_load_roundtrip(self, small_config):
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            path = f.name
        small_config.save(path)
        loaded = DatasetConfig.load(path)
        assert loaded.name == small_config.name
        assert loaded.algorithm == small_config.algorithm
        assert len(loaded.sizes) == len(small_config.sizes)
        assert loaded.total_count() == small_config.total_count()
        Path(path).unlink()


class TestMazeDataset:
    def test_generate_count(self, small_dataset):
        assert len(small_dataset) == 15

    def test_generate_sizes(self, small_dataset):
        counts = small_dataset.size_counts
        assert counts["3x3"] == 10
        assert counts["4x4"] == 5

    def test_indexing(self, small_dataset):
        record = small_dataset[0]
        assert isinstance(record, MazeRecord)

    def test_iteration(self, small_dataset):
        records = list(small_dataset)
        assert len(records) == 15

    def test_filter_by_size(self, small_dataset):
        threes = small_dataset.filter_by_size(3, 3)
        assert len(threes) == 10
        assert all(r.width == 3 and r.height == 3 for r in threes)

    def test_filter_by_solution_length(self, small_dataset):
        short = small_dataset.filter_by_solution_length(max_len=5)
        assert all(r.solution_length <= 5 for r in short)

    def test_save_load_roundtrip(self, small_dataset):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False) as f:
            path = f.name
        small_dataset.save(path)
        loaded = MazeDataset.load(path)
        assert len(loaded) == len(small_dataset)
        for orig, restored in zip(small_dataset, loaded):
            assert orig.id == restored.id
            assert orig.solution_moves == restored.solution_moves
            assert orig.walls == restored.walls
        Path(path).unlink()

    def test_reconstructed_mazes_are_valid(self, small_dataset):
        for record in small_dataset:
            maze = record.to_maze()
            assert maze.solution[0] == maze.entry
            assert maze.solution[-1] == maze.exit
            assert len(maze.solution_moves) == record.solution_length

    def test_summary(self, small_dataset):
        s = small_dataset.summary()
        assert "15 records" in s
        assert "3x3" in s

    def test_solution_length_distribution(self, small_dataset):
        dist = small_dataset.solution_length_distribution
        assert sum(dist.values()) == 15

    def test_deterministic(self, small_config):
        ds1 = MazeDataset.generate(small_config, progress=False)
        ds2 = MazeDataset.generate(small_config, progress=False)
        for r1, r2 in zip(ds1, ds2):
            assert r1.walls == r2.walls
            assert r1.solution_moves == r2.solution_moves
