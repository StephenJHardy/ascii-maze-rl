"""Tests for maze generation and solving."""

import pytest

from src.maze_gen import Maze, can_move, generate, path_to_moves, solve


class TestGenerate:
    def test_returns_maze(self):
        maze = generate(3, 3, seed=42)
        assert isinstance(maze, Maze)
        assert maze.width == 3
        assert maze.height == 3

    def test_entry_and_exit(self):
        maze = generate(4, 4, seed=0)
        assert maze.entry == (0, 0)
        assert maze.exit == (3, 3)

    def test_deterministic(self):
        a = generate(5, 5, seed=123)
        b = generate(5, 5, seed=123)
        assert a == b

    def test_different_seeds_differ(self):
        a = generate(5, 5, seed=1)
        b = generate(5, 5, seed=2)
        assert a.walls != b.walls

    def test_has_solution(self):
        for seed in range(20):
            maze = generate(5, 5, seed=seed)
            assert maze.solution is not None
            assert len(maze.solution) >= 2
            assert maze.solution[0] == maze.entry
            assert maze.solution[-1] == maze.exit

    def test_solution_is_valid_path(self):
        maze = generate(4, 4, seed=7)
        for i in range(len(maze.solution) - 1):
            curr = maze.solution[i]
            next_ = maze.solution[i + 1]
            dr = abs(next_[0] - curr[0])
            dc = abs(next_[1] - curr[1])
            assert dr + dc == 1, f"Non-adjacent step: {curr} -> {next_}"
            assert can_move(curr, next_, maze), f"Wall between {curr} and {next_}"

    def test_perfect_maze_all_cells_reachable(self):
        maze = generate(4, 4, seed=99)
        for r in range(maze.height):
            for c in range(maze.width):
                path = solve(maze.width, maze.height, maze.walls, (0, 0), (r, c))
                assert path is not None, f"Cell ({r},{c}) unreachable"

    def test_minimum_size(self):
        maze = generate(2, 2, seed=0)
        assert maze.width == 2
        assert maze.height == 2
        assert len(maze.solution) >= 2

    def test_rectangular_maze(self):
        maze = generate(3, 5, seed=0)
        assert maze.width == 3
        assert maze.height == 5
        assert maze.exit == (4, 2)

    def test_too_small_raises(self):
        with pytest.raises(ValueError):
            generate(1, 3, seed=0)
        with pytest.raises(ValueError):
            generate(3, 1, seed=0)

    def test_various_sizes(self):
        for size in range(2, 10):
            maze = generate(size, size, seed=size)
            assert maze.solution[0] == (0, 0)
            assert maze.solution[-1] == (size - 1, size - 1)


class TestSolve:
    def test_shortest_path(self):
        maze = generate(3, 3, seed=42)
        path = solve(maze.width, maze.height, maze.walls, maze.entry, maze.exit)
        assert path is not None
        assert path[0] == maze.entry
        assert path[-1] == maze.exit

    def test_start_equals_end(self):
        maze = generate(3, 3, seed=0)
        path = solve(maze.width, maze.height, maze.walls, (0, 0), (0, 0))
        assert path == [(0, 0)]


class TestCanMove:
    def test_valid_move(self):
        maze = generate(3, 3, seed=42)
        first = maze.solution[0]
        second = maze.solution[1]
        assert can_move(first, second, maze)

    def test_out_of_bounds(self):
        maze = generate(3, 3, seed=42)
        assert not can_move((0, 0), (-1, 0), maze)
        assert not can_move((0, 0), (0, -1), maze)

    def test_wall_blocks(self):
        maze = generate(3, 3, seed=42)
        for wall in maze.walls:
            a, b = tuple(wall)
            assert not can_move(a, b, maze)


class TestPathToMoves:
    def test_simple_path(self):
        path = ((0, 0), (0, 1), (1, 1), (2, 1))
        assert path_to_moves(path) == ("r", "d", "d")

    def test_all_directions(self):
        path = ((1, 1), (0, 1), (0, 2), (1, 2), (1, 1))
        assert path_to_moves(path) == ("u", "r", "d", "l")

    def test_single_cell(self):
        assert path_to_moves(((0, 0),)) == ()

    def test_invalid_step_raises(self):
        with pytest.raises(ValueError):
            path_to_moves(((0, 0), (2, 2)))

    def test_matches_solution(self):
        maze = generate(4, 4, seed=10)
        moves = path_to_moves(maze.solution)
        assert len(moves) == len(maze.solution) - 1
        assert all(m in "udlr" for m in moves)
