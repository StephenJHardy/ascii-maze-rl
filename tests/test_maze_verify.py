"""Tests for move parsing, path simulation, and verification."""

import pytest

from src.maze_gen import generate
from src.maze_verify import (
    extract_moves,
    manhattan_progress,
    reached_exit,
    simulate,
)


class TestExtractMoves:
    def test_simple_sequence(self):
        assert extract_moves("r r d d l d r r") == ["r", "r", "d", "d", "l", "d", "r", "r"]

    def test_single_move(self):
        assert extract_moves("r") == ["r"]

    def test_last_match_wins(self):
        text = "First I'll go r d then actually: r r d d"
        result = extract_moves(text)
        assert result == ["r", "r", "d", "d"]

    def test_ignores_surrounding_text(self):
        text = "Let me solve this. The answer is: r d r d r. Done!"
        result = extract_moves(text)
        assert result == ["r", "d", "r", "d", "r"]

    def test_returns_none_for_gibberish(self):
        assert extract_moves("I cannot solve this maze") is None

    def test_returns_none_for_empty(self):
        assert extract_moves("") is None

    def test_returns_none_for_uppercase(self):
        assert extract_moves("R R D D") is None

    def test_returns_none_for_invalid_chars(self):
        assert extract_moves("x y z w") is None

    def test_multiline(self):
        text = "thinking...\nr d r d"
        result = extract_moves(text)
        assert result == ["r", "d", "r", "d"]

    def test_mixed_valid_invalid(self):
        text = "move r then x then d"
        result = extract_moves(text)
        assert result is not None  # should find individual 'r' or 'd'

    def test_tab_separated(self):
        result = extract_moves("r\td\tr")
        assert result == ["r", "d", "r"]


class TestSimulate:
    def test_follows_solution(self):
        maze = generate(3, 3, seed=42)
        moves = list(maze.solution_moves)
        path = simulate(moves, maze)
        assert path[0] == maze.entry
        assert path[-1] == maze.exit
        assert len(path) == len(moves) + 1

    def test_stops_at_wall(self):
        maze = generate(3, 3, seed=42)
        walls = maze.walls
        # Find a direction from entry that's blocked
        blocked = None
        for move_char, (dr, dc) in [("u", (-1, 0)), ("d", (1, 0)), ("l", (0, -1)), ("r", (0, 1))]:
            next_pos = (maze.entry[0] + dr, maze.entry[1] + dc)
            wall = frozenset({maze.entry, next_pos})
            in_bounds = 0 <= next_pos[0] < maze.height and 0 <= next_pos[1] < maze.width
            if wall in walls or not in_bounds:
                blocked = move_char
                break
        if blocked:
            path = simulate([blocked] * 5, maze)
            assert len(path) == 1  # never moved

    def test_empty_moves(self):
        maze = generate(3, 3, seed=42)
        path = simulate([], maze)
        assert path == [maze.entry]

    def test_partial_path(self):
        maze = generate(3, 3, seed=42)
        moves = list(maze.solution_moves)
        half = moves[: len(moves) // 2]
        path = simulate(half, maze)
        assert len(path) == len(half) + 1
        assert path[0] == maze.entry


class TestReachedExit:
    def test_full_solution(self):
        maze = generate(3, 3, seed=42)
        path = simulate(list(maze.solution_moves), maze)
        assert reached_exit(path, maze)

    def test_partial_path(self):
        maze = generate(3, 3, seed=42)
        moves = list(maze.solution_moves)[:-1]
        path = simulate(moves, maze)
        assert not reached_exit(path, maze)

    def test_start_only(self):
        maze = generate(3, 3, seed=42)
        assert not reached_exit([maze.entry], maze)


class TestManhattanProgress:
    def test_at_origin(self):
        assert manhattan_progress((0, 0), (4, 4), (0, 0)) == pytest.approx(0.0)

    def test_at_target(self):
        assert manhattan_progress((4, 4), (4, 4), (0, 0)) == pytest.approx(1.0)

    def test_halfway(self):
        assert manhattan_progress((2, 2), (4, 4), (0, 0)) == pytest.approx(0.5)

    def test_wrong_direction(self):
        progress = manhattan_progress((0, 0), (2, 2), (1, 1))
        assert progress < 0.0

    def test_same_start_and_end(self):
        assert manhattan_progress((3, 3), (3, 3), (3, 3)) == pytest.approx(1.0)
