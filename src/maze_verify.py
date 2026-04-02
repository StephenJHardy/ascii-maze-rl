"""
Move parsing, path simulation, and solution verification.

Extracts move sequences from model output, simulates them through a maze,
and scores the result.
"""

from __future__ import annotations

import re

from src.maze_gen import Maze, can_move

DIRECTIONS: dict[str, tuple[int, int]] = {
    "u": (-1, 0),
    "d": (1, 0),
    "l": (0, -1),
    "r": (0, 1),
}

VALID_MOVES = frozenset(DIRECTIONS.keys())

# Matches a sequence of space-separated single direction characters.
# Captures the full sequence so we can split it.
_MOVE_SEQ_PATTERN = re.compile(r"\b([udlr](?:\s+[udlr])+)\b")
_SINGLE_MOVE_PATTERN = re.compile(r"\b([udlr])\b")


def extract_moves(text: str) -> list[str] | None:
    """
    Extract a move sequence from model output.

    Looks for the last sequence of space-separated u/d/l/r characters.
    Takes the last match so the model can reason before its answer.
    Returns None if no valid moves are found.
    """
    matches = _MOVE_SEQ_PATTERN.findall(text)
    if matches:
        return matches[-1].split()

    m = _SINGLE_MOVE_PATTERN.search(text)
    if m:
        return [m.group(1)]

    return None


def simulate(moves: list[str], maze: Maze) -> list[tuple[int, int]]:
    """
    Simulate a move sequence through a maze.

    Starts at maze.entry. Stops at the first wall collision, out-of-bounds
    move, or when moves are exhausted. Returns the path of cells visited
    (including the starting cell).
    """
    pos = maze.entry
    path = [pos]
    for move in moves:
        delta = DIRECTIONS.get(move)
        if delta is None:
            break
        next_pos = (pos[0] + delta[0], pos[1] + delta[1])
        if not can_move(pos, next_pos, maze):
            break
        pos = next_pos
        path.append(pos)
    return path


def reached_exit(path: list[tuple[int, int]], maze: Maze) -> bool:
    """Check if the path reaches the maze exit."""
    return len(path) > 0 and path[-1] == maze.exit


def manhattan_progress(
    pos: tuple[int, int],
    target: tuple[int, int],
    origin: tuple[int, int],
) -> float:
    """
    Measure progress from origin toward target, normalized to [0, 1].

    Returns 0.0 if pos is at origin, 1.0 if pos is at target.
    Can return negative values if pos is further from target than origin is.
    """
    total = abs(target[0] - origin[0]) + abs(target[1] - origin[1])
    if total == 0:
        return 1.0
    remaining = abs(target[0] - pos[0]) + abs(target[1] - pos[1])
    return 1.0 - remaining / total
