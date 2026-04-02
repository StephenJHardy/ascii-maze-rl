"""
Maze generation and solving.

Generates perfect mazes (exactly one path between any two cells). Two
algorithms are available:

  - Wilson's algorithm (default): uniform random spanning tree sampling
    via loop-erased random walks. Every possible maze is equally likely.
  - DFS backtracker: randomized depth-first search. Produces mazes with
    long corridors but can only generate a small subset of all possible
    mazes (e.g. 14 of 192 for 3×3).

A maze is represented as a Maze dataclass containing:
  - width, height: logical dimensions (number of cells)
  - walls: set of frozensets, each a pair of adjacent cells with a wall between them
  - entry: always (0, 0)
  - exit: always (height-1, width-1)
  - solution: shortest path as a list of (row, col) tuples
  - seed: the seed used to generate this maze
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from enum import Enum


class Algorithm(Enum):
    WILSON = "wilson"
    DFS = "dfs"


@dataclass(frozen=True)
class Maze:
    width: int
    height: int
    walls: frozenset[frozenset[tuple[int, int]]]
    entry: tuple[int, int]
    exit: tuple[int, int]
    solution: tuple[tuple[int, int], ...]
    seed: int

    @property
    def solution_moves(self) -> tuple[str, ...]:
        """Convert the solution path to a sequence of move characters."""
        return path_to_moves(self.solution)


def generate(
    width: int,
    height: int,
    seed: int,
    algorithm: Algorithm = Algorithm.WILSON,
) -> Maze:
    """
    Generate a perfect maze.

    Every cell is reachable from every other cell, and there is exactly one
    path between any two cells.

    Entry is always (0, 0), exit is always (height-1, width-1).
    """
    if width < 2 or height < 2:
        raise ValueError(f"Maze must be at least 2x2, got {width}x{height}")

    rng = random.Random(seed)

    if algorithm == Algorithm.WILSON:
        walls = _generate_wilson(width, height, rng)
    elif algorithm == Algorithm.DFS:
        walls = _generate_dfs(width, height, rng)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    entry = (0, 0)
    exit_ = (height - 1, width - 1)

    solution = solve(width, height, walls, entry, exit_)
    if solution is None:
        raise RuntimeError(f"Generated maze has no solution (seed={seed})")

    return Maze(
        width=width,
        height=height,
        walls=walls,
        entry=entry,
        exit=exit_,
        solution=tuple(solution),
        seed=seed,
    )


def _generate_wilson(
    width: int,
    height: int,
    rng: random.Random,
) -> frozenset[frozenset[tuple[int, int]]]:
    """
    Wilson's algorithm: uniform random spanning tree via loop-erased random walks.

    Every spanning tree is equally likely to be generated, giving full coverage
    of the maze space.
    """
    cells = [(r, c) for r in range(height) for c in range(width)]
    cell_set = set(cells)

    all_walls = _all_edges(width, height)

    in_tree: set[tuple[int, int]] = set()
    # Start with a random cell in the tree
    in_tree.add(cells[rng.randrange(len(cells))])

    carved: set[frozenset[tuple[int, int]]] = set()

    remaining = [c for c in cells if c not in in_tree]
    rng.shuffle(remaining)

    for start_cell in remaining:
        if start_cell in in_tree:
            continue

        # Random walk from start_cell until we hit the tree
        path: dict[tuple[int, int], tuple[int, int]] = {}
        current = start_cell
        while current not in in_tree:
            neighbors = _cell_neighbors(current, cell_set)
            next_cell = neighbors[rng.randrange(len(neighbors))]
            path[current] = next_cell
            current = next_cell

        # Trace the loop-erased path and add to tree
        current = start_cell
        while current not in in_tree:
            next_cell = path[current]
            in_tree.add(current)
            carved.add(frozenset({current, next_cell}))
            current = next_cell

    return frozenset(all_walls - carved)


def _generate_dfs(
    width: int,
    height: int,
    rng: random.Random,
) -> frozenset[frozenset[tuple[int, int]]]:
    """
    DFS backtracker: randomized depth-first search.

    Produces mazes with long winding corridors. Only generates a subset
    of all possible spanning trees (DFS-trees from the start cell).
    """
    cells = {(r, c) for r in range(height) for c in range(width)}
    all_walls = _all_edges(width, height)

    visited: set[tuple[int, int]] = set()
    stack: list[tuple[int, int]] = []

    start = (0, 0)
    visited.add(start)
    stack.append(start)

    carved: set[frozenset[tuple[int, int]]] = set()

    while stack:
        current = stack[-1]
        neighbors = _unvisited_neighbors(current, cells, visited)
        if neighbors:
            chosen = neighbors[rng.randrange(len(neighbors))]
            carved.add(frozenset({current, chosen}))
            visited.add(chosen)
            stack.append(chosen)
        else:
            stack.pop()

    return frozenset(all_walls - carved)


def solve(
    width: int,
    height: int,
    walls: frozenset[frozenset[tuple[int, int]]],
    start: tuple[int, int],
    end: tuple[int, int],
) -> list[tuple[int, int]] | None:
    """BFS to find the shortest path from start to end."""
    if start == end:
        return [start]

    visited: set[tuple[int, int]] = {start}
    parent: dict[tuple[int, int], tuple[int, int]] = {}
    queue: deque[tuple[int, int]] = deque([start])
    cells = {(r, c) for r in range(height) for c in range(width)}

    while queue:
        current = queue.popleft()
        for neighbor in _cell_neighbors(current, cells):
            if neighbor in visited:
                continue
            wall = frozenset({current, neighbor})
            if wall in walls:
                continue
            visited.add(neighbor)
            parent[neighbor] = current
            if neighbor == end:
                return _reconstruct_path(parent, start, end)
            queue.append(neighbor)

    return None


def can_move(
    pos: tuple[int, int],
    next_pos: tuple[int, int],
    maze: Maze,
) -> bool:
    """Check if movement from pos to next_pos is valid (in bounds, no wall)."""
    if not (0 <= next_pos[0] < maze.height and 0 <= next_pos[1] < maze.width):
        return False
    wall = frozenset({pos, next_pos})
    return wall not in maze.walls


def path_to_moves(path: tuple[tuple[int, int], ...] | list[tuple[int, int]]) -> tuple[str, ...]:
    """Convert a path of (row, col) tuples to a sequence of move characters."""
    move_map = {(-1, 0): "u", (1, 0): "d", (0, -1): "l", (0, 1): "r"}
    moves: list[str] = []
    for i in range(len(path) - 1):
        dr = path[i + 1][0] - path[i][0]
        dc = path[i + 1][1] - path[i][1]
        move = move_map.get((dr, dc))
        if move is None:
            raise ValueError(f"Invalid step from {path[i]} to {path[i+1]}")
        moves.append(move)
    return tuple(moves)


def _all_edges(width: int, height: int) -> set[frozenset[tuple[int, int]]]:
    edges: set[frozenset[tuple[int, int]]] = set()
    for r in range(height):
        for c in range(width):
            if r + 1 < height:
                edges.add(frozenset({(r, c), (r + 1, c)}))
            if c + 1 < width:
                edges.add(frozenset({(r, c), (r, c + 1)}))
    return edges


def _unvisited_neighbors(
    cell: tuple[int, int],
    cells: set[tuple[int, int]],
    visited: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    return [n for n in _cell_neighbors(cell, cells) if n not in visited]


def _cell_neighbors(
    cell: tuple[int, int],
    cells: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    r, c = cell
    return [(r + dr, c + dc) for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
            if (r + dr, c + dc) in cells]


def _reconstruct_path(
    parent: dict[tuple[int, int], tuple[int, int]],
    start: tuple[int, int],
    end: tuple[int, int],
) -> list[tuple[int, int]]:
    path = [end]
    current = end
    while current != start:
        current = parent[current]
        path.append(current)
    path.reverse()
    return path
