"""
Enumerate and categorize mazes by size and solution length.

For small grids, exhaustively enumerates all spanning trees (perfect mazes)
of the grid graph. For larger grids, samples randomly and estimates the
distribution.

A perfect maze is a spanning tree of the grid graph — exactly one path
between any two cells, so exactly one loop-free solution per maze.

Usage:
    uv run python src/maze_census.py
"""

from __future__ import annotations

import sys
import time
from collections import Counter
from itertools import combinations

from src.maze_gen import generate, solve


def grid_edges(width: int, height: int) -> list[frozenset[tuple[int, int]]]:
    """All possible internal edges (walls that could be removed) in a grid."""
    edges = []
    for r in range(height):
        for c in range(width):
            if c + 1 < width:
                edges.append(frozenset({(r, c), (r, c + 1)}))
            if r + 1 < height:
                edges.append(frozenset({(r, c), (r + 1, c)}))
    return edges


def is_spanning_tree(
    edges_kept: set[frozenset[tuple[int, int]]],
    num_cells: int,
    start: tuple[int, int],
    all_cells: set[tuple[int, int]],
) -> bool:
    """Check if the kept edges form a spanning tree (connected + n-1 edges)."""
    if len(edges_kept) != num_cells - 1:
        return False

    adj: dict[tuple[int, int], list[tuple[int, int]]] = {c: [] for c in all_cells}
    for edge in edges_kept:
        a, b = tuple(edge)
        adj[a].append(b)
        adj[b].append(a)

    visited = {start}
    stack = [start]
    while stack:
        node = stack.pop()
        for neighbor in adj[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                stack.append(neighbor)

    return len(visited) == num_cells


def enumerate_mazes(
    width: int, height: int,
) -> list[tuple[frozenset[frozenset[tuple[int, int]]], int]]:
    """
    Exhaustively enumerate all perfect mazes for a grid.

    Returns list of (walls_as_frozenset, solution_length) tuples.
    """
    edges = grid_edges(width, height)
    num_cells = width * height
    num_tree_edges = num_cells - 1
    all_cells = {(r, c) for r in range(height) for c in range(width)}
    entry = (0, 0)
    exit_ = (height - 1, width - 1)

    results = []
    num_edges = len(edges)
    edges_to_remove = num_edges - num_tree_edges

    for removed in combinations(range(num_edges), edges_to_remove):
        removed_set = {edges[i] for i in removed}
        kept = set(edges) - removed_set
        walls = frozenset(removed_set)

        if is_spanning_tree(kept, num_cells, entry, all_cells):
            path = solve(width, height, walls, entry, exit_)
            if path is not None:
                solution_length = len(path) - 1  # number of moves
                results.append((walls, solution_length))

    return results


def sample_mazes(
    width: int,
    height: int,
    num_samples: int = 100_000,
) -> Counter[int]:
    """Sample random mazes and return solution length distribution."""
    length_counts: Counter[int] = Counter()
    seen: set[frozenset[frozenset[tuple[int, int]]]] = set()

    for seed in range(num_samples):
        maze = generate(width, height, seed=seed)
        if maze.walls not in seen:
            seen.add(maze.walls)
            solution_length = len(maze.solution) - 1
            length_counts[solution_length] += 1

    return length_counts


def main():
    print("=" * 70)
    print("Maze Census: Unique mazes and solution length distributions")
    print("=" * 70)
    print()
    print("Perfect mazes have exactly ONE loop-free solution per maze.")
    print("We categorize by solution length (number of moves).")
    print()

    # Exhaustive enumeration for small sizes
    exhaustive_sizes = [(2, 2), (2, 3), (3, 2), (3, 3)]

    # Check if user wants to include 3x4 / 4x3 (slower)
    if "--include-4" in sys.argv:
        exhaustive_sizes += [(3, 4), (4, 3), (4, 4)]

    for width, height in exhaustive_sizes:
        t0 = time.perf_counter()
        results = enumerate_mazes(width, height)
        elapsed = time.perf_counter() - t0

        length_dist: Counter[int] = Counter()
        for _, sol_len in results:
            length_dist[sol_len] += 1

        print(f"--- {width}×{height} (exhaustive, {elapsed:.1f}s) ---")
        print(f"  Total unique mazes: {len(results)}")
        print("  Solution length distribution:")
        for length in sorted(length_dist):
            count = length_dist[length]
            pct = 100 * count / len(results)
            bar = "█" * int(pct / 2)
            print(f"    {length:3d} moves: {count:6d} ({pct:5.1f}%) {bar}")
        print()

    # Sampling for larger sizes
    sample_sizes = [(4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9)]
    num_samples = 100_000

    for width, height in sample_sizes:
        t0 = time.perf_counter()
        length_dist = sample_mazes(width, height, num_samples)
        elapsed = time.perf_counter() - t0
        total_unique = sum(length_dist.values())

        print(f"--- {width}×{height} (sampled {num_samples:,} seeds, {elapsed:.1f}s) ---")
        print(f"  Unique mazes found: {total_unique:,}")
        if total_unique < num_samples:
            print(f"  (exhausted unique mazes — total population ≈ {total_unique})")
        else:
            print("  (sampling — true population is larger)")
        print("  Solution length distribution:")
        for length in sorted(length_dist):
            count = length_dist[length]
            pct = 100 * count / total_unique
            bar = "█" * int(pct / 2)
            print(f"    {length:3d} moves: {count:6d} ({pct:5.1f}%) {bar}")
        print()


if __name__ == "__main__":
    main()
