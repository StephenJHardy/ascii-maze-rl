"""
Maze rendering and prompt formatting.

Converts a Maze into the space-separated expanded grid format designed for
clean tokenization (1 token per grid position with Qwen tokenizers).

A W×H maze of logical cells maps to a (2W+1) × (2H+1) character grid:
  - Even row, even col → corner (always '#')
  - Even row, odd col  → horizontal wall ('#') or passage ('.')
  - Odd row, even col  → vertical wall ('#') or passage ('.')
  - Odd row, odd col   → cell interior (always '.')

Entry '>' appears at the left edge of (0,0).
Exit '>' appears at the right edge of (H-1, W-1).
"""

from __future__ import annotations

from src.maze_gen import Maze

WALL = "#"
OPEN = "."
ENTRY_EXIT = ">"


def to_grid(maze: Maze) -> list[list[str]]:
    """
    Render a Maze as a 2D character grid.

    Returns a list of rows, each a list of single characters.
    Grid dimensions: (2*height+1) rows × (2*width+1) cols,
    plus the entry/exit markers which extend the first and last data rows.
    """
    h, w = maze.height, maze.width
    rows = 2 * h + 1
    cols = 2 * w + 1

    grid = [[WALL for _ in range(cols)] for _ in range(rows)]

    for r in range(h):
        for c in range(w):
            grid[2 * r + 1][2 * c + 1] = OPEN

    for r in range(h):
        for c in range(w):
            if c + 1 < w:
                wall = frozenset({(r, c), (r, c + 1)})
                if wall not in maze.walls:
                    grid[2 * r + 1][2 * c + 2] = OPEN
            if r + 1 < h:
                wall = frozenset({(r, c), (r + 1, c)})
                if wall not in maze.walls:
                    grid[2 * r + 2][2 * c + 1] = OPEN

    entry_row = 2 * maze.entry[0] + 1
    grid[entry_row][0] = ENTRY_EXIT

    exit_row = 2 * maze.exit[0] + 1
    grid[exit_row][cols - 1] = ENTRY_EXIT

    return grid


def grid_to_str(grid: list[list[str]]) -> str:
    """Convert a character grid to a space-separated string with newlines."""
    return "\n".join(" ".join(row) for row in grid)


def to_str(maze: Maze) -> str:
    """Render a maze as a space-separated character grid string."""
    return grid_to_str(to_grid(maze))


def solution_to_str(moves: tuple[str, ...]) -> str:
    """Format a move sequence as a space-separated string."""
    return " ".join(moves)


PROMPT_TEMPLATE = """\
Solve this maze. Find a path from the entrance (>) on the left side to the exit (>) on the right side.

Maze:
{maze}

Output ONLY your sequence of moves (u/d/l/r), space-separated.
Moves:"""


def to_prompt(maze: Maze) -> str:
    """Build the full prompt for a maze."""
    return PROMPT_TEMPLATE.format(maze=to_str(maze))
