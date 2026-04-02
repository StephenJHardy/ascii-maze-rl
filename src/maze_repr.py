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
    Grid dimensions: (2*height+1) rows × (2*width+1) cols.
    Entry and exit markers overwrite the boundary wall cells on the
    leftmost and rightmost columns.
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


SYSTEM_PROMPT = (
    "You solve mazes. Output ONLY moves as space-separated letters."
    "\nExample output: d r r u d"
)


def to_chat_messages(maze: Maze) -> list[dict[str, str]]:
    """Build chat messages for a maze (system + user)."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": to_str(maze)},
    ]


def to_prompt(maze: Maze, tokenizer=None) -> str:
    """
    Build the full prompt for a maze.

    If a tokenizer is provided, applies the chat template. Otherwise returns
    a plain text version (for display/testing).
    """
    messages = to_chat_messages(maze)
    if tokenizer is not None:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return f"{SYSTEM_PROMPT}\n\n{to_str(maze)}"
