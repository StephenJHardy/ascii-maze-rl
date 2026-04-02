"""Tests for maze rendering and prompt formatting."""

from src.maze_gen import generate
from src.maze_repr import grid_to_str, solution_to_str, to_grid, to_prompt, to_str


class TestToGrid:
    def test_grid_dimensions(self):
        maze = generate(3, 3, seed=42)
        grid = to_grid(maze)
        assert len(grid) == 7  # 2*3 + 1
        assert all(len(row) == 7 for row in grid)

    def test_grid_dimensions_rectangular(self):
        maze = generate(4, 3, seed=0)
        grid = to_grid(maze)
        assert len(grid) == 7   # 2*3 + 1
        assert all(len(row) == 9 for row in grid)  # 2*4 + 1

    def test_corners_are_walls(self):
        maze = generate(3, 3, seed=42)
        grid = to_grid(maze)
        for r in range(0, 7, 2):
            for c in range(0, 7, 2):
                assert grid[r][c] == "#", f"Corner ({r},{c}) should be '#'"

    def test_cell_interiors_are_open(self):
        maze = generate(3, 3, seed=42)
        grid = to_grid(maze)
        for r in range(1, 7, 2):
            for c in range(1, 7, 2):
                assert grid[r][c] == ".", f"Cell interior ({r},{c}) should be '.'"

    def test_entry_marker(self):
        maze = generate(3, 3, seed=42)
        grid = to_grid(maze)
        assert grid[1][0] == ">"

    def test_exit_marker(self):
        maze = generate(3, 3, seed=42)
        grid = to_grid(maze)
        assert grid[5][6] == ">"

    def test_only_wall_and_open_and_markers(self):
        maze = generate(5, 5, seed=42)
        grid = to_grid(maze)
        valid = {"#", ".", ">"}
        for r, row in enumerate(grid):
            for c, ch in enumerate(row):
                assert ch in valid, f"Unexpected char '{ch}' at ({r},{c})"

    def test_passages_match_no_walls(self):
        """Verify that open passages in the grid correspond to missing walls in the maze."""
        maze = generate(3, 3, seed=42)
        grid = to_grid(maze)
        for r in range(maze.height):
            for c in range(maze.width):
                if c + 1 < maze.width:
                    wall = frozenset({(r, c), (r, c + 1)})
                    grid_char = grid[2 * r + 1][2 * c + 2]
                    if wall in maze.walls:
                        assert grid_char == "#"
                    else:
                        assert grid_char == "."
                if r + 1 < maze.height:
                    wall = frozenset({(r, c), (r + 1, c)})
                    grid_char = grid[2 * r + 2][2 * c + 1]
                    if wall in maze.walls:
                        assert grid_char == "#"
                    else:
                        assert grid_char == "."


class TestToStr:
    def test_space_separated(self):
        maze = generate(3, 3, seed=42)
        s = to_str(maze)
        lines = s.split("\n")
        assert len(lines) == 7
        for line in lines:
            chars = line.split(" ")
            assert all(len(ch) == 1 for ch in chars), f"Multi-char token: {chars}"

    def test_roundtrip_grid_to_str(self):
        maze = generate(3, 3, seed=42)
        grid = to_grid(maze)
        s = grid_to_str(grid)
        lines = s.split("\n")
        for r, line in enumerate(lines):
            chars = line.split(" ")
            assert chars == grid[r]


class TestSolutionToStr:
    def test_format(self):
        assert solution_to_str(("r", "r", "d", "l")) == "r r d l"

    def test_empty(self):
        assert solution_to_str(()) == ""

    def test_single(self):
        assert solution_to_str(("d",)) == "d"


class TestToPrompt:
    def test_contains_maze(self):
        maze = generate(3, 3, seed=42)
        prompt = to_prompt(maze)
        assert to_str(maze) in prompt

    def test_contains_system_prompt(self):
        maze = generate(3, 3, seed=42)
        prompt = to_prompt(maze)
        assert "solve" in prompt.lower() or "maze" in prompt.lower()

    def test_plain_text_without_tokenizer(self):
        maze = generate(3, 3, seed=42)
        prompt = to_prompt(maze)
        assert isinstance(prompt, str)
        assert len(prompt) > 0
