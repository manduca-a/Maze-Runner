#!/usr/bin/env python3
"""
maze_generate.py

Author: Qizhen Dong

Generate a grid maze and save it as maze.json in the maze_runner root directory.

Recommended project structure:
maze_runner/
├── maze.json
└── code/
    └── maze_generate.py

Usage examples:
    python maze_generate.py
    python maze_generate.py --rows 5 --cols 5 --seed 42
    python maze_generate.py --rows 50 --cols 50 --keys 3
    python maze_generate.py --rows 100 --cols 100 --output ../maze.json
    python maze_generate.py --rows 50 --cols 50 --loop-rate 0.08
"""

from __future__ import annotations

import argparse
import json
import random
from collections import deque
from pathlib import Path
from typing import List, Tuple


UP = 1
RIGHT = 2
DOWN = 4
LEFT = 8

DIRS = {
    "U": (-1, 0, UP, DOWN),
    "R": (0, 1, RIGHT, LEFT),
    "D": (1, 0, DOWN, UP),
    "L": (0, -1, LEFT, RIGHT),
}


Coord = Tuple[int, int]


def in_bounds(r: int, c: int, rows: int, cols: int) -> bool:
    return 0 <= r < rows and 0 <= c < cols


def make_full_wall_grid(rows: int, cols: int) -> List[List[int]]:
    return [[UP | RIGHT | DOWN | LEFT for _ in range(cols)] for _ in range(rows)]


def carve_passage(grid: List[List[int]], a: Coord, b: Coord) -> None:
    ar, ac = a
    br, bc = b
    dr = br - ar
    dc = bc - ac

    if dr == -1 and dc == 0:
        grid[ar][ac] &= ~UP
        grid[br][bc] &= ~DOWN
    elif dr == 1 and dc == 0:
        grid[ar][ac] &= ~DOWN
        grid[br][bc] &= ~UP
    elif dr == 0 and dc == 1:
        grid[ar][ac] &= ~RIGHT
        grid[br][bc] &= ~LEFT
    elif dr == 0 and dc == -1:
        grid[ar][ac] &= ~LEFT
        grid[br][bc] &= ~RIGHT
    else:
        raise ValueError(f"Cells {a} and {b} are not adjacent.")


def generate_perfect_maze(rows: int, cols: int, rng: random.Random) -> List[List[int]]:
    """
    Generate a perfect maze with recursive backtracking (iterative DFS).
    Every cell is reachable and there is exactly one simple path between any two cells.
    """
    grid = make_full_wall_grid(rows, cols)
    visited = [[False for _ in range(cols)] for _ in range(rows)]

    start = (0, 0)
    stack = [start]
    visited[0][0] = True

    while stack:
        r, c = stack[-1]
        neighbors = []

        for dr, dc, _, _ in DIRS.values():
            nr, nc = r + dr, c + dc
            if in_bounds(nr, nc, rows, cols) and not visited[nr][nc]:
                neighbors.append((nr, nc))

        if neighbors:
            nr, nc = rng.choice(neighbors)
            carve_passage(grid, (r, c), (nr, nc))
            visited[nr][nc] = True
            stack.append((nr, nc))
        else:
            stack.pop()

    return grid


def add_loops(grid: List[List[int]], rng: random.Random, loop_rate: float) -> None:
    """
    Randomly remove a small portion of remaining walls after generating a perfect maze.
    This introduces cycles so the maze is no longer path-unique.

    loop_rate is the fraction of candidate interior walls to open, usually 0.05~0.10.
    """
    if loop_rate <= 0:
        return

    rows, cols = len(grid), len(grid[0])
    candidate_walls = []

    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols and (grid[r][c] & RIGHT) != 0:
                candidate_walls.append(((r, c), (r, c + 1)))
            if r + 1 < rows and (grid[r][c] & DOWN) != 0:
                candidate_walls.append(((r, c), (r + 1, c)))

    if not candidate_walls:
        return

    rng.shuffle(candidate_walls)
    open_count = max(1, int(len(candidate_walls) * loop_rate))
    open_count = min(open_count, len(candidate_walls))

    for a, b in candidate_walls[:open_count]:
        carve_passage(grid, a, b)



def get_open_neighbors(grid: List[List[int]], pos: Coord) -> List[Coord]:
    rows, cols = len(grid), len(grid[0])
    r, c = pos
    result = []

    if (grid[r][c] & UP) == 0 and in_bounds(r - 1, c, rows, cols):
        result.append((r - 1, c))
    if (grid[r][c] & RIGHT) == 0 and in_bounds(r, c + 1, rows, cols):
        result.append((r, c + 1))
    if (grid[r][c] & DOWN) == 0 and in_bounds(r + 1, c, rows, cols):
        result.append((r + 1, c))
    if (grid[r][c] & LEFT) == 0 and in_bounds(r, c - 1, rows, cols):
        result.append((r, c - 1))

    return result


def bfs_distances(grid: List[List[int]], start: Coord) -> List[List[int]]:
    rows, cols = len(grid), len(grid[0])
    dist = [[-1 for _ in range(cols)] for _ in range(rows)]
    q = deque([start])
    sr, sc = start
    dist[sr][sc] = 0

    while q:
        cur = q.popleft()
        for nxt in get_open_neighbors(grid, cur):
            nr, nc = nxt
            if dist[nr][nc] == -1:
                cr, cc = cur
                dist[nr][nc] = dist[cr][cc] + 1
                q.append(nxt)

    return dist


def choose_start_exit(grid: List[List[int]]) -> Tuple[Coord, Coord]:
    """
    Pick start at bottom-left and exit at top-right by default.
    This matches the project description well and keeps the layout easy to explain.
    """
    rows, cols = len(grid), len(grid[0])
    start = (rows - 1, 0)
    exit_pos = (0, cols - 1)
    return start, exit_pos


def choose_keys(
    grid: List[List[int]],
    start: Coord,
    exit_pos: Coord,
    key_count: int,
    rng: random.Random,
) -> List[Coord]:
    """
    Choose key positions uniformly at random from all maze cells,
    excluding only start and exit.

    This intentionally removes the previous distance-based heuristic so
    keys can appear anywhere in the maze, including near the center.
    """
    rows, cols = len(grid), len(grid[0])
    all_cells = [(r, c) for r in range(rows) for c in range(cols)]
    candidates = [cell for cell in all_cells if cell != start and cell != exit_pos]

    if len(candidates) < key_count:
        raise ValueError("Not enough cells to place all keys.")

    return rng.sample(candidates, key_count)


def manhattan(a: Coord, b: Coord) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def validate_maze(grid: List[List[int]], start: Coord, exit_pos: Coord, keys: List[Coord]) -> None:
    """
    Basic correctness checks:
    - all special cells are distinct
    - start can reach every key and exit
    - wall symmetry is consistent
    """
    rows, cols = len(grid), len(grid[0])

    specials = [start, exit_pos] + list(keys)
    if len(set(specials)) != len(specials):
        raise ValueError("Start, exit, and keys must all be in distinct cells.")

    dist = bfs_distances(grid, start)
    for cell in [exit_pos] + list(keys):
        r, c = cell
        if dist[r][c] == -1:
            raise ValueError(f"Special cell {cell} is unreachable from start.")

    for r in range(rows):
        for c in range(cols):
            if r > 0:
                up_open = (grid[r][c] & UP) == 0
                neighbor_down_open = (grid[r - 1][c] & DOWN) == 0
                if up_open != neighbor_down_open:
                    raise ValueError(f"Wall mismatch between {(r, c)} and {(r - 1, c)}")
            if c < cols - 1:
                right_open = (grid[r][c] & RIGHT) == 0
                neighbor_left_open = (grid[r][c + 1] & LEFT) == 0
                if right_open != neighbor_left_open:
                    raise ValueError(f"Wall mismatch between {(r, c)} and {(r, c + 1)}")


def build_maze_data(
    rows: int,
    cols: int,
    key_count: int,
    seed: int | None,
    generator_name: str = "dfs_backtracking",
    loop_rate: float = 0.08,
) -> dict:
    rng = random.Random(seed)

    grid = generate_perfect_maze(rows, cols, rng)
    add_loops(grid, rng, loop_rate)
    start, exit_pos = choose_start_exit(grid)
    keys = choose_keys(grid, start, exit_pos, key_count, rng)

    validate_maze(grid, start, exit_pos, keys)

    return {
        "meta": {
            "rows": rows,
            "cols": cols,
            "generator": generator_name,
            "seed": seed,
            "key_count": key_count,
            "loop_rate": loop_rate,
            "wall_encoding": {
                "UP": UP,
                "RIGHT": RIGHT,
                "DOWN": DOWN,
                "LEFT": LEFT,
            },
        },
        "start": [start[0], start[1]],
        "exit": [exit_pos[0], exit_pos[1]],
        "keys": [[r, c] for r, c in keys],
        "cells": grid,
    }



def default_output_path() -> Path:
    """
    Save maze.json in the maze_runner root directory.
    If this script is placed in maze_runner/code/, this returns maze_runner/maze.json.
    """
    script_path = Path(__file__).resolve()
    root_dir = script_path.parent.parent
    return root_dir / "maze.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a maze.json file for the CS5800 final project.")
    parser.add_argument("--rows", type=int, default=5, help="Number of maze rows.")
    parser.add_argument("--cols", type=int, default=5, help="Number of maze columns.")
    parser.add_argument("--keys", type=int, default=3, help="Number of keys to place.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible maze generation.")
    parser.add_argument(
        "--loop-rate",
        type=float,
        default=0.08,
        help="Fraction of candidate walls to open after maze generation. Recommended: 0.05~0.10.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output path. Default: maze_runner/maze.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.rows <= 0 or args.cols <= 0:
        raise ValueError("rows and cols must both be positive integers.")
    if args.keys < 0:
        raise ValueError("keys must be a non-negative integer.")
    if args.keys > args.rows * args.cols - 2:
        raise ValueError("Too many keys for the maze size.")
    if not 0 <= args.loop_rate <= 1:
        raise ValueError("loop_rate must be between 0 and 1.")

    maze_data = build_maze_data(
        rows=args.rows,
        cols=args.cols,
        key_count=args.keys,
        seed=args.seed,
        loop_rate=args.loop_rate,
    )

    output_path = Path(args.output).resolve() if args.output else default_output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(maze_data, f, indent=2)

    print(f"Maze saved to: {output_path}")
    print(f"Size: {args.rows}x{args.cols}")
    print(f"Start: {tuple(maze_data['start'])}")
    print(f"Exit: {tuple(maze_data['exit'])}")
    print(f"Keys: {[tuple(k) for k in maze_data['keys']]}")
    print(f"Loop rate: {maze_data['meta']['loop_rate']}")
    print("Wall encoding: UP=1, RIGHT=2, DOWN=4, LEFT=8")


if __name__ == "__main__":
    main()
