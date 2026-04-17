#!/usr/bin/env python3
"""
dijkstra.py

Read maze.json from the maze_runner root directory and complete the maze task
using standard Dijkstra shortest-path search.

Task rules implemented here:
1. Start from the entrance.
2. The exit position is known.
3. The three key positions exist in maze.json, but the agent does not use them
   as goals before stepping onto those cells.
4. Before collecting all keys, the agent prioritizes exploration of unknown
   cells by repeatedly moving to the nearest unvisited cell using Dijkstra.
5. After all keys are collected, the agent sets the exit as the goal and moves
   directly to the exit using Dijkstra.

Recommended project structure:
maze_runner/
├── maze.json
└── code/
    ├── maze_generate.py
    ├── maze_display.py
    └── dijkstra.py

Usage:
    python dijkstra.py
    python dijkstra.py --input ../maze.json
    python dijkstra.py --output ../dijkstra_result.json
"""

from __future__ import annotations

import argparse
import heapq
import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


UP = 1
RIGHT = 2
DOWN = 4
LEFT = 8

Coord = Tuple[int, int]


def default_input_path() -> Path:
    """
    If this script is placed in maze_runner/code/, this returns maze_runner/maze.json.
    """
    script_path = Path(__file__).resolve()
    root_dir = script_path.parent.parent
    return root_dir / "maze.json"


def default_output_path() -> Path:
    """
    If this script is placed in maze_runner/code/, this returns
    maze_runner/dijkstra_result.json.
    """
    script_path = Path(__file__).resolve()
    root_dir = script_path.parent.parent
    return root_dir / "dijkstra_result.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Traverse the maze with Dijkstra-based exploration and exit routing."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Optional input path. Default: maze_runner/maze.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output path. Default: maze_runner/dijkstra_result.json",
    )
    return parser.parse_args()


def load_maze(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"maze.json not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    required_keys = {"meta", "start", "exit", "keys", "cells"}
    missing = required_keys - set(data.keys())
    if missing:
        raise ValueError(f"maze.json is missing required fields: {sorted(missing)}")

    return data


def validate_maze(data: Dict) -> None:
    rows = data["meta"]["rows"]
    cols = data["meta"]["cols"]
    cells = data["cells"]

    if len(cells) != rows:
        raise ValueError("Row count in cells does not match meta.rows.")
    for row in cells:
        if len(row) != cols:
            raise ValueError("Column count in cells does not match meta.cols.")

    specials = [tuple(data["start"]), tuple(data["exit"])] + [tuple(k) for k in data["keys"]]
    for r, c in specials:
        if not (0 <= r < rows and 0 <= c < cols):
            raise ValueError(f"Special cell {(r, c)} is out of bounds.")

    if len(set(specials)) != len(specials):
        raise ValueError("Start, exit, and keys must all be in distinct cells.")


def in_bounds(r: int, c: int, rows: int, cols: int) -> bool:
    return 0 <= r < rows and 0 <= c < cols


def get_open_neighbors(cells: List[List[int]], pos: Coord) -> List[Coord]:
    rows, cols = len(cells), len(cells[0])
    r, c = pos
    cell = cells[r][c]
    neighbors: List[Coord] = []

    if (cell & UP) == 0 and in_bounds(r - 1, c, rows, cols):
        neighbors.append((r - 1, c))
    if (cell & RIGHT) == 0 and in_bounds(r, c + 1, rows, cols):
        neighbors.append((r, c + 1))
    if (cell & DOWN) == 0 and in_bounds(r + 1, c, rows, cols):
        neighbors.append((r + 1, c))
    if (cell & LEFT) == 0 and in_bounds(r, c - 1, rows, cols):
        neighbors.append((r, c - 1))

    return neighbors


def dijkstra_shortest_path(cells: List[List[int]], start: Coord, goal: Coord) -> List[Coord]:
    """
    Standard Dijkstra shortest path for an unweighted grid graph.
    Each move has cost 1.
    Returns the full path from start to goal (inclusive).
    """
    if start == goal:
        return [start]

    pq: List[Tuple[int, Coord]] = [(0, start)]
    dist: Dict[Coord, int] = {start: 0}
    parent: Dict[Coord, Optional[Coord]] = {start: None}
    settled: Set[Coord] = set()

    while pq:
        cur_dist, cur = heapq.heappop(pq)

        if cur in settled:
            continue
        settled.add(cur)

        if cur == goal:
            break

        for nxt in get_open_neighbors(cells, cur):
            new_dist = cur_dist + 1
            if nxt not in dist or new_dist < dist[nxt]:
                dist[nxt] = new_dist
                parent[nxt] = cur
                heapq.heappush(pq, (new_dist, nxt))

    if goal not in parent:
        raise ValueError(f"No path found from {start} to {goal}.")

    path: List[Coord] = []
    node: Optional[Coord] = goal
    while node is not None:
        path.append(node)
        node = parent[node]
    path.reverse()
    return path


def dijkstra_to_nearest_unvisited(
    cells: List[List[int]],
    start: Coord,
    visited_cells: Set[Coord],
) -> List[Coord]:
    """
    Standard Dijkstra search from start until the nearest unvisited cell
    is settled. Returns the shortest path to that cell.
    """
    if start not in visited_cells:
        return [start]

    pq: List[Tuple[int, Coord]] = [(0, start)]
    dist: Dict[Coord, int] = {start: 0}
    parent: Dict[Coord, Optional[Coord]] = {start: None}
    settled: Set[Coord] = set()

    while pq:
        cur_dist, cur = heapq.heappop(pq)

        if cur in settled:
            continue
        settled.add(cur)

        if cur not in visited_cells:
            path: List[Coord] = []
            node: Optional[Coord] = cur
            while node is not None:
                path.append(node)
                node = parent[node]
            path.reverse()
            return path

        for nxt in get_open_neighbors(cells, cur):
            new_dist = cur_dist + 1
            if nxt not in dist or new_dist < dist[nxt]:
                dist[nxt] = new_dist
                parent[nxt] = cur
                heapq.heappush(pq, (new_dist, nxt))

    raise ValueError("No reachable unvisited cell remains.")


def expand_path(
    full_path: List[Coord],
    segment: List[Coord],
    visit_counts: Dict[Coord, int],
    visited_cells: Set[Coord],
    actual_keys: Set[Coord],
    collected_keys: Set[Coord],
    discovered_key_order: List[Coord],
) -> None:
    """
    Append a path segment to the global path and update traversal stats.
    The first node of segment is not duplicated if it matches the current tail.
    """
    if not segment:
        return

    start_index = 0
    if full_path and segment[0] == full_path[-1]:
        start_index = 1

    for pos in segment[start_index:]:
        full_path.append(pos)
        visit_counts[pos] = visit_counts.get(pos, 0) + 1
        visited_cells.add(pos)

        if pos in actual_keys and pos not in collected_keys:
            collected_keys.add(pos)
            discovered_key_order.append(pos)


def run_dijkstra_agent(data: Dict) -> Dict:
    cells: List[List[int]] = data["cells"]
    start: Coord = tuple(data["start"])
    exit_pos: Coord = tuple(data["exit"])
    actual_keys: Set[Coord] = {tuple(k) for k in data["keys"]}
    target_key_count = len(actual_keys)

    current = start
    full_path: List[Coord] = [start]

    visit_counts: Dict[Coord, int] = {start: 1}
    visited_cells: Set[Coord] = {start}
    collected_keys: Set[Coord] = set()
    discovered_key_order: List[Coord] = []

    if current in actual_keys:
        collected_keys.add(current)
        discovered_key_order.append(current)

    # Phase 1: explore until all keys are reached.
    while len(collected_keys) < target_key_count:
        segment = dijkstra_to_nearest_unvisited(cells, current, visited_cells)
        expand_path(
            full_path=full_path,
            segment=segment,
            visit_counts=visit_counts,
            visited_cells=visited_cells,
            actual_keys=actual_keys,
            collected_keys=collected_keys,
            discovered_key_order=discovered_key_order,
        )
        current = full_path[-1]

    # Phase 2: go directly to the exit.
    exit_segment = dijkstra_shortest_path(cells, current, exit_pos)
    expand_path(
        full_path=full_path,
        segment=exit_segment,
        visit_counts=visit_counts,
        visited_cells=visited_cells,
        actual_keys=actual_keys,
        collected_keys=collected_keys,
        discovered_key_order=discovered_key_order,
    )

    total_steps = max(0, len(full_path) - 1)
    unique_cells_visited = len(visited_cells)
    repeated_visits = sum(count - 1 for count in visit_counts.values() if count > 1)
    backtracking_ratio = repeated_visits / total_steps if total_steps > 0 else 0.0

    seen_keys_for_report: Set[Coord] = set()
    key_collection_step_indices: List[int] = []
    for step_index, pos in enumerate(full_path):
        if pos in actual_keys and pos not in seen_keys_for_report:
            seen_keys_for_report.add(pos)
            key_collection_step_indices.append(step_index)

    steps_before_all_keys = key_collection_step_indices[-1] if key_collection_step_indices else 0
    steps_after_final_key = total_steps - steps_before_all_keys

    result = {
        "algorithm": "dijkstra_explorer_then_exit",
        "success": len(collected_keys) == target_key_count and full_path[-1] == exit_pos,
        "rules": {
            "exit_known": True,
            "keys_unknown_until_visited": True,
            "pre_key_strategy": "repeated standard Dijkstra to nearest unvisited cell",
            "post_key_strategy": "standard Dijkstra shortest path to exit",
        },
        "meta": data["meta"],
        "start": list(start),
        "exit": list(exit_pos),
        "actual_keys": [list(k) for k in sorted(actual_keys)],
        "discovered_key_order": [list(k) for k in discovered_key_order],
        "total_steps": total_steps,
        "unique_cells_visited": unique_cells_visited,
        "repeated_visits": repeated_visits,
        "backtracking_ratio": backtracking_ratio,
        "steps_before_all_keys": steps_before_all_keys,
        "steps_after_final_key": steps_after_final_key,
        "final_position": list(full_path[-1]),
        "path": [list(p) for p in full_path],
    }
    return result


def save_result(result: Dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


def print_summary(result: Dict, output_path: Path) -> None:
    print("Dijkstra traversal finished.")
    print(f"Success: {result['success']}")
    print(f"Start: {tuple(result['start'])}")
    print(f"Exit: {tuple(result['exit'])}")
    print(f"Actual keys: {[tuple(k) for k in result['actual_keys']]}")
    print(f"Discovered key order: {[tuple(k) for k in result['discovered_key_order']]}")
    print(f"Total steps: {result['total_steps']}")
    print(f"Unique cells visited: {result['unique_cells_visited']}")
    print(f"Repeated visits: {result['repeated_visits']}")
    print(f"Backtracking ratio: {result['backtracking_ratio']:.4f}")
    print(f"Steps before all keys collected: {result['steps_before_all_keys']}")
    print(f"Steps after final key to exit: {result['steps_after_final_key']}")
    print(f"Final position: {tuple(result['final_position'])}")
    print(f"Result saved to: {output_path}")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve() if args.input else default_input_path()
    output_path = Path(args.output).resolve() if args.output else default_output_path()

    data = load_maze(input_path)
    validate_maze(data)

    result = run_dijkstra_agent(data)
    save_result(result, output_path)
    print_summary(result, output_path)


if __name__ == "__main__":
    main()
