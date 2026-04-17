#!/usr/bin/env python3
"""
astar.py

Author: Qizhen Dong

Read maze.json from the maze_runner root directory and complete the maze task
using WEIGHTED A* search.

New rules applied:
1. Weighted grid:
   - outer ring cells have weight 1
   - each deeper ring increases weight by +1
   - moving into a cell costs the weight of that destination cell
2. Directional search hint for A*:
   - at the start of each key-search phase, compute the current position's
     center-symmetric cell: (rows - 1 - r, cols - 1 - c)
   - this mirrored cell is used as a reference direction
   - when searching for the next key, ties are biased toward cells that are
     closer to this mirrored reference target

Unified search / routing rule:
1. Start from the entrance.
2. The exit position is known.
3. During the SEARCH phase, keys are treated as unknown targets:
   - run a weighted A*-style outward search from the current cell
   - stop as soon as the first remaining key is popped/settled
4. After a key is discovered, separately run weighted A* shortest-path search
   from the current cell to that key.
5. After all keys are collected, run weighted A* shortest-path search to exit.

This keeps:
- searching for which key to target next
- routing to that discovered target
as two separate stages.

Usage:
    python astar.py
    python astar.py --input ../maze.json
    python astar.py --output ../astar_result.json
"""

from __future__ import annotations

import argparse
import heapq
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


UP = 1
RIGHT = 2
DOWN = 4
LEFT = 8

Coord = Tuple[int, int]


def default_input_path() -> Path:
    script_path = Path(__file__).resolve()
    root_dir = script_path.parent.parent
    return root_dir / "maze.json"


def default_output_path() -> Path:
    script_path = Path(__file__).resolve()
    root_dir = script_path.parent.parent
    return root_dir / "astar_result.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Traverse the maze with weighted A* search and routing."
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
        help="Optional output path. Default: maze_runner/astar_result.json",
    )
    return parser.parse_args()


def load_maze(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"maze.json not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    required_keys = {"meta", "start", "exit", "keys", "cells"}
    missing = required_keys - set(data.keys())
    if missing:
        raise ValueError(f"maze.json is missing required fields: {sorted(missing)}")

    return data


def validate_maze(data: Dict[str, Any]) -> None:
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


def cell_weight(rows: int, cols: int, r: int, c: int) -> int:
    """
    Outer ring has weight 1.
    Each deeper ring increases weight by 1.
    """
    depth = min(r, c, rows - 1 - r, cols - 1 - c)
    return depth + 1


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


def move_cost(cells: List[List[int]], to_cell: Coord) -> int:
    rows, cols = len(cells), len(cells[0])
    r, c = to_cell
    return cell_weight(rows, cols, r, c)


def manhattan(a: Coord, b: Coord) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def mirrored_reference(rows: int, cols: int, pos: Coord) -> Coord:
    """
    Center-symmetric cell / opposite diagonal-like reference target.
    """
    r, c = pos
    return (rows - 1 - r, cols - 1 - c)


def weighted_astar_shortest_path(cells: List[List[int]], start: Coord, goal: Coord) -> List[Coord]:
    """
    Standard weighted A* shortest path.
    g(n): accumulated weighted travel cost
    h(n): Manhattan distance * 1 (admissible lower bound because minimum move cost is 1)
    """
    if start == goal:
        return [start]

    open_heap: List[Tuple[int, int, Coord]] = [(manhattan(start, goal), 0, start)]
    g_cost: Dict[Coord, int] = {start: 0}
    parent: Dict[Coord, Optional[Coord]] = {start: None}
    closed: Set[Coord] = set()

    while open_heap:
        _, g_cur, cur = heapq.heappop(open_heap)

        if cur in closed:
            continue
        closed.add(cur)

        if cur == goal:
            break

        for nxt in get_open_neighbors(cells, cur):
            tentative_g = g_cur + move_cost(cells, nxt)

            if nxt in closed and tentative_g >= g_cost.get(nxt, float("inf")):
                continue

            if tentative_g < g_cost.get(nxt, float("inf")):
                g_cost[nxt] = tentative_g
                parent[nxt] = cur
                f_nxt = tentative_g + manhattan(nxt, goal)
                heapq.heappush(open_heap, (f_nxt, tentative_g, nxt))

    if goal not in parent:
        raise ValueError(f"No path found from {start} to {goal}.")

    path: List[Coord] = []
    node: Optional[Coord] = goal
    while node is not None:
        path.append(node)
        node = parent[node]
    path.reverse()
    return path


def weighted_astar_search_next_key(
    cells: List[List[int]],
    start: Coord,
    remaining_keys: Set[Coord],
) -> Coord:
    """
    SEARCH phase:
    Run a weighted A*-style outward search from the current cell and stop
    as soon as the first remaining key is popped/settled.

    Search priority:
    1. smaller accumulated weighted travel cost g
    2. smaller Manhattan distance to the mirrored reference target
       for directional bias
    3. row/col tuple order naturally from heap tuple

    This makes search influenced by BOTH:
    - the weighted terrain
    - the mirrored reference direction
    """
    if start in remaining_keys:
        return start

    rows, cols = len(cells), len(cells[0])
    ref = mirrored_reference(rows, cols, start)

    open_heap: List[Tuple[int, int, Coord]] = [(0, manhattan(start, ref), start)]
    g_cost: Dict[Coord, int] = {start: 0}
    closed: Set[Coord] = set()

    while open_heap:
        _, _, cur = heapq.heappop(open_heap)

        if cur in closed:
            continue
        closed.add(cur)

        if cur in remaining_keys:
            return cur

        g_cur = g_cost[cur]
        for nxt in get_open_neighbors(cells, cur):
            tentative_g = g_cur + move_cost(cells, nxt)
            if tentative_g < g_cost.get(nxt, float("inf")):
                g_cost[nxt] = tentative_g
                directional_bias = manhattan(nxt, ref)
                heapq.heappush(open_heap, (tentative_g, directional_bias, nxt))

    raise RuntimeError(f"Weighted A* search could not find any remaining key from {start}.")


def path_weight_cost(cells: List[List[int]], path: List[Coord]) -> int:
    """
    Total weighted travel cost of a path.
    Starting cell contributes 0.
    Each move contributes the weight of the destination cell.
    """
    if len(path) <= 1:
        return 0

    total = 0
    for pos in path[1:]:
        total += move_cost(cells, pos)
    return total


def compute_stats(cells: List[List[int]], path: List[Coord], key_set: Set[Coord], exit_pos: Coord) -> Dict[str, Any]:
    visit_counts: Dict[Coord, int] = {}
    discovered_keys: List[Coord] = []
    seen_keys: Set[Coord] = set()
    step_keys: List[Optional[int]] = [None, None, None]
    weighted_cost_keys: List[Optional[int]] = [None, None, None]

    cumulative_cost = 0
    for step, pos in enumerate(path):
        visit_counts[pos] = visit_counts.get(pos, 0) + 1

        if step > 0:
            cumulative_cost += move_cost(cells, pos)

        if pos in key_set and pos not in seen_keys:
            seen_keys.add(pos)
            discovered_keys.append(pos)
            idx = len(discovered_keys) - 1
            if idx < 3:
                step_keys[idx] = step
                weighted_cost_keys[idx] = cumulative_cost

    total_steps = max(0, len(path) - 1)
    total_weighted_cost = path_weight_cost(cells, path)
    repeated_visits = sum(v - 1 for v in visit_counts.values() if v > 1)
    repeated_visit_ratio = repeated_visits / total_steps if total_steps > 0 else 0.0
    step_to_exit = total_steps if (path and path[-1] == exit_pos) else None
    weighted_cost_to_exit = total_weighted_cost if (path and path[-1] == exit_pos) else None
    avg_weight_per_step = total_weighted_cost / total_steps if total_steps > 0 else 0.0

    return {
        "discovered_key_order": [[r, c] for r, c in discovered_keys],
        "step_to_first_key": step_keys[0],
        "step_to_second_key": step_keys[1],
        "step_to_third_key": step_keys[2],
        "weighted_cost_to_first_key": weighted_cost_keys[0],
        "weighted_cost_to_second_key": weighted_cost_keys[1],
        "weighted_cost_to_third_key": weighted_cost_keys[2],
        "step_to_exit": step_to_exit,
        "weighted_cost_to_exit": weighted_cost_to_exit,
        "repeated_visits": repeated_visits,
        "repeated_visit_ratio": repeated_visit_ratio,
        "unique_cells_visited": len(visit_counts),
        "total_steps": total_steps,
        "total_weighted_cost": total_weighted_cost,
        "avg_weight_per_step": avg_weight_per_step,
    }


def run_astar_agent(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Weighted A* agent with separated search and routing phases.

    Search phase:
        weighted A*-style outward search until the first remaining key is reached,
        with directional bias toward the mirrored reference target

    Routing phase:
        weighted A* shortest path to that discovered key
    """
    cells: List[List[int]] = data["cells"]
    start: Coord = tuple(data["start"])
    exit_pos: Coord = tuple(data["exit"])
    key_list: List[Coord] = [tuple(k) for k in data["keys"]]

    remaining_keys: Set[Coord] = set(key_list)
    full_path: List[Coord] = [start]
    current: Coord = start

    while remaining_keys:
        discovered_key = weighted_astar_search_next_key(cells, current, remaining_keys)
        route_to_key = weighted_astar_shortest_path(cells, current, discovered_key)

        full_path.extend(route_to_key[1:])
        current = discovered_key
        remaining_keys.remove(discovered_key)

    exit_route = weighted_astar_shortest_path(cells, current, exit_pos)
    full_path.extend(exit_route[1:])

    rows, cols = len(cells), len(cells[0])
    reference_target = mirrored_reference(rows, cols, start)

    stats = compute_stats(cells, full_path, set(key_list), exit_pos)

    return {
        "algorithm": "astar_weighted",
        "path": [[r, c] for r, c in full_path],
        "exit": list(exit_pos),
        "actual_keys": [[r, c] for r, c in key_list],
        "success": full_path[-1] == exit_pos,
        "weight_rule": "outer ring = 1, each deeper ring +1, move cost = destination cell weight",
        "search_reference_rule": "each key-search phase uses current position's center-symmetric cell as directional reference",
        "initial_reference_target": [reference_target[0], reference_target[1]],
        **stats,
    }


def save_result(result: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


def print_summary(result: Dict[str, Any], output_path: Path) -> None:
    print("Weighted A* traversal finished.")
    print(f"Success: {result['success']}")
    print(f"Start: {tuple(result['path'][0])}")
    print(f"Exit: {tuple(result['exit'])}")
    print(f"Actual keys: {[tuple(k) for k in result['actual_keys']]}")
    print(f"Discovered key order: {[tuple(k) for k in result['discovered_key_order']]}")
    print(f"Steps to first key: {result['step_to_first_key']}")
    print(f"Steps to second key: {result['step_to_second_key']}")
    print(f"Steps to third key: {result['step_to_third_key']}")
    print(f"Weighted cost to first key: {result['weighted_cost_to_first_key']}")
    print(f"Weighted cost to second key: {result['weighted_cost_to_second_key']}")
    print(f"Weighted cost to third key: {result['weighted_cost_to_third_key']}")
    print(f"Steps to exit: {result['step_to_exit']}")
    print(f"Weighted cost to exit: {result['weighted_cost_to_exit']}")
    print(f"Total steps: {result['total_steps']}")
    print(f"Total weighted cost: {result['total_weighted_cost']}")
    print(f"Avg weight per step: {result['avg_weight_per_step']:.4f}")
    print(f"Unique cells visited: {result['unique_cells_visited']}")
    print(f"Repeated visits: {result['repeated_visits']}")
    print(f"Repeated visit ratio: {result['repeated_visit_ratio']:.4f}")
    print(f"Initial mirrored reference target: {tuple(result['initial_reference_target'])}")
    print(f"Result saved to: {output_path}")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve() if args.input else default_input_path()
    output_path = Path(args.output).resolve() if args.output else default_output_path()

    data = load_maze(input_path)
    validate_maze(data)

    result = run_astar_agent(data)
    save_result(result, output_path)
    print_summary(result, output_path)


if __name__ == "__main__":
    main()
