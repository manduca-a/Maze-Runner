#!/usr/bin/env python3
"""
bfs.py

Author: Yanzhe Xi

BFS maze solver for the CS5800 final project.

Task:
    Start at maze["start"], collect ALL keys (positions unknown to agent
    until stepped on), then reach maze["exit"].

Strategy:
    Phase-based BFS.  At each phase the agent does NOT know where the
    remaining keys are — it runs a full BFS from its current position
    over the entire reachable maze and returns the shortest path to the
    NEAREST undiscovered key it encounters during that search.  After
    all keys are collected a final BFS phase navigates to the exit.

    Because BFS visits cells in breadth-first order the agent always
    reaches the closest key first, and the path recorded is the true
    shortest path — zero backtracking, minimal repeated visits.

Path recorded:
    The full concatenation of every phase's sub-path, start → key1 →
    key2 → key3 → exit.  maze_run.py uses this path for animation and
    statistics.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple

UP    = 1
RIGHT = 2
DOWN  = 4
LEFT  = 8

Coord = Tuple[int, int]


def _open_neighbors(cells: List[List[int]], pos: Coord) -> List[Coord]:
    """Return all grid neighbors reachable from pos (no wall between them)."""
    rows, cols = len(cells), len(cells[0])
    r, c = pos
    result: List[Coord] = []
    if r > 0        and not (cells[r][c] & UP):    result.append((r - 1, c))
    if c < cols - 1 and not (cells[r][c] & RIGHT): result.append((r, c + 1))
    if r < rows - 1 and not (cells[r][c] & DOWN):  result.append((r + 1, c))
    if c > 0        and not (cells[r][c] & LEFT):  result.append((r, c - 1))
    return result


def _bfs_to_nearest(
    cells: List[List[int]],
    src: Coord,
    targets: Set[Coord],
) -> Optional[List[Coord]]:
    """
    BFS from src.  Stop as soon as ANY cell in *targets* is reached.
    Returns the shortest path [src, ..., target], or None if unreachable.
    This simulates a blind agent: it does not know which target is closer
    in advance — it simply expands outward and stops at the first hit.
    """
    if src in targets:
        return [src]

    parent: Dict[Coord, Optional[Coord]] = {src: None}
    queue: deque[Coord] = deque([src])

    while queue:
        cur = queue.popleft()
        for nb in _open_neighbors(cells, cur):
            if nb not in parent:
                parent[nb] = cur
                if nb in targets:
                    # reconstruct path
                    path: List[Coord] = []
                    node: Optional[Coord] = nb
                    while node is not None:
                        path.append(node)
                        node = parent[node]
                    path.reverse()
                    return path
                queue.append(nb)

    return None


def _bfs_shortest_path(
    cells: List[List[int]],
    src: Coord,
    dst: Coord,
) -> Optional[List[Coord]]:
    """BFS shortest path from src to a single destination."""
    return _bfs_to_nearest(cells, src, {dst})



def _compute_stats(
    path: List[Coord],
    key_set: Set[Coord],
    exit_pos: Coord,
) -> Dict[str, Any]:
    visit_counts: Dict[Coord, int] = {}
    seen_keys: List[Coord] = []
    discovered: Set[Coord] = set()
    step_keys: List[Optional[int]] = [None, None, None]

    for step, pos in enumerate(path):
        visit_counts[pos] = visit_counts.get(pos, 0) + 1
        if pos in key_set and pos not in discovered:
            discovered.add(pos)
            seen_keys.append(pos)
            idx = len(seen_keys) - 1
            if idx < 3:
                step_keys[idx] = step

    total    = max(0, len(path) - 1)
    repeated = sum(v - 1 for v in visit_counts.values() if v > 1)
    ratio    = repeated / total if total > 0 else 0.0
    step_exit = total if (path and path[-1] == exit_pos) else None

    return {
        "discovered_key_order":  [[r, c] for r, c in seen_keys],
        "step_to_first_key":     step_keys[0],
        "step_to_second_key":    step_keys[1],
        "step_to_third_key":     step_keys[2],
        "step_to_exit":          step_exit,
        "repeated_visits":       repeated,
        "repeated_visit_ratio":  round(ratio, 6),
        "unique_cells_visited":  len(visit_counts),
        "total_steps":           total,
    }


def run_bfs_agent(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    BFS agent.

    The agent does NOT receive key positions in advance.  Each phase it
    runs BFS from its current cell and heads toward the nearest key it
    finds during the breadth-first expansion.  After all keys are
    collected it runs one final BFS to the exit.
    """
    cells: List[List[int]] = data["cells"]
    start: Coord           = tuple(data["start"])
    exit_pos: Coord        = tuple(data["exit"])
    key_list: List[Coord]  = [tuple(k) for k in data["keys"]]

    remaining: Set[Coord]   = set(key_list)
    full_path: List[Coord]  = [start]
    current: Coord          = start

    while remaining:
        sub = _bfs_to_nearest(cells, current, remaining)
        if sub is None:
            raise RuntimeError(
                f"BFS: no reachable key from {current}. "
                "Check maze connectivity."
            )
        # sub[0] == current, skip to avoid duplicate
        full_path.extend(sub[1:])
        found_key = sub[-1]
        remaining.remove(found_key)
        current = found_key

    exit_sub = _bfs_shortest_path(cells, current, exit_pos)
    if exit_sub is None:
        raise RuntimeError(
            f"BFS: exit unreachable from {current}. "
            "Check maze connectivity."
        )
    full_path.extend(exit_sub[1:])
    
    stats = _compute_stats(full_path, set(key_list), exit_pos)

    return {
        "algorithm":   "bfs",
        "path":        [[r, c] for r, c in full_path],
        "exit":        list(exit_pos),
        "actual_keys": [[r, c] for r, c in key_list],
        "success":     full_path[-1] == exit_pos,
        **stats,
    }
