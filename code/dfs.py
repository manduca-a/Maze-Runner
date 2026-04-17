#!/usr/bin/env python3
"""
dfs.py

DFS maze solver for the CS5800 final project.

Task:
    Start at maze["start"], collect ALL keys (positions unknown to agent
    until stepped on), then reach maze["exit"].

Strategy:
    Phase-based DFS.  At each phase the agent does NOT know where the
    remaining keys are — it runs a full iterative DFS from its current
    position and stops as soon as it first steps on an undiscovered key.
    The path returned is the COMPLETE DFS traversal trace, including
    every backtracking move, faithfully representing what DFS visits.

    After all keys are found a final DFS phase navigates to the exit.

Path recorded:
    Every cell push AND every backtrack step is appended to the trace,
    so repeated_visits and repeated_visit_ratio will be noticeably higher
    than BFS — this is the expected and educationally interesting result.
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


def _bfs_distance(cells: List[List[int]], src: Coord, dst: Coord) -> int:
    """BFS shortest distance (steps), or -1 if unreachable."""
    if src == dst:
        return 0
    visited: Set[Coord] = {src}
    queue: deque[Tuple[Coord, int]] = deque([(src, 0)])
    while queue:
        cur, d = queue.popleft()
        for nb in _open_neighbors(cells, cur):
            if nb == dst:
                return d + 1
            if nb not in visited:
                visited.add(nb)
                queue.append((nb, d + 1))
    return -1


def _dfs_trace_to_nearest(
    cells: List[List[int]],
    src: Coord,
    targets: Set[Coord],
) -> Optional[List[Coord]]:
    """
    Iterative DFS from src.  Stop as soon as ANY cell in *targets* is
    first reached.

    Returns the FULL traversal trace — every forward step and every
    backtrack move is recorded — so the path length reflects true DFS
    behaviour, not just the solution length.

    Returns None if no target is reachable.
    """
    if src in targets:
        return [src]

    nb_cache: Dict[Coord, List[Coord]] = {}

    def neighbors(pos: Coord) -> List[Coord]:
        if pos not in nb_cache:
            nb_cache[pos] = _open_neighbors(cells, pos)
        return nb_cache[pos]

    stack: List[Tuple[Coord, int]] = [(src, 0)]
    visited: Set[Coord] = {src}
    trace: List[Coord] = [src]

    while stack:
        cur, nb_idx = stack[-1]
        nb_list = neighbors(cur)
        found_next = False

        while nb_idx < len(nb_list):
            nb = nb_list[nb_idx]
            nb_idx += 1
            if nb not in visited:
                # update iterator index for current frame
                stack[-1] = (cur, nb_idx)
                visited.add(nb)
                stack.append((nb, 0))
                trace.append(nb)

                if nb in targets:
                    return trace  

                found_next = True
                break

        if not found_next:
            stack.pop()
            if stack:
                parent_pos = stack[-1][0]
                trace.append(parent_pos)

    return None


def _dfs_trace_to_target(
    cells: List[List[int]],
    src: Coord,
    dst: Coord,
) -> Optional[List[Coord]]:
    """DFS trace to a single destination cell."""
    return _dfs_trace_to_nearest(cells, src, {dst})



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



def run_dfs_agent(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    DFS agent.

    The agent does NOT receive key positions in advance.  Each phase it
    runs DFS from its current cell and heads toward the FIRST key it
    encounters during depth-first expansion.  The full backtracking
    trace is recorded so statistics reflect true DFS behaviour.

    Key collection order is determined by DFS discovery order (whichever
    key DFS happens to reach first from the current position), making it
    genuinely different from BFS's nearest-first ordering.
    """
    cells: List[List[int]] = data["cells"]
    start: Coord           = tuple(data["start"])
    exit_pos: Coord        = tuple(data["exit"])
    key_list: List[Coord]  = [tuple(k) for k in data["keys"]]

    remaining: Set[Coord]   = set(key_list)
    full_path: List[Coord]  = [start]
    current: Coord          = start

    while remaining:
        sub = _dfs_trace_to_nearest(cells, current, remaining)
        if sub is None:
            raise RuntimeError(
                f"DFS: no reachable key from {current}. "
                "Check maze connectivity."
            )
        # sub[0] == current, skip to avoid duplicate
        full_path.extend(sub[1:])
        found_key = sub[-1]
        remaining.remove(found_key)
        current = found_key

    exit_sub = _dfs_trace_to_target(cells, current, exit_pos)
    if exit_sub is None:
        raise RuntimeError(
            f"DFS: exit unreachable from {current}. "
            "Check maze connectivity."
        )
    full_path.extend(exit_sub[1:])

    stats = _compute_stats(full_path, set(key_list), exit_pos)

    return {
        "algorithm":   "dfs",
        "path":        [[r, c] for r, c in full_path],
        "exit":        list(exit_pos),
        "actual_keys": [[r, c] for r, c in key_list],
        "success":     full_path[-1] == exit_pos,
        **stats,
    }
