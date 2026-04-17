#!/usr/bin/env python3
"""
bfs.py

BFS maze solver for the CS5800 final project.

Task:
  Start at maze["start"], collect ALL keys, then reach maze["exit"].
  The agent does NOT have global knowledge of key positions — it reads
  them from maze data the same way a "known-map, unknown-goal-order" agent would.

Strategy (BFS):
  Phase-based BFS:
    Phase 0..N-1 : collect keys one by one (nearest uncollected key each phase)
    Phase N       : reach the exit

  Within each phase, BFS finds the shortest path from current position to
  the next target.  The sub-paths are concatenated (without duplicating the
  shared endpoint) to form the full path.

Return value (dict):
  {
    "algorithm":          "bfs",
    "path":               [[r, c], ...],   # full trajectory, start → ... → exit
    "exit":               [r, c],
    "actual_keys":        [[r, c], ...],   # all key positions used
    "discovered_key_order": [[r, c], ...], # order in which keys were collected
    "step_to_first_key":  int | None,
    "step_to_second_key": int | None,
    "step_to_third_key":  int | None,
    "step_to_exit":       int,
    "repeated_visits":    int,
    "repeated_visit_ratio": float,
    "unique_cells_visited": int,
    "total_steps":        int,
    "success":            bool,
  }

  maze_run.py will recompute any missing metrics from "path", so only
  "path", "exit", and "actual_keys" are strictly required.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple

# ── wall-bit constants (must match maze_generate.py) ──────────────────────────
UP    = 1
RIGHT = 2
DOWN  = 4
LEFT  = 8

Coord = Tuple[int, int]


# ── low-level helpers ──────────────────────────────────────────────────────────

def _open_neighbors(cells: List[List[int]], pos: Coord) -> List[Coord]:
    """Return all grid neighbors reachable from *pos* (no wall between them)."""
    rows, cols = len(cells), len(cells[0])
    r, c = pos
    result: List[Coord] = []

    if r > 0        and not (cells[r][c] & UP):    result.append((r - 1, c))
    if c < cols - 1 and not (cells[r][c] & RIGHT): result.append((r, c + 1))
    if r < rows - 1 and not (cells[r][c] & DOWN):  result.append((r + 1, c))
    if c > 0        and not (cells[r][c] & LEFT):  result.append((r, c - 1))

    return result


def _bfs_shortest_path(
    cells: List[List[int]],
    src: Coord,
    dst: Coord,
) -> Optional[List[Coord]]:
    """
    Classic BFS on the maze grid.
    Returns the shortest path [src, ..., dst], or None if unreachable.
    """
    if src == dst:
        return [src]

    parent: Dict[Coord, Optional[Coord]] = {src: None}
    queue: deque[Coord] = deque([src])

    while queue:
        cur = queue.popleft()
        for nb in _open_neighbors(cells, cur):
            if nb not in parent:
                parent[nb] = cur
                if nb == dst:
                    # Reconstruct path
                    path: List[Coord] = []
                    node: Optional[Coord] = dst
                    while node is not None:
                        path.append(node)
                        node = parent[node]
                    path.reverse()
                    return path
                queue.append(nb)

    return None  # unreachable


# ── statistics helper ──────────────────────────────────────────────────────────

def _compute_stats(
    path: List[Coord],
    key_set: Set[Coord],
    exit_pos: Coord,
) -> Dict[str, Any]:
    visit_counts: Dict[Coord, int] = {}
    seen_keys: List[Coord] = []
    discovered: Set[Coord] = set()

    step_keys = [None, None, None]   # steps to 1st / 2nd / 3rd key

    for step, pos in enumerate(path):
        visit_counts[pos] = visit_counts.get(pos, 0) + 1
        if pos in key_set and pos not in discovered:
            discovered.add(pos)
            seen_keys.append(pos)
            idx = len(seen_keys) - 1
            if idx < 3:
                step_keys[idx] = step

    total_steps = max(0, len(path) - 1)
    repeated    = sum(v - 1 for v in visit_counts.values() if v > 1)
    ratio       = repeated / total_steps if total_steps > 0 else 0.0

    step_exit = total_steps if (path and path[-1] == exit_pos) else None

    return {
        "discovered_key_order":  [[r, c] for r, c in seen_keys],
        "step_to_first_key":     step_keys[0],
        "step_to_second_key":    step_keys[1],
        "step_to_third_key":     step_keys[2],
        "step_to_exit":          step_exit,
        "repeated_visits":       repeated,
        "repeated_visit_ratio":  round(ratio, 6),
        "unique_cells_visited":  len(visit_counts),
        "total_steps":           total_steps,
    }


# ── main agent entry point ─────────────────────────────────────────────────────

def run_bfs_agent(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    BFS agent: collect all keys (nearest-first BFS each phase), then reach exit.

    Parameters
    ----------
    data : dict
        Parsed maze.json content.

    Returns
    -------
    dict
        Result dict compatible with maze_run.py's normalize_result().
    """
    cells: List[List[int]] = data["cells"]
    start: Coord           = tuple(data["start"])
    exit_pos: Coord        = tuple(data["exit"])
    key_list: List[Coord]  = [tuple(k) for k in data["keys"]]

    remaining_keys: List[Coord] = list(key_list)   # keys not yet collected
    full_path: List[Coord]      = [start]           # trajectory accumulator
    current: Coord              = start

    # ── Phase 1 : collect every key ───────────────────────────────────────────
    while remaining_keys:
        # Among remaining keys, pick the one with the shortest BFS distance.
        best_sub_path: Optional[List[Coord]] = None
        best_target:   Optional[Coord]       = None

        for key in remaining_keys:
            sub = _bfs_shortest_path(cells, current, key)
            if sub is not None:
                if best_sub_path is None or len(sub) < len(best_sub_path):
                    best_sub_path = sub
                    best_target   = key

        if best_sub_path is None:
            raise RuntimeError(f"BFS: unreachable key from {current}. Check maze connectivity.")

        # Append sub-path (skip first cell — already in full_path)
        full_path.extend(best_sub_path[1:])
        current = best_target
        remaining_keys.remove(best_target)

    # ── Phase 2 : reach the exit ───────────────────────────────────────────────
    exit_sub = _bfs_shortest_path(cells, current, exit_pos)
    if exit_sub is None:
        raise RuntimeError(f"BFS: exit unreachable from {current}. Check maze connectivity.")

    full_path.extend(exit_sub[1:])

    # ── Build result ───────────────────────────────────────────────────────────
    stats = _compute_stats(full_path, set(key_list), exit_pos)

    return {
        "algorithm":   "bfs",
        "path":        [[r, c] for r, c in full_path],
        "exit":        list(exit_pos),
        "actual_keys": [[r, c] for r, c in key_list],
        "success":     full_path[-1] == exit_pos,
        **stats,
    }