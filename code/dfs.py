#!/usr/bin/env python3
"""
dfs.py

DFS maze solver for the CS5800 final project.

Strategy (DFS):
  Same phase structure as bfs.py — collect keys one by one (nearest by BFS
  distance, because DFS path-length is not meaningful for "nearest"), then
  reach the exit — but the path within each phase is produced by an iterative
  DFS that records the actual traversal order (including backtracking steps),
  so the path faithfully represents what DFS *visits*, not just the solution.

  This gives DFS higher repeated_visits and longer total_steps than BFS on
  the same maze, which is the expected and interesting comparison result.

Why "nearest key" for phase ordering?
  DFS doesn't optimise distance, so the order in which keys are targeted
  doesn't affect the DFS path quality — using nearest-by-BFS keeps the
  comparison fair (same key-collection order as bfs.py) so that differences
  in the statistics come purely from the traversal strategy.

Return value: same schema as bfs.py (compatible with maze_run.py).
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple

UP    = 1
RIGHT = 2
DOWN  = 4
LEFT  = 8

Coord = Tuple[int, int]


# ── low-level helpers (identical to bfs.py) ────────────────────────────────────

def _open_neighbors(cells: List[List[int]], pos: Coord) -> List[Coord]:
    rows, cols = len(cells), len(cells[0])
    r, c = pos
    result: List[Coord] = []

    if r > 0        and not (cells[r][c] & UP):    result.append((r - 1, c))
    if c < cols - 1 and not (cells[r][c] & RIGHT): result.append((r, c + 1))
    if r < rows - 1 and not (cells[r][c] & DOWN):  result.append((r + 1, c))
    if c > 0        and not (cells[r][c] & LEFT):  result.append((r, c - 1))

    return result


def _bfs_distance(cells: List[List[int]], src: Coord, dst: Coord) -> int:
    """Return BFS shortest-path length (number of steps), or -1 if unreachable."""
    if src == dst:
        return 0
    visited: Set[Coord] = {src}
    queue: deque[Tuple[Coord, int]] = deque([(src, 0)])
    while queue:
        cur, dist = queue.popleft()
        for nb in _open_neighbors(cells, cur):
            if nb == dst:
                return dist + 1
            if nb not in visited:
                visited.add(nb)
                queue.append((nb, dist + 1))
    return -1


# ── DFS traversal (with full backtracking trace) ───────────────────────────────

def _dfs_path_with_trace(
    cells: List[List[int]],
    src: Coord,
    dst: Coord,
) -> Optional[List[Coord]]:
    """
    Iterative DFS from *src* to *dst*.

    Returns the *full traversal trace* — every cell push and every backtrack
    step is recorded — so the returned list accurately reflects what DFS
    visits, including backtracking.  The list starts at src and ends at dst.

    Returns None if dst is unreachable.
    """
    if src == dst:
        return [src]

    # Stack entries: (position, index-of-next-neighbor-to-try)
    # We store the neighbor list alongside so we can resume from where we left off.
    neighbor_cache: Dict[Coord, List[Coord]] = {}

    def neighbors(pos: Coord) -> List[Coord]:
        if pos not in neighbor_cache:
            neighbor_cache[pos] = _open_neighbors(cells, pos)
        return neighbor_cache[pos]

    # DFS stack: list of (coord, iterator-index)
    stack: List[Tuple[Coord, int]] = [(src, 0)]
    visited: Set[Coord] = {src}

    # Full trace (every push and pop recorded as a sequence of positions)
    trace: List[Coord] = [src]

    while stack:
        cur, nb_idx = stack[-1]

        # Find next unvisited neighbor
        nb_list = neighbors(cur)
        found = False
        while nb_idx < len(nb_list):
            nb = nb_list[nb_idx]
            nb_idx += 1
            if nb not in visited:
                # Update the iterator index on the current frame
                stack[-1] = (cur, nb_idx)

                visited.add(nb)
                stack.append((nb, 0))
                trace.append(nb)

                if nb == dst:
                    return trace

                found = True
                break

        if not found:
            # Backtrack: pop current cell and record the step back to parent
            stack.pop()
            if stack:
                parent_pos = stack[-1][0]
                trace.append(parent_pos)   # ← backtrack move recorded in trace

    return None  # dst unreachable


# ── statistics helper (identical to bfs.py) ───────────────────────────────────

def _compute_stats(
    path: List[Coord],
    key_set: Set[Coord],
    exit_pos: Coord,
) -> Dict[str, Any]:
    visit_counts: Dict[Coord, int] = {}
    seen_keys: List[Coord] = []
    discovered: Set[Coord] = set()
    step_keys = [None, None, None]

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
    step_exit   = total_steps if (path and path[-1] == exit_pos) else None

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

def run_dfs_agent(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    DFS agent: collect all keys (nearest-first ordering), then reach exit.
    Each sub-path is produced by DFS with full backtracking trace.
    """
    cells: List[List[int]] = data["cells"]
    start: Coord           = tuple(data["start"])
    exit_pos: Coord        = tuple(data["exit"])
    key_list: List[Coord]  = [tuple(k) for k in data["keys"]]

    remaining_keys: List[Coord] = list(key_list)
    full_path: List[Coord]      = [start]
    current: Coord              = start

    # ── Phase 1 : collect every key ───────────────────────────────────────────
    while remaining_keys:
        # Nearest key by BFS distance (keeps phase order comparable to BFS)
        nearest     = min(remaining_keys, key=lambda k: _bfs_distance(cells, current, k))
        sub_path    = _dfs_path_with_trace(cells, current, nearest)

        if sub_path is None:
            raise RuntimeError(f"DFS: unreachable key {nearest} from {current}.")

        full_path.extend(sub_path[1:])   # skip first cell (already recorded)
        current = nearest
        remaining_keys.remove(nearest)

    # ── Phase 2 : reach the exit ───────────────────────────────────────────────
    exit_sub = _dfs_path_with_trace(cells, current, exit_pos)
    if exit_sub is None:
        raise RuntimeError(f"DFS: exit unreachable from {current}.")

    full_path.extend(exit_sub[1:])

    # ── Build result ───────────────────────────────────────────────────────────
    stats = _compute_stats(full_path, set(key_list), exit_pos)

    return {
        "algorithm":   "dfs",
        "path":        [[r, c] for r, c in full_path],
        "exit":        list(exit_pos),
        "actual_keys": [[r, c] for r, c in key_list],
        "success":     full_path[-1] == exit_pos,
        **stats,
    }