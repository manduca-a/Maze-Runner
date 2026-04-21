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


def run_bfs_optimal_agent(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    BFS Agent (UNIFIED / OPTIMAL APPROACH)
    
    A differenza dell'approccio 'a fasi', questo agente esplora l'intero 
    spazio degli stati usando la Tupla Booleana: S = ((r,c), (b1, b2, ...))
    Questo garantisce l'ottimalità globale del percorso.
    """
    cells: List[List[int]] = data["cells"]
    start: Coord           = tuple(data["start"])
    exit_pos: Coord        = tuple(data["exit"])
    key_list: List[Coord]  = [tuple(k) for k in data["keys"]]
    
    # Mappiamo ogni chiave a un indice fisso (es. chiave A è indice 0, B è 1...)
    key_idx_map = {pos: i for i, pos in enumerate(key_list)}
    num_keys = len(key_list)
    
    # STATO INIZIALE: Coordinate + Tupla di False
    init_bools = tuple([False] * num_keys)
    init_state = (start, init_bools)
    
    # La coda ora salva una tupla con (Stato_Logico, Percorso_Fino_A_Qui)
    from collections import deque
    queue = deque([(init_state, [start])])
    
    # EXPLORED SET: Salva interi stati logici (Proprietà di Markov rispettata)
    visited = {init_state}
    
    full_path = None

    while queue:
        (curr_pos, curr_bools), path = queue.popleft()
        
        # GOAL TEST: Sei all'uscita E hai tutte le chiavi (tutti True)
        if curr_pos == exit_pos and all(curr_bools):
            full_path = path
            break
            
        # Funzione Successore
        for nb in _open_neighbors(cells, curr_pos):
            # Copiamo la tupla booleana per non sporcare gli altri rami
            new_bools = list(curr_bools)
            
            # Se la cella vicina ha una chiave, aggiorniamo il suo booleano a True
            if nb in key_idx_map:
                new_bools[key_idx_map[nb]] = True
                
            next_state = (nb, tuple(new_bools))
            
            # GRAPH SEARCH DUPLICATE DETECTION SAFE
            # Passare sulla cella (5,5) con 0 chiavi è un nodo diverso
            # dal passare sulla cella (5,5) con 1 chiave!
            if next_state not in visited:
                visited.add(next_state)
                queue.append((next_state, path + [nb]))

    if full_path is None:
        raise RuntimeError("BFS: nessun percorso ottimale globale trovato.")

    # Calcoliamo le statistiche usando la funzione già presente nel file originale
    stats = _compute_stats(full_path, set(key_list), exit_pos)

    return {
        "algorithm":   "bfs",
        "path":        [[r, c] for r, c in full_path],
        "exit":        list(exit_pos),
        "actual_keys": [[r, c] for r, c in key_list],
        "success":     full_path[-1] == exit_pos,
        **stats,
    }