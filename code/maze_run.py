#!/usr/bin/env python3
"""
maze_run.py

Visualize maze traversal results for BFS / DFS / Dijkstra / A* in a pygame window.

Layout:
  - Default: 2x2 grid, each algorithm gets its own maze panel with stats below
  - Press 1/2/3/4 to focus on a single algorithm (full-screen panel)
  - Press 0 to return to 2x2 grid view

Features:
1. Different colors for different algorithms:
   - BFS:      red
   - DFS:      green
   - Dijkstra: yellow
   - A*:       blue
2. Animate traversal paths simultaneously across all panels.
3. Per-panel statistics bar showing key/exit steps and repeat ratio.
4. Support command-line filtering:
   - python maze_run.py --run dijkstra
5. Keyboard controls:
   - 1/2/3/4 : focus single algorithm
   - 0       : back to 2x2 grid
   - Space   : pause / resume
   - N       : restart animation
   - G       : toggle grid
   - ESC     : quit
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pygame


UP    = 1
RIGHT = 2
DOWN  = 4
LEFT  = 8

Coord = Tuple[int, int]

ALGO_ORDER = ["bfs", "dfs", "dijkstra", "astar"]
ALGO_DISPLAY_NAME = {
    "bfs":      "BFS",
    "dfs":      "DFS",
    "dijkstra": "Dijkstra",
    "astar":    "A*",
}
ALGO_COLORS = {
    "bfs":      (229,  57,  53),   # red
    "dfs":      ( 67, 160,  71),   # green
    "dijkstra": (251, 192,  45),   # yellow
    "astar":    ( 66, 133, 244),   # blue
}

BACKGROUND_COLOR = (230, 230, 230)
CELL_COLOR       = (255, 255, 255)
WALL_COLOR       = ( 20,  20,  20)
GRID_COLOR       = (220, 220, 220)
TEXT_COLOR       = ( 20,  20,  20)
TEXT_LIGHT       = (255, 255, 255)
START_COLOR      = ( 76, 175,  80)
EXIT_COLOR       = (229,  57,  53)
KEY_COLOR        = (255, 193,   7)
HUD_BG           = (255, 255, 255, 240)
PANEL_BORDER     = (180, 180, 180)
STAT_BG          = (245, 245, 245)

DEFAULT_WINDOW_WIDTH  = 1500
DEFAULT_WINDOW_HEIGHT = 960
MIN_CELL_SIZE         = 2.0
MAX_CELL_SIZE         = 120.0
TOP_HUD_HEIGHT        = 64       # title + controls bar
PANEL_GAP             = 10       # gap between panels and screen edges
STAT_BAR_HEIGHT       = 96       # stats strip at bottom of each panel
MIN_ANIM_SECONDS      = 3.0
MAX_ANIM_SECONDS      = 8.0



def default_input_path() -> Path:
    script_path = Path(__file__).resolve()
    return script_path.parent.parent / "maze.json"


def default_code_dir() -> Path:
    return Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run and visualize maze algorithms.")
    parser.add_argument("--input",  type=str,   default=None)
    parser.add_argument("--run",    type=str,   default=None, choices=ALGO_ORDER)
    parser.add_argument("--width",  type=int,   default=DEFAULT_WINDOW_WIDTH)
    parser.add_argument("--height", type=int,   default=DEFAULT_WINDOW_HEIGHT)
    parser.add_argument("--speed",  type=float, default=1.0)
    return parser.parse_args()


def load_maze(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"maze.json not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    required = {"meta", "start", "exit", "keys", "cells"}
    missing = required - set(data.keys())
    if missing:
        raise ValueError(f"maze.json missing fields: {sorted(missing)}")
    return data


def validate_maze(data: Dict[str, Any]) -> None:
    rows, cols = data["meta"]["rows"], data["meta"]["cols"]
    cells = data["cells"]
    if len(cells) != rows:
        raise ValueError("Row count mismatch.")
    for row in cells:
        if len(row) != cols:
            raise ValueError("Col count mismatch.")
    specials = [tuple(data["start"]), tuple(data["exit"])] + [tuple(k) for k in data["keys"]]
    for r, c in specials:
        if not (0 <= r < rows and 0 <= c < cols):
            raise ValueError(f"Special cell {(r,c)} out of bounds.")
    if len(set(specials)) != len(specials):
        raise ValueError("Start, exit, and keys must all be distinct.")


def normalize_path(path_value: Any) -> List[Coord]:
    if not isinstance(path_value, list) or not path_value:
        raise ValueError("Result path must be a non-empty list.")
    return [(int(p[0]), int(p[1])) for p in path_value]


def compute_stats_from_path(
    path: List[Coord], key_set: Set[Coord], exit_pos: Coord
) -> Dict[str, Any]:
    visit_counts: Dict[Coord, int] = {}
    discovered: List[Coord] = []
    seen: Set[Coord] = set()
    step_keys: List[Optional[int]] = [None, None, None]

    for step, pos in enumerate(path):
        visit_counts[pos] = visit_counts.get(pos, 0) + 1
        if pos in key_set and pos not in seen:
            seen.add(pos)
            discovered.append(pos)
            idx = len(discovered) - 1
            if idx < 3:
                step_keys[idx] = step

    total = max(0, len(path) - 1)
    repeated = sum(v - 1 for v in visit_counts.values() if v > 1)
    ratio = repeated / total if total > 0 else 0.0
    step_exit = total if (path and path[-1] == exit_pos) else None

    return {
        "path": [[r, c] for r, c in path],
        "discovered_key_order": [[r, c] for r, c in discovered],
        "step_to_first_key":    step_keys[0],
        "step_to_second_key":   step_keys[1],
        "step_to_third_key":    step_keys[2],
        "step_to_exit":         step_exit,
        "repeated_visits":      repeated,
        "repeated_visit_ratio": repeated_visit_ratio if (repeated_visit_ratio := ratio) else 0.0,
        "unique_cells_visited": len(visit_counts),
        "total_steps":          total,
    }


def normalize_result(
    name: str, raw: Dict[str, Any], maze_data: Dict[str, Any]
) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"{name}: result must be a dict.")
    path = normalize_path(raw.get("path"))
    start = tuple(maze_data["start"])
    if path[0] != start:
        raise ValueError(f"{name}: path must start at {start}, got {path[0]}.")
    exit_pos = tuple(raw.get("exit", maze_data["exit"]))
    actual_keys = {tuple(k) for k in raw.get("actual_keys", maze_data["keys"])}
    computed = compute_stats_from_path(path, actual_keys, exit_pos)
    return {
        "algorithm":            raw.get("algorithm", name),
        "display_name":         ALGO_DISPLAY_NAME[name],
        "color":                ALGO_COLORS[name],
        "path":                 computed["path"],
        "actual_keys":          [list(k) for k in sorted(actual_keys)],
        "discovered_key_order": raw.get("discovered_key_order", computed["discovered_key_order"]),
        "step_to_first_key":    raw.get("step_to_first_key",  computed["step_to_first_key"]),
        "step_to_second_key":   raw.get("step_to_second_key", computed["step_to_second_key"]),
        "step_to_third_key":    raw.get("step_to_third_key",  computed["step_to_third_key"]),
        "step_to_exit":         raw.get("step_to_exit",       computed["step_to_exit"]),
        "repeated_visits":      raw.get("repeated_visits",    computed["repeated_visits"]),
        "repeated_visit_ratio": raw.get("repeated_visit_ratio", computed["repeated_visit_ratio"]),
        "unique_cells_visited": raw.get("unique_cells_visited", computed["unique_cells_visited"]),
        "total_steps":          raw.get("total_steps",        computed["total_steps"]),
        "success":              raw.get("success", path[-1] == exit_pos),
    }


def load_algorithm_module(module_path: Path, import_name: str):
    spec = importlib.util.spec_from_file_location(import_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def find_runner_function(module: Any, algo_name: str):
    for name in [f"run_{algo_name}_agent", "run_algorithm", "solve"]:
        fn = getattr(module, name, None)
        if callable(fn):
            return fn
    return None


def run_available_algorithms(
    maze_data: Dict[str, Any],
    code_dir: Path,
    requested_algo: Optional[str],
) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    selected = [requested_algo] if requested_algo else ALGO_ORDER

    for algo_name in selected:
        module_path = code_dir / f"{algo_name}.py"
        if not module_path.exists():
            if requested_algo:
                raise FileNotFoundError(f"Algorithm file not found: {module_path}")
            continue
        module = load_algorithm_module(module_path, f"maze_run_{algo_name}")
        runner = find_runner_function(module, algo_name)
        if runner is None:
            if requested_algo:
                raise AttributeError(f"{algo_name}.py has no supported runner function.")
            continue
        raw = runner(maze_data)
        results[algo_name] = normalize_result(algo_name, raw, maze_data)

    if requested_algo and requested_algo not in results:
        raise RuntimeError(f"Could not run: {requested_algo}")
    if not results:
        raise RuntimeError("No runnable algorithm found (bfs.py / dfs.py / dijkstra.py / astar.py).")
    return results

#  Viewer

class MazeRunViewer:

    def __init__(
        self,
        maze_data: Dict[str, Any],
        results: Dict[str, Dict[str, Any]],
        window_width: int,
        window_height: int,
        speed_multiplier: float,
    ) -> None:
        self.data            = maze_data
        self.results         = results
        self.available_algos = [n for n in ALGO_ORDER if n in results]

        self.rows  = maze_data["meta"]["rows"]
        self.cols  = maze_data["meta"]["cols"]
        self.cells: List[List[int]] = maze_data["cells"]
        self.start: Coord = tuple(maze_data["start"])
        self.exit:  Coord = tuple(maze_data["exit"])
        self.keys:  List[Coord] = [tuple(k) for k in maze_data["keys"]]

        self.speed_multiplier = max(0.1, speed_multiplier)

        pygame.init()
        pygame.display.set_caption("Maze Run Viewer")
        self.screen = pygame.display.set_mode(
            (max(800, window_width), max(600, window_height)), pygame.RESIZABLE
        )
        self.clock = pygame.time.Clock()

        self.font       = pygame.font.SysFont(None, 22)
        self.small_font = pygame.font.SysFont(None, 18)
        self.big_font   = pygame.font.SysFont(None, 28)
        self.mono_font  = pygame.font.SysFont("consolas", 17)

        self.show_grid          = True
        self.current_filter     = "all"   # "all" | algo_name
        self.animation_start_time = time.time()
        self.animation_paused   = False
        self.pause_elapsed      = 0.0


    def _anim_duration(self, algo_name: str) -> float:
        steps = self.results[algo_name]["total_steps"]
        base  = max(MIN_ANIM_SECONDS, min(MAX_ANIM_SECONDS, 0.005 * steps + 3.0))
        return base / self.speed_multiplier

    def _anim_progress(self, algo_name: str) -> int:
        path_len = len(self.results[algo_name]["path"])
        duration = self._anim_duration(algo_name)
        elapsed  = self.pause_elapsed if self.animation_paused \
                   else time.time() - self.animation_start_time
        ratio = min(1.0, max(0.0, elapsed / duration)) if duration > 0 else 1.0
        return max(1, int(ratio * path_len))

    def restart_animation(self) -> None:
        self.animation_start_time = time.time()
        self.pause_elapsed = 0.0
        self.animation_paused = False

    def toggle_pause(self) -> None:
        if not self.animation_paused:
            self.pause_elapsed = time.time() - self.animation_start_time
            self.animation_paused = True
        else:
            self.animation_start_time = time.time() - self.pause_elapsed
            self.animation_paused = False

    def _visible_algos(self) -> List[str]:
        if self.current_filter == "all":
            return self.available_algos
        if self.current_filter in self.available_algos:
            return [self.current_filter]
        return []

    def _panel_rects(self) -> Dict[str, pygame.Rect]:
        """
        Return the bounding Rect for each visible algorithm's panel.
        Single algo → one full-canvas panel.
        Multiple    → 2×2 grid (or 1×N for fewer than 4).
        """
        sw = self.screen.get_width()
        sh = self.screen.get_height()
        canvas_top = TOP_HUD_HEIGHT
        canvas_h   = sh - canvas_top

        visible = self._visible_algos()

        if len(visible) == 1:
            name = visible[0]
            return {name: pygame.Rect(0, canvas_top, sw, canvas_h)}

        # 2 columns, up to 2 rows
        n_cols = 2
        n_rows = (len(visible) + 1) // 2

        total_w = sw  - PANEL_GAP * (n_cols + 1)
        total_h = canvas_h - PANEL_GAP * (n_rows + 1)
        pw = total_w  // n_cols
        ph = total_h  // n_rows

        rects: Dict[str, pygame.Rect] = {}
        for idx, name in enumerate(visible):
            row = idx // n_cols
            col = idx % n_cols
            x = PANEL_GAP + col * (pw + PANEL_GAP)
            y = canvas_top + PANEL_GAP + row * (ph + PANEL_GAP)
            rects[name] = pygame.Rect(x, y, pw, ph)
        return rects

    def _draw_panel(self, algo_name: str, panel: pygame.Rect) -> None:
        """Draw one complete maze panel (maze + path + stat bar) inside *panel*."""
        result   = self.results[algo_name]
        color    = result["color"]
        path     = [tuple(p) for p in result["path"]]
        progress = self._anim_progress(algo_name)
        sub_path = path[:progress]

        is_solo  = (self.current_filter != "all")
        stat_h   = 0 if is_solo else STAT_BAR_HEIGHT

        maze_area = pygame.Rect(panel.x, panel.y, panel.width, panel.height - stat_h)
        stat_area = pygame.Rect(panel.x, panel.y + panel.height - stat_h,
                                panel.width, stat_h)

        pygame.draw.rect(self.screen, CELL_COLOR, maze_area)
        pygame.draw.rect(self.screen, PANEL_BORDER, panel, 1)

        margin = 8
        cs = max(MIN_CELL_SIZE, min(MAX_CELL_SIZE,
            min((maze_area.width  - margin * 2) / self.cols,
                (maze_area.height - margin * 2) / self.rows)))
        mw = self.cols * cs
        mh = self.rows * cs
        ox = maze_area.x + (maze_area.width  - mw) / 2
        oy = maze_area.y + (maze_area.height - mh) / 2

        def cc(r: int, c: int) -> Tuple[int, int]:
            return (int(round(ox + (c + 0.5) * cs)),
                    int(round(oy + (r + 0.5) * cs)))

        if self.show_grid and cs >= 8:
            for gc in range(self.cols + 1):
                x = int(round(ox + gc * cs))
                pygame.draw.line(self.screen, GRID_COLOR,
                                 (x, int(oy)),
                                 (x, int(round(oy + self.rows * cs))), 1)
            for gr in range(self.rows + 1):
                y = int(round(oy + gr * cs))
                pygame.draw.line(self.screen, GRID_COLOR,
                                 (int(ox), y),
                                 (int(round(ox + self.cols * cs)), y), 1)
                
        for r in range(self.rows):
            for c in range(self.cols):
                cx, cy = cc(r, c)
                pad = max(1, int(cs * 0.13))
                inner = pygame.Rect(
                    int(round(ox + c * cs)) + pad,
                    int(round(oy + r * cs)) + pad,
                    max(1, int(cs) - pad * 2),
                    max(1, int(cs) - pad * 2),
                )
                if (r, c) == self.start:
                    pygame.draw.rect(self.screen, START_COLOR, inner)
                elif (r, c) == self.exit:
                    pygame.draw.rect(self.screen, EXIT_COLOR, inner)
                elif (r, c) in self.keys:
                    pygame.draw.circle(self.screen, KEY_COLOR,
                                       (cx, cy), max(2, int(cs * 0.20)))

                if cs >= 14:
                    label = lc = None
                    if   (r, c) == self.start: label, lc = "I", TEXT_LIGHT
                    elif (r, c) == self.exit:  label, lc = "O", TEXT_LIGHT
                    elif (r, c) in self.keys:  label, lc = "K", (70, 50, 0)
                    if label:
                        fs = max(10, min(26, int(cs * 0.50)))
                        lf = pygame.font.SysFont(None, fs, bold=True)
                        s  = lf.render(label, True, lc)
                        self.screen.blit(s, s.get_rect(center=(cx, cy)))

        ww = 1 if cs < 20 else 2 if cs < 60 else 3
        for r in range(self.rows):
            for c in range(self.cols):
                x0 = int(round(ox + c       * cs))
                y0 = int(round(oy + r       * cs))
                x1 = int(round(ox + (c + 1) * cs))
                y1 = int(round(oy + (r + 1) * cs))
                cell = self.cells[r][c]
                if cell & UP:    pygame.draw.line(self.screen, WALL_COLOR, (x0,y0),(x1,y0), ww)
                if cell & RIGHT: pygame.draw.line(self.screen, WALL_COLOR, (x1,y0),(x1,y1), ww)
                if cell & DOWN:  pygame.draw.line(self.screen, WALL_COLOR, (x0,y1),(x1,y1), ww)
                if cell & LEFT:  pygame.draw.line(self.screen, WALL_COLOR, (x0,y0),(x0,y1), ww)

        if len(sub_path) >= 2:
            line_w = max(1, int(cs * 0.18))
            points = [cc(r, c) for r, c in sub_path]
            pygame.draw.lines(self.screen, color, False, points, line_w)

        if sub_path:
            hr, hc = sub_path[-1]
            pygame.draw.circle(self.screen, color, cc(hr, hc),
                                max(3, int(cs * 0.22)))

        dot_x = maze_area.x + 10
        dot_y = maze_area.y + 12
        pygame.draw.circle(self.screen, color, (dot_x, dot_y), 7)
        name_surf = self.big_font.render(
            ALGO_DISPLAY_NAME[algo_name], True, TEXT_COLOR)
        self.screen.blit(name_surf, (dot_x + 14, dot_y - name_surf.get_height() // 2))

        if stat_h > 0:
            self._draw_stat_bar(algo_name, stat_area)

    def _draw_stat_bar(self, algo_name: str, area: pygame.Rect) -> None:
        """Draw the statistics strip at the bottom of a panel."""
        pygame.draw.rect(self.screen, STAT_BG, area)
        pygame.draw.line(self.screen, PANEL_BORDER,
                         (area.x, area.y), (area.right, area.y), 1)

        result = self.results[algo_name]

        def sv(v: Any) -> str:
            if v is None:       return "—"
            if isinstance(v, float): return f"{v:.3f}"
            return str(v)

        stats_row1 = [
            ("1st key",  sv(result["step_to_first_key"])),
            ("2nd key",  sv(result["step_to_second_key"])),
            ("3rd key",  sv(result["step_to_third_key"])),
            ("Exit",     sv(result["step_to_exit"])),
        ]
        stats_row2 = [
            ("Total steps",   sv(result["total_steps"])),
            ("Unique cells",  sv(result["unique_cells_visited"])),
            ("Repeats",       sv(result["repeated_visits"])),
            ("Repeat ratio",  sv(result["repeated_visit_ratio"])),
        ]

        col_w = area.width // 4
        row1_y = area.y + 10
        row2_y = area.y + 48

        for i, (label, value) in enumerate(stats_row1):
            x = area.x + i * col_w + col_w // 2
            lbl_s = self.small_font.render(label, True, (120, 120, 120))
            val_s = self.font.render(value, True, TEXT_COLOR)
            self.screen.blit(lbl_s, lbl_s.get_rect(centerx=x, y=row1_y))
            self.screen.blit(val_s, val_s.get_rect(centerx=x, y=row1_y + 18))

        for i, (label, value) in enumerate(stats_row2):
            x = area.x + i * col_w + col_w // 2
            lbl_s = self.small_font.render(label, True, (120, 120, 120))
            val_s = self.font.render(value, True, TEXT_COLOR)
            self.screen.blit(lbl_s, lbl_s.get_rect(centerx=x, y=row2_y))
            self.screen.blit(val_s, val_s.get_rect(centerx=x, y=row2_y + 18))


    def _draw_solo_sidebar(self, algo_name: str) -> None:
        """In single-algo full-screen mode, draw a stats sidebar on the right."""
        sw = self.screen.get_width()
        sh = self.screen.get_height()
        sidebar_w = 280
        rect = pygame.Rect(sw - sidebar_w, TOP_HUD_HEIGHT,
                           sidebar_w, sh - TOP_HUD_HEIGHT)

        surf = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        surf.fill((250, 250, 250, 230))
        self.screen.blit(surf, rect.topleft)
        pygame.draw.line(self.screen, PANEL_BORDER,
                         (rect.x, rect.y), (rect.x, rect.bottom), 1)

        result = self.results[algo_name]
        color  = result["color"]
        x = rect.x + 16
        y = rect.y + 16

        pygame.draw.circle(self.screen, color, (x + 8, y + 10), 8)
        self.screen.blit(
            self.big_font.render(ALGO_DISPLAY_NAME[algo_name], True, TEXT_COLOR),
            (x + 22, y))
        y += 36

        pygame.draw.line(self.screen, (210, 210, 210),
                         (x, y), (rect.right - 16, y), 1)
        y += 12

        def sv(v: Any) -> str:
            if v is None: return "—"
            if isinstance(v, float): return f"{v:.4f}"
            return str(v)

        rows = [
            ("1st key steps",   sv(result["step_to_first_key"])),
            ("2nd key steps",   sv(result["step_to_second_key"])),
            ("3rd key steps",   sv(result["step_to_third_key"])),
            ("Exit steps",      sv(result["step_to_exit"])),
            ("Total steps",     sv(result["total_steps"])),
            ("Unique visited",  sv(result["unique_cells_visited"])),
            ("Repeated cells",  sv(result["repeated_visits"])),
            ("Repeat ratio",    sv(result["repeated_visit_ratio"])),
        ]
        mf = pygame.font.SysFont("consolas", 18)
        for label, value in rows:
            lbl_s = self.small_font.render(label, True, (100, 100, 100))
            val_s = mf.render(value, True, TEXT_COLOR)
            self.screen.blit(lbl_s, (x, y))
            self.screen.blit(val_s, (rect.right - 16 - val_s.get_width(), y))
            y += 26
        y += 8
        pygame.draw.line(self.screen, (210, 210, 210),
                         (x, y), (rect.right - 16, y), 1)
        y += 12
        progress = self._anim_progress(algo_name)
        total    = len(result["path"])
        pct      = progress / total if total > 0 else 0.0
        bar_w    = rect.width - 32
        bar_rect = pygame.Rect(x, y, bar_w, 8)
        pygame.draw.rect(self.screen, (210, 210, 210), bar_rect, border_radius=4)
        fill_w = int(bar_w * pct)
        if fill_w > 0:
            pygame.draw.rect(self.screen, color,
                             pygame.Rect(x, y, fill_w, 8), border_radius=4)
        y += 20
        self.screen.blit(
            self.small_font.render(
                f"Step {min(progress, total)} / {total}", True, (120, 120, 120)),
            (x, y))


    def _draw_top_hud(self) -> None:
        sw = self.screen.get_width()
        surf = pygame.Surface((sw, TOP_HUD_HEIGHT), pygame.SRCALPHA)
        surf.fill(HUD_BG)
        self.screen.blit(surf, (0, 0))
        pygame.draw.line(self.screen, (185, 185, 185),
                         (0, TOP_HUD_HEIGHT - 1), (sw, TOP_HUD_HEIGHT - 1), 1)

        filter_label = (self.current_filter.upper()
                        if self.current_filter != "all" else "ALL")
        title = (f"Maze Run Viewer   {self.rows}×{self.cols}   "
                 f"Keys: {len(self.keys)}   View: {filter_label}   "
                 f"{'⏸ Paused' if self.animation_paused else '▶ Playing'}   "
                 f"Speed ×{self.speed_multiplier:.1f}")
        controls = ("1 BFS   2 DFS   3 Dijkstra   4 A*   0 Grid   "
                    "Space pause   N restart   G grid   ESC quit")

        self.screen.blit(self.big_font.render(title, True, TEXT_COLOR), (14, 8))
        self.screen.blit(self.small_font.render(controls, True, (100, 100, 100)),
                         (14, 38))

    def draw_all(self) -> None:
        self.screen.fill(BACKGROUND_COLOR)
        panel_rects = self._panel_rects()
        for algo_name, rect in panel_rects.items():
            self._draw_panel(algo_name, rect)
        if self.current_filter != "all" and len(self._visible_algos()) == 1:
            self._draw_solo_sidebar(self._visible_algos()[0])
        self._draw_top_hud()

    def handle_keydown(self, event: pygame.event.Event) -> None:
        if event.key == pygame.K_ESCAPE:
            raise SystemExit
        elif event.key == pygame.K_SPACE:
            self.toggle_pause()
        elif event.key == pygame.K_n:
            self.restart_animation()
        elif event.key == pygame.K_g:
            self.show_grid = not self.show_grid
        elif event.key == pygame.K_0:
            self.current_filter = "all"
            self.restart_animation()
        elif event.key == pygame.K_1 and "bfs"      in self.available_algos:
            self.current_filter = "bfs";      self.restart_animation()
        elif event.key == pygame.K_2 and "dfs"      in self.available_algos:
            self.current_filter = "dfs";      self.restart_animation()
        elif event.key == pygame.K_3 and "dijkstra" in self.available_algos:
            self.current_filter = "dijkstra"; self.restart_animation()
        elif event.key == pygame.K_4 and "astar"    in self.available_algos:
            self.current_filter = "astar";    self.restart_animation()

    def run(self) -> None:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                if event.type == pygame.VIDEORESIZE:
                    self.screen = pygame.display.set_mode(
                        (event.w, event.h), pygame.RESIZABLE)
                elif event.type == pygame.KEYDOWN:
                    self.handle_keydown(event)

            self.draw_all()
            pygame.display.flip()
            self.clock.tick(60)


def main() -> None:
    args       = parse_args()
    input_path = Path(args.input).resolve() if args.input else default_input_path()
    code_dir   = default_code_dir()

    maze_data = load_maze(input_path)
    validate_maze(maze_data)

    results = run_available_algorithms(
        maze_data=maze_data,
        code_dir=code_dir,
        requested_algo=args.run,
    )

    viewer = MazeRunViewer(
        maze_data=maze_data,
        results=results,
        window_width=args.width,
        window_height=args.height,
        speed_multiplier=args.speed,
    )
    viewer.run()


if __name__ == "__main__":
    main()
