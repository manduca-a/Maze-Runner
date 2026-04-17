#!/usr/bin/env python3
"""
maze_run.py

Visualize maze traversal results for BFS / DFS / Dijkstra / A* in a pygame window.

Features:
1. Different colors for different algorithms:
   - BFS: red
   - DFS: green
   - Dijkstra: yellow
   - A*: blue
2. Animate traversal paths over a few seconds.
3. Show statistics:
   - steps to first key
   - steps to second key
   - steps to third key
   - steps to exit
   - repeated visited cells count
   - repeated visit ratio
4. Support command-line filtering:
   - python maze_run.py --run dijkstra
   - without --run, try to display all available algorithms
5. Support in-window toggles:
   - 1: BFS
   - 2: DFS
   - 3: Dijkstra
   - 4: A*
   - 0: all available algorithms

Design notes:
- This script reads maze.json from the project root by default.
- It dynamically loads algorithm files from code/:
    bfs.py, dfs.py, dijkstra.py, astar.py
- The script tries several common function names in each module, including:
    run_bfs_agent(data), run_dfs_agent(data), run_dijkstra_agent(data), run_astar_agent(data)
    run_algorithm(data), solve(data)
- The loaded function must return a dict containing at least:
    {
      "path": [[r, c], ...],
      "exit": [r, c],
      "actual_keys": [[r, c], ...]   # preferred
    }
- If some metrics are missing, maze_run.py computes them from the path.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pygame


UP = 1
RIGHT = 2
DOWN = 4
LEFT = 8

Coord = Tuple[int, int]

ALGO_ORDER = ["bfs", "dfs", "dijkstra", "astar"]
ALGO_DISPLAY_NAME = {
    "bfs": "BFS",
    "dfs": "DFS",
    "dijkstra": "Dijkstra",
    "astar": "A*",
}
ALGO_COLORS = {
    "bfs": (229, 57, 53),       # red
    "dfs": (67, 160, 71),       # green
    "dijkstra": (251, 192, 45), # yellow
    "astar": (66, 133, 244),    # blue
}

BACKGROUND_COLOR = (245, 245, 245)
CELL_COLOR = (255, 255, 255)
WALL_COLOR = (20, 20, 20)
GRID_COLOR = (225, 225, 225)
TEXT_COLOR = (20, 20, 20)
START_COLOR = (76, 175, 80)
EXIT_COLOR = (229, 57, 53)
KEY_COLOR = (255, 193, 7)

HUD_BG = (255, 255, 255, 235)
SIDEBAR_BG = (250, 250, 250, 235)
SIDEBAR_BORDER = (200, 200, 200)

DEFAULT_WINDOW_WIDTH = 1500
DEFAULT_WINDOW_HEIGHT = 950
MIN_CELL_SIZE = 3.0
MAX_CELL_SIZE = 120.0
TOP_HUD_HEIGHT = 72
RIGHT_PANEL_WIDTH = 360
BOTTOM_HUD_HEIGHT = 72
MIN_ANIM_SECONDS = 3.0
MAX_ANIM_SECONDS = 8.0


def default_input_path() -> Path:
    script_path = Path(__file__).resolve()
    root_dir = script_path.parent.parent
    return root_dir / "maze.json"


def default_code_dir() -> Path:
    script_path = Path(__file__).resolve()
    return script_path.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run and visualize maze algorithms.")
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Optional maze.json path. Default: maze_runner/maze.json",
    )
    parser.add_argument(
        "--run",
        type=str,
        default=None,
        choices=ALGO_ORDER,
        help="Run and display only one algorithm.",
    )
    parser.add_argument("--width", type=int, default=DEFAULT_WINDOW_WIDTH, help="Window width.")
    parser.add_argument("--height", type=int, default=DEFAULT_WINDOW_HEIGHT, help="Window height.")
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Animation speed multiplier. Larger is faster.",
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
        raise ValueError("Start, exit, and keys must all be distinct.")


def normalize_path(path_value: Any) -> List[Coord]:
    if not isinstance(path_value, list):
        raise ValueError("Result path must be a list.")
    path: List[Coord] = []
    for item in path_value:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError("Each path item must be a coordinate pair.")
        path.append((int(item[0]), int(item[1])))
    if not path:
        raise ValueError("Result path must not be empty.")
    return path


def compute_stats_from_path(path: List[Coord], key_set: Set[Coord], exit_pos: Coord) -> Dict[str, Any]:
    visit_counts: Dict[Coord, int] = {}
    discovered_keys: List[Coord] = []
    seen_keys: Set[Coord] = set()

    step_first_key: Optional[int] = None
    step_second_key: Optional[int] = None
    step_third_key: Optional[int] = None

    for step_idx, pos in enumerate(path):
        visit_counts[pos] = visit_counts.get(pos, 0) + 1

        if pos in key_set and pos not in seen_keys:
            seen_keys.add(pos)
            discovered_keys.append(pos)
            if len(discovered_keys) == 1:
                step_first_key = step_idx
            elif len(discovered_keys) == 2:
                step_second_key = step_idx
            elif len(discovered_keys) == 3:
                step_third_key = step_idx

    total_steps = max(0, len(path) - 1)
    repeated_visits = sum(v - 1 for v in visit_counts.values() if v > 1)
    repeated_visit_ratio = repeated_visits / total_steps if total_steps > 0 else 0.0

    step_exit = None
    if path and path[-1] == exit_pos:
        step_exit = total_steps
    else:
        for idx, pos in enumerate(path):
            if pos == exit_pos:
                step_exit = idx
                break

    return {
        "path": [[r, c] for r, c in path],
        "discovered_key_order": [[r, c] for r, c in discovered_keys],
        "step_to_first_key": step_first_key,
        "step_to_second_key": step_second_key,
        "step_to_third_key": step_third_key,
        "step_to_exit": step_exit,
        "repeated_visits": repeated_visits,
        "repeated_visit_ratio": repeated_visit_ratio,
        "unique_cells_visited": len(visit_counts),
        "total_steps": total_steps,
    }


def normalize_result(name: str, raw_result: Dict[str, Any], maze_data: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(raw_result, dict):
        raise ValueError(f"{name}: algorithm result must be a dict.")

    path = normalize_path(raw_result.get("path"))
    start = tuple(maze_data["start"])
    if path[0] != start:
        raise ValueError(f"{name}: path must start at maze start {start}, got {path[0]}.")

    exit_pos = tuple(raw_result.get("exit", maze_data["exit"]))
    key_list_raw = raw_result.get("actual_keys", maze_data["keys"])
    actual_keys = {tuple(k) for k in key_list_raw}

    computed = compute_stats_from_path(path, actual_keys, exit_pos)

    result = {
        "algorithm": raw_result.get("algorithm", name),
        "display_name": ALGO_DISPLAY_NAME[name],
        "color": ALGO_COLORS[name],
        "path": computed["path"],
        "actual_keys": [list(k) for k in sorted(actual_keys)],
        "discovered_key_order": raw_result.get("discovered_key_order", computed["discovered_key_order"]),
        "step_to_first_key": raw_result.get("step_to_first_key", computed["step_to_first_key"]),
        "step_to_second_key": raw_result.get("step_to_second_key", computed["step_to_second_key"]),
        "step_to_third_key": raw_result.get("step_to_third_key", computed["step_to_third_key"]),
        "step_to_exit": raw_result.get("step_to_exit", computed["step_to_exit"]),
        "repeated_visits": raw_result.get("repeated_visits", computed["repeated_visits"]),
        "repeated_visit_ratio": raw_result.get("repeated_visit_ratio", computed["repeated_visit_ratio"]),
        "unique_cells_visited": raw_result.get("unique_cells_visited", computed["unique_cells_visited"]),
        "total_steps": raw_result.get("total_steps", computed["total_steps"]),
        "success": raw_result.get("success", path[-1] == exit_pos),
    }
    return result


def load_algorithm_module(module_path: Path, import_name: str):
    spec = importlib.util.spec_from_file_location(import_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def find_runner_function(module: Any, algo_name: str):
    candidates = [
        f"run_{algo_name}_agent",
        "run_algorithm",
        "solve",
    ]
    for func_name in candidates:
        func = getattr(module, func_name, None)
        if callable(func):
            return func
    return None


def run_available_algorithms(
    maze_data: Dict[str, Any],
    code_dir: Path,
    requested_algo: Optional[str],
) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    selected_algos = [requested_algo] if requested_algo else ALGO_ORDER

    for algo_name in selected_algos:
        module_filename = f"{algo_name}.py"
        module_path = code_dir / module_filename

        if not module_path.exists():
            if requested_algo:
                raise FileNotFoundError(f"Requested algorithm file not found: {module_path}")
            continue

        module = load_algorithm_module(module_path, f"maze_run_{algo_name}")
        runner = find_runner_function(module, algo_name)
        if runner is None:
            if requested_algo:
                raise AttributeError(
                    f"{module_filename} does not define a supported runner function."
                )
            continue

        raw_result = runner(maze_data)
        normalized = normalize_result(algo_name, raw_result, maze_data)
        results[algo_name] = normalized

    if requested_algo and requested_algo not in results:
        raise RuntimeError(f"Could not run requested algorithm: {requested_algo}")

    if not results:
        raise RuntimeError(
            "No runnable algorithm module was found. Expected files like bfs.py, dfs.py, dijkstra.py, astar.py in code/."
        )

    return results


class MazeRunViewer:
    def __init__(
        self,
        maze_data: Dict[str, Any],
        results: Dict[str, Dict[str, Any]],
        window_width: int,
        window_height: int,
        speed_multiplier: float,
    ) -> None:
        self.data = maze_data
        self.results = results
        self.available_algos = [name for name in ALGO_ORDER if name in results]

        self.rows = self.data["meta"]["rows"]
        self.cols = self.data["meta"]["cols"]
        self.cells: List[List[int]] = self.data["cells"]
        self.start: Coord = tuple(self.data["start"])
        self.exit: Coord = tuple(self.data["exit"])
        self.keys: List[Coord] = [tuple(k) for k in self.data["keys"]]

        self.window_width = max(1000, window_width)
        self.window_height = max(700, window_height)
        self.speed_multiplier = max(0.1, speed_multiplier)

        pygame.init()
        pygame.display.set_caption("Maze Run Viewer")
        self.screen = pygame.display.set_mode((self.window_width, self.window_height), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()

        self.font = pygame.font.SysFont(None, 24)
        self.small_font = pygame.font.SysFont(None, 20)
        self.big_font = pygame.font.SysFont(None, 30)
        self.mono_font = pygame.font.SysFont("consolas", 20)

        self.cell_size = 20.0
        self.offset_x = 0.0
        self.offset_y = 0.0

        self.show_grid = True
        self.dragging = False
        self.last_mouse_pos = (0, 0)

        self.current_filter = "all"
        self.animation_start_time = time.time()
        self.animation_paused = False
        self.pause_elapsed = 0.0

        self.reset_view()

    def maze_area_rect(self) -> pygame.Rect:
        width = self.screen.get_width() - RIGHT_PANEL_WIDTH
        height = self.screen.get_height() - TOP_HUD_HEIGHT - BOTTOM_HUD_HEIGHT
        return pygame.Rect(0, TOP_HUD_HEIGHT, max(100, width), max(100, height))

    def sidebar_rect(self) -> pygame.Rect:
        return pygame.Rect(
            self.screen.get_width() - RIGHT_PANEL_WIDTH,
            TOP_HUD_HEIGHT,
            RIGHT_PANEL_WIDTH,
            self.screen.get_height() - TOP_HUD_HEIGHT,
        )

    def reset_view(self) -> None:
        area = self.maze_area_rect()
        fit_size = min((area.width - 40) / self.cols, (area.height - 40) / self.rows)
        self.cell_size = max(MIN_CELL_SIZE, min(MAX_CELL_SIZE, fit_size))

        maze_px_w = self.cols * self.cell_size
        maze_px_h = self.rows * self.cell_size
        self.offset_x = area.x + (area.width - maze_px_w) / 2
        self.offset_y = area.y + (area.height - maze_px_h) / 2

    def cell_rect(self, r: int, c: int) -> pygame.Rect:
        x = int(round(self.offset_x + c * self.cell_size))
        y = int(round(self.offset_y + r * self.cell_size))
        w = max(1, int(round(self.cell_size)))
        h = max(1, int(round(self.cell_size)))
        return pygame.Rect(x, y, w, h)

    def visible_algos(self) -> List[str]:
        if self.current_filter == "all":
            return self.available_algos
        if self.current_filter in self.available_algos:
            return [self.current_filter]
        return []

    def animation_duration_for_algo(self, algo_name: str) -> float:
        total_steps = self.results[algo_name]["total_steps"]
        base = max(MIN_ANIM_SECONDS, min(MAX_ANIM_SECONDS, 0.005 * total_steps + 3.0))
        return base / self.speed_multiplier

    def animation_progress(self, algo_name: str) -> int:
        result = self.results[algo_name]
        path_len = len(result["path"])
        duration = self.animation_duration_for_algo(algo_name)

        if self.animation_paused:
            elapsed = self.pause_elapsed
        else:
            elapsed = time.time() - self.animation_start_time

        if duration <= 0:
            return path_len

        progress_ratio = min(1.0, max(0.0, elapsed / duration))
        return max(1, int(progress_ratio * path_len))

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

    def zoom_at(self, factor: float, mouse_pos: Tuple[int, int]) -> None:
        old_size = self.cell_size
        new_size = max(MIN_CELL_SIZE, min(MAX_CELL_SIZE, old_size * factor))
        if abs(new_size - old_size) < 1e-6:
            return

        mx, my = mouse_pos
        world_x = (mx - self.offset_x) / old_size
        world_y = (my - self.offset_y) / old_size

        self.cell_size = new_size
        self.offset_x = mx - world_x * new_size
        self.offset_y = my - world_y * new_size

    def draw_base_maze(self) -> None:
        self.screen.fill(BACKGROUND_COLOR)
        area = self.maze_area_rect()
        pygame.draw.rect(self.screen, CELL_COLOR, area)

        maze_rect = pygame.Rect(
            int(self.offset_x),
            int(self.offset_y),
            int(round(self.cols * self.cell_size)),
            int(round(self.rows * self.cell_size)),
        )
        pygame.draw.rect(self.screen, CELL_COLOR, maze_rect)

        if self.show_grid and self.cell_size >= 10:
            for c in range(self.cols + 1):
                x = int(round(self.offset_x + c * self.cell_size))
                pygame.draw.line(
                    self.screen,
                    GRID_COLOR,
                    (x, int(self.offset_y)),
                    (x, int(round(self.offset_y + self.rows * self.cell_size))),
                    1,
                )
            for r in range(self.rows + 1):
                y = int(round(self.offset_y + r * self.cell_size))
                pygame.draw.line(
                    self.screen,
                    GRID_COLOR,
                    (int(self.offset_x), y),
                    (int(round(self.offset_x + self.cols * self.cell_size)), y),
                    1,
                )

        for r in range(self.rows):
            for c in range(self.cols):
                rect = self.cell_rect(r, c)
                center = rect.center

                if (r, c) == self.start:
                    pygame.draw.rect(
                        self.screen,
                        START_COLOR,
                        rect.inflate(-max(2, rect.width // 6), -max(2, rect.height // 6)),
                    )
                elif (r, c) == self.exit:
                    pygame.draw.rect(
                        self.screen,
                        EXIT_COLOR,
                        rect.inflate(-max(2, rect.width // 6), -max(2, rect.height // 6)),
                    )
                elif (r, c) in self.keys:
                    radius = max(3, int(self.cell_size * 0.22))
                    pygame.draw.circle(self.screen, KEY_COLOR, center, radius)

                if self.cell_size >= 18:
                    label = None
                    label_color = TEXT_COLOR
                    if (r, c) == self.start:
                        label = "I"
                        label_color = (255, 255, 255)
                    elif (r, c) == self.exit:
                        label = "O"
                        label_color = (255, 255, 255)
                    elif (r, c) in self.keys:
                        label = "K"
                        label_color = (70, 50, 0)

                    if label is not None:
                        font_size = max(14, min(34, int(self.cell_size * 0.55)))
                        label_font = pygame.font.SysFont(None, font_size, bold=True)
                        surf = label_font.render(label, True, label_color)
                        self.screen.blit(surf, surf.get_rect(center=center))

        wall_width = 1 if self.cell_size < 20 else 2 if self.cell_size < 60 else 3
        for r in range(self.rows):
            for c in range(self.cols):
                x0 = int(round(self.offset_x + c * self.cell_size))
                y0 = int(round(self.offset_y + r * self.cell_size))
                x1 = int(round(self.offset_x + (c + 1) * self.cell_size))
                y1 = int(round(self.offset_y + (r + 1) * self.cell_size))
                cell = self.cells[r][c]

                if cell & UP:
                    pygame.draw.line(self.screen, WALL_COLOR, (x0, y0), (x1, y0), wall_width)
                if cell & RIGHT:
                    pygame.draw.line(self.screen, WALL_COLOR, (x1, y0), (x1, y1), wall_width)
                if cell & DOWN:
                    pygame.draw.line(self.screen, WALL_COLOR, (x0, y1), (x1, y1), wall_width)
                if cell & LEFT:
                    pygame.draw.line(self.screen, WALL_COLOR, (x0, y0), (x0, y1), wall_width)

    def draw_paths(self) -> None:
        visible = self.visible_algos()
        if not visible:
            return

        for algo_name in visible:
            result = self.results[algo_name]
            color = result["color"]
            path = [tuple(p) for p in result["path"]]
            progress = self.animation_progress(algo_name)
            sub_path = path[:progress]

            if len(sub_path) >= 2:
                width = max(2, int(self.cell_size * 0.18))
                points = []
                for r, c in sub_path:
                    rect = self.cell_rect(r, c)
                    points.append(rect.center)
                pygame.draw.lines(self.screen, color, False, points, width)

            if sub_path:
                r, c = sub_path[-1]
                rect = self.cell_rect(r, c)
                radius = max(4, int(self.cell_size * 0.24))
                pygame.draw.circle(self.screen, color, rect.center, radius)

    def draw_top_hud(self) -> None:
        rect = pygame.Rect(0, 0, self.screen.get_width(), TOP_HUD_HEIGHT)
        surf = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        surf.fill(HUD_BG)
        self.screen.blit(surf, rect.topleft)
        pygame.draw.line(self.screen, (185, 185, 185), (0, TOP_HUD_HEIGHT - 1), (self.screen.get_width(), TOP_HUD_HEIGHT - 1), 1)

        title = f"Maze Run Viewer   {self.rows}x{self.cols}   Keys: {len(self.keys)}   Filter: {self.current_filter.upper() if self.current_filter != 'all' else 'ALL'}"
        controls = "1 BFS   2 DFS   3 Dijkstra   4 A*   0 All   Space pause   N restart   G grid   R reset view   +/- zoom   Drag pan"

        self.screen.blit(self.big_font.render(title, True, TEXT_COLOR), (14, 10))
        self.screen.blit(self.small_font.render(controls, True, TEXT_COLOR), (14, 42))

    def draw_bottom_hud(self) -> None:
        y = self.screen.get_height() - BOTTOM_HUD_HEIGHT
        rect = pygame.Rect(0, y, self.screen.get_width() - RIGHT_PANEL_WIDTH, BOTTOM_HUD_HEIGHT)
        surf = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        surf.fill(HUD_BG)
        self.screen.blit(surf, rect.topleft)
        pygame.draw.line(self.screen, (185, 185, 185), (0, y), (rect.width, y), 1)

        legend_x = 16
        legend_y = y + 14
        for algo_name in self.available_algos:
            color = self.results[algo_name]["color"]
            label = ALGO_DISPLAY_NAME[algo_name]
            pygame.draw.circle(self.screen, color, (legend_x + 10, legend_y + 10), 8)
            self.screen.blit(self.font.render(label, True, TEXT_COLOR), (legend_x + 24, legend_y))
            legend_x += 130

        status = "Paused" if self.animation_paused else "Playing"
        self.screen.blit(self.font.render(f"Animation: {status}   Speed: x{self.speed_multiplier:.2f}", True, TEXT_COLOR), (16, y + 40))

    def stat_value_str(self, value: Any) -> str:
        if value is None:
            return "-"
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    def draw_sidebar(self) -> None:
        rect = self.sidebar_rect()
        surf = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        surf.fill(SIDEBAR_BG)
        self.screen.blit(surf, rect.topleft)
        pygame.draw.line(self.screen, SIDEBAR_BORDER, (rect.x, rect.y), (rect.x, rect.bottom), 1)

        x = rect.x + 16
        y = rect.y + 14

        header = self.big_font.render("Statistics", True, TEXT_COLOR)
        self.screen.blit(header, (x, y))
        y += 36

        visible = self.visible_algos()
        if not visible:
            self.screen.blit(self.font.render("No algorithm selected.", True, TEXT_COLOR), (x, y))
            return

        for algo_name in visible:
            result = self.results[algo_name]
            color = result["color"]

            pygame.draw.circle(self.screen, color, (x + 10, y + 12), 8)
            title = self.font.render(ALGO_DISPLAY_NAME[algo_name], True, TEXT_COLOR)
            self.screen.blit(title, (x + 24, y + 2))
            y += 28

            lines = [
                f"1st key steps:   {self.stat_value_str(result['step_to_first_key'])}",
                f"2nd key steps:   {self.stat_value_str(result['step_to_second_key'])}",
                f"3rd key steps:   {self.stat_value_str(result['step_to_third_key'])}",
                f"Exit steps:      {self.stat_value_str(result['step_to_exit'])}",
                f"Repeated cells:  {self.stat_value_str(result['repeated_visits'])}",
                f"Repeat ratio:    {self.stat_value_str(result['repeated_visit_ratio'])}",
                f"Unique visited:  {self.stat_value_str(result['unique_cells_visited'])}",
                f"Total steps:     {self.stat_value_str(result['total_steps'])}",
            ]

            for line in lines:
                surf_line = self.mono_font.render(line, True, TEXT_COLOR)
                self.screen.blit(surf_line, (x + 6, y))
                y += 24

            y += 12
            pygame.draw.line(self.screen, (220, 220, 220), (x, y), (rect.right - 16, y), 1)
            y += 16

    def draw_all(self) -> None:
        self.draw_base_maze()
        self.draw_paths()
        self.draw_top_hud()
        self.draw_bottom_hud()
        self.draw_sidebar()

    def handle_keydown(self, event: pygame.event.Event) -> None:
        pan_step = max(20, int(self.cell_size * 0.8))

        if event.key == pygame.K_ESCAPE:
            raise SystemExit
        elif event.key == pygame.K_r:
            self.reset_view()
        elif event.key == pygame.K_g:
            self.show_grid = not self.show_grid
        elif event.key == pygame.K_SPACE:
            self.toggle_pause()
        elif event.key == pygame.K_n:
            self.restart_animation()
        elif event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
            center = (self.maze_area_rect().centerx, self.maze_area_rect().centery)
            self.zoom_at(1.15, center)
        elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
            center = (self.maze_area_rect().centerx, self.maze_area_rect().centery)
            self.zoom_at(1 / 1.15, center)
        elif event.key == pygame.K_LEFT:
            self.offset_x += pan_step
        elif event.key == pygame.K_RIGHT:
            self.offset_x -= pan_step
        elif event.key == pygame.K_UP:
            self.offset_y += pan_step
        elif event.key == pygame.K_DOWN:
            self.offset_y -= pan_step
        elif event.key == pygame.K_0:
            self.current_filter = "all"
            self.restart_animation()
        elif event.key == pygame.K_1 and "bfs" in self.available_algos:
            self.current_filter = "bfs"
            self.restart_animation()
        elif event.key == pygame.K_2 and "dfs" in self.available_algos:
            self.current_filter = "dfs"
            self.restart_animation()
        elif event.key == pygame.K_3 and "dijkstra" in self.available_algos:
            self.current_filter = "dijkstra"
            self.restart_animation()
        elif event.key == pygame.K_4 and "astar" in self.available_algos:
            self.current_filter = "astar"
            self.restart_animation()

    def run(self) -> None:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

                if event.type == pygame.VIDEORESIZE:
                    self.screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)

                elif event.type == pygame.KEYDOWN:
                    self.handle_keydown(event)

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1 and self.maze_area_rect().collidepoint(event.pos):
                        self.dragging = True
                        self.last_mouse_pos = event.pos
                    elif event.button == 4 and self.maze_area_rect().collidepoint(event.pos):
                        self.zoom_at(1.12, event.pos)
                    elif event.button == 5 and self.maze_area_rect().collidepoint(event.pos):
                        self.zoom_at(1 / 1.12, event.pos)

                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        self.dragging = False

                elif event.type == pygame.MOUSEMOTION and self.dragging:
                    mx, my = event.pos
                    lx, ly = self.last_mouse_pos
                    self.offset_x += mx - lx
                    self.offset_y += my - ly
                    self.last_mouse_pos = event.pos

            self.draw_all()
            pygame.display.flip()
            self.clock.tick(60)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve() if args.input else default_input_path()
    code_dir = default_code_dir()

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
