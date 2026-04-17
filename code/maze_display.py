#!/usr/bin/env python3
"""
maze_display.py

Author: Qizhen Dong

Render maze.json with pygame so the user can view the maze clearly in a window.

This version adds:
1. Grid weight visualization by grayscale depth:
   - outer ring weight = 1 and shown as white
   - each deeper ring adds +1 weight
   - the center keeps a neutral gray
2. Adaptive window sizing:
   - by default the window opens near screen-maximized size
   - width / height arguments still work if explicitly provided

Recommended project structure:
maze_runner/
├── maze.json
└── code/
    ├── maze_generate.py
    └── maze_display.py

Usage:
    python maze_display.py
    python maze_display.py --input ../maze.json
    python maze_display.py --width 1400 --height 900

Controls:
    Mouse wheel / +/- : zoom in or out
    Drag with left mouse button: pan
    Arrow keys: pan
    R: reset view to fit maze
    G: toggle helper grid
    ESC or close window: quit
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pygame


UP = 1
RIGHT = 2
DOWN = 4
LEFT = 8

Coord = Tuple[int, int]

BACKGROUND_COLOR = (245, 245, 245)
WALL_COLOR = (20, 20, 20)
GRID_COLOR = (220, 220, 220)
TEXT_COLOR = (20, 20, 20)
START_COLOR = (76, 175, 80)
EXIT_COLOR = (229, 57, 53)
KEY_COLOR = (255, 193, 7)
HUD_BG = (255, 255, 255, 230)

DEFAULT_WINDOW_WIDTH = 1200
DEFAULT_WINDOW_HEIGHT = 900
MIN_CELL_SIZE = 4.0
MAX_CELL_SIZE = 120.0
HUD_HEIGHT = 82

OUTER_RING_GRAY = 255
CENTER_RING_GRAY = 100


def default_input_path() -> Path:
    """
    If this script is placed in maze_runner/code/, this returns maze_runner/maze.json.
    """
    script_path = Path(__file__).resolve()
    root_dir = script_path.parent.parent
    return root_dir / "maze.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Display maze.json in a pygame window.")
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Optional input path. Default: maze_runner/maze.json",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Optional window width. If omitted, uses an adaptive near-maximized width.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Optional window height. If omitted, uses an adaptive near-maximized height.",
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


def cell_weight(rows: int, cols: int, r: int, c: int) -> int:
    """
    Outer ring has weight 1.
    Each deeper ring increases weight by 1.
    """
    depth = min(r, c, rows - 1 - r, cols - 1 - c)
    return depth + 1


def max_weight(rows: int, cols: int) -> int:
    return min(rows, cols) // 2 + (1 if min(rows, cols) % 2 == 1 else 0)


def weight_to_gray(weight: int, max_w: int) -> Tuple[int, int, int]:
    """
    Map weight to grayscale:
    - outer ring -> white
    - center -> neutral gray
    """
    if max_w <= 1:
        value = OUTER_RING_GRAY
    else:
        ratio = (weight - 1) / (max_w - 1)
        value = int(round(OUTER_RING_GRAY + ratio * (CENTER_RING_GRAY - OUTER_RING_GRAY)))
    value = max(0, min(255, value))
    return (value, value, value)


class MazeViewer:
    def __init__(
        self,
        maze_data: Dict,
        source_path: Path,
        window_width: Optional[int],
        window_height: Optional[int],
    ) -> None:
        self.data = maze_data
        self.source_path = source_path

        self.rows = self.data["meta"]["rows"]
        self.cols = self.data["meta"]["cols"]
        self.cells: List[List[int]] = self.data["cells"]
        self.start: Coord = tuple(self.data["start"])
        self.exit: Coord = tuple(self.data["exit"])
        self.keys: List[Coord] = [tuple(k) for k in self.data["keys"]]
        self.max_w = max_weight(self.rows, self.cols)

        pygame.init()
        pygame.display.set_caption("Maze Viewer")

        display_info = pygame.display.Info()
        adaptive_width = max(900, display_info.current_w - 80)
        adaptive_height = max(700, display_info.current_h - 120)

        self.window_width = max(600, window_width if window_width is not None else adaptive_width)
        self.window_height = max(500, window_height if window_height is not None else adaptive_height)

        self.screen = pygame.display.set_mode((self.window_width, self.window_height), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()

        self.font = pygame.font.SysFont(None, 24)
        self.small_font = pygame.font.SysFont(None, 20)
        self.big_font = pygame.font.SysFont(None, 30)

        self.cell_size = 20.0
        self.offset_x = 0.0
        self.offset_y = float(HUD_HEIGHT)
        self.show_grid = True

        self.dragging = False
        self.last_mouse_pos = (0, 0)

        self.reset_view()

    def reset_view(self) -> None:
        usable_w = max(100, self.screen.get_width() - 40)
        usable_h = max(100, self.screen.get_height() - HUD_HEIGHT - 40)

        fit_size = min(usable_w / self.cols, usable_h / self.rows)
        self.cell_size = max(MIN_CELL_SIZE, min(MAX_CELL_SIZE, fit_size))

        maze_px_w = self.cols * self.cell_size
        maze_px_h = self.rows * self.cell_size

        self.offset_x = (self.screen.get_width() - maze_px_w) / 2
        self.offset_y = HUD_HEIGHT + (self.screen.get_height() - HUD_HEIGHT - maze_px_h) / 2

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

    def cell_rect(self, r: int, c: int) -> pygame.Rect:
        x = int(round(self.offset_x + c * self.cell_size))
        y = int(round(self.offset_y + r * self.cell_size))
        w = max(1, int(round(self.cell_size)))
        h = max(1, int(round(self.cell_size)))
        return pygame.Rect(x, y, w, h)

    def draw_maze(self) -> None:
        self.screen.fill(BACKGROUND_COLOR)

        maze_rect = pygame.Rect(
            int(self.offset_x),
            int(self.offset_y),
            int(round(self.cols * self.cell_size)),
            int(round(self.rows * self.cell_size)),
        )

        pygame.draw.rect(self.screen, (255, 255, 255), maze_rect)

        for r in range(self.rows):
            for c in range(self.cols):
                rect = self.cell_rect(r, c)
                weight = cell_weight(self.rows, self.cols, r, c)
                fill_color = weight_to_gray(weight, self.max_w)
                pygame.draw.rect(self.screen, fill_color, rect)

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
                    color = TEXT_COLOR
                    if (r, c) == self.start:
                        label = "I"
                        color = (255, 255, 255)
                    elif (r, c) == self.exit:
                        label = "O"
                        color = (255, 255, 255)
                    elif (r, c) in self.keys:
                        label = "K"
                        color = (70, 50, 0)

                    if label is not None:
                        font_size = max(14, min(36, int(self.cell_size * 0.55)))
                        label_font = pygame.font.SysFont(None, font_size, bold=True)
                        surf = label_font.render(label, True, color)
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

        self.draw_hud()

    def draw_hud(self) -> None:
        hud_surface = pygame.Surface((self.screen.get_width(), HUD_HEIGHT), pygame.SRCALPHA)
        hud_surface.fill(HUD_BG)
        self.screen.blit(hud_surface, (0, 0))
        pygame.draw.line(
            self.screen,
            (180, 180, 180),
            (0, HUD_HEIGHT - 1),
            (self.screen.get_width(), HUD_HEIGHT - 1),
            1,
        )

        generator = self.data["meta"].get("generator", "unknown")
        seed = self.data["meta"].get("seed", None)

        title = f"Maze Viewer   {self.rows}x{self.cols}   Generator: {generator}   Seed: {seed}"
        controls = "Zoom: mouse wheel / +/-   Pan: drag or arrow keys   R: reset   G: grid toggle   ESC: quit"
        points = f"I = entrance {self.start}    O = exit {self.exit}    K = keys {self.keys}"
        weights = f"Weight rings: outer = 1 (white), deeper rings increase by +1, center = neutral gray ({self.max_w} max)"

        self.screen.blit(self.big_font.render(title, True, TEXT_COLOR), (12, 8))
        self.screen.blit(self.small_font.render(points, True, TEXT_COLOR), (12, 35))
        self.screen.blit(self.small_font.render(weights, True, TEXT_COLOR), (12, 54))
        controls_surf = self.small_font.render(controls, True, TEXT_COLOR)
        self.screen.blit(controls_surf, (12, 72))

    def handle_keydown(self, event: pygame.event.Event) -> None:
        pan_step = max(20, int(self.cell_size * 0.8))

        if event.key == pygame.K_ESCAPE:
            raise SystemExit
        elif event.key == pygame.K_r:
            self.reset_view()
        elif event.key == pygame.K_g:
            self.show_grid = not self.show_grid
        elif event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
            center = (self.screen.get_width() // 2, self.screen.get_height() // 2)
            self.zoom_at(1.15, center)
        elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
            center = (self.screen.get_width() // 2, self.screen.get_height() // 2)
            self.zoom_at(1 / 1.15, center)
        elif event.key == pygame.K_LEFT:
            self.offset_x += pan_step
        elif event.key == pygame.K_RIGHT:
            self.offset_x -= pan_step
        elif event.key == pygame.K_UP:
            self.offset_y += pan_step
        elif event.key == pygame.K_DOWN:
            self.offset_y -= pan_step

    def run(self) -> None:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

                if event.type == pygame.VIDEORESIZE:
                    self.screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
                    self.reset_view()

                elif event.type == pygame.KEYDOWN:
                    self.handle_keydown(event)

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        self.dragging = True
                        self.last_mouse_pos = event.pos
                    elif event.button == 4:
                        self.zoom_at(1.12, event.pos)
                    elif event.button == 5:
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

            self.draw_maze()
            pygame.display.flip()
            self.clock.tick(60)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve() if args.input else default_input_path()

    data = load_maze(input_path)
    validate_maze(data)

    viewer = MazeViewer(
        maze_data=data,
        source_path=input_path,
        window_width=args.width,
        window_height=args.height,
    )
    viewer.run()


if __name__ == "__main__":
    main()
