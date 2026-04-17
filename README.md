# CS5800 Final Project: Maze Runner

This project is a maze-based algorithm comparison framework for the CS5800 final project.

The maze is a 2D grid with walls between neighboring cells.  
An agent starts from the entrance, must collect 3 hidden keys, and then reach the exit.

At the current stage, this repository includes:

- `maze_generate.py`: generate a maze and save it as `maze.json`
- `maze_display.py`: read `maze.json` and render the maze with pygame

---

## Project Structure

```text
maze_runner/
├── maze.json
├── README.md
├── requirements.txt
└── code/
    ├── maze_generate.py
    └── maze_display.py
```

- `maze.json` is stored in the project root.
- Python scripts are stored in `code/`.

---

## Environment Setup

It is recommended to create a virtual environment first.

### Option 1: using conda

```bash
conda create -n maze python=3.10 -y
conda activate maze
pip install -r requirements.txt
```

### Option 2: using venv

```bash
python -m venv venv
```

Activate the environment:

#### On Windows PowerShell
```bash
venv\Scripts\Activate.ps1
```

#### On Windows CMD
```bash
venv\Scripts\activate
```

#### On macOS / Linux
```bash
source venv/bin/activate
```

Then install dependencies:

```bash
pip install -r requirements.txt
```

---

## 1. Generate a Maze

Go to the `code` directory:

```bash
cd code
```

Run the generator:

```bash
python maze_generate.py
```

This will generate a default `5 x 5` maze and save it to:

```text
../maze.json
```

That means the output file will appear in the project root as `maze.json`.

### Example: generate a larger maze

```bash
python maze_generate.py --rows 10 --cols 10 --seed 42
```

### Example: generate a 50 x 50 maze

```bash
python maze_generate.py --rows 50 --cols 50 --seed 42
```

### Optional arguments

- `--rows`: number of rows
- `--cols`: number of columns
- `--keys`: number of keys
- `--seed`: random seed for reproducibility
- `--output`: optional custom output path

Example:

```bash
python maze_generate.py --rows 20 --cols 20 --keys 3 --seed 123 --output ../maze.json
```

---

## 2. Display the Maze

After `maze.json` has been generated, run:

```bash
python maze_display.py
```

This reads `../maze.json` by default and opens a pygame window.

### Optional input path

```bash
python maze_display.py --input ../maze.json
```

### Optional window size

```bash
python maze_display.py --width 1400 --height 1000
```

---

## 3. Controls in the Pygame Viewer

Inside the pygame window:

- Mouse wheel: zoom in / out
- `+` or `-`: zoom in / out
- Left mouse drag: pan
- Arrow keys: pan
- `R`: reset the view to fit the maze
- `G`: toggle helper grid
- `ESC`: quit

### Symbols

- `I`: entrance
- `O`: exit
- `K`: key

---

## 4. Maze JSON Format

The generated `maze.json` follows this structure:

```json
{
  "meta": {
    "rows": 5,
    "cols": 5,
    "generator": "dfs_backtracking",
    "seed": 42,
    "key_count": 3,
    "wall_encoding": {
      "UP": 1,
      "RIGHT": 2,
      "DOWN": 4,
      "LEFT": 8
    }
  },
  "start": [4, 0],
  "exit": [0, 4],
  "keys": [[1, 0], [2, 4], [3, 1]],
  "cells": [
    [9, 1, 1, 1, 3],
    [8, 0, 1, 1, 2],
    [8, 1, 2, 1, 2],
    [8, 0, 4, 0, 2],
    [12, 5, 5, 5, 6]
  ]
}
```

### Important notes

- `start` and `exit` are known positions
- `keys` are stored in the file, but future agent code does not have to expose them directly
- `cells[r][c]` stores the wall bitmask for each cell

Wall encoding:

- `UP = 1`
- `RIGHT = 2`
- `DOWN = 4`
- `LEFT = 8`

A direction is blocked if the corresponding bit is present.

---

## 5. Recommended Workflow

From the project root:

```bash
cd code
python maze_generate.py --rows 10 --cols 10 --seed 42
python maze_display.py
```

This is the standard workflow for testing.

---

## 6. Notes for Teammates

- Always keep `maze_generate.py` and `maze_display.py` inside `code/`
- Always keep `maze.json` in the project root unless you intentionally change the path
- If pygame is missing, install dependencies again with:

```bash
pip install -r requirements.txt
```

---

## 7. Current Status

Currently completed:

- Maze generation
- Maze JSON export
- Pygame maze visualization

Planned next steps:

- Maze environment interface
- Agent algorithms
- Path animation
- Performance statistics and comparison charts
