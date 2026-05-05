# Maze Runner Project

## Overview

This project implements and compares multiple classical search algorithms in a grid-based maze environment. The maze includes:

* A start position
* An exit
* Three keys that must be collected before reaching the exit

The project supports both **unweighted** and **weighted** maze traversal, enabling analysis of algorithm behavior under different cost structures.

---

## Algorithms

* **BFS Optimal**
* **BFS Greedy**
* **DFS (Depth-First Search)**
* **A***

---

## Maze Design

### Structure

* Grid-based maze
* Walls represented using bit encoding
* Fully connected (every cell reachable)

### Key Placement

* Keys are randomly generated
* Positions are uniformly sampled from all valid cells
* Start and exit positions are excluded

### Weight Rules (for weighted algorithms)

* Outer ring weight = 1
* Each inner layer increases weight by +1
* Movement cost = weight of destination cell

---

## Task Definition

All agents follow the same task pipeline:

1. Start from the entrance
2. Search for keys (unknown positions)
3. Collect all keys
4. Navigate to the exit

Search and routing are separated into two phases:

* **Search phase**: find next key
* **Routing phase**: compute shortest path to that key

---

## File Structure

```
maze_runner/
├── maze.json
├── README.md
├── requirements.txt
├── .gitignore
└── code/
    ├── maze_generate.py
    ├── maze_display.py
    ├── maze_run.py
    ├── bfs_optimal.py
    ├── bfs_greedy.py
    ├── dfs.py
    └── astar.py
```

---

## Setup

Before running the project, make sure to install the required dependencies. It is recommended to use a virtual environment, especially on newer macOS/Linux systems to avoid environment errors.

```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Generate Maze

```
python code/maze_generate.py
python code/maze_generate.py --rows 20 --cols 20 --seed 42 --loop-rate 0.1
python code/maze_display.py
```

### Run Algorithms

```
python code/bfs_optimal.py
python code/bfs_greedy.py
python code/dfs.py
python code/astar.py
```

### Visualize & Compare

```
python code/maze_run.py
```

---

## Output Metrics

Each algorithm produces:

* Path taken
* Total steps
* Total weighted cost
* Key discovery order
* Repeated visits
* Average cost per step

---

## Key Observations

* BFS Optimal finds the shortest path optimally
* BFS Greedy finds a path quickly but might not be optimal
* DFS is not optimal but useful as a baseline
* A* introduces directional bias and heuristic guidance

When loops are introduced into the maze, algorithm differences become significantly more visible.

---

## Notes

* Weighted and unweighted behaviors are intentionally separated
* Maze generation supports randomness via seed
* Visualization highlights traversal differences clearly

---

## Authors

* BFS Optimal / BFS Greedy / DFS: Yanzhe Xi
* A* / System Design & Integration: Qizhen Dong
