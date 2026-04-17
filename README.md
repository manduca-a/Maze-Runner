# Maze Runner Project

## Overview

This project implements and compares multiple classical search algorithms in a grid-based maze environment. The maze includes:

* A start position
* An exit
* Three keys that must be collected before reaching the exit

The project supports both **unweighted** and **weighted** maze traversal, enabling analysis of algorithm behavior under different cost structures.

---

## Algorithms

### Unweighted Search

* **BFS (Breadth-First Search)**
  Author: Yanzhe Xi

* **DFS (Depth-First Search)**
  Author: Yanzhe Xi

### Weighted Search

* **Dijkstra**
  Author: Qizhen Dong

* **A***
  Author: Qizhen Dong

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
├── astar_result.json
├── dijkstra_result.json
├── README.md
├── requirements.txt
├── .gitignore
└── code/
    ├── maze_generate.py
    ├── maze_display.py
    ├── maze_run.py
    ├── bfs.py
    ├── dfs.py
    ├── dijkstra.py
    └── astar.py
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
python code/bfs.py
python code/dfs.py
python code/dijkstra.py
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

* BFS minimizes steps, not cost
* Dijkstra guarantees optimal cost
* A* introduces directional bias and heuristic guidance
* DFS is not optimal but useful as a baseline

When loops are introduced into the maze, algorithm differences become significantly more visible.

---

## Notes

* Weighted and unweighted behaviors are intentionally separated
* Maze generation supports randomness via seed
* Visualization highlights traversal differences clearly

---

## Authors

* BFS / DFS: Yanzhe Xi
* Dijkstra / A*: Qizhen Dong
* System Design & Integration: Qizhen Dong
