# Travelling Salesman Problem with Guided Local Search (TSP-GLS)

## Problem

Design an **edge-distance update heuristic** for **Guided Local Search (GLS)** applied to the **Travelling Salesman Problem** (TSP). The evolved function is called after each local search phase to penalize frequently used edges, guiding the search away from local optima.

The evaluation runs GLS on 3 random TSP instances with 100 cities each (TSP100), using a time limit of 10 seconds per instance and up to 1000 GLS iterations. Fitness is the **negative average percentage gap** to the optimal tour length — higher is better (0 means optimal).

This problem is adapted from the [EoH benchmark suite](https://github.com/FeiLiu36/EoH).

## Interface

The program must expose a function:

```python
def update_edge_distance(
    edge_distance: float,
    local_opt_tour: list,
    edge_n_used: int,
) -> float:
    ...
```

- `edge_distance`: current distance/cost of the edge.
- `local_opt_tour`: the locally optimal tour found (list of node indices).
- `edge_n_used`: number of times this edge has been penalized.
- Returns: updated (penalized) edge cost used to guide the next local search.

## Metrics

| Key | Description | Goal |
|-----|-------------|------|
| `fitness` | avg_gap_pct (higher is better) | maximize |
| `avg_gap_pct` | average percentage gap to optimal across 3 instances | minimize |
| `gaps` | per-instance gaps | minimize |
| `eval_time` | wall-clock seconds | minimize |

## Notes

- Instances are loaded from `TrainingData/TSPAEL64.pkl` (64 pre-generated TSP100 instances).
- The GLS framework is fixed; only the `update_edge_distance` function is evolved.
- Typical gap percentages range from 0–5% for good heuristics; the EoH baseline achieves ~1–2%.
