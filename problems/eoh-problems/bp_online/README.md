# Online Bin Packing (BP-Online)

## Problem

Design a **priority scoring heuristic** for the **online bin packing** problem: items arrive one at a time and must be immediately assigned to a bin without knowledge of future items. The goal is to minimize the total number of bins used across a set of benchmark instances.

Each item has a size in $(0, 1]$ and each bin has capacity 1. At each step, the item is placed in the **feasible bin with the highest score** returned by the evolved `score` function. If no bin can fit the item, a new empty bin is opened.

**Objective:** Minimize the average excess ratio over the L1 lower bound (higher fitness = fewer excess bins used = better).

This problem is adapted from the [EoH benchmark suite](https://github.com/FeiLiu36/EoH).

## Interface

The program must expose a function:

```python
def score(item: float, bins: np.ndarray) -> np.ndarray:
    ...
```

- `item`: size of the current item to be packed (scalar float in $(0,1]$).
- `bins`: remaining capacities of feasible bins (1D array, all entries $\ge$ `item`).
- Returns: 1D array of scores (higher score = preferred bin).

## Metrics

| Key | Description | Goal |
|-----|-------------|------|
| `fitness` | avg_excess over all datasets (higher is better) | maximize |
| `excess_pct` | average excess percentage over L1 lower bound | minimize |
| `avg_num_bins` | average bins used per instance | minimize |
| `eval_time` | wall-clock seconds | minimize |

## Notes

- The evaluator uses a fixed set of benchmark instances from `get_instance.py`.
- Fitness is the **negative** average excess ratio, so values near 0 are best and negative values indicate performance worse than the L1 lower bound.
- The best-fit heuristic (baseline) typically achieves fitness around −0.05 to −0.10.
