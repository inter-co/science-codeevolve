# Flow Shop Scheduling with Guided Local Search (FSSP-GLS)

## Problem

Design a **penalty and perturbation heuristic** for **Guided Local Search (GLS)** applied to the **Permutation Flow Shop Scheduling Problem** (PFSP). The evolved function modifies job processing times and selects jobs to perturb, enabling the GLS framework to escape local optima.

The goal is to minimize the **makespan** (total completion time) of all jobs across all machines. The evaluation uses 3 benchmark PFSP instances, runs up to 1000 GLS iterations with a 30-second time limit per instance, and reports the average gap to the best known solution.

This problem is adapted from the [EoH benchmark suite](https://github.com/FeiLiu36/EoH).

## Interface

The program must expose a function:

```python
def get_matrix_and_jobs(
    current_sequence: list,
    time_matrix: np.ndarray,
    m: int,
    n: int,
) -> tuple[np.ndarray, list]:
    ...
```

- `current_sequence`: current job ordering (list of job indices, length `n`).
- `time_matrix`: `np.ndarray` of shape `(n, m)` — processing time of job $j$ on machine $k$.
- `m`: number of machines.
- `n`: number of jobs.
- Returns:
  - `new_matrix`: updated `np.ndarray` of shape `(n, m)` with modified processing times (penalties).
  - `perturb_jobs`: list of job indices to perturb (length must be $> 1$ and $\le 5$).

## Metrics

| Key | Description | Goal |
|-----|-------------|------|
| `fitness` | avg_gap across instances (higher is better) | maximize |
| `avg_gap_pct` | average percentage gap to best-known makespan | minimize |
| `eval_time` | wall-clock seconds | minimize |

## Notes

- Training instances are located in `TrainingData/` (64 PFSP instances).
- The GLS loop, local search (swap + relocate), and makespan computation are fixed; only `get_matrix_and_jobs` is evolved.
- Core scheduling routines use Numba JIT compilation for performance.
- The EoH baseline typically achieves gaps of 1–5% on standard benchmarks.
