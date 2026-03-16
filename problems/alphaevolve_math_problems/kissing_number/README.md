# Kissing Number Problem — Dimension 11

## Problem

Find the maximum number of non-overlapping unit spheres that can simultaneously touch (kiss) a central unit sphere in **11-dimensional space**.

Formally, construct a set of integer-coordinate vectors in $\mathbb{Z}^{11}$ such that:

1. No vector is the zero vector.
2. The minimum pairwise squared distance between any two vectors is at least as large as the maximum squared norm of any vector in the set.

This guarantees a valid kissing configuration after normalization. The **kissing number** $K_{11}$ is the size of the largest such set.

**Known bounds:** $K_{11} \le 2432$ (linear programming bound); the current record is **592** and the benchmark target is **≥ 593** (AlphaEvolve's reported best).

## Discrete formulation

The program returns an integer matrix of shape $(m, 11)$ where each row is a sphere center. The evaluator:

1. Rounds all entries to the nearest integer.
2. Checks that no center is the zero vector.
3. Verifies $\min_{i \ne j} \|c_i - c_j\|^2 \ge \max_i \|c_i\|^2$ (non-overlap condition).
4. Reports `num_points = m` as the primary metric.

## Interface

```python
def kissing_number11() -> np.ndarray:
    ...
```

- Returns `np.ndarray` of shape `(m, 11)` with integer (or near-integer) coordinates.
- All entries are rounded to the nearest integer before validation.
- Must satisfy the non-overlap constraint; otherwise a `ValueError` is raised.

## Metrics

| Key | Description | Goal |
|-----|-------------|------|
| `num_points` | number of kissing spheres $m$ | maximize |
| `benchmark_ratio` | $m / 593$ (fitness) | maximize; $> 1.0$ beats benchmark |
| `eval_time` | wall-clock seconds | minimize |
