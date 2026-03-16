# Minimizing Max/Min Distance Ratio — Dimension 3, n = 14

## Problem

Place **14 points** in $\mathbb{R}^3$ to **maximize the ratio of minimum to maximum pairwise distance**:

$$\text{maximize} \quad \left(\frac{d_{\min}}{d_{\max}}\right)^2, \quad d_{\min} = \min_{i \ne j} \|p_i - p_j\|, \quad d_{\max} = \max_{i \ne j} \|p_i - p_j\|$$

This is the 3-dimensional analogue of the 2D problem. Optimal configurations tend to be highly symmetric arrangements (e.g., based on icosahedra or other Platonic solids).

**Benchmark:** `min_max_ratio > 1/4.165849767² ≈ 0.057594` (AlphaEvolve's reported best).

## Interface

```python
def min_max_dist_dim3_14() -> np.ndarray:
    ...
```

- Returns `np.ndarray` of shape `(14, 3)` — the $(x, y, z)$ coordinates of 14 points.
- No boundary constraint; points can be placed anywhere in $\mathbb{R}^3$.

## Metrics

| Key | Description | Goal |
|-----|-------------|------|
| `min_max_ratio` | $(d_{\min}/d_{\max})^2$ (fitness) | maximize |
| `benchmark_ratio` | `min_max_ratio / (1/4.165849767²)` | maximize; $> 1.0$ beats benchmark |
| `eval_time` | wall-clock seconds | minimize |