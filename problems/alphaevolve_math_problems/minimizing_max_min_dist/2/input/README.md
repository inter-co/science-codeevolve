# Minimizing Max/Min Distance Ratio — Dimension 2, n = 16

## Problem

Place **16 points** in $\mathbb{R}^2$ to **maximize the ratio of minimum to maximum pairwise distance**:

$$\text{maximize} \quad \left(\frac{d_{\min}}{d_{\max}}\right)^2, \quad d_{\min} = \min_{i \ne j} \|p_i - p_j\|, \quad d_{\max} = \max_{i \ne j} \|p_i - p_j\|$$

The squared ratio is used as the fitness metric to avoid numerical issues near zero. A perfect configuration (all pairwise distances equal) achieves a ratio of 1; a degenerate configuration achieves 0.

**Benchmark:** `min_max_ratio > 1/12.889266112² ≈ 0.006017` (AlphaEvolve's reported best).

## Interface

```python
def min_max_dist_dim2_16() -> np.ndarray:
    ...
```

- Returns `np.ndarray` of shape `(16, 2)` — the $(x, y)$ coordinates of 16 points.
- No boundary constraint; points can be placed anywhere in $\mathbb{R}^2$.

## Metrics

| Key | Description | Goal |
|-----|-------------|------|
| `min_max_ratio` | $(d_{\min}/d_{\max})^2$ (fitness) | maximize |
| `benchmark_ratio` | `min_max_ratio / (1/12.889266112²)` | maximize; $> 1.0$ beats benchmark |
| `eval_time` | wall-clock seconds | minimize |