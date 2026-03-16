# Heilbronn Triangle Problem — Convex Region, n = 14

## Problem

Place 14 points in 2D space to **maximize the area of the smallest triangle** formed by any three of these points, normalized by the convex hull area of the point set.

$$\text{maximize} \quad \frac{\min_{i < j < k} \text{area}(p_i, p_j, p_k)}{\text{area}(\text{ConvexHull}(p_1, \ldots, p_{14}))}$$

The normalization by the convex hull area makes the problem scale- and translation-invariant.

**Benchmark:** `min_area_normalized > 0.027835571458482138` (AlphaEvolve's reported best).

Note: the benchmark value is slightly lower than for $n = 13$, reflecting the increased difficulty of packing one additional point.

## Interface

```python
def heilbronn_convex14() -> np.ndarray:
    ...
```

- Returns `np.ndarray` of shape `(14, 2)` — the `(x, y)` coordinates of 14 points
- Points can be anywhere in $\mathbb{R}^2$; no boundary constraint
- The convex hull must be non-degenerate (positive area)

## Metrics

| Key | Description | Goal |
|-----|-------------|------|
| `min_area_normalized` | min triangle area / convex hull area | maximize |
| `benchmark_ratio` | `min_area_normalized / 0.027835...` (fitness) | maximize; $> 1.0$ beats benchmark |
| `eval_time` | wall-clock seconds | minimize |
| `convex_hull_fraction` | fraction of 14 points on the convex hull | MAP-Elites feature |
| `min_nn_dist` | minimum pairwise distance between any two points | MAP-Elites feature |