# Heilbronn Triangle Problem — Convex Region, n = 13

## Problem

Place 13 points in 2D space to **maximize the area of the smallest triangle** formed by any three of these points, normalized by the convex hull area of the point set.

$$\text{maximize} \quad \frac{\min_{i < j < k} \text{area}(p_i, p_j, p_k)}{\text{area}(\text{ConvexHull}(p_1, \ldots, p_{13}))}$$

The normalization by the convex hull area makes the problem scale- and translation-invariant: what matters is the *relative* spread of triangles within the configuration.

**Benchmark:** `min_area_normalized > 0.030936889034895654` (AlphaEvolve's reported best).

## Interface

```python
def heilbronn_convex13() -> np.ndarray:
    ...
```

- Returns `np.ndarray` of shape `(13, 2)` — the `(x, y)` coordinates of 13 points
- Points can be anywhere in $\mathbb{R}^2$; no boundary constraint
- The convex hull must be non-degenerate (positive area)

## Metrics

| Key | Description | Goal |
|-----|-------------|------|
| `min_area_normalized` | min triangle area / convex hull area | maximize |
| `benchmark_ratio` | `min_area_normalized / 0.030936...` (fitness) | maximize; $> 1.0$ beats benchmark |
| `eval_time` | wall-clock seconds | minimize |
| `convex_hull_fraction` | fraction of 13 points on the convex hull | MAP-Elites feature |
| `min_nn_dist` | minimum pairwise distance between any two points | MAP-Elites feature |