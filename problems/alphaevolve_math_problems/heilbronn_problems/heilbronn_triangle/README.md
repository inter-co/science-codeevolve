# Heilbronn Triangle Problem — Equilateral Triangle Region, n = 11

## Problem

Place 11 points **inside** a fixed equilateral triangle to **maximize the area of the smallest triangle** formed by any three of these points, normalized by the enclosing triangle area.

$$\text{maximize} \quad \frac{\min_{i < j < k} \text{area}(p_i, p_j, p_k)}{\text{area}(\triangle)}$$

where $\triangle$ is the equilateral triangle with vertices $(0, 0)$, $(1, 0)$, $(0.5,\, \sqrt{3}/2)$, whose area is $\sqrt{3}/4 \approx 0.4330$.

**Benchmark:** `min_area_normalized > 0.036529889880030156` (AlphaEvolve's reported best).

## Interface

```python
def heilbronn_triangle11() -> np.ndarray:
    ...
```

- Returns `np.ndarray` of shape `(11, 2)` — the `(x, y)` coordinates of 11 points
- **All points must lie strictly inside the enclosing triangle** (tolerance $10^{-6}$):
  - $y \geq 0$
  - $\sqrt{3}\,x \leq \sqrt{3} - y$
  - $y \leq \sqrt{3}\,x$
- Constraint violations raise `ValueError` and result in zero fitness

## Metrics

| Key | Description | Goal |
|-----|-------------|------|
| `min_area_normalized` | min triangle area / enclosing triangle area | maximize |
| `benchmark_ratio` | `min_area_normalized / 0.036529...` (fitness) | maximize; $> 1.0$ beats benchmark |
| `eval_time` | wall-clock seconds | minimize |
| `convex_hull_fraction` | fraction of 11 points on their own convex hull | MAP-Elites feature |
| `min_nn_dist` | minimum pairwise distance between any two points | MAP-Elites feature |