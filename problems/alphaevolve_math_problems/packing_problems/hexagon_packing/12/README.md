# Hexagon Packing — n = 12

## Problem

Pack **12 non-overlapping unit regular hexagons** inside the **smallest possible regular hexagon**, minimizing the side length $s$ of the outer hexagon (equivalently, maximizing $1/s$).

$$\text{maximize} \quad \frac{1}{s}$$

subject to:
- All 12 inner hexagons (each with side length 1) are fully **contained** inside the outer hexagon of side length $s$.
- All pairs of inner hexagons are **disjoint** (non-overlapping, tolerance $10^{-6}$).
- The outer hexagon can be positioned and rotated freely.

**Benchmark:** `inv_outer_hex_side_length > 1/3.9419123 ≈ 0.25368` (AlphaEvolve's reported best).

## Interface

```python
def hexagon_packing_12() -> tuple[np.ndarray, np.ndarray, float]:
    ...
```

Returns a tuple of:
- `inner_hex_data`: `np.ndarray` of shape `(12, 3)` — each row is `(x, y, angle_degrees)` for an inner hexagon.
- `outer_hex_data`: `np.ndarray` of shape `(3,)` — `(center_x, center_y, angle_degrees)` for the outer hexagon.
- `outer_hex_side_length`: `float` — side length of the outer hexagon (to minimize).

## Metrics

| Key | Description | Goal |
|-----|-------------|------|
| `inv_outer_hex_side_length` | $1/s$ | maximize |
| `benchmark_ratio` | `inv_outer_hex_side_length / (1/3.9419123)` (fitness) | maximize; $> 1.0$ beats benchmark |
| `eval_time` | wall-clock seconds | minimize |