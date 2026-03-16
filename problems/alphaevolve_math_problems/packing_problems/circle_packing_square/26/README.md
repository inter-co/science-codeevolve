# Circle Packing in a Unit Square — n = 26

## Problem

Pack **26 non-overlapping circles** of possibly **different radii** inside the **unit square** $[0,1]^2$, maximizing the **sum of radii**.

$$\text{maximize} \quad \sum_{i=1}^{26} r_i$$

subject to:
- Each circle is contained in $[0,1]^2$: $r_i \le x_i \le 1 - r_i$ and $r_i \le y_i \le 1 - r_i$.
- No two circles overlap: $\|c_i - c_j\| \ge r_i + r_j$ (tolerance $10^{-6}$).
- All radii are non-negative and finite.

**Benchmark:** `radii_sum > 2.6358627564136983` (AlphaEvolve's reported best).

## Interface

```python
def circle_packing26() -> np.ndarray:
    ...
```

- Returns `np.ndarray` of shape `(26, 3)` where each row is `(x, y, r)`.
- Constraints are validated with tolerance $10^{-6}$.

## Metrics

| Key | Description | Goal |
|-----|-------------|------|
| `radii_sum` | $\sum r_i$ | maximize |
| `benchmark_ratio` | `radii_sum / 2.6358...` (fitness) | maximize; $> 1.0$ beats benchmark |
| `eval_time` | wall-clock seconds | minimize |