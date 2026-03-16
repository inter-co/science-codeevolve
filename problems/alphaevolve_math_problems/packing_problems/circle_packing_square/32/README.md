# Circle Packing in a Unit Square — n = 32

## Problem

Pack **32 non-overlapping circles** of possibly **different radii** inside the **unit square** $[0,1]^2$, maximizing the **sum of radii**.

$$\text{maximize} \quad \sum_{i=1}^{32} r_i$$

subject to:
- Each circle is contained in $[0,1]^2$: $r_i \le x_i \le 1 - r_i$ and $r_i \le y_i \le 1 - r_i$.
- No two circles overlap: $\|c_i - c_j\| \ge r_i + r_j$ (tolerance $10^{-6}$).
- All radii are non-negative and finite.

**Benchmark:** `radii_sum > 2.937944526205518` (AlphaEvolve's reported best).

## Interface

```python
def circle_packing32() -> np.ndarray:
    ...
```

- Returns `np.ndarray` of shape `(32, 3)` where each row is `(x, y, r)`.
- Constraints are validated with tolerance $10^{-6}$.

## Metrics

| Key | Description | Goal |
|-----|-------------|------|
| `radii_sum` | $\sum r_i$ | maximize |
| `benchmark_ratio` | `radii_sum / 2.9379...` (fitness) | maximize; $> 1.0$ beats benchmark |
| `eval_time` | wall-clock seconds | minimize |