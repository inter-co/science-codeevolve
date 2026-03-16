# Circle Packing in a Rectangle of Perimeter 4 — n = 21

## Problem

Pack **21 non-overlapping circles** of possibly **different radii** inside a rectangle whose **perimeter equals 4** (i.e., $2(w + h) = 4$, so $w + h = 2$), maximizing the **sum of radii**.

$$\text{maximize} \quad \sum_{i=1}^{21} r_i$$

subject to:
- All circles fit inside the rectangle (no circle extends beyond the boundary).
- No two circles overlap (pairwise center distances $\ge r_i + r_j$, with tolerance $10^{-6}$).
- All radii are non-negative.

The rectangle dimensions $(w, h)$ are determined by the minimum bounding rectangle of the circle set; the constraint is that $w + h \le 2$ (perimeter $\le 4$).

**Benchmark:** `radii_sum > 2.3658321334167627` (AlphaEvolve's reported best).

## Interface

```python
def circle_packing21() -> np.ndarray:
    ...
```

- Returns `np.ndarray` of shape `(21, 3)` where each row is `(x, y, r)` — center coordinates and radius.
- Radii must be non-negative and finite.
- Circles must not overlap (tolerance $10^{-6}$).
- All circles must fit inside a rectangle with $w + h \le 2 + 10^{-6}$.

## Metrics

| Key | Description | Goal |
|-----|-------------|------|
| `radii_sum` | $\sum r_i$ | maximize |
| `benchmark_ratio` | `radii_sum / 2.3658...` (fitness) | maximize; $> 1.0$ beats benchmark |
| `eval_time` | wall-clock seconds | minimize |