# First Autocorrelation Inequality (FirstAutocorrIneq)

## Problem

Find a non-negative step function that minimizes the constant C₁, defined as the largest constant satisfying

$$\max_{-1/2 \le t \le 1/2}(f * f)(t) \ge C_1 \left(\int_{-1/4}^{1/4} f(x)\,dx\right)^2$$

for all non-negative $f : \mathbb{R} \to \mathbb{R}$. Upper bounds on C₁ are obtained by explicit step-function constructions ([Matolcsi & Vinuesa, 2009](https://arxiv.org/abs/0907.1379)).

**Known bounds:** $1.28 \le C_1 \le 1.5098$

**Benchmark:** $C_1 < 1.5031$ (equivalently $1/C_1 > 0.6653$), matching AlphaEvolve's reported best.

## Discrete formulation

A step function with $n$ equal-width steps and heights $a = [a_1, \ldots, a_n]$ (non-negative) satisfies:

$$C_1 = \frac{2n \cdot \max(b)}{(\sum a)^2}, \quad b = a * a \text{ (autoconvolution)}$$

The objective is to **minimize** $C_1$, equivalently **maximize** $1/C_1$.

## Interface

The program must expose a function:

```python
def search_for_best_sequence() -> list[float]:
    ...
```

- Returns a list of non-negative floats (step heights)
- Heights are clamped to $[0, 1000]$ by the evaluator
- `sum(a)` must exceed $0.01$; otherwise the evaluation is rejected

## Metrics

| Key | Description | Goal |
|-----|-------------|------|
| `inv_c1` | $1/C_1$ | maximize |
| `benchmark_ratio` | $1.5031 / C_1$ (fitness) | maximize; $> 1.0$ beats benchmark |
| `eval_time` | wall-clock seconds | minimize |
| `seq_length` | number of steps $n$ | MAP-Elites feature |
| `density` | fraction of steps with height $> 10^{-6}$ | MAP-Elites feature |
| `conv_concentration` | $\max(b) / \sum(b)$, peakedness of autoconvolution | MAP-Elites feature |