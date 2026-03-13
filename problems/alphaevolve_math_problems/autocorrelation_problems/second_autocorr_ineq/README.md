# Second Autocorrelation Inequality (SecondAutocorrIneq)

## Problem

Find a non-negative step function that maximizes the constant C₂, defined as the smallest constant satisfying

$$\|f * f\|_2^2 \le C_2 \cdot \|f * f\|_1 \cdot \|f * f\|_\infty$$

for all non-negative $f : \mathbb{R} \to \mathbb{R}$. Hölder's inequality immediately gives $C_2 \le 1$. Lower bounds on C₂ are obtained by explicit step-function constructions ([Matolcsi & Vinuesa, 2009](https://arxiv.org/abs/0907.1379)).

**Known bounds:** $C_2 \le 1$

**Benchmark:** $C_2 > 0.962$, matching AlphaEvolve's reported best.

## Discrete formulation

For step heights $a = [a_1, \ldots, a_n]$ (non-negative), let $b = a * a$ (autoconvolution). The evaluator computes:

$$C_2 = \frac{\|b\|_2^2}{\|b\|_1 \cdot \|b\|_\infty}$$

where the norms are approximated via piecewise linear integration over a unit interval. The objective is to **maximize** $C_2$.

A high $C_2$ requires the autoconvolution $b$ to be as **flat** as possible relative to its peak — uniform $b$ would give $C_2 = 1$; a delta-like $b$ gives $C_2 \to 0$.

## Interface

The program must expose a function:

```python
def construct_function() -> list[float]:
    ...
```

- Returns a list or `np.ndarray` of non-negative floats (step heights)
- Values slightly below zero (floating-point noise, $> -10^{-6}$) are clamped to zero
- Values strictly below $-10^{-6}$ raise a validation error

## Metrics

| Key | Description | Goal |
|-----|-------------|------|
| `c2` | the computed $C_2$ value | maximize |
| `benchmark_ratio` | $C_2 / 0.962$ (fitness) | maximize; $> 1.0$ beats benchmark |
| `eval_time` | wall-clock seconds | minimize |
| `seq_length` | number of steps $n$ | MAP-Elites feature |
| `density` | fraction of steps with height $> 10^{-6}$ | MAP-Elites feature |
| `conv_concentration` | $\max(b) / \sum(b)$, peakedness of autoconvolution | MAP-Elites feature |
