# ===--------------------------------------------------------------------------------------===#
#
# Part of the CodeEvolve Project, under the Apache License v2.0.
# See https://github.com/inter-co/science-codeevolve/blob/main/LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0
#
# ===--------------------------------------------------------------------------------------===#
#
# This file implements the evaluator for the second autocorrelation inequality problem.
#
# ===--------------------------------------------------------------------------------------===#
#
# Some of the code in this file is adapted from:
#
# https://github.com/google-deepmind/alphaevolve_results:
# Licensed under the Apache License v2.0.
#
# ===--------------------------------------------------------------------------------------===#

import sys
import os
import time
import json
import numpy as np
from importlib import __import__

BENCHMARK = 0.962

def verify_c2_solution(f_values: np.ndarray) -> dict:
    """
    Verifies the C2 lower bound solution using the rigorous, unitless, piecewise linear integral method.

    Returns:
        A dict with keys:
            c2: the computed C2 lower bound (maximize this).
            seq_length: number of steps n in the step function.
            density: fraction of steps with positive height (> 1e-6).
            conv_concentration: max(f*f) / sum(f*f), peakedness of the autoconvolution.
    """
    n_points = len(f_values)
    if n_points == 0 or f_values is None:
        raise ValueError("Received empty function values.")
    if f_values.shape != (n_points,):
        raise ValueError(f"Expected function values shape {(n_points,)}. Got {f_values.shape}.")
    if np.any(f_values < -1e-6):  # Allow for small floating point errors
        raise ValueError("Function must be non-negative.")

    f_nonneg = np.maximum(f_values, 0.0)

    convolution = np.convolve(f_nonneg, f_nonneg, mode="full")

    num_conv_points = len(convolution)
    x_points = np.linspace(-0.5, 0.5, num_conv_points + 2)
    x_intervals = np.diff(x_points)
    y_points = np.concatenate(([0], convolution, [0]))

    l2_norm_squared = 0.0
    for i in range(len(convolution) + 1):
        y1, y2, h = y_points[i], y_points[i + 1], x_intervals[i]
        interval_l2_squared = (h / 3) * (y1**2 + y1 * y2 + y2**2)
        l2_norm_squared += interval_l2_squared

    norm_1 = np.sum(np.abs(convolution)) / (len(convolution) + 1)

    norm_inf = float(np.max(np.abs(convolution)))

    if norm_1 * norm_inf < 1e-12:
        raise ValueError(f"Norm product too close to zero: norm_1={norm_1}, norm_inf={norm_inf}")

    computed_c2 = float(l2_norm_squared / (norm_1 * norm_inf))

    density = float(np.sum(f_nonneg > 1e-6) / n_points)
    sum_conv = float(np.sum(convolution))
    conv_concentration = float(norm_inf / sum_conv) if sum_conv > 0 else 0.0

    return {
        "c2": computed_c2,
        "seq_length": n_points,
        "density": density,
        "conv_concentration": conv_concentration,
    }


def evaluate(program_path: str, results_path: str):
    abs_program_path = os.path.abspath(program_path)
    program_dir = os.path.dirname(abs_program_path)
    module_name = os.path.splitext(os.path.basename(program_path))[0]
    
    f_values = None
    eval_time = 0
    
    try:
        sys.path.insert(0, program_dir)
        program = __import__(module_name)
        start_time = time.time()
        f_values_list = program.construct_function()
        end_time = time.time()
        eval_time = end_time - start_time
        
        if not isinstance(f_values_list, (list, np.ndarray)):
            raise ValueError(f"construct_function must return list or np.ndarray, got {type(f_values_list)}")
        f_values = np.array(f_values_list, dtype=float)
        
    except Exception as err:
        raise err
    finally:
        if program_dir in sys.path:
            sys.path.remove(program_dir)
    
    seq_metrics = verify_c2_solution(f_values)
    c2 = seq_metrics["c2"]

    with open(results_path, "w") as f:
        json.dump(
            {
                "c2": float(c2),
                "benchmark_ratio": float(c2) / BENCHMARK,
                "eval_time": float(eval_time),
                "seq_length": float(seq_metrics["seq_length"]),
                "density": seq_metrics["density"],
                "conv_concentration": seq_metrics["conv_concentration"],
            },
            f,
            indent=4,
        )


if __name__ == "__main__":
    program_path = sys.argv[1]
    results_path = sys.argv[2]
    evaluate(program_path, results_path)
