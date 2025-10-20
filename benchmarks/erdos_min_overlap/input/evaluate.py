# ===--------------------------------------------------------------------------------------===#
#
# Part of the CodeEvolve Project, under the Apache License v2.0.
# See https://github.com/inter-co/science-codeevolve/blob/main/LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0
#
# ===--------------------------------------------------------------------------------------===#
#
# This file implements the evaluator for the erdos minimum overlap problem.
#
# ===--------------------------------------------------------------------------------------===#
#
# Some of the code in this file is adapted from:
#
# google-deepmind/alphaevolve_results:
# Licensed under the Apache License v2.0.
#
# ===--------------------------------------------------------------------------------------===#

import sys
import os
from importlib import __import__
import time
import json
import numpy as np

# Known bounds
BENCHMARK = 0.38092303510845016


def verify_c5_solution(h_values: np.ndarray, c5_achieved: float, n_points: int):
    """Verifies the C5 upper bound solution."""

    if h_values.shape != (n_points,):
        raise ValueError(f"Expected h shape ({n_points},), got {h_values.shape}")

    # Verify h(x) in [0, 1] constraint
    if np.any(h_values < 0) or np.any(h_values > 1):
        raise ValueError(f"h(x) is not in [0, 1]. Range: [{h_values.min()}, {h_values.max()}]")

    # Verify integral of h = 1 constraint
    dx = 2.0 / n_points
    integral_h = np.sum(h_values) * dx
    if not np.isclose(integral_h, 1.0, atol=1e-3):
        raise ValueError(f"Integral of h is not close to 1. Got: {integral_h:.6f}")

    # Re-calculate the C5 bound using np.correlate
    j_values = 1.0 - h_values
    correlation = np.correlate(h_values, j_values, mode="full") * dx
    computed_c5 = np.max(correlation)

    # Check for consistency
    if not np.isclose(computed_c5, c5_achieved, atol=1e-4):
        raise ValueError(f"C5 mismatch: reported {c5_achieved:.6f}, computed {computed_c5:.6f}")


def evaluate(program_path: str, results_path: str):
    abs_program_path = os.path.abspath(program_path)
    program_dir = os.path.dirname(abs_program_path)
    module_name = os.path.splitext(os.path.basename(program_path))[0]

    try:
        sys.path.insert(0, program_dir)
        program = __import__(module_name)
        start_time = time.time()
        h_values, c5_bound, n_points = program.run()
        end_time = time.time()
        eval_time = end_time - start_time
    finally:
        if program_dir in sys.path:
            sys.path.remove(program_dir)

    verify_c5_solution(h_values, c5_bound, n_points)

    with open(results_path, "w") as f:
        json.dump(
            {
                "c5_bound": float(c5_bound),
                "inverse_c5": BENCHMARK / float(c5_bound),
                "n_points": int(n_points),
                "eval_time": float(eval_time),
            },
            f,
            indent=4,
        )


if __name__ == "__main__":
    program_path = sys.argv[1]
    results_path = sys.argv[2]

    evaluate(program_path, results_path)
