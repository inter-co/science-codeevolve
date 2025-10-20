# ===--------------------------------------------------------------------------------------===#
#
# Part of the CodeEvolve Project, under the Apache License v2.0.
# See https://github.com/inter-co/science-codeevolve/blob/main/LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0
#
# ===--------------------------------------------------------------------------------------===#
#
# This file implements the evaluator for the third autocorrelation inequality problem.
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
BENCHMARK = 1.4556427953745406


def verify_c3_solution(f_values: np.ndarray, c3_achieved: float, n_points: int):
    """Verify the solution for the C3 UPPER BOUND optimization."""

    if f_values.shape != (n_points,):
        raise ValueError(f"Expected function values shape {(n_points,)}. Got {f_values.shape}.")

    # Recompute C3 using NumPy to verify
    dx = 0.5 / n_points

    # Squared integral of f
    integral_f_sq = (np.sum(f_values) * dx) ** 2

    if integral_f_sq < 1e-9:
        raise ValueError("Function integral is close to zero, ratio is unstable.")

    # Max absolute value of the scaled autoconvolution
    conv = np.convolve(f_values, f_values, mode="full")
    scaled_conv = conv * dx
    max_abs_conv = np.max(np.abs(scaled_conv))

    computed_c3 = max_abs_conv / integral_f_sq

    delta = abs(computed_c3 - c3_achieved)
    if delta > 1e-3:
        raise ValueError(
            f"C3 mismatch: reported {c3_achieved:.6f}, computed {computed_c3:.6f}, delta: {delta:.6f}"
        )


def evaluate(program_path: str, results_path: str):
    abs_program_path = os.path.abspath(program_path)
    program_dir = os.path.dirname(abs_program_path)
    module_name = os.path.splitext(os.path.basename(program_path))[0]

    try:
        sys.path.insert(0, program_dir)
        program = __import__(module_name)
        start_time = time.time()
        f_values, c3_achieved, loss, n_points = program.run()
        end_time = time.time()
        eval_time = end_time - start_time
    finally:
        if program_dir in sys.path:
            sys.path.remove(program_dir)

    verify_c3_solution(f_values, c3_achieved, n_points)

    with open(results_path, "w") as f:
        json.dump(
            {
                "c3": float(c3_achieved),
                "c3_ratio": BENCHMARK / float(c3_achieved),
                "loss": float(loss),
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
