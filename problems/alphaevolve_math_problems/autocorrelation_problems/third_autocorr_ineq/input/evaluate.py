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
BENCHMARK = 1.4556


def verify_c3_solution(f_values: np.ndarray) -> float:
    """Verify the solution for the C3 UPPER BOUND optimization."""

    n_points = len(f_values)
    if n_points == 0 or f_values is None:
        raise ValueError("Received empty function values.")
    if f_values.shape != (n_points,):
        raise ValueError(f"Expected function values shape {(n_points,)}. Got {f_values.shape}.")

    convolution = np.convolve(f_values, f_values)
    den = (np.sum(f_values)**2)
    if den < 1e-12:
        raise ValueError(f"Sum of squared values of the function is too close to zero {den}.")
    c3 = abs(2 * len(f_values) * np.max(convolution) / den)

    return c3


def evaluate(program_path: str, results_path: str):
    abs_program_path = os.path.abspath(program_path)
    program_dir = os.path.dirname(abs_program_path)
    module_name = os.path.splitext(os.path.basename(program_path))[0]

    try:
        sys.path.insert(0, program_dir)
        program = __import__(module_name)
        start_time = time.time()
        f_values_list = program.construct_function()
        end_time = time.time()
        eval_time = end_time - start_time
        
        # Convert to numpy array
        if not isinstance(f_values_list, (list, np.ndarray)):
            raise ValueError(f"construct_function must return list or np.ndarray, got {type(f_values_list)}")
        f_values = np.array(f_values_list, dtype=float)
    finally:
        if program_dir in sys.path:
            sys.path.remove(program_dir)

    c3 = verify_c3_solution(f_values)

    with open(results_path, "w") as f:
        json.dump(
            {
                "inv_c3": float(1/c3),
                "benchmark_ratio": BENCHMARK / float(c3),
                "eval_time": float(eval_time),
            },
            f,
            indent=4,
        )


if __name__ == "__main__":
    program_path = sys.argv[1]
    results_path = sys.argv[2]

    evaluate(program_path, results_path)
