# ===--------------------------------------------------------------------------------------===#
#
# Part of the CodeEvolve Project, under the Apache License v2.0.
# See https://github.com/inter-co/science-codeevolve/blob/main/LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0
#
# ===--------------------------------------------------------------------------------------===#
#
# This file implements the evaluator for the first autocorrelation inequality.
#
# ===--------------------------------------------------------------------------------------===#
#
# Some of the code in this file is adapted from:
#
# google-deepmind/alphaevolve_repository_of_problems:
# Licensed under the Apache License v2.0.
#
# ===--------------------------------------------------------------------------------------===#

import numpy as np 
import sys
import os
import time
import json
from importlib import __import__

BENCHMARK: float = 1.5031

def evaluate_sequence(sequence: list[float]) -> float:
    """
    Evaluates a sequence of coefficients with enhanced security checks.
    """
    if not isinstance(sequence, list):
        raise ValueError(f"Sequence type expected to be list, received {type(sequence)}")

    if not sequence:
        raise ValueError("Sequence cannot be None.")

    for x in sequence:
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            raise ValueError("Sequence entries must be integers or floats.")
        if np.isnan(x) or np.isinf(x):
            raise ValueError("Sequence cannot contain nans or infs.")

    sequence = [float(x) for x in sequence]
    sequence = [max(0, x) for x in sequence]
    sequence = [min(1000.0, x) for x in sequence]

    n = len(sequence)
    b_sequence = np.convolve(sequence, sequence)
    max_b = max(b_sequence)
    sum_a = np.sum(sequence)

    if sum_a < 0.01:
        raise ValueError(f"Sum of sequence entries too close to zero: {sum_a}.")

    return float(2 * n * max_b / (sum_a**2))

def evaluate(program_path: str, results_path: str = None) -> None:
    abs_program_path = os.path.abspath(program_path)
    program_dir = os.path.dirname(abs_program_path)
    module_name = os.path.splitext(os.path.basename(program_path))[0]

    sequence = None
    eval_time = 0
    try:
        sys.path.insert(0, program_dir)
        program = __import__(module_name)

        start_time = time.time()
        sequence = program.search_for_best_sequence()
        end_time = time.time()
        eval_time = end_time - start_time
    except Exception as err:
        raise err
    finally:
        if program_dir in sys.path:
            sys.path.remove(program_dir)

    c1 = evaluate_sequence(sequence)

    with open(results_path, "w") as f:
        json.dump(
            {
                "inv_c1": float(1/c1),
                "benchmark_ratio": float(BENCHMARK/c1),
                "eval_time": float(eval_time),
            },
            f,
            indent=4,
        )


if __name__ == "__main__":
    program_path = sys.argv[1]
    results_path = sys.argv[2]

    evaluate(program_path, results_path)
