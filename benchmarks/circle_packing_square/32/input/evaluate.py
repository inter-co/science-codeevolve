# ===--------------------------------------------------------------------------------------===#
#
# Part of the CodeEvolve Project, under the Apache License v2.0.
# See https://github.com/inter-co/science-codeevolve/blob/main/LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0
#
# ===--------------------------------------------------------------------------------------===#
#
# This file implements the evaluator for the circle packing problem on unit square
# for 32 circles.
#
# ===--------------------------------------------------------------------------------------===#
#
# Some of the code in this file is adapted from:
#
# google-deepmind/alphaevolve_results:
# Licensed under the Apache License v2.0.
#
# ===--------------------------------------------------------------------------------------===#

import time
import numpy as np
import json
import sys
import os
from importlib import __import__

BENCHMARK = 2.937944526205518
NUM_CIRCLES = 32
TOL = 1e-6


def validate_packing_radii(radii: np.ndarray) -> None:
    n = len(radii)
    for i in range(n):
        if radii[i] < 0:
            raise ValueError(f"Circle {i} has negative radius {radii[i]}")
        elif np.isnan(radii[i]):
            raise ValueError(f"Circle {i} has nan radius")


def validate_packing_unit_square_wtol(circles: np.ndarray, tol: float = 1e-6) -> None:
    n = len(circles)
    for i in range(n):
        x, y, r = circles[i]
        if (x - r < -tol) or (x + r > 1 + tol) or (y - r < -tol) or (y + r > 1 + tol):
            raise ValueError(
                f"Circle {i} at ({x}, {y}) with radius {r} is outside the unit square"
            )


def validate_packing_overlap_wtol(circles: np.ndarray, tol: float = 1e-6) -> None:
    n = len(circles)
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((circles[i, :2] - circles[j, :2]) ** 2))
            if dist < circles[i, 2] + circles[j, 2] - tol:
                raise ValueError(
                    f"Circles {i} and {j} overlap: dist={dist}, r1+r2={circles[i,2]+circles[j,2]}"
                )


def evaluate(program_path: str, results_path: str = None) -> None:
    abs_program_path = os.path.abspath(program_path)
    program_dir = os.path.dirname(abs_program_path)
    module_name = os.path.splitext(os.path.basename(program_path))[0]

    circles = None
    eval_time = 0
    try:
        sys.path.insert(0, program_dir)
        program = __import__(module_name)

        start_time = time.time()
        circles = program.circle_packing32()
        end_time = time.time()
        eval_time = end_time - start_time

    except Exception as err:
        raise err
    finally:
        if program_dir in sys.path:
            sys.path.remove(program_dir)

    if not isinstance(circles, np.ndarray):
        circles = np.array(circles)

    if circles.shape != (NUM_CIRCLES, 3):
        raise ValueError(
            f"Invalid shapes: circles = {circles.shape}, expected {(NUM_CIRCLES,3)}"
        )

    validate_packing_radii(circles[:, -1])
    validate_packing_overlap_wtol(circles, TOL)
    validate_packing_unit_square_wtol(circles, TOL)

    radii_sum = np.sum(circles[:, -1])
    with open(results_path, "w") as f:
        json.dump(
            {
                "radii_sum": float(radii_sum),
                "benchmark_ratio": float(radii_sum / BENCHMARK),
                "eval_time": float(eval_time),
            },
            f,
            indent=4,
        )


if __name__ == "__main__":
    program_path = sys.argv[1]
    results_path = sys.argv[2]

    evaluate(program_path, results_path)
