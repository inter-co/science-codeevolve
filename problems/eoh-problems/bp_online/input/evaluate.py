# ===--------------------------------------------------------------------------------------===#
#
# Part of the CodeEvolve Project, under the Apache License v2.0.
# See https://github.com/inter-co/science-codeevolve/blob/main/LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0
#
# ===--------------------------------------------------------------------------------------===#
#
# Online Bin Packing Evaluator 
#
# This evaluator loads bin packing instances, runs the evolved heuristic score
# function on them, and reports fitness metrics in JSON format.
#
# ===--------------------------------------------------------------------------------------===#
#
# Some of the code in this file is adapted from:
#
# https://github.com/FeiLiu36/EoH:
# Licensed under the MIT License.
#
# ===--------------------------------------------------------------------------------------===#

import sys
import os
import json
import time
import numpy as np


# ---------------------------------------------------------------------------
# Instance data
# ---------------------------------------------------------------------------

def load_instances():
    """Load bin packing instances from the co-located get_instance module."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, script_dir)
    from get_instance import GetData
    data = GetData()
    datasets, lower_bounds = data.get_instances()
    if script_dir in sys.path:
        sys.path.remove(script_dir)
    return datasets, lower_bounds


# ---------------------------------------------------------------------------
# Online bin packing simulation
# ---------------------------------------------------------------------------

def get_valid_bin_indices(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns indices of bins in which item can fit."""
    return np.nonzero((bins - item) >= 0)[0]


def online_binpack(items: tuple, bins: np.ndarray, score_fn) -> tuple:
    """Performs online bin packing of *items* into *bins* using *score_fn*.

    At each step, the item is placed into the feasible bin with the highest
    score returned by ``score_fn(item, feasible_bins)``.

    Returns:
        packing: list of lists (items packed in each used bin).
        bins:    remaining capacities after packing.
    """
    packing = [[] for _ in bins]
    for item in items:
        valid_bin_indices = get_valid_bin_indices(item, bins)
        if len(valid_bin_indices) == 0:
            continue
        priorities = score_fn(item, bins[valid_bin_indices])
        best_bin = valid_bin_indices[np.argmax(priorities)]
        bins[best_bin] -= item
        packing[best_bin].append(item)
    packing = [bin_items for bin_items in packing if bin_items]
    return packing, bins


# ---------------------------------------------------------------------------
# Fitness computation
# ---------------------------------------------------------------------------

def compute_fitness(datasets, lower_bounds, score_fn):
    """Run the heuristic on all instances and return aggregate metrics.

    Fitness is defined as the *negative* excess ratio over the L1 lower bound,
    so that **higher is better** (0.0 means optimal packing).
    """
    total_bins_used = 0
    total_instances = 0
    excess_per_dataset = {}

    for name, dataset in datasets.items():
        num_bins_list = []
        for _, instance in dataset.items():
            capacity = instance["capacity"]
            items = np.array(instance["items"])
            bins = np.array([capacity] * instance["num_items"])
            _, bins_packed = online_binpack(items, bins, score_fn)
            num_bins = int((bins_packed != capacity).sum())
            num_bins_list.append(num_bins)

        avg_num_bins = float(np.mean(num_bins_list))
        lb = lower_bounds[name]
        excess = (avg_num_bins - lb) / lb
        excess_per_dataset[name] = excess
        total_bins_used += sum(num_bins_list)
        total_instances += len(num_bins_list)

    avg_excess = float(np.mean(list(excess_per_dataset.values())))
    avg_bins = total_bins_used / max(total_instances, 1)

    return {
        "fitness": float(-avg_excess),
        "excess_pct": float(avg_excess * 100),
        "avg_num_bins": float(avg_bins),
    }


# ---------------------------------------------------------------------------
# Main evaluator entry point
# ---------------------------------------------------------------------------

def evaluate(program_path: str, results_path: str) -> None:
    """CodeEvolve evaluator interface.

    Args:
        program_path: Path to the evolved Python file containing ``score()``.
        results_path: Path where JSON results should be written.
    """
    abs_program_path = os.path.abspath(program_path)
    program_dir = os.path.dirname(abs_program_path)
    module_name = os.path.splitext(os.path.basename(program_path))[0]

    try:
        sys.path.insert(0, program_dir)
        program = __import__(module_name)
    except Exception as err:
        raise err
    finally:
        if program_dir in sys.path:
            sys.path.remove(program_dir)

    if not hasattr(program, "score"):
        raise AttributeError(
            f"Evolved program at {program_path} does not define a 'score' function."
        )

    datasets, lower_bounds = load_instances()

    start_time = time.time()
    metrics = compute_fitness(datasets, lower_bounds, program.score)
    eval_time = time.time() - start_time
    metrics["eval_time"] = float(eval_time)

    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    program_path = sys.argv[1]
    results_path = sys.argv[2]
    evaluate(program_path, results_path)
