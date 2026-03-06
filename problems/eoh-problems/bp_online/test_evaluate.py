#!/usr/bin/env python3
# ===--------------------------------------------------------------------------------------===#
#
# Part of the CodeEvolve Project, under the Apache License v2.0.
# See https://github.com/inter-co/science-codeevolve/blob/main/LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0
#
# ===--------------------------------------------------------------------------------------===#
#
# Test evaluation for Online Bin Packing -- generates Table 1 from the EoH paper.
#
# Usage:
#     python3 test_evaluate.py <heuristic_file> [--test-data-dir <path>]
#
# The heuristic file must define a `score(item, bins) -> scores` function.
# Test data should contain test_dataset_{1k,5k,10k}.pkl from the EoH repo.
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
import argparse
import pickle
import math
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_TEST_DATA = os.path.join(
    SCRIPT_DIR, "..", "..", "..", "..", "EoH",
    "examples", "bp_online", "evaluation", "testingdata",
)

# EoH paper Table 1 values (excess % to L1 lower bound)
EOH_TABLE = {
    ("1k", 100):  {"First Fit": 5.32, "Best Fit": 4.87, "FunSearch": 3.78, "EoH": 2.24},
    ("5k", 100):  {"First Fit": 4.40, "Best Fit": 4.08, "FunSearch": 0.80, "EoH": 0.80},
    ("10k", 100): {"First Fit": 4.44, "Best Fit": 4.09, "FunSearch": 0.33, "EoH": 0.61},
    ("1k", 500):  {"First Fit": 4.97, "Best Fit": 4.50, "FunSearch": 6.75, "EoH": 2.13},
    ("5k", 500):  {"First Fit": 4.27, "Best Fit": 3.91, "FunSearch": 1.47, "EoH": 0.78},
    ("10k", 500): {"First Fit": 4.28, "Best Fit": 3.95, "FunSearch": 0.74, "EoH": 0.61},
}

SIZES = ["1k", "5k", "10k"]
CAPACITIES = [100, 500]
SIZE_TO_NUM = {"1k": 1000, "5k": 5000, "10k": 10000}


# ---------------------------------------------------------------------------
# Bin packing simulation (same as training evaluator)
# ---------------------------------------------------------------------------

def online_binpack(items, capacity, score_fn):
    """Run online bin packing with the given scoring function."""
    bins = np.array([capacity] * len(items))
    for item in items:
        valid = np.where(bins >= item)[0]
        used = np.where(bins < capacity)[0]
        feasible = np.intersect1d(valid, used)
        if len(feasible) == 0:
            new_bins = np.where((bins == capacity) & (bins >= item))[0]
            if len(new_bins) == 0:
                continue
            best = new_bins[0]
        else:
            scores = score_fn(item, bins[feasible])
            best = feasible[np.argmax(scores)]
        bins[best] -= item
    num_used = int(np.sum(bins < capacity))
    return num_used


def l1_lower_bound(items, capacity):
    return math.ceil(sum(items) / capacity)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_test_data(test_data_dir):
    """Load test datasets from pickle files."""
    datasets = {}
    for size_label in SIZES:
        num = SIZE_TO_NUM[size_label]
        pkl_path = os.path.join(test_data_dir, f"test_dataset_{size_label}.pkl")
        if not os.path.exists(pkl_path):
            print(f"Warning: {pkl_path} not found, skipping {size_label}")
            continue
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        instances = data[num]
        datasets[size_label] = [list(map(int, inst)) for inst in instances]
    return datasets


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_on_test(datasets, score_fn):
    """Evaluate heuristic on all test configurations, return excess % dict."""
    results = {}
    for size_label in SIZES:
        if size_label not in datasets:
            continue
        instances = datasets[size_label]
        for capacity in CAPACITIES:
            excess_list = []
            for inst in instances:
                lb = l1_lower_bound(inst, capacity)
                n_bins = online_binpack(inst, capacity, score_fn)
                excess = (n_bins - lb) / lb
                excess_list.append(excess)
            avg_excess = float(np.mean(excess_list)) * 100
            results[(size_label, capacity)] = avg_excess
            print(f"  {size_label} C{capacity}: {avg_excess:.2f}% excess ({len(instances)} instances)")
    return results


# ---------------------------------------------------------------------------
# Table generation
# ---------------------------------------------------------------------------

def print_table(results):
    """Print a markdown table matching Table 1 from the EoH paper."""
    columns = [(s, c) for c in CAPACITIES for s in SIZES]
    col_headers = [f"{s} C{c}" for s, c in columns]

    methods = ["First Fit", "Best Fit", "FunSearch", "EoH", "CodeEvolve"]

    header = "| Method | " + " | ".join(col_headers) + " |"
    sep = "|---|" + "|".join(["---"] * len(columns)) + "|"

    print()
    print("### Table 1: Online Bin Packing Results")
    print("Fraction of excess bins to lower bound (lower is better) on Weibull instances.")
    print()
    print(header)
    print(sep)

    for method in methods:
        row = f"| {method} |"
        for col in columns:
            if method == "CodeEvolve":
                val = results.get(col)
                if val is not None:
                    row += f" **{val:.2f}%** |"
                else:
                    row += " - |"
            else:
                val = EOH_TABLE.get(col, {}).get(method, None)
                if val is not None:
                    row += f" {val:.2f}% |"
                else:
                    row += " - |"
        print(row)

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Test evaluation for Online Bin Packing (Table 1)")
    parser.add_argument("heuristic_file", help="Path to the evolved heuristic .py file")
    parser.add_argument("--test-data-dir", default=DEFAULT_TEST_DATA,
                        help="Path to directory containing test_dataset_*.pkl files")
    args = parser.parse_args()

    # Load heuristic
    abs_path = os.path.abspath(args.heuristic_file)
    module_dir = os.path.dirname(abs_path)
    module_name = os.path.splitext(os.path.basename(abs_path))[0]

    sys.path.insert(0, module_dir)
    try:
        heuristic_module = __import__(module_name)
    finally:
        sys.path.pop(0)

    if not hasattr(heuristic_module, "score"):
        print(f"Error: {args.heuristic_file} does not define a 'score' function.")
        sys.exit(1)

    score_fn = heuristic_module.score

    # Load test data
    test_data_dir = os.path.abspath(args.test_data_dir)
    print(f"Loading test data from: {test_data_dir}")
    datasets = load_test_data(test_data_dir)

    if not datasets:
        print("Error: No test data found.")
        sys.exit(1)

    # Evaluate
    print(f"\nEvaluating heuristic: {args.heuristic_file}")
    results = evaluate_on_test(datasets, score_fn)

    # Print table
    print_table(results)


if __name__ == "__main__":
    main()
