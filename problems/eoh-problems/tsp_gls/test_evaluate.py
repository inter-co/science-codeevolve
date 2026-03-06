#!/usr/bin/env python3
# ===--------------------------------------------------------------------------------------===#
#
# Part of the CodeEvolve Project, under the Apache License v2.0.
# See https://github.com/inter-co/science-codeevolve/blob/main/LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0
#
# ===--------------------------------------------------------------------------------------===#
# Test evaluation for TSP with Guided Local Search -- generates Table 10 from the EoH paper.
#
# Usage:
#     python3 test_evaluate.py <heuristic_file> [--test-data-dir <path>] [--max-instances N]
#
# The heuristic file must define an `update_edge_distance(edge_distance, local_opt_tour, edge_n_used)`
# function. Test data should contain TSP20.pkl, TSP50.pkl, TSP100.pkl from the EoH repo.
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
import time
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(SCRIPT_DIR, "input")

# Add input/ to path so gls/ and utils/ are importable
if INPUT_DIR not in sys.path:
    sys.path.insert(0, INPUT_DIR)

from gls.gls_run import solve_instance

DEFAULT_TEST_DATA = os.path.join(
    SCRIPT_DIR, "..", "..", "..", "..", "EoH",
    "examples", "user_tsp_gls", "TestingData",
)

# GLS parameters (same as EoH paper)
TIME_LIMIT = 10
ITE_MAX = 1000
PERTURBATION_MOVES = 1

TSP_SIZES = ["TSP20", "TSP50", "TSP100"]

# EoH paper Table 10 values (gap % to Concorde optimal)
EOH_TABLE = {
    "TSP20":  {"Concorde": 0.000, "LKH3": 0.000, "NN": 17.448, "FI": 2.242,
               "AM": 0.069, "GLS": 0.004, "EBGLS": 0.002, "KGLS": 0.000, "EoH": 0.000},
    "TSP50":  {"Concorde": 0.000, "LKH3": 0.000, "NN": 23.230, "FI": 7.263,
               "AM": 0.494, "GLS": 0.045, "EBGLS": 0.003, "KGLS": 0.000, "EoH": 0.000},
    "TSP100": {"Concorde": 0.000, "LKH3": 0.011, "NN": 25.104, "FI": 12.456,
               "AM": 2.368, "GLS": 0.659, "EBGLS": 0.155, "KGLS": 0.035, "EoH": 0.025},
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_test_data(test_data_dir):
    """Load TSP test instances from pickle files."""
    datasets = {}
    for name in TSP_SIZES:
        pkl_path = os.path.join(test_data_dir, f"{name}.pkl")
        if not os.path.exists(pkl_path):
            print(f"Warning: {pkl_path} not found, skipping {name}")
            continue
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        datasets[name] = {
            "coords": data["coordinate"],
            "instances": data["distance_matrix"],
            "opt_costs": data["cost"],
        }
        print(f"  {name}: {len(data['distance_matrix'])} instances, "
              f"size {data['distance_matrix'][0].shape[0]}")
    return datasets


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_on_test(datasets, heuristic_module, max_instances):
    """Evaluate GLS heuristic on all test datasets, return gap % dict."""
    results = {}

    for name in TSP_SIZES:
        if name not in datasets:
            continue

        data = datasets[name]
        coords = data["coords"]
        instances = data["instances"]
        opt_costs = data["opt_costs"]

        n_inst = len(instances)
        if max_instances > 0:
            n_inst = min(n_inst, max_instances)

        gaps = []
        times = []
        t_total = time.time()

        for i in range(n_inst):
            t_start = time.time()
            gap = solve_instance(
                i, opt_costs[i], instances[i], coords[i],
                TIME_LIMIT, ITE_MAX, PERTURBATION_MOVES, heuristic_module,
            )
            elapsed = time.time() - t_start
            gaps.append(gap)
            times.append(elapsed)

            if (i + 1) % 50 == 0 or i == n_inst - 1:
                avg_gap = np.mean(gaps)
                avg_time = np.mean(times)
                print(f"  {name}: {i+1}/{n_inst} instances, "
                      f"avg gap={avg_gap:.3f}%, avg time={avg_time:.3f}s")

        avg_gap = float(np.mean(gaps))
        avg_time = float(np.mean(times))
        results[name] = {"gap": avg_gap, "time": avg_time}

        total_elapsed = time.time() - t_total
        print(f"  {name} done: gap={avg_gap:.3f}%, "
              f"avg_time={avg_time:.3f}s, total={total_elapsed:.1f}s\n")

    return results


# ---------------------------------------------------------------------------
# Table generation
# ---------------------------------------------------------------------------

def print_table(results):
    """Print a markdown table matching Table 10 from the EoH paper."""
    baselines = ["NN", "FI", "AM", "GLS", "EBGLS", "KGLS", "EoH"]

    header = "| Method | " + " | ".join(TSP_SIZES) + " |"
    sep = "|---|" + "|".join(["---"] * len(TSP_SIZES)) + "|"

    print()
    print("### Table 10: TSP Results (Gap % to Concorde Optimal)")
    print("Average gap (%) over test instances (lower is better).")
    print()
    print(header)
    print(sep)

    for method in baselines:
        row = f"| {method} |"
        for name in TSP_SIZES:
            val = EOH_TABLE.get(name, {}).get(method, None)
            if val is not None:
                row += f" {val:.3f} |"
            else:
                row += " - |"
        print(row)

    # CodeEvolve row
    row = "| **CodeEvolve** |"
    for name in TSP_SIZES:
        if name in results:
            row += f" **{results[name]['gap']:.3f}** |"
        else:
            row += " - |"
    print(row)

    # Time row
    print()
    print("#### Average time per instance (seconds)")
    print()
    header2 = "| Method | " + " | ".join(TSP_SIZES) + " |"
    sep2 = "|---|" + "|".join(["---"] * len(TSP_SIZES)) + "|"
    print(header2)
    print(sep2)

    eoh_times = {"TSP20": 0.498, "TSP50": 1.494, "TSP100": 4.510}
    row = "| EoH |"
    for name in TSP_SIZES:
        row += f" {eoh_times.get(name, '-'):.3f} |"
    print(row)

    row = "| **CodeEvolve** |"
    for name in TSP_SIZES:
        if name in results:
            row += f" **{results[name]['time']:.3f}** |"
        else:
            row += " - |"
    print(row)

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Test evaluation for TSP GLS (Table 10)")
    parser.add_argument("heuristic_file",
                        help="Path to the evolved heuristic .py file")
    parser.add_argument("--test-data-dir", default=DEFAULT_TEST_DATA,
                        help="Path to directory containing TSP{20,50,100}.pkl")
    parser.add_argument("--max-instances", type=int, default=0,
                        help="Max instances per size (0 = all, default: all)")
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

    if not hasattr(heuristic_module, "update_edge_distance"):
        print(f"Error: {args.heuristic_file} does not define "
              f"an 'update_edge_distance' function.")
        sys.exit(1)

    # Load test data
    test_data_dir = os.path.abspath(args.test_data_dir)
    print(f"Loading test data from: {test_data_dir}")
    datasets = load_test_data(test_data_dir)

    if not datasets:
        print("Error: No test data found.")
        sys.exit(1)

    n_str = f" (max {args.max_instances} per size)" if args.max_instances > 0 else ""
    print(f"\nEvaluating heuristic: {args.heuristic_file}{n_str}\n")

    results = evaluate_on_test(datasets, heuristic_module, args.max_instances)
    print_table(results)


if __name__ == "__main__":
    main()
