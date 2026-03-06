#!/usr/bin/env python3
# ===--------------------------------------------------------------------------------------===#
#
# Part of the CodeEvolve Project, under the Apache License v2.0.
# See https://github.com/inter-co/science-codeevolve/blob/main/LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0
#
# ===--------------------------------------------------------------------------------------===#
#
# Test evaluation for Flow Shop Scheduling with GLS -- generates Table 12 from the EoH paper.
#
# Usage:
#     python3 test_evaluate.py <heuristic_file> [--test-data-dir <path>]
#
# The heuristic file must define a
# `get_matrix_and_jobs(current_sequence, time_matrix, m, n)` function.
# Test data should contain the Taillard benchmark files from the EoH repo.
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
import time
import random
import warnings
import numpy as np

from numba import jit
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning

warnings.simplefilter("ignore", category=NumbaDeprecationWarning)
warnings.simplefilter("ignore", category=NumbaPendingDeprecationWarning)
warnings.filterwarnings(
    "ignore", message="loaded more than 1 DLL from .libs", category=UserWarning
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_TEST_DATA = os.path.join(
    SCRIPT_DIR, "..", "..", "..", "..", "EoH",
    "examples", "user_fssp_gls", "TestingData", "Taillard",
)

# GLS parameters (same as EoH paper)
TIME_MAX = 60   # 60s per instance for test (paper: "maximum running time 60 seconds")
ITER_MAX = 1000

# Taillard test sets in paper order
TAILLARD_SETS = [
    ("20", "5"),  ("20", "10"), ("20", "20"),
    ("50", "5"),  ("50", "10"), ("50", "20"),
    ("100", "5"), ("100", "10"), ("100", "20"),
    ("200", "10"), ("200", "20"),
]

# EoH paper Table 12 values (gap % to Taillard upper bound)
EOH_TABLE_12 = {
    ("20", "5"):   {"GUPTA": 12.89, "CDS": 9.03,  "NEH": 3.24, "NEHFF": 2.30, "LS": 1.91, "ILS1": 0.42, "ILS2": 0.18, "EoH": 0.09},
    ("20", "10"):  {"GUPTA": 23.42, "CDS": 12.87, "NEH": 4.05, "NEHFF": 4.15, "LS": 2.77, "ILS1": 0.33, "ILS2": 0.25, "EoH": 0.30},
    ("20", "20"):  {"GUPTA": 21.79, "CDS": 10.35, "NEH": 3.06, "NEHFF": 2.72, "LS": 2.60, "ILS1": 0.29, "ILS2": 0.25, "EoH": 0.10},
    ("50", "5"):   {"GUPTA": 12.23, "CDS": 6.98,  "NEH": 0.57, "NEHFF": 0.40, "LS": 0.32, "ILS1": 0.15, "ILS2": 0.32, "EoH": 0.02},
    ("50", "10"):  {"GUPTA": 20.11, "CDS": 12.72, "NEH": 3.47, "NEHFF": 3.62, "LS": 3.33, "ILS1": 1.47, "ILS2": 0.29, "EoH": 0.19},
    ("50", "20"):  {"GUPTA": 22.78, "CDS": 15.03, "NEH": 5.48, "NEHFF": 5.10, "LS": 4.67, "ILS1": 2.13, "ILS2": 0.34, "EoH": 0.60},
    ("100", "5"):  {"GUPTA": 5.98,  "CDS": 5.10,  "NEH": 0.39, "NEHFF": 0.31, "LS": 0.28, "ILS1": 0.20, "ILS2": 0.38, "EoH": -0.04},
    ("100", "10"): {"GUPTA": 15.03, "CDS": 9.36,  "NEH": 2.07, "NEHFF": 1.88, "LS": 1.38, "ILS1": 0.77, "ILS2": 0.34, "EoH": 0.14},
    ("100", "20"): {"GUPTA": 21.00, "CDS": 13.55, "NEH": 3.58, "NEHFF": 3.73, "LS": 3.51, "ILS1": 2.27, "ILS2": 0.43, "EoH": 0.41},
    ("200", "10"): {"GUPTA": 11.59, "CDS": 7.22,  "NEH": 0.98, "NEHFF": 0.70, "LS": 0.87, "ILS1": 0.74, "ILS2": 0.54, "EoH": 0.12},
    ("200", "20"): {"GUPTA": 18.09, "CDS": 11.89, "NEH": 2.90, "NEHFF": 2.52, "LS": 2.53, "ILS1": 2.26, "ILS2": 0.59, "EoH": 0.61},
}


# ---------------------------------------------------------------------------
# Core scheduling functions (from EoH, with numba JIT)
# ---------------------------------------------------------------------------

@jit(nopython=True)
def makespan(order, tasks, machines_val):
    times = []
    for i in range(machines_val):
        times.append(0)
    for j in order:
        times[0] += tasks[j][0]
        for k in range(1, machines_val):
            if times[k] < times[k - 1]:
                times[k] = times[k - 1]
            times[k] += tasks[j][k]
    return max(times)


@jit(nopython=True)
def local_search(sequence, cmax_old, tasks, machines_val):
    new_seq = sequence[:]
    for i in range(len(new_seq)):
        for j in range(i + 1, len(new_seq)):
            temp_seq = new_seq[:]
            temp_seq[i], temp_seq[j] = temp_seq[j], temp_seq[i]
            cmax = makespan(temp_seq, tasks, machines_val)
            if cmax < cmax_old:
                new_seq = temp_seq[:]
                cmax_old = cmax
    for i in range(1, len(new_seq)):
        for j in range(1, len(new_seq)):
            temp_seq = new_seq[:]
            temp_seq.remove(i)
            temp_seq.insert(j, i)
            cmax = makespan(temp_seq, tasks, machines_val)
            if cmax < cmax_old:
                new_seq = temp_seq[:]
                cmax_old = cmax
    return new_seq


@jit(nopython=True)
def local_search_perturb(sequence, cmax_old, tasks, machines_val, job):
    new_seq = sequence[:]
    for i in job:
        for j in range(i + 1, len(new_seq)):
            temp_seq = new_seq[:]
            temp_seq[i], temp_seq[j] = temp_seq[j], temp_seq[i]
            cmax = makespan(temp_seq, tasks, machines_val)
            if cmax < cmax_old:
                new_seq = temp_seq[:]
                cmax_old = cmax
    for i in job:
        for j in range(1, len(new_seq)):
            temp_seq = new_seq[:]
            temp_seq.remove(i)
            temp_seq.insert(j, i)
            cmax = makespan(temp_seq, tasks, machines_val)
            if cmax < cmax_old:
                new_seq = temp_seq[:]
                cmax_old = cmax
    return new_seq


# ---------------------------------------------------------------------------
# NEH heuristic
# ---------------------------------------------------------------------------

def sum_and_order(tasks_val, machines_val, tasks):
    tab = [0] * tasks_val
    tab1 = [0] * tasks_val
    for j in range(tasks_val):
        for k in range(machines_val):
            tab[j] += tasks[j][k]
    place = 0
    iter_count = 0
    while iter_count != tasks_val:
        max_time = 1
        for i in range(tasks_val):
            if max_time < tab[i]:
                max_time = tab[i]
                place = i
        tab[place] = 1
        tab1[iter_count] = place
        iter_count += 1
    return tab1


def insert_neh(sequence, position, value):
    new_seq = sequence[:]
    new_seq.insert(position, value)
    return new_seq


def neh(tasks, machines_val, tasks_val):
    order = sum_and_order(tasks_val, machines_val, tasks)
    current_seq = [order[0]]
    for i in range(1, tasks_val):
        min_cmax = float("inf")
        best_seq = None
        for j in range(i + 1):
            tmp = insert_neh(current_seq, j, order[i])
            cmax_tmp = makespan(tmp, tasks, machines_val)
            if min_cmax > cmax_tmp:
                best_seq = tmp
                min_cmax = cmax_tmp
        current_seq = best_seq
    return current_seq, makespan(current_seq, tasks, machines_val)


# ---------------------------------------------------------------------------
# Guided Local Search
# ---------------------------------------------------------------------------

def gls(heuristic_module, tasks_val, tasks, machines_val):
    cmax_best = 1e10
    random.seed(2024)
    try:
        pi, cmax = neh(tasks, machines_val, tasks_val)
        n = len(pi)
        pi_best = pi
        cmax_best = cmax
        n_itr = 0
        time_start = time.time()

        while time.time() - time_start < TIME_MAX and n_itr < ITER_MAX:
            piprim = local_search(pi, cmax, tasks, machines_val)
            pi = piprim
            cmax = makespan(pi, tasks, machines_val)
            if cmax < cmax_best:
                pi_best = pi
                cmax_best = cmax

            tasks_perturb, jobs = heuristic_module.get_matrix_and_jobs(
                pi, tasks.copy(), machines_val, n
            )
            if len(jobs) <= 1:
                return 1e10
            if len(jobs) > 5:
                jobs = jobs[:5]

            cmax = makespan(pi, tasks_perturb, machines_val)
            pi = local_search_perturb(pi, cmax, tasks_perturb, machines_val, jobs)
            n_itr += 1
            if n_itr % 50 == 0:
                pi = pi_best
                cmax = cmax_best
    except Exception:
        cmax_best = 1e10
    return cmax_best


# ---------------------------------------------------------------------------
# Taillard instance parsing
# ---------------------------------------------------------------------------

def read_taillard_file(filepath):
    """Parse a Taillard benchmark file. Returns list of (tasks_val, machines_val, tasks, upper_bound)."""
    instances = []
    with open(filepath, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # Look for header line
        if line.startswith("number of jobs"):
            i += 1
            parts = lines[i].split()
            tasks_val = int(parts[0])
            machines_val = int(parts[1])
            upper_bound = int(parts[3])
            i += 1  # skip "processing times :" line
            i += 1

            # Read processing times: machines_val rows of tasks_val values
            time_matrix = np.zeros((tasks_val, machines_val))
            for m_idx in range(machines_val):
                row_vals = list(map(int, lines[i].split()))
                for j_idx in range(tasks_val):
                    time_matrix[j_idx][m_idx] = row_vals[j_idx]
                i += 1

            instances.append((tasks_val, machines_val, time_matrix, upper_bound))
        else:
            i += 1

    return instances


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_test_data(test_data_dir):
    """Load all Taillard test sets."""
    datasets = {}
    for n_jobs, n_machines in TAILLARD_SETS:
        filename = f"t_j{n_jobs}_m{n_machines}.txt"
        filepath = os.path.join(test_data_dir, filename)
        if not os.path.exists(filepath):
            print(f"  Warning: {filepath} not found, skipping {n_jobs}x{n_machines}")
            continue
        instances = read_taillard_file(filepath)
        key = (n_jobs, n_machines)
        datasets[key] = instances
        print(f"  {n_jobs}x{n_machines}: {len(instances)} instances")
    return datasets


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_on_test(datasets, heuristic_module):
    """Evaluate GLS heuristic on all Taillard test sets."""
    results = {}

    for n_jobs, n_machines in TAILLARD_SETS:
        key = (n_jobs, n_machines)
        if key not in datasets:
            continue

        instances = datasets[key]
        gaps = []
        t_total = time.time()

        for idx, (tasks_val, machines_val, tasks, ub) in enumerate(instances):
            cmax = gls(heuristic_module, tasks_val, tasks, machines_val)
            gap = (cmax - ub) / ub * 100
            gaps.append(gap)
            print(f"  {n_jobs}x{n_machines} inst {idx+1}/{len(instances)}: "
                  f"makespan={cmax:.0f}, UB={ub}, gap={gap:.2f}%")

        avg_gap = float(np.mean(gaps))
        elapsed = time.time() - t_total
        results[key] = avg_gap
        print(f"  {n_jobs}x{n_machines} done: avg gap={avg_gap:.2f}%, "
              f"time={elapsed:.1f}s\n")

    return results


# ---------------------------------------------------------------------------
# Table generation
# ---------------------------------------------------------------------------

def print_table(results):
    """Print a markdown table matching Table 12 from the EoH paper."""
    baselines = ["NEH", "NEHFF", "LS", "ILS1", "ILS2", "EoH"]
    col_headers = [f"n{n}m{m}" for n, m in TAILLARD_SETS]

    header = "| Method | " + " | ".join(col_headers) + " | Avg |"
    sep = "|---|" + "|".join(["---"] * len(TAILLARD_SETS)) + "|---|"

    print()
    print("### Table 12: Flow Shop Scheduling Results (Taillard Benchmarks)")
    print("Average gap (%) to Taillard upper bound (lower is better). 10 instances per set.")
    print()
    print(header)
    print(sep)

    for method in baselines:
        row = f"| {method} |"
        vals = []
        for key in TAILLARD_SETS:
            val = EOH_TABLE_12.get(key, {}).get(method, None)
            if val is not None:
                row += f" {val:.2f} |"
                vals.append(val)
            else:
                row += " - |"
        avg = np.mean(vals) if vals else float("nan")
        row += f" {avg:.2f} |"
        print(row)

    # CodeEvolve row
    row = "| **CodeEvolve** |"
    vals = []
    for key in TAILLARD_SETS:
        if key in results:
            row += f" **{results[key]:.2f}** |"
            vals.append(results[key])
        else:
            row += " - |"
    avg = np.mean(vals) if vals else float("nan")
    row += f" **{avg:.2f}** |"
    print(row)

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Test evaluation for FSSP GLS (Table 12)")
    parser.add_argument("heuristic_file",
                        help="Path to the evolved heuristic .py file")
    parser.add_argument("--test-data-dir", default=DEFAULT_TEST_DATA,
                        help="Path to directory containing Taillard t_j*_m*.txt files")
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

    if not hasattr(heuristic_module, "get_matrix_and_jobs"):
        print(f"Error: {args.heuristic_file} does not define "
              f"a 'get_matrix_and_jobs' function.")
        sys.exit(1)

    # Load test data
    test_data_dir = os.path.abspath(args.test_data_dir)
    print(f"Loading test data from: {test_data_dir}")
    datasets = load_test_data(test_data_dir)

    if not datasets:
        print("Error: No test data found.")
        sys.exit(1)

    # Warmup numba JIT (first call compiles)
    print("\nWarming up numba JIT...")
    dummy_tasks = np.array([[10.0, 20.0], [30.0, 40.0]])
    _ = makespan([0, 1], dummy_tasks, 2)
    _ = local_search([0, 1], 999999, dummy_tasks, 2)
    print("Done.\n")

    print(f"Evaluating heuristic: {args.heuristic_file}\n")
    results = evaluate_on_test(datasets, heuristic_module)
    print_table(results)


if __name__ == "__main__":
    main()
