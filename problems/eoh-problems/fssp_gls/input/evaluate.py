# ===--------------------------------------------------------------------------------------===#
#
# Part of the CodeEvolve Project, under the Apache License v2.0.
# See https://github.com/inter-co/science-codeevolve/blob/main/LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0
#
# ===--------------------------------------------------------------------------------------===#
#
# Flow Shop Scheduling with Guided Local Search Evaluator 
#
# This evaluator loads FSSP instances, runs the GLS framework with the evolved
# get_matrix_and_jobs heuristic, and reports fitness metrics in JSON format.
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


# ---------------------------------------------------------------------------
# Configuration (matches EoH paper)
# ---------------------------------------------------------------------------

N_INST_EVA = 3       # Number of instances used during evolution (EoH default)
TIME_MAX = 30        # Seconds per instance (EoH default)
ITER_MAX = 1000      # Max GLS iterations per instance (EoH default)


# ---------------------------------------------------------------------------
# Core scheduling functions (from EoH prob.py, with numba JIT)
# ---------------------------------------------------------------------------

@jit(nopython=True)
def makespan(order, tasks, machines_val):
    """Compute the makespan of a given job sequence."""
    times = []
    for i in range(0, machines_val):
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
    """Full local search: swap + relocate neighborhoods."""
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
    """Perturbation-focused local search: only moves specified jobs."""
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
# NEH heuristic for initial solution
# ---------------------------------------------------------------------------

def sum_and_order(tasks_val, machines_val, tasks):
    """Order jobs by decreasing total processing time (NEH ordering)."""
    tab = []
    tab1 = []
    for i in range(0, tasks_val):
        tab.append(0)
        tab1.append(0)
    for j in range(0, tasks_val):
        for k in range(0, machines_val):
            tab[j] += tasks[j][k]
    place = 0
    iter_count = 0
    while iter_count != tasks_val:
        max_time = 1
        for i in range(0, tasks_val):
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
    """NEH heuristic for FSSP: builds a schedule by inserting jobs."""
    order = sum_and_order(tasks_val, machines_val, tasks)
    current_seq = [order[0]]
    for i in range(1, tasks_val):
        min_cmax = float("inf")
        best_seq = None
        for j in range(0, i + 1):
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
    """Run GLS on a single FSSP instance with the evolved heuristic."""
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
# Instance loading
# ---------------------------------------------------------------------------

def read_instances():
    """Load all 64 FSSP training instances from text files."""
    tasks_val_list = []
    machines_val_list = []
    tasks_list = []

    for i in range(1, 65):
        filepath = os.path.join(SCRIPT_DIR, "TrainingData", f"{i}.txt")
        with open(filepath, "r") as f:
            tasks_val, machines_val = f.readline().split()
            tasks_val = int(tasks_val)
            machines_val = int(machines_val)

            tasks = np.zeros((tasks_val, machines_val))
            for row in range(tasks_val):
                tmp = f.readline().split()
                for col in range(machines_val):
                    tasks[row][col] = int(float(tmp[col * 2 + 1]))

        tasks_val_list.append(tasks_val)
        machines_val_list.append(machines_val)
        tasks_list.append(tasks)

    return tasks_val_list, machines_val_list, tasks_list


# ---------------------------------------------------------------------------
# Fitness computation
# ---------------------------------------------------------------------------

def compute_fitness(tasks_val_list, machines_val_list, tasks_list, heuristic_module):
    """Run the GLS heuristic on N_INST_EVA instances and return aggregate metrics.

    Fitness is defined as the *negative* average makespan,
    so that **higher is better** (less negative means shorter makespan).
    """
    cmax_list = np.zeros(N_INST_EVA)

    for idx in range(N_INST_EVA):
        cmax_list[idx] = gls(
            heuristic_module,
            tasks_val_list[idx],
            tasks_list[idx],
            machines_val_list[idx],
        )

    avg_makespan = float(np.mean(cmax_list))

    return {
        "fitness": float(-avg_makespan),
        "avg_makespan": float(avg_makespan),
        "makespans": [float(c) for c in cmax_list],
    }


# ---------------------------------------------------------------------------
# Main evaluator entry point
# ---------------------------------------------------------------------------

def evaluate(program_path: str, results_path: str) -> None:
    """CodeEvolve evaluator interface.

    Args:
        program_path: Path to the evolved Python file containing
            ``get_matrix_and_jobs(current_sequence, time_matrix, m, n)``.
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

    if not hasattr(program, "get_matrix_and_jobs"):
        raise AttributeError(
            f"Evolved program at {program_path} does not define "
            f"a 'get_matrix_and_jobs' function."
        )

    tasks_val_list, machines_val_list, tasks_list = read_instances()

    start_time = time.time()
    metrics = compute_fitness(
        tasks_val_list, machines_val_list, tasks_list, program
    )
    eval_time = time.time() - start_time
    metrics["eval_time"] = float(eval_time)

    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    program_path = sys.argv[1]
    results_path = sys.argv[2]
    evaluate(program_path, results_path)
