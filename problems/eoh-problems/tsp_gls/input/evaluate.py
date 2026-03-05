# ===--------------------------------------------------------------------------------------===#
#
# Part of the CodeEvolve Project, under the Apache License v2.0.
# See https://github.com/inter-co/science-codeevolve/blob/main/LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0
#
# ===--------------------------------------------------------------------------------------===#
#
# TSP with Guided Local Search Evaluator
#
# This evaluator loads TSP100 instances, runs the GLS framework with the evolved
# update_edge_distance heuristic, and reports fitness metrics in JSON format.
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
import types
import warnings
import numpy as np

# Ensure the input directory is on the path so gls/ and utils/ are importable.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from utils.readTSPRandom import read_instance_all
from gls.gls_run import solve_instance


# ---------------------------------------------------------------------------
# Configuration (matches EoH paper)
# ---------------------------------------------------------------------------

N_INST_EVA = 3        # Number of instances used during evolution (EoH default)
TIME_LIMIT = 10       # Seconds per instance (EoH default)
ITE_MAX = 1000        # Max GLS iterations per instance (EoH default)
PERTURBATION_MOVES = 1  # Perturbation moves per edge (EoH default)


# ---------------------------------------------------------------------------
# Instance data
# ---------------------------------------------------------------------------

def load_instances():
    """Load TSP100 instances from the TSPAEL64.pkl pickle file."""
    instance_path = os.path.join(SCRIPT_DIR, "TrainingData", "TSPAEL64.pkl")
    coords, instances, opt_costs = read_instance_all(instance_path)
    return coords, instances, opt_costs


# ---------------------------------------------------------------------------
# Fitness computation
# ---------------------------------------------------------------------------

def compute_fitness(coords, instances, opt_costs, heuristic_module):
    """Run the GLS heuristic on N_INST_EVA instances and return aggregate metrics.

    Fitness is defined as the *negative* average gap (%) to optimal,
    so that **higher is better** (0.0 means optimal tours found).
    """
    gaps = np.zeros(N_INST_EVA)

    for i in range(N_INST_EVA):
        gap = solve_instance(
            i,
            opt_costs[i],
            instances[i],
            coords[i],
            TIME_LIMIT,
            ITE_MAX,
            PERTURBATION_MOVES,
            heuristic_module,
        )
        gaps[i] = gap

    avg_gap = float(np.mean(gaps))

    return {
        "fitness": float(-avg_gap),
        "avg_gap_pct": float(avg_gap),
        "gaps": [float(g) for g in gaps],
    }


# ---------------------------------------------------------------------------
# Main evaluator entry point
# ---------------------------------------------------------------------------

def evaluate(program_path: str, results_path: str) -> None:
    """CodeEvolve evaluator interface.

    Args:
        program_path: Path to the evolved Python file containing
            ``update_edge_distance(edge_distance, local_opt_tour, edge_n_used)``.
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

    if not hasattr(program, "update_edge_distance"):
        raise AttributeError(
            f"Evolved program at {program_path} does not define "
            f"an 'update_edge_distance' function."
        )

    coords, instances, opt_costs = load_instances()

    start_time = time.time()
    metrics = compute_fitness(coords, instances, opt_costs, program)
    eval_time = time.time() - start_time
    metrics["eval_time"] = float(eval_time)

    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    program_path = sys.argv[1]
    results_path = sys.argv[2]
    evaluate(program_path, results_path)
