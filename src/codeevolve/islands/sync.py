# ===--------------------------------------------------------------------------------------===#
#
# Part of the CodeEvolve Project, under the Apache License v2.0.
# See https://github.com/inter-co/science-codeevolve/blob/main/LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0
#
# ===--------------------------------------------------------------------------------------===#
#
# This file implements the synchronization structures for coordinating the distributed
# islands algorithm.
#
# ===--------------------------------------------------------------------------------------===#

from typing import Dict
from dataclasses import dataclass
import multiprocessing as mp
import multiprocessing.sharedctypes as mpsct
import multiprocessing.synchronize as mps

from codeevolve.database import Program


# ---------------------------------------------------------------------------
# Global best program tracking
# ---------------------------------------------------------------------------


@dataclass
class GlobalBestProg:
    """Tracks the globally best program across all islands using shared memory.

    This class maintains synchronized access to information about the best
    program found across all islands in a distributed evolutionary system.

    Attributes:
        fitness: Synchronized fitness value of the best program.
        iteration_found: Synchronized iteration number when best program was found.
        island_found: Synchronized ID of the island that found the best program.
        depth: Synchronized depth of the best program in the evolutionary tree.
        eval_metrics: Shared dictionary of evaluation metric names to values.
    """

    fitness: mpsct.Synchronized
    iteration_found: mpsct.Synchronized
    island_found: mpsct.Synchronized
    depth: mpsct.Synchronized
    eval_metrics: Dict[str, float]

    def __repr__(self) -> str:
        """Returns a string representation of the global best program.

        Returns:
            A formatted string showing fitness, iteration found, island found,
            depth, and evaluation metrics (each on a separate line).
        """
        lines = [
            f"{self.__class__.__name__}(",
            f"  fitness={self.fitness.value:.8f},",
            f"  iteration_found={self.iteration_found.value},",
            f"  island_found={self.island_found.value},",
            f"  depth={self.depth.value},",
        ]

        if self.eval_metrics:
            lines.append("  eval_metrics={")
            for key, value in self.eval_metrics.items():
                lines.append(f"    {key}: {value},")
            lines.append("  }")
        else:
            lines.append("  eval_metrics={}")

        lines.append(")")
        return "\n".join(lines)

    def update_from_program(self, prog: Program) -> None:
        """Updates the global best from a Program instance.

        Args:
            prog: The program to update from. Must have fitness, iteration_found,
                  island_found, depth, and eval_metrics attributes.
        """
        self.fitness.value = prog.fitness
        self.iteration_found.value = prog.iteration_found
        self.island_found.value = prog.island_found
        self.depth.value = prog.depth
        self.eval_metrics.clear()
        self.eval_metrics.update(prog.eval_metrics)


# ---------------------------------------------------------------------------
# Global synchronization data
# ---------------------------------------------------------------------------


@dataclass
class GlobalSyncData:
    """Contains synchronization data structures for coordinating distributed islands.

    This class encapsulates all shared memory objects and synchronization
    primitives needed for coordinating multiple islands in a distributed
    evolutionary algorithm.

    Attributes:
        best_sol: Information about the globally best program found.
        early_stop_counter: Counter for consecutive iterations without improvement.
        early_stop_aux: Auxiliary counter for early stopping coordination.
        lock: Mutex for protecting shared data access.
        barrier: Synchronization barrier for coordinating island phases.
        log_queue: Queue for collecting log messages from all islands.
        start_time: Synchronized start time of the algorithm (Unix timestamp).
        elapsed_time_offset: Synchronized offset for elapsed time when resuming from checkpoint.
        cpu_count: Synchronized count of CPUs available to the process.
    """

    best_sol: GlobalBestProg
    early_stop_counter: mpsct.Synchronized
    early_stop_aux: mpsct.Synchronized
    lock: mps.Lock
    barrier: mps.Barrier
    log_queue: mp.Queue
    start_time: mpsct.Synchronized
    elapsed_time_offset: mpsct.Synchronized
    cpu_count: mpsct.Synchronized
