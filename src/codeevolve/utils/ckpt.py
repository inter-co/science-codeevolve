# ===--------------------------------------------------------------------------------------===#
#
# Part of the CodeEvolve Project, under the Apache License v2.0.
# See https://github.com/inter-co/science-codeevolve/blob/main/LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0
#
# ===--------------------------------------------------------------------------------------===#
#
# This file implements checkpointing routines.
#
# ===--------------------------------------------------------------------------------------===#

import json
import logging
import pickle as pkl
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from codeevolve.database import ProgramDatabase
from codeevolve.islands.sync import GlobalBestProg
from codeevolve.scheduler import Scheduler
from codeevolve.utils.constants import CHECKPOINT_FILE_FORMAT, RUN_METADATA_FILE


def save_ckpt(
    curr_epoch: int,
    prompt_db: ProgramDatabase,
    sol_db: ProgramDatabase,
    evolve_state: Dict[str, Any],
    exploration_scheduler: Optional[Scheduler],
    best_sol_path: str | Path,
    best_prompt_path: str | Path,
    ckpt_dir: str | Path,
    logger: Optional[logging.Logger] = None,
    timeout_scheduler: Optional[Scheduler] = None,
) -> None:
    """Saves a checkpoint of the evolutionary algorithm state.

    This function creates a checkpoint by serializing the current state of the
    evolutionary algorithm, including program databases and algorithm state.
    It also saves the best solution and prompt as separate text files.

    Args:
        curr_epoch: Current epoch number for checkpoint naming.
        prompt_db: Database containing prompt population.
        sol_db: Database containing solution population.
        evolve_state: Dictionary containing the current state of the evolution algorithm.
        exploration_scheduler: Exploration rate scheduler.
        best_sol_path: File path where the best solution code will be saved.
        best_prompt_path: File path where the best prompt code will be saved.
        ckpt_dir: Directory where the checkpoint file will be saved.
        logger: Logger instance for logging checkpoint operations.
        timeout_scheduler: Timeout scheduler (optional).
    """

    data: Dict[str, Any] = {
        "prompt_db": prompt_db,
        "sol_db": sol_db,
        "evolve_state": evolve_state,
    }
    if exploration_scheduler is not None:
        data["exploration_scheduler"] = exploration_scheduler
    if timeout_scheduler is not None:
        data["timeout_scheduler"] = timeout_scheduler
    if isinstance(best_sol_path, str):
        best_sol_path = Path(best_sol_path)
    if isinstance(best_prompt_path, str):
        best_prompt_path = Path(best_prompt_path)
    if isinstance(ckpt_dir, str):
        ckpt_dir = Path(ckpt_dir)

    with open(ckpt_dir.joinpath(CHECKPOINT_FILE_FORMAT.format(epoch=curr_epoch)), "wb") as f:
        pkl.dump(data, f, protocol=pkl.HIGHEST_PROTOCOL)

    with open(best_sol_path, "w") as f:
        f.write(sol_db.programs[sol_db.best_prog_id].code)

    with open(best_prompt_path, "w") as f:
        f.write(prompt_db.programs[prompt_db.best_prog_id].code)

    logger.info(f"Saved best solution at '{best_sol_path}'.")
    logger.info(f"Saved best prompt at '{best_prompt_path}'.")
    logger.info(f"Checkpoint {curr_epoch} successfully saved.")


def load_ckpt(epoch: int, ckpt_dir: str | Path) -> Tuple[
    Optional[ProgramDatabase],
    Optional[ProgramDatabase],
    Optional[Dict[str, Any]],
    Optional[Scheduler],
    Optional[Scheduler],
]:
    """Loads a checkpoint of the evolutionary algorithm state.

    This function restores the state of the evolutionary algorithm from a
    previously saved checkpoint file, including program databases and algorithm state.

    Args:
        epoch: Epoch number of the checkpoint to load.
        ckpt_dir: Directory containing the checkpoint files.

    Returns:
        A tuple containing:
            - Prompt database with evolved prompts, None if not found
            - Solution database with evolved programs, None if not found
            - Dictionary with the evolution algorithm state, None if not found
            - Exploration rate scheduler, None if not found
            - Timeout scheduler, None if not found
    """
    if isinstance(ckpt_dir, str):
        ckpt_dir = Path(ckpt_dir)

    with open(ckpt_dir.joinpath(CHECKPOINT_FILE_FORMAT.format(epoch=epoch)), "rb") as f:
        data: Dict[str, Any] = pkl.load(f)

    return (
        data.get("prompt_db", None),
        data.get("sol_db", None),
        data.get("evolve_state", None),
        data.get("exploration_scheduler", None),
        data.get("timeout_scheduler", None),
    )


def save_run_metadata(
    out_dir: str | Path,
    epoch: int,
    elapsed_time: float,
    cpu_count: int,
    global_best_sol: GlobalBestProg,
    early_stop_counter: int,
) -> None:
    """Saves run metadata to a JSON file.

    This function saves metadata at each checkpoint epoch to enable
    accurate tracking when resuming from checkpoints.

    Args:
        out_dir: Output directory where the metadata file will be saved.
        epoch: Current epoch number.
        elapsed_time: Total elapsed time in seconds up to this checkpoint.
        cpu_count: Number of CPUs available to the process.
        global_best_prog: best solution found globally
        early_stop_counter: global early stopping counter
    """
    if isinstance(out_dir, str):
        out_dir = Path(out_dir)

    metadata_file: Path = out_dir.joinpath(RUN_METADATA_FILE)

    data: Dict[str, Any] = {}
    if metadata_file.exists():
        with open(metadata_file, "r") as f:
            data = json.load(f)

    data[str(epoch)] = {
        "elapsed_time": elapsed_time,
        "cpu_count": cpu_count,
        "best_sol": {
            "fitness": global_best_sol.fitness.value,
            "iteration_found": global_best_sol.iteration_found.value,
            "island_found": global_best_sol.island_found.value,
            "depth": global_best_sol.depth.value,
            "eval_metrics": global_best_sol.eval_metrics.copy(),
        },
        "early_stop_counter": early_stop_counter,
    }

    with open(metadata_file, "w") as f:
        json.dump(data, f, indent=2)


def load_run_metadata(out_dir: str | Path, epoch: int) -> Optional[Dict[str, Any]]:
    """Loads run metadata from a JSON file for a specific checkpoint epoch.

    Args:
        out_dir: Output directory containing the metadata file.
        epoch: Epoch number to load metadata for.

    Returns:
        Dictionary with run metadata. None if epoch data not found.
    """
    if isinstance(out_dir, str):
        out_dir = Path(out_dir)

    metadata_file: Path = out_dir.joinpath(RUN_METADATA_FILE)

    if not metadata_file.exists():
        return None

    with open(metadata_file, "r") as f:
        data: Dict[str, Any] = json.load(f)

    epoch_data = data.get(str(epoch), None)
    return epoch_data
