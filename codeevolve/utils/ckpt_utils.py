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

from typing import Any, Dict, Tuple, Optional
import logging
import pickle as pkl
import pathlib

from codeevolve.database import ProgramDatabase


def save_ckpt(
    curr_epoch: int,
    prompt_db: ProgramDatabase,
    sol_db: ProgramDatabase,
    evolve_state: Dict[str, Any],
    best_sol_path: str | pathlib.Path,
    best_prompt_path: str | pathlib.Path,
    ckpt_dir: str | pathlib.Path,
    logger: Optional[logging.Logger] = None,
):
    """Saves a checkpoint of the evolutionary algorithm state.

    This function creates a checkpoint by serializing the current state of the
    evolutionary algorithm, including program databases and algorithm state.
    It also saves the best solution and prompt as separate text files.

    Args:
        curr_epoch: Current epoch number for checkpoint naming.
        prompt_db: Database containing prompt population.
        sol_db: Database containing solution population.
        evolve_state: Dictionary containing the current state of the evolution algorithm.
        best_sol_path: File path where the best solution code will be saved.
        best_prompt_path: File path where the best prompt code will be saved.
        ckpt_dir: Directory where the checkpoint file will be saved.
        logger: Logger instance for logging checkpoint operations.
    """

    data: Dict[str, Any] = {
        "prompt_db": prompt_db,
        "sol_db": sol_db,
        "evolve_state": evolve_state,
    }
    if isinstance(best_sol_path, str):
        best_sol_path = pathlib.Path(best_sol_path)
    if isinstance(best_prompt_path, str):
        best_prompt_path = pathlib.Path(best_prompt_path)
    if isinstance(ckpt_dir, str):
        ckpt_dir = pathlib.Path(ckpt_dir)

    with open(ckpt_dir.joinpath(f"ckpt_{curr_epoch}.pkl"), "wb") as f:
        pkl.dump(data, f, protocol=pkl.HIGHEST_PROTOCOL)

    with open(best_sol_path, "w") as f:
        f.write(sol_db.programs[sol_db.best_prog_id].code)

    with open(best_prompt_path, "w") as f:
        f.write(prompt_db.programs[prompt_db.best_prog_id].code)

    logger.info(f"Saved best solution at '{best_sol_path}'.")
    logger.info(f"Saved best prompt at '{best_prompt_path}'.")
    logger.info(f"Checkpoint {curr_epoch} sucessfully saved.")


def load_ckpt(
    epoch: int, ckpt_dir: str | pathlib.Path
) -> Tuple[ProgramDatabase, ProgramDatabase, Dict[str, Any]]:
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
    """
    if isinstance(ckpt_dir, str):
        ckpt_dir = pathlib.Path(ckpt_dir)

    with open(ckpt_dir.joinpath(f"ckpt_{epoch}.pkl"), "rb") as f:
        data: Dict[str, Any] = pkl.load(f)

    return (
        data.get("prompt_db", None),
        data.get("sol_db", None),
        data.get("evolve_state", None),
    )
