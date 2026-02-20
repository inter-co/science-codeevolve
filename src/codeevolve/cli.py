# ===--------------------------------------------------------------------------------------===#
#
# Part of the CodeEvolve Project, under the Apache License v2.0.
# See https://github.com/inter-co/science-codeevolve/blob/main/LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0
#
# ===--------------------------------------------------------------------------------------===#
#
# This file implements the command-line interface entry point for CodeEvolve.
#
# ===--------------------------------------------------------------------------------------===#

import argparse
import ctypes
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from codeevolve.islands.graph import PipeEdge, setup_island_topology
from codeevolve.islands.sync import GlobalBestProg, GlobalSyncData
from codeevolve.runner import (
    cleanup_log_daemon,
    get_cleanup_state,
    monitor_island_processes,
    setup_signal_handlers,
    spawn_island_processes,
    start_log_daemon,
)
from codeevolve.utils.ckpt import load_run_metadata
from codeevolve.utils.cli_setup import (
    create_config_copy,
    display_run_data,
    load_config,
    setup_island_args,
    validate_environment,
    validate_paths,
)
from codeevolve.utils.lock import DirectoryLock, check_directory_lock


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for CodeEvolve execution.

    Returns:
        Parsed command-line arguments containing input directory, config path,
        output directory, checkpoint settings, and logging preferences.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="CodeEvolve")
    parser.add_argument(
        "--inpt_dir",
        type=str,
        required=True,
        help="Path to input directory containing initial solution and evaluation file",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Path to directory that will contain the outputs of CodeEvolve",
    )
    parser.add_argument(
        "--cfg_path",
        type=str,
        help="Path to .yaml config file (required when starting new run)",
    )
    parser.add_argument(
        "--load_ckpt",
        type=int,
        default=0,
        help="Checkpoint to load: 0 for new run, -1 for latest, or specific epoch number",
    )
    parser.add_argument(
        "--terminal_logging",
        action="store_true",
        help="Enable dynamic log display from all islands in terminal",
    )
    parser.add_argument(
        "--y",
        action="store_true",
        help="Skips all user inputs in cli",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> int:
    """Main entry point for CodeEvolve.

    Orchestrates the complete execution flow:
    1. Parses arguments and validates environment
    2. Checks for directory lock (prevents concurrent runs)
    3. Loads configuration and sets up output directories
    4. Creates shared memory and synchronization primitives
    5. Configures island communication topology
    6. Spawns island processes for distributed evolution
    7. Monitors island processes and handles failures
    8. Coordinates shutdown

    Returns:
        Exit code (0 for success, 1 for failure, 128+signum for signal).
    """
    warnings: List[str] = []

    cleanup_state: Dict[str, Any] = get_cleanup_state()

    setup_signal_handlers()

    args_ns: argparse.Namespace = parse_args()
    args: Dict[str, Any] = vars(args_ns)

    args["inpt_dir"] = Path(args["inpt_dir"])
    args["cfg_path"] = Path(args["cfg_path"]) if args["cfg_path"] else None
    args["out_dir"] = Path(args["out_dir"])

    api_base: str
    api_key: str
    api_base, api_key = validate_environment()
    args["api_base"] = api_base
    args["api_key"] = api_key

    loading_checkpoint: bool = args["load_ckpt"] != 0
    validate_paths(args["inpt_dir"], args["cfg_path"], loading_checkpoint)

    if not args["out_dir"].exists():
        if loading_checkpoint:
            loading_checkpoint = False
            warnings.append(f"Warning: directory {args["out_dir"]} not found. Starting anew.")
        os.makedirs(args["out_dir"])

    config: Dict[str, Any]
    cfg_copy_path: Path
    if loading_checkpoint:
        config, cfg_copy_path = load_config(args)
    else:
        config, cfg_copy_path = create_config_copy(args)

    evolve_config: Dict[str, Any] = config["EVOLVE_CONFIG"]

    isl2args: Dict[int, Dict[str, Any]] = setup_island_args(
        args, evolve_config["num_islands"], cfg_copy_path
    )

    global_best_sol: GlobalBestProg = GlobalBestProg()
    elapsed_time_offset: float = 0.0
    cpu_count: int = len(os.sched_getaffinity(0))
    global_ckpt: int = 0
    metadata: Optional[Dict[str, Any]] = None

    if loading_checkpoint:
        global_ckpt: int = isl2args[0]["load_ckpt"]
        metadata = load_run_metadata(args["out_dir"], global_ckpt)
        if metadata is not None:
            global_best_sol.from_dict(metadata["best_sol"])
            elapsed_time_offset = metadata["elapsed_time"]
            ckpt_cpu_count: int = metadata["cpu_count"]
            if ckpt_cpu_count > 0 and ckpt_cpu_count != cpu_count:
                warnings.append(
                    f"Warning: CPU count changed from {ckpt_cpu_count} (ckpt) to {cpu_count} (current)."
                )

    if args["load_ckpt"] >= 0 and global_ckpt != args["load_ckpt"]:
        warnings.append(f"Warning: ckpt {args["load_ckpt"]}, global ckpt set to {global_ckpt}.")

    global_data: GlobalSyncData = GlobalSyncData(
        best_sol=global_best_sol,
        early_stop_counter=mp.Value(
            ctypes.c_int, 0, lock=False
        ),  # early stop counter should be in metadata, need to change this
        early_stop_aux=mp.Value(ctypes.c_int, 0, lock=False),  # same as above
        lock=mp.Lock(),
        barrier=mp.Barrier(parties=evolve_config["num_islands"]),
        log_queue=mp.Queue(),
        start_time=mp.Value(ctypes.c_double, time.time(), lock=False),
        elapsed_time_offset=mp.Value(ctypes.c_double, elapsed_time_offset, lock=False),
        cpu_count=mp.Value(ctypes.c_int, cpu_count, lock=False),
    )

    in_adj: Optional[List[PipeEdge]]
    out_adj: Optional[List[PipeEdge]]
    in_adj, out_adj = setup_island_topology(
        evolve_config["num_islands"],
        evolve_config["migration_topology"],
    )

    display_run_data(args, config, global_ckpt, metadata, warnings)

    directory_lock: DirectoryLock = DirectoryLock(args["out_dir"])
    check_directory_lock(directory_lock)
    cleanup_state["directory_lock"] = directory_lock

    log_daemon: Optional[mp.Process] = start_log_daemon(
        args, global_data, evolve_config["num_islands"]
    )

    cleanup_state["log_daemon"] = log_daemon
    cleanup_state["log_queue"] = global_data.log_queue

    processes: List[mp.Process] = spawn_island_processes(
        num_islands=evolve_config["num_islands"],
        isl2args=isl2args,
        in_adj=in_adj,
        out_adj=out_adj,
        global_data=global_data,
    )
    cleanup_state["processes"] = processes

    exit_code: int = monitor_island_processes(
        processes=processes,
        global_data=global_data,
        log_daemon=log_daemon,
        out_dir=args["out_dir"],
        poll_interval=1.0,
    )

    cleanup_log_daemon(log_daemon, global_data.log_queue)
    directory_lock.release()

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
