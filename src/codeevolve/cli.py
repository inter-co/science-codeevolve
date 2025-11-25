# ===--------------------------------------------------------------------------------------===#
#
# Part of the CodeEvolve Project, under the Apache License v2.0.
# See https://github.com/inter-co/science-codeevolve/blob/main/LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0
#
# ===--------------------------------------------------------------------------------------===#
#
# This file implements the command-line interface of CodeEvolve.
#
# ===--------------------------------------------------------------------------------------===#

from typing import Any, Dict, List, Tuple, Optional

import argparse
import asyncio
import multiprocessing as mp
import multiprocessing.sharedctypes as mpsct
import multiprocessing.synchronize as mps
import ctypes
import os
from pathlib import Path
import re
import sys

import yaml

from codeevolve.islands import (
    PipeEdge,
    IslandData,
    GlobalData,
    GlobalBestProg,
    get_edge_list,
    get_pipe_graph,
)
from codeevolve.evolution import codeevolve
from codeevolve.utils.logging_utils import cli_logger


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for CodeEvolve execution.

    Returns:
        Parsed command-line arguments containing input directory, config path,
        output directory, checkpoint settings, and logging preferences.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inpt_dir",
        type=str,
        help="path to input directory containing initial solution and evaluation file.",
        required=True,
    )
    parser.add_argument("--cfg_path", type=str, help="path to .yaml config file.")
    parser.add_argument(
        "--out_dir",
        type=str,
        help="path to directory that will contain the outputs of CodeEvolve.",
        required=True,
    )
    parser.add_argument(
        "--load_ckpt",
        type=int,
        default=0,
        help="checkpoint to be loaded, if 0 will start anew, if -1 will load latest.",
    )
    parser.add_argument(
        "--terminal_logging",
        action="store_true",
        help="if true, dynamically displays logs from all islands in terminal.",
    )

    args = parser.parse_args()

    return args


def setup_isl_args(args: Dict[str, Any], num_islands: int) -> Dict[int, Dict[str, Any]]:
    """Sets up island-specific arguments and determines checkpoint synchronization.

    This function creates separate output directories and checkpoint directories
    for each island, identifies available checkpoints, and ensures all islands
    start from the same checkpoint epoch for consistency.

    Args:
        args: Global command-line arguments dictionary.
        num_islands: Total number of islands in the distributed system.

    Returns:
        Dictionary mapping island IDs to their specific argument configurations,
        with synchronized checkpoint loading across all islands.
    """
    isl2args: Dict[int, Dict[str, Any]] = {}

    common_ckpts: set[str] = {}
    for island_id in range(num_islands):
        isl_args: Dict[str, Any] = args.copy()
        isl_args["isl_out_dir"] = isl_args["out_dir"].joinpath(f"{island_id}/")
        isl_args["ckpt_dir"] = isl_args["isl_out_dir"].joinpath("ckpt/")
        isl2args[island_id] = isl_args

        os.makedirs(isl_args["isl_out_dir"], exist_ok=True)
        os.makedirs(isl_args["ckpt_dir"], exist_ok=True)

        ckpts: List[str] = [
            f for f in os.listdir(isl_args["ckpt_dir"]) if re.match(r"ckpt_\d+\.pkl$", f)
        ]
        common_ckpts = common_ckpts.intersection(set(ckpts)) if island_id else set(ckpts)

    global_ckpt: int = 0
    if len(common_ckpts):
        latest_common_ckpt = max(
            int(re.search(r"ckpt_(\d+)\.pkl$", f).group(1)) for f in common_ckpts
        )
        if args["load_ckpt"] and f"ckpt_{args["load_ckpt"]}.pkl" in common_ckpts:
            global_ckpt = args["load_ckpt"]
            print(f"Loading common checkpoint: {global_ckpt}")
        else:
            global_ckpt = latest_common_ckpt
            print(f"Resuming from latest common checkpoint: {global_ckpt}")
    else:
        print("No common checkpoints found. Starting from epoch 0.")

    for island_id in range(num_islands):
        isl2args[island_id]["load_ckpt"] = global_ckpt

    return isl2args


def main():
    """Main entry point for CodeEvolve execution with multiprocessing setup.

    This function orchestrates the entire CodeEvolve system by:
    1. Loading and copying configuration files
    2. Setting up shared memory and synchronization primitives
    3. Creating island communication topology
    4. Spawning island processes for distributed evolution
    5. Managing optional terminal logging daemon
    6. Coordinating process cleanup

    Args:
        args: Dictionary containing parsed command-line arguments including
              input/output paths, configuration, and execution settings.
    """

    def _async_run_evolve(run_args: Dict[str, Any], isl_data: IslandData, global_data: GlobalData):
        asyncio.run(codeevolve(run_args, isl_data, global_data))

    # args
    args: Dict[str, Any] = vars(parse_args())
    args["inpt_dir"] = Path(args["inpt_dir"])
    args["cfg_path"] = Path(args["cfg_path"])
    args["out_dir"] = Path(args["out_dir"])

    try:
        for path in [args["inpt_dir"], args["cfg_path"]]:
            assert os.path.exists(path), f"Path {path} not found."
    except AssertionError as err:
        print(str(err))
        sys.exit(1)

    try:
        args["api_base"] = os.environ["API_BASE"]
        args["api_key"] = os.environ["API_KEY"]
    except Exception as err:
        print("Export API_KEY and API_BASE as environment variables before running CodeEvolve.")
        sys.exit(1)

    # config
    os.makedirs(args["out_dir"], exist_ok=True)
    cfg_copy_path: Path = args["out_dir"].joinpath(args["cfg_path"].name)

    try:
        if os.path.exists(cfg_copy_path) and args["load_ckpt"]:
            config: Dict[Any, Any] = yaml.safe_load(open(cfg_copy_path, "r"))
        else:
            config: Dict[Any, Any] = yaml.safe_load(open(args["cfg_path"], "r"))
            yaml.safe_dump(config, open(cfg_copy_path, "w"))
    except Exception as err:
        print(str(err))
        sys.exit(1)

    evolve_config: Dict[str, Any] = config["EVOLVE_CONFIG"]
    isl2args: Dict[int, Dict[str, Any]] = setup_isl_args(args, evolve_config["num_islands"])

    # synchronization primitives
    global_best_sol: GlobalBestProg = GlobalBestProg(
        fitness=mp.Value(ctypes.c_longdouble, 0, lock=False),
        iteration_found=mp.Value(ctypes.c_uint, 0, lock=False),
        island_found=mp.Value(ctypes.c_int, -1, lock=False),
    )
    early_stop_counter: mpsct.Synchronized = mp.Value(ctypes.c_uint, 0, lock=False)
    early_stop_aux: mpsct.Synchronized = mp.Value(ctypes.c_int, 0, lock=False)
    lock: mps.Lock = mp.Lock()
    barrier: mps.Barrier = mp.Barrier(parties=evolve_config["num_islands"])
    log_queue: mp.Queue = mp.Queue()

    global_data: GlobalData = GlobalData(
        best_sol=global_best_sol,
        early_stop_counter=early_stop_counter,
        early_stop_aux=early_stop_aux,
        lock=lock,
        barrier=barrier,
        log_queue=log_queue,
    )

    # islands
    try:
        edge_list: List[Tuple[int, int]] = get_edge_list(
            evolve_config["num_islands"], evolve_config["migration_topology"]
        )
    except Exception as err:
        print(str(err))
        sys.exit(1)

    in_adj: Optional[List[PipeEdge]] = None
    out_adj: Optional[List[PipeEdge]] = None
    if len(edge_list):
        in_adj, out_adj = get_pipe_graph(evolve_config["num_islands"], edge_list)

    processes: List[mp.Process] = []

    if args.get("terminal_logging", False):
        log_formatter_daemon = mp.Process(
            target=cli_logger,
            args=(args, global_data, log_queue, evolve_config["num_islands"]),
            daemon=True,
        )
        log_formatter_daemon.start()

    # spawn processes
    for island_id in range(evolve_config["num_islands"]):
        isl_data: IslandData = IslandData(
            id=island_id,
            in_neigh=in_adj[island_id] if in_adj else None,
            out_neigh=out_adj[island_id] if out_adj else None,
        )

        process = mp.Process(
            target=_async_run_evolve, args=(isl2args[island_id], isl_data, global_data)
        )
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    if args.get("terminal_logging", False):
        # kill log daemon
        log_queue.put(None)
        log_formatter_daemon.join()

    return 0


if __name__ == "__main__":
    sys.exit(main())
