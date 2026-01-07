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
from typing import Any, Dict, List, Tuple, Optional, Set
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
    return parser.parse_args()


def validate_environment() -> Tuple[str, str]:
    """Validates required environment variables are set.

    Returns:
        Tuple of (api_base, api_key) environment variable values.

    Raises:
        SystemExit: If required environment variables are not set.
    """
    try:
        api_base: str = os.environ["API_BASE"]
        api_key: str = os.environ["API_KEY"]
        return api_base, api_key
    except KeyError:
        print(
            "Error: Export API_KEY and API_BASE as environment variables before running CodeEvolve."
        )
        sys.exit(1)


def validate_paths(inpt_dir: Path, cfg_path: Optional[Path], loading_checkpoint: bool) -> None:
    """Validates that required input paths exist.

    Args:
        inpt_dir: Path to input directory.
        cfg_path: Path to configuration file (may be None if loading checkpoint).
        loading_checkpoint: Whether a checkpoint is being loaded.

    Raises:
        SystemExit: If any required path does not exist.
    """
    if not inpt_dir.exists():
        print(f"Error: Input directory not found: {inpt_dir}")
        sys.exit(1)

    if not loading_checkpoint:
        if cfg_path is None:
            print("Error: --cfg_path is required when starting a new run (load_ckpt=0)")
            sys.exit(1)
        if not cfg_path.exists():
            print(f"Error: Config file not found: {cfg_path}")
            sys.exit(1)


def load_or_copy_config(args: Dict[str, Any]) -> Tuple[Dict[Any, Any], Path]:
    """Loads configuration file and creates a copy in output directory if starting new run.

    Args:
        args: Dictionary of command-line arguments.

    Returns:
        Tuple of (config dictionary, path to config copy in output directory).

    Raises:
        SystemExit: If config file operations fail.
    """
    out_dir: Path = args["out_dir"]

    if args["load_ckpt"] == 0:
        os.makedirs(out_dir, exist_ok=True)
        cfg_path: Path = args["cfg_path"]
        cfg_copy_path: Path = out_dir / cfg_path.name

        try:
            with open(cfg_path, "r") as f:
                config: Dict[Any, Any] = yaml.safe_load(f)
            with open(cfg_copy_path, "w") as f:
                yaml.safe_dump(config, f)
            return config, cfg_copy_path
        except Exception as err:
            print(f"Error loading config: {err}")
            sys.exit(1)
    else:
        cfg_files: List[str] = [f for f in os.listdir(out_dir) if f.endswith(".yaml")]

        if len(cfg_files) == 0:
            print(f"Error: No config file found in {out_dir} while loading checkpoint.")
            sys.exit(1)
        elif len(cfg_files) > 1:
            print(
                f"Error: Multiple config files found in {out_dir} (expected one), found: {cfg_files}"
            )
            sys.exit(1)

        cfg_copy_path: Path = out_dir / cfg_files[0]
        try:
            with open(cfg_copy_path, "r") as f:
                config: Dict[Any, Any] = yaml.safe_load(f)
            return config, cfg_copy_path
        except Exception as err:
            print(f"Error loading config: {err}")
            sys.exit(1)


def find_common_checkpoints(ckpt_dirs: List[Path]) -> Set[str]:
    """Finds checkpoints that exist across all island directories.

    Args:
        ckpt_dirs: List of checkpoint directory paths for each island.

    Returns:
        Set of checkpoint filenames common to all islands.
    """
    common_ckpts: Optional[Set[str]] = None
    checkpoint_pattern: re.Pattern[str] = re.compile(r"ckpt_\d+\.pkl$")

    for ckpt_dir in ckpt_dirs:
        if not ckpt_dir.exists():
            continue

        ckpts: Set[str] = {f for f in os.listdir(ckpt_dir) if checkpoint_pattern.match(f)}
        common_ckpts = ckpts if common_ckpts is None else common_ckpts.intersection(ckpts)

    return common_ckpts or set()


def determine_checkpoint_to_load(common_ckpts: Set[str], requested_ckpt: int) -> int:
    """Determines which checkpoint epoch to load based on availability and user request.

    Args:
        common_ckpts: Set of available checkpoint filenames.
        requested_ckpt: User-requested checkpoint (0 for new, -1 for latest, or specific epoch).

    Returns:
        Checkpoint epoch number to load (0 if starting new).
    """
    if not common_ckpts:
        print("No common checkpoints found. Starting from epoch 0.")
        return 0

    checkpoint_epochs: List[int] = [
        int(re.search(r"ckpt_(\d+)\.pkl$", f).group(1)) for f in common_ckpts
    ]
    latest_epoch: int = max(checkpoint_epochs)

    if requested_ckpt > 0 and f"ckpt_{requested_ckpt}.pkl" in common_ckpts:
        print(f"Loading common checkpoint: {requested_ckpt}")
        return requested_ckpt
    else:
        print(f"Resuming from latest common checkpoint: {latest_epoch}")
        return latest_epoch


def setup_isl_args(
    args: Dict[str, Any], num_islands: int, cfg_copy_path: Path
) -> Dict[int, Dict[str, Any]]:
    """Sets up island-specific arguments with synchronized checkpoint loading.

    Creates separate output and checkpoint directories for each island and ensures
    all islands start from the same checkpoint epoch for consistency.

    Args:
        args: Global command-line arguments dictionary.
        num_islands: Total number of islands in the distributed system.
        cfg_copy_path: Path to config file copy in experiment directory.

    Returns:
        Dictionary mapping island IDs to their specific argument configurations.
    """
    isl2args: Dict[int, Dict[str, Any]] = {}
    ckpt_dirs: List[Path] = []

    for island_id in range(num_islands):
        isl_args: Dict[str, Any] = args.copy()
        isl_out_dir: Path = args["out_dir"] / f"{island_id}"
        ckpt_dir: Path = isl_out_dir / "ckpt"

        isl_args["isl_out_dir"] = isl_out_dir
        isl_args["ckpt_dir"] = ckpt_dir
        isl_args["cfg_path"] = cfg_copy_path

        os.makedirs(isl_out_dir, exist_ok=True)
        os.makedirs(ckpt_dir, exist_ok=True)

        isl2args[island_id] = isl_args
        ckpt_dirs.append(ckpt_dir)

    common_ckpts: Set[str] = find_common_checkpoints(ckpt_dirs)
    global_ckpt: int = determine_checkpoint_to_load(common_ckpts, args["load_ckpt"])

    for island_id in range(num_islands):
        isl2args[island_id]["load_ckpt"] = global_ckpt

    return isl2args


def create_global_data(num_islands: int) -> GlobalData:
    """Creates shared memory structures for inter-island coordination.

    Args:
        num_islands: Total number of islands in the system.

    Returns:
        GlobalData instance containing synchronization primitives and shared state.
    """
    global_best_sol: GlobalBestProg = GlobalBestProg(
        fitness=mp.Value(ctypes.c_longdouble, 0, lock=False),
        iteration_found=mp.Value(ctypes.c_uint, 0, lock=False),
        island_found=mp.Value(ctypes.c_int, -1, lock=False),
    )

    return GlobalData(
        best_sol=global_best_sol,
        early_stop_counter=mp.Value(ctypes.c_uint, 0, lock=False),
        early_stop_aux=mp.Value(ctypes.c_int, 0, lock=False),
        lock=mp.Lock(),
        barrier=mp.Barrier(parties=num_islands),
        log_queue=mp.Queue(),
    )


def setup_island_topology(
    num_islands: int, topology_type: str
) -> Tuple[Optional[List[PipeEdge]], Optional[List[PipeEdge]]]:
    """Configures communication topology between islands.

    Args:
        num_islands: Total number of islands.
        topology_type: Type of migration topology to create.

    Returns:
        Tuple of (incoming adjacency list, outgoing adjacency list) or (None, None) if no edges.

    Raises:
        SystemExit: If topology configuration fails.
    """
    try:
        edge_list: List[Tuple[int, int]] = get_edge_list(num_islands, topology_type)
    except Exception as err:
        print(f"Error creating migration topology: {err}")
        sys.exit(1)

    if not edge_list:
        return None, None

    in_adj: List[PipeEdge]
    out_adj: List[PipeEdge]
    in_adj, out_adj = get_pipe_graph(num_islands, edge_list)
    return in_adj, out_adj


def spawn_island_processes(
    num_islands: int,
    isl2args: Dict[int, Dict[str, Any]],
    in_adj: Optional[List[PipeEdge]],
    out_adj: Optional[List[PipeEdge]],
    global_data: GlobalData,
) -> List[mp.Process]:
    """Spawns evolution processes for each island.

    Args:
        num_islands: Total number of islands to spawn.
        isl2args: Island-specific argument configurations.
        in_adj: Incoming pipe adjacency list for migrations.
        out_adj: Outgoing pipe adjacency list for migrations.
        global_data: Shared global data structures.

    Returns:
        List of spawned process objects.
    """

    def async_run_evolve(
        run_args: Dict[str, Any], isl_data: IslandData, global_data: GlobalData
    ) -> None:
        asyncio.run(codeevolve(run_args, isl_data, global_data))

    processes: List[mp.Process] = []
    for island_id in range(num_islands):
        isl_data: IslandData = IslandData(
            id=island_id,
            in_neigh=in_adj[island_id] if in_adj else None,
            out_neigh=out_adj[island_id] if out_adj else None,
        )

        process: mp.Process = mp.Process(
            target=async_run_evolve,
            args=(isl2args[island_id], isl_data, global_data),
        )
        processes.append(process)
        process.start()

    return processes


def main() -> int:
    """Main entry point for CodeEvolve.

    Orchestrates the complete execution flow:
    1. Parses arguments and validates environment
    2. Loads configuration and sets up output directories
    3. Creates shared memory and synchronization primitives
    4. Configures island communication topology
    5. Spawns island processes for distributed evolution
    6. Manages optional terminal logging
    7. Coordinates shutdown

    Returns:
        Exit code (0 for success).
    """
    args_ns: argparse.Namespace = parse_args()
    args: Dict[str, Any] = vars(args_ns)

    args["inpt_dir"] = Path(args["inpt_dir"])
    args["cfg_path"] = Path(args["cfg_path"]) if args["cfg_path"] else None
    args["out_dir"] = Path(args["out_dir"])

    loading_checkpoint: bool = args["load_ckpt"] != 0
    validate_paths(args["inpt_dir"], args["cfg_path"], loading_checkpoint)

    api_base: str
    api_key: str
    api_base, api_key = validate_environment()
    args["api_base"] = api_base
    args["api_key"] = api_key

    config: Dict[Any, Any]
    cfg_copy_path: Path
    config, cfg_copy_path = load_or_copy_config(args)
    evolve_config: Dict[str, Any] = config["EVOLVE_CONFIG"]

    isl2args: Dict[int, Dict[str, Any]] = setup_isl_args(
        args, evolve_config["num_islands"], cfg_copy_path
    )
    global_data: GlobalData = create_global_data(evolve_config["num_islands"])

    in_adj: Optional[List[PipeEdge]]
    out_adj: Optional[List[PipeEdge]]
    in_adj, out_adj = setup_island_topology(
        evolve_config["num_islands"],
        evolve_config["migration_topology"],
    )

    log_daemon: Optional[mp.Process] = None
    if args.get("terminal_logging", False):
        log_daemon = mp.Process(
            target=cli_logger,
            args=(args, global_data, global_data.log_queue, evolve_config["num_islands"]),
            daemon=True,
        )
        log_daemon.start()

    processes: List[mp.Process] = spawn_island_processes(
        num_islands=evolve_config["num_islands"],
        isl2args=isl2args,
        in_adj=in_adj,
        out_adj=out_adj,
        global_data=global_data,
    )

    process: mp.Process
    for process in processes:
        process.join()

    if log_daemon:
        global_data.log_queue.put(None)
        log_daemon.join()

    return 0


if __name__ == "__main__":
    sys.exit(main())
