# ===--------------------------------------------------------------------------------------===#
#
# Part of the CodeEvolve Project, under the Apache License v2.0.
# See https://github.com/inter-co/science-codeevolve/blob/main/LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0
#
# ===--------------------------------------------------------------------------------------===#
#
# This file implements configuration loading and validation for CodeEvolve.
#
# ===--------------------------------------------------------------------------------------===#

import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml

from codeevolve.utils.constants import (
    ASCII_LOGO,
    ASCII_NAME,
    CHECKPOINT_FILE_FORMAT,
    CHECKPOINT_PATTERN,
)

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


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
            print("Error: --cfg_path is required when starting a new run")
            sys.exit(1)
        if not cfg_path.exists():
            print(f"Error: Config file not found: {cfg_path}")
            sys.exit(1)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def create_config_copy(args: Dict[str, Any]) -> Tuple[Dict[str, Any], Path]:
    """Loads configuration file and creates a copy in output directory.

    Args:
        args: Dictionary of command-line arguments.

    Returns:
        Tuple of (config dictionary, path to config copy in output directory).

    Raises:
        SystemExit: If config file operations fail.
    """
    out_dir: Path = args["out_dir"]
    cfg_path: Path = args["cfg_path"]
    cfg_copy_path: Path = out_dir.joinpath(cfg_path.name)

    try:
        with open(cfg_path, "r") as f:
            config: Dict[str, Any] = yaml.safe_load(f)
        with open(cfg_copy_path, "w") as f:
            yaml.safe_dump(config, f)
        return config, cfg_copy_path
    except Exception as err:
        print(f"Error loading config: {err}")
        sys.exit(1)


def load_config(args: Dict[str, Any]) -> Tuple[Dict[str, Any], Path]:
    """Loads configuration file in output directory.

    Args:
        args: Dictionary of command-line arguments.

    Returns:
        Tuple of (config dictionary, path to config in output directory).

    Raises:
        SystemExit: If config file operations fail.
    """
    out_dir: Path = args["out_dir"]
    cfg_files: List[str] = [f for f in os.listdir(out_dir) if f.endswith(".yaml")]

    if len(cfg_files) == 0:
        print(f"Error: No config file found in {out_dir} while loading checkpoint.")
        sys.exit(1)
    elif len(cfg_files) > 1:
        print(f"Error: Multiple config files found in {out_dir} (expected one), found: {cfg_files}")
        sys.exit(1)

    cfg_copy_path: Path = out_dir.joinpath(cfg_files[0])
    try:
        with open(cfg_copy_path, "r") as f:
            config: Dict[str, Any] = yaml.safe_load(f)
        return config, cfg_copy_path
    except Exception as err:
        print(f"Error loading config: {err}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------


def find_common_checkpoints(ckpt_dirs: List[Path]) -> Set[str]:
    """Finds checkpoints that exist across all island directories.

    Args:
        ckpt_dirs: List of checkpoint directory paths for each island.

    Returns:
        Set of checkpoint filenames common to all islands.
    """
    common_ckpts: Optional[Set[str]] = None
    checkpoint_pattern: re.Pattern[str] = re.compile(CHECKPOINT_PATTERN)

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
    if not common_ckpts or requested_ckpt == 0:
        return 0

    checkpoint_epochs: List[int] = [
        int(re.search(CHECKPOINT_PATTERN, f).group(1)) for f in common_ckpts
    ]
    latest_epoch: int = max(checkpoint_epochs)

    if requested_ckpt > 0 and CHECKPOINT_FILE_FORMAT.format(epoch=requested_ckpt) in common_ckpts:
        return requested_ckpt
    else:
        return latest_epoch


def compute_cpu_affinity_sets(
    budget_config: Dict[str, Any],
    num_islands: int,
) -> Tuple[List[Optional[Set[int]]], List[str]]:
    """Partitions available CPUs across islands when num_cpus_per_eval is configured.

    When ``num_cpus_per_eval`` is set in budget_config, this function partitions the
    process-available CPUs (as reported by ``os.sched_getaffinity``) into consecutive
    slices — one per island. Each island then pins itself to its slice at startup,
    preventing cross-island CPU contention and making wall-clock evaluation time
    roughly equal to CPU time.

    Falls back to unpartitioned execution (all-None, no pinning) when:
    - ``num_cpus_per_eval`` is absent or ``None``
    - ``num_cpus_per_eval`` is not a positive integer
    - The platform does not support ``os.sched_getaffinity`` (non-Linux)
    - The required total CPUs exceed available CPUs

    Args:
        budget_config: The ``BUDGET_CONFIG`` dictionary from the run config.
        num_islands: Total number of islands.

    Returns:
        Tuple of (affinity_sets, warnings):
          - affinity_sets: List indexed by island ID; each element is a set of CPU
            indices to pin to, or ``None`` if no pinning should be applied.
          - warnings: List of human-readable warning strings describing any fallback.
    """
    no_pinning: List[Optional[Set[int]]] = [None] * num_islands
    warnings: List[str] = []

    num_cpus_per_eval: Optional[int] = budget_config.get("num_cpus_per_eval", None)
    if num_cpus_per_eval is None:
        return no_pinning, warnings

    if not isinstance(num_cpus_per_eval, int) or num_cpus_per_eval < 1:
        warnings.append(
            f"Warning: num_cpus_per_eval must be a positive integer, "
            f"got {num_cpus_per_eval!r}. Falling back to unpartitioned execution."
        )
        return no_pinning, warnings

    if not hasattr(os, "sched_getaffinity"):
        warnings.append(
            "Warning: num_cpus_per_eval is set but CPU affinity pinning is not supported "
            "on this platform (requires Linux). Falling back to unpartitioned execution."
        )
        return no_pinning, warnings

    available_cpus: List[int] = sorted(os.sched_getaffinity(0))
    total_needed: int = num_cpus_per_eval * num_islands
    if total_needed > len(available_cpus):
        warnings.append(
            f"Warning: num_cpus_per_eval={num_cpus_per_eval} × num_islands={num_islands} "
            f"= {total_needed} CPUs needed, but only {len(available_cpus)} are available. "
            "Falling back to unpartitioned execution."
        )
        return no_pinning, warnings

    affinity_sets: List[Optional[Set[int]]] = []
    for isl_id in range(num_islands):
        start: int = isl_id * num_cpus_per_eval
        affinity_sets.append(set(available_cpus[start : start + num_cpus_per_eval]))
    return affinity_sets, warnings


def setup_island_args(
    args: Dict[str, Any],
    num_islands: int,
    cfg_copy_path: Path,
    cpu_affinity_sets: Optional[List[Optional[Set[int]]]] = None,
) -> Dict[int, Dict[str, Any]]:
    """Sets up island-specific arguments with synchronized checkpoint loading.

    Creates separate output and checkpoint directories for each island and ensures
    all islands start from the same checkpoint epoch for consistency.

    Args:
        args: Global command-line arguments dictionary.
        num_islands: Total number of islands in the distributed system.
        cfg_copy_path: Path to config file copy in experiment directory.
        cpu_affinity_sets: Optional list of per-island CPU affinity sets computed
            by ``compute_cpu_affinity_sets``. When provided, each island's args
            receive a ``cpu_affinity_set`` key with its assigned core set (or
            ``None`` if no pinning applies to that island).

    Returns:
        Dictionary mapping island IDs to their specific argument configurations.
    """
    isl2args: Dict[int, Dict[str, Any]] = {}
    ckpt_dirs: List[Path] = []

    for island_id in range(num_islands):
        isl_args: Dict[str, Any] = args.copy()
        isl_out_dir: Path = args["out_dir"].joinpath(f"island_{island_id}")
        ckpt_dir: Path = isl_out_dir.joinpath("ckpt")
        logs_dir: Path = isl_out_dir.joinpath("logs")

        isl_args["isl_out_dir"] = isl_out_dir
        isl_args["ckpt_dir"] = ckpt_dir
        isl_args["logs_dir"] = logs_dir
        isl_args["cfg_path"] = cfg_copy_path
        isl_args["cpu_affinity_set"] = (
            cpu_affinity_sets[island_id] if cpu_affinity_sets is not None else None
        )

        os.makedirs(isl_out_dir, exist_ok=True)
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)

        isl2args[island_id] = isl_args
        ckpt_dirs.append(ckpt_dir)

    common_ckpts: Set[str] = find_common_checkpoints(ckpt_dirs)
    global_ckpt: int = determine_checkpoint_to_load(common_ckpts, args["load_ckpt"])

    for island_id in range(num_islands):
        isl2args[island_id]["load_ckpt"] = global_ckpt

    return isl2args


# ---------------------------------------------------------------------------
# CLI display
# ---------------------------------------------------------------------------


def print_dict_rec(
    base_dict: Dict[str, Any],
    forbidden_keys: Optional[List[str]] = None,
    indent: int = 0,
) -> None:
    """Recursively prints a dictionary in a formatted hierarchical structure.

    For nested dictionaries, prints a header with the key name surrounded by
    '===' and then recursively prints the nested dictionary. For leaf values,
    prints the key-value pair on a single line.

    Args:
        base_dict: The dictionary to print. Values can be any type, but nested
            dictionaries will be recursively printed with indentation headers.
        forbidden_keys: List of forbidden keys that will not be printed.
                        Recursively passed to nested dicts.
        indent: Auxiliary integer for indentation.
    """
    prefix: str = "  " * indent
    for key, value in base_dict.items():
        if isinstance(value, dict):
            print(f"{prefix}{key}:")
            print_dict_rec(value, forbidden_keys, indent + 1)
        else:
            if forbidden_keys is None or key not in forbidden_keys:
                print(f"{prefix}{key}: {value}")
            else:
                print(f"{prefix}{key}: ***")


def display_run_data(
    args: Dict[str, Any],
    config: Dict[str, Any],
    global_ckpt: int,
    metadata: Optional[Dict[str, Any]],
    warnings: List[str],
) -> None:
    """
    Displays the relevant data about the run to be executed. Unless the '--y' flag is set,
    prompts the user to confirm before proceeding. If the user does not confirm
    with 'y', the program exits.

    Args:
        args: Dictionary of command-line arguments to display.
        config: Dictionary of configuration settings loaded from the YAML file.
        global_ckpt: Integer with the global checkpoint that each island will load.
        metadata: CodeEvolve metadata
        warnings: Warnings from cli.py
    """
    print("=" * 100)
    print(ASCII_LOGO)
    print(ASCII_NAME)
    print("=" * 100)
    print("=" * 10 + " TERMINAL ARGS " + "=" * 10)
    print_dict_rec(args, ["api_key"])
    print("=" * 10 + " CONFIG " + "=" * 10)
    print_dict_rec(config)
    if len(warnings) > 0:
        print("=" * 10 + " WARNINGS " + "=" * 10)
        for msg in warnings:
            print(msg)
    print("=" * 10 + f" STARTING FROM CKPT {global_ckpt} " + "=" * 10)
    if metadata is not None:
        print("=" * 10 + " LOADED METADATA " + "=" * 10)
        print_dict_rec(metadata)
    print("=" * 100)
    if not args.get("y", False):
        cont: str = input("Do you wish to continue? [y/[^y]]")
        if cont != "y":
            sys.exit(1)
