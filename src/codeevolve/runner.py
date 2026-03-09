# ===--------------------------------------------------------------------------------------===#
#
# Part of the CodeEvolve Project, under the Apache License v2.0.
# See https://github.com/inter-co/science-codeevolve/blob/main/LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0
#
# ===--------------------------------------------------------------------------------------===#
#
# This file implements process management for CodeEvolve's distributed island evolution.
#
# ===--------------------------------------------------------------------------------------===#

import asyncio
import multiprocessing as mp
import signal
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from codeevolve.evolution import codeevolve
from codeevolve.islands.graph import IslandCommunicationData, PipeEdge
from codeevolve.islands.sync import GlobalSyncData
from codeevolve.utils.constants import CRASH_LOG_FILE
from codeevolve.utils.lock import DirectoryLock
from codeevolve.utils.logging import cli_logger

# ---------------------------------------------------------------------------
# Global cleanup state for signal handlers
# ---------------------------------------------------------------------------

_cleanup_state: Dict[str, Any] = {
    "processes": [],
    "log_daemon": None,
    "log_queue": None,
    "directory_lock": None,
    "cleaned_up": False,
}


def get_cleanup_state() -> Dict[str, Any]:
    """Returns the global cleanup state dictionary.

    This is used by cli.py to register resources that need cleanup.
    """
    return _cleanup_state


# ---------------------------------------------------------------------------
# Signal handling
# ---------------------------------------------------------------------------


def _terminate_processes(processes: List[mp.Process], timeout: float = 1) -> None:
    """Terminates a list of processes.

    Args:
        processes: List of processes to terminate.
        timeout: Seconds to wait for termination before force killing.
    """
    for process in processes:
        if process.is_alive():
            process.terminate()
            process.join(timeout=timeout)
            if process.is_alive():
                process.kill()
                process.join()


def _cleanup_on_signal(signum: int, frame: Any) -> None:
    """Signal handler for shutdown on SIGTSTP, SIGTERM, etc.

    This handler ensures all child processes are terminated and the directory
    lock is released when the process receives a termination signal.

    Args:
        signum: Signal number received.
        frame: Current stack frame (unused).
    """
    if _cleanup_state["cleaned_up"]:
        return

    _cleanup_state["cleaned_up"] = True

    signal_name: str = signal.Signals(signum).name
    print(f"\n\nReceived {signal_name}. Shutting down...", file=sys.stderr)

    processes: List[mp.Process] = _cleanup_state["processes"]
    _terminate_processes(processes)

    log_daemon: Optional[mp.Process] = _cleanup_state["log_daemon"]
    log_queue: Optional[mp.Queue] = _cleanup_state["log_queue"]
    if log_daemon and log_daemon.is_alive():
        if log_queue:
            log_queue.put(None)
        log_daemon.join(timeout=2.0)
        if log_daemon.is_alive():
            log_daemon.terminate()

    directory_lock: Optional[DirectoryLock] = _cleanup_state["directory_lock"]
    if directory_lock:
        directory_lock.release()

    print("Cleanup complete.", file=sys.stderr)
    sys.exit(128 + signum)


def setup_signal_handlers() -> None:
    """Set up signal handlers for shutdown.

    Registers handlers for SIGTERM, SIGTSTP (CTRL-Z), and SIGQUIT (CTRL-\\)
    to ensure proper cleanup of child processes and lock files.
    """
    signal.signal(signal.SIGTERM, _cleanup_on_signal)
    signal.signal(signal.SIGTSTP, _cleanup_on_signal)
    signal.signal(signal.SIGQUIT, _cleanup_on_signal)


# ---------------------------------------------------------------------------
# Process spawning
# ---------------------------------------------------------------------------


def async_run_codeevolve(
    run_args: Dict[str, Any], isl_data: IslandCommunicationData, global_data: GlobalSyncData
) -> None:
    """Wrapper to run evolution with exception handling."""
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGTSTP, signal.SIG_DFL)
    signal.signal(signal.SIGQUIT, signal.SIG_DFL)

    try:
        asyncio.run(codeevolve(run_args, isl_data, global_data))
    except Exception as e:
        error_msg = f"Island {isl_data.id} crashed with exception: {type(e).__name__}: {str(e)}"
        full_traceback = traceback.format_exc()
        try:
            isl_logs_dir: Path = run_args.get("logs_dir")
            if isl_logs_dir:
                _write_island_crash_log(
                    isl_logs_dir, error_msg, full_traceback, int(global_data.start_time.value)
                )
        except Exception:
            pass

        try:
            global_data.log_queue.put(f"[CRITICAL] {error_msg}")
        except Exception:
            pass

        raise


def spawn_island_processes(
    num_islands: int,
    isl2args: Dict[int, Dict[str, Any]],
    in_adj: Optional[List[PipeEdge]],
    out_adj: Optional[List[PipeEdge]],
    global_data: GlobalSyncData,
) -> List[mp.Process]:
    """Spawns evolution processes for each island in the distributed islands algorithm.

    Args:
        num_islands: Total number of islands to spawn.
        isl2args: Island-specific argument configurations.
        in_adj: Incoming pipe adjacency list for migrations.
        out_adj: Outgoing pipe adjacency list for migrations.
        global_data: Shared global data structures.

    Returns:
        List of spawned process objects.
    """

    processes: List[mp.Process] = []
    for island_id in range(num_islands):
        isl_data: IslandCommunicationData = IslandCommunicationData(
            id=island_id,
            in_neigh=in_adj[island_id] if in_adj else None,
            out_neigh=out_adj[island_id] if out_adj else None,
        )

        process: mp.Process = mp.Process(
            target=async_run_codeevolve,
            args=(isl2args[island_id], isl_data, global_data),
        )
        processes.append(process)
        process.start()

    return processes


# ---------------------------------------------------------------------------
# Log daemon management
# ---------------------------------------------------------------------------


def start_log_daemon(
    args: Dict[str, Any],
    global_data: GlobalSyncData,
    num_islands: int,
) -> Optional[mp.Process]:
    """Starts the logging daemon process if terminal logging is enabled.

    Args:
        args: Command-line arguments dictionary.
        global_data: Shared global data structures containing log queue.
        num_islands: Number of islands to monitor.

    Returns:
        The log daemon process if started, None otherwise.
    """
    if not args.get("terminal_logging", False):
        return None

    log_daemon: mp.Process = mp.Process(
        target=cli_logger,
        args=(args, global_data, global_data.log_queue, num_islands),
        daemon=True,
    )
    log_daemon.start()
    return log_daemon


def cleanup_log_daemon(
    log_daemon: Optional[mp.Process],
    log_queue: mp.Queue,
    timeout: float = 2.0,
) -> None:
    """Shuts down the log daemon process.

    Sends a sentinel value to signal shutdown, waits for termination,
    and force-terminates if necessary.

    Args:
        log_daemon: The logging daemon process to shut down (may be None).
        log_queue: Queue to send shutdown signal through.
        timeout: Maximum seconds to wait for shutdown.
    """
    if log_daemon and log_daemon.is_alive():
        log_queue.put(None)
        log_daemon.join(timeout=timeout)
        if log_daemon.is_alive():
            log_daemon.terminate()


# ---------------------------------------------------------------------------
# Crash logging
# ---------------------------------------------------------------------------


def _write_island_crash_log(
    isl_logs_dir: Path, error_msg: str, full_traceback: str, time: int = 0
) -> None:
    """Writes detailed crash information to an island's crash log file.

    Called from within the child process where the exception occurred.

    Args:
        isl_logs_dir: Island's logs directory.
        error_msg: Summary error message.
        full_traceback: Full Python traceback string.
        time: Int identifying global time of the run.
    """
    crash_log_path: Path = isl_logs_dir.joinpath(CRASH_LOG_FILE.format(time=time))
    with open(crash_log_path, "a") as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"CRASH REPORT\n")
        f.write(f"{'='*60}\n")
        f.write(f"{error_msg}\n\n")
        f.write("Full Traceback:\n")
        f.write(full_traceback)
        f.write(f"{'='*60}\n\n")


def _write_crash_summary(
    out_dir: Path, island_id: int, exit_code: int, message: str, time: int = 0
) -> None:
    """Writes crash summary to the main output directory's crash log.

    Called from the parent process when it detects an island has crashed.

    Args:
        out_dir: Main output directory.
        island_id: ID of the island that crashed.
        exit_code: Exit code of the crashed process.
        message: Summary message.
        time: Int identifying global time of the run.
    """
    crash_log_path: Path = out_dir.joinpath(CRASH_LOG_FILE.format(time=time))
    with open(crash_log_path, "a") as f:
        f.write(f"\n{'='*60}\n")
        f.write(f"CRASH REPORT\n")
        f.write(f"{'='*60}\n")
        f.write(f"Island: {island_id}\n")
        f.write(f"Exit Code: {exit_code}\n")
        f.write(f"Message: {message}\n")
        f.write(
            f"See island_{island_id}/logs/{CRASH_LOG_FILE.format(time=time)} for full traceback.\n"
        )
        f.write(f"{'='*60}\n\n")


# ---------------------------------------------------------------------------
# Process monitoring
# ---------------------------------------------------------------------------


def monitor_island_processes(
    processes: List[mp.Process],
    global_data: GlobalSyncData,
    log_daemon: Optional[mp.Process],
    out_dir: Path,
    poll_interval: float = 1.0,
) -> int:
    """Monitors island processes and handles failures.

    This function continuously monitors all island processes. If any island
    dies unexpectedly, it:
    1. Stops the log daemon
    2. Logs the failure to console and crash.log file
    3. Terminates all remaining islands
    4. Returns with exit code 1 if any island failed, 0 if all completed successfully

    Args:
        processes: List of island process objects to monitor.
        global_data: Shared data structures containing log queue.
        log_daemon: Optional logging daemon process.
        out_dir: Output directory for crash logs.
        poll_interval: Time in seconds between process health checks.

    Returns:
        Exit code: 0 if all processes completed successfully, 1 if any failed.
    """
    num_islands: int = len(processes)
    completed: List[bool] = [False] * num_islands
    time: int = int(global_data.start_time.value)

    try:
        while not all(completed):
            for i, process in enumerate(processes):
                if completed[i]:
                    continue

                process.join(timeout=poll_interval)

                if not process.is_alive():
                    completed[i] = True

                    if process.exitcode != 0:
                        cleanup_log_daemon(log_daemon, global_data.log_queue)

                        error_msg: str = (
                            f"Island {i} died unexpectedly with exit code {process.exitcode}"
                        )
                        _write_crash_summary(out_dir, i, process.exitcode, error_msg, time)

                        print(f"\n{'='*60}", file=sys.stderr)
                        print(f"ERROR: {error_msg}", file=sys.stderr)
                        print(
                            f"See {out_dir}/{CRASH_LOG_FILE.format(time=time)} and island logs for details.",
                            file=sys.stderr,
                        )
                        print(f"{'='*60}\n", file=sys.stderr)

                        print("Terminating all remaining islands...", file=sys.stderr)
                        other_processes: List[mp.Process] = [
                            other_process
                            for j, other_process in enumerate(processes)
                            if j != i and other_process.is_alive()
                        ]
                        _terminate_processes(processes=other_processes)

                        return 1

        print("\nAll islands completed successfully.")
        return 0

    except KeyboardInterrupt:
        cleanup_log_daemon(log_daemon, global_data.log_queue)

        print("\n\nKeyboard interrupt received. Shutting down islands...", file=sys.stderr)
        _terminate_processes(processes)

        directory_lock: Optional[DirectoryLock] = _cleanup_state.get("directory_lock")
        if directory_lock:
            directory_lock.release()

        return 130
