# ===--------------------------------------------------------------------------------------===#
#
# Part of the CodeEvolve Project, under the Apache License v2.0.
# See https://github.com/inter-co/science-codeevolve/blob/main/LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0
#
# ===--------------------------------------------------------------------------------------===#
#
# This file implements the evaluator class for executing programs.
#
# ===--------------------------------------------------------------------------------------===#

import json
import logging
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import psutil

from codeevolve.database import Program
from codeevolve.utils.constants import (
    DEFAULT_EXTENSION,
    DEFAULT_RESOURCE_CHECK_INTERVAL_S,
    LANGUAGE_TO_EXTENSION,
)

# ---------------------------------------------------------------------------
# Process utilities
# ---------------------------------------------------------------------------


def get_process_tree(parent: psutil.Process) -> List[psutil.Process]:
    """Retrieves all processes in a process tree including the parent.

    Args:
        parent: The root process of the tree to retrieve.

    Returns:
        A list containing the parent process and all its descendants.
        Returns an empty list if the process no longer exists.
    """
    try:
        children: List[psutil.Process] = parent.children(recursive=True)
        return [parent] + children
    except psutil.NoSuchProcess:
        return []


def kill_process_tree(parent: psutil.Process) -> None:
    """Terminates a process tree, first attempting graceful termination then forcing.

    This function first sends SIGTERM to all processes in the tree, waits briefly,
    then sends SIGKILL to any surviving processes.

    Args:
        parent: The root process of the tree to terminate.
    """
    processes: List[psutil.Process] = get_process_tree(parent)

    for proc in processes:
        try:
            proc.terminate()
        except psutil.NoSuchProcess:
            pass

    gone, alive = psutil.wait_procs(processes, timeout=0.5)

    for proc in alive:
        try:
            proc.kill()
        except psutil.NoSuchProcess:
            pass


def resource_monitor(
    process: psutil.Process,
    check_interval_s: float,
    kill_flag: threading.Event,
    mem_exceeded_flag: threading.Event,
    cpu_exceeded_flag: threading.Event,
    max_mem_b: Optional[int] = None,
    cpu_limit_s: Optional[float] = None,
) -> None:
    """Monitors resource usage of a process tree and kills it when any limit is exceeded.

    This function runs in a separate thread, polling the entire process tree at
    a fixed interval.  On each tick it optionally checks:

    * **Memory** (RSS across the tree) against ``max_mem_b``.
    * **CPU time** (accumulated user + system time across the tree) against
      ``cpu_limit_s``.

    Args:
        process: The root psutil Process of the evaluation subprocess tree.
        check_interval_s: Seconds to sleep between resource checks.
        kill_flag: Event set by the caller to stop monitoring.
        mem_exceeded_flag: Event set by this function when memory limit is exceeded.
        cpu_exceeded_flag: Event set by this function when CPU time limit is exceeded.
        max_mem_b: Maximum combined RSS in bytes across the process tree.
            If None, memory is not checked.
        cpu_limit_s: Maximum combined CPU time (user + system) in seconds across
            the process tree.  If None, CPU time is not checked.
    """
    try:
        while not kill_flag.is_set():
            try:
                if not process.is_running():
                    return
                total_mem: int = 0
                total_cpu_s: float = 0.0
                processes: List[psutil.Process] = get_process_tree(process)
                for proc in processes:
                    try:
                        if max_mem_b is not None:
                            total_mem += proc.memory_info().rss
                        if cpu_limit_s is not None:
                            cpu = proc.cpu_times()
                            total_cpu_s += cpu.user + cpu.system
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                if max_mem_b is not None and total_mem > max_mem_b:
                    kill_process_tree(process)
                    mem_exceeded_flag.set()
                    return
                if cpu_limit_s is not None and total_cpu_s > cpu_limit_s:
                    kill_process_tree(process)
                    cpu_exceeded_flag.set()
                    return
            except psutil.NoSuchProcess:
                return
            time.sleep(check_interval_s)
    except Exception:
        return


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class Evaluator:
    """Evaluates programs by executing them in a controlled environment.

    This class provides functionality to execute programs with resource limits
    (time and memory), capture their output and errors, and extract evaluation
    metrics from the results. Programs are executed in isolated temporary
    directories to prevent interference.
    """

    def __init__(
        self,
        eval_path: Path | str,
        cwd: Optional[Path | str],
        timeout_s: int,
        max_mem_b: Optional[int],
        resource_check_interval_s: Optional[float],
        logger: Optional[logging.Logger] = None,
    ):
        """Initializes the evaluator with execution parameters and resource limits.

        Args:
            eval_path: Path to the evaluation script that will execute the programs.
            cwd: Working directory for program execution. If provided, it will be
                copied to a temporary directory for isolated execution.
            timeout_s: Maximum execution time in seconds before killing the process.
            max_mem_b: Maximum memory usage in bytes. If None, no memory limit is enforced.
            resource_check_interval_s: Polling interval in seconds for the resource monitor
                thread (memory and CPU). Defaults to 0.1. Required when max_mem_b is specified.
            logger: Logger instance for logging evaluation activities. If None, creates
                a default logger.

        Raises:
            ValueError: If timeout_s is not positive, or if max_mem_b is specified but
                resource_check_interval_s is None or not positive.
        """
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")

        if max_mem_b is not None:
            if max_mem_b <= 0:
                raise ValueError("max_mem_b must be positive if specified")
            if resource_check_interval_s is None or resource_check_interval_s <= 0:
                raise ValueError(
                    "resource_check_interval_s must be positive when max_mem_b is specified"
                )

        self.eval_path: Path = Path(eval_path)
        self.cwd: Optional[Path] = Path(cwd) if cwd is not None else None
        self.timeout_s: int = timeout_s
        self.max_mem_b: Optional[int] = max_mem_b
        self.resource_check_interval_s: Optional[float] = resource_check_interval_s
        self.logger: logging.Logger = logger if logger is not None else logging.getLogger(__name__)

    def __repr__(self):
        """Returns a string representation of the Evaluator instance.

        Returns:
            A formatted string showing the evaluator's configuration including
            eval path, working directory, timeout, and memory limits.
        """
        return (
            f"{self.__class__.__name__}"
            "("
            f"eval_path={self.eval_path},"
            f"cwd={self.cwd},"
            f"timeout_s={self.timeout_s},"
            f"max_mem_b={self.max_mem_b},"
            f"resource_check_interval_s={self.resource_check_interval_s}"
            ")"
        )

    def execute(
        self, prog: Program, timeout_s: Optional[int] = None
    ) -> Tuple[int, Optional[str], Optional[str], Optional[str], Dict[str, Any]]:
        """Executes a program and updates it with execution results and metrics.

        This method creates temporary files for the program code, executes it using
        the evaluation script with resource monitoring, and returns the execution results
        including return code, errors, and evaluation metrics.

        The execution happens in an isolated temporary directory. If a working directory
        is configured, it is copied to a temporary location to prevent modifications
        to the original.

        Args:
            prog: Program object containing the code to execute. This object will be
                modified in-place with execution results including returncode, error
                messages, and evaluation metrics.
            timeout_s: Optional timeout override in seconds.  When provided,
                this value is used instead of ``self.timeout_s``.

        Returns:
            returncode: Exit code of the program (0 for success)
            output: String with stdout
            warning: String with warning
            error: String with stderr
            eval_metrics: Dictionary of evaluation metrics if successful
        """
        effective_timeout: int = timeout_s if timeout_s is not None else self.timeout_s
        self.logger.info(f"Attempting to evaluate program (timeout={effective_timeout}s)...")

        extension: str = LANGUAGE_TO_EXTENSION.get(prog.language, DEFAULT_EXTENSION)
        returncode: int = 1
        output: Optional[str] = None
        error: Optional[str] = None
        warning: Optional[str] = None
        eval_metrics: Dict[str, float] = {}

        process: Optional[subprocess.Popen] = None
        ps_process: Optional[psutil.Process] = None
        monitor_daemon: Optional[threading.Thread] = None
        kill_flag: threading.Event = threading.Event()
        mem_exceeded_flag: threading.Event = threading.Event()
        cpu_exceeded_flag: threading.Event = threading.Event()

        tmp_dir: Optional[tempfile.TemporaryDirectory] = None
        temp_cwd_dir: Optional[tempfile.TemporaryDirectory] = None
        temp_cwd: Optional[Path] = None

        try:
            # we copy cwd to temp and pass this temp directory as
            # the cwd for the program being executed
            tmp_dir = tempfile.TemporaryDirectory()

            if self.cwd:
                temp_cwd_dir = tempfile.TemporaryDirectory()
                temp_cwd = Path(temp_cwd_dir.name)
                shutil.copytree(self.cwd, temp_cwd, dirs_exist_ok=True)

            with tempfile.NamedTemporaryFile(
                mode="w", delete=False, suffix=extension, dir=tmp_dir.name
            ) as code_file:
                code_file.write(prog.code)
                code_file.flush()
                code_file_path: str = code_file.name

            with tempfile.NamedTemporaryFile(
                mode="w", delete=False, suffix=".json", dir=tmp_dir.name
            ) as results_file:
                result_file_path: str = results_file.name

            # resolve eval_path against temp_cwd when absolute so that sys.path[0]
            # inside the subprocess points to the isolated copy, not the original cwd.
            effective_eval_path: Path = self.eval_path
            if temp_cwd is not None and self.eval_path.is_absolute() and self.cwd is not None:
                try:
                    effective_eval_path = temp_cwd / self.eval_path.relative_to(self.cwd)
                except ValueError:
                    pass

            # launch evaluation subprocess
            process = subprocess.Popen(
                [sys.executable, str(effective_eval_path), code_file_path, result_file_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=1,
                text=True,
                cwd=str(temp_cwd) if temp_cwd else None,
            )

            ps_process = psutil.Process(process.pid)

            monitor_daemon = threading.Thread(
                target=resource_monitor,
                kwargs=dict(
                    process=ps_process,
                    check_interval_s=self.resource_check_interval_s
                    or DEFAULT_RESOURCE_CHECK_INTERVAL_S,
                    kill_flag=kill_flag,
                    mem_exceeded_flag=mem_exceeded_flag,
                    cpu_exceeded_flag=cpu_exceeded_flag,
                    max_mem_b=self.max_mem_b,
                    cpu_limit_s=float(effective_timeout),
                ),
                daemon=True,
            )
            monitor_daemon.start()

            try:
                stdout, stderr = process.communicate(timeout=effective_timeout)
                kill_flag.set()
                if monitor_daemon is not None:
                    monitor_daemon.join(timeout=1)

                output = stdout

                if mem_exceeded_flag.is_set():
                    error = f"MemoryExceededError: Evaluation memory usage exceeded maximum limit of {self.max_mem_b} bytes."
                    returncode = 1
                elif cpu_exceeded_flag.is_set():
                    error = f"CPUTimeExceededError: Evaluation CPU time usage exceeded maximum limit of {effective_timeout} seconds."
                    returncode = 1
                elif process.returncode == 0:
                    try:
                        with open(result_file_path, "r") as f:
                            eval_metrics = json.load(f)
                        warning = stderr
                        returncode = 0
                    except (json.JSONDecodeError, FileNotFoundError) as err:
                        error = f"Failed to load evaluation metrics: {err}"
                        returncode = 1
                else:
                    returncode = process.returncode
                    error = stderr if stderr else f"Process exited with code {returncode}"

            except subprocess.TimeoutExpired:
                kill_flag.set()
                if ps_process:
                    kill_process_tree(ps_process)
                try:
                    process.communicate(timeout=1)
                except Exception:
                    pass
                if monitor_daemon is not None:
                    monitor_daemon.join(timeout=1)
                error = f"TimeoutError: Evaluation time usage exceeded maximum time limit of {effective_timeout} seconds."

        except Exception as err:
            self.logger.error(f"Unexpected error during evaluation: {err}")
            error = f"EvaluationError: {str(err)}"

        finally:
            kill_flag.set()

            if process is not None and process.poll() is None:
                if ps_process is not None:
                    kill_process_tree(ps_process)
                else:
                    process.kill()

            if monitor_daemon is not None:
                monitor_daemon.join(timeout=1)

            if tmp_dir is not None:
                try:
                    tmp_dir.cleanup()
                except Exception as err:
                    self.logger.warning(f"Failed to cleanup tmp_dir: {err}")

            if temp_cwd_dir is not None:
                try:
                    temp_cwd_dir.cleanup()
                except Exception as err:
                    self.logger.warning(f"Failed to cleanup temp_cwd_dir: {err}")

        if not error:
            self.logger.info("Evaluated program without errors.")
        else:
            self.logger.error(f"Error in evaluating program -> '{error}'.")

        return returncode, output, warning, error, eval_metrics
