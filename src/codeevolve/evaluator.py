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

from typing import Any, Optional, Dict, List, Tuple
import tempfile
import logging
import subprocess
import threading
import json
import time
import psutil
import pathlib
import shutil
import sys
from codeevolve.database import Program


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


def mem_monitor(
    process: psutil.Process,
    max_mem_b: int,
    mem_check_interval_s: float,
    kill_flag: threading.Event,
    mem_exceeded_flag: threading.Event,
) -> None:
    """Monitors memory usage of a process and kills it if it exceeds the limit.

    This function runs in a separate thread to continuously monitor the memory
    usage of a subprocess and its entire process tree. It terminates the process
    tree if total memory consumption exceeds the specified threshold.

    Args:
        process: The psutil Process object to monitor.
        max_mem_b: Maximum memory usage in bytes before killing the process.
        mem_check_interval_s: Time interval in seconds between memory checks.
        kill_flag: Event to signal when monitoring should stop.
        mem_exceeded_flag: Event to signal when memory limit is exceeded.
    """
    try:
        while not kill_flag.is_set():
            try:
                if not process.is_running():
                    return
                total_mem: int = 0
                processes: List[psutil.Process] = get_process_tree(process)
                for proc in processes:
                    try:
                        mem_info = proc.memory_info()
                        total_mem += mem_info.rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                if total_mem > max_mem_b:
                    kill_process_tree(process)
                    mem_exceeded_flag.set()
                    return
            except psutil.NoSuchProcess:
                return
            time.sleep(mem_check_interval_s)
    except Exception:
        return


class Evaluator:
    """Evaluates programs by executing them in a controlled environment.

    This class provides functionality to execute programs with resource limits
    (time and memory), capture their output and errors, and extract evaluation
    metrics from the results. Programs are executed in isolated temporary
    directories to prevent interference.
    """

    def __init__(
        self,
        eval_path: pathlib.Path | str,
        cwd: Optional[pathlib.Path | str],
        timeout_s: int,
        max_mem_b: Optional[int],
        mem_check_interval_s: Optional[float],
        logger: Optional[logging.Logger] = None,
    ):
        """Initializes the evaluator with execution parameters and resource limits.

        Args:
            eval_path: Path to the evaluation script that will execute the programs.
            cwd: Working directory for program execution. If provided, it will be
                copied to a temporary directory for isolated execution.
            timeout_s: Maximum execution time in seconds before killing the process.
            max_mem_b: Maximum memory usage in bytes. If None, no memory limit is enforced.
            mem_check_interval_s: Interval for memory usage checks in seconds. Defaults to 0.1.
                Only used if max_mem_b is specified.
            logger: Logger instance for logging evaluation activities. If None, creates
                a default logger.

        Raises:
            ValueError: If timeout_s is not positive, or if max_mem_b is specified but
                mem_check_interval_s is None or not positive.
        """
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")

        if max_mem_b is not None:
            if max_mem_b <= 0:
                raise ValueError("max_mem_b must be positive if specified")
            if mem_check_interval_s is None or mem_check_interval_s <= 0:
                raise ValueError(
                    "mem_check_interval_s must be positive when max_mem_b is specified"
                )

        self.eval_path: pathlib.Path = pathlib.Path(eval_path)
        self.cwd: Optional[pathlib.Path] = pathlib.Path(cwd) if cwd is not None else None
        self.timeout_s: int = timeout_s
        self.max_mem_b: Optional[int] = max_mem_b
        self.mem_check_interval_s: Optional[float] = mem_check_interval_s
        self.language2extension: Dict[str, str] = {
            "python": ".py",
            "javascript": ".js",
            "java": ".java",
            "cpp": ".cpp",
            "c": ".c",
            "csharp": ".cs",
            "go": ".go",
            "rust": ".rs",
            "typescript": ".ts",
            "php": ".php",
            "ruby": ".rb",
            "swift": ".swift",
            "kotlin": ".kt",
            "scala": ".scala",
            "r": ".r",
            "matlab": ".m",
            "shell": ".sh",
            "powershell": ".ps1",
            "sql": ".sql",
        }
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
            f"mem_check_interval_s={self.mem_check_interval_s}"
            ")"
        )

    def execute(
        self, prog: Program
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

        Returns:
            returncode: Exit code of the program (0 for success)
            output: String with stdout
            warning: String with warning
            error: String with stderr
            eval_metrics: Dictionary of evaluation metrics if successful
        """
        self.logger.info("Attempting to evaluate program...")

        extension: str = self.language2extension.get(prog.language, ".txt")
        returncode: int = 1
        output: Optional[str] = None
        error: Optional[str] = None
        warning: Optional[str] = None
        eval_metrics: Dict[str, float] = {}

        process: Optional[subprocess.Popen] = None
        ps_process: Optional[psutil.Process] = None
        mem_monitor_daemon: Optional[threading.Thread] = None
        kill_flag: threading.Event = threading.Event()
        mem_exceeded_flag: threading.Event = threading.Event()

        tmp_dir: Optional[tempfile.TemporaryDirectory] = None
        temp_cwd_dir: Optional[tempfile.TemporaryDirectory] = None
        temp_cwd: Optional[pathlib.Path] = None

        # we copy cwd to temp and pass this temp directory as
        # the cwd for the program being executed
        tmp_dir: tempfile.TemporaryDirectory = tempfile.TemporaryDirectory(delete=False)
        temp_cwd: Optional[tempfile.TemporaryDirectory] = None
        temp_cwd_dir: Optional[tempfile.TemporaryDirectory] = None

        try:
            tmp_dir = tempfile.TemporaryDirectory(delete=False)

            if self.cwd:
                temp_cwd_dir = tempfile.TemporaryDirectory(delete=False)
                temp_cwd = pathlib.Path(temp_cwd_dir.name)
                try:
                    shutil.copytree(self.cwd, temp_cwd, dirs_exist_ok=True)
                except Exception as err:
                    self.logger.warning(f"Failed to copy cwd directory: {err}. Using original cwd.")
                    temp_cwd = self.cwd
                    if temp_cwd_dir:
                        try:
                            temp_cwd_dir.cleanup()
                        except Exception:
                            pass
                        temp_cwd_dir = None

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

            # Launch evaluation subprocess
            process = subprocess.Popen(
                [sys.executable, str(self.eval_path), code_file_path, result_file_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=1,
                text=True,
                cwd=str(temp_cwd) if temp_cwd else None,
            )

            ps_process = psutil.Process(process.pid)

            if self.max_mem_b is not None:
                mem_monitor_daemon = threading.Thread(
                    target=mem_monitor,
                    args=(
                        ps_process,
                        self.max_mem_b,
                        self.mem_check_interval_s,
                        kill_flag,
                        mem_exceeded_flag,
                    ),
                    daemon=True,
                )
                mem_monitor_daemon.start()

            try:
                stdout, stderr = process.communicate(timeout=self.timeout_s)
                kill_flag.set()
                if mem_monitor_daemon is not None:
                    mem_monitor_daemon.join(timeout=1)

                output = stdout

                if mem_exceeded_flag.is_set():
                    error = f"MemoryExceededError: Evaluation memory usage exceeded maximum limit of {self.max_mem_b} bytes."
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
                error = f"TimeoutError: Evaluation time usage exceeded maximum time limit of {self.timeout_s} seconds."

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

            if mem_monitor_daemon is not None:
                mem_monitor_daemon.join(timeout=1)

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
            error_preview: str = error[:128] + "[...]" if len(error) > 128 else error
            self.logger.error(f"Error in evaluating program -> '{error_preview}'.")

        return returncode, output, warning, error, eval_metrics
