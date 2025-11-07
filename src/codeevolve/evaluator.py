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

from typing import Optional, Dict
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

# TODO: better sandboxing (e.g. firejail)


def mem_monitor(
    process: psutil.Process,
    max_mem_b: int,
    mem_check_interval_s: float,
    kill_flag: threading.Event,
    mem_exceeded_flag: threading.Event,
) -> None:
    """Monitors memory usage of a process and kills it if it exceeds the limit.

    This function runs in a separate thread to continuously monitor the memory
    usage of a subprocess and terminate it if memory consumption exceeds the
    specified threshold.

    Args:
        process: The psutil Process object to monitor.
        max_mem_b: Maximum memory usage in bytes before killing the process.
        mem_check_interval_s: Time interval in seconds between memory checks.
        kill_flag: Event to signal when monitoring should stop.
        mem_exceeded_flag: Event to signal when memory limit is exceeded.
    """
    try:
        while not kill_flag.is_set():
            if not process.is_running():
                return
            mem_info = process.memory_info()
            if mem_info.rss > max_mem_b:
                process.kill()
                mem_exceeded_flag.set()
                return
            time.sleep(mem_check_interval_s)
    except:
        return


class Evaluator:
    """Evaluates programs by executing them in a controlled environment.

    This class provides functionality to execute programs with resource limits
    (time and memory), capture their output and errors, and extract evaluation
    metrics from the results.
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
            mem_check_interval_s: Interval for memory usage checks in seconds.
            logger: Logger instance for logging evaluation activities.
        """
        self.eval_path: pathlib.Path | str = eval_path
        self.cwd: Optional[pathlib.Path | str] = cwd
        self.timeout_s: int = timeout_s
        self.max_mem_b: Optional[int] = max_mem_b
        self.mem_check_interval_s: Optional[float] = mem_check_interval_s
        self.language2extension = {
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

    def execute(self, prog: Program) -> None:
        """Executes a program and updates it with execution results and metrics.

        This method creates temporary files for the program code, executes it using
        the evaluation script with resource monitoring, and updates the Program object
        with the execution results including return code, errors, and evaluation metrics.

        Args:
            prog: Program object containing the code to execute. This object will be
                 modified in-place with execution results including returncode, error
                 messages, and evaluation metrics.
        """
        self.logger.info("Attempting to evaluate program...")
        extension: str = self.language2extension[prog.language]
        returncode: int = 1
        error: Optional[str] = None
        warning: Optional[str] = None
        eval_metrics: Dict[str, float] = {}

        # we copy cwd to temp and pass this temp directory as
        # the cwd for the program being executed
        tmp_dir: tempfile.TemporaryDirectory = tempfile.TemporaryDirectory(delete=False)
        temp_cwd: Optional[tempfile.TemporaryDirectory] = None
        temp_cwd_dir: Optional[tempfile.TemporaryDirectory] = None

        if self.cwd:
            temp_cwd_dir = tempfile.TemporaryDirectory(delete=False)
            temp_cwd = temp_cwd_dir.name
            try:
                shutil.copytree(self.cwd, temp_cwd, dirs_exist_ok=True)
            except Exception as e:
                self.logger.warning(f"Failed to copy cwd directory: {e}. Using original cwd.")
                temp_cwd = self.cwd
                if temp_cwd_dir:
                    try:
                        temp_cwd_dir.cleanup()
                    except:
                        pass
                    temp_cwd_dir = None

        try:
            with tempfile.NamedTemporaryFile(
                mode="w", delete=False, suffix=extension, dir=tmp_dir.name
            ) as code_file:
                code_file.write(prog.code)
                code_file.flush()
                code_file_path = code_file.name
            with tempfile.NamedTemporaryFile(
                mode="w", delete=False, suffix=".json", dir=tmp_dir.name
            ) as results_file:
                result_file_path = results_file.name

            # run evaluate.py in subprocess using the temporary cwd copy
            process = subprocess.Popen(
                [sys.executable, self.eval_path, code_file_path, result_file_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=1,
                text=True,
                cwd=temp_cwd,
            )

            # memory monitor daemon if desired
            ps_process = psutil.Process(process.pid)
            kill_flag = threading.Event()
            mem_exceeded_flag = threading.Event()
            if self.max_mem_b:
                mem_monitor_daemon = threading.Thread(
                    target=mem_monitor,
                    args=(
                        ps_process,
                        self.max_mem_b,
                        self.mem_check_interval_s,
                        kill_flag,
                        mem_exceeded_flag,
                    ),
                )
                mem_monitor_daemon.daemon = True
                mem_monitor_daemon.start()

            # run program as subprocess
            stdout: Optional[str] = None
            stderr: Optional[str] = None
            try:
                stdout, stderr = process.communicate(timeout=self.timeout_s)
                returncode = process.returncode
                kill_flag.set()
                if not mem_exceeded_flag.is_set():
                    if returncode == 0:
                        with open(result_file_path, "r") as f:
                            eval_metrics: Dict[str, float] = json.load(f)
                        if len(stderr):
                            warning = stderr
                    else:
                        error = stderr
                else:
                    error = (
                        "MemoryExceededError: Evaluation memory usage exceeded maximum"
                        f" limit of {self.max_mem_b} bytes."
                    )
            except subprocess.TimeoutExpired:
                kill_flag.set()
                process.kill()
                error = (
                    "TimeoutError: Evaluation time usage exceeded maximum"
                    f" time limit of {self.timeout_s} seconds."
                )

        finally:
            try:
                tmp_dir.cleanup()
            except:
                pass
            if temp_cwd_dir:
                try:
                    temp_cwd_dir.cleanup()
                except:
                    pass

        if not error:
            self.logger.info(f"Evaluated program without errors.")
        else:
            self.logger.error(f"Error in evaluating program -> '{error[:128]}[...]'.")
        prog.returncode = returncode
        prog.error = error
        prog.eval_metrics = eval_metrics

        # TODO: figure a good way of using stdout and warning, they might be really big
        # prog.output = stdout
        # prog.warning = warning
