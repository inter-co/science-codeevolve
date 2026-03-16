# ===--------------------------------------------------------------------------------------===#
#
# Part of the CodeEvolve Project, under the Apache License v2.0.
# See https://github.com/inter-co/science-codeevolve/blob/main/LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0
#
# ===--------------------------------------------------------------------------------------===#
#
# This file implements distributed logging for CodeEvolve.
#
# ===--------------------------------------------------------------------------------------===#

import logging
import multiprocessing as mp
import os
import re
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, Optional

from codeevolve.islands.sync import GlobalSyncData
from codeevolve.utils.constants import ASCII_NAME, DEFAULT_MAX_LOG_MSG_SIZE, ISLAND_LOG_FILE

# ---------------------------------------------------------------------------
# Dashboard shutdown signalling
# ---------------------------------------------------------------------------


class ShutdownReason(Enum):
    """Reason why the CLI dashboard is being shut down."""

    FINISHED = auto()
    ERROR = auto()
    INTERRUPTED = auto()


@dataclass
class DashboardShutdown:
    """Sentinel message sent to the dashboard queue to trigger a clean shutdown.

    Replaces the bare ``None`` sentinel so the dashboard can render an
    appropriate final banner depending on *why* it is being stopped.

    Attributes:
        reason: The high-level cause of shutdown.
        error_msg: Human-readable error summary (populated for ERROR reason).
        crash_log_path: Path to the crash log file (populated for ERROR reason).
    """

    reason: ShutdownReason
    error_msg: Optional[str] = field(default=None)
    crash_log_path: Optional[str] = field(default=None)


# ---------------------------------------------------------------------------
# Time formatting utilities
# ---------------------------------------------------------------------------


def format_elapsed_time(seconds: float) -> str:
    """Formats elapsed time in seconds to a human-readable string.

    Args:
        seconds: Elapsed time in seconds.

    Returns:
        Formatted string like "1h 23m 45s" or "5m 30s" or "45s".
    """
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def get_elapsed_time(global_data: GlobalSyncData) -> float:
    """Calculates the total elapsed time including checkpoint offset.

    Args:
        global_data: Shared data structure containing timing information.

    Returns:
        Total elapsed time in seconds.
    """
    current_elapsed: float = time.time() - global_data.start_time.value
    return current_elapsed + global_data.elapsed_time_offset.value


# ---------------------------------------------------------------------------
# Formatters and handlers
# ---------------------------------------------------------------------------


class SizeLimitedFormatter(logging.Formatter):
    """Custom logging formatter that enforces a maximum message size.

    This formatter extends the standard logging.Formatter to automatically truncate
    log messages that exceed a specified character limit. Messages longer than the
    limit are cut off and marked with a truncation indicator to preserve log
    readability and prevent extremely long messages from cluttering output.

    The truncation is applied to the raw message content before standard formatting
    (timestamp, level, etc.) is added, ensuring that the size limit refers specifically
    to the user's message content rather than the entire formatted log entry.

    Attributes:
        max_msg_sz: Maximum allowed length for log message content in characters.
    """

    def __init__(
        self,
        fmt: Optional[str] = None,
        datefmt: Optional[str] = None,
        max_msg_sz: int = DEFAULT_MAX_LOG_MSG_SIZE,
    ) -> None:
        """Initialize the size-limited formatter.

        Args:
            fmt: Format string for log messages. If None, uses the default format.
            datefmt: Format string for date/time portion of log messages. If None,
                uses the default date format.
            max_msg_sz: Maximum allowed length for the core message content in
                characters. Messages exceeding this limit will be truncated with
                a "... [TRUNCATED]" suffix. Must be at least 20 characters to
                accommodate the truncation indicator.

        Raises:
            ValueError: If max_msg_sz is less than 15 characters.
        """
        if max_msg_sz < 15:
            raise ValueError(
                "max_msg_sz must be at least 15" "characters to accommodate truncation indicator"
            )

        super().__init__(fmt, datefmt)
        self.max_msg_sz: int = max_msg_sz

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record, truncating the message if it exceeds size limits.

        This method checks if the message content exceeds the configured maximum
        size. If so, it temporarily modifies the record's message to a truncated
        version, formats it using the parent formatter, then restores the original
        message to avoid side effects if the record is used elsewhere.

        Args:
            record: The LogRecord instance containing the message and metadata
                to be formatted.

        Returns:
            The formatted log message string, with the core message content
            truncated if it originally exceeded max_msg_sz characters.
        """
        message_content: str = record.getMessage()

        if len(message_content) > self.max_msg_sz:
            original_msg: str = record.msg

            truncate_length: int = self.max_msg_sz - 15
            truncated_msg: str = str(record.msg)[:truncate_length] + "... [TRUNCATED]"
            record.msg = truncated_msg

            formatted: str = super().format(record)

            record.msg = original_msg
            return formatted

        return super().format(record)


class QueueHandler(logging.Handler):
    """Custom logging handler that sends log records to a multiprocessing queue.

    This handler enables logging from multiple processes by putting formatted
    log messages into a shared queue that can be processed by a central logger.
    """

    def __init__(self, queue: mp.Queue):
        """Initializes the queue handler with a multiprocessing queue.

        Args:
            queue: Multiprocessing queue to send log messages to.
        """
        super().__init__()
        self.queue = queue

    def emit(self, record: logging.LogRecord) -> None:
        """Emits a log record by formatting it and putting it in the queue.

        Args:
            record: The LogRecord to be formatted and queued.
        """
        try:
            msg = self.format(record)
            self.queue.put(msg)
        except Exception:
            self.handleError(record)


# ---------------------------------------------------------------------------
# Logger factory
# ---------------------------------------------------------------------------


def get_logger(
    island_id: int = 0,
    logs_dir: Optional[Path] = None,
    time: int = 0,
    log_queue: Optional[mp.Queue] = None,
    max_msg_sz: int = DEFAULT_MAX_LOG_MSG_SIZE,
) -> logging.Logger:
    """Creates a logger instance for an island with file and optional queue handlers.
    This function sets up a logger that writes to both a file and optionally to
    a multiprocessing queue for centralized log collection. Each log message
    is prefixed with the island ID + start time of the run for identification.

    If no logs_dir is provided, the logger will only output to stdout.

    Args:
        island_id: Unique identifier for the island creating the logger.
        logs_dir: Directory where the log file will be created. If None, logs only to stdout.
        time: Int representing current time. Defaults to zero.
        log_queue: Optional multiprocessing queue for centralized logging.
        max_msg_sz: Maximum size for log messages in characters.
    Returns:
        Configured Logger instance for the island.
    """
    if logs_dir:
        sanitized_dir: str = str(logs_dir).replace("/", "_").replace("\\", "_")
        logger_name: str = f"logger_{sanitized_dir}"
    else:
        logger_name: str = f"logger_stdout_{island_id}"

    logger: logging.Logger = logging.getLogger(logger_name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        log_fmt_str: str = (
            f"[island {island_id}] %(asctime)s | %(levelname)s | %(process)d"
            f" | %(filename)s:%(lineno)d | %(message)s"
        )
        truncatingFormatter = SizeLimitedFormatter(log_fmt_str, max_msg_sz=max_msg_sz)
        fullFormatter = logging.Formatter(log_fmt_str)
        logger.propagate = False

        if log_queue:
            queue_handler: QueueHandler = QueueHandler(log_queue)
            queue_handler.setFormatter(truncatingFormatter)
            logger.addHandler(queue_handler)
        else:
            logStreamHandler: logging.StreamHandler = logging.StreamHandler()
            logStreamHandler.setFormatter(truncatingFormatter)
            logger.addHandler(logStreamHandler)

        if logs_dir:
            fh: logging.FileHandler = logging.FileHandler(
                logs_dir.joinpath(ISLAND_LOG_FILE.format(time=time)), mode="w"
            )
            fh.setLevel(logging.INFO)
            fh.setFormatter(fullFormatter)
            logger.addHandler(fh)

    return logger


# ---------------------------------------------------------------------------
# CLI dashboard
# ---------------------------------------------------------------------------


def _print_global_status(args: Dict[str, Any], global_data: GlobalSyncData) -> None:
    """Prints the global status of the CodeEvolve algorithm.

    Args:
        args: Dictionary containing command-line arguments and configuration.
        global_data: Shared data structure containing global algorithm state.
    """
    elapsed: float = get_elapsed_time(global_data)
    elapsed_str: str = format_elapsed_time(elapsed)
    print("=" * 100)
    print(ASCII_NAME)
    print("=" * 47 + " INFO " + "=" * 47)
    print(f"> INPT DIR = {args['inpt_dir']}")
    print(f"> CFG PATH = {args['cfg_path']}")
    print(f"> OUT DIR = {args['out_dir']}")
    print(f"> CPU COUNT = {global_data.cpu_count.value}")

    print("=" * 46 + " STATUS " + "=" * 46)
    print(f"> ELAPSED TIME = {elapsed_str}")
    print(f"> GLOBAL BEST SOLUTION = {global_data.best_sol}")
    print(f"> GLOBAL EARLY STOPPING COUNTER = {global_data.early_stop_counter.value}")


def cli_dashboard(
    args: Dict[str, Any],
    global_data: GlobalSyncData,
    queue: mp.Queue,
    num_islands: int,
    refresh_rate: float = 0.5,
    island_hist_len: int = 10,
) -> None:
    """Formats and displays real-time logs from multiple islands in a dashboard format.

    This function runs as a separate process to collect log messages from all islands
    and display them in a continuously updating console dashboard showing the status
    of each island and global progress.

    Shutdown is triggered by placing a :class:`DashboardShutdown` message on the
    queue.  The ``reason`` field controls which final banner is rendered:

    * ``FINISHED``    — normal algorithm completion
    * ``ERROR``       — an island crashed; ``error_msg`` and ``crash_log_path`` are shown
    * ``INTERRUPTED`` — the run was cancelled by the user or a signal

    Args:
        args: Dictionary containing command-line arguments and configuration.
        global_data: Shared data structure containing global algorithm state.
        queue: Multiprocessing queue containing log messages from all islands.
        num_islands: Total number of islands in the system.
        refresh_rate: Time in seconds between dashboard refreshes.
        island_hist_len: Maximum number of log messages to keep per island.
    """
    island_logs: Dict[int, deque] = {i: deque(maxlen=island_hist_len) for i in range(num_islands)}
    island_id_pattern = re.compile(r"\[island (\d+)\]")

    island_epochs: Dict[int, str] = {i: "Initializing..." for i in range(num_islands)}
    epoch_pattern = re.compile(r"========= EPOCH (\d+) =========")

    while True:
        while not queue.empty():
            message = queue.get_nowait()
            if isinstance(message, DashboardShutdown):
                os.system("cls" if os.name == "nt" else "clear")
                _print_global_status(args, global_data)
                if message.reason == ShutdownReason.FINISHED:
                    print("=" * 45 + " FINISHED " + "=" * 45)
                elif message.reason == ShutdownReason.ERROR:
                    print(f"\n{'=' * 46} ERROR {'=' * 47}")
                    if message.error_msg:
                        print(message.error_msg)
                    if message.crash_log_path:
                        print(f"See {message.crash_log_path} and island logs for details.")
                    print("=" * 100)
                elif message.reason == ShutdownReason.INTERRUPTED:
                    print("=" * 43 + " INTERRUPTED " + "=" * 44)
                return

            match = island_id_pattern.search(message)
            if match:
                island_id = int(match.group(1))

                epoch_match = epoch_pattern.search(message)
                if epoch_match:
                    epoch_num = epoch_match.group(1)
                    island_epochs[island_id] = epoch_num

                if island_id in island_logs:
                    clean_message = island_id_pattern.sub("", message).strip()
                    island_logs[island_id].append(clean_message)

        os.system("cls" if os.name == "nt" else "clear")

        _print_global_status(args, global_data)
        print("=" * 47 + " LOGS " + "=" * 47)
        for i in sorted(island_logs.keys()):
            current_epoch = island_epochs.get(i, "?")
            print(f"=== ISLAND {i} | EPOCH {current_epoch} ===")
            if island_logs[i] is not None:
                for msg in island_logs[i]:
                    print(f"  > {msg}")
            print("-" * 45)

        print("=" * 100)
        time.sleep(refresh_rate)
