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

from typing import Any, Dict, Optional

import logging
import multiprocessing as mp
import time
from collections import deque
import re
import os
import pathlib

from codeevolve.islands import GlobalData

from typing import Optional
import logging


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
        self, fmt: Optional[str] = None, datefmt: Optional[str] = None, max_msg_sz: int = 256
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


def get_logger(
    island_id: int = 0,
    results_dir: Optional[pathlib.Path] = None,
    append_mode: bool = False,
    log_queue: Optional[mp.Queue] = None,
    max_msg_sz: int = 256,
) -> logging.Logger:
    """Creates a logger instance for an island with file and optional queue handlers.
    This function sets up a logger that writes to both a file and optionally to
    a multiprocessing queue for centralized log collection. Each log message
    is prefixed with the island ID for identification.

    If no results_dir is provided, the logger will only output to stdout.

    Args:
        island_id: Unique identifier for the island creating the logger.
        results_dir: Directory where the log file will be created. If None, logs only to stdout.
        append_mode: If True, append to existing log file; if False, overwrite.
        log_queue: Optional multiprocessing queue for centralized logging.
        max_msg_sz: Maximum size for log messages in characters.
    Returns:
        Configured Logger instance for the island.
    """
    if results_dir:
        sanitized_dir: str = str(results_dir).replace("/", "_").replace("\\", "_")
        logger_name: str = f"logger_{sanitized_dir}"
    else:
        logger_name: str = f"logger_stdout_{island_id}"

    logger: logging.Logger = logging.getLogger(logger_name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        logFormatter = SizeLimitedFormatter(
            f"[island {island_id}] %(asctime)s | %(levelname)s | %(process)d | %(message)s",
            max_msg_sz=max_msg_sz,
        )
        logger.propagate = False

        if log_queue:
            queue_handler: QueueHandler = QueueHandler(log_queue)
            queue_handler.setFormatter(logFormatter)
            logger.addHandler(queue_handler)
        else:
            logStreamHandler: logging.StreamHandler = logging.StreamHandler()
            logStreamHandler.setFormatter(logFormatter)
            logger.addHandler(logStreamHandler)

        if results_dir:
            fh: logging.FileHandler = logging.FileHandler(
                results_dir.joinpath("results.log"), mode="a" if append_mode else "w"
            )
            fh.setLevel(logging.INFO)
            fh.setFormatter(logFormatter)
            logger.addHandler(fh)

    return logger


def cli_logger(
    args: Dict[str, Any],
    global_data: GlobalData,
    queue: mp.Queue,
    num_islands: int,
    refresh_rate: float = 0.5,
    island_hist_len: int = 10,
) -> None:
    """Formats and displays real-time logs from multiple islands in a dashboard format.

    This function runs as a separate process to collect log messages from all islands
    and display them in a continuously updating console dashboard showing the status
    of each island and global progress.

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

    try:
        while True:
            while not queue.empty():
                message = queue.get_nowait()
                if message is None:
                    os.system("cls" if os.name == "nt" else "clear")
                    print("Program finished.")
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

            print("=" * 15 + " CODEEVOLVE STATUS " + "=" * 15)
            print(f"> INPT DIR = {args['inpt_dir']}")
            print(f"> CFG PATH = {args['cfg_path']}")
            print(f"> OUT DIR = {args['out_dir']}")
            print(f"> GLOBAL BEST SOLUTION = {global_data.best_sol}")
            print(f"> GLOBAL EARLY STOPPING COUNTER = {global_data.early_stop_counter.value}")
            for i in sorted(island_logs.keys()):
                current_epoch = island_epochs.get(i, "N/A")
                print(f"=== ISLAND {i} | EPOCH {current_epoch} ===")
                if not island_logs[i]:
                    print("(Waiting for messages...)")
                else:
                    for msg in island_logs[i]:
                        print(f"  > {msg}")
                print("-" * 45)

            time.sleep(refresh_rate)

    except (KeyboardInterrupt, ValueError):
        os.system("cls" if os.name == "nt" else "clear")
        print("\nProgram interrupted.")
