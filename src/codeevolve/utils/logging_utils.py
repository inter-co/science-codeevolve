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


def log_formatter(
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


def get_logger(
    island_id: int,
    results_dir: pathlib.Path,
    append_mode: bool,
    log_queue: Optional[mp.Queue] = None,
) -> logging.Logger:
    """Creates a logger instance for an island with file and optional queue handlers.

    This function sets up a logger that writes to both a file and optionally to
    a multiprocessing queue for centralized log collection. Each log message
    is prefixed with the island ID for identification.

    Args:
        island_id: Unique identifier for the island creating the logger.
        results_dir: Directory where the log file will be created.
        append_mode: If True, append to existing log file; if False, overwrite.
        log_queue: Optional multiprocessing queue for centralized logging.

    Returns:
        Configured Logger instance for the island.
    """

    sanitized_dir = str(results_dir).replace("/", "_").replace("\\", "_")
    logger_name = f"logger_{sanitized_dir}"
    logger = logging.getLogger(logger_name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)
        logFormatter = logging.Formatter(
            f"[island {island_id}] %(asctime)s | %(levelname)s | %(process)d | %(message)s"
        )
        logger.propagate = False

        if log_queue:
            queue_handler = QueueHandler(log_queue)
            queue_handler.setFormatter(logFormatter)
            logger.addHandler(queue_handler)
        else:
            logStreamHandler = logging.StreamHandler()
            logStreamHandler.setFormatter(logFormatter)
            logger.addHandler(logStreamHandler)

        fh = logging.FileHandler(
            results_dir.joinpath("results.log"), mode="a" if append_mode else "w"
        )
        fh.setLevel(logging.INFO)
        fh.setFormatter(logFormatter)
        logger.addHandler(fh)

    return logger
