# ===--------------------------------------------------------------------------------------===#
#
# Part of the CodeEvolve Project, under the Apache License v2.0.
# See https://github.com/inter-co/science-codeevolve/blob/main/LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0
#
# ===--------------------------------------------------------------------------------------===#
#
# This file implements unit tests for the logging utilities.
#
# ===--------------------------------------------------------------------------------------===#

import logging
import multiprocessing as mp
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import pytest

from codeevolve.utils.logging import (
    DashboardShutdown,
    ShutdownReason,
    SizeLimitedFormatter,
    cli_dashboard,
    format_elapsed_time,
    get_logger,
)

# ---------------------------------------------------------------------------
# format_elapsed_time
# ---------------------------------------------------------------------------


class TestFormatElapsedTime:
    """Test suite for the format_elapsed_time function."""

    def test_seconds_only(self):
        """Tests formatting for durations under a minute."""
        result: str = format_elapsed_time(45)
        assert result == "45s"

    def test_minutes_and_seconds(self):
        """Tests formatting for durations under an hour."""
        result: str = format_elapsed_time(330)
        assert result == "5m 30s"

    def test_hours_minutes_seconds(self):
        """Tests formatting for durations over an hour."""
        result: str = format_elapsed_time(5025)
        assert result == "1h 23m 45s"

    def test_zero_seconds(self):
        """Tests formatting for zero duration."""
        result: str = format_elapsed_time(0)
        assert result == "0s"

    def test_exact_hour(self):
        """Tests formatting for exactly one hour."""
        result: str = format_elapsed_time(3600)
        assert result == "1h 0m 0s"


# ---------------------------------------------------------------------------
# SizeLimitedFormatter
# ---------------------------------------------------------------------------


class TestSizeLimitedFormatter:
    """Test suite for the SizeLimitedFormatter class."""

    def test_creation(self):
        """Tests that formatter can be created with valid max_msg_sz."""
        fmt: SizeLimitedFormatter = SizeLimitedFormatter(max_msg_sz=100)
        assert fmt.max_msg_sz == 100

    def test_invalid_max_msg_sz(self):
        """Tests that too-small max_msg_sz raises ValueError."""
        with pytest.raises(ValueError):
            SizeLimitedFormatter(max_msg_sz=5)

    def test_short_message_not_truncated(self):
        """Tests that short messages are not truncated."""
        fmt: SizeLimitedFormatter = SizeLimitedFormatter(fmt="%(message)s", max_msg_sz=100)
        record: logging.LogRecord = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="short message",
            args=(),
            exc_info=None,
        )
        result: str = fmt.format(record)
        assert result == "short message"

    def test_long_message_truncated(self):
        """Tests that long messages are truncated with indicator."""
        fmt: SizeLimitedFormatter = SizeLimitedFormatter(fmt="%(message)s", max_msg_sz=30)
        long_msg: str = "A" * 100
        record: logging.LogRecord = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg=long_msg,
            args=(),
            exc_info=None,
        )
        result: str = fmt.format(record)
        assert len(result) < 100
        assert "[TRUNCATED]" in result

    def test_exact_limit_not_truncated(self):
        """Tests that messages at exactly the limit are not truncated."""
        fmt: SizeLimitedFormatter = SizeLimitedFormatter(fmt="%(message)s", max_msg_sz=20)
        msg: str = "A" * 20
        record: logging.LogRecord = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg=msg,
            args=(),
            exc_info=None,
        )
        result: str = fmt.format(record)
        assert result == msg


# ---------------------------------------------------------------------------
# get_logger
# ---------------------------------------------------------------------------


class TestGetLogger:
    """Test suite for the get_logger factory function."""

    def test_logger_without_results_dir(self):
        """Tests that a logger can be created without a results directory."""
        logger: logging.Logger = get_logger(island_id=99)
        assert logger is not None
        assert logger.level == logging.INFO

    def test_logger_with_logs_dir(self, tmp_path: Path):
        """Tests that a logger creates a log file in logs_dir."""
        logger: logging.Logger = get_logger(island_id=0, logs_dir=tmp_path, time=100)
        logger.info("test message")
        log_file: Path = tmp_path / "run_100.log"
        assert log_file.exists()

    def test_logger_unique_names(self, tmp_path: Path):
        """Tests that loggers for different directories have unique names."""
        dir1: Path = tmp_path / "island0"
        dir2: Path = tmp_path / "island1"
        dir1.mkdir()
        dir2.mkdir()

        logger1: logging.Logger = get_logger(island_id=0, logs_dir=dir1)
        logger2: logging.Logger = get_logger(island_id=1, logs_dir=dir2)
        assert logger1.name != logger2.name

    def test_file_handler_not_truncated(self, tmp_path: Path):
        """Tests that the file log handler does NOT truncate long messages."""
        import multiprocessing as mp

        long_msg: str = "X" * 2000
        queue: mp.Queue = mp.Queue()
        logs_dir: Path = tmp_path / "island_trunc_test"
        logs_dir.mkdir()
        logger: logging.Logger = get_logger(
            island_id=0,
            logs_dir=logs_dir,
            time=0,
            log_queue=queue,
            max_msg_sz=256,
        )
        logger.info(long_msg)

        log_file: Path = logs_dir / "run_0.log"
        assert log_file.exists()
        content: str = log_file.read_text()
        assert "[TRUNCATED]" not in content, "File log should not truncate messages"
        assert long_msg in content, "Full message should appear in the file log"

    def test_queue_handler_truncated(self, tmp_path: Path):
        """Tests that the queue log handler truncates long messages."""
        import multiprocessing as mp

        long_msg: str = "Y" * 2000
        queue: mp.Queue = mp.Queue()
        logs_dir: Path = tmp_path / "island_queue_test"
        logs_dir.mkdir()
        logger: logging.Logger = get_logger(
            island_id=0,
            logs_dir=logs_dir,
            time=0,
            log_queue=queue,
            max_msg_sz=256,
        )
        logger.info(long_msg)

        queued: str = queue.get_nowait()
        assert "[TRUNCATED]" in queued, "Queue log should truncate long messages"


# ---------------------------------------------------------------------------
# DashboardShutdown
# ---------------------------------------------------------------------------


class TestDashboardShutdown:
    """Test suite for the DashboardShutdown dataclass and ShutdownReason enum."""

    def test_finished_defaults(self):
        """Tests that FINISHED shutdown has None error fields by default."""
        shutdown: DashboardShutdown = DashboardShutdown(reason=ShutdownReason.FINISHED)
        assert shutdown.reason == ShutdownReason.FINISHED
        assert shutdown.error_msg is None
        assert shutdown.crash_log_path is None

    def test_interrupted_defaults(self):
        """Tests that INTERRUPTED shutdown has None error fields by default."""
        shutdown: DashboardShutdown = DashboardShutdown(reason=ShutdownReason.INTERRUPTED)
        assert shutdown.reason == ShutdownReason.INTERRUPTED
        assert shutdown.error_msg is None
        assert shutdown.crash_log_path is None

    def test_error_with_details(self):
        """Tests that ERROR shutdown carries error_msg and crash_log_path."""
        shutdown: DashboardShutdown = DashboardShutdown(
            reason=ShutdownReason.ERROR,
            error_msg="Island 1 died unexpectedly",
            crash_log_path="/out/crash_42.log",
        )
        assert shutdown.reason == ShutdownReason.ERROR
        assert shutdown.error_msg == "Island 1 died unexpectedly"
        assert shutdown.crash_log_path == "/out/crash_42.log"

    def test_error_partial_fields(self):
        """Tests that ERROR shutdown can be created with only some fields set."""
        shutdown: DashboardShutdown = DashboardShutdown(
            reason=ShutdownReason.ERROR,
            error_msg="crash",
        )
        assert shutdown.error_msg == "crash"
        assert shutdown.crash_log_path is None

    def test_all_reasons_are_distinct(self):
        """Tests that each ShutdownReason value is distinct."""
        reasons = {ShutdownReason.FINISHED, ShutdownReason.ERROR, ShutdownReason.INTERRUPTED}
        assert len(reasons) == 3
