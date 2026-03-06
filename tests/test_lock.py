# ===--------------------------------------------------------------------------------------===#
#
# Part of the CodeEvolve Project, under the Apache License v2.0.
# See https://github.com/inter-co/science-codeevolve/blob/main/LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0
#
# ===--------------------------------------------------------------------------------------===#
#
# This file implements unit tests for the directory locking mechanism.
#
# ===--------------------------------------------------------------------------------------===#

from pathlib import Path
from typing import Optional

import pytest

from codeevolve.utils.lock import DirectoryLock

# ---------------------------------------------------------------------------
# DirectoryLock
# ---------------------------------------------------------------------------


class TestDirectoryLock:
    """Test suite for the DirectoryLock class."""

    def test_creation(self, tmp_path: Path):
        """Tests that a DirectoryLock can be created."""
        lock: DirectoryLock = DirectoryLock(tmp_path)
        assert lock.out_dir == tmp_path
        assert lock.locked is False
        assert lock.lock_file is None

    def test_acquire_and_release(self, tmp_path: Path):
        """Tests acquire/release lifecycle."""
        lock: DirectoryLock = DirectoryLock(tmp_path)
        acquired: bool = lock.acquire()
        assert acquired is True
        assert lock.locked is True
        assert lock.lock_file_path.exists()

        lock.release()
        assert lock.locked is False
        assert not lock.lock_file_path.exists()

    def test_double_release_is_safe(self, tmp_path: Path):
        """Tests that releasing twice does not raise."""
        lock: DirectoryLock = DirectoryLock(tmp_path)
        lock.acquire()
        lock.release()
        lock.release()
        assert lock.locked is False

    def test_get_lock_info(self, tmp_path: Path):
        """Tests that lock info contains PID and hostname after acquisition."""
        lock: DirectoryLock = DirectoryLock(tmp_path)
        lock.acquire()
        info: Optional[str] = lock.get_lock_info()
        assert info is not None
        assert "PID:" in info
        assert "Host:" in info
        lock.release()

    def test_get_lock_info_no_lock(self, tmp_path: Path):
        """Tests that get_lock_info returns None when no lock file exists."""
        lock: DirectoryLock = DirectoryLock(tmp_path)
        info: Optional[str] = lock.get_lock_info()
        assert info is None

    def test_is_stale_no_lock_file(self, tmp_path: Path):
        """Tests that is_stale returns False when no lock file exists."""
        lock: DirectoryLock = DirectoryLock(tmp_path)
        assert lock.is_stale() is False

    def test_is_stale_with_active_process(self, tmp_path: Path):
        """Tests that is_stale returns False for an active process lock."""
        lock: DirectoryLock = DirectoryLock(tmp_path)
        lock.acquire()
        assert lock.is_stale() is False
        lock.release()

    def test_concurrent_lock_fails(self, tmp_path: Path):
        """Tests that a second lock on the same directory fails."""
        lock1: DirectoryLock = DirectoryLock(tmp_path)
        lock2: DirectoryLock = DirectoryLock(tmp_path)
        assert lock1.acquire() is True
        assert lock2.acquire() is False
        lock1.release()
