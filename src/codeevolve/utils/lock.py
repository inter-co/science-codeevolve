# ===--------------------------------------------------------------------------------------===#
#
# Part of the CodeEvolve Project, under the Apache License v2.0.
# See https://github.com/inter-co/science-codeevolve/blob/main/LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0
#
# ===--------------------------------------------------------------------------------------===#
#
# This file implements directory locking to prevent multiple CodeEvolve runs on the same
# output directory.
#
# ===--------------------------------------------------------------------------------------===#

import atexit
import datetime
import fcntl
import os
import socket
import sys
from pathlib import Path
from typing import Optional

from codeevolve.utils.constants import LOCK_FILE


class DirectoryLock:
    """Manages an exclusive lock on an output directory to prevent concurrent runs.

    This uses file-based locking (fcntl.flock) to ensure only one CodeEvolve instance
    can write to a given output directory at a time. The lock is automatically released
    when the process exits, even if it crashes.

    Attributes:
        out_dir: Path to the output directory being locked.
        lock_file_path: Path to the lock file (.codeevolve.lock).
        lock_file: Open file object holding the lock.
        locked: Whether the lock is currently held.
    """

    def __init__(self, out_dir: Path) -> None:
        """Initialize the directory lock.

        Args:
            out_dir: Path to the output directory to lock.
        """
        self.out_dir: Path = out_dir
        self.lock_file_path: Path = out_dir.joinpath(LOCK_FILE)
        self.lock_file: Optional[object] = None
        self.locked: bool = False

    def acquire(self) -> bool:
        """Attempt to acquire an exclusive lock on the output directory.

        Returns:
            True if lock was successfully acquired, False if directory is already locked.
        """
        try:
            self.lock_file = open(self.lock_file_path, "w")

            try:
                fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.locked = True

                pid: int = os.getpid()
                hostname: str = socket.gethostname()
                timestamp: str = datetime.datetime.now().isoformat()

                self.lock_file.write(f"PID: {pid}\n")
                self.lock_file.write(f"Host: {hostname}\n")
                self.lock_file.write(f"Started: {timestamp}\n")
                self.lock_file.flush()

                atexit.register(self.release)

                return True

            except BlockingIOError:
                self.lock_file.close()
                self.lock_file = None
                return False

        except Exception as e:
            print(f"Warning: Could not acquire directory lock: {e}", file=sys.stderr)
            return True

    def release(self) -> None:
        """Release the lock and clean up lock file."""
        if not self.locked:
            return

        try:
            if self.lock_file:
                try:
                    fcntl.flock(self.lock_file.fileno(), fcntl.LOCK_UN)
                except (AttributeError, OSError):
                    pass

                self.lock_file.close()
                self.lock_file = None

            if self.lock_file_path.exists():
                self.lock_file_path.unlink()

            self.locked = False
        except Exception as e:
            print(f"Warning: Error releasing lock: {e}", file=sys.stderr)

    def get_lock_info(self) -> Optional[str]:
        """Get information about who currently holds the lock.

        Returns:
            String with lock holder information, or None if lock file can't be read.
        """
        try:
            if self.lock_file_path.exists():
                with open(self.lock_file_path, "r") as f:
                    content: str = f.read().strip()
                    return content
        except Exception:
            pass
        return None

    def is_stale(self) -> bool:
        """Check if lock file is stale (process no longer running).

        This checks if the PID in the lock file still corresponds to a running process.

        Returns:
            True if lock is stale (can be safely removed), False otherwise.
        """
        try:
            if not self.lock_file_path.exists():
                return False

            with open(self.lock_file_path, "r") as f:
                first_line: str = f.readline()
                if first_line.startswith("PID: "):
                    pid_str: str = first_line.split(": ")[1].strip()
                    pid: int = int(pid_str)

                    try:
                        os.kill(pid, 0)
                        return False
                    except OSError:
                        return True

            return False
        except Exception:
            return False


def check_directory_lock(lock: DirectoryLock) -> None:
    """Check if output directory is locked by another CodeEvolve instance.

    This function attempts to acquire a lock on the output directory. If the lock
    cannot be acquired (because another instance is using it), the program exits
    with an error message. If successful, it returns the lock object which should
    be kept alive for the duration of the program.

    Args:
        out_dir: Output directory path to check and lock.

    Raises:
        SystemExit: If directory is locked by another instance.
    """
    if not lock.acquire():
        lock_info: Optional[str] = lock.get_lock_info()
        error_msg: str = (
            f"\n{'='*70}\n"
            f"ERROR: Output directory is already in use by another CodeEvolve instance\n"
            f"{'='*70}\n"
            f"Output directory: {lock.out_dir}\n"
        )

        if lock_info:
            error_msg += f"\nLock held by:\n{lock_info}\n"

        error_msg += (
            f"\nThis prevents data corruption from concurrent writes.\n"
            f"Please either:\n"
            f"  1. Wait for the other instance to complete\n"
            f"  2. Use a different output directory\n"
            f"  3. Stop the other instance if it's no longer needed\n"
            f"\nIf you're certain no other instance is running, you can manually\n"
            f"remove the lock file: {lock.out_dir / '.codeevolve.lock'}\n"
            f"{'='*70}\n"
        )

        print(error_msg, file=sys.stderr)
        sys.exit(1)
