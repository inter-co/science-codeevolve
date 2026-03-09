# ===--------------------------------------------------------------------------------------===#
#
# Part of the CodeEvolve Project, under the Apache License v2.0.
# See https://github.com/inter-co/science-codeevolve/blob/main/LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0
#
# ===--------------------------------------------------------------------------------------===#
#
# This file implements unit tests for the Evaluator class.
#
# ===--------------------------------------------------------------------------------------===#

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pytest

from codeevolve.database import Program
from codeevolve.evaluator import Evaluator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_eval_script(tmpdir: Path, script_content: str) -> Path:
    """Creates an evaluation script file in the given directory."""
    eval_path: Path = tmpdir / "evaluate.py"
    eval_path.write_text(script_content)
    return eval_path


def _make_program(code: str = "print('hello')", language: str = "python") -> Program:
    """Creates a minimal Program for testing."""
    return Program(id="test_prog", code=code, language=language)


# ---------------------------------------------------------------------------
# Evaluator init
# ---------------------------------------------------------------------------


class TestEvaluatorInit:
    """Test suite for the Evaluator constructor validation."""

    def test_creation_valid(self, tmp_path: Path):
        """Tests that Evaluator can be created with valid parameters."""
        eval_path: Path = _make_eval_script(tmp_path, "pass")
        evaluator: Evaluator = Evaluator(
            eval_path=eval_path,
            cwd=tmp_path,
            timeout_s=10,
            max_mem_b=1024 * 1024,
            resource_check_interval_s=0.1,
        )
        assert evaluator.timeout_s == 10
        assert evaluator.max_mem_b == 1024 * 1024

    def test_invalid_timeout(self, tmp_path: Path):
        """Tests that non-positive timeout raises ValueError."""
        with pytest.raises(ValueError, match="timeout_s must be positive"):
            Evaluator(
                eval_path="eval.py",
                cwd=tmp_path,
                timeout_s=0,
                max_mem_b=None,
                resource_check_interval_s=None,
            )

    def test_invalid_max_mem(self, tmp_path: Path):
        """Tests that non-positive max_mem_b raises ValueError."""
        with pytest.raises(ValueError, match="max_mem_b must be positive"):
            Evaluator(
                eval_path="eval.py",
                cwd=tmp_path,
                timeout_s=10,
                max_mem_b=-1,
                resource_check_interval_s=0.1,
            )

    def test_invalid_resource_check_interval(self, tmp_path: Path):
        """Tests that invalid resource_check_interval_s raises ValueError."""
        with pytest.raises(ValueError, match="resource_check_interval_s must be positive"):
            Evaluator(
                eval_path="eval.py",
                cwd=tmp_path,
                timeout_s=10,
                max_mem_b=1024,
                resource_check_interval_s=0,
            )

    def test_no_mem_limit(self, tmp_path: Path):
        """Tests that Evaluator can be created without memory limit."""
        evaluator: Evaluator = Evaluator(
            eval_path="eval.py",
            cwd=tmp_path,
            timeout_s=10,
            max_mem_b=None,
            resource_check_interval_s=None,
        )
        assert evaluator.max_mem_b is None


# ---------------------------------------------------------------------------
# Evaluator execution
# ---------------------------------------------------------------------------


class TestEvaluatorExecution:
    """Test suite for the Evaluator.execute method."""

    def test_successful_evaluation(self, tmp_path: Path):
        """Tests that a correct program is evaluated successfully."""
        eval_script: str = """
import sys, json
code_path = sys.argv[1]
results_path = sys.argv[2]
with open(results_path, 'w') as f:
    json.dump({"fitness": 42.0}, f)
"""
        eval_path: Path = _make_eval_script(tmp_path, eval_script)
        evaluator: Evaluator = Evaluator(
            eval_path=eval_path,
            cwd=tmp_path,
            timeout_s=30,
            max_mem_b=None,
            resource_check_interval_s=None,
        )

        prog: Program = _make_program("x = 1")
        returncode: int
        output: Optional[str]
        warning: Optional[str]
        error: Optional[str]
        eval_metrics: Dict[str, Any]
        returncode, output, warning, error, eval_metrics = evaluator.execute(prog)

        assert returncode == 0
        assert eval_metrics["fitness"] == 42.0
        assert error is None or error == ""

    def test_evaluation_with_error(self, tmp_path: Path):
        """Tests that a failing evaluation script returns non-zero returncode."""
        eval_script: str = """
import sys
sys.exit(1)
"""
        eval_path: Path = _make_eval_script(tmp_path, eval_script)
        evaluator: Evaluator = Evaluator(
            eval_path=eval_path,
            cwd=tmp_path,
            timeout_s=30,
            max_mem_b=None,
            resource_check_interval_s=None,
        )

        prog: Program = _make_program("x = 1")
        returncode, _, _, error, eval_metrics = evaluator.execute(prog)
        assert returncode != 0
        assert eval_metrics == {}

    def test_evaluation_timeout(self, tmp_path: Path):
        """Tests that long-running evaluations are terminated with a timeout error."""
        eval_script: str = """
import sys, time
time.sleep(60)
"""
        eval_path: Path = _make_eval_script(tmp_path, eval_script)
        evaluator: Evaluator = Evaluator(
            eval_path=eval_path,
            cwd=tmp_path,
            timeout_s=1,
            max_mem_b=None,
            resource_check_interval_s=None,
        )

        prog: Program = _make_program("x = 1")
        returncode, _, _, error, eval_metrics = evaluator.execute(prog)
        assert returncode == 1
        assert "TimeoutError" in error

    def test_evaluation_timeout_override(self, tmp_path: Path):
        """Tests that a timeout_s override on execute() is respected."""
        eval_script: str = """
import sys, time
time.sleep(60)
"""
        eval_path: Path = _make_eval_script(tmp_path, eval_script)
        evaluator: Evaluator = Evaluator(
            eval_path=eval_path,
            cwd=tmp_path,
            timeout_s=30,
            max_mem_b=None,
            resource_check_interval_s=None,
        )

        prog: Program = _make_program("x = 1")
        returncode, _, _, error, _ = evaluator.execute(prog, timeout_s=1)
        assert returncode == 1
        assert "TimeoutError" in error
        assert "1 seconds" in error

    def test_evaluation_invalid_json(self, tmp_path: Path):
        """Tests that invalid JSON in results causes an error."""
        eval_script: str = """
import sys
results_path = sys.argv[2]
with open(results_path, 'w') as f:
    f.write("not valid json")
"""
        eval_path: Path = _make_eval_script(tmp_path, eval_script)
        evaluator: Evaluator = Evaluator(
            eval_path=eval_path,
            cwd=tmp_path,
            timeout_s=30,
            max_mem_b=None,
            resource_check_interval_s=None,
        )

        prog: Program = _make_program("x = 1")
        returncode, _, _, error, eval_metrics = evaluator.execute(prog)
        assert returncode == 1
        assert "Failed to load evaluation metrics" in error

    def test_evaluation_cpu_exceeded(self, tmp_path: Path):
        """Tests that programs whose cumulative CPU time exceeds the limit are killed.

        Spawns several worker processes that each burn CPU in a tight loop so that
        their combined CPU time accumulates faster than wall-clock time.  With a
        generous wall-clock timeout (30 s) and a short CPU limit (timeout_s=3),
        the CPU budget should be exhausted well before the wall-clock guard fires.
        """
        eval_script: str = """
import sys
from multiprocessing import Process

def _burn():
    while True:
        pass

if __name__ == "__main__":
    workers = [Process(target=_burn) for _ in range(4)]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
"""
        eval_path: Path = _make_eval_script(tmp_path, eval_script)
        evaluator: Evaluator = Evaluator(
            eval_path=eval_path,
            cwd=tmp_path,
            timeout_s=3,
            max_mem_b=None,
            resource_check_interval_s=0.1,
        )

        prog: Program = _make_program("x = 1")
        returncode, _, _, error, _ = evaluator.execute(prog)
        assert returncode == 1
        assert error is not None
        assert "CPUTimeExceededError" in error or "TimeoutError" in error

    def test_evaluation_with_cwd_copy(self, tmp_path: Path):
        """Tests that evaluation copies the cwd to a temp directory."""
        cwd_dir: Path = tmp_path / "cwd"
        cwd_dir.mkdir()
        (cwd_dir / "helper.txt").write_text("helper data")

        eval_script: str = """
import sys, json, os
code_path = sys.argv[1]
results_path = sys.argv[2]
helper_exists = os.path.exists("helper.txt")
with open(results_path, 'w') as f:
    json.dump({"fitness": 1.0, "helper_exists": helper_exists}, f)
"""
        eval_path: Path = _make_eval_script(cwd_dir, eval_script)
        evaluator: Evaluator = Evaluator(
            eval_path=eval_path,
            cwd=cwd_dir,
            timeout_s=30,
            max_mem_b=None,
            resource_check_interval_s=None,
        )

        prog: Program = _make_program("x = 1")
        returncode, _, _, error, eval_metrics = evaluator.execute(prog)
        assert returncode == 0
        assert eval_metrics["helper_exists"] is True
