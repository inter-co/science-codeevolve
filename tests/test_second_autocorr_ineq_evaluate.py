# ===--------------------------------------------------------------------------------------===#
#
# Part of the CodeEvolve Project, under the Apache License v2.0.
# See https://github.com/inter-co/science-codeevolve/blob/main/LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0
#
# ===--------------------------------------------------------------------------------------===#
#
# This file implements unit tests for the second autocorrelation inequality evaluator.
#
# ===--------------------------------------------------------------------------------------===#

import importlib.util
import json
from pathlib import Path

import pytest

EVALUATOR_PATH = (
    Path(__file__).resolve().parents[1]
    / "problems"
    / "alphaevolve_math_problems"
    / "autocorrelation_problems"
    / "second_autocorr_ineq"
    / "input"
    / "evaluate.py"
)


def _load_evaluator():
    """Loads the problem-specific evaluator module from its file path."""
    spec = importlib.util.spec_from_file_location("second_autocorr_ineq_evaluate", EVALUATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_program(tmp_path: Path, values: str) -> Path:
    """Writes a program whose construct_function returns the given values.

    The file name is unique per test because evaluate() imports it by module
    name and Python caches imported modules for the whole process.
    """
    program = tmp_path / f"{tmp_path.name}.py"
    program.write_text(
        "def construct_function():\n" f"    return [{values}]\n",
        encoding="utf-8",
    )
    return program


@pytest.mark.parametrize("value", ["float('nan')", "float('inf')", "float('-inf')"])
def test_rejects_non_finite_function_values(tmp_path: Path, value: str) -> None:
    """NaN and infinity must be rejected instead of producing a NaN result."""
    evaluator = _load_evaluator()
    program = _write_program(tmp_path, value)
    results = tmp_path / "results.json"

    with pytest.raises(ValueError, match="finite"):
        evaluator.evaluate(str(program), str(results))

    assert not results.exists()


def test_accepts_finite_function_values(tmp_path: Path) -> None:
    """A finite non-negative function still evaluates to a finite c2."""
    evaluator = _load_evaluator()
    program = _write_program(tmp_path, "1.0, 1.0, 1.0, 1.0")
    results = tmp_path / "results.json"

    evaluator.evaluate(str(program), str(results))

    payload = json.loads(results.read_text(encoding="utf-8"))
    # f = [1, 1, 1, 1]: autoconvolution [1, 2, 3, 4, 3, 2, 1] gives
    # l2^2 = 16/3, norm_1 = 2, norm_inf = 4, so c2 = (16/3) / 8 = 2/3.
    assert payload["c2"] == pytest.approx(2.0 / 3.0)
    assert payload["benchmark_ratio"] == pytest.approx((2.0 / 3.0) / 0.962)
