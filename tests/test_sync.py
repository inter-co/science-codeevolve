# ===--------------------------------------------------------------------------------------===#
#
# Part of the CodeEvolve Project, under the Apache License v2.0.
# See https://github.com/inter-co/science-codeevolve/blob/main/LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0
#
# ===--------------------------------------------------------------------------------------===#
#
# This file implements unit tests for the synchronization structures.
#
# ===--------------------------------------------------------------------------------------===#

from typing import Any, Dict

import pytest

from codeevolve.database import Program
from codeevolve.islands.sync import GlobalBestProg


# ---------------------------------------------------------------------------
# GlobalBestProg
# ---------------------------------------------------------------------------


class TestGlobalBestProg:
    """Test suite for the GlobalBestProg shared-memory tracker."""

    def test_creation_has_expected_attributes(self):
        """Tests that GlobalBestProg has the expected shared-memory attributes."""
        best: GlobalBestProg = GlobalBestProg()
        assert hasattr(best, "fitness")
        assert hasattr(best, "iteration_found")
        assert hasattr(best, "island_found")
        assert hasattr(best, "depth")
        assert hasattr(best, "eval_metrics")

    def test_update_from_program(self):
        """Tests updating the global best from a Program instance."""
        best: GlobalBestProg = GlobalBestProg()
        prog: Program = Program(
            id="p1",
            code="def f(): return 1",
            language="python",
            fitness=42.0,
            iteration_found=10,
            island_found=2,
            depth=5,
            eval_metrics={"accuracy": 0.95},
        )
        best.update_from_program(prog)
        assert best.fitness.value == 42.0
        assert best.iteration_found.value == 10
        assert best.island_found.value == 2
        assert best.depth.value == 5
        assert dict(best.eval_metrics) == {"accuracy": 0.95}

    def test_update_overwrites_previous(self):
        """Tests that update_from_program fully overwrites previous state."""
        best: GlobalBestProg = GlobalBestProg()

        prog1: Program = Program(
            id="p1",
            code="",
            language="python",
            fitness=10.0,
            iteration_found=1,
            island_found=0,
            depth=1,
            eval_metrics={"a": 1.0, "b": 2.0},
        )
        best.update_from_program(prog1)

        prog2: Program = Program(
            id="p2",
            code="",
            language="python",
            fitness=20.0,
            iteration_found=5,
            island_found=1,
            depth=3,
            eval_metrics={"c": 3.0},
        )
        best.update_from_program(prog2)

        assert best.fitness.value == 20.0
        assert best.iteration_found.value == 5
        assert "c" in best.eval_metrics
        assert "a" not in best.eval_metrics

    def test_from_dict(self):
        """Tests initialization from a dictionary."""
        best: GlobalBestProg = GlobalBestProg()
        data: Dict[str, Any] = {
            "fitness": 99.0,
            "iteration_found": 50,
            "island_found": 3,
            "depth": 10,
            "eval_metrics": {"loss": 0.01},
        }
        best.from_dict(data)
        assert best.fitness.value == 99.0
        assert best.iteration_found.value == 50
        assert best.island_found.value == 3
        assert best.depth.value == 10

    def test_from_dict_partial(self):
        """Tests that from_dict with missing keys keeps defaults."""
        best: GlobalBestProg = GlobalBestProg()
        original_fitness: float = best.fitness.value
        best.from_dict({"island_found": 7})
        assert best.island_found.value == 7
        assert best.fitness.value == original_fitness