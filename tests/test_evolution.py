# ===--------------------------------------------------------------------------------------===#
#
# Part of the CodeEvolve Project, under the Apache License v2.0.
# See https://github.com/inter-co/science-codeevolve/blob/main/LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0
#
# ===--------------------------------------------------------------------------------------===#
#
# This file implements unit tests for the evolution module helper functions.
#
# ===--------------------------------------------------------------------------------------===#

import logging
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock

import pytest

from codeevolve.database import Program, ProgramDatabase
from codeevolve.evolution import _get_markers, compute_dynamic_timeout, select_parents

# ---------------------------------------------------------------------------
# _get_markers
# ---------------------------------------------------------------------------


class TestGetMarkers:
    """Test suite for the _get_markers helper function."""

    def test_default_markers(self):
        """Tests that default markers are returned when not configured."""
        evolve_config: Dict[str, Any] = {}
        markers: Tuple[str, str, str, str] = _get_markers(evolve_config)
        assert markers[0] == "# EVOLVE-BLOCK-START"
        assert markers[1] == "# EVOLVE-BLOCK-END"
        assert markers[2] == "# PROMPT-BLOCK-START"
        assert markers[3] == "# PROMPT-BLOCK-END"

    def test_custom_markers(self):
        """Tests that custom markers override defaults."""
        evolve_config: Dict[str, Any] = {
            "evolve_start_marker": "// BEGIN",
            "evolve_end_marker": "// FINISH",
            "mp_start_marker": "/* PSTART */",
            "mp_end_marker": "/* PEND */",
        }
        markers: Tuple[str, str, str, str] = _get_markers(evolve_config)
        assert markers[0] == "// BEGIN"
        assert markers[1] == "// FINISH"
        assert markers[2] == "/* PSTART */"
        assert markers[3] == "/* PEND */"

    def test_partial_custom_markers(self):
        """Tests that unset markers fall back to defaults."""
        evolve_config: Dict[str, Any] = {
            "evolve_start_marker": "// BEGIN",
        }
        markers: Tuple[str, str, str, str] = _get_markers(evolve_config)
        assert markers[0] == "// BEGIN"
        assert markers[1] == "# EVOLVE-BLOCK-END"


# ---------------------------------------------------------------------------
# select_parents
# ---------------------------------------------------------------------------


class TestSelectParents:
    """Test suite for the select_parents function."""

    def _make_prog(self, id: str, fitness: float = 0.0) -> Program:
        """Helper to create a minimal evaluated program."""
        prog: Program = Program(
            id=id,
            code=f"# {id}",
            language="python",
            fitness=fitness,
            island_found=0,
            iteration_found=0,
            generation=0,
            returncode=0,
            eval_metrics={"fitness": fitness},
            features={"fitness": fitness},
        )
        prog.prog_msg = f"Program {id}"
        return prog

    def test_init_pop_returns_initial_programs(self):
        """Tests that during init population, initial programs are selected."""
        sol_db: ProgramDatabase = ProgramDatabase(id=0, seed=42)
        prompt_db: ProgramDatabase = ProgramDatabase(id=0, seed=42)

        init_sol: Program = self._make_prog("init_sol", fitness=1.0)
        init_prompt: Program = Program(id="init_prompt", code="prompt", language="text")

        sol_db.add(init_sol)
        prompt_db.add(init_prompt)

        evolve_config: Dict[str, Any] = {
            "num_inspirations": 0,
            "selection_policy": "tournament",
            "selection_kwargs": {"tournament_size": 3},
        }
        logger: logging.Logger = logging.getLogger("test_select")

        parent_sol: Program
        parent_prompt: Program
        inspirations: List[Program]
        parent_sol, parent_prompt, inspirations = select_parents(
            sol_db=sol_db,
            prompt_db=prompt_db,
            init_sol=init_sol,
            init_prompt=init_prompt,
            evolve_config=evolve_config,
            gen_init_pop=True,
            exploration=False,
            logger=logger,
        )

        assert parent_sol.id == "init_sol"
        assert parent_prompt.id == "init_prompt"
        assert inspirations == []


# ---------------------------------------------------------------------------
# compute_dynamic_timeout
# ---------------------------------------------------------------------------


class TestComputeDynamicTimeout:
    """Test suite for the compute_dynamic_timeout function."""

    def test_depth_zero_returns_min(self):
        """Tests that depth 0 returns min_timeout_s."""
        assert compute_dynamic_timeout(depth=0, max_timeout_s=60, min_timeout_s=5, max_depth=10) == 5

    def test_depth_zero_default_min(self):
        """Tests that depth 0 returns 1 with min_timeout_s=1."""
        assert compute_dynamic_timeout(depth=0, max_timeout_s=60, min_timeout_s=1, max_depth=10) == 1

    def test_depth_equal_max_returns_full(self):
        """Tests that depth == max_depth returns max_timeout_s."""
        assert compute_dynamic_timeout(depth=10, max_timeout_s=60, min_timeout_s=1, max_depth=10) == 60

    def test_depth_exceeds_max_clamped(self):
        """Tests that depth > max_depth is clamped to max_timeout_s."""
        assert compute_dynamic_timeout(depth=20, max_timeout_s=60, min_timeout_s=1, max_depth=10) == 60

    def test_linear_midpoint(self):
        """Tests linear interpolation at midpoint: int(1 + 99*0.5) = 50."""
        assert compute_dynamic_timeout(depth=5, max_timeout_s=100, min_timeout_s=1, max_depth=10) == 50

    def test_linear_midpoint_custom_min(self):
        """Tests midpoint with custom min: int(10 + 50*0.5) = 35."""
        assert compute_dynamic_timeout(depth=5, max_timeout_s=60, min_timeout_s=10, max_depth=10) == 35

    def test_linear_various_depths(self):
        """Tests linear scaling at several depth values with min=1."""
        assert compute_dynamic_timeout(depth=1, max_timeout_s=100, min_timeout_s=1, max_depth=10) == 10
        assert compute_dynamic_timeout(depth=3, max_timeout_s=100, min_timeout_s=1, max_depth=10) == 30
        assert compute_dynamic_timeout(depth=7, max_timeout_s=100, min_timeout_s=1, max_depth=10) == 70

    def test_floor_clamp_small_timeout(self):
        """Tests that the min floor applies when computed value is small."""
        assert compute_dynamic_timeout(depth=0, max_timeout_s=5, min_timeout_s=2, max_depth=100) == 2
        assert compute_dynamic_timeout(depth=1, max_timeout_s=5, min_timeout_s=2, max_depth=100) == 2
