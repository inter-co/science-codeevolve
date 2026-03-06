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
from codeevolve.evolution import _get_markers, select_parents


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
        init_prompt: Program = Program(
            id="init_prompt", code="prompt", language="text"
        )

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
