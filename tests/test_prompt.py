# ===--------------------------------------------------------------------------------------===#
#
# Part of the CodeEvolve Project, under the Apache License v2.0.
# See https://github.com/inter-co/science-codeevolve/blob/main/LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0
#
# ===--------------------------------------------------------------------------------------===#
#
# This file implements unit tests for the prompt sampler and templates.
#
# ===--------------------------------------------------------------------------------------===#

from typing import Any, Dict, List

import pytest

from codeevolve.database import Program, ProgramDatabase
from codeevolve.prompt.sampler import PromptSampler, format_prog_msg
from codeevolve.prompt.template import (
    get_evolve_prompt_task_template,
    get_evolve_task_template,
    get_evolve_with_inspirations_task_template,
    get_explore_task_template,
    get_explore_with_inspirations_task_template,
)


# ---------------------------------------------------------------------------
# format_prog_msg
# ---------------------------------------------------------------------------


class TestFormatProgMsg:
    """Test suite for the format_prog_msg utility function."""

    def test_format_valid_program(self):
        """Tests formatting a program with valid execution results."""
        prog: Program = Program(
            id="p1",
            code="def foo(): return 1",
            language="python",
            returncode=0,
            eval_metrics={"fitness": 1.0},
            warning=None,
            error=None,
        )
        msg: str = format_prog_msg(prog)
        assert "python" in msg
        assert "def foo(): return 1" in msg
        assert "fitness" in msg
        assert "RETURNCODE: 0" in msg

    def test_format_program_with_error(self):
        """Tests formatting a program that had an execution error."""
        prog: Program = Program(
            id="p1",
            code="import bad",
            language="python",
            returncode=1,
            eval_metrics={},
            error="ImportError: No module named bad",
        )
        msg: str = format_prog_msg(prog)
        assert "RETURNCODE: 1" in msg
        assert "ImportError" in msg

    def test_format_program_no_returncode_raises(self):
        """Tests that formatting raises ValueError when returncode is None."""
        prog: Program = Program(id="p1", code="x=1", language="python")
        with pytest.raises(ValueError, match="returncode"):
            format_prog_msg(prog)


# ---------------------------------------------------------------------------
# PromptSampler
# ---------------------------------------------------------------------------


class TestPromptSampler:
    """Test suite for the PromptSampler class."""

    def _make_sampler(self) -> PromptSampler:
        """Helper to create a PromptSampler with a mock LM."""
        aux_lm_cfg: Dict[str, Any] = {"model_name": "MOCK"}
        return PromptSampler(
            aux_lm_cfg=aux_lm_cfg,
            api_key="test_key",
            api_base="http://localhost",
        )

    def _make_evaluated_prog(
        self,
        id: str,
        code: str = "def f(): return 1",
        fitness: float = 1.0,
        parent_id: str = None,
    ) -> Program:
        """Helper to create an evaluated program with a prog_msg."""
        prog: Program = Program(
            id=id,
            code=code,
            language="python",
            returncode=0,
            eval_metrics={"fitness": fitness},
            fitness=fitness,
            parent_id=parent_id,
        )
        prog.prog_msg = format_prog_msg(prog)
        return prog

    def test_creation(self):
        """Tests that PromptSampler can be created with mock LM."""
        sampler: PromptSampler = self._make_sampler()
        assert sampler.aux_lm is not None

    def test_build_basic(self):
        """Tests building a basic conversation prompt without inspirations."""
        sampler: PromptSampler = self._make_sampler()
        db: ProgramDatabase = ProgramDatabase(id=0, seed=42)

        prompt: Program = Program(id="prompt1", code="You are an expert.", language="text")
        prog: Program = self._make_evaluated_prog("p1")
        db.add(prog)

        messages: List[Dict[str, str]] = sampler.build(
            prompt=prompt, prog=prog, db=db, inspirations=[], exploitation=False
        )
        assert len(messages) >= 2
        assert messages[0]["role"] == "system"
        assert "You are an expert." in messages[0]["content"]

    def test_build_with_inspirations(self):
        """Tests building a prompt with inspiration programs."""
        sampler: PromptSampler = self._make_sampler()
        db: ProgramDatabase = ProgramDatabase(id=0, seed=42)

        prompt: Program = Program(id="prompt1", code="You are an expert.", language="text")
        prog: Program = self._make_evaluated_prog("p1")
        db.add(prog)

        insp: Program = self._make_evaluated_prog("insp1", code="def g(): return 2")
        db.add(insp)

        messages: List[Dict[str, str]] = sampler.build(
            prompt=prompt, prog=prog, db=db, inspirations=[insp], exploitation=True
        )
        found_inspiration: bool = any("INSPIRATION" in m.get("content", "") for m in messages)
        assert found_inspiration

    def test_build_with_chat_depth(self):
        """Tests that max_chat_depth limits conversation history."""
        sampler: PromptSampler = self._make_sampler()
        db: ProgramDatabase = ProgramDatabase(id=0, seed=42)

        prompt: Program = Program(id="prompt1", code="Expert.", language="text")

        p1: Program = self._make_evaluated_prog("p1", code="v1")
        p1.model_msg = "diff1"
        db.add(p1)

        p2: Program = self._make_evaluated_prog("p2", code="v2", parent_id="p1")
        p2.model_msg = "diff2"
        db.add(p2)

        p3: Program = self._make_evaluated_prog("p3", code="v3", parent_id="p2")
        p3.model_msg = "diff3"
        db.add(p3)

        messages_full: List[Dict[str, str]] = sampler.build(
            prompt=prompt, prog=p3, db=db, max_chat_depth=None, exploitation=True
        )
        messages_limited: List[Dict[str, str]] = sampler.build(
            prompt=prompt, prog=p3, db=db, max_chat_depth=1, exploitation=True
        )
        assert len(messages_limited) < len(messages_full)

    @pytest.mark.asyncio
    async def test_meta_prompt(self):
        """Tests that meta_prompt returns a diff string."""
        sampler: PromptSampler = self._make_sampler()
        prompt: Program = Program(
            id="prompt1",
            code="# PROMPT-BLOCK-START\nYou are an expert.\n# PROMPT-BLOCK-END",
            language="text",
        )
        prog: Program = self._make_evaluated_prog("p1")
        diff: str
        prompt_tok: int
        compl_tok: int
        diff, prompt_tok, compl_tok = await sampler.meta_prompt(prompt=prompt, prog=prog)
        assert isinstance(diff, str)


# ---------------------------------------------------------------------------
# Template factory functions
# ---------------------------------------------------------------------------


class TestTemplateFactories:
    """Test suite for template factory functions."""

    def test_evolve_task_template(self):
        """Tests that evolve task template contains expected sections."""
        template: str = get_evolve_task_template("# START", "# END")
        assert "CODE EVOLUTION" in template
        assert "SEARCH/REPLACE" in template
        assert "# START" in template

    def test_evolve_with_inspirations_task_template(self):
        """Tests that inspiration template includes inspiration analysis section."""
        template: str = get_evolve_with_inspirations_task_template("# START", "# END")
        assert "INSPIRATION" in template
        assert "CODE EVOLUTION" in template

    def test_explore_task_template(self):
        """Tests that explore template contains exploration instructions."""
        template: str = get_explore_task_template("# START", "# END")
        assert "EXPLORATION" in template
        assert "DIVERSIFICATION" in template

    def test_explore_with_inspirations_task_template(self):
        """Tests that explore with inspirations template contains both sections."""
        template: str = get_explore_with_inspirations_task_template("# START", "# END")
        assert "EXPLORATION" in template
        assert "INSPIRATION" in template

    def test_evolve_prompt_task_template(self):
        """Tests that prompt evolution template contains expected sections."""
        template: str = get_evolve_prompt_task_template("# PS", "# PE")
        assert "PROMPT EVOLUTION" in template
        assert "# PS" in template
        assert "# PE" in template
