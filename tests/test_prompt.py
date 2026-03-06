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
    format_eval_budget,
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
# format_eval_budget
# ---------------------------------------------------------------------------


class TestFormatEvalBudget:
    """Test suite for the format_eval_budget utility function."""

    def test_timeout_only(self):
        """Tests formatting with only a timeout (no memory limit)."""
        result: str = format_eval_budget(timeout_s=60)
        assert "60 seconds" in result
        assert "Time limit" in result
        assert "Memory limit" not in result

    def test_timeout_with_memory_gb(self):
        """Tests formatting with a memory limit in the GB range."""
        result: str = format_eval_budget(timeout_s=120, max_mem_b=2 * 1024**3)
        assert "120 seconds" in result
        assert "2.0 GB" in result

    def test_timeout_with_memory_mb(self):
        """Tests formatting with a memory limit in the MB range."""
        result: str = format_eval_budget(timeout_s=30, max_mem_b=512 * 1024**2)
        assert "30 seconds" in result
        assert "512.0 MB" in result

    def test_timeout_with_memory_bytes(self):
        """Tests formatting with a memory limit below the MB range."""
        result: str = format_eval_budget(timeout_s=10, max_mem_b=1024)
        assert "10 seconds" in result
        assert "1024 bytes" in result

    def test_memory_none(self):
        """Tests that memory line is omitted when max_mem_b is None."""
        result: str = format_eval_budget(timeout_s=60, max_mem_b=None)
        assert "Memory limit" not in result

    def test_exact_gb_boundary(self):
        """Tests formatting at exactly 1 GB."""
        result: str = format_eval_budget(timeout_s=60, max_mem_b=1024**3)
        assert "1.0 GB" in result


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

    def test_build_with_eval_budget(self):
        """Tests that eval_budget is injected into the system message."""
        sampler: PromptSampler = self._make_sampler()
        db: ProgramDatabase = ProgramDatabase(id=0, seed=42)

        prompt: Program = Program(id="prompt1", code="You are an expert.", language="text")
        prog: Program = self._make_evaluated_prog("p1")
        db.add(prog)

        budget: str = format_eval_budget(timeout_s=60, max_mem_b=1024**3)
        messages: List[Dict[str, str]] = sampler.build(
            prompt=prompt, prog=prog, db=db, inspirations=[], exploitation=False,
            eval_budget=budget,
        )

        sys_content: str = messages[0]["content"]
        assert "You are an expert." in sys_content
        assert "60 seconds" in sys_content
        assert "1.0 GB" in sys_content

    def test_build_without_eval_budget(self):
        """Tests that system message has no budget section when eval_budget is None."""
        sampler: PromptSampler = self._make_sampler()
        db: ProgramDatabase = ProgramDatabase(id=0, seed=42)

        prompt: Program = Program(id="prompt1", code="You are an expert.", language="text")
        prog: Program = self._make_evaluated_prog("p1")
        db.add(prog)

        messages: List[Dict[str, str]] = sampler.build(
            prompt=prompt, prog=prog, db=db, inspirations=[], exploitation=False,
        )

        sys_content: str = messages[0]["content"]
        assert "You are an expert." in sys_content
        assert "COMPUTATIONAL BUDGET" not in sys_content

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
