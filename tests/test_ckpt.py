# ===--------------------------------------------------------------------------------------===#
#
# Part of the CodeEvolve Project, under the Apache License v2.0.
# See https://github.com/inter-co/science-codeevolve/blob/main/LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0
#
# ===--------------------------------------------------------------------------------------===#
#
# This file implements unit tests for checkpointing routines.
#
# ===--------------------------------------------------------------------------------------===#

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

from codeevolve.database import Program, ProgramDatabase
from codeevolve.islands.sync import GlobalBestProg
from codeevolve.scheduler import ExponentialScheduler, Scheduler
from codeevolve.utils.ckpt import load_ckpt, load_run_metadata, save_ckpt, save_run_metadata
from codeevolve.utils.constants import RUN_METADATA_FILE

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_db_with_program(island_id: int = 0) -> ProgramDatabase:
    """Creates a ProgramDatabase with one program for testing."""
    db: ProgramDatabase = ProgramDatabase(id=island_id, seed=42)
    prog: Program = Program(
        id="test_prog",
        code="def f(): return 1",
        language="python",
        fitness=10.0,
        island_found=island_id,
        iteration_found=0,
        generation=0,
        returncode=0,
        eval_metrics={"fitness": 10.0},
    )
    db.add(prog)
    return db


# ---------------------------------------------------------------------------
# save_ckpt / load_ckpt
# ---------------------------------------------------------------------------


class TestCheckpointing:
    """Test suite for checkpoint save and load operations."""

    def test_save_and_load_ckpt(self, tmp_path: Path):
        """Tests that checkpoint round-trip preserves database state."""
        sol_db: ProgramDatabase = _make_db_with_program()
        prompt_db: ProgramDatabase = ProgramDatabase(id=0, seed=42)
        prompt_prog: Program = Program(
            id="prompt1",
            code="You are an expert.",
            language="text",
            fitness=0.0,
            iteration_found=0,
            generation=0,
        )
        prompt_db.add(prompt_prog)

        evolve_state: Dict[str, Any] = {
            "early_stop_counter": 3,
            "best_fit_hist": [1.0, 2.0, 10.0],
            "avg_fit_hist": [0.5, 1.0, 5.0],
            "errors": [],
            "tok_usage": [],
            "exploration": [True, False, True],
        }

        logger: logging.Logger = logging.getLogger("test_ckpt")

        best_sol_path: Path = tmp_path / "best_sol.py"
        best_prompt_path: Path = tmp_path / "best_prompt.txt"
        ckpt_dir: Path = tmp_path / "ckpt"
        ckpt_dir.mkdir()

        save_ckpt(
            curr_epoch=10,
            prompt_db=prompt_db,
            sol_db=sol_db,
            evolve_state=evolve_state,
            exploration_scheduler=None,
            best_sol_path=best_sol_path,
            best_prompt_path=best_prompt_path,
            ckpt_dir=ckpt_dir,
            logger=logger,
        )

        assert best_sol_path.exists()
        assert best_prompt_path.exists()
        assert (ckpt_dir / "ckpt_10.pkl").exists()

        loaded_prompt_db: Optional[ProgramDatabase]
        loaded_sol_db: Optional[ProgramDatabase]
        loaded_state: Optional[Dict[str, Any]]
        loaded_sched: Optional[Scheduler]
        loaded_ts: Optional[Scheduler]
        loaded_prompt_db, loaded_sol_db, loaded_state, loaded_sched, loaded_ts = load_ckpt(
            10, ckpt_dir
        )

        assert loaded_sol_db is not None
        assert loaded_prompt_db is not None
        assert loaded_state is not None
        assert loaded_sched is None
        assert loaded_ts is None
        assert loaded_sol_db.best_prog_id == "test_prog"
        assert loaded_state["early_stop_counter"] == 3

    def test_save_and_load_with_scheduler(self, tmp_path: Path):
        """Tests checkpoint round-trip with a scheduler."""
        sol_db: ProgramDatabase = _make_db_with_program()
        prompt_db: ProgramDatabase = ProgramDatabase(id=0, seed=42)
        prompt_db.add(Program(id="pr", code="p", language="text"))

        scheduler: ExponentialScheduler = ExponentialScheduler(
            value=0.5, max_value=1.0, min_value=0.01, weight=0.99
        )

        logger: logging.Logger = logging.getLogger("test_ckpt_sched")
        ckpt_dir: Path = tmp_path / "ckpt"
        ckpt_dir.mkdir()

        save_ckpt(
            curr_epoch=5,
            prompt_db=prompt_db,
            sol_db=sol_db,
            evolve_state={
                "early_stop_counter": 0,
                "best_fit_hist": [],
                "avg_fit_hist": [],
                "errors": [],
                "tok_usage": [],
                "exploration": [],
            },
            exploration_scheduler=scheduler,
            best_sol_path=tmp_path / "best.py",
            best_prompt_path=tmp_path / "best_prompt.txt",
            ckpt_dir=ckpt_dir,
            logger=logger,
        )

        _, _, _, loaded_sched, _ = load_ckpt(5, ckpt_dir)
        assert loaded_sched is not None
        assert isinstance(loaded_sched, ExponentialScheduler)
        assert loaded_sched.weight == 0.99

    def test_best_files_content(self, tmp_path: Path):
        """Tests that best solution and prompt files contain correct code."""
        sol_db: ProgramDatabase = _make_db_with_program()
        prompt_db: ProgramDatabase = ProgramDatabase(id=0, seed=42)
        prompt_db.add(Program(id="pr", code="Expert prompt.", language="text"))

        logger: logging.Logger = logging.getLogger("test_ckpt_content")
        ckpt_dir: Path = tmp_path / "ckpt"
        ckpt_dir.mkdir()

        best_sol_path: Path = tmp_path / "best_sol.py"
        best_prompt_path: Path = tmp_path / "best_prompt.txt"

        save_ckpt(
            curr_epoch=1,
            prompt_db=prompt_db,
            sol_db=sol_db,
            evolve_state={
                "early_stop_counter": 0,
                "best_fit_hist": [],
                "avg_fit_hist": [],
                "errors": [],
                "tok_usage": [],
                "exploration": [],
            },
            exploration_scheduler=None,
            best_sol_path=best_sol_path,
            best_prompt_path=best_prompt_path,
            ckpt_dir=ckpt_dir,
            logger=logger,
        )

        sol_content: str = best_sol_path.read_text()
        prompt_content: str = best_prompt_path.read_text()
        assert "def f(): return 1" in sol_content
        assert "Expert prompt." in prompt_content


# ---------------------------------------------------------------------------
# Run metadata
# ---------------------------------------------------------------------------


class TestRunMetadata:
    """Test suite for run metadata save and load operations."""

    def test_save_and_load_metadata(self, tmp_path: Path):
        """Tests save/load round-trip for run metadata."""
        best_sol: GlobalBestProg = GlobalBestProg()
        best_sol.fitness.value = 42.0
        best_sol.iteration_found.value = 10
        best_sol.island_found.value = 0
        best_sol.depth.value = 5

        save_run_metadata(
            tmp_path,
            epoch=10,
            elapsed_time=120.5,
            cpu_count=8,
            global_best_sol=best_sol,
            early_stop_counter=5,
        )

        metadata: Optional[Dict[str, Any]] = load_run_metadata(tmp_path, epoch=10)
        assert metadata is not None
        assert metadata["elapsed_time"] == 120.5
        assert metadata["cpu_count"] == 8
        assert metadata["early_stop_counter"] == 5
        assert metadata["best_sol"]["fitness"] == 42.0

    def test_load_nonexistent_metadata(self, tmp_path: Path):
        """Tests loading metadata when file doesn't exist returns None."""
        metadata: Optional[Dict[str, Any]] = load_run_metadata(tmp_path, epoch=99)
        assert metadata is None

    def test_load_missing_epoch(self, tmp_path: Path):
        """Tests loading metadata for a missing epoch returns empty dict."""
        best_sol: GlobalBestProg = GlobalBestProg()
        save_run_metadata(
            tmp_path,
            epoch=10,
            elapsed_time=100.0,
            cpu_count=4,
            global_best_sol=best_sol,
            early_stop_counter=5,
        )

        metadata: Optional[Dict[str, Any]] = load_run_metadata(tmp_path, epoch=99)
        assert metadata == None

    def test_metadata_accumulates(self, tmp_path: Path):
        """Tests that multiple saves accumulate in the same file."""
        best_sol: GlobalBestProg = GlobalBestProg()
        save_run_metadata(
            tmp_path,
            epoch=10,
            elapsed_time=100.0,
            cpu_count=4,
            global_best_sol=best_sol,
            early_stop_counter=5,
        )
        save_run_metadata(
            tmp_path,
            epoch=20,
            elapsed_time=200.0,
            cpu_count=4,
            global_best_sol=best_sol,
            early_stop_counter=10,
        )

        metadata_file: Path = tmp_path / RUN_METADATA_FILE
        with open(metadata_file, "r") as f:
            data: Dict[str, Any] = json.load(f)

        assert "10" in data
        assert "20" in data
