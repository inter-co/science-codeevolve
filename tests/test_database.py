# ===--------------------------------------------------------------------------------------===#
#
# Part of the CodeEvolve Project, under the Apache License v2.0.
# See https://github.com/inter-co/science-codeevolve/blob/main/LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0
#
# ===--------------------------------------------------------------------------------------===#
#
# This file implements unit tests for the Program and ProgramDatabase classes.
#
# ===--------------------------------------------------------------------------------------===#

import pytest

from codeevolve.database import (
    CVTEliteMap,
    EliteFeature,
    GridEliteMap,
    Program,
    ProgramDatabase,
)

# ---------------------------------------------------------------------------
# Program dataclass
# ---------------------------------------------------------------------------


class TestProgram:
    """Test suite for the Program dataclass."""

    def test_program_creation(self):
        """Tests that a Program can be created with required fields."""
        prog: Program = Program(id="p1", code="print('hello')", language="python")
        assert prog.id == "p1"
        assert prog.code == "print('hello')"
        assert prog.language == "python"
        assert prog.fitness == 0
        assert prog.depth == 0
        assert prog.parent_id is None
        assert prog.eval_metrics == {}

    def test_program_defaults(self):
        """Tests that all optional fields have correct default values."""
        prog: Program = Program(id="p1", code="", language="python")
        assert prog.returncode is None
        assert prog.output is None
        assert prog.error is None
        assert prog.warning is None
        assert prog.eval_metrics == {}
        assert prog.fitness == 0
        assert prog.parent_id is None
        assert prog.iteration_found is None
        assert prog.generation is None
        assert prog.island_found is None
        assert prog.prompt_id is None
        assert prog.inspiration_ids == []
        assert prog.model_id is None
        assert prog.model_msg is None
        assert prog.prog_msg is None
        assert prog.features == {}
        assert prog.embedding is None
        assert prog.depth == 0

    def test_program_with_eval_metrics(self):
        """Tests that eval_metrics are stored correctly."""
        prog: Program = Program(
            id="p1",
            code="",
            language="python",
            eval_metrics={"accuracy": 0.95, "loss": 0.05},
        )
        assert prog.eval_metrics["accuracy"] == 0.95
        assert prog.eval_metrics["loss"] == 0.05


# ---------------------------------------------------------------------------
# GridEliteMap
# ---------------------------------------------------------------------------


class TestGridEliteMap:
    """Test suite for the GridEliteMap class."""

    def test_grid_elite_map_creation(self):
        """Tests that a GridEliteMap can be created with valid features."""
        features: list[EliteFeature] = [
            EliteFeature(name="f1", min_val=0.0, max_val=1.0, num_bins=5),
            EliteFeature(name="f2", min_val=0.0, max_val=10.0, num_bins=10),
        ]
        grid: GridEliteMap = GridEliteMap(features=features)
        assert grid.num_cells == 50
        assert len(grid.map) == 0

    def test_grid_elite_map_missing_num_bins(self):
        """Tests that GridEliteMap raises ValueError when num_bins is None."""
        features: list[EliteFeature] = [
            EliteFeature(name="f1", min_val=0.0, max_val=1.0, num_bins=None),
        ]
        with pytest.raises(ValueError):
            GridEliteMap(features=features)

    def test_grid_elite_map_add_single(self):
        """Tests adding a single elite to the grid map."""
        features: list[EliteFeature] = [
            EliteFeature(name="score", min_val=0.0, max_val=1.0, num_bins=5),
        ]
        grid: GridEliteMap = GridEliteMap(features=features)
        prog: Program = Program(
            id="p1", code="", language="python", fitness=0.5, features={"score": 0.3}
        )
        grid.add_elite(prog)
        assert len(grid.map) == 1
        assert "p1" in grid.get_elite_ids()

    def test_grid_elite_map_replacement(self):
        """Tests that a higher-fitness program replaces a lower-fitness one in the same cell."""
        features: list[EliteFeature] = [
            EliteFeature(name="score", min_val=0.0, max_val=1.0, num_bins=5),
        ]
        grid: GridEliteMap = GridEliteMap(features=features)

        prog1: Program = Program(
            id="p1", code="", language="python", fitness=0.3, features={"score": 0.5}
        )
        prog2: Program = Program(
            id="p2", code="", language="python", fitness=0.8, features={"score": 0.5}
        )
        grid.add_elite(prog1)
        grid.add_elite(prog2)

        elite_ids: list[str] = grid.get_elite_ids()
        assert "p2" in elite_ids
        assert "p1" not in elite_ids

    def test_grid_elite_map_no_replacement_on_lower_fitness(self):
        """Tests that a lower-fitness program does not replace a higher-fitness one."""
        features: list[EliteFeature] = [
            EliteFeature(name="score", min_val=0.0, max_val=1.0, num_bins=5),
        ]
        grid: GridEliteMap = GridEliteMap(features=features)

        prog1: Program = Program(
            id="p1", code="", language="python", fitness=0.9, features={"score": 0.5}
        )
        prog2: Program = Program(
            id="p2", code="", language="python", fitness=0.1, features={"score": 0.5}
        )
        grid.add_elite(prog1)
        grid.add_elite(prog2)

        elite_ids: list[str] = grid.get_elite_ids()
        assert "p1" in elite_ids
        assert "p2" not in elite_ids

    def test_grid_elite_map_missing_feature(self):
        """Tests that a program with missing features is not added."""
        features: list[EliteFeature] = [
            EliteFeature(name="score", min_val=0.0, max_val=1.0, num_bins=5),
        ]
        grid: GridEliteMap = GridEliteMap(features=features)
        prog: Program = Program(id="p1", code="", language="python", fitness=0.5, features={})
        grid.add_elite(prog)
        assert len(grid.map) == 0


# ---------------------------------------------------------------------------
# CVTEliteMap
# ---------------------------------------------------------------------------


class TestCVTEliteMap:
    """Test suite for the CVTEliteMap class."""

    def test_cvt_elite_map_creation(self):
        """Tests that a CVTEliteMap can be created with valid parameters."""
        features: list[EliteFeature] = [
            EliteFeature(name="f1", min_val=0.0, max_val=1.0),
        ]
        cvt_map: CVTEliteMap = CVTEliteMap(
            features=features, num_centroids=10, num_init_samples=100
        )
        assert cvt_map.centroids.shape == (10, 1)
        assert len(cvt_map.map) == 0

    def test_cvt_elite_map_add_and_retrieve(self):
        """Tests adding programs and retrieving elite IDs from CVTEliteMap."""
        features: list[EliteFeature] = [
            EliteFeature(name="f1", min_val=0.0, max_val=1.0),
        ]
        cvt_map: CVTEliteMap = CVTEliteMap(features=features, num_centroids=5, num_init_samples=50)
        prog: Program = Program(
            id="p1", code="", language="python", fitness=0.5, features={"f1": 0.3}
        )
        cvt_map.add_elite(prog)
        assert "p1" in cvt_map.get_elite_ids()

    def test_cvt_elite_map_replacement(self):
        """Tests that a higher-fitness program replaces a lower-fitness one in CVT."""
        features: list[EliteFeature] = [
            EliteFeature(name="f1", min_val=0.0, max_val=1.0),
        ]
        cvt_map: CVTEliteMap = CVTEliteMap(features=features, num_centroids=1, num_init_samples=50)
        prog1: Program = Program(
            id="p1", code="", language="python", fitness=0.3, features={"f1": 0.5}
        )
        prog2: Program = Program(
            id="p2", code="", language="python", fitness=0.9, features={"f1": 0.5}
        )
        cvt_map.add_elite(prog1)
        cvt_map.add_elite(prog2)
        assert "p2" in cvt_map.get_elite_ids()


# ---------------------------------------------------------------------------
# ProgramDatabase
# ---------------------------------------------------------------------------


class TestProgramDatabase:
    """Test suite for the ProgramDatabase class."""

    def _make_prog(self, id: str, fitness: float = 0.0, island: int = 0) -> Program:
        """Helper to create a program with minimal fields."""
        return Program(
            id=id,
            code=f"# {id}",
            language="python",
            fitness=fitness,
            island_found=island,
            iteration_found=0,
            generation=0,
            returncode=0,
            eval_metrics={"fitness": fitness},
            features={"fitness": fitness},
        )

    def test_db_creation(self):
        """Tests that ProgramDatabase can be created with defaults."""
        db: ProgramDatabase = ProgramDatabase(id=0, seed=42)
        assert db.id == 0
        assert db.num_alive == 0
        assert db.best_prog_id is None
        assert db.worst_prog_id is None
        assert len(db.programs) == 0

    def test_db_add_single(self):
        """Tests adding a single program to the database."""
        db: ProgramDatabase = ProgramDatabase(id=0, seed=42)
        prog: Program = self._make_prog("p1", fitness=1.0)
        db.add(prog)
        assert db.num_alive == 1
        assert db.best_prog_id == "p1"
        assert db.worst_prog_id == "p1"
        assert "p1" in db.programs

    def test_db_add_duplicate_raises(self):
        """Tests that adding a program with duplicate ID raises ValueError."""
        db: ProgramDatabase = ProgramDatabase(id=0, seed=42)
        prog: Program = self._make_prog("p1", fitness=1.0)
        db.add(prog)
        with pytest.raises(ValueError):
            db.add(prog)

    def test_db_best_worst_tracking(self):
        """Tests that best and worst program IDs are updated correctly."""
        db: ProgramDatabase = ProgramDatabase(id=0, seed=42)
        db.add(self._make_prog("p1", fitness=1.0))
        db.add(self._make_prog("p2", fitness=5.0))
        db.add(self._make_prog("p3", fitness=3.0))
        assert db.best_prog_id == "p2"
        assert db.worst_prog_id == "p1"

    def test_db_max_alive(self):
        """Tests that population is capped at max_alive."""
        db: ProgramDatabase = ProgramDatabase(id=0, seed=42, max_alive=3)
        db.add(self._make_prog("p1", fitness=1.0))
        db.add(self._make_prog("p2", fitness=5.0))
        db.add(self._make_prog("p3", fitness=3.0))
        assert db.num_alive == 3

        db.add(self._make_prog("p4", fitness=4.0))
        assert db.num_alive == 3
        assert db.is_alive["p1"] is False
        assert db.is_alive["p4"] is True

    def test_db_max_alive_rejects_worse(self):
        """Tests that a worse program is not added when population is full."""
        db: ProgramDatabase = ProgramDatabase(id=0, seed=42, max_alive=2)
        db.add(self._make_prog("p1", fitness=5.0))
        db.add(self._make_prog("p2", fitness=3.0))

        db.add(self._make_prog("p3", fitness=1.0))
        assert db.is_alive["p3"] is False
        assert db.num_alive == 2

    def test_db_roots(self):
        """Tests that root programs (no parent) are tracked."""
        db: ProgramDatabase = ProgramDatabase(id=0, seed=42)
        p1: Program = self._make_prog("p1", fitness=1.0)
        db.add(p1)
        p2: Program = self._make_prog("p2", fitness=2.0)
        p2.parent_id = "p1"
        db.add(p2)
        assert "p1" in db.roots
        assert "p2" not in db.roots

    # Selection tests

    def test_random_selection(self):
        """Tests uniform random selection from the database."""
        db: ProgramDatabase = ProgramDatabase(id=0, seed=42)
        db.add(self._make_prog("p1", fitness=1.0))
        db.add(self._make_prog("p2", fitness=2.0))
        db.add(self._make_prog("p3", fitness=3.0))

        parent, inspirations = db.sample(selection_policy="random", num_inspirations=1)
        assert parent is not None
        assert parent.id in {"p1", "p2", "p3"}

    def test_tournament_selection(self):
        """Tests tournament selection from the database."""
        db: ProgramDatabase = ProgramDatabase(id=0, seed=42)
        db.add(self._make_prog("p1", fitness=1.0))
        db.add(self._make_prog("p2", fitness=2.0))
        db.add(self._make_prog("p3", fitness=3.0))

        parent, inspirations = db.sample(
            selection_policy="tournament",
            num_inspirations=1,
            tournament_size=3,
        )
        assert parent is not None

    def test_roulette_selection(self):
        """Tests roulette selection from the database."""
        db: ProgramDatabase = ProgramDatabase(id=0, seed=42)
        db.add(self._make_prog("p1", fitness=1.0))
        db.add(self._make_prog("p2", fitness=2.0))
        db.add(self._make_prog("p3", fitness=3.0))

        parent, inspirations = db.sample(selection_policy="roulette", num_inspirations=0)
        assert parent is not None

    def test_roulette_selection_by_rank(self):
        """Tests rank-based roulette selection."""
        db: ProgramDatabase = ProgramDatabase(id=0, seed=42)
        db.add(self._make_prog("p1", fitness=1.0))
        db.add(self._make_prog("p2", fitness=2.0))

        parent, _ = db.sample(
            selection_policy="roulette", num_inspirations=0, roulette_by_rank=True
        )
        assert parent is not None

    def test_best_selection(self):
        """Tests best selection always returns the highest-fitness program."""
        db: ProgramDatabase = ProgramDatabase(id=0, seed=42)
        db.add(self._make_prog("p1", fitness=1.0))
        db.add(self._make_prog("p2", fitness=5.0))
        db.add(self._make_prog("p3", fitness=3.0))

        parent, _ = db.sample(selection_policy="best", num_inspirations=0)
        assert parent is not None
        assert parent.id == "p2"

    def test_invalid_selection_policy(self):
        """Tests that invalid selection policy raises ValueError."""
        db: ProgramDatabase = ProgramDatabase(id=0, seed=42)
        db.add(self._make_prog("p1", fitness=1.0))
        with pytest.raises(ValueError):
            db.sample(selection_policy="invalid_policy")

    def test_sample_with_inspirations(self):
        """Tests that inspirations are returned correctly."""
        db: ProgramDatabase = ProgramDatabase(id=0, seed=42)
        db.add(self._make_prog("p1", fitness=1.0))
        db.add(self._make_prog("p2", fitness=2.0))
        db.add(self._make_prog("p3", fitness=3.0))

        parent, inspirations = db.sample(selection_policy="random", num_inspirations=2)
        assert parent is not None
        assert len(inspirations) <= 2
        for insp in inspirations:
            assert insp.id != parent.id

    def test_sample_empty_db(self):
        """Tests that sampling from an empty database returns None."""
        db: ProgramDatabase = ProgramDatabase(id=0, seed=42)
        parent, inspirations = db.sample(selection_policy="random", num_inspirations=0)
        assert parent is None
        assert inspirations == []

    # Migration tests

    def test_get_migrants(self):
        """Tests that migration returns eligible programs."""
        db: ProgramDatabase = ProgramDatabase(id=0, seed=42)
        db.add(self._make_prog("p1", fitness=5.0, island=0))
        db.add(self._make_prog("p2", fitness=3.0, island=0))
        db.add(self._make_prog("p3", fitness=1.0, island=0))

        migrants: list[Program] = db.get_migrants(migration_rate=0.5)
        assert len(migrants) > 0
        for migrant in migrants:
            assert migrant.id != db.best_prog_id

    def test_get_migrants_excludes_already_migrated(self):
        """Tests that already-migrated programs are not selected again."""
        db: ProgramDatabase = ProgramDatabase(id=0, seed=42)
        db.add(self._make_prog("p1", fitness=5.0, island=0))
        db.add(self._make_prog("p2", fitness=3.0, island=0))
        db.add(self._make_prog("p3", fitness=1.0, island=0))

        db.has_migrated["p2"] = True
        migrants: list[Program] = db.get_migrants(migration_rate=1.0)
        migrant_ids: list[str] = [m.id for m in migrants]
        assert "p2" not in migrant_ids

    def test_get_migrants_excludes_foreign_island(self):
        """Tests that programs from other islands are not selected as migrants."""
        db: ProgramDatabase = ProgramDatabase(id=0, seed=42)
        db.add(self._make_prog("p1", fitness=5.0, island=0))
        p2: Program = self._make_prog("p2", fitness=3.0, island=1)
        p2.parent_id = None
        db.add(p2)

        migrants: list[Program] = db.get_migrants(migration_rate=1.0)
        migrant_ids: list[str] = [m.id for m in migrants]
        assert "p2" not in migrant_ids

    # MAP-Elites mode

    def test_db_map_elites_grid_mode(self):
        """Tests database in MAP-Elites grid mode."""
        features: list[EliteFeature] = [
            EliteFeature(name="fitness", min_val=0.0, max_val=10.0, num_bins=5),
        ]
        db: ProgramDatabase = ProgramDatabase(
            id=0, seed=42, features=features, elite_map_type="grid"
        )
        db.add(self._make_prog("p1", fitness=1.0))
        db.add(self._make_prog("p2", fitness=5.0))
        assert db.elite_map is not None
        assert db.best_prog_id == "p2"

    def test_db_map_elites_cvt_mode(self):
        """Tests database in MAP-Elites CVT mode."""
        features: list[EliteFeature] = [
            EliteFeature(name="fitness", min_val=0.0, max_val=10.0),
        ]
        db: ProgramDatabase = ProgramDatabase(
            id=0,
            seed=42,
            features=features,
            elite_map_type="cvt",
            num_centroids=5,
            num_init_samples=50,
        )
        db.add(self._make_prog("p1", fitness=1.0))
        assert db.elite_map is not None
        assert len(db.elite_map.get_elite_ids()) > 0

    def test_db_invalid_elite_map_type(self):
        """Tests that invalid elite_map_type raises ValueError."""
        features: list[EliteFeature] = [
            EliteFeature(name="f1", min_val=0.0, max_val=1.0, num_bins=5),
        ]
        with pytest.raises(ValueError):
            ProgramDatabase(id=0, seed=42, features=features, elite_map_type="invalid")
