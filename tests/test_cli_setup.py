# ===--------------------------------------------------------------------------------------===#
#
# Part of the CodeEvolve Project, under the Apache License v2.0.
# See https://github.com/inter-co/science-codeevolve/blob/main/LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0
#
# ===--------------------------------------------------------------------------------------===#
#
# This file implements unit tests for the CLI setup utilities.
#
# ===--------------------------------------------------------------------------------------===#

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pytest
import yaml

import codeevolve.utils.cli_setup as _cli_setup_mod
from codeevolve.utils.cli_setup import (
    compute_cpu_affinity_sets,
    create_config_copy,
    determine_checkpoint_to_load,
    find_common_checkpoints,
    load_config,
    print_dict_rec,
    setup_island_args,
    validate_environment,
    validate_paths,
)

# ---------------------------------------------------------------------------
# validate_environment
# ---------------------------------------------------------------------------


class TestValidateEnvironment:
    """Test suite for the validate_environment function."""

    def test_valid_environment(self, monkeypatch: pytest.MonkeyPatch):
        """Tests that valid environment variables are returned correctly."""
        monkeypatch.setenv("API_BASE", "http://localhost:8000")
        monkeypatch.setenv("API_KEY", "test_key_123")
        api_base: str
        api_key: str
        api_base, api_key = validate_environment()
        assert api_base == "http://localhost:8000"
        assert api_key == "test_key_123"

    def test_missing_environment(self, monkeypatch: pytest.MonkeyPatch):
        """Tests that missing environment variables cause SystemExit."""
        monkeypatch.delenv("API_BASE", raising=False)
        monkeypatch.delenv("API_KEY", raising=False)
        with pytest.raises(SystemExit):
            validate_environment()


# ---------------------------------------------------------------------------
# validate_paths
# ---------------------------------------------------------------------------


class TestValidatePaths:
    """Test suite for the validate_paths function."""

    def test_valid_paths(self, tmp_path: Path):
        """Tests that valid paths pass validation."""
        inpt_dir: Path = tmp_path / "input"
        inpt_dir.mkdir()
        cfg_path: Path = tmp_path / "config.yaml"
        cfg_path.write_text("key: value")
        validate_paths(inpt_dir, cfg_path, loading_checkpoint=False)

    def test_missing_input_dir(self, tmp_path: Path):
        """Tests that missing input directory causes SystemExit."""
        with pytest.raises(SystemExit):
            validate_paths(tmp_path / "nonexistent", None, loading_checkpoint=True)

    def test_missing_cfg_new_run(self, tmp_path: Path):
        """Tests that missing config on new run causes SystemExit."""
        inpt_dir: Path = tmp_path / "input"
        inpt_dir.mkdir()
        with pytest.raises(SystemExit):
            validate_paths(inpt_dir, None, loading_checkpoint=False)

    def test_missing_cfg_file(self, tmp_path: Path):
        """Tests that non-existent config file causes SystemExit."""
        inpt_dir: Path = tmp_path / "input"
        inpt_dir.mkdir()
        with pytest.raises(SystemExit):
            validate_paths(inpt_dir, tmp_path / "missing.yaml", loading_checkpoint=False)

    def test_loading_checkpoint_no_cfg_needed(self, tmp_path: Path):
        """Tests that config is not required when loading checkpoint."""
        inpt_dir: Path = tmp_path / "input"
        inpt_dir.mkdir()
        validate_paths(inpt_dir, None, loading_checkpoint=True)


# ---------------------------------------------------------------------------
# create_config_copy / load_config
# ---------------------------------------------------------------------------


class TestConfigLoading:
    """Test suite for config loading functions."""

    def test_create_config_copy(self, tmp_path: Path):
        """Tests that config is copied to output directory."""
        cfg_path: Path = tmp_path / "config.yaml"
        config_data: Dict[str, Any] = {"EVOLVE_CONFIG": {"num_epochs": 10}}
        with open(cfg_path, "w") as f:
            yaml.safe_dump(config_data, f)

        out_dir: Path = tmp_path / "output"
        out_dir.mkdir()

        args: Dict[str, Any] = {"cfg_path": cfg_path, "out_dir": out_dir}
        config: Dict[str, Any]
        copy_path: Path
        config, copy_path = create_config_copy(args)

        assert config["EVOLVE_CONFIG"]["num_epochs"] == 10
        assert copy_path.exists()
        assert copy_path.parent == out_dir

    def test_load_config_from_output(self, tmp_path: Path):
        """Tests loading config from output directory."""
        config_data: Dict[str, Any] = {"EVOLVE_CONFIG": {"num_islands": 4}}
        cfg_path: Path = tmp_path / "my_config.yaml"
        with open(cfg_path, "w") as f:
            yaml.safe_dump(config_data, f)

        args: Dict[str, Any] = {"out_dir": tmp_path}
        config: Dict[str, Any]
        copy_path: Path
        config, copy_path = load_config(args)
        assert config["EVOLVE_CONFIG"]["num_islands"] == 4

    def test_load_config_no_yaml(self, tmp_path: Path):
        """Tests that missing config in output directory causes SystemExit."""
        args: Dict[str, Any] = {"out_dir": tmp_path}
        with pytest.raises(SystemExit):
            load_config(args)

    def test_load_config_multiple_yaml(self, tmp_path: Path):
        """Tests that multiple config files cause SystemExit."""
        (tmp_path / "a.yaml").write_text("a: 1")
        (tmp_path / "b.yaml").write_text("b: 2")
        args: Dict[str, Any] = {"out_dir": tmp_path}
        with pytest.raises(SystemExit):
            load_config(args)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


class TestCheckpointHelpers:
    """Test suite for checkpoint helper functions."""

    def test_find_common_checkpoints(self, tmp_path: Path):
        """Tests finding common checkpoints across island directories."""
        for i in range(2):
            ckpt_dir: Path = tmp_path / f"island_{i}" / "ckpt"
            ckpt_dir.mkdir(parents=True)
            (ckpt_dir / "ckpt_10.pkl").write_text("")
            (ckpt_dir / "ckpt_20.pkl").write_text("")

        ckpt_dirs: List[Path] = [tmp_path / f"island_{i}" / "ckpt" for i in range(2)]
        common: Set[str] = find_common_checkpoints(ckpt_dirs)
        assert "ckpt_10.pkl" in common
        assert "ckpt_20.pkl" in common

    def test_find_common_checkpoints_partial(self, tmp_path: Path):
        """Tests that only checkpoints present in all directories are returned."""
        dir0: Path = tmp_path / "island_0" / "ckpt"
        dir1: Path = tmp_path / "island_1" / "ckpt"
        dir0.mkdir(parents=True)
        dir1.mkdir(parents=True)

        (dir0 / "ckpt_10.pkl").write_text("")
        (dir0 / "ckpt_20.pkl").write_text("")
        (dir1 / "ckpt_10.pkl").write_text("")

        common: Set[str] = find_common_checkpoints([dir0, dir1])
        assert "ckpt_10.pkl" in common
        assert "ckpt_20.pkl" not in common

    def test_find_common_checkpoints_empty(self, tmp_path: Path):
        """Tests that empty directories return empty set."""
        dir0: Path = tmp_path / "island_0" / "ckpt"
        dir0.mkdir(parents=True)
        common: Set[str] = find_common_checkpoints([dir0])
        assert len(common) == 0

    def test_determine_checkpoint_latest(self):
        """Tests that -1 loads the latest available checkpoint."""
        common: Set[str] = {"ckpt_10.pkl", "ckpt_20.pkl", "ckpt_30.pkl"}
        epoch: int = determine_checkpoint_to_load(common, requested_ckpt=-1)
        assert epoch == 30

    def test_determine_checkpoint_specific(self):
        """Tests loading a specific requested checkpoint."""
        common: Set[str] = {"ckpt_10.pkl", "ckpt_20.pkl"}
        epoch: int = determine_checkpoint_to_load(common, requested_ckpt=10)
        assert epoch == 10

    def test_determine_checkpoint_missing_fallback(self):
        """Tests fallback to latest when requested checkpoint is not found."""
        common: Set[str] = {"ckpt_10.pkl", "ckpt_20.pkl"}
        epoch: int = determine_checkpoint_to_load(common, requested_ckpt=99)
        assert epoch == 20

    def test_determine_checkpoint_empty(self):
        """Tests that empty common set returns 0 (new run)."""
        epoch: int = determine_checkpoint_to_load(set(), requested_ckpt=-1)
        assert epoch == 0


# ---------------------------------------------------------------------------
# setup_island_args
# ---------------------------------------------------------------------------


class TestSetupIslandArgs:
    """Test suite for the setup_island_args function."""

    def test_setup_creates_dirs(self, tmp_path: Path):
        """Tests that setup creates island output and checkpoint directories."""
        out_dir: Path = tmp_path / "output"
        out_dir.mkdir()
        cfg_path: Path = tmp_path / "config.yaml"
        cfg_path.write_text("key: value")

        args: Dict[str, Any] = {
            "out_dir": out_dir,
            "load_ckpt": 0,
        }
        isl2args: Dict[int, Dict[str, Any]] = setup_island_args(
            args, num_islands=3, cfg_copy_path=cfg_path
        )

        assert len(isl2args) == 3
        for i in range(3):
            assert isl2args[i]["isl_out_dir"].exists()
            assert isl2args[i]["ckpt_dir"].exists()
            assert isl2args[i]["load_ckpt"] == 0

    def test_cpu_affinity_set_stored(self, tmp_path: Path):
        """Tests that cpu_affinity_set is stored in each island's args."""
        out_dir: Path = tmp_path / "output"
        out_dir.mkdir()
        cfg_path: Path = tmp_path / "config.yaml"
        cfg_path.write_text("key: value")

        args: Dict[str, Any] = {"out_dir": out_dir, "load_ckpt": 0}
        affinity: List[Optional[Set[int]]] = [{0, 1}, {2, 3}, {4, 5}]
        isl2args: Dict[int, Dict[str, Any]] = setup_island_args(
            args, num_islands=3, cfg_copy_path=cfg_path, cpu_affinity_sets=affinity
        )

        assert isl2args[0]["cpu_affinity_set"] == {0, 1}
        assert isl2args[1]["cpu_affinity_set"] == {2, 3}
        assert isl2args[2]["cpu_affinity_set"] == {4, 5}

    def test_cpu_affinity_none_when_not_provided(self, tmp_path: Path):
        """Tests that cpu_affinity_set is None when no sets are supplied."""
        out_dir: Path = tmp_path / "output"
        out_dir.mkdir()
        cfg_path: Path = tmp_path / "config.yaml"
        cfg_path.write_text("key: value")

        args: Dict[str, Any] = {"out_dir": out_dir, "load_ckpt": 0}
        isl2args: Dict[int, Dict[str, Any]] = setup_island_args(
            args, num_islands=2, cfg_copy_path=cfg_path
        )

        assert isl2args[0]["cpu_affinity_set"] is None
        assert isl2args[1]["cpu_affinity_set"] is None


# ---------------------------------------------------------------------------
# compute_cpu_affinity_sets
# ---------------------------------------------------------------------------


class TestComputeCpuAffinitySets:
    """Test suite for the compute_cpu_affinity_sets function."""

    def test_no_pinning_when_not_configured(self):
        """Returns all-None sets and no warnings when num_cpus_per_eval is absent."""
        result: List[Optional[Set[int]]]
        warnings: List[str]
        result, warnings = compute_cpu_affinity_sets({}, num_islands=3)
        assert result == [None, None, None]
        assert warnings == []

    def test_no_pinning_when_explicitly_none(self):
        """Returns all-None sets when num_cpus_per_eval is explicitly None."""
        result, warnings = compute_cpu_affinity_sets({"num_cpus_per_eval": None}, num_islands=2)
        assert result == [None, None]
        assert warnings == []

    def test_invalid_value_warns_and_falls_back(self):
        """Returns all-None sets with a warning for non-positive num_cpus_per_eval."""
        for bad in (-1, 0, 1.5, "two"):
            result, warnings = compute_cpu_affinity_sets({"num_cpus_per_eval": bad}, num_islands=2)
            assert result == [None, None], f"expected no pinning for {bad!r}"
            assert len(warnings) == 1
            assert "positive integer" in warnings[0]

    def test_no_sched_getaffinity_warns(self, monkeypatch: pytest.MonkeyPatch):
        """Returns all-None with warning when platform lacks sched_getaffinity."""
        monkeypatch.delattr(_cli_setup_mod.os, "sched_getaffinity", raising=False)
        result, warnings = compute_cpu_affinity_sets({"num_cpus_per_eval": 2}, num_islands=2)
        assert result == [None, None]
        assert len(warnings) == 1
        assert "platform" in warnings[0].lower()

    def test_insufficient_cpus_warns(self, monkeypatch: pytest.MonkeyPatch):
        """Returns all-None with warning when required CPUs exceed available CPUs."""
        monkeypatch.setattr(_cli_setup_mod.os, "sched_getaffinity", lambda _: {0, 1}, raising=False)
        result, warnings = compute_cpu_affinity_sets({"num_cpus_per_eval": 2}, num_islands=3)
        assert result == [None, None, None]
        assert len(warnings) == 1
        assert "6 CPUs needed" in warnings[0]
        assert "2 are available" in warnings[0]

    def test_correct_partition_consecutive(self, monkeypatch: pytest.MonkeyPatch):
        """Assigns consecutive CPU slices to each island."""
        monkeypatch.setattr(
            _cli_setup_mod.os, "sched_getaffinity", lambda _: {0, 1, 2, 3}, raising=False
        )
        result, warnings = compute_cpu_affinity_sets({"num_cpus_per_eval": 2}, num_islands=2)
        assert result == [{0, 1}, {2, 3}]
        assert warnings == []

    def test_partial_cpu_use(self, monkeypatch: pytest.MonkeyPatch):
        """Uses only the first num_cpus_per_eval * num_islands CPUs; leaves extras unassigned."""
        monkeypatch.setattr(
            _cli_setup_mod.os, "sched_getaffinity", lambda _: {0, 1, 2, 3, 4, 5}, raising=False
        )
        result, warnings = compute_cpu_affinity_sets({"num_cpus_per_eval": 2}, num_islands=2)
        assert result == [{0, 1}, {2, 3}]
        assert warnings == []

    def test_single_cpu_per_island(self, monkeypatch: pytest.MonkeyPatch):
        """Works correctly when each island gets exactly one CPU."""
        monkeypatch.setattr(
            _cli_setup_mod.os, "sched_getaffinity", lambda _: {0, 1, 2}, raising=False
        )
        result, warnings = compute_cpu_affinity_sets({"num_cpus_per_eval": 1}, num_islands=3)
        assert result == [{0}, {1}, {2}]
        assert warnings == []

    def test_non_contiguous_cpus_sorted(self, monkeypatch: pytest.MonkeyPatch):
        """Partitions are assigned in sorted CPU order even for non-contiguous sets."""
        monkeypatch.setattr(
            _cli_setup_mod.os, "sched_getaffinity", lambda _: {0, 2, 4, 6}, raising=False
        )
        result, warnings = compute_cpu_affinity_sets({"num_cpus_per_eval": 2}, num_islands=2)
        assert result == [{0, 2}, {4, 6}]
        assert warnings == []
