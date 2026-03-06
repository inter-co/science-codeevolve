# ===--------------------------------------------------------------------------------------===#
#
# Part of the CodeEvolve Project, under the Apache License v2.0.
# See https://github.com/inter-co/science-codeevolve/blob/main/LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0
#
# ===--------------------------------------------------------------------------------------===#
#
# This file implements unit tests for the exploration rate schedulers.
#
# ===--------------------------------------------------------------------------------------===#

import pytest

from codeevolve.scheduler import (
    SCHEDULER_TYPES,
    CosineScheduler,
    ExponentialDecayScheduler,
    PlateauScheduler,
)

# ---------------------------------------------------------------------------
# ExponentialDecayScheduler
# ---------------------------------------------------------------------------


class TestExponentialDecayScheduler:
    """Test suite for the ExponentialDecayScheduler."""

    def test_creation(self):
        """Tests that scheduler is created with valid parameters."""
        sched: ExponentialDecayScheduler = ExponentialDecayScheduler(
            exploration_rate=0.5, max_rate=1.0, min_rate=0.01, decay_weight=0.99
        )
        assert sched.exploration_rate == 0.5
        assert sched.decay_weight == 0.99
        assert sched.initial_rate == 0.5

    def test_invalid_decay_weight(self):
        """Tests that invalid decay_weight raises ValueError."""
        with pytest.raises(ValueError):
            ExponentialDecayScheduler(
                exploration_rate=0.5, max_rate=1.0, min_rate=0.01, decay_weight=0.0
            )
        with pytest.raises(ValueError):
            ExponentialDecayScheduler(
                exploration_rate=0.5, max_rate=1.0, min_rate=0.01, decay_weight=1.5
            )

    def test_decay_over_epochs(self):
        """Tests that exploration rate decays over epochs."""
        sched: ExponentialDecayScheduler = ExponentialDecayScheduler(
            exploration_rate=1.0, max_rate=1.0, min_rate=0.0, decay_weight=0.5
        )
        rate_0: float = sched(epoch=0)
        rate_1: float = sched(epoch=1)
        rate_2: float = sched(epoch=2)

        assert rate_0 == 1.0
        assert rate_1 == 0.5
        assert rate_2 == 0.25

    def test_min_rate_clamp(self):
        """Tests that exploration rate is clamped to min_rate."""
        sched: ExponentialDecayScheduler = ExponentialDecayScheduler(
            exploration_rate=0.5, max_rate=1.0, min_rate=0.1, decay_weight=0.01
        )
        rate: float = sched(epoch=100)
        assert rate == 0.1

    def test_reset(self):
        """Tests that reset restores initial state."""
        sched: ExponentialDecayScheduler = ExponentialDecayScheduler(
            exploration_rate=0.5, max_rate=1.0, min_rate=0.01, decay_weight=0.99
        )
        sched(epoch=50)
        sched.reset(exploration_rate=0.8)
        assert sched.exploration_rate == 0.8
        assert sched.initial_rate == 0.8


# ---------------------------------------------------------------------------
# PlateauScheduler
# ---------------------------------------------------------------------------


class TestPlateauScheduler:
    """Test suite for the PlateauScheduler."""

    def test_creation(self):
        """Tests that scheduler is created with valid parameters."""
        sched: PlateauScheduler = PlateauScheduler(
            exploration_rate=0.5,
            max_rate=1.0,
            min_rate=0.01,
            plateau_threshold=5,
            increase_factor=1.5,
            decrease_factor=0.9,
        )
        assert sched.exploration_rate == 0.5
        assert sched.plateau_threshold == 5

    def test_invalid_plateau_threshold(self):
        """Tests that non-positive plateau_threshold raises ValueError."""
        with pytest.raises(ValueError):
            PlateauScheduler(
                exploration_rate=0.5,
                max_rate=1.0,
                min_rate=0.01,
                plateau_threshold=0,
                increase_factor=1.5,
                decrease_factor=0.9,
            )

    def test_invalid_increase_factor(self):
        """Tests that increase_factor <= 1.0 raises ValueError."""
        with pytest.raises(ValueError):
            PlateauScheduler(
                exploration_rate=0.5,
                max_rate=1.0,
                min_rate=0.01,
                plateau_threshold=5,
                increase_factor=0.5,
                decrease_factor=0.9,
            )

    def test_invalid_decrease_factor(self):
        """Tests that decrease_factor outside (0, 1) raises ValueError."""
        with pytest.raises(ValueError):
            PlateauScheduler(
                exploration_rate=0.5,
                max_rate=1.0,
                min_rate=0.01,
                plateau_threshold=5,
                increase_factor=1.5,
                decrease_factor=1.5,
            )

    def test_decrease_on_improvement(self):
        """Tests that rate decreases when fitness improves."""
        sched: PlateauScheduler = PlateauScheduler(
            exploration_rate=0.5,
            max_rate=1.0,
            min_rate=0.01,
            plateau_threshold=5,
            increase_factor=1.5,
            decrease_factor=0.9,
        )
        rate: float = sched(best_fitness=1.0)
        assert rate < 0.5

    def test_increase_on_plateau(self):
        """Tests that rate increases after plateau_threshold stagnant epochs."""
        sched: PlateauScheduler = PlateauScheduler(
            exploration_rate=0.5,
            max_rate=1.0,
            min_rate=0.01,
            plateau_threshold=3,
            increase_factor=2.0,
            decrease_factor=0.9,
        )
        sched(best_fitness=1.0)
        rate_before: float = sched.exploration_rate

        for _ in range(3):
            sched(best_fitness=0.5)

        assert sched.exploration_rate > rate_before

    def test_reset_clears_state(self):
        """Tests that reset clears epochs_without_improvement and last_best_fitness."""
        sched: PlateauScheduler = PlateauScheduler(
            exploration_rate=0.5,
            max_rate=1.0,
            min_rate=0.01,
            plateau_threshold=5,
            increase_factor=1.5,
            decrease_factor=0.9,
        )
        sched(best_fitness=1.0)
        sched.reset(exploration_rate=0.6)
        assert sched.epochs_without_improvement == 0
        assert sched.last_best_fitness == float("-inf")


# ---------------------------------------------------------------------------
# CosineScheduler
# ---------------------------------------------------------------------------


class TestCosineScheduler:
    """Test suite for the CosineScheduler."""

    def test_creation(self):
        """Tests that scheduler is created with valid parameters."""
        sched: CosineScheduler = CosineScheduler(
            exploration_rate=0.5, max_rate=1.0, min_rate=0.0, cycle_length=10
        )
        assert sched.cycle_length == 10

    def test_invalid_cycle_length(self):
        """Tests that non-positive cycle_length raises ValueError."""
        with pytest.raises(ValueError):
            CosineScheduler(exploration_rate=0.5, max_rate=1.0, min_rate=0.0, cycle_length=0)

    def test_rate_at_epoch_zero(self):
        """Tests that rate at epoch 0 equals max_rate."""
        sched: CosineScheduler = CosineScheduler(
            exploration_rate=0.5, max_rate=1.0, min_rate=0.0, cycle_length=10
        )
        rate: float = sched(epoch=0)
        assert abs(rate - 1.0) < 1e-6

    def test_rate_at_half_cycle(self):
        """Tests that rate at half cycle equals midpoint between max and min."""
        sched: CosineScheduler = CosineScheduler(
            exploration_rate=0.5, max_rate=1.0, min_rate=0.0, cycle_length=10
        )
        rate: float = sched(epoch=5)
        assert abs(rate - 0.5) < 1e-6

    def test_rate_cycles(self):
        """Tests that rate is periodic with cycle_length."""
        sched: CosineScheduler = CosineScheduler(
            exploration_rate=0.5, max_rate=1.0, min_rate=0.0, cycle_length=10
        )
        rate_0: float = sched(epoch=0)
        rate_10: float = sched(epoch=10)
        assert abs(rate_0 - rate_10) < 1e-6


# ---------------------------------------------------------------------------
# Base class and registry
# ---------------------------------------------------------------------------


class TestSchedulerBase:
    """Test suite for scheduler base class validation and registry."""

    def test_invalid_min_max_rate(self):
        """Tests that min_rate > max_rate raises ValueError."""
        with pytest.raises(ValueError):
            ExponentialDecayScheduler(
                exploration_rate=0.5, max_rate=0.1, min_rate=0.9, decay_weight=0.99
            )

    def test_rate_outside_bounds(self):
        """Tests that exploration_rate outside [min_rate, max_rate] raises ValueError."""
        with pytest.raises(ValueError):
            ExponentialDecayScheduler(
                exploration_rate=1.5, max_rate=1.0, min_rate=0.0, decay_weight=0.99
            )

    def test_reset_with_invalid_rate(self):
        """Tests that reset with rate outside bounds raises ValueError."""
        sched: ExponentialDecayScheduler = ExponentialDecayScheduler(
            exploration_rate=0.5, max_rate=1.0, min_rate=0.0, decay_weight=0.99
        )
        with pytest.raises(ValueError):
            sched.reset(exploration_rate=2.0)

    def test_scheduler_registry(self):
        """Tests that all scheduler types are registered."""
        assert "ExponentialDecayScheduler" in SCHEDULER_TYPES
        assert "PlateauScheduler" in SCHEDULER_TYPES
        assert "CosineScheduler" in SCHEDULER_TYPES
        assert len(SCHEDULER_TYPES) == 3
