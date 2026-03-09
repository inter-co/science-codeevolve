# ===--------------------------------------------------------------------------------------===#
#
# Part of the CodeEvolve Project, under the Apache License v2.0.
# See https://github.com/inter-co/science-codeevolve/blob/main/LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0
#
# ===--------------------------------------------------------------------------------------===#
#
# This file implements unit tests for the schedulers.
#
# ===--------------------------------------------------------------------------------------===#

import pytest

from codeevolve.scheduler import (
    SCHEDULER_TYPES,
    CosineScheduler,
    ExponentialScheduler,
    PlateauScheduler,
)

# ---------------------------------------------------------------------------
# ExponentialScheduler
# ---------------------------------------------------------------------------


class TestExponentialScheduler:
    """Test suite for the ExponentialScheduler."""

    def test_creation(self):
        """Tests that scheduler is created with valid parameters."""
        sched: ExponentialScheduler = ExponentialScheduler(
            value=0.5, max_value=1.0, min_value=0.01, weight=0.99
        )
        assert sched.value == 0.5
        assert sched.weight == 0.99
        assert sched.initial_value == 0.5

    def test_invalid_weight(self):
        """Tests that non-positive weight raises ValueError."""
        with pytest.raises(ValueError):
            ExponentialScheduler(value=0.5, max_value=1.0, min_value=0.01, weight=0.0)
        with pytest.raises(ValueError):
            ExponentialScheduler(value=0.5, max_value=1.0, min_value=0.01, weight=-1.0)

    def test_decay_over_epochs(self):
        """Tests that value decays over epochs with weight < 1."""
        sched: ExponentialScheduler = ExponentialScheduler(
            value=1.0, max_value=1.0, min_value=0.0, weight=0.5
        )
        rate_0: float = sched(epoch=0)
        rate_1: float = sched(epoch=1)
        rate_2: float = sched(epoch=2)

        assert rate_0 == 1.0
        assert rate_1 == 0.5
        assert rate_2 == 0.25

    def test_growth_over_epochs(self):
        """Tests that value grows over epochs with weight > 1."""
        sched: ExponentialScheduler = ExponentialScheduler(
            value=5.0, max_value=60.0, min_value=5.0, weight=1.5
        )
        val_0: float = sched(epoch=0)
        val_1: float = sched(epoch=1)
        val_2: float = sched(epoch=2)

        assert val_0 == 5.0
        assert val_1 == 7.5
        assert val_2 == 11.25

    def test_growth_clamped_to_max(self):
        """Tests that exponential growth is clamped to max_value."""
        sched: ExponentialScheduler = ExponentialScheduler(
            value=5.0, max_value=20.0, min_value=5.0, weight=2.0
        )
        val: float = sched(epoch=100)
        assert val == 20.0

    def test_min_value_clamp(self):
        """Tests that value is clamped to min_value."""
        sched: ExponentialScheduler = ExponentialScheduler(
            value=0.5, max_value=1.0, min_value=0.1, weight=0.01
        )
        rate: float = sched(epoch=100)
        assert rate == 0.1

    def test_reset(self):
        """Tests that reset restores initial state."""
        sched: ExponentialScheduler = ExponentialScheduler(
            value=0.5, max_value=1.0, min_value=0.01, weight=0.99
        )
        sched(epoch=50)
        sched.reset(value=0.8)
        assert sched.value == 0.8
        assert sched.initial_value == 0.8


# ---------------------------------------------------------------------------
# PlateauScheduler
# ---------------------------------------------------------------------------


class TestPlateauScheduler:
    """Test suite for the PlateauScheduler."""

    def test_creation(self):
        """Tests that scheduler is created with valid parameters."""
        sched: PlateauScheduler = PlateauScheduler(
            value=0.5,
            max_value=1.0,
            min_value=0.01,
            plateau_threshold=5,
            increase_factor=1.5,
            decrease_factor=0.9,
        )
        assert sched.value == 0.5
        assert sched.plateau_threshold == 5

    def test_invalid_plateau_threshold(self):
        """Tests that non-positive plateau_threshold raises ValueError."""
        with pytest.raises(ValueError):
            PlateauScheduler(
                value=0.5,
                max_value=1.0,
                min_value=0.01,
                plateau_threshold=0,
                increase_factor=1.5,
                decrease_factor=0.9,
            )

    def test_invalid_increase_factor(self):
        """Tests that increase_factor <= 1.0 raises ValueError."""
        with pytest.raises(ValueError):
            PlateauScheduler(
                value=0.5,
                max_value=1.0,
                min_value=0.01,
                plateau_threshold=5,
                increase_factor=0.5,
                decrease_factor=0.9,
            )

    def test_invalid_decrease_factor(self):
        """Tests that decrease_factor outside (0, 1) raises ValueError."""
        with pytest.raises(ValueError):
            PlateauScheduler(
                value=0.5,
                max_value=1.0,
                min_value=0.01,
                plateau_threshold=5,
                increase_factor=1.5,
                decrease_factor=1.5,
            )

    def test_decrease_on_improvement(self):
        """Tests that value decreases when fitness improves."""
        sched: PlateauScheduler = PlateauScheduler(
            value=0.5,
            max_value=1.0,
            min_value=0.01,
            plateau_threshold=5,
            increase_factor=1.5,
            decrease_factor=0.9,
        )
        rate: float = sched(best_fitness=1.0)
        assert rate < 0.5

    def test_increase_on_plateau(self):
        """Tests that value increases after plateau_threshold stagnant epochs."""
        sched: PlateauScheduler = PlateauScheduler(
            value=0.5,
            max_value=1.0,
            min_value=0.01,
            plateau_threshold=3,
            increase_factor=2.0,
            decrease_factor=0.9,
        )
        sched(best_fitness=1.0)
        rate_before: float = sched.value

        for _ in range(3):
            sched(best_fitness=0.5)

        assert sched.value > rate_before

    def test_reset_clears_state(self):
        """Tests that reset clears epochs_without_improvement and last_best_fitness."""
        sched: PlateauScheduler = PlateauScheduler(
            value=0.5,
            max_value=1.0,
            min_value=0.01,
            plateau_threshold=5,
            increase_factor=1.5,
            decrease_factor=0.9,
        )
        sched(best_fitness=1.0)
        sched.reset(value=0.6)
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
            value=0.5, max_value=1.0, min_value=0.0, cycle_length=10
        )
        assert sched.cycle_length == 10

    def test_invalid_cycle_length(self):
        """Tests that non-positive cycle_length raises ValueError."""
        with pytest.raises(ValueError):
            CosineScheduler(value=0.5, max_value=1.0, min_value=0.0, cycle_length=0)

    def test_rate_at_epoch_zero(self):
        """Tests that value at epoch 0 equals max_value."""
        sched: CosineScheduler = CosineScheduler(
            value=0.5, max_value=1.0, min_value=0.0, cycle_length=10
        )
        rate: float = sched(epoch=0)
        assert abs(rate - 1.0) < 1e-6

    def test_rate_at_half_cycle(self):
        """Tests that value at half cycle equals midpoint between max and min."""
        sched: CosineScheduler = CosineScheduler(
            value=0.5, max_value=1.0, min_value=0.0, cycle_length=10
        )
        rate: float = sched(epoch=5)
        assert abs(rate - 0.5) < 1e-6

    def test_rate_cycles(self):
        """Tests that value is periodic with cycle_length."""
        sched: CosineScheduler = CosineScheduler(
            value=0.5, max_value=1.0, min_value=0.0, cycle_length=10
        )
        rate_0: float = sched(epoch=0)
        rate_10: float = sched(epoch=10)
        assert abs(rate_0 - rate_10) < 1e-6


# ---------------------------------------------------------------------------
# Base class and registry
# ---------------------------------------------------------------------------


class TestSchedulerBase:
    """Test suite for scheduler base class validation and registry."""

    def test_invalid_min_max_value(self):
        """Tests that min_value > max_value raises ValueError."""
        with pytest.raises(ValueError):
            ExponentialScheduler(value=0.5, max_value=0.1, min_value=0.9, weight=0.99)

    def test_value_outside_bounds(self):
        """Tests that value outside [min_value, max_value] raises ValueError."""
        with pytest.raises(ValueError):
            ExponentialScheduler(value=1.5, max_value=1.0, min_value=0.0, weight=0.99)

    def test_reset_with_invalid_value(self):
        """Tests that reset with value outside bounds raises ValueError."""
        sched: ExponentialScheduler = ExponentialScheduler(
            value=0.5, max_value=1.0, min_value=0.0, weight=0.99
        )
        with pytest.raises(ValueError):
            sched.reset(value=2.0)

    def test_scheduler_registry(self):
        """Tests that all scheduler types are registered."""
        assert "ExponentialScheduler" in SCHEDULER_TYPES
        assert "PlateauScheduler" in SCHEDULER_TYPES
        assert "CosineScheduler" in SCHEDULER_TYPES
        assert len(SCHEDULER_TYPES) == 3
