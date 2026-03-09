# ===--------------------------------------------------------------------------------------===#
#
# Part of the CodeEvolve Project, under the Apache License v2.0.
# See https://github.com/inter-co/science-codeevolve/blob/main/LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0
#
# ===--------------------------------------------------------------------------------------===#
#
# This file implements generic bounded-value schedulers used for exploration
# rate scheduling, timeout scheduling, and other per-epoch parameter control.
#
# ===--------------------------------------------------------------------------------------===#

from abc import ABC, abstractmethod
from typing import Dict, Optional, Type

import numpy as np

# ---------------------------------------------------------------------------
# Scheduler classes
# ---------------------------------------------------------------------------


class Scheduler(ABC):
    """Abstract base class for bounded-value schedulers.

    A scheduler manages a single numeric value that is adjusted each epoch
    according to a concrete scheduling strategy.  The value is always kept
    within ``[min_value, max_value]``.

    Concrete subclasses are used for exploration-rate scheduling, evaluation
    timeout scheduling, and any other per-epoch parameter that needs
    adaptive or predetermined control.

    Attributes:
        value: Current scheduled value.
        max_value: Upper bound (inclusive).
        min_value: Lower bound (inclusive).
    """

    def __init__(self, value: float, max_value: float, min_value: float):
        """
        Initialize the scheduler.

        Args:
            value: Initial value.
            max_value: Maximum value to clip to.
            min_value: Minimum value to clip to.

        Raises:
            ValueError: If min_value > max_value or value is outside bounds.
        """
        if min_value > max_value:
            raise ValueError(f"min_value ({min_value}) must be <= max_value ({max_value})")
        if not (min_value <= value <= max_value):
            raise ValueError(
                f"value ({value}) must be between "
                f"min_value ({min_value}) and max_value ({max_value})"
            )

        self.value: float = value
        self.max_value: float = max_value
        self.min_value: float = min_value

    @abstractmethod
    def __call__(self, **kwargs) -> float:
        """
        Compute and update the scheduled value.

        Args:
            **kwargs: Scheduler-specific arguments (e.g., epoch, fitness).

        Returns:
            Updated value after applying the scheduling logic.
        """
        pass

    def reset(self, value: Optional[float] = None) -> None:
        """
        Reset the scheduler to its initial state.

        Args:
            value: New initial value. If None, keeps current value.
        """
        if value is not None:
            if not (self.min_value <= value <= self.max_value):
                raise ValueError(
                    f"value ({value}) must be between "
                    f"min_value ({self.min_value}) and max_value ({self.max_value})"
                )
            self.value = value


class ExponentialScheduler(Scheduler):
    """Exponential scheduler that scales a value geometrically over epochs.

    The value evolves according to::

        value(t) = initial_value * (weight ^ t)

    Use ``weight < 1`` for decay (e.g. exploration rate) and ``weight > 1``
    for growth (e.g. evaluation timeout ramp-up).

    Attributes:
        weight: Multiplicative factor applied each epoch.
        initial_value: Value at epoch 0 (used for reset).
    """

    def __init__(self, value: float, max_value: float, min_value: float, weight: float):
        """
        Initialize the exponential scheduler.

        Args:
            value: Initial value.
            max_value: Upper bound.
            min_value: Lower bound.
            weight: Multiplicative factor per epoch. Must be positive.
                Use values in (0, 1) for decay and values > 1 for growth.

        Raises:
            ValueError: If weight is not positive.
        """
        super().__init__(value, max_value, min_value)
        if weight <= 0:
            raise ValueError(f"weight ({weight}) must be positive")
        self.weight: float = weight
        self.initial_value: float = value

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            "("
            f"value={self.value},"
            f"min_value={self.min_value},"
            f"max_value={self.max_value},"
            f"weight={self.weight},"
            f"initial_value={self.initial_value}"
            ")"
        )

    def __call__(self, epoch: int, **kwargs) -> float:
        """
        Compute value with exponential scaling.

        Args:
            epoch: Current epoch number (0-indexed).
            **kwargs: Additional arguments (ignored).

        Returns:
            Updated value after scaling.
        """
        raw: float = self.initial_value * (self.weight**epoch)
        self.value = float(np.clip(raw, self.min_value, self.max_value))
        return self.value

    def reset(self, value: Optional[float] = None) -> None:
        """Reset scheduler and update initial value if provided."""
        super().reset(value)
        if value is not None:
            self.initial_value = value


class PlateauScheduler(Scheduler):
    """Adaptive scheduler that adjusts value based on fitness improvements.

    This scheduler monitors fitness progress and reacts:

    * **Improvement** — multiplies by ``decrease_factor`` (shrinks value).
    * **Plateau** (no improvement for ``plateau_threshold`` epochs) —
      multiplies by ``increase_factor`` (grows value).

    For exploration-rate scheduling, this means exploring more when stuck.
    For timeout scheduling, it means giving programs more time when stuck.

    Attributes:
        plateau_threshold: Epochs without improvement before increasing.
        increase_factor: Multiplicative factor when increasing (> 1.0).
        decrease_factor: Multiplicative factor when decreasing (in (0, 1)).
        epochs_without_improvement: Counter for stagnant epochs.
        last_best_fitness: Best fitness value seen so far.
    """

    def __init__(
        self,
        value: float,
        max_value: float,
        min_value: float,
        plateau_threshold: int,
        increase_factor: float,
        decrease_factor: float,
    ):
        """
        Initialize the plateau-based scheduler.

        Args:
            value: Initial value.
            max_value: Upper bound.
            min_value: Lower bound.
            plateau_threshold: Epochs without improvement before value increase.
            increase_factor: Factor to multiply value by when stuck (> 1.0).
            decrease_factor: Factor to multiply value by when improving (in (0, 1)).

        Raises:
            ValueError: If factors are invalid or plateau_threshold is non-positive.
        """
        super().__init__(value, max_value, min_value)
        if plateau_threshold <= 0:
            raise ValueError(f"plateau_threshold ({plateau_threshold}) must be positive")
        if increase_factor <= 1.0:
            raise ValueError(f"increase_factor ({increase_factor}) must be > 1.0")
        if not (0 < decrease_factor < 1.0):
            raise ValueError(f"decrease_factor ({decrease_factor}) must be in (0, 1)")

        self.plateau_threshold: int = plateau_threshold
        self.increase_factor: float = increase_factor
        self.decrease_factor: float = decrease_factor
        self.epochs_without_improvement: int = 0
        self.last_best_fitness: float = float("-inf")

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            "("
            f"value={self.value},"
            f"min_value={self.min_value},"
            f"max_value={self.max_value},"
            f"plateau_threshold={self.plateau_threshold},"
            f"increase_factor={self.increase_factor},"
            f"decrease_factor={self.decrease_factor},"
            ")"
        )

    def __call__(self, best_fitness: float, **kwargs) -> float:
        """
        Adjust value based on fitness improvement.

        Args:
            best_fitness: Best fitness value in current epoch.
            **kwargs: Additional arguments (ignored).

        Returns:
            Updated value after adjustment.
        """
        raw: float = self.value

        if best_fitness > self.last_best_fitness:
            raw *= self.decrease_factor
            self.epochs_without_improvement = 0
            self.last_best_fitness = best_fitness
        else:
            self.epochs_without_improvement += 1
            if self.epochs_without_improvement >= self.plateau_threshold:
                raw *= self.increase_factor
                self.epochs_without_improvement = 0

        self.value = float(np.clip(raw, self.min_value, self.max_value))
        return self.value

    def reset(self, value: Optional[float] = None) -> None:
        """Reset scheduler state including fitness tracking."""
        super().reset(value)
        self.epochs_without_improvement = 0
        self.last_best_fitness = float("-inf")


class CosineScheduler(Scheduler):
    """Cosine annealing scheduler that cycles value between bounds.

    The value follows a cosine curve, smoothly transitioning between
    ``max_value`` and ``min_value`` over a configurable cycle length.

    Attributes:
        cycle_length: Number of epochs per complete cosine cycle.
    """

    def __init__(self, value: float, max_value: float, min_value: float, cycle_length: int):
        """Initialize the cosine annealing scheduler.

        Args:
            value: Initial value.
            max_value: Upper bound.
            min_value: Lower bound.
            cycle_length: Number of epochs for one complete cycle.

        Raises:
            ValueError: If cycle_length is not positive.
        """
        super().__init__(value, max_value, min_value)
        if cycle_length <= 0:
            raise ValueError(f"cycle_length ({cycle_length}) must be positive")
        self.cycle_length: int = cycle_length

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            "("
            f"value={self.value},"
            f"min_value={self.min_value},"
            f"max_value={self.max_value},"
            f"cycle_length={self.cycle_length}"
            ")"
        )

    def __call__(self, epoch: int, **kwargs) -> float:
        """Compute value using cosine annealing.

        Args:
            epoch: Current epoch number (0-indexed).
            **kwargs: Additional arguments (ignored).

        Returns:
            Updated value following cosine curve.
        """
        cycle_progress: float = (epoch % self.cycle_length) / self.cycle_length
        cosine_factor: float = 0.5 * (1 + np.cos(np.pi * cycle_progress))
        raw: float = self.min_value + (self.max_value - self.min_value) * cosine_factor

        self.value = float(raw)
        return self.value


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SCHEDULER_TYPES: Dict[str, Type[Scheduler]] = {
    "ExponentialScheduler": ExponentialScheduler,
    "PlateauScheduler": PlateauScheduler,
    "CosineScheduler": CosineScheduler,
}
