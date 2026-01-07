# ===--------------------------------------------------------------------------------------===#
#
# Part of the CodeEvolve Project, under the Apache License v2.0.
# See https://github.com/inter-co/science-codeevolve/blob/main/LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0
#
# ===--------------------------------------------------------------------------------------===#
#
# This file implements exploration rate schedulers.
#
# ===--------------------------------------------------------------------------------------===#

from typing import Dict, Optional
from abc import ABC, abstractmethod

import numpy as np


class ExplorationRateScheduler(ABC):
    """
    Abstract base class for exploration rate schedulers in genetic algorithms.

    Exploration rate schedulers dynamically adjust the exploration-exploitation
    balance during evolutionary optimization. Higher rates encourage exploration
    of the search space, while lower rates favor exploitation of known solutions.

    Attributes:
        exploration_rate: Current exploration rate value.
        max_rate: Maximum allowed exploration rate (upper bound).
        min_rate: Minimum allowed exploration rate (lower bound).
    """

    def __init__(self, exploration_rate: float, max_rate: float, min_rate: float):
        """
        Initialize the exploration rate scheduler.

        Args:
            exploration_rate: Initial exploration rate.
            max_rate: Maximum exploration rate to clip to.
            min_rate: Minimum exploration rate to clip to.

        Raises:
            ValueError: If min_rate > max_rate or exploration_rate is outside bounds.
        """
        if min_rate > max_rate:
            raise ValueError(f"min_rate ({min_rate}) must be <= max_rate ({max_rate})")
        if not (min_rate <= exploration_rate <= max_rate):
            raise ValueError(
                f"exploration_rate ({exploration_rate}) must be between "
                f"min_rate ({min_rate}) and max_rate ({max_rate})"
            )

        self.exploration_rate: float = exploration_rate
        self.max_rate: float = max_rate
        self.min_rate: float = min_rate

    @abstractmethod
    def __call__(self, **kwargs) -> float:
        """
        Compute and update the exploration rate.

        Args:
            **kwargs: Scheduler-specific arguments (e.g., epoch, fitness).

        Returns:
            Updated exploration rate after applying scheduling logic.
        """
        pass

    def reset(self, exploration_rate: Optional[float] = None) -> None:
        """
        Reset the scheduler to its initial state.

        Args:
            exploration_rate: New initial rate. If None, keeps current rate.
        """
        if exploration_rate is not None:
            if not (self.min_rate <= exploration_rate <= self.max_rate):
                raise ValueError(
                    f"exploration_rate ({exploration_rate}) must be between "
                    f"min_rate ({self.min_rate}) and max_rate ({self.max_rate})"
                )
            self.exploration_rate = exploration_rate


class ExponentialDecayScheduler(ExplorationRateScheduler):
    """
    Exponential decay scheduler that reduces exploration rate over time.

    The exploration rate decays exponentially according to:
        rate(t) = rate(0) * (decay_weight ^ t)

    This scheduler is useful when you want gradual reduction in exploration
    as the algorithm progresses, transitioning from exploration to exploitation.

    Attributes:
        decay_weight: Multiplicative decay factor applied each epoch (0 < decay_weight < 1).
    """

    def __init__(
        self, exploration_rate: float, max_rate: float, min_rate: float, decay_weight: float
    ):
        """
        Initialize the exponential decay scheduler.

        Args:
            exploration_rate: Initial exploration rate.
            max_rate: Maximum exploration rate bound.
            min_rate: Minimum exploration rate bound.
            decay_weight: Decay factor per epoch (typically 0.9-0.99).

        Raises:
            ValueError: If decay_weight is not in (0, 1].
        """
        super().__init__(exploration_rate, max_rate, min_rate)
        if not (0 < decay_weight <= 1):
            raise ValueError(f"decay_weight ({decay_weight}) must be in (0, 1]")
        self.decay_weight: float = decay_weight
        self.initial_rate: float = exploration_rate

    def __repr__(self) -> str:
        """Returns a string representation of the ExponentialDecayScheduler instance.

        Returns:
            A formatted string showing the scheduler's configuration.
        """
        return (
            f"{self.__class__.__name__}"
            "("
            f"exploration_rate={self.exploration_rate},"
            f"min_rate={self.min_rate},"
            f"max_rate={self.max_rate},"
            f"decay_weight={self.decay_weight},"
            f"initial_rate={self.initial_rate}"
            ")"
        )

    def __call__(self, epoch: int, **kwargs) -> float:
        """
        Compute exploration rate with exponential decay.

        Args:
            epoch: Current epoch number (0-indexed).
            **kwargs: Additional arguments (ignored).

        Returns:
            Updated exploration rate after decay.
        """
        rate: float = self.initial_rate * (self.decay_weight**epoch)
        self.exploration_rate = float(np.clip(rate, self.min_rate, self.max_rate))
        return self.exploration_rate

    def reset(self, exploration_rate: Optional[float] = None) -> None:
        """Reset scheduler and update initial rate if provided."""
        super().reset(exploration_rate)
        if exploration_rate is not None:
            self.initial_rate = exploration_rate


class PlateauScheduler(ExplorationRateScheduler):
    """
    Adaptive scheduler that adjusts exploration rate based on fitness improvements.

    This scheduler monitors fitness progress and adapts the exploration rate:
    - Decreases rate when fitness improves (exploit good solutions)
    - Increases rate after plateau threshold (explore when stuck)

    This creates a dynamic balance that responds to optimization progress.

    Attributes:
        plateau_threshold: Number of epochs without improvement before increasing rate.
        increase_factor: Multiplicative factor when increasing rate (> 1.0).
        decrease_factor: Multiplicative factor when decreasing rate (< 1.0).
        epochs_without_improvement: Counter for stagnant epochs.
        last_best_fitness: Best fitness value seen so far.
    """

    def __init__(
        self,
        exploration_rate: float,
        max_rate: float,
        min_rate: float,
        plateau_threshold: int,
        increase_factor: float,
        decrease_factor: float,
    ):
        """
        Initialize the plateau-based scheduler.

        Args:
            exploration_rate: Initial exploration rate.
            max_rate: Maximum exploration rate bound.
            min_rate: Minimum exploration rate bound.
            plateau_threshold: Epochs without improvement before rate increase.
            increase_factor: Factor to multiply rate by when stuck (> 1.0).
            decrease_factor: Factor to multiply rate by when improving (< 1.0).

        Raises:
            ValueError: If factors are invalid or plateau_threshold is non-positive.
        """
        super().__init__(exploration_rate, max_rate, min_rate)
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
        """Returns a string representation of the PlateauScheduler instance.

        Returns:
            A formatted string showing the scheduler's configuration.
        """
        return (
            f"{self.__class__.__name__}"
            "("
            f"exploration_rate={self.exploration_rate},"
            f"min_rate={self.min_rate},"
            f"max_rate={self.max_rate},"
            f"plateau_threshold={self.plateau_threshold},"
            f"increase_factor={self.increase_factor},"
            f"decrease_factor={self.decrease_factor},"
            ")"
        )

    def __call__(self, best_fitness: float, **kwargs) -> float:
        """
        Adjust exploration rate based on fitness improvement.

        Args:
            best_fitness: Best fitness value in current epoch.
            **kwargs: Additional arguments (ignored).

        Returns:
            Updated exploration rate after adjustment.
        """
        rate: float = self.exploration_rate

        if best_fitness > self.last_best_fitness:
            rate *= self.decrease_factor
            self.epochs_without_improvement = 0
            self.last_best_fitness = best_fitness
        else:
            self.epochs_without_improvement += 1
            if self.epochs_without_improvement >= self.plateau_threshold:
                rate *= self.increase_factor
                self.epochs_without_improvement = 0

        self.exploration_rate = float(np.clip(rate, self.min_rate, self.max_rate))
        return self.exploration_rate

    def reset(self, exploration_rate: Optional[float] = None) -> None:
        """Reset scheduler state including fitness tracking."""
        super().reset(exploration_rate)
        self.epochs_without_improvement = 0
        self.last_best_fitness = float("-inf")


class CosineScheduler(ExplorationRateScheduler):
    """Cosine annealing scheduler that oscillates exploration rate periodically.
    
    This scheduler uses a cosine wave to smoothly vary the exploration rate
    between min_rate and max_rate over a fixed cycle length, allowing for
    periodic transitions between exploration and exploitation.
    
    Attributes:
        cycle_length: Number of epochs for one complete cosine cycle.
    """

    def __init__(
        self, exploration_rate: float, max_rate: float, min_rate: float, cycle_length: int
    ):
        """Initialize the cosine annealing scheduler.
        
        Args:
            exploration_rate: Initial exploration rate.
            max_rate: Maximum exploration rate bound.
            min_rate: Minimum exploration rate bound.
            cycle_length: Number of epochs per complete cosine cycle.
            
        Raises:
            ValueError: If cycle_length is not positive.
        """
        super().__init__(exploration_rate, max_rate, min_rate)
        if cycle_length <= 0:
            raise ValueError(f"cycle_length ({cycle_length}) must be positive")
        self.cycle_length = cycle_length

    def __call__(self, epoch: int, **kwargs) -> float:
        """Compute exploration rate using cosine annealing.
        
        Args:
            epoch: Current epoch number.
            **kwargs: Additional arguments (ignored).
            
        Returns:
            Updated exploration rate following cosine wave.
        """
        cycle_progress: float = (epoch % self.cycle_length) / self.cycle_length
        cosine_factor: float = 0.5 * (1 + np.cos(np.pi * cycle_progress))
        rate: float = self.min_rate + (self.max_rate - self.min_rate) * cosine_factor

        self.exploration_rate = float(rate)
        return self.exploration_rate


SCHEDULER_TYPES: Dict[str, ExplorationRateScheduler] = {
    "ExponentialDecayScheduler": ExponentialDecayScheduler,
    "PlateauScheduler": PlateauScheduler,
    "CosineScheduler": CosineScheduler,
}
