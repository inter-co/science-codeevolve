# ===--------------------------------------------------------------------------------------===#
#
# Part of the CodeEvolve Project, under the Apache License v2.0.
# See https://github.com/inter-co/science-codeevolve/blob/main/LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0
#
# ===--------------------------------------------------------------------------------------===#
#
# This file implements a the program and program database classes of CodeEvolve.
#
# ===--------------------------------------------------------------------------------------===#

from typing import Dict, List, Optional, Callable, Tuple

from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import random
import math

import numpy as np

from codeevolve.utils.cvt_utils import cvt, closest_centroid_idx


@dataclass
class Program:
    """Represents a program with execution results and evolutionary metadata.

    This class stores information about a program including its code, execution
    results, fitness metrics, and genealogical information for evolutionary
    programming applications.

    Attributes:
        id: Unique identifier for the program.
        code: The source code of the program.
        language: Programming language of the code.
        returncode: Exit code from program execution.
        output: Standard output from program execution.
        error: Error messages from program execution.
        warning: Warning messages from program execution.
        eval_metrics: Dictionary of evaluation metric names and values.
        fitness: Fitness score for evolutionary selection.
        parent_id: ID of the parent program in evolutionary lineage.
        iteration_found: Iteration number when program was discovered.
        generation: Generation number in evolutionary process.
        island_found: Island ID where program was found (for island models).
        prompt_id: ID of the prompt used to generate this program.
        inspiration_ids: List of IDs of programs used as inspiration.
        model_id: ID of the model that generated this program.
        model_msg: Message that generated this program.
        prog_msg: Formatted program message obtained with this program's info
        (see sampler.py and template.py).
        features: A dictionary of feature names to values, used for MAP-Elites.
        embedding: Word embedding of program code.
    """

    id: str
    code: str
    language: str

    returncode: Optional[int] = None
    output: Optional[str] = None
    error: Optional[str] = None
    warning: Optional[str] = None

    eval_metrics: Dict[str, float] = field(default_factory=dict)
    fitness: float = 0

    parent_id: Optional[str] = None
    iteration_found: Optional[int] = None
    generation: Optional[int] = None
    island_found: Optional[int] = None

    prompt_id: Optional[str] = None
    inspiration_ids: List[str] = field(default_factory=list)
    model_id: Optional[int] = None
    model_msg: Optional[str] = None
    prog_msg: Optional[str] = None

    features: Dict[str, float] = field(default_factory=dict)

    embedding: Optional[List[float]] = None

    def __repr__(self) -> str:
        """Returns a string representation of the Program instance.

        Returns:
            A formatted string showing key program attributes including ID,
            fitness, location found, and execution status.
        """
        return (
            f"{self.__class__.__name__}"
            "("
            f"id={self.id},"
            f"fitness={self.fitness:.8f},"
            f"island_found={self.island_found},"
            f"iteration_found={self.iteration_found},"
            f"returncode={self.returncode},"
            f"eval_metrics={self.eval_metrics}"
            ")"
        )


@dataclass
class EliteFeature:
    name: str
    min_val: float
    max_val: float
    num_bins: Optional[int] = None


class EliteMap(ABC):
    """Abstract base class for a MAP-Elites feature map.
    This class defines the interface for different MAP-Elites strategies,
    such as grid-based or CVT-based maps. Subclasses must implement methods
    for adding elites and retrieving their IDs.
    """

    def __init__(self, features: List[EliteFeature]):
        """Initializes the EliteMap.

        Args:
            features: list of EliteFeature objects.
        """
        self.features: List[EliteFeature] = features

    @abstractmethod
    def add_elite(self, prog: Program) -> None:
        """Adds a program to the elite map if it qualifies.
        The specific logic for determining if a program is an elite and where
        it belongs in the map is implemented by subclasses.
        Args:
            prog: The Program instance to consider adding.
        """
        pass

    @abstractmethod
    def get_elite_ids(self) -> List[str]:
        """Retrieves the IDs of all current elites in the map.
        Returns:
            A list of string identifiers for all elite programs.
        """
        pass


class GridEliteMap(EliteMap):
    """Implements MAP-Elites using a fixed grid in the feature space.
    Each dimension of the feature space is discretized into a fixed number of
    bins, creating a grid of cells. Each cell stores the id and fitness of the single best program
    (elite) found for that region of the feature space.
    """

    def __init__(self, features: List[EliteFeature]):
        """Initializes the GridEliteMap.
        Args:
            features: A list of MAPElitesFeature objects defining the dimensions
                      and binning of the feature space.
        """
        super().__init__(features)
        for f in self.features:
            if f.num_bins is None:
                raise ValueError("All features must have 'num_bins' defined for GridEliteMap.")
        self.num_cells: int = math.prod(f.num_bins for f in self.features)
        self.map: Dict[Tuple[int, ...], Tuple[str, float]] = {}

    def __repr__(self):
        """String representation of GridEliteMap."""
        return f"{self.__class__.__name__}(num_cells={self.num_cells})"

    def _get_cell_idx(self, prog: Program) -> Optional[Tuple[int, ...]]:
        """Calculates the cell index for a program based on its features.
        Args:
            prog: The Program instance to map to a cell.
        Returns:
            A tuple of indices representing the cell coordinates, or None if
            any feature is out of bounds.
        """
        indices: List[int] = []
        for feature in self.features:
            value: Optional[float] = prog.features.get(feature.name, None)
            if value is None:
                return None

            value = max(feature.min_val, min(value, feature.max_val))

            proportion: float = (value - feature.min_val) / (feature.max_val - feature.min_val)
            idx: int = int(proportion * (feature.num_bins - 1))
            indices.append(idx)

        return tuple(indices)

    def add_elite(self, prog: Program) -> None:
        """Adds a program to the MAP-Elites map if it's a new or better elite.
        This method checks if the program's feature cell is empty or if the
        program has a higher fitness score than the existing elite in that cell.
        If so, it updates the map.
        Args:
            prog: The program to consider for the elite map.
        """
        cell_idx: Optional[Tuple[int, ...]] = self._get_cell_idx(prog)
        if cell_idx is None:
            return None

        elite_id, elite_fitness = self.map.get(cell_idx, (None, None))
        if elite_id is None or prog.fitness > elite_fitness:
            self.map[cell_idx] = (prog.id, prog.fitness)

    def get_elite_ids(self) -> List[str]:
        """Retrieves the IDs of all elites currently in the grid map.
        Returns:
            A list of string identifiers for all stored elite programs.
        """
        return [pid for pid, _ in self.map.values()]


class CVTEliteMap(EliteMap):
    """Implements MAP-Elites using Centroidal Voronoi Tesselations (CVT).
    Instead of a fixed grid, this approach partitions the feature space into
    a set of regions (Voronoi cells) defined by centroids. Each centroid
    maintains the id and fitness of the best program found within its region.
    """

    def __init__(
        self,
        features: List[EliteFeature],
        num_centroids: int,
        num_init_samples: int,
        max_iter: int = 300,
        tolerance: float = 1e-6,
    ):
        """Initializes the CVTEliteMap by generating centroids.
        Args:
            features: A list of MAPElitesFeature objects defining the feature space.
            num_centroids: The number of centroids to create.
            num_init_samples: The number of samples to use for the CVT algorithm.
            max_iter: Maximum iterations for the CVT algorithm.
            tolerance: Convergence tolerance for the CVT algorithm.
        """
        super().__init__(features=features)
        feature_bounds: List[Tuple[float, float]] = [(f.min_val, f.max_val) for f in features]
        self.centroids: np.ndarray = cvt(
            num_centroids, num_init_samples, feature_bounds, max_iter, tolerance
        )
        self.map: Dict[int, Tuple[str, float]] = {}

    def __repr__(self):
        """String representation of CVTEliteMap."""
        return f"{self.__class__.__name__}(num_centroids={len(self.centroids)})"

    def add_elite(self, prog: Program) -> None:
        """Adds a program to the MAP-Elites map if it's a new or better elite.
        This method checks if the program's centroid is empty or if the
        program has a higher fitness score than the existing elite in that centroid.
        If so, it updates the map.
        Args:
            prog: The program to consider for the elite map.
        """
        prog_feats: np.ndarray = np.zeros(len(self.features))
        for i, feature in enumerate(self.features):
            value: Optional[float] = prog.features.get(feature.name, None)
            if value is None:
                return None

            prog_feats[i] = value

        centroid_idx: int = closest_centroid_idx(prog_feats, self.centroids)

        elite_id, elite_fitness = self.map.get(centroid_idx, (None, None))
        if elite_id is None or prog.fitness > elite_fitness:
            self.map[centroid_idx] = (prog.id, prog.fitness)

    def get_elite_ids(self) -> List[str]:
        """Retrieves the IDs of all elites currently in the CVT map.
        Returns:
            A list of string identifiers for all stored elite programs.
        """
        return [pid for pid, _ in self.map.values()]


class ProgramDatabase:
    """Manages a collection of programs for an evolutionary algorithm.
    This class can operate in two modes:
    1.  Standard EA Mode: Manages a population of a fixed maximum size
        (`max_alive`), where new individuals replace the least fit ones.
    2.  MAP-Elites Mode: Enabled by providing a list of `features`. In this mode,
        it maintains a map of "elites," which are the highest-performing programs
        found for each cell in a discrete feature space. The concept of `max_alive`
        is ignored, as the population consists of the elites in the map.
    """

    def __init__(
        self,
        id: int,
        seed: Optional[int] = None,
        max_alive: Optional[int] = None,
        features: Optional[List[EliteFeature]] = None,
        elite_map_type: Optional[str] = None,
        **elite_map_kwargs,
    ):
        """Initializes the program database.

        Args:
            id: Unique identifier for this database instance.
            max_alive: Maximum number of programs to keep alive simultaneously.
                      If None, no limit is enforced.
            seed: Random seed for reproducible selection operations.
        """
        self.id = id
        self.seed: Optional[int] = seed
        self.random_state: random.Random = random.Random()
        if self.seed:
            self.random_state.seed(self.seed)

        self.programs: Dict[str, Program] = {}
        self.roots: List[str] = []

        self.num_alive: int = 0
        self.max_alive: Optional[int] = max_alive
        self.is_alive: Dict[str, bool] = {}

        self.best_prog_id: Optional[str] = None
        self.worst_prog_id: Optional[str] = None

        self.has_migrated: Dict[str, bool] = {}

        self._pids_pool_cache: List[str] = []
        self._rank_cache: Dict[str, int] = {}

        self.elite_map_type: Optional[str] = elite_map_type.lower() if elite_map_type else None
        self.elite_map: Optional[EliteMap] = None

        if features and self.elite_map_type:
            match self.elite_map_type:
                case "grid":
                    self.elite_map = GridEliteMap(features=features)
                case "cvt":
                    self.elite_map = CVTEliteMap(features=features, **elite_map_kwargs)
                case _:
                    raise ValueError(f"Invalid elite_map_type {self.elite_map_type}.")

        self._selection_methods: Dict[str, Callable] = {
            "random": self.random_selection,
            "roulette": self.roulette_selection,
            "tournament": self.tournament_selection,
            "best": self.best_selection,
        }

    def __repr__(self) -> str:
        """Returns a string representation of the database.

        Returns:
            A formatted string showing database statistics.
        """
        db_str: str = (
            f"{self.__class__.__name__}"
            "("
            f"id={self.id},"
            f"total_programs={len(self.programs)},"
        )
        if getattr(self, "elite_map", None) is not None:
            db_str += f"mode=MAP-Elites," f"elite_map={self.elite_map}"
        else:
            db_str += f"mode=standard," f"num_alive={self.num_alive}," f"max_alive={self.max_alive}"
        db_str += ")"
        return db_str

    # program management
    ## TODO: improve insertion logic if we are to make more insertions per epoch
    # (currently each insertion takes NlogN worst case, we can use bisect or
    # heapq to improve this).

    def _update_caches(self) -> None:
        """Updates internal caches for programs and their fitness rankings.

        This method rebuilds the program cache, sorts programs by fitness,
        updates rank mappings, and identifies best and worst programs.
        """
        if getattr(self, "map", None) is not None:
            self._pids_pool_cache = self.elite_map.get_elite_ids()
        else:
            self._pids_pool_cache = [pid for pid, is_alive in self.is_alive.items() if is_alive]

        if not self._pids_pool_cache:
            self._rank_cache = {}
            return

        desc_pids: List[str] = sorted(
            self._pids_pool_cache, key=lambda pid: self.programs[pid].fitness, reverse=True
        )
        self._rank_cache = {pid: i for i, pid in enumerate(desc_pids)}

    def add(self, prog: Program) -> None:
        """Adds a program to the database.

        Args:
            prog: The Program instance to add to the database.

        Raises:
            ValueError: If a program with the same ID already exists.
        """
        if prog.id in self.programs:
            raise ValueError(f"ID {prog.id} is already in db.")

        self.programs[prog.id] = prog
        if prog.parent_id is None:
            self.roots.append(prog.id)

        if self.elite_map is not None:
            self.is_alive[prog.id] = True
            self.elite_map.add_elite(prog)
            self.num_alive += 1  # every new program is alive in elite_map mode
        else:
            if self.max_alive is None or self.num_alive < self.max_alive:
                self.is_alive[prog.id] = True
                self.num_alive += 1
            elif prog.fitness >= self.programs[self.worst_prog_id].fitness:
                self.is_alive[self.worst_prog_id] = False
                self.is_alive[prog.id] = True
            else:
                self.is_alive[prog.id] = False

        if self.best_prog_id is None or self.programs[self.best_prog_id].fitness < prog.fitness:
            self.best_prog_id = prog.id

        if self.is_alive[prog.id] and (
            self.worst_prog_id is None or prog.fitness < self.programs[self.worst_prog_id].fitness
        ):
            self.worst_prog_id = prog.id

        self._update_caches()

    # parent selection

    def random_selection(self, pids_pool: List[str], k: int = 1) -> Optional[List[Program]]:
        """Selects k programs uniformly at random from the pids_pool.

        Args:
            pids_pool: List of target program ids.
            k: Number of programs to select.

        Returns:
            List of selected Program instances, or None if k=0 or no pids_pool is empty.
        """
        if k and len(pids_pool):
            pids: List[str] = self.random_state.choices(pids_pool, k=min(len(pids_pool), k))
            return [self.programs[pid] for pid in pids]
        else:
            return None

    def roulette_selection(
        self, pids_pool: List[str], k: int = 1, roulette_by_rank: bool = False
    ) -> Optional[List[Program]]:
        """Selects k programs according to a roulette wheel selection from the pids_pool.

        Args:
            pids_pool: List of target program ids.
            k: Number of programs to select.
            roulette_by_rank: If True, use rank-based weights; if False, use
            fitness-based weights.

        Returns:
            List of selected Program instances, or None if k=0 or no pids_pool is empty.
        """
        if k and len(pids_pool):
            weights: List[float] = [1 / len(pids_pool) for _ in range(len(pids_pool))]

            if roulette_by_rank:
                weights = [1 / (1 + self._rank_cache[pid]) for pid in pids_pool]
                wsum: float = sum(weights)
                weights = [weight / wsum for weight in weights]
            else:
                fitness_list: List[float] = [self.programs[pid].fitness for pid in pids_pool]
                fit_sum: float = sum(fitness_list)
                if fit_sum > 0:
                    weights = [fit / fit_sum for fit in fitness_list]

            pids: List[str] = self.random_state.choices(
                pids_pool, weights, k=min(len(pids_pool), k)
            )
            return [self.programs[pid] for pid in pids]
        else:
            return None

    def tournament_selection(
        self,
        pids_pool: List[str],
        k: int = 1,
        tournament_size: int = 3,
    ) -> Optional[List[Program]]:
        """Selects k programs according to a tournament from the pids_pool.

        Args:
            pids_pool: List of target program ids.
            k: Number of programs to select.
            tournament_size: Number of programs in each tournament.

        Returns:
            List of selected Program instances, or None if k=0 or no pids_pool is empty.
        """
        if k and tournament_size and len(pids_pool):
            tournament_pids: List[str] = self.random_state.choices(
                pids_pool, k=min(tournament_size, len(pids_pool))
            )
            best_pids: List[str] = sorted(
                tournament_pids,
                key=lambda pid: self._rank_cache[pid],
                reverse=False,
            )[:k]

            return [self.programs[pid] for pid in best_pids]
        else:
            return None

    def best_selection(self, pids_pool: List[str], k: int = 1) -> Optional[List[Program]]:
        """Selects the k best programs by fitness.

        Args:
            k: Number of programs to select.
            restricted_pids: List of program IDs to exclude from selection.

        Returns:
            List of the k best Program instances, or None if no valid programs.
        """
        return self.tournament_selection(pids_pool=pids_pool, k=k, tournament_size=len(pids_pool))

    def sample(
        self, selection_policy: str, num_inspirations: int = 0, **kwargs
    ) -> Tuple[Optional[Program], List[Program]]:
        """Samples a parent and inspiration programs using a selection policy.
        Args:
            selection_policy: Method to use ('random', 'roulette', 'tournament', 'best').
            num_inspirations: Number of inspiration programs to sample.
            **kwargs: Additional arguments for the selection method.
        Returns:
            A tuple of (parent Program, list of inspiration Programs).
        """
        selection_func: Callable = self._selection_methods.get(selection_policy)
        if not selection_func:
            raise ValueError(f"Selection policy must be in {self._selection_methods.keys()}")

        self._update_caches()
        if not self._pids_pool_cache:
            return None, []

        parent_progs: Optional[List[Program]] = selection_func(
            pids_pool=self._pids_pool_cache, k=1, **kwargs
        )
        parent: Optional[Program] = parent_progs[0] if parent_progs else None

        inspirations: List[Program] = []
        if parent and num_inspirations > 0:
            inspiration_pool: List[str] = [pid for pid in self._pids_pool_cache if pid != parent.id]
            if inspiration_pool:
                inspirations = selection_func(
                    pids_pool=inspiration_pool, k=num_inspirations, **kwargs
                )
        return parent, inspirations

    def get_migrants(self, migration_rate: float) -> List[Program]:
        """Returns a list of migrants according a given migration rate.

        Only migrates programs who haven't migrated and are not
        themselves migrants. Might lead to fewer than
        migration_rate*self.num_alive migrants.

        Args:
            migration rate: controls the number of migrants

        Returns:
            List of migrant programs.
        """
        self._update_caches()

        eligible_progs: List[str] = [
            pid
            for pid in self._pids_pool_cache
            if self.programs[pid].island_found == self.id and not self.has_migrated.get(pid)
        ]
        if not eligible_progs:
            return []

        num_migrants: int = min(
            len(eligible_progs), int(max(1, migration_rate * len(self._pids_pool_cache)))
        )

        migrant_pids: List[str] = sorted(eligible_progs, key=lambda pid: self._rank_cache[pid])[
            :num_migrants
        ]
        return [self.programs[pid] for pid in migrant_pids]
