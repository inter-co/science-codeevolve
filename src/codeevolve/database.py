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
import random


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


class ProgramDatabase:
    """A database for managing programs in an evolutionary programming system.

    This class maintains a collection of programs with fitness-based survival
    mechanics, supporting various selection methods for evolutionary algorithms.
    It manages program lifecycles, fitness rankings, and provides selection
    methods for parent and inspiration sampling.
    """

    def __init__(self, id: int, max_alive: Optional[int] = None, seed: Optional[int] = None):
        """Initializes the program database.

        Args:
            id: Unique identifier for this database instance.
            max_alive: Maximum number of programs to keep alive simultaneously.
                      If None, no limit is enforced.
            seed: Random seed for reproducible selection operations.
        """
        self.id = id
        self.max_alive: Optional[int] = max_alive

        self.seed: Optional[int] = seed
        self.random_state: random.Random = random.Random()
        if self.seed:
            self.random_state.seed(self.seed)

        self.programs: Dict[str, Program] = {}
        self.roots: List[str] = []
        self.is_alive: Dict[str, bool] = {}
        self.num_alive: int = 0

        self.alive_pid_cache: List[str] = []
        self.alive_rank_cache: Dict[str, int] = {}
        self.best_prog_id: Optional[str] = None
        self.worst_prog_id: Optional[str] = None

        self.has_migrated: Dict[str, bool] = {}

        self._selection_methods: Dict[str, Callable] = {
            "random": self.random_selection,
            "roulette": self.roulette_selection,
            "tournament": self.tournament_selection,
            "best": self.best_selection,
        }

    def __repr__(self) -> str:
        """Returns a string representation of the database.

        Returns:
            A formatted string showing database statistics including number
            of alive programs, maximum capacity, total programs, and root programs.
        """
        return (
            f"{self.__class__.__name__}"
            "("
            f"num_alive={self.num_alive},"
            f"max_alive={self.max_alive},"
            f"total={len(self.programs)},"
            f"roots={len(self.roots)}"
            ")"
        )

    # program management
    ## TODO: improve insertion logic if we are to make more insertions per epoch
    # (currently each insertion takes NlogN worst case, we can use bisect or
    # heapq to improve this).

    def update_alive_caches(self) -> None:
        """Updates internal caches for alive programs and their fitness rankings.

        This method rebuilds the alive program cache, sorts programs by fitness,
        updates rank mappings, and identifies best and worst programs.
        """
        self.alive_pid_cache = [pid for pid in self.programs.keys() if self.is_alive[pid]]

        desc_alive_pids: List[str] = sorted(
            self.alive_pid_cache,
            key=lambda pid: self.programs[pid].fitness,
            reverse=True,
        )

        self.alive_rank_cache = {pid: i for i, pid in enumerate(desc_alive_pids)}
        self.best_prog_id = desc_alive_pids[0]
        self.worst_prog_id = desc_alive_pids[-1]

    def add(self, prog: Program) -> None:
        """Adds a program to the database with survival-of-the-fittest logic.

        If the database is at capacity, the new program replaces the worst
        performing program only if it has better fitness.

        Args:
            prog: The Program instance to add to the database.

        Raises:
            ValueError: If a program with the same ID already exists.
        """
        if self.programs.get(prog.id, None):
            raise ValueError(f"ID {prog.id} is already in db.")

        self.programs[prog.id] = prog
        if prog.parent_id is None:
            self.roots.append(prog.id)

        if self.max_alive is None or self.num_alive < self.max_alive:
            self.is_alive[prog.id] = True
            self.num_alive += 1
            self.update_alive_caches()
        elif prog.fitness >= self.programs[self.worst_prog_id].fitness:
            self.is_alive[self.worst_prog_id] = 0
            self.is_alive[prog.id] = True
            self.update_alive_caches()
        else:
            self.is_alive[prog.id] = False

    # parent selection

    def random_selection(
        self, k: int = 1, restricted_pids: Optional[List[str]] = None
    ) -> Optional[List[Program]]:
        """Selects programs randomly from the alive population.

        Args:
            k: Number of programs to select.
            restricted_pids: List of program IDs to exclude from selection.

        Returns:
            List of selected Program instances, or None if k=0 or no valid programs.
        """
        if k:
            self.update_alive_caches()
            pids_pool: List[str] = self.alive_pid_cache
            if restricted_pids and len(restricted_pids):
                pids_pool = [pid for pid in pids_pool if pid not in restricted_pids]
            if not len(pids_pool):
                return None

            pids: List[str] = self.random_state.choices(pids_pool, k=min(len(pids_pool), k))
            return [self.programs[pid] for pid in pids]
        else:
            return None

    def roulette_selection(
        self,
        k: int = 1,
        roulette_by_rank: bool = False,
        restricted_pids: Optional[List[str]] = None,
    ) -> Optional[List[Program]]:
        """Selects programs using roulette wheel selection based on fitness or rank.

        Args:
            k: Number of programs to select.
            roulette_by_rank: If True, use rank-based weights; if False, use
            fitness-based weights.
            restricted_pids: List of program IDs to exclude from selection.

        Returns:
            List of selected Program instances, or None if k=0 or no valid programs.
        """
        if k:
            self.update_alive_caches()
            pids_pool: List[str] = self.alive_pid_cache
            if restricted_pids and len(restricted_pids):
                pids_pool = [pid for pid in pids_pool if pid not in restricted_pids]
            if not len(pids_pool):
                return None

            weights: List[float] = [1 / len(pids_pool) for _ in range(len(pids_pool))]

            if roulette_by_rank:
                weights = [1 / (1 + self.alive_rank_cache[pid]) for pid in pids_pool]
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
        k: int = 1,
        tournament_size: int = 3,
        restricted_pids: Optional[List[str]] = None,
    ) -> Optional[List[Program]]:
        """Selects programs using tournament selection.

        Randomly samples a tournament group and selects the best performers.

        Args:
            k: Number of programs to select.
            tournament_size: Number of programs in each tournament.
            restricted_pids: List of program IDs to exclude from selection.

        Returns:
            List of selected Program instances, or None if k=0 or tournament_size=0.
        """
        if k and tournament_size:
            self.update_alive_caches()
            pids_pool: List[str] = self.alive_pid_cache
            if restricted_pids and len(restricted_pids):
                pids_pool = [pid for pid in pids_pool if pid not in restricted_pids]
            if not len(pids_pool):
                return None

            tournament_pids: List[str] = self.random_state.choices(
                pids_pool, k=min(len(pids_pool), tournament_size)
            )
            best_pids: List[str] = sorted(
                tournament_pids,
                key=lambda pid: self.alive_rank_cache[pid],
                reverse=False,
            )[:k]

            return [self.programs[pid] for pid in best_pids]
        else:
            return None

    def best_selection(
        self, k: int = 1, restricted_pids: Optional[List[str]] = None
    ) -> Optional[List[Program]]:
        """Selects the k best programs by fitness.

        Args:
            k: Number of programs to select.
            restricted_pids: List of program IDs to exclude from selection.

        Returns:
            List of the k best Program instances, or None if no valid programs.
        """
        return self.tournament_selection(
            k=k, tournament_size=self.num_alive, restricted_pids=restricted_pids
        )

    def sample(
        self, selection_policy: str, num_inspirations: int = 0, **kwargs
    ) -> Tuple[Optional[Program], List[Program]]:
        """Samples a parent program and inspiration programs using the specified
        selection policy.

        Args:
            selection_policy: Name of the selection method to use ('random', 'roulette',
                            'tournament', or 'best').
            num_inspirations: Number of inspiration programs to sample.
            **kwargs: Additional arguments passed to the selection method.

        Returns:
            A tuple containing:
                - The selected parent Program (or None if no programs available)
                - List of inspiration Programs (empty if num_inspirations=0)

        Raises:
            ValueError: If no live programs exist or selection_policy is invalid.
        """
        if self.num_alive == 0:
            raise ValueError("No live programs found in database.")
        elif selection_policy not in self._selection_methods.keys():
            raise ValueError(f"Selection policy must be in {self._selection_methods.keys()}")

        sampled_parent: Optional[Program] = self._selection_methods[selection_policy](k=1, **kwargs)
        if sampled_parent:
            sampled_parent = sampled_parent[0]

        inspirations: List[Program] = []
        if num_inspirations:
            restricted_pids: List[Program] = [sampled_parent.id] if sampled_parent else []
            inspirations = self._selection_methods[selection_policy](
                k=num_inspirations, restricted_pids=restricted_pids, **kwargs
            )

        return sampled_parent, inspirations
