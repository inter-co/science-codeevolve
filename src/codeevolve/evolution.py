# ===--------------------------------------------------------------------------------------===#
#
# Part of the CodeEvolve Project, under the Apache License v2.0.
# See https://github.com/inter-co/science-codeevolve/blob/main/LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0
#
# ===--------------------------------------------------------------------------------------===#
#
# This file implements the main evolutionary loop of CodeEvolve.
# Refactored for modularity and readability.
#
# ===--------------------------------------------------------------------------------------===#

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from uuid import uuid4
import logging
from pathlib import Path
import yaml
import numpy as np

from codeevolve.database import Program, ProgramDatabase, EliteFeature
from codeevolve.lm.openai import OpenAIEnsemble, OpenAIEmbedding
from codeevolve.evaluator import Evaluator
from codeevolve.prompt.sampler import PromptSampler, format_prog_msg

from codeevolve.islands.sync import GlobalSyncData
from codeevolve.islands.graph import IslandCommunicationData
from codeevolve.islands.migration import sync_migrate

from codeevolve.scheduler import ExplorationRateScheduler, SCHEDULER_TYPES
from codeevolve.utils.parsing import apply_diff
from codeevolve.utils.logging import get_logger, get_elapsed_time
from codeevolve.utils.ckpt import save_ckpt, load_ckpt, save_run_metadata
from codeevolve.utils.constants import (
    DEFAULT_EVOLVE_START_MARKER,
    DEFAULT_EVOLVE_END_MARKER,
    DEFAULT_PROMPT_START_MARKER,
    DEFAULT_PROMPT_END_MARKER,
    DEFAULT_EVAL_TIMEOUT_S,
    DEFAULT_MAX_MEM_BYTES,
    DEFAULT_MEM_CHECK_INTERVAL_S,
    DEFAULT_MIGRATION_INTERVAL,
    DEFAULT_MIGRATION_RATE,
    DEFAULT_MAX_LOG_MSG_SIZE,
    BEST_SOLUTION_FILE,
    BEST_PROMPT_FILE,
    LANGUAGE_TO_EXTENSION,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_markers(evolve_config: Dict[str, Any]) -> Tuple[str, str, str, str]:
    """Extracts code and prompt block markers from configuration.

    Args:
        evolve_config: Evolution-specific configuration.

    Returns:
        Tuple of (evolve_start, evolve_end, prompt_start, prompt_end) markers.
    """
    return (
        evolve_config.get("evolve_start_marker", DEFAULT_EVOLVE_START_MARKER),
        evolve_config.get("evolve_end_marker", DEFAULT_EVOLVE_END_MARKER),
        evolve_config.get("mp_start_marker", DEFAULT_PROMPT_START_MARKER),
        evolve_config.get("mp_end_marker", DEFAULT_PROMPT_END_MARKER),
    )


# ---------------------------------------------------------------------------
# Evolutionary loop functions
# ---------------------------------------------------------------------------


def early_stopping_check(
    num_islands: int,
    improved_local_fitness: bool,
    global_data: GlobalSyncData,
    logger: logging.Logger,
) -> None:
    """Coordinates early stopping decision across all islands.

    This function implements a distributed early stopping mechanism where
    all islands must report no improvement before the early stopping counter
    is incremented. Uses barriers to ensure all islands participate in the decision.

    Args:
        num_islands: Total number of islands in the system.
        improved_local_fitness: Whether this island improved its best fitness.
        global_data: Shared data structures for coordination.
        logger: Logger instance for this island.
    """
    if not improved_local_fitness:
        with global_data.lock:
            # indicates if an island didnt improve locally
            global_data.early_stop_aux.value += 1

    logger.info("Waiting for all islands to report progress...")
    global_data.barrier.wait()
    logger.info("All islands synced.")

    with global_data.lock:
        # first to arrive is the leader, makes the early stop check,
        # and then sets the aux to -1 so no other island can do the same
        if global_data.early_stop_aux.value != -1:
            if global_data.early_stop_aux.value == num_islands:
                global_data.early_stop_counter.value += 1
            else:
                global_data.early_stop_counter.value = 0

            global_data.early_stop_aux.value = (
                -1
            )  # flag for other islands to not repeat the above code

    logger.info("Waiting for other islands to finish early stopping check...")
    global_data.barrier.wait()
    logger.info("All islands synced.")

    global_data.early_stop_aux.value = 0  # reset to zero


def select_parents(
    sol_db: ProgramDatabase,
    prompt_db: ProgramDatabase,
    init_sol: Program,
    init_prompt: Program,
    evolve_config: Dict[str, Any],
    gen_init_pop: bool,
    exploration: bool,
    logger: logging.Logger,
) -> Tuple[Program, Program, List[Program]]:
    """
    Select parent solution, prompt, and inspiration programs for evolution.

    This function implements the selection phase of the evolutionary algorithm,
    choosing parents based on the current mode (initialization, exploration, or
    exploitation). Selection policies are configurable and can include fitness-based,
    novelty-based, or random selection.

    Selection Modes:
        - **Initialization**: Returns the initial solution and prompt to seed population
        - **Exploration**: Samples uniformly at random to encourage diversity
        - **Exploitation**: Uses configured selection policy (e.g., tournament, roulette)

    Args:
        sol_db: Database containing solution programs with fitness scores
        prompt_db: Database containing prompt programs
        init_sol: Initial solution program used during population initialization
        init_prompt: Initial prompt program used during population initialization
        evolve_config: Configuration dictionary containing:
            - selection_policy: Policy name (e.g., 'tournament', 'roulette')
            - selection_kwargs: Policy-specific parameters
            - num_inspirations: Number of inspiration programs to sample
            - migration_interval: Epochs between migrations (default: 20)
        gen_init_pop: Whether currently generating initial population
        exploration: Whether in exploration mode (vs exploitation)
        logger: Logger for recording selection decisions

    Returns:
        Tuple containing:
            - parent_sol: Selected parent solution program
            - parent_prompt: Selected parent prompt program
            - inspirations: List of inspiration programs
    """
    logger.info("=== SELECTION STEP ===")

    parent_sol: Program
    parent_prompt: Program
    inspirations: List[Program] = []

    if gen_init_pop:
        logger.info(
            "Generating initial population: selecting initial solution and prompt as parents."
        )
        parent_sol = init_sol
        parent_prompt = init_prompt
    elif exploration:
        logger.info("Exploration: selecting parents uniformly at random.")
        parent_sol, _ = sol_db.sample(
            selection_policy="random",
            num_inspirations=evolve_config["num_inspirations"],
            pids_pool=[sol_id for sol_id, is_alive in sol_db.is_alive.items() if is_alive],
        )
        parent_prompt, _ = prompt_db.sample(
            selection_policy="random",
            num_inspirations=0,
            pids_pool=[prompt_id for prompt_id, is_alive in prompt_db.is_alive.items() if is_alive],
        )
    else:
        selection_kwargs: Dict[str, Any] = evolve_config.get("selection_kwargs", {})
        logger.info(
            f"Exploitation: Selecting parents using {evolve_config['selection_policy']} "
            f"with kwargs {selection_kwargs}."
        )
        parent_sol, inspirations = sol_db.sample(
            selection_policy=evolve_config["selection_policy"],
            num_inspirations=evolve_config["num_inspirations"],
            **selection_kwargs,
        )
        parent_prompt, _ = prompt_db.sample(
            selection_policy=evolve_config["selection_policy"],
            num_inspirations=0,
            **selection_kwargs,
        )

    logger.info(f"Selected {len(inspirations)} inspirations.")
    return parent_sol, parent_prompt, inspirations


async def run_meta_prompting(
    prompt_sampler: PromptSampler,
    prompt_db: ProgramDatabase,
    parent_prompt: Program,
    parent_sol: Program,
    epoch: int,
    isl_id: int,
    evolve_config: Dict[str, Any],
    evolve_state: Dict[str, Any],
    gen_init_pop: bool,
    logger: logging.Logger,
) -> Tuple[Optional[Program], bool]:
    """
    Evolve a parent prompt by generating and applying modifications via LLM.

    This function implements the meta-prompting phase, where an auxiliary LLM proposes
    changes to the system prompt based on the current best solution's performance.
    The LLM generates a diff in SEARCH/REPLACE format, which is then applied to
    produce a child prompt.

    Process:
        1. Query LLM to generate prompt modification diff
        2. Parse and validate the SEARCH/REPLACE blocks
        3. Apply diff to parent prompt within designated markers
        4. Add child prompt to database if successful

    Args:
        prompt_sampler: Sampler containing the auxiliary LLM for meta-prompting
        prompt_db: Database to store evolved prompts
        parent_prompt: Current prompt program to evolve
        parent_sol: Current best solution (provides performance context)
        epoch: Current epoch number
        isl_id: Island identifier for tracking provenance
        evolve_config: Configuration containing:
            - mp_start_marker: Start marker for prompt evolution block (default: "# PROMPT-BLOCK-START")
            - mp_end_marker: End marker for prompt evolution block (default: "# PROMPT-BLOCK-END")
        evolve_state: State dictionary to record token usage and errors
        gen_init_pop: Whether generating initial population (affects parent tracking)
        logger: Logger instance

    Returns:
        Tuple of (child_prompt, success) where:
            - child_prompt: Newly created prompt Program or None if failed
            - success: Boolean indicating whether meta-prompting succeeded
    """
    logger.info("=== META-PROMPT STEP ===")

    _, _, mp_start_marker, mp_end_marker = _get_markers(evolve_config)

    prompt_diff: str = ""

    ## GENERATE DIFF
    try:
        logger.info(f"Attempting to run meta_prompt on {prompt_sampler.aux_lm}...")
        prompt_diff, prompt_tok, compl_tok = await prompt_sampler.meta_prompt(
            prompt=parent_prompt, prog=parent_sol
        )
        logger.info(
            (
                f"Successfully retrieved response, using {prompt_tok} prompt tokens"
                f" and {compl_tok} completion tokens."
            )
        )

        evolve_state["tok_usage"].append(
            {
                "epoch": epoch,
                "motive": "meta_prompt",
                "prompt_tok": prompt_tok,
                "compl_tok": compl_tok,
                "model_name": prompt_sampler.aux_lm.model_name,
            }
        )
    except Exception as err:
        logger.error(f"Error when running meta-prompt on LM: {str(err)}.")
        evolve_state["errors"].append(
            {
                "epoch": epoch,
                "motive": "meta_prompt",
                "error_msg": str(err),
            }
        )
        return None, False

    ## APPLY DIFF
    try:
        logger.info("Attempting to SEARCH/REPLACE on prompt...")
        child_prompt_txt = apply_diff(
            parent_code=parent_prompt.code,
            diff=prompt_diff,
            start_marker=mp_start_marker,
            end_marker=mp_end_marker,
        )
        logger.info("Successfully modified parent prompt.")
    except Exception as err:
        logger.error(f"Error with SEARCH/REPLACE (meta-prompt): '{str(err)}'.")
        evolve_state["errors"].append(
            {
                "epoch": epoch,
                "motive": "sr_meta_prompt",
                "parent_prompt_id": parent_prompt.id,
                "parent_sol_id": parent_sol.id,
                "prompt_diff": prompt_diff,
                "error_msg": str(err),
            }
        )
        return None, False

    ## ADD TO DB
    logger.info("Adding child_prompt to prompt_db.")
    child_prompt = Program(
        id=str(uuid4()),
        code=child_prompt_txt,
        language=parent_prompt.language,
        iteration_found=epoch,
        generation=epoch,
        island_found=isl_id,
        model_id=0,
        model_msg=prompt_diff,
        depth=parent_prompt.depth + 1,
    )
    if not gen_init_pop:
        child_prompt.parent_id = parent_prompt.id

    prompt_db.add(child_prompt)
    return child_prompt, True


async def generate_solution(
    ensemble: OpenAIEnsemble,
    prompt_sampler: PromptSampler,
    sol_db: ProgramDatabase,
    prompt: Program,
    parent_sol: Program,
    inspirations: List[Program],
    epoch: int,
    isl_id: int,
    evolve_config: Dict[str, Any],
    evolve_state: Dict[str, Any],
    gen_init_pop: bool,
    chat_depth: Optional[int],
    exploitation: bool,
    logger: logging.Logger,
) -> Tuple[Optional[Program], bool]:
    """
    Generate a new solution program by querying an LLM ensemble with structured context.

        This function constructs a conversation context from the prompt, parent solution,
        and optional inspiration programs, then queries the LLM ensemble to generate
        code modifications. The LLM produces a diff in SEARCH/REPLACE format, which is
        applied to the parent solution to create a child program.

        Process:
            1. Build chat messages from prompt, parent, and inspirations
            2. Query LLM ensemble to generate code modification diff
            3. Parse and validate SEARCH/REPLACE blocks
            4. Apply diff to parent solution within designated markers
            5. Create child Program object (not yet evaluated)

        Args:
            ensemble: LLM ensemble for code generation (exploration or exploitation)
            prompt_sampler: Sampler for building conversation context
            sol_db: Solution database for retrieving context
            prompt: System prompt to guide LLM behavior
            parent_sol: Parent solution program to modify
            inspirations: List of inspiration programs for context (empty during exploration)
            epoch: Current epoch number
            isl_id: Island identifier
            evolve_config: Configuration containing:
                - evolve_start_marker: Start marker for code evolution block (default: "# EVOLVE-BLOCK-START")
                - evolve_end_marker: End marker for code evolution block (default: "# EVOLVE-BLOCK-END")
            evolve_state: State dictionary for tracking token usage and errors
            gen_init_pop: Whether generating initial population
            logger: Logger instance

        Returns:
            Tuple of (child_sol, success) where:
                - child_sol: Unevaluated Program object or None if generation failed
                - success: Boolean indicating success
    """
    logger.info("=== EVOLVE CODE STEP ===")

    evolve_start_marker, evolve_end_marker, _, _ = _get_markers(evolve_config)

    ## BUILD MESSAGE CHAT
    messages = prompt_sampler.build(
        prompt=prompt,
        prog=parent_sol,
        db=sol_db,
        inspirations=inspirations,
        max_chat_depth=chat_depth,
        exploitation=exploitation,
    )
    logger.info(f"Chat consists of {len(messages)} messages (max_chat_depth = {chat_depth}).")

    ## GENERATE DIFF
    try:
        model_id, sol_diff, prompt_tok, compl_tok = await ensemble.generate(messages=messages)
        evolve_state["tok_usage"].append(
            {
                "epoch": epoch,
                "motive": "generate_prog",
                "prompt_tok": prompt_tok,
                "compl_tok": compl_tok,
                "model_name": ensemble.models[model_id].model_name,
            }
        )
    except Exception as err:
        logger.error(f"Error when generating program on LM: {str(err)}.")
        evolve_state["errors"].append(
            {
                "epoch": epoch,
                "motive": "generate_prog",
                "error_msg": str(err),
            }
        )
        return None, False

    ## APPLY DIFF
    try:
        logger.info("Attempting to SEARCH/REPLACE on solution...")
        child_sol_code = apply_diff(
            parent_code=parent_sol.code,
            diff=sol_diff,
            start_marker=evolve_start_marker,
            end_marker=evolve_end_marker,
        )
        logger.info("Successfully modified parent solution.")
    except Exception as err:
        logger.error(f"Error with SEARCH/REPLACE (evolve solution): '{str(err)}'.")
        evolve_state["errors"].append(
            {
                "epoch": epoch,
                "motive": "sr_evolve_prog",
                "parent_sol_id": parent_sol.id,
                "sol_diff": sol_diff,
                "error_msg": str(err),
            }
        )
        return None, False

    # currently both iteration_found and generation are the same
    # as only one program is generated at each epoch
    child_sol = Program(
        id=str(uuid4()),
        code=child_sol_code,
        language=parent_sol.language,
        parent_id=parent_sol.id if not gen_init_pop else None,
        iteration_found=epoch,
        generation=epoch,
        island_found=isl_id,
        prompt_id=prompt.id,
        inspiration_ids=[ins.id for ins in inspirations],
        model_id=model_id,
        model_msg=sol_diff,
        depth=parent_sol.depth + 1,
    )
    return child_sol, True


async def evaluate_and_store(
    child_sol: Program,
    prompt: Program,
    evaluator: Evaluator,
    sol_db: ProgramDatabase,
    prompt_db: ProgramDatabase,
    embedding: Optional[OpenAIEmbedding],
    evolve_config: Dict[str, Any],
    evolve_state: Dict[str, Any],
    epoch: int,
    logger: logging.Logger,
) -> bool:
    """
    Evaluate a solution program and add it to the database if valid.

    This function executes the child solution in a sandboxed environment, computes
    fitness metrics, optionally generates code embeddings, and adds the program to
    the solution database. It also updates the associated prompt's fitness if the
    child improves upon it.

    Evaluation Steps:
        1. Execute program with resource limits (time, memory)
        2. Extract fitness from evaluation metrics
        3. Generate code embedding (optional)
        4. Update prompt fitness if child improves
        5. Add child to solution database
        6. Check if new global best was found

    Args:
        child_sol: Unevaluated child solution program
        prompt: Prompt used to generate this solution
        evaluator: Program evaluator with sandboxing
        sol_db: Solution database for storage
        prompt_db: Prompt database for updating prompt fitness
        embedding: Optional embedding model for code vectorization
        evolve_config: Configuration containing:
            - fitness_key: Metric name to use as fitness (e.g., 'accuracy')
            - use_embedding: Whether to generate embeddings
        evolve_state: State dictionary for tracking token usage and errors
        epoch: Current epoch number
        logger: Logger instance

    Returns:
        Boolean indicating whether this child became the new global best solution
    """
    ## EVALUATING CHILD PROGRAM
    child_sol.returncode, _, _, child_sol.error, child_sol.eval_metrics = evaluator.execute(
        child_sol
    )
    child_sol.fitness = child_sol.eval_metrics.get(evolve_config["fitness_key"], 0)

    logger.info(f"Child solution -> {child_sol}.")

    child_sol.prog_msg = format_prog_msg(prog=child_sol)
    child_sol.features = child_sol.eval_metrics

    if child_sol.fitness >= prompt.fitness:
        logger.info("Child solution improves on parent prompt fitness.")
        prompt.fitness = child_sol.fitness
        prompt.features = child_sol.features
        prompt_db.update_caches()

    ## EMBEDDING (Optional)
    if evolve_config.get("use_embedding", False) and embedding is not None:
        try:
            logger.info(f"Attempting to obtain embedding with model {embedding.model_name}...")
            child_sol.embedding, prompt_tok = await embedding.embed(child_sol.code)
            logger.info(f"Successfully retrieved embedding, used {prompt_tok} tokens")
            evolve_state["tok_usage"].append(
                {
                    "epoch": epoch,
                    "motive": "generate_embedding",
                    "prompt_tok": prompt_tok,
                    "compl_tok": 0,
                    "model_name": embedding.model_name,
                }
            )
        except Exception as err:
            logger.error(f"Error when generating embedding: '{str(err)}'.")
            evolve_state["errors"].append(
                {
                    "epoch": epoch,
                    "motive": "generate_embedding",
                    "error_msg": str(err),
                }
            )

    ## ADD TO DB
    logger.info("Adding child_sol to sol_db.")
    sol_db.add(child_sol)

    if child_sol.id == sol_db.best_prog_id:
        logger.info(f"New best program found -> {child_sol.fitness}.")
        return True

    logger.info(
        f"New program is worse than best -> {child_sol.fitness} "
        f"<= {sol_db.programs[sol_db.best_prog_id].fitness}."
    )
    return False


def handle_migration(
    epoch: int,
    isl_data: IslandCommunicationData,
    global_data: GlobalSyncData,
    sol_db: ProgramDatabase,
    evolve_config: Dict[str, Any],
    logger: logging.Logger,
) -> None:
    """
    Perform inter-island migration of solution programs at scheduled intervals.

    This function implements the migration phase of the island model, where islands
    periodically exchange their best solutions to maintain genetic diversity and
    accelerate convergence. Migration occurs synchronously across all islands using
    barrier synchronization.

    Migration Process:
        1. Select top programs from local database (migrants)
        2. Synchronize with other islands at barrier
        3. Send migrants to outgoing neighbor
        4. Receive migrants from incoming neighbor
        5. Add received programs to local database
        6. Mark outgoing programs as "migrated" to prevent re-selection

    Args:
        epoch: Current epoch number
        isl_data: Island data containing:
            - in_neigh: Incoming neighbor's communication channel
            - out_neigh: Outgoing neighbor's communication channel
        global_data: Global data containing barrier for synchronization
        sol_db: Local solution database
        evolve_config: Configuration containing:
            - migration_interval: Epochs between migrations (default: 20)
            - migration_rate: Fraction of population to migrate (default: 0.1)
        logger: Logger instance
    """
    if isl_data.in_neigh is None and isl_data.out_neigh is None:
        return

    if epoch % evolve_config.get("migration_interval", DEFAULT_MIGRATION_INTERVAL) == 0:
        logger.info("=== MIGRATION STEP ===")
        out_migrants = sol_db.get_migrants(
            migration_rate=evolve_config.get("migration_rate", DEFAULT_MIGRATION_RATE)
        )
        in_migrants = sync_migrate(
            out_migrants=out_migrants,
            isl_data=isl_data,
            barrier=global_data.barrier,
            logger=logger,
        )

        for out_migrant in out_migrants:
            sol_db.has_migrated[out_migrant.id] = True

        for in_migrant in in_migrants:
            in_migrant.parent_id = None
            in_migrant.prompt_id = None
            sol_db.add(in_migrant)


async def codeevolve_loop(
    start_epoch: int,
    evolve_state: Dict[str, Any],
    init_sol: Program,
    init_prompt: Program,
    evolve_config: Dict[str, Any],
    args: Dict[str, Any],
    isl_data: IslandCommunicationData,
    global_data: GlobalSyncData,
    sol_db: ProgramDatabase,
    prompt_db: ProgramDatabase,
    prompt_sampler: PromptSampler,
    exploration_ensemble: OpenAIEnsemble,
    exploitation_ensemble: OpenAIEnsemble,
    evaluator: Evaluator,
    embedding: Optional[OpenAIEmbedding],
    scheduler: Optional[ExplorationRateScheduler],
    logger: logging.Logger,
) -> None:
    """Executes the main evolutionary loop for program and prompt co-evolution.

    This function implements the core evolutionary algorithm. It has been refactored
    to delegate specific tasks (selection, prompt evolution, code generation, evaluation)
    to helper functions, improving readability and maintainability.

    The loop iterates through epochs, performing the following steps:
    1.  **State Setup**: Updates exploration rates and logs status.
    2.  **Selection**: Calls `select_parents` to choose programs/prompts.
    3.  **Meta-Prompting**: Calls `run_meta_prompting` to evolve prompts (if enabled).
    4.  **Generation**: Calls `generate_solution` to create new code via LLM.
    5.  **Evaluation**: Calls `evaluate_and_store` to run code and update DB.
    6.  **Migration**: Calls `handle_migration` to sync with other islands.
    7.  **Maintenance**: Handles metrics recording, checkpointing, and early stopping.

    Args:
        start_epoch: Starting epoch number.
        evolve_state: Dictionary tracking algorithm state.
        init_sol: Initial solution program.
        init_prompt: Initial prompt program.
        config: Full configuration dictionary.
        evolve_config: Evolution-specific configuration.
        args: Command-line arguments.
        isl_data: Island communication data.
        global_data: Shared data structures.
        sol_db: Solution database.
        prompt_db: Prompt database.
        prompt_sampler: Prompt sampling component.
        exploration_ensemble: Ensemble for exploration.
        exploitation_ensemble: Ensemble for exploitation.
        evaluator: Program evaluator.
        embedding: Embedding model (optional).
        scheduler: Exploration rate scheduler (optional).
        logger: Logger instance.
    """
    logger.info("============ STARTING EVOLUTIONARY LOOP ============")
    logger.info(f"Starting from epoch {start_epoch} with evolve_config = {evolve_config}")

    def _do_checkpoint(epoch_num: int) -> None:
        """Save checkpoint and optionally run metadata (island 0 only)."""
        best_lang: str = sol_db.programs[sol_db.best_prog_id].language
        save_ckpt(
            curr_epoch=epoch_num,
            prompt_db=prompt_db,
            sol_db=sol_db,
            evolve_state=evolve_state,
            scheduler=scheduler,
            best_sol_path=args["isl_out_dir"].joinpath(
                BEST_SOLUTION_FILE + LANGUAGE_TO_EXTENSION.get(best_lang, ".txt")
            ),
            best_prompt_path=args["isl_out_dir"].joinpath(BEST_PROMPT_FILE),
            ckpt_dir=args["ckpt_dir"],
            logger=logger,
        )
        if isl_data.id == 0:
            elapsed_s: float = get_elapsed_time(global_data)
            cpus: int = global_data.cpu_count.value
            save_run_metadata(args["out_dir"], epoch_num, elapsed_s, cpus)
            logger.info(
                f"Saved run metadata (elapsed={elapsed_s:.1f}s, cpus={cpus})"
                f" for epoch {epoch_num}."
            )

    meta_prompting: bool = evolve_config.get("meta_prompting", False)
    use_map_elites: bool = evolve_config.get("use_map_elites", False)
    exploration_rate: float = (
        scheduler.exploration_rate if scheduler is not None else evolve_config["exploration_rate"]
    )
    epoch: int = start_epoch + 1

    for epoch in range(start_epoch + 1, evolve_config["num_epochs"] + 1):
        logger.info(f"========= EPOCH {epoch} =========")

        # LOGGING AND SCHEDULER
        logger.info(
            f"Global early stopping counter: {evolve_state['early_stop_counter']}"
            f"/{evolve_config['early_stopping_rounds']}"
        )
        logger.info(f"Exploration rate: {exploration_rate}")
        logger.info(f"Best prompt: {prompt_db.programs[prompt_db.best_prog_id]}")
        logger.info(f"Best solution: {sol_db.programs[sol_db.best_prog_id]}")
        if use_map_elites:
            logger.info(f"sol_db EliteMap: {sol_db.elite_map.map}")

        init_pop_size: int = evolve_config.get("init_pop", sol_db.num_alive)
        gen_init_pop: bool = sol_db.num_alive < init_pop_size

        if not gen_init_pop and scheduler is not None:
            exploration_rate = scheduler(
                epoch=epoch - init_pop_size,
                best_fitness=sol_db.programs[sol_db.best_prog_id].fitness,
            )

        exploration: bool = (
            not gen_init_pop and sol_db.random_state.uniform(0, 1) <= exploration_rate
        )
        exploitation: bool = not gen_init_pop and not exploration

        logger.info(f"Generating initial populations: {gen_init_pop}")
        logger.info(f"Exploration: {exploration}")
        logger.info(f"Exploitation: {exploitation}")

        # PARENT SELECTION
        parent_sol, parent_prompt, inspirations = select_parents(
            sol_db=sol_db,
            prompt_db=prompt_db,
            init_sol=init_sol,
            init_prompt=init_prompt,
            evolve_config=evolve_config,
            gen_init_pop=gen_init_pop,
            exploration=exploration,
            logger=logger,
        )

        # META-PROMPTING (OPTIONAL)
        child_prompt: Optional[Program] = None
        meta_prompt_success: bool = False

        if meta_prompting and not exploitation:
            child_prompt, meta_prompt_success = await run_meta_prompting(
                prompt_sampler=prompt_sampler,
                prompt_db=prompt_db,
                parent_prompt=parent_prompt,
                parent_sol=parent_sol,
                epoch=epoch,
                isl_id=isl_data.id,
                evolve_config=evolve_config,
                evolve_state=evolve_state,
                gen_init_pop=gen_init_pop,
                logger=logger,
            )

        active_prompt: Program = (
            child_prompt if (meta_prompt_success and child_prompt) else parent_prompt
        )

        # EVOLVE SOLUTION
        ensemble: OpenAIEnsemble = (
            exploration_ensemble if not exploitation else exploitation_ensemble
        )
        chat_depth: Optional[int] = evolve_config.get("max_chat_depth", None) if exploitation else 0

        child_sol, evolve_success = await generate_solution(
            ensemble=ensemble,
            prompt_sampler=prompt_sampler,
            sol_db=sol_db,
            prompt=active_prompt,
            parent_sol=parent_sol,
            inspirations=inspirations,
            epoch=epoch,
            isl_id=isl_data.id,
            evolve_config=evolve_config,
            evolve_state=evolve_state,
            gen_init_pop=gen_init_pop,
            chat_depth=chat_depth,
            exploitation=exploitation,
            logger=logger,
        )

        # EVALUATE AND ADD TO DB
        improved_local_fitness: bool = False
        if evolve_success and child_sol:
            improved_local_fitness = await evaluate_and_store(
                child_sol=child_sol,
                prompt=active_prompt,
                evaluator=evaluator,
                sol_db=sol_db,
                prompt_db=prompt_db,
                embedding=embedding,
                evolve_config=evolve_config,
                evolve_state=evolve_state,
                epoch=epoch,
                logger=logger,
            )

        # MIGRATION
        handle_migration(
            epoch=epoch,
            isl_data=isl_data,
            global_data=global_data,
            sol_db=sol_db,
            evolve_config=evolve_config,
            logger=logger,
        )

        # CKPTS
        evolve_state["best_fit_hist"].append(sol_db.programs[sol_db.best_prog_id].fitness)
        evolve_state["avg_fit_hist"].append(
            np.mean(np.array([sol.fitness for sol in sol_db.programs.values()]))
        )
        evolve_state["exploration"].append(exploration)

        if epoch % evolve_config["ckpt"] == 0:
            logger.info("=== CHECKPOINT STEP ===")
            logger.info("Waiting for other islands to arrive at barrier...")
            global_data.barrier.wait()
            logger.info("All islands arrived. Proceeding to save ckpt.")
            _do_checkpoint(epoch)

        # EARLY STOPPING
        logger.info("=== GLOBAL EARLY STOPPING CHECK STEP ===")
        if improved_local_fitness and child_sol:
            with global_data.lock:
                if global_data.best_sol.fitness.value <= child_sol.fitness:
                    logger.info("Global best solution improved.")
                    global_data.best_sol.update_from_program(child_sol)

        early_stopping_check(
            num_islands=evolve_config["num_islands"],
            improved_local_fitness=improved_local_fitness,
            global_data=global_data,
            logger=logger,
        )

        if global_data.early_stop_counter.value > evolve_state["early_stop_counter"]:
            logger.info(
                f"Early stopping counter increased: {global_data.early_stop_counter.value}"
                f"/{evolve_config['early_stopping_rounds']}"
            )

        evolve_state["early_stop_counter"] = global_data.early_stop_counter.value

        if evolve_state["early_stop_counter"] == evolve_config["early_stopping_rounds"]:
            logger.info(
                f"EARLY STOPPING: {evolve_state['early_stop_counter']} "
                "global consecutive epochs without improvement."
            )
            break

        # SYNC
        logger.info("=== END EPOCH SYNC STEP ===")
        logger.info("Waiting for other islands to finish epoch...")
        global_data.barrier.wait()
        logger.info("All islands finished. Moving to next epoch.")

    # END
    logger.info("====== ALGORITHM FINISHED ======")
    logger.info(f"Best solution: {sol_db.programs[sol_db.best_prog_id]}")
    logger.info(f"Best prompt: {prompt_db.programs[prompt_db.best_prog_id]}")
    _do_checkpoint(epoch)


# ---------------------------------------------------------------------------
# CodeEvolve components dataclass and helper functions
# ---------------------------------------------------------------------------


@dataclass
class CodeEvolveComponents:
    """Container for all components required by the CodeEvolve algorithm.

    This dataclass groups together all the initialized components needed to run
    the evolutionary loop, including databases, ensembles, evaluators, and state.

    Attributes:
        config: Full configuration dictionary loaded from YAML.
        evolve_config: Evolution-specific configuration extracted from config.
        start_epoch: Starting epoch number (0 for new run, >0 for checkpoint resume).
        evolve_state: Dictionary tracking algorithm state including fitness history,
                     token usage, errors, and early stopping counter.
        init_sol: Initial solution program used to seed the population.
        init_prompt: Initial prompt program used to seed the prompt population.
        sol_db: Database managing the solution program population.
        prompt_db: Database managing the prompt program population.
        exploration_ensemble: LLM ensemble used during exploration phases.
        exploitation_ensemble: LLM ensemble used during exploitation phases.
        prompt_sampler: Sampler for building conversation prompts from lineages.
        evaluator: Program evaluator with sandboxing and resource limits.
        embedding: Optional embedding model for code vectorization.
        scheduler: Optional exploration rate scheduler.
        logger: Logger instance for this island.
    """

    config: Dict[str, Any]
    evolve_config: Dict[str, Any]
    start_epoch: int
    evolve_state: Dict[str, Any]
    init_sol: Program
    init_prompt: Program
    sol_db: ProgramDatabase
    prompt_db: ProgramDatabase
    exploration_ensemble: OpenAIEnsemble
    exploitation_ensemble: OpenAIEnsemble
    prompt_sampler: PromptSampler
    evaluator: Evaluator
    embedding: Optional[OpenAIEmbedding]
    scheduler: Optional[ExplorationRateScheduler]
    logger: logging.Logger


def _create_ensembles(
    config: Dict[str, Any],
    evolve_config: Dict[str, Any],
    args: Dict[str, Any],
    logger: logging.Logger,
) -> Tuple[OpenAIEnsemble, OpenAIEnsemble]:
    """Creates and configures the exploration and exploitation LLM ensembles.

    Args:
        config: Full configuration dictionary.
        evolve_config: Evolution-specific configuration.
        args: Command-line arguments containing API credentials.
        logger: Logger instance for the ensembles.

    Returns:
        Tuple of (exploration_ensemble, exploitation_ensemble).
    """
    evolve_start_marker, evolve_end_marker, _, _ = _get_markers(evolve_config)

    exploration_ensemble: OpenAIEnsemble = OpenAIEnsemble(
        models_cfg=config.get("EXPLORATION_ENSEMBLE", config.get("ENSEMBLE")),
        api_key=args["api_key"],
        api_base=args["api_base"],
        logger=logger,
        start_marker=evolve_start_marker,
        end_marker=evolve_end_marker,
    )
    exploitation_ensemble: OpenAIEnsemble = OpenAIEnsemble(
        models_cfg=config.get("EXPLOITATION_ENSEMBLE", config.get("ENSEMBLE")),
        api_key=args["api_key"],
        api_base=args["api_base"],
        logger=logger,
        start_marker=evolve_start_marker,
        end_marker=evolve_end_marker,
    )

    return exploration_ensemble, exploitation_ensemble


def _create_prompt_sampler(
    config: Dict[str, Any],
    evolve_config: Dict[str, Any],
    args: Dict[str, Any],
) -> PromptSampler:
    """Creates and configures the prompt sampler.

    Args:
        config: Full configuration dictionary.
        evolve_config: Evolution-specific configuration.
        args: Command-line arguments containing API credentials.

    Returns:
        Configured PromptSampler instance.
    """
    evolve_start_marker, evolve_end_marker, mp_start_marker, mp_end_marker = _get_markers(
        evolve_config
    )

    return PromptSampler(
        aux_lm_cfg=config["SAMPLER_AUX_LM"],
        api_key=args["api_key"],
        api_base=args["api_base"],
        evolve_start_marker=evolve_start_marker,
        evolve_end_marker=evolve_end_marker,
        prompt_start_marker=mp_start_marker,
        prompt_end_marker=mp_end_marker,
    )


def _create_evaluator(
    config: Dict[str, Any],
    args: Dict[str, Any],
    logger: logging.Logger,
) -> Evaluator:
    """Creates and configures the program evaluator.

    Args:
        config: Full configuration dictionary.
        args: Command-line arguments containing input directory path.
        logger: Logger instance for the evaluator.

    Returns:
        Configured Evaluator instance.
    """
    return Evaluator(
        eval_path=Path(config["EVAL_FILE_NAME"]),
        cwd=args["inpt_dir"],
        timeout_s=config.get("EVAL_TIMEOUT", DEFAULT_EVAL_TIMEOUT_S),
        max_mem_b=config.get("MAX_MEM_BYTES", DEFAULT_MAX_MEM_BYTES),
        mem_check_interval_s=config.get("MEM_CHECK_INTERVAL_S", DEFAULT_MEM_CHECK_INTERVAL_S),
        logger=logger,
    )


def _create_embedding(
    config: Dict[str, Any],
    evolve_config: Dict[str, Any],
    args: Dict[str, Any],
) -> Optional[OpenAIEmbedding]:
    """Creates the embedding model if configured.

    Args:
        config: Full configuration dictionary.
        evolve_config: Evolution-specific configuration.
        args: Command-line arguments containing API credentials.

    Returns:
        OpenAIEmbedding instance if use_embedding is enabled, None otherwise.

    Raises:
        AssertionError: If use_embedding is True but EMBEDDING config is missing.
    """
    if not evolve_config.get("use_embedding", False):
        return None

    assert (
        config.get("EMBEDDING", None) is not None
    ), "EMBEDDING model must be defined in config.yaml when use_embedding is true."

    return OpenAIEmbedding(
        **config["EMBEDDING"],
        api_key=args["api_key"],
        api_base=args["api_base"],
    )


def _create_scheduler(
    evolve_config: Dict[str, Any],
) -> Optional[ExplorationRateScheduler]:
    """Creates the exploration rate scheduler if configured.

    Args:
        evolve_config: Evolution-specific configuration.

    Returns:
        ExplorationRateScheduler instance if use_scheduler is enabled, None otherwise.
    """
    if not evolve_config.get("use_scheduler", False):
        return None

    scheduler_type: str = evolve_config.get("type", "ExponentialDecayScheduler")
    return SCHEDULER_TYPES[scheduler_type](
        exploration_rate=evolve_config["exploration_rate"],
        **evolve_config["scheduler_kwargs"],
    )


def _initialize_from_checkpoint(
    args: Dict[str, Any],
    scheduler: Optional[ExplorationRateScheduler],
) -> Tuple[
    ProgramDatabase,
    ProgramDatabase,
    Dict[str, Any],
    Program,
    Program,
    Optional[ExplorationRateScheduler],
]:
    """Initializes databases and programs from a checkpoint.

    Args:
        args: Command-line arguments containing checkpoint path.
        scheduler: Previously created scheduler (may be replaced by checkpoint).

    Returns:
        Tuple of (prompt_db, sol_db, evolve_state, init_prompt, init_sol, scheduler).
    """
    prompt_db: ProgramDatabase
    sol_db: ProgramDatabase
    evolve_state: Dict[str, Any]
    sched: Optional[ExplorationRateScheduler]

    prompt_db, sol_db, evolve_state, sched = load_ckpt(args["load_ckpt"], args["ckpt_dir"])

    init_prompt: Program = prompt_db.programs[prompt_db.best_prog_id]
    init_sol: Program = sol_db.programs[sol_db.best_prog_id]
    init_sol.prompt_id = init_prompt.id

    final_scheduler: Optional[ExplorationRateScheduler] = sched if sched is not None else scheduler

    return prompt_db, sol_db, evolve_state, init_prompt, init_sol, final_scheduler


def _initialize_new(
    config: Dict[str, Any],
    evolve_config: Dict[str, Any],
    args: Dict[str, Any],
    isl_id: int,
    evaluator: Evaluator,
    logger: logging.Logger,
) -> Tuple[ProgramDatabase, ProgramDatabase, Dict[str, Any], Program, Program]:
    """Initializes databases and programs for a new run.

    Args:
        config: Full configuration dictionary.
        evolve_config: Evolution-specific configuration.
        args: Command-line arguments containing input directory path.
        isl_id: Island identifier.
        evaluator: Program evaluator for executing the initial solution.
        logger: Logger instance.

    Returns:
        Tuple of (prompt_db, sol_db, evolve_state, init_prompt, init_sol).

    Raises:
        AssertionError: If use_map_elites is True but MAP_ELITES config is missing.
    """
    logger.info("Starting anew.")

    evolve_state: Dict[str, Any] = {
        "early_stop_counter": 0,
        "best_fit_hist": [],
        "avg_fit_hist": [],
        "errors": [],
        "tok_usage": [],
        "exploration": [],
    }

    features: Optional[List[EliteFeature]] = None
    map_elites_cfg: Dict[str, Any] = config.get("MAP_ELITES", {})

    if evolve_config.get("use_map_elites", False):
        assert (
            len(map_elites_cfg) > 0
        ), "MAP_ELITES must be defined in config.yaml when use_map_elites is true."
        features = []
        for feature in map_elites_cfg["features"]:
            features.append(
                EliteFeature(
                    name=feature["name"],
                    min_val=feature["min_val"],
                    max_val=feature["max_val"],
                    num_bins=feature.get("num_bins", None),
                )
            )

    prompt_db: ProgramDatabase = ProgramDatabase(
        id=isl_id,
        seed=config.get("SEED", None),
        max_alive=evolve_config.get("max_size", None),
        elite_map_type=None,
        features=None,
    )
    sol_db: ProgramDatabase = ProgramDatabase(
        id=isl_id,
        seed=config.get("SEED", None),
        max_alive=evolve_config.get("max_size", None),
        elite_map_type=map_elites_cfg.get("elite_map_type", None),
        features=features,
        **map_elites_cfg.get("elite_map_kwargs", {}),
    )

    init_prompt: Program = Program(
        id=str(uuid4()),
        code=config["SYS_MSG"],
        language="text",
        iteration_found=0,
        generation=0,
        island_found=isl_id,
        depth=0,
    )
    prompt_db.add(init_prompt)

    init_sol_path: Path = (
        args["inpt_dir"]
        .joinpath(config["CODEBASE_PATH"])
        .joinpath(config["INIT_FILE_DATA"]["filename"])
    )
    with open(init_sol_path) as f:
        init_sol: Program = Program(
            id=str(uuid4()),
            code=f.read(),
            language=config["INIT_FILE_DATA"]["language"],
            iteration_found=0,
            generation=0,
            island_found=isl_id,
            depth=0,
        )

    init_sol.returncode, _, _, init_sol.error, init_sol.eval_metrics = evaluator.execute(init_sol)
    if init_sol.returncode == 0:
        init_sol.fitness = init_sol.eval_metrics[evolve_config["fitness_key"]]
    init_sol.prog_msg = format_prog_msg(prog=init_sol)
    init_sol.features = init_sol.eval_metrics
    sol_db.add(init_sol)

    return prompt_db, sol_db, evolve_state, init_prompt, init_sol


def setup_codeevolve_components(
    args: Dict[str, Any],
    isl_data: IslandCommunicationData,
    global_data: GlobalSyncData,
) -> CodeEvolveComponents:
    """Sets up and initializes all components required for the CodeEvolve algorithm.

    This function handles all initialization logic including:
    - Logger setup
    - Configuration loading
    - LLM ensemble creation (exploration and exploitation)
    - Prompt sampler creation
    - Evaluator creation with resource limits
    - Optional embedding model and scheduler creation
    - Database initialization (either from checkpoint or new)
    - Initial solution evaluation

    Args:
        args: Dictionary containing command-line arguments and runtime configuration
              including paths, API keys, checkpoint settings, etc.
        isl_data: Island-specific data including ID and communication channels.
        global_data: Shared data structures for coordinating between islands.

    Returns:
        CodeEvolveComponents instance containing all initialized components ready
        for the evolutionary loop.
    """
    logger: logging.Logger = get_logger(
        island_id=isl_data.id,
        results_dir=args["isl_out_dir"],
        append_mode=(args["load_ckpt"] != 0),
        log_queue=global_data.log_queue,
        max_msg_sz=DEFAULT_MAX_LOG_MSG_SIZE,
    )
    logger.info("=== CodeEvolve ===")
    logger.info("====== PREPARING COMPONENTS ======")

    with open(args["cfg_path"], "r") as f:
        config: Dict[str, Any] = yaml.safe_load(f)
    evolve_config: Dict[str, Any] = config["EVOLVE_CONFIG"]

    exploration_ensemble: OpenAIEnsemble
    exploitation_ensemble: OpenAIEnsemble
    exploration_ensemble, exploitation_ensemble = _create_ensembles(
        config, evolve_config, args, logger
    )

    prompt_sampler: PromptSampler = _create_prompt_sampler(config, evolve_config, args)
    evaluator: Evaluator = _create_evaluator(config, args, logger)
    embedding: Optional[OpenAIEmbedding] = _create_embedding(config, evolve_config, args)
    scheduler: Optional[ExplorationRateScheduler] = _create_scheduler(evolve_config)

    start_epoch: int = args["load_ckpt"]
    prompt_db: ProgramDatabase
    sol_db: ProgramDatabase
    evolve_state: Dict[str, Any]
    init_prompt: Program
    init_sol: Program

    if args["load_ckpt"]:
        prompt_db, sol_db, evolve_state, init_prompt, init_sol, scheduler = (
            _initialize_from_checkpoint(args, scheduler)
        )
    else:
        prompt_db, sol_db, evolve_state, init_prompt, init_sol = _initialize_new(
            config, evolve_config, args, isl_data.id, evaluator, logger
        )

    return CodeEvolveComponents(
        config=config,
        evolve_config=evolve_config,
        start_epoch=start_epoch,
        evolve_state=evolve_state,
        init_sol=init_sol,
        init_prompt=init_prompt,
        sol_db=sol_db,
        prompt_db=prompt_db,
        exploration_ensemble=exploration_ensemble,
        exploitation_ensemble=exploitation_ensemble,
        prompt_sampler=prompt_sampler,
        evaluator=evaluator,
        embedding=embedding,
        scheduler=scheduler,
        logger=logger,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


async def codeevolve(
    args: Dict[str, Any],
    isl_data: IslandCommunicationData,
    global_data: GlobalSyncData,
) -> None:
    """Main entry point for the CodeEvolve algorithm on a single island.

    This function orchestrates the evolutionary program synthesis process by:
    1. Setting up all required components via setup_codeevolve_components()
    2. Synchronizing the global best solution across islands
    3. Checking for early termination conditions
    4. Launching the evolutionary loop

    The algorithm co-evolves programs and prompts using language models, with support
    for distributed execution across multiple islands, fitness-based selection,
    migration between islands, and early stopping mechanisms.

    Args:
        args: Dictionary containing command-line arguments and runtime configuration
              including paths, API keys, checkpoint settings, etc.
        isl_data: Island-specific data including ID and communication channels for
                 distributed execution.
        global_data: Shared data structures for coordinating between islands including
                    global best solution tracking and synchronization primitives.
    """
    components: CodeEvolveComponents = setup_codeevolve_components(args, isl_data, global_data)
    components.logger.info(f"sol_db={components.sol_db}")
    components.logger.info(f"prompt_db={components.prompt_db}")
    components.logger.info(f"exploration_ensemble={components.exploration_ensemble}")
    components.logger.info(f"exploitation_ensemble={components.exploitation_ensemble}")
    components.logger.info(f"prompt_sampler={components.prompt_sampler}")
    components.logger.info(f"evaluator={components.evaluator}")
    components.logger.info(f"embedding={components.embedding}")
    components.logger.info(f"scheduler={components.scheduler}")
    components.logger.info(f"init_prog={components.init_sol}")

    with global_data.lock:
        global_data.early_stop_counter.value = components.evolve_state["early_stop_counter"]
        if global_data.best_sol.fitness.value <= components.init_sol.fitness:
            global_data.best_sol.update_from_program(components.init_sol)

    is_already_complete: bool = (
        components.start_epoch == components.evolve_config["num_epochs"]
        or components.evolve_state["early_stop_counter"]
        == components.evolve_config["early_stopping_rounds"]
    )
    if is_already_complete:
        components.logger.info("Loaded checkpoint already finished the algorithm.")
        return

    components.logger.info("Waiting for other islands to finish setup...")
    global_data.barrier.wait()
    components.logger.info("All islands finished. Starting CodeEvolve loop.")

    await codeevolve_loop(
        components.start_epoch,
        components.evolve_state,
        components.init_sol,
        components.init_prompt,
        components.evolve_config,
        args,
        isl_data,
        global_data,
        components.sol_db,
        components.prompt_db,
        components.prompt_sampler,
        components.exploration_ensemble,
        components.exploitation_ensemble,
        components.evaluator,
        components.embedding,
        components.scheduler,
        components.logger,
    )
