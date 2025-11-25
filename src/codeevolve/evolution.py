# ===--------------------------------------------------------------------------------------===#
#
# Part of the CodeEvolve Project, under the Apache License v2.0.
# See https://github.com/inter-co/science-codeevolve/blob/main/LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0
#
# ===--------------------------------------------------------------------------------------===#
#
# This file implements the main evolutionary loop of CodeEvolve.
#
# ===--------------------------------------------------------------------------------------===#

from typing import Any, Dict, List, Optional
from uuid import uuid4
import logging
from pathlib import Path

import yaml
import numpy as np

from codeevolve.database import Program, ProgramDatabase, EliteFeature
from codeevolve.lm import OpenAILM, LMEnsemble, OpenAIEmbedding
from codeevolve.evaluator import Evaluator
from codeevolve.prompt.sampler import PromptSampler, format_prog_msg
from codeevolve.islands import (
    IslandData,
    GlobalData,
    sync_migrate,
    early_stopping_check,
)

from codeevolve.utils.parsing_utils import apply_diff
from codeevolve.utils.logging_utils import get_logger
from codeevolve.utils.ckpt_utils import save_ckpt, load_ckpt

MAX_LOG_MSG_SZ: int = 256


async def evolve_loop(
    start_epoch: int,
    evolve_state: Dict[str, Any],
    init_sol: Program,
    init_prompt: Program,
    config: Dict[Any, Any],
    evolve_config: Dict[str, Any],
    args: Dict[str, Any],
    isl_data: IslandData,
    global_data: GlobalData,
    sol_db: ProgramDatabase,
    prompt_db: ProgramDatabase,
    prompt_sampler: PromptSampler,
    ensemble: LMEnsemble,
    evaluator: Evaluator,
    embedding: Optional[OpenAIEmbedding],
    logger: logging.Logger,
) -> None:
    """Executes the main evolutionary loop for program and prompt co-evolution.

    This function implements the core evolutionary algorithm that iteratively:
    1. Selects parent programs and prompts based on fitness
    2. Evolves prompts using meta-prompting (if enabled)
    3. Generates new programs using evolved prompts
    4. Evaluates and adds successful programs to the database
    5. Handles migration between islands
    6. Performs early stopping checks and checkpointing

    Args:
        start_epoch: Starting epoch number (0 for new runs, >0 for resumed runs).
        evolve_state: Dictionary tracking algorithm state including fitness history and errors.
        init_sol: Initial solution program to bootstrap the evolution.
        init_prompt: Initial prompt program for generating solutions.
        config: Full configuration dictionary loaded from YAML.
        evolve_config: Evolution-specific configuration parameters.
        args: Command-line arguments and runtime parameters.
        isl_data: Island communication data for distributed execution.
        global_data: Shared data structures for inter-island coordination.
        sol_db: Database containing solution programs.
        prompt_db: Database containing prompt programs.
        prompt_sampler: Component for building conversation prompts.
        ensemble: Language model ensemble for program generation.
        evaluator: Component for executing and evaluating programs.
        logger: Logger instance for this island.
    """
    logger.info("============ STARTING EVOLUTIONARY LOOP ============")
    logger.info(f"Starting from epoch {start_epoch} with evolve_config = {evolve_config}")

    meta_prompting: bool = evolve_config.get("meta_prompting", False)
    use_embedding: bool = evolve_config.get("use_embedding", False)

    mp_start_marker: str = evolve_config.get("mp_start_marker", "# PROMPT-BLOCK-START")
    mp_end_marker: str = evolve_config.get("mp_end_marker", "# PROMPT-BLOCK-END")
    evolve_start_marker: str = evolve_config.get("evolve_start_marker", "# EVOLVE-BLOCK-START")
    evolve_end_marker: str = evolve_config.get("evolve_end_marker", "# EVOLVE-BLOCK-END")

    for epoch in range(start_epoch + 1, evolve_config["num_epochs"] + 1):
        logger.info(f"========= EPOCH {epoch} =========")
        logger.info(
            (
                f"Global early stopping counter: {evolve_state['early_stop_counter']}"
                f"/{evolve_config['early_stopping_rounds']}"
            )
        )
        logger.info(f"Prompt database: {prompt_db}")
        logger.info(f"Best prompt: {prompt_db.programs[prompt_db.best_prog_id]}")
        logger.info(f"Solution database: {sol_db}")
        logger.info(f"Best solution: {sol_db.programs[sol_db.best_prog_id]}")
        if config.get("MAP_ELITES", None):
            logger.info(f"sol_db EliteMap: {sol_db.elite_map.map}")
            logger.info(f"prompt_db EliteMap: {prompt_db.elite_map.map}")

        gen_init_pop: bool = sol_db.num_alive < evolve_config.get("init_pop", sol_db.num_alive)
        logger.info(f"Generating initial populations: {gen_init_pop}")

        exploration: bool = sol_db.random_state.uniform(0, 1) <= evolve_config["exploration_rate"]
        logger.info(f"Exploration: {exploration}")

        # SELECTING PARENT PROGRAM
        logger.info("=== SELECTION STEP ===")
        if not gen_init_pop:
            selection_policy: str = "random" if exploration else evolve_config["selection_policy"]
            selection_kwargs: dict = {} if exploration else evolve_config["selection_kwargs"]
            logger.info(
                (
                    f"Selecting parents according to {selection_policy}"
                    f" with kwargs {selection_kwargs}."
                )
            )
            parent_sol, inspirations = sol_db.sample(
                selection_policy=selection_policy,
                num_inspirations=evolve_config["num_inspirations"],
                **selection_kwargs,
            )
            if parent_sol.prompt_id:
                parent_prompt: Program = prompt_db.programs[parent_sol.prompt_id]
            else:
                parent_prompt, _ = prompt_db.sample(
                    selection_policy=selection_policy,
                    num_inspirations=0,
                    **selection_kwargs,
                )
        else:
            logger.info("Selecting init_prompt and init_sol as parents.")
            parent_sol: Program = init_sol
            inspirations: List[Program] = []
            parent_prompt: Program = init_prompt

        if (
            isl_data.in_neigh or isl_data.out_neigh
        ):  # only use inspirations after first migration if migrating
            inspirations = (
                inspirations if (epoch > evolve_config.get("migration_interval", 20)) else []
            )

        logger.info(f"Selected {len(inspirations)} inspirations.")

        # META-PROMPTING
        if meta_prompting and (gen_init_pop or exploration):
            logger.info("=== META-PROMPT STEP ===")
            meta_prompt_success: bool = False
            ## GENERATE DIFF
            try:
                # TODO: maybe move the logger from inside the sampler class to here
                prompt_diff, prompt_tok, compl_tok = await prompt_sampler.meta_prompt(
                    prompt=parent_prompt, prog=parent_sol
                )
                meta_prompt_success = True

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
                logger.error(f"Error when running prompt on LM: {str(err)}.")
                error_info: Dict[str, Any] = {
                    "epoch": epoch,
                    "motive": "meta_prompt",
                    "error_msg": str(err),
                }
                evolve_state["errors"].append(error_info)

            ## APPLY DIFF
            if meta_prompt_success:
                try:
                    logger.info("Attempting to SEARCH/REPLACE...")
                    child_prompt_txt: str = apply_diff(
                        parent_code=parent_prompt.code,
                        diff=prompt_diff,
                        start_marker=mp_start_marker,
                        end_marker=mp_end_marker,
                    )
                    logger.info("Successfully modified parent prompt.")
                except Exception as err:
                    logger.error(f"Error with SEARCH/REPLACE: '{str(err)}'.")
                    meta_prompt_success = False

                    error_info: Dict[str, Any] = {
                        "epoch": epoch,
                        "motive": "sr_meta_prompt",
                        "parent_prompt_id": parent_prompt.id,
                        "parent_sol_id": parent_sol.id,
                        "prompt_diff": prompt_diff,
                        "error_msg": str(err),
                    }
                    evolve_state["errors"].append(error_info)

            ## ADD TO DB
            if meta_prompt_success:
                logger.info("Adding child_prompt to prompt_db.")
                child_prompt: Program = Program(
                    id=str(uuid4()),
                    code=child_prompt_txt,
                    language=parent_prompt.language,
                    iteration_found=epoch,
                    generation=epoch,
                    island_found=isl_data.id,
                    model_id=0,
                    model_msg=prompt_diff,
                )
                if not gen_init_pop:
                    child_prompt.parent_id = parent_prompt.id

                prompt_db.add(child_prompt)

        # EVOLVE PARENT PROG
        logger.info("=== EVOLVE CODE STEP=== ")
        evolve_success: bool = False
        improved_local_fitness: bool = False

        prompt: Program = (
            parent_prompt
            if not (meta_prompting and (gen_init_pop or exploration) and meta_prompt_success)
            else child_prompt
        )

        ## BUILD MESSAGE CHAT
        messages: List[Dict[str, str]] = prompt_sampler.build(
            prompt=prompt,
            prog=parent_sol,
            db=sol_db,
            inspirations=inspirations,
            max_chat_depth=(0 if exploration else evolve_config.get("max_chat_depth", None)),
        )
        logger.info(f"Chat consists of {len(messages)} messages.")

        ## GENERATE DIFF
        try:
            # TODO: maybe move the logger from inside the ensemble class to here
            model_id, sol_diff, prompt_tok, compl_tok = await ensemble.generate(messages=messages)
            evolve_success = True

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
            logger.error(f"Error when running prompt on LM: {str(err)}.")
            error_info: Dict[str, Any] = {
                "epoch": epoch,
                "motive": "generate_prog",
                "error_msg": str(err),
            }
            evolve_state["errors"].append(error_info)

        ## APPLY DIFF
        if evolve_success:
            try:
                logger.info("Attempting to SEARCH/REPLACE...")
                child_sol_code: str = apply_diff(
                    parent_code=parent_sol.code,
                    diff=sol_diff,
                    start_marker=evolve_start_marker,
                    end_marker=evolve_end_marker,
                )
                logger.info("Successfully modified parent solution.")
            except Exception as err:
                logger.error(f"Error with SEARCH/REPLACE: '{str(err)}'.")
                evolve_success = False
                error_info: Dict[str, Any] = {
                    "epoch": epoch,
                    "motive": "sr_evolve_prog",
                    "parent_sol_id": parent_sol.id,
                    "sol_diff": sol_diff,
                    "error_msg": str(err),
                }
                evolve_state["errors"].append(error_info)

        if evolve_success:
            # currently both iteration_found and generation are the same
            # as only one program is generated at each epoch
            child_sol: Program = Program(
                id=str(uuid4()),
                code=child_sol_code,
                language=parent_sol.language,
                parent_id=parent_sol.id if not gen_init_pop else None,
                iteration_found=epoch,
                generation=epoch,
                island_found=isl_data.id,
                prompt_id=prompt.id,
                inspiration_ids=[inspiration.id for inspiration in inspirations],
                model_id=model_id,
                model_msg=sol_diff,
            )

            ## EVALUATING CHILD PROGRAM
            evaluator.execute(child_sol)
            if child_sol.returncode == 0:
                child_sol.fitness = child_sol.eval_metrics[evolve_config["fitness_key"]]
            child_sol.prog_msg = format_prog_msg(prog=child_sol)
            child_sol.features = child_sol.eval_metrics

            if child_sol.fitness > prompt.fitness:
                logger.info("Child solution improves on parent prompt fitness.")
                prompt.fitness = child_sol.fitness
                prompt.features = child_sol.features

            ## CHILD SOL FEATURES
            if use_embedding:
                try:
                    logger.info(
                        f"Attempting to obtain embedding with model {embedding.model_name}..."
                    )
                    child_sol.embedding, prompt_tok = await embedding.embed(child_sol.code)
                    logger.info(
                        f"Successfully retrieved response, using {prompt_tok} prompt tokens"
                    )

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
                    error_info: Dict[str, Any] = {
                        "epoch": epoch,
                        "motive": "generate_embedding",
                        "error_msg": str(err),
                    }
                    evolve_state["errors"].append(error_info)

            logger.info(f"Child solution -> {child_sol}.")

            ## ADD TO DB
            logger.info("Adding child_sol to sol_db.")
            sol_db.add(child_sol)

            if child_sol.id == sol_db.best_prog_id:
                logger.info(f"New best program found -> {child_sol.fitness}.")
                improved_local_fitness = True
            else:
                logger.info(
                    (
                        f"New program is worse than best -> {child_sol.fitness}"
                        f" <= {sol_db.programs[sol_db.best_prog_id].fitness}."
                    )
                )

        # MIGRATION
        if isl_data.in_neigh or isl_data.out_neigh:
            if epoch % evolve_config.get("migration_interval", 20) == 0:
                logger.info("=== MIGRATION STEP ===")
                out_migrants: List[Program] = sol_db.get_migrants(
                    migration_rate=evolve_config.get("migration_rate", 0.1)
                )
                in_migrants: List[Program] = sync_migrate(
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

        # UPDATE EVOLVE STATE
        evolve_state["best_fit_hist"].append(sol_db.programs[sol_db.best_prog_id].fitness)
        evolve_state["avg_fit_hist"].append(
            np.mean(np.array([sol.fitness for sol in sol_db.programs.values()]))
        )
        evolve_state["exploration"].append(exploration)

        # CHECKPOINTING
        if epoch % evolve_config["ckpt"] == 0:
            # we synchronize here to avoid a relatively common problem where
            # a slower island fails to save a CKPT when an experiment is interrupted,
            # resulting in desynchronized ckpts. this does not solve the problem completely,
            # but it does make it extremely unlikely to occur.
            logger.info("=== CHECKPOINT STEP ===")
            logger.info("Waiting for other islands to arrive at barrier...")
            global_data.barrier.wait()
            logger.info("All islands arrived. Proceeding to save ckpt.")

            save_ckpt(
                curr_epoch=epoch,
                prompt_db=prompt_db,
                sol_db=sol_db,
                evolve_state=evolve_state,
                best_sol_path=args["isl_out_dir"].joinpath(
                    "best_sol"
                    + evaluator.language2extension[sol_db.programs[sol_db.best_prog_id].language]
                ),
                best_prompt_path=args["isl_out_dir"].joinpath("best_prompt.txt"),
                ckpt_dir=args["ckpt_dir"],
                logger=logger,
            )

        # EARLY STOPPING
        logger.info("=== GLOBAL EARLY STOPPING CHECK STEP ===")
        if improved_local_fitness:
            with global_data.lock:
                if global_data.best_sol.fitness.value < child_sol.fitness:
                    logger.info("Global best solution improved.")
                    global_data.best_sol.fitness.value = child_sol.fitness
                    global_data.best_sol.iteration_found.value = child_sol.iteration_found
                    global_data.best_sol.island_found.value = child_sol.island_found

        early_stopping_check(
            island_id=isl_data.id,
            num_islands=evolve_config["num_islands"],
            improved_local_fitness=improved_local_fitness,
            global_data=global_data,
            logger=logger,
        )

        if global_data.early_stop_counter.value > evolve_state["early_stop_counter"]:
            logger.info(
                (
                    f"Early stopping counter increased: {global_data.early_stop_counter.value}"
                    f"/{evolve_config['early_stopping_rounds']}"
                )
            )

        evolve_state["early_stop_counter"] = global_data.early_stop_counter.value
        if evolve_state["early_stop_counter"] == evolve_config["early_stopping_rounds"]:
            logger.info(
                (
                    f"EARLY STOPPING: {evolve_state['early_stop_counter']}"
                    " global consecutive epochs without improvement."
                )
            )
            break

        # END EPOCH SYNC
        logger.info("=== END EPOCH SYNC STEP ===")
        logger.info("Waiting for other islands to finish epoch...")
        global_data.barrier.wait()
        logger.info("All islands finished. Moving to next epoch.")

    logger.info("====== ALGORITHM FINISHED ======")

    logger.info(f"Best solution: {sol_db.programs[sol_db.best_prog_id]}")
    logger.info(f"Best prompt: {prompt_db.programs[prompt_db.best_prog_id]}")
    save_ckpt(
        curr_epoch=epoch,
        prompt_db=prompt_db,
        sol_db=sol_db,
        evolve_state=evolve_state,
        best_sol_path=args["isl_out_dir"].joinpath(
            "best_sol" + evaluator.language2extension[sol_db.programs[sol_db.best_prog_id].language]
        ),
        best_prompt_path=args["isl_out_dir"].joinpath("best_prompt.txt"),
        ckpt_dir=args["ckpt_dir"],
        logger=logger,
    )


async def codeevolve(args: Dict[str, Any], isl_data: IslandData, global_data: GlobalData) -> None:
    """Main entry point for the CodeEvolve algorithm on a single island.

    This function initializes all components needed for evolutionary program synthesis,
    sets up the initial population, and launches the evolutionary loop. It handles
    both fresh starts and checkpoint resumption.

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
    # LOGGER
    logger: logging.Logger = get_logger(
        island_id=isl_data.id,
        results_dir=args["isl_out_dir"],
        append_mode=(args["load_ckpt"] != 0),
        log_queue=global_data.log_queue,
        max_msg_sz=MAX_LOG_MSG_SZ,
    )

    logger.info("=== CodeEvolve ===")

    # EVOLVE COMPONENTS
    logger.info("====== PREPARING COMPONENTS ======")
    start_epoch: int = args["load_ckpt"]
    evolve_state: Dict[str, Any] = {
        "early_stop_counter": 0,
        "best_fit_hist": [],
        "avg_fit_hist": [],
        "errors": [],
        "tok_usage": [],
        "exploration": [],
    }

    config: Dict[Any, Any] = yaml.safe_load(open(args["cfg_path"], "r"))
    evolve_config = config["EVOLVE_CONFIG"]

    ensemble: LMEnsemble = LMEnsemble(
        models_cfg=config["ENSEMBLE"],
        api_key=args["api_key"],
        api_base=args["api_base"],
        logger=logger,
    )

    prompt_sampler = PromptSampler(
        aux_lm=OpenAILM(
            **config["SAMPLER_AUX_LM"],
            api_key=args["api_key"],
            api_base=args["api_base"],
        ),
        logger=logger,
    )

    evaluator: Evaluator = Evaluator(
        eval_path=Path(config["EVAL_FILE_NAME"]),
        cwd=args["inpt_dir"],
        timeout_s=config.get("EVAL_TIMEOUT", 1 * 60),
        max_mem_b=config.get("MAX_MEM_BYTES", 1 * 1024 * 1024 * 1024),
        mem_check_interval_s=config.get("MEM_CHECK_INTERVAL_S", 0.1),
        logger=logger,
    )

    embedding: Optional[OpenAIEmbedding] = None
    if evolve_config.get("use_embedding", False):
        assert (
            config.get("EMBEDDING", None) is not None
        ), "EMBEDDING model must be defined in config.yaml when use_embedding is true."
        embedding = OpenAIEmbedding(
            **config["EMBEDDING"],
            api_key=args["api_key"],
            api_base=args["api_base"],
        )

    if args["load_ckpt"]:
        prompt_db, sol_db, evolve_state = load_ckpt(args["load_ckpt"], args["ckpt_dir"])

        init_prompt: Program = prompt_db.programs[prompt_db.best_prog_id]
        init_sol: Program = sol_db.programs[sol_db.best_prog_id]
        init_sol.prompt_id = init_prompt.id

    else:
        logger.info("Starting anew.")
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
            id=isl_data.id,
            seed=config.get("SEED", None),
            max_alive=evolve_config.get("max_size", None),
            elite_map_type=map_elites_cfg.get("elite_map_type", None),
            features=features,
            **map_elites_cfg.get("elite_map_kwargs", {}),
        )

        sol_db: ProgramDatabase = ProgramDatabase(
            id=isl_data.id,
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
            island_found=isl_data.id,
        )
        prompt_db.add(init_prompt)

        with open(
            args["inpt_dir"]
            .joinpath(config["CODEBASE_PATH"])
            .joinpath(config["INIT_FILE_DATA"]["filename"])
        ) as f:
            init_sol: Program = Program(
                id=str(uuid4()),
                code=f.read(),
                language=config["INIT_FILE_DATA"]["language"],
                iteration_found=0,
                generation=0,
                island_found=isl_data.id,
            )

        evaluator.execute(init_sol)
        if init_sol.returncode == 0:
            init_sol.fitness = init_sol.eval_metrics[evolve_config["fitness_key"]]

        init_sol.prog_msg = format_prog_msg(prog=init_sol)
        init_sol.features = init_sol.eval_metrics

        sol_db.add(init_sol)

    logger.info(f"sol_db={sol_db}")
    logger.info(f"prompt_db={prompt_db}")
    logger.info(f"ensemble={ensemble}")
    logger.info(f"prompt_sampler={prompt_sampler}")
    logger.info(f"evaluator={evaluator}")
    logger.info(f"embedding={embedding}")
    logger.info(f"init_prog={init_sol}")

    # UPDATE GLOBAL BEST
    with global_data.lock:
        global_data.early_stop_counter.value = evolve_state["early_stop_counter"]
        if global_data.best_sol.fitness.value < init_sol.fitness:
            global_data.best_sol.fitness.value = init_sol.fitness
            global_data.best_sol.iteration_found.value = init_sol.iteration_found
            global_data.best_sol.island_found.value = init_sol.island_found

    if start_epoch == evolve_config["num_epochs"] or (
        evolve_state["early_stop_counter"] == evolve_config["early_stopping_rounds"]
    ):
        logger.info("Loaded checkpoint already finished the algorithm.")
        return

    await evolve_loop(
        start_epoch,
        evolve_state,
        init_sol,
        init_prompt,
        config,
        evolve_config,
        args,
        isl_data,
        global_data,
        sol_db,
        prompt_db,
        prompt_sampler,
        ensemble,
        evaluator,
        embedding,
        logger,
    )
