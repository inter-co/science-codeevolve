# ===--------------------------------------------------------------------------------------===#
#
# Part of the CodeEvolve Project, under the Apache License v2.0.
# See https://github.com/inter-co/science-codeevolve/blob/main/LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0
#
# ===--------------------------------------------------------------------------------------===#
#
# This file implements the prompt sampler that builds the prompts for CodeEvolve.
#
# ===--------------------------------------------------------------------------------------===#

from typing import Any, Dict, List, Tuple, Optional, Union
from collections import deque

from codeevolve.lm.openai import (
    OpenAILM,
    MockOpenAILM,
    _create_lm_from_config,
)
from codeevolve.database import Program, ProgramDatabase
from codeevolve.prompt.template import (
    PROG_TEMPLATE,
    EVOLVE_PROG_TEMPLATE,
    INSP_PROG_TEMPLATE,
    EVOLVE_PROMPT_TEMPLATE,
    get_evolve_task_template,
    get_evolve_with_inspirations_task_template,
    get_explore_task_template,
    get_explore_with_inspirations_task_template,
    get_evolve_prompt_task_template,
)
from codeevolve.utils.constants import (
    DEFAULT_EVOLVE_START_MARKER,
    DEFAULT_EVOLVE_END_MARKER,
    DEFAULT_PROMPT_START_MARKER,
    DEFAULT_PROMPT_END_MARKER,
)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def format_prog_msg(prog: Program) -> str:
    """Formats a program's execution results into a standardized message string.

    This function creates a formatted message containing the program's code,
    execution results, and evaluation metrics using a predefined template.

    Args:
        prog: Program object containing code and execution results.

    Returns:
        A formatted string representation of the program and its execution results.

    Raises:
        ValueError: If the program does not have a returncode (hasn't been executed).
    """
    if prog.returncode is None:
        raise ValueError("Program must have a returncode in order to format message.")

    return PROG_TEMPLATE.format(
        language=prog.language,
        code=prog.code,
        eval_metrics=prog.eval_metrics,
        returncode=prog.returncode,
        warning=prog.warning,
        error=prog.error,
    )


# ---------------------------------------------------------------------------
# Prompt sampler
# ---------------------------------------------------------------------------


class PromptSampler:
    """Builds conversation prompts for evolutionary program generation.

    This class constructs prompts for language models by creating conversation
    histories from program lineages and incorporating inspiration programs.
    It supports both program evolution and meta-prompt evolution.

    Supports mock models for debugging: set model_name to "MOCK" (or any name
    starting with "MOCK") in the configuration to use a MockOpenAILM that returns
    identity SEARCH/REPLACE operations without making API requests.

    Attributes:
        aux_lm_cfg: Configuration dictionary for the auxiliary language model.
        aux_lm: The auxiliary language model instance.
        evolve_start_marker: Marker indicating the start of an evolve block.
        evolve_end_marker: Marker indicating the end of an evolve block.
        prompt_start_marker: Marker indicating the start of a prompt block.
        prompt_end_marker: Marker indicating the end of a prompt block.
    """

    def __init__(
        self,
        aux_lm_cfg: Dict[str, Any],
        api_key: str,
        api_base: str,
        evolve_start_marker: str = DEFAULT_EVOLVE_START_MARKER,
        evolve_end_marker: str = DEFAULT_EVOLVE_END_MARKER,
        prompt_start_marker: str = DEFAULT_PROMPT_START_MARKER,
        prompt_end_marker: str = DEFAULT_PROMPT_END_MARKER,
    ):
        """Initializes the prompt sampler with an auxiliary language model configuration.

        Args:
            aux_lm_cfg: Configuration dictionary for the auxiliary language model.
                       To use a mock model, set model_name to "MOCK" or any
                       name starting with "MOCK".
            api_key: API key for authentication.
            api_base: Base URL for the API endpoint.
            evolve_start_marker: Marker indicating the start of an evolve block.
            evolve_end_marker: Marker indicating the end of an evolve block.
            prompt_start_marker: Marker indicating the start of a prompt block.
            prompt_end_marker: Marker indicating the end of a prompt block.
        """
        self.aux_lm_cfg: Dict[str, Any] = aux_lm_cfg
        self.aux_lm: Union[OpenAILM, MockOpenAILM] = _create_lm_from_config(
            model_cfg=aux_lm_cfg,
            api_key=api_key,
            api_base=api_base,
        )

        self.evolve_start_marker: str = evolve_start_marker
        self.evolve_end_marker: str = evolve_end_marker
        self.prompt_start_marker: str = prompt_start_marker
        self.prompt_end_marker: str = prompt_end_marker

        self.configure_mocks(prompt_start_marker, prompt_end_marker)

    def __repr__(self) -> str:
        """Returns a string representation of the PromptSampler.

        Returns:
            A formatted string showing the auxiliary language model configuration.
        """
        return f"{self.__class__.__name__}(aux_lm={self.aux_lm})"

    def configure_mocks(self, start_marker: str, end_marker: str) -> None:
        """Configures evolve block markers for the auxiliary LM if it's a mock.

        This method should be called after creating the sampler to set the
        correct markers from evolve_config. Only affects MockOpenAILM instances.

        Args:
            start_marker: The marker indicating the start of an evolve block.
            end_marker: The marker indicating the end of an evolve block.
        """
        if isinstance(self.aux_lm, MockOpenAILM):
            self.aux_lm.start_marker = start_marker
            self.aux_lm.end_marker = end_marker

    async def meta_prompt(self, prompt: Program, prog: Program) -> Tuple[str, int, int]:
        """Generates an evolved prompt using meta-prompting.

        This method uses the auxiliary language model to evolve a prompt based
        on a program's performance, creating potentially better prompts for
        future program generation.

        Args:
            prompt: The current prompt program to evolve.
            prog: The program generated using the prompt, with execution results.

        Returns:
            A tuple containing:
                - The evolved prompt text
                - Number of prompt tokens used
                - Number of completion tokens used
        """
        evolve_prompt_task = get_evolve_prompt_task_template(
            self.prompt_start_marker, self.prompt_end_marker
        )

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": evolve_prompt_task},
            {
                "role": "user",
                "content": EVOLVE_PROMPT_TEMPLATE.format(prompt=prompt.code, program=prog.prog_msg),
            },
        ]

        response, prompt_tok, compl_tok = await self.aux_lm.generate(messages)

        return (response, prompt_tok, compl_tok)

    def build(
        self,
        prompt: Program,
        prog: Program,
        db: ProgramDatabase,
        inspirations: Optional[List[Program]] = None,
        max_chat_depth: Optional[int] = None,
        exploitation: bool = False,
    ) -> List[Dict[str, str]]:
        """Builds a conversation prompt from program lineage and inspirations.

        This method constructs a conversation history by tracing back through
        a program's evolutionary lineage, creating a chat-like sequence that
        can be used to generate the next program iteration. It optionally
        includes inspiration programs and limits conversation depth.

        Args:
            prompt: The system prompt program defining the task and instructions.
            prog: The current program to build conversation history from.
            db: Program database containing the evolutionary lineage.
            inspirations: Optional list of programs to include as inspiration examples.
            max_chat_depth: Maximum depth to trace back in the conversation history.
                           If None, traces back to the root program.
            exploitation: If True, use exploitation templates; if False, use exploration templates.

        Returns:
            A list of message dictionaries following the OpenAI chat format,
            with 'role' and 'content' keys representing the conversation history.
        """
        messages: deque[Dict[str, str]] = deque()

        # Recover chat history
        curr_pid: str = prog.id
        curr_depth: int = 0
        while db.programs[curr_pid].parent_id is not None and (
            (max_chat_depth is None) or (curr_depth < max_chat_depth)
        ):
            messages.appendleft(
                {
                    "role": "user",
                    "content": EVOLVE_PROG_TEMPLATE.format(program=db.programs[curr_pid].prog_msg),
                }
            )
            messages.appendleft({"role": "assistant", "content": db.programs[curr_pid].model_msg})
            curr_pid = db.programs[curr_pid].parent_id
            curr_depth += 1

        messages.appendleft(
            {
                "role": "user",
                "content": EVOLVE_PROG_TEMPLATE.format(program=db.programs[curr_pid].prog_msg),
            }
        )
        messages.appendleft({"role": "system", "content": prompt.code})

        task_template: str
        if inspirations and len(inspirations):
            insp_str: str = ""
            for i, inspiration in enumerate(inspirations):
                insp_str += INSP_PROG_TEMPLATE.format(counter=i + 1, program=inspiration.prog_msg)

            messages[-1]["content"] = insp_str + messages[-1]["content"]

            if exploitation:
                task_template = get_evolve_with_inspirations_task_template(
                    self.evolve_start_marker, self.evolve_end_marker
                )
            else:
                task_template = get_explore_with_inspirations_task_template(
                    self.evolve_start_marker, self.evolve_end_marker
                )
        else:
            if exploitation:
                task_template = get_evolve_task_template(
                    self.evolve_start_marker, self.evolve_end_marker
                )
            else:
                task_template = get_explore_task_template(
                    self.evolve_start_marker, self.evolve_end_marker
                )

        messages[0]["content"] += task_template

        return list(messages)
