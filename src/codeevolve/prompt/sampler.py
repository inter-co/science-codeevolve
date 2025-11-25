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

from typing import Dict, List, Tuple, Optional
from collections import deque
import logging

from codeevolve.lm import OpenAILM
from codeevolve.database import Program, ProgramDatabase
from codeevolve.prompt.template import (
    PROG_TEMPLATE,
    EVOLVE_PROG_TASK_TEMPLATE,
    EVOLVE_PROG_WINSP_TASK_TEMPLATE,
    EVOLVE_PROMPT_TASK_TEMPLATE,
    EVOLVE_PROMPT_TEMPLATE,
    EVOLVE_PROG_TEMPLATE,
    INSP_PROG_TEMPLATE,
)


def format_prog_msg(prog: Program):
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


class PromptSampler:
    """Builds conversation prompts for evolutionary program generation.

    This class constructs prompts for language models by creating conversation
    histories from program lineages and incorporating inspiration programs.
    It supports both program evolution and meta-prompt evolution.
    """

    def __init__(self, aux_lm: OpenAILM, logger: Optional[logging.Logger] = None):
        """Initializes the prompt sampler with an auxiliary language model.

        Args:
            aux_lm: OpenAI language model instance for meta-prompt generation.
            logger: Logger instance for logging prompt operations.
        """

        self.aux_lm: OpenAILM = aux_lm

        self.logger: logging.Logger = logger if logger is not None else logging.getLogger(__name__)

    def __repr__(self) -> str:
        """Returns a string representation of the PromptSampler.

        Returns:
            A formatted string showing the auxiliary language model configuration.
        """
        return f"{self.__class__.__name__}(aux_lm={self.aux_lm})"

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
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": EVOLVE_PROMPT_TASK_TEMPLATE},
            {
                "role": "user",
                "content": EVOLVE_PROMPT_TEMPLATE.format(prompt=prompt.code, program=prog.prog_msg),
            },
        ]

        self.logger.info(f"Attempting to run meta_prompt on {self.aux_lm}...")

        response, prompt_tok, compl_tok = await self.aux_lm.generate(messages)

        self.logger.info(
            (
                f"Successfully retrieved response, using {prompt_tok} prompt tokens"
                f" and {compl_tok} completion tokens."
            )
        )

        return (response, prompt_tok, compl_tok)

    def build(
        self,
        prompt: Program,
        prog: Program,
        db: ProgramDatabase,
        inspirations: Optional[List[Program]] = None,
        max_chat_depth: Optional[int] = None,
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

        Returns:
            A list of message dictionaries following the OpenAI chat format,
            with 'role' and 'content' keys representing the conversation history.
        """
        messages: deque[Dict[str, str]] = deque()

        # recover chat
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

        # inspirations
        if inspirations and len(inspirations):
            insp_str: str = ""
            for i, inspiration in enumerate(inspirations):
                insp_str += INSP_PROG_TEMPLATE.format(counter=i + 1, program=inspiration.prog_msg)

            messages[-1]["content"] = insp_str + messages[-1]["content"]
            messages[0]["content"] += EVOLVE_PROG_WINSP_TASK_TEMPLATE
        else:
            messages[0]["content"] += EVOLVE_PROG_TASK_TEMPLATE

        return list(messages)
