# ===--------------------------------------------------------------------------------------===#
#
# Part of the CodeEvolve Project, under the Apache License v2.0.
# See https://github.com/inter-co/science-codeevolve/blob/main/LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0
#
# ===--------------------------------------------------------------------------------------===#
#
# This file implements wrappers for making requests to LLM providers.
#
# ===--------------------------------------------------------------------------------------===#

import asyncio
import logging
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import uuid4

import httpx
from openai import AsyncOpenAI

from codeevolve.lm.base import BaseEmbedding, BaseEnsemble, BaseLM
from codeevolve.utils.constants import (
    DEFAULT_EVOLVE_END_MARKER,
    DEFAULT_EVOLVE_START_MARKER,
    MOCK_MODEL_PREFIX,
)
from codeevolve.utils.parsing import find_evolve_block_spans

# ---------------------------------------------------------------------------
# Language model classes
# ---------------------------------------------------------------------------


@dataclass
class OpenAILM(BaseLM):
    """A dataclass for managing OpenAI language model interactions.

    This class provides an interface for communicating with OpenAI-compatible APIs,
    handling configuration parameters, retries, and response generation.

    Attributes:
        model_name: The name of the model to use for generation.
        temp: Temperature parameter for controlling randomness.
        top_p: Nucleus sampling parameter for controlling diversity.
        max_tok: Maximum number of tokens to generate.
        seed: Random seed for reproducible outputs.
        weight: Weight for ensemble selection when used in LMEnsemble.
        retries: Number of retry attempts on failure.
        api_base: Base URL for the API endpoint.
        api_key: API key for authentication.
        verify_ssl: Whether to verify SSL certificates.
        client: The async OpenAI client instance (auto-initialized).
    """

    model_name: Optional[str] = None

    temp: float = 0.7
    top_p: float = 0.95
    max_tok: Optional[int] = None

    seed: Optional[int] = None
    weight: float = 1
    retries: int = 3

    api_base: Optional[str] = None
    api_key: Optional[str] = None
    verify_ssl: Optional[bool] = None

    client: AsyncOpenAI = field(init=False, repr=False)

    def __repr__(self):
        """Returns a string representation of the OpenAILM instance.

        Returns:
            A formatted string showing key configuration parameters.
        """
        return (
            f"{self.__class__.__name__}"
            "("
            f"model_name={self.model_name},"
            f"temp={self.temp},"
            f"top_p={self.top_p},"
            f"weight={self.weight}"
            ")"
        )

    def __post_init__(self):
        """Initializes the AsyncOpenAI client after dataclass initialization.

        Sets up the HTTP client with SSL verification settings and creates
        the AsyncOpenAI client instance with the provided configuration.
        """
        http_client = httpx.AsyncClient(verify=self.verify_ssl)

        self.client = AsyncOpenAI(
            api_key=self.api_key, base_url=self.api_base, http_client=http_client
        )

    async def generate(self, messages: List[Dict[str, str]]) -> Tuple[str, int, int]:
        """Generates a response from the language model.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
                     following the OpenAI chat format.

        Returns:
            A tuple containing:
                - Generated text response
                - Number of prompt tokens used
                - Number of completion tokens used

        Raises:
            ConnectionError: If all retry attempts fail to get a response.
        """
        params: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "max_completion_tokens": self.max_tok,
            "user": f"user_{str(uuid4())}",
            "seed": self.seed,
            "top_p": self.top_p,
            "temperature": self.temp,
        }

        retry_delay: int = 1
        for attempt in range(self.retries + 1):
            try:
                ret = await self.client.chat.completions.create(**params)
                content: str = ret.choices[0].message.content
                content = content if content is not None else ""
                return (content, ret.usage.prompt_tokens, ret.usage.completion_tokens)
            except Exception as err:
                if attempt < self.retries:
                    await asyncio.sleep(retry_delay)
                    retry_delay = retry_delay << 1
                else:
                    raise ConnectionError(
                        (
                            f"Failed to fetch LM response after {self.retries+1} attempts"
                            f"(Error:{str(err)})."
                        )
                    )


@dataclass
class MockOpenAILM(BaseLM):
    """A mock language model that returns identity SEARCH/REPLACE operations.

    This class is designed for debugging purposes, allowing CodeEvolve to run
    without making actual API requests. When given a solution to evolve, it
    returns a SEARCH/REPLACE diff that keeps the code unchanged (identity operation).

    To use this mock, set the model_name to "MOCK" (or any name starting with "MOCK")
    in your configuration.

    Attributes:
        model_name: The name of the mock model (for logging/display purposes).
        start_marker: The marker indicating the start of an evolve block.
        end_marker: The marker indicating the end of an evolve block.
        weight: Weight for ensemble selection when used in an ensemble.
    """

    model_name: str = MOCK_MODEL_PREFIX
    start_marker: str = DEFAULT_EVOLVE_START_MARKER
    end_marker: str = DEFAULT_EVOLVE_END_MARKER
    weight: float = 1.0

    temp: float = 0.0
    top_p: float = 1.0
    max_tok: Optional[int] = None
    seed: Optional[int] = None
    retries: int = 0
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    verify_ssl: Optional[bool] = None

    debug_sleep: float = 0.0

    def __repr__(self) -> str:
        """Returns a string representation of the MockOpenAILM instance.

        Returns:
            A formatted string showing key configuration parameters.
        """
        return f"{self.__class__.__name__}" f"(model_name={self.model_name}"

    def _extract_code_from_messages(self, messages: List[Dict[str, str]]) -> Optional[str]:
        """Extracts the target program code from the conversation messages.

        The last message is always the user message containing the solution code
        formatted as a markdown code block (```language ... ```).

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys.

        Returns:
            The extracted code string, or None if no code block is found.
        """
        if not messages:
            return None

        last_message_content: str = messages[-1].get("content", "")
        code_block_pattern: str = r"```\w*\n(.*?)```"
        matches: List[str] = re.findall(code_block_pattern, last_message_content, re.DOTALL)

        if matches:
            return matches[-1]

        return None

    def _generate_identity_diff(self, code: str) -> str:
        """Generates identity SEARCH/REPLACE diffs for all evolve blocks.

        For each evolve block in the code, generates a diff that replaces the
        block content with itself (identity operation).

        Args:
            code: The source code containing evolve blocks.

        Returns:
            A string containing SEARCH/REPLACE blocks for each evolve block.
            Returns an empty diff block if no evolve blocks are found.
        """
        evolve_regex: str = (
            rf"\s*{re.escape(self.start_marker)}\s*\n?(.*?)\n?\s*{re.escape(self.end_marker)}"
        )

        try:
            evolve_spans: List[Tuple[int, int]] = find_evolve_block_spans(
                parent_code=code, evolve_regex=evolve_regex
            )
        except Exception:
            return "<<<<<<< SEARCH\n\n=======\n\n>>>>>>> REPLACE"

        diff_parts: List[str] = []
        for start, end in evolve_spans:
            block_content: str = code[start:end]
            diff_block: str = (
                f"<<<<<<< SEARCH\n"
                f"{block_content}\n"
                f"=======\n"
                f"{block_content}\n"
                f">>>>>>> REPLACE"
            )
            diff_parts.append(diff_block)

        return "\n\n".join(diff_parts)

    async def generate(self, messages: List[Dict[str, str]]) -> Tuple[str, int, int]:
        """Generates an identity SEARCH/REPLACE response for the input code.

        This method extracts the code from the messages, finds all evolve blocks,
        and returns a diff that keeps the code unchanged.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
                     following the OpenAI chat format.

        Returns:
            A tuple containing:
                - Generated identity diff string
                - Number of prompt tokens used (mocked as 0)
                - Number of completion tokens used (mocked as 0)
        """
        code: Optional[str] = self._extract_code_from_messages(messages)

        if code is None:
            response: str = "<<<<<<< SEARCH\n\n=======\n\n>>>>>>> REPLACE"
            return (response, 0, 0)

        response: str = self._generate_identity_diff(code)

        prompt_tokens: int = 0
        completion_tokens: int = 0

        time.sleep(self.debug_sleep)

        return (response, prompt_tokens, completion_tokens)


# ---------------------------------------------------------------------------
# Factory and ensemble
# ---------------------------------------------------------------------------


def _create_lm_from_config(
    model_cfg: Dict[str, Any],
    api_key: str,
    api_base: str,
    mock_start_marker: str = DEFAULT_EVOLVE_START_MARKER,
    mock_end_marker: str = DEFAULT_EVOLVE_END_MARKER,
) -> Union[OpenAILM, MockOpenAILM]:
    """Creates an LM instance based on the model configuration.

    If the model_name starts with "MOCK", creates a MockOpenAILM instance.
    Otherwise, creates a regular OpenAILM instance.

    Args:
        model_cfg: Configuration dictionary for the model.
        api_key: API key for authentication (used only for real models).
        api_base: Base URL for the API endpoint (used only for real models).
        mock_start_marker: Start marker for mock model evolve blocks.
        mock_end_marker: End marker for mock model evolve blocks.

    Returns:
        Either an OpenAILM or MockOpenAILM instance based on the model name.
    """
    model_name: str = model_cfg.get("model_name", "")
    is_mock_model: bool = model_name.upper().startswith(MOCK_MODEL_PREFIX)
    if is_mock_model:
        mock_cfg: Dict[str, Any] = {
            "model_name": model_cfg.get("model_name", MOCK_MODEL_PREFIX),
            "weight": model_cfg.get("weight", 1.0),
            "debug_sleep": model_cfg.get("debug_sleep", 0.0),
            "start_marker": mock_start_marker,
            "end_marker": mock_end_marker,
        }
        return MockOpenAILM(**mock_cfg)
    else:
        return OpenAILM(**model_cfg, api_key=api_key, api_base=api_base)


class OpenAIEnsemble(BaseEnsemble):
    """An ensemble of language models for weighted random selection.

    This class manages multiple OpenAI language models and selects one randomly
    based on their configured weights for each generation request.

    Supports mock models for debugging: set model_name to "MOCK" (or any name
    starting with "MOCK") in the configuration to use a MockOpenAILM that returns
    identity SEARCH/REPLACE operations without making API requests.
    """

    def __init__(
        self,
        models_cfg: List[Dict[str, Any]],
        api_key: str,
        api_base: str,
        seed: Optional[int] = None,
        logger: Optional[logging.Logger] = None,
        start_marker: str = DEFAULT_EVOLVE_START_MARKER,
        end_marker: str = DEFAULT_EVOLVE_END_MARKER,
    ):
        """Initializes the language model ensemble.

        Args:
            models_cfg: List of configuration dictionaries for each model.
                       To use a mock model, set model_name to "MOCK" or any
                       name starting with "MOCK".
            api_key: API key for authentication.
            api_base: Base URL for the API endpoint.
            seed: Random seed for reproducible model selection.
            logger: Logger instance for logging operations.
            start_marker: Start marker for mock model evolve blocks.
            end_marker: End marker for mock model evolve blocks.
        """

        self.models_cfg: List[Dict[str, Any]] = models_cfg
        self.models: List[Union[OpenAILM, MockOpenAILM]] = [
            _create_lm_from_config(
                model_cfg,
                api_key=api_key,
                api_base=api_base,
                mock_start_marker=start_marker,
                mock_end_marker=end_marker,
            )
            for model_cfg in models_cfg
        ]

        self.weights: List[float] = [model.weight for model in self.models]
        total: float = sum(self.weights)
        self.weights = [weight / total for weight in self.weights]

        self.random_state: random.Random = random.Random()
        self.seed: Optional[int] = seed
        if self.seed:
            self.random_state.seed(self.seed)

        self.logger: logging.Logger = logger if logger is not None else logging.getLogger(__name__)

    def __repr__(self) -> str:
        """Returns a string representation of the ensemble.

        Returns:
            A multi-line string showing the number of models and their details.
        """
        lines: List[str] = [f"{self.__class__.__name__}("]
        lines.append(f"  (model): {len(self.models)}")

        for i, model in enumerate(self.models):
            lines.append(f"  ({i}): {model}")

        lines.append(")")
        return "\n".join(lines)

    def configure_mocks(self, start_marker: str, end_marker: str) -> None:
        """Configures evolve block markers for any mock models in the ensemble.

        This method should be called after creating the ensemble to set the
        correct markers from evolve_config. Only affects MockOpenAILM instances.

        Args:
            start_marker: The marker indicating the start of an evolve block.
            end_marker: The marker indicating the end of an evolve block.
        """
        for model in self.models:
            if isinstance(model, MockOpenAILM):
                model.start_marker = start_marker
                model.end_marker = end_marker

    async def generate(self, messages: List[Dict[str, str]]) -> Tuple[int, int, str, int]:
        """Generates a response using a randomly selected model from the ensemble.

        Args:
            messages: List of message dictionaries following OpenAI chat format.

        Returns:
            A tuple containing:
                - Selected model ID (index in the ensemble)
                - Generated text response
                - Number of prompt tokens used
                - Number of completion tokens used

        Raises:
            ConnectionError: If the selected model fails to generate a response.
        """
        model_id: int = self.random_state.choices([*range(len(self.models))], self.weights)[0]

        self.logger.info(f"Attempting to run prompt on {self.models[model_id]}...")

        response, prompt_tok, compl_tok = await self.models[model_id].generate(messages)

        self.logger.info(
            (
                f"Successfully retrieved response, using {prompt_tok} prompt tokens"
                f" and {compl_tok} completion tokens."
            )
        )

        return (model_id, response, prompt_tok, compl_tok)


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------


@dataclass
class OpenAIEmbedding(BaseEmbedding):
    """A dataclass for managing OpenAI embedding computations.

    This class provides an interface for computing text embeddings using
    OpenAI-compatible APIs, handling configuration parameters, retries,
    and batch processing.

    Attributes:
        model_name: The name of the embedding model to use.
        dimensions: Optional dimensionality reduction for the embeddings.
        encoding_format: Format for returned embeddings ('float' or 'base64').
        retries: Number of retry attempts on failure.
        api_base: Base URL for the API endpoint.
        api_key: API key for authentication.
        verify_ssl: Whether to verify SSL certificates.
        client: The async OpenAI client instance (auto-initialized).
    """

    model_name: Optional[str] = None
    dimensions: Optional[int] = None
    encoding_format: str = "float"

    retries: int = 3
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    verify_ssl: Optional[bool] = None

    client: AsyncOpenAI = field(init=False, repr=False)

    def __repr__(self):
        """Returns a string representation of the OpenAIEmbedding instance.

        Returns:
            A formatted string showing key configuration parameters.
        """
        return (
            f"{self.__class__.__name__}"
            "("
            f"model_name={self.model_name},"
            f"dimensions={self.dimensions}"
            ")"
        )

    def __post_init__(self):
        """Initializes the AsyncOpenAI client after dataclass initialization.

        Sets up the HTTP client with SSL verification settings and creates
        the AsyncOpenAI client instance with the provided configuration.
        """
        http_client = httpx.AsyncClient(verify=self.verify_ssl)
        self.client = AsyncOpenAI(
            api_key=self.api_key, base_url=self.api_base, http_client=http_client
        )

    async def embed_batch(self, texts: List[str]) -> Tuple[List[List[float]], int]:
        """Computes embeddings for multiple text inputs in a single request.

        Args:
            texts: List of text strings to embed.

        Returns:
            A tuple containing:
                - List of embedding vectors (each vector is a list of floats)
                - Total number of tokens used

        Raises:
            ConnectionError: If all retry attempts fail to get a response.
        """
        params: Dict[str, Any] = {
            "model": self.model_name,
            "input": texts,
            "encoding_format": self.encoding_format,
        }

        if self.dimensions is not None:
            params["dimensions"] = self.dimensions

        retry_delay: int = 1
        for attempt in range(self.retries + 1):
            try:
                response = await self.client.embeddings.create(**params)
                embeddings = [data.embedding for data in response.data]
                total_tokens = response.usage.total_tokens

                if len(texts) == 1:
                    return (embeddings[0], total_tokens)
                return (embeddings, total_tokens)

            except Exception as err:
                if attempt < self.retries:
                    await asyncio.sleep(retry_delay)
                    retry_delay = retry_delay << 1
                else:
                    raise ConnectionError(
                        f"Failed to compute embeddings after {self.retries + 1} attempts "
                        f"(Error: {str(err)})."
                    )

    async def embed(self, text: str) -> Tuple[List[float], int]:
        """Computes embeddings for a single text input.

        Args:
            text: The text string to embed.

        Returns:
            A tuple containing:
                - List of embedding values (floats)
                - Number of tokens used

        Raises:
            ConnectionError: If all retry attempts fail to get a response.
        """
        return await self.embed_batch([text])
