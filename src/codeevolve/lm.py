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

from typing import Any, Dict, List, Optional, Tuple

import asyncio
from dataclasses import dataclass, field
import logging
import random
import httpx

from uuid import uuid4

from openai import AsyncOpenAI

# TODO: classes for open-source LM's executing locally.


@dataclass
class OpenAILM:
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
    max_tok: int = None

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
        format_msgs = messages.copy()
        params: Dict[str, Any] = {
            "model": self.model_name,
            "messages": format_msgs,
            "max_completion_tokens": self.max_tok,
            "user": f"user_{str(uuid4())}",
            "seed": getattr(self, "seed", None),
            "top_p": getattr(self, "top_p", 0.95),
            "temperature": getattr(self, "temp", 1),
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


class LMEnsemble:
    """An ensemble of language models for weighted random selection.

    This class manages multiple OpenAI language models and selects one randomly
    based on their configured weights for each generation request.
    """

    def __init__(
        self,
        models_cfg: List[Dict[Any, Any]],
        api_key: str,
        api_base: str,
        seed: int = None,
        logger: Optional[logging.Logger] = None,
    ):
        """Initializes the language model ensemble.

        Args:
            models_cfg: List of configuration dictionaries for each model.
            api_key: API key for authentication.
            api_base: Base URL for the API endpoint.
            seed: Random seed for reproducible model selection.
            logger: Logger instance for logging operations.
        """
        self.models_cfg: List[Dict[Any, Any]] = models_cfg
        self.models: List[OpenAILM] = [
            OpenAILM(**model_cfg, api_key=api_key, api_base=api_base) for model_cfg in models_cfg
        ]

        self.weights: List[float] = [model.weight for model in self.models]
        total = sum(self.weights)
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

    async def generate(self, messages: List[Dict[str, str]]) -> Tuple[int, str, int]:
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


@dataclass
class OpenAIEmbedding:
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
