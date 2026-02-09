# ===--------------------------------------------------------------------------------------===#
#
# Part of the CodeEvolve Project, under the Apache License v2.0.
# See https://github.com/inter-co/science-codeevolve/blob/main/LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0
#
# ===--------------------------------------------------------------------------------------===#
#
# This file defines abstract base classes for language model interfaces.
#
# ===--------------------------------------------------------------------------------------===#

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple


class BaseLM(ABC):
    """Abstract base class for language models."""

    @abstractmethod
    async def generate(self, messages: List[Dict[str, str]]) -> Tuple[str, int, int]:
        """Generates a response from the language model.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys.

        Returns:
            A tuple containing:
                - Generated text response
                - Number of prompt tokens used
                - Number of completion tokens used
        """
        pass


class BaseEnsemble(ABC):
    """Abstract base class for language model ensembles."""

    @abstractmethod
    async def generate(self, messages: List[Dict[str, str]]) -> Tuple[int, str, int, int]:
        """Generates a response using a model from the ensemble.

        Args:
            messages: List of message dictionaries.

        Returns:
            A tuple containing:
                - Selected model ID
                - Generated text response
                - Number of prompt tokens used
                - Number of completion tokens used
        """
        pass


class BaseEmbedding(ABC):
    """Abstract base class for embedding models."""

    @abstractmethod
    async def embed(self, text: str) -> Tuple[List[float], int]:
        """Computes embeddings for a single text input.

        Args:
            text: The text string to embed.

        Returns:
            A tuple containing:
                - List of embedding values
                - Number of tokens used
        """
        pass

    @abstractmethod
    async def embed_batch(self, texts: List[str]) -> Tuple[List[List[float]], int]:
        """Computes embeddings for multiple text inputs.

        Args:
            texts: List of text strings to embed.

        Returns:
            A tuple containing:
                - List of embedding vectors
                - Total number of tokens used
        """
        pass
