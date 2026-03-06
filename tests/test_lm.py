# ===--------------------------------------------------------------------------------------===#
#
# Part of the CodeEvolve Project, under the Apache License v2.0.
# See https://github.com/inter-co/science-codeevolve/blob/main/LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0
#
# ===--------------------------------------------------------------------------------------===#
#
# This file implements unit tests for the language model wrappers.
#
# ===--------------------------------------------------------------------------------------===#

from typing import Any, Dict, List, Optional, Tuple, Union

import pytest

from codeevolve.lm.openai import (
    MockOpenAILM,
    OpenAIEnsemble,
    OpenAILM,
    _create_lm_from_config,
)

# ---------------------------------------------------------------------------
# MockOpenAILM
# ---------------------------------------------------------------------------


class TestMockOpenAILM:
    """Test suite for the MockOpenAILM class."""

    def test_creation(self):
        """Tests that MockOpenAILM can be created with defaults."""
        mock: MockOpenAILM = MockOpenAILM()
        assert mock.model_name == "MOCK"
        assert mock.weight == 1.0

    def test_custom_markers(self):
        """Tests that custom markers can be set."""
        mock: MockOpenAILM = MockOpenAILM(start_marker="// START", end_marker="// END")
        assert mock.start_marker == "// START"
        assert mock.end_marker == "// END"

    @pytest.mark.asyncio
    async def test_generate_identity_diff(self):
        """Tests that generate returns identity SEARCH/REPLACE for evolve blocks."""
        mock: MockOpenAILM = MockOpenAILM()
        code: str = """# EVOLVE-BLOCK-START
def foo():
    return 1
# EVOLVE-BLOCK-END"""

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"```python\n{code}\n```"},
        ]
        response: str
        prompt_tok: int
        compl_tok: int
        response, prompt_tok, compl_tok = await mock.generate(messages)

        assert "<<<<<<< SEARCH" in response
        assert "=======" in response
        assert ">>>>>>> REPLACE" in response
        assert prompt_tok == 0
        assert compl_tok == 0

    @pytest.mark.asyncio
    async def test_generate_no_code(self):
        """Tests fallback when no code block is found in messages."""
        mock: MockOpenAILM = MockOpenAILM()
        messages: List[Dict[str, str]] = [
            {"role": "user", "content": "no code here"},
        ]
        response, _, _ = await mock.generate(messages)
        assert "<<<<<<< SEARCH" in response

    @pytest.mark.asyncio
    async def test_generate_empty_messages(self):
        """Tests fallback for empty message list."""
        mock: MockOpenAILM = MockOpenAILM()
        response, _, _ = await mock.generate([])
        assert "<<<<<<< SEARCH" in response

    def test_extract_code_from_messages(self):
        """Tests code extraction from formatted messages."""
        mock: MockOpenAILM = MockOpenAILM()
        messages: List[Dict[str, str]] = [
            {"role": "user", "content": "```python\nprint('hello')\n```"},
        ]
        code: Optional[str] = mock._extract_code_from_messages(messages)
        assert code is not None
        assert "print('hello')" in code

    def test_extract_code_no_block(self):
        """Tests that extraction returns None when no code block exists."""
        mock: MockOpenAILM = MockOpenAILM()
        messages: List[Dict[str, str]] = [
            {"role": "user", "content": "plain text message"},
        ]
        code: Optional[str] = mock._extract_code_from_messages(messages)
        assert code is None


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


class TestCreateLmFromConfig:
    """Test suite for the _create_lm_from_config factory function."""

    def test_creates_mock_for_mock_name(self):
        """Tests that MOCK model name creates a MockOpenAILM."""
        config: Dict[str, Any] = {"model_name": "MOCK"}
        lm: Union[OpenAILM, MockOpenAILM] = _create_lm_from_config(
            config, api_key="key", api_base="http://localhost"
        )
        assert isinstance(lm, MockOpenAILM)

    def test_creates_mock_for_mock_prefix(self):
        """Tests that model names starting with MOCK create MockOpenAILM."""
        config: Dict[str, Any] = {"model_name": "MOCK_CUSTOM_V2"}
        lm: Union[OpenAILM, MockOpenAILM] = _create_lm_from_config(
            config, api_key="key", api_base="http://localhost"
        )
        assert isinstance(lm, MockOpenAILM)

    def test_creates_mock_case_insensitive(self):
        """Tests that mock detection is case-insensitive."""
        config: Dict[str, Any] = {"model_name": "mock_test"}
        lm: Union[OpenAILM, MockOpenAILM] = _create_lm_from_config(
            config, api_key="key", api_base="http://localhost"
        )
        assert isinstance(lm, MockOpenAILM)

    def test_creates_real_for_real_name(self):
        """Tests that non-MOCK model name creates an OpenAILM."""
        config: Dict[str, Any] = {"model_name": "gpt-4", "temp": 0.5}
        lm: Union[OpenAILM, MockOpenAILM] = _create_lm_from_config(
            config, api_key="test_key", api_base="http://localhost"
        )
        assert isinstance(lm, OpenAILM)
        assert lm.model_name == "gpt-4"
        assert lm.temp == 0.5

    def test_mock_passes_custom_markers(self):
        """Tests that custom markers are passed through to the mock."""
        config: Dict[str, Any] = {"model_name": "MOCK"}
        lm: Union[OpenAILM, MockOpenAILM] = _create_lm_from_config(
            config,
            api_key="key",
            api_base="http://localhost",
            mock_start_marker="// START",
            mock_end_marker="// END",
        )
        assert isinstance(lm, MockOpenAILM)
        assert lm.start_marker == "// START"
        assert lm.end_marker == "// END"


# ---------------------------------------------------------------------------
# OpenAIEnsemble (with mock models)
# ---------------------------------------------------------------------------


class TestOpenAIEnsemble:
    """Test suite for the OpenAIEnsemble class using mock models."""

    def _make_mock_ensemble(self, num_models: int = 2) -> OpenAIEnsemble:
        """Helper to create an ensemble of mock models."""
        models_cfg: List[Dict[str, Any]] = [
            {"model_name": f"MOCK_{i}", "weight": 1.0} for i in range(num_models)
        ]
        return OpenAIEnsemble(
            models_cfg=models_cfg,
            api_key="test_key",
            api_base="http://localhost",
            seed=42,
        )

    def test_creation(self):
        """Tests that ensemble is created with the correct number of models."""
        ensemble: OpenAIEnsemble = self._make_mock_ensemble(3)
        assert len(ensemble.models) == 3
        assert len(ensemble.weights) == 3

    def test_weights_normalized(self):
        """Tests that model weights are normalized to sum to 1."""
        models_cfg: List[Dict[str, Any]] = [
            {"model_name": "MOCK_0", "weight": 2.0},
            {"model_name": "MOCK_1", "weight": 3.0},
        ]
        ensemble: OpenAIEnsemble = OpenAIEnsemble(
            models_cfg=models_cfg,
            api_key="test_key",
            api_base="http://localhost",
        )
        total: float = sum(ensemble.weights)
        assert abs(total - 1.0) < 1e-6

    @pytest.mark.asyncio
    async def test_generate_returns_tuple(self):
        """Tests that ensemble generate returns the expected 4-tuple."""
        ensemble: OpenAIEnsemble = self._make_mock_ensemble(2)
        messages: List[Dict[str, str]] = [
            {
                "role": "user",
                "content": "```python\n# EVOLVE-BLOCK-START\nx=1\n# EVOLVE-BLOCK-END\n```",
            },
        ]
        result: Tuple[int, str, int, int] = await ensemble.generate(messages)
        assert len(result) == 4
        model_id: int = result[0]
        response: str = result[1]
        assert model_id in {0, 1}
        assert isinstance(response, str)

    def test_configure_mocks(self):
        """Tests that configure_mocks updates all mock model markers."""
        ensemble: OpenAIEnsemble = self._make_mock_ensemble(2)
        ensemble.configure_mocks("// START", "// END")
        for model in ensemble.models:
            if isinstance(model, MockOpenAILM):
                assert model.start_marker == "// START"
                assert model.end_marker == "// END"
