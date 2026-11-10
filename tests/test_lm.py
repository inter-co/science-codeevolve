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

import asyncio
import json
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import httpx
import pytest
from openai import AsyncOpenAI

from codeevolve.lm.openai import (
    MockOpenAILM,
    OpenAIEmbedding,
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


# ---------------------------------------------------------------------------
# Offline provider fixtures for OpenAILM / OpenAIEmbedding
# ---------------------------------------------------------------------------

_MESSAGES: List[Dict[str, str]] = [{"role": "user", "content": "hello"}]


def _make_completion_body(
    content: Optional[str] = "ok",
    usage: Optional[Dict[str, int]] = None,
    include_usage_key: bool = True,
    num_choices: int = 1,
) -> Dict[str, Any]:
    """Builds a chat.completion JSON body as returned by an OpenAI-compatible API."""
    body: Dict[str, Any] = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "test-model",
        "choices": [
            {
                "index": i,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
            for i in range(num_choices)
        ],
    }
    if include_usage_key:
        body["usage"] = usage
    return body


def _make_embedding_body(
    num_vectors: int = 1,
    usage: Optional[Dict[str, int]] = None,
    include_usage_key: bool = True,
) -> Dict[str, Any]:
    """Builds an embeddings JSON body as returned by an OpenAI-compatible API."""
    body: Dict[str, Any] = {
        "object": "list",
        "model": "test-embedding",
        "data": [
            {"object": "embedding", "index": i, "embedding": [0.1 * (i + 1), 0.2]}
            for i in range(num_vectors)
        ],
    }
    if include_usage_key:
        body["usage"] = usage
    return body


_USAGE: Dict[str, int] = {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}


class _FakeProvider:
    """Serves a scripted sequence of HTTP responses and counts requests."""

    def __init__(self, responses: List[Tuple[int, Dict[str, Any]]]):
        self.responses: List[Tuple[int, Dict[str, Any]]] = responses
        self.calls: int = 0

    def handler(self, request: httpx.Request) -> httpx.Response:
        status, body = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return httpx.Response(status, content=json.dumps(body).encode())

    def client(self) -> AsyncOpenAI:
        # max_retries=0 disables the SDK's own retries so that only the
        # wrapper's retry loop is under test.
        return AsyncOpenAI(
            api_key="test_key",
            base_url="http://fake-provider.test/v1",
            max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(self.handler)),
        )


def _error_body(status: int) -> Dict[str, Any]:
    return {"error": {"message": f"simulated {status}", "type": "test", "code": None}}


@pytest.fixture
def no_backoff(monkeypatch):
    """Makes the wrapper's exponential backoff instantaneous."""

    async def _instant(_delay: float) -> None:
        return None

    monkeypatch.setattr("codeevolve.lm.openai.asyncio.sleep", _instant)


def _make_lm(provider: _FakeProvider, retries: int = 3) -> OpenAILM:
    lm: OpenAILM = OpenAILM(
        model_name="test-model", api_key="test_key", api_base="http://x", retries=retries
    )
    lm.client = provider.client()
    return lm


def _make_embedding(provider: _FakeProvider, retries: int = 3) -> OpenAIEmbedding:
    emb: OpenAIEmbedding = OpenAIEmbedding(
        model_name="test-embedding", api_key="test_key", api_base="http://x", retries=retries
    )
    emb.client = provider.client()
    return emb


class TestOpenAILMUsageMetadata:
    """Usage metadata is optional and must never cause a successful request to be replayed."""

    @pytest.mark.asyncio
    async def test_reported_usage_is_returned(self, no_backoff):
        provider = _FakeProvider([(200, _make_completion_body(usage=_USAGE))])
        result = await _make_lm(provider).generate(_MESSAGES)
        assert result == ("ok", 11, 7)
        assert provider.calls == 1

    @pytest.mark.asyncio
    async def test_absent_usage_key_is_not_retried(self, no_backoff):
        provider = _FakeProvider([(200, _make_completion_body(include_usage_key=False))])
        result = await _make_lm(provider).generate(_MESSAGES)
        assert result == ("ok", None, None)
        assert provider.calls == 1

    @pytest.mark.asyncio
    async def test_null_usage_is_not_retried(self, no_backoff):
        provider = _FakeProvider([(200, _make_completion_body(usage=None))])
        result = await _make_lm(provider).generate(_MESSAGES)
        assert result == ("ok", None, None)
        assert provider.calls == 1

    @pytest.mark.asyncio
    async def test_partial_usage(self, no_backoff):
        provider = _FakeProvider([(200, _make_completion_body(usage={"prompt_tokens": 5}))])
        result = await _make_lm(provider).generate(_MESSAGES)
        assert result == ("ok", 5, None)
        assert provider.calls == 1

    @pytest.mark.asyncio
    async def test_null_content_becomes_empty_string(self, no_backoff):
        provider = _FakeProvider([(200, _make_completion_body(content=None, usage=_USAGE))])
        result = await _make_lm(provider).generate(_MESSAGES)
        assert result == ("", 11, 7)
        assert provider.calls == 1


class TestOpenAILMRetries:
    """Retry behaviour for empty responses and provider errors."""

    @pytest.mark.asyncio
    async def test_empty_choices_retried_until_exhausted(self, no_backoff):
        provider = _FakeProvider([(200, _make_completion_body(num_choices=0, usage=_USAGE))])
        with pytest.raises(ConnectionError, match="no choices"):
            await _make_lm(provider, retries=2).generate(_MESSAGES)
        assert provider.calls == 3

    @pytest.mark.asyncio
    async def test_empty_choices_then_success(self, no_backoff):
        provider = _FakeProvider(
            [
                (200, _make_completion_body(num_choices=0)),
                (200, _make_completion_body(usage=_USAGE)),
            ]
        )
        result = await _make_lm(provider).generate(_MESSAGES)
        assert result == ("ok", 11, 7)
        assert provider.calls == 2

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", [429, 500, 503])
    async def test_transient_errors_are_retried(self, no_backoff, status):
        provider = _FakeProvider(
            [(status, _error_body(status)), (200, _make_completion_body(usage=_USAGE))]
        )
        result = await _make_lm(provider).generate(_MESSAGES)
        assert result == ("ok", 11, 7)
        assert provider.calls == 2

    @pytest.mark.asyncio
    async def test_persistent_transient_error_exhausts_retries(self, no_backoff):
        provider = _FakeProvider([(500, _error_body(500))])
        with pytest.raises(ConnectionError, match="after 4 attempts"):
            await _make_lm(provider, retries=3).generate(_MESSAGES)
        assert provider.calls == 4

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
    async def test_non_retryable_errors_fail_fast(self, no_backoff, status):
        provider = _FakeProvider([(status, _error_body(status))])
        with pytest.raises(ConnectionError, match="non-retryable"):
            await _make_lm(provider, retries=3).generate(_MESSAGES)
        assert provider.calls == 1

    @pytest.mark.asyncio
    async def test_backoff_is_exponential(self, monkeypatch):
        delays: List[float] = []

        async def _record(delay: float) -> None:
            delays.append(delay)

        monkeypatch.setattr("codeevolve.lm.openai.asyncio.sleep", _record)
        provider = _FakeProvider([(500, _error_body(500))])
        with pytest.raises(ConnectionError):
            await _make_lm(provider, retries=3).generate(_MESSAGES)
        assert delays == [1, 2, 4]


class TestOpenAIEnsembleUsageWarning:
    """The ensemble surfaces missing usage metadata through its logger."""

    def _make_ensemble(self, provider: _FakeProvider, logger: logging.Logger) -> OpenAIEnsemble:
        ensemble = OpenAIEnsemble(
            models_cfg=[{"model_name": "test-model", "retries": 0}],
            api_key="test_key",
            api_base="http://x",
            logger=logger,
        )
        ensemble.models[0].client = provider.client()
        return ensemble

    @pytest.mark.asyncio
    async def test_missing_usage_logs_warning(self, caplog):
        logger = logging.getLogger("test_lm.ensemble.missing")
        provider = _FakeProvider([(200, _make_completion_body(include_usage_key=False))])
        with caplog.at_level(logging.WARNING, logger=logger.name):
            result = await self._make_ensemble(provider, logger).generate(_MESSAGES)
        assert result == (0, "ok", None, None)
        assert provider.calls == 1
        assert any("did not report full token usage" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_reported_usage_logs_no_warning(self, caplog):
        logger = logging.getLogger("test_lm.ensemble.reported")
        provider = _FakeProvider([(200, _make_completion_body(usage=_USAGE))])
        with caplog.at_level(logging.WARNING, logger=logger.name):
            result = await self._make_ensemble(provider, logger).generate(_MESSAGES)
        assert result == (0, "ok", 11, 7)
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


class TestOpenAIEmbedding:
    """Usage handling and retries for the embedding wrapper."""

    @pytest.mark.asyncio
    async def test_single_text_with_usage(self, no_backoff):
        provider = _FakeProvider(
            [(200, _make_embedding_body(usage={"prompt_tokens": 3, "total_tokens": 3}))]
        )
        embedding, tokens = await _make_embedding(provider).embed("hello")
        assert embedding == [0.1, 0.2]
        assert tokens == 3
        assert provider.calls == 1

    @pytest.mark.asyncio
    async def test_missing_usage_is_not_retried(self, no_backoff):
        provider = _FakeProvider([(200, _make_embedding_body(include_usage_key=False))])
        embedding, tokens = await _make_embedding(provider).embed("hello")
        assert embedding == [0.1, 0.2]
        assert tokens is None
        assert provider.calls == 1

    @pytest.mark.asyncio
    async def test_batch_returns_all_vectors(self, no_backoff):
        provider = _FakeProvider([(200, _make_embedding_body(num_vectors=2, usage=None))])
        embeddings, tokens = await _make_embedding(provider).embed_batch(["a", "b"])
        assert len(embeddings) == 2
        assert tokens is None
        assert provider.calls == 1

    @pytest.mark.asyncio
    async def test_wrong_vector_count_is_retried(self, no_backoff):
        provider = _FakeProvider(
            [
                (200, _make_embedding_body(num_vectors=1)),
                (200, _make_embedding_body(num_vectors=2)),
            ]
        )
        embeddings, _ = await _make_embedding(provider).embed_batch(["a", "b"])
        assert len(embeddings) == 2
        assert provider.calls == 2

    @pytest.mark.asyncio
    async def test_non_retryable_error_fails_fast(self, no_backoff):
        provider = _FakeProvider([(401, _error_body(401))])
        with pytest.raises(ConnectionError, match="non-retryable"):
            await _make_embedding(provider).embed("hello")
        assert provider.calls == 1
