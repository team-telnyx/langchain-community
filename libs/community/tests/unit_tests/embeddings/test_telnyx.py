"""Test Telnyx embeddings."""

import os
from typing import cast

import pytest
from pydantic import SecretStr

from langchain_community.embeddings import TelnyxEmbeddings


def test_telnyx_initialization() -> None:
    embeddings = TelnyxEmbeddings(telnyx_api_key="test")  # type: ignore[arg-type]
    assert cast(SecretStr, embeddings.telnyx_api_key).get_secret_value() == "test"


def test_telnyx_api_key_is_secret_string() -> None:
    embeddings = TelnyxEmbeddings(telnyx_api_key="secret-api-key")  # type: ignore[arg-type]
    assert isinstance(embeddings.telnyx_api_key, SecretStr)


def test_telnyx_api_base_default() -> None:
    embeddings = TelnyxEmbeddings(telnyx_api_key="test")  # type: ignore[arg-type]
    assert embeddings.telnyx_api_base == "https://api.telnyx.com/v2/ai/openai"


def test_telnyx_api_base_custom() -> None:
    embeddings = TelnyxEmbeddings(
        telnyx_api_key="test",  # type: ignore[arg-type]
        telnyx_api_base="https://custom.api",
    )
    assert embeddings.telnyx_api_base == "https://custom.api"


def test_telnyx_model_default() -> None:
    embeddings = TelnyxEmbeddings(telnyx_api_key="test")  # type: ignore[arg-type]
    assert embeddings.model == "thenlper/gte-large"


def test_telnyx_model_custom() -> None:
    embeddings = TelnyxEmbeddings(
        telnyx_api_key="test",  # type: ignore[arg-type]
        model="custom-model",
    )
    assert embeddings.model == "custom-model"


def test_telnyx_tiktoken_disabled() -> None:
    embeddings = TelnyxEmbeddings(telnyx_api_key="test")  # type: ignore[arg-type]
    assert embeddings.tiktoken_enabled is False


def test_telnyx_initialization_from_env() -> None:
    os.environ["TELNYX_API_KEY"] = "env-api-key"
    embeddings = TelnyxEmbeddings()
    assert cast(SecretStr, embeddings.telnyx_api_key).get_secret_value() == "env-api-key"
    del os.environ["TELNYX_API_KEY"]


def test_telnyx_initialization_missing_api_key() -> None:
    if "TELNYX_API_KEY" in os.environ:
        del os.environ["TELNYX_API_KEY"]
    with pytest.raises(ValueError):
        TelnyxEmbeddings()
