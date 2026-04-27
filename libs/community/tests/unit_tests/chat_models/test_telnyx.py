"""Test Telnyx chat model."""

from pydantic import SecretStr

from langchain_community.chat_models import ChatTelnyx


def test_telnyx_model_name_param() -> None:
    llm = ChatTelnyx(model_name="foo", telnyx_api_key="test")  # type: ignore[arg-type]
    assert llm.model_name == "foo"


def test_telnyx_model_param() -> None:
    llm = ChatTelnyx(model="foo", telnyx_api_key="test")  # type: ignore[arg-type]
    assert llm.model_name == "foo"


def test_telnyx_api_key_is_secret_string() -> None:
    llm = ChatTelnyx(telnyx_api_key="secret-api-key")  # type: ignore[arg-type]
    assert isinstance(llm.telnyx_api_key, SecretStr)


def test_telnyx_api_base_default() -> None:
    llm = ChatTelnyx(telnyx_api_key="test")  # type: ignore[arg-type]
    assert llm.telnyx_api_base == "https://api.telnyx.com/v2/ai/openai"


def test_telnyx_api_base_custom() -> None:
    llm = ChatTelnyx(telnyx_api_key="test", telnyx_api_base="https://custom.api")  # type: ignore[arg-type]
    assert llm.telnyx_api_base == "https://custom.api"


def test_telnyx_tiktoken_disabled() -> None:
    llm = ChatTelnyx(telnyx_api_key="test")  # type: ignore[arg-type]
    assert llm.tiktoken_enabled is False
