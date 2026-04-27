"""Telnyx Inference embeddings wrapper."""

from __future__ import annotations

from typing import Dict, Optional

from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env, pre_init
from pydantic import Field, SecretStr

from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.utils.openai import is_openai_v1

DEFAULT_API_BASE = "https://api.telnyx.com/v2/ai/openai"
DEFAULT_MODEL = "thenlper/gte-large"


class TelnyxEmbeddings(OpenAIEmbeddings):
    """`Telnyx` Embeddings API.

    See https://telnyx.com for information about Telnyx.

    To use, you should have the ``openai`` python package installed, and the
    environment variable ``TELNYX_API_KEY`` set with your API key.
    Alternatively, you can use the telnyx_api_key keyword argument.

    Example:
        .. code-block:: python

            from langchain_community.embeddings import TelnyxEmbeddings
            embeddings = TelnyxEmbeddings(model="thenlper/gte-large")
    """

    telnyx_api_key: Optional[SecretStr] = Field(default=None)
    """Telnyx API key."""
    model: str = Field(default=DEFAULT_MODEL)
    """Model name to use."""
    telnyx_api_base: str = Field(default=DEFAULT_API_BASE)
    """Base URL path for API requests."""
    tiktoken_enabled: bool = False
    """Set this to False for non-OpenAI implementations of the embeddings API"""
    embedding_ctx_length: int = 500
    """The maximum number of tokens to embed at once."""

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {
            "telnyx_api_key": "TELNYX_API_KEY",
        }

    @pre_init
    def validate_environment(cls, values: dict) -> dict:
        """Validate that api key and python package exists in environment."""
        values["telnyx_api_key"] = convert_to_secret_str(
            get_from_dict_or_env(
                values,
                "telnyx_api_key",
                "TELNYX_API_KEY",
            )
        )
        values["telnyx_api_base"] = get_from_dict_or_env(
            values,
            "telnyx_api_base",
            "TELNYX_API_BASE",
            default=DEFAULT_API_BASE,
        )
        try:
            import openai

        except ImportError:
            raise ImportError(
                "Could not import openai python package. "
                "Please install it with `pip install openai`."
            )
        if is_openai_v1():
            client_params = {
                "api_key": values["telnyx_api_key"].get_secret_value(),
                "base_url": values["telnyx_api_base"],
            }
            values["client"] = openai.OpenAI(**client_params).embeddings
        else:
            values["openai_api_base"] = values["telnyx_api_base"]
            values["openai_api_key"] = values["telnyx_api_key"].get_secret_value()
            values["client"] = openai.Embedding
        return values

    @property
    def _llm_type(self) -> str:
        return "telnyx-embedding"
