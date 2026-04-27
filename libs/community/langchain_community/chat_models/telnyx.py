"""Telnyx Inference chat wrapper. Relies heavily on ChatOpenAI."""

from __future__ import annotations

import logging
from typing import Any, Dict

from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env
from pydantic import Field, SecretStr, model_validator

from langchain_community.chat_models.openai import ChatOpenAI
from langchain_community.utils.openai import is_openai_v1

logger = logging.getLogger(__name__)

DEFAULT_API_BASE = "https://api.telnyx.com/v2/ai/openai"
DEFAULT_MODEL = "moonshotai/Kimi-K2.6"


class ChatTelnyx(ChatOpenAI):
    """`Telnyx` Chat large language models.

    See https://telnyx.com for information about Telnyx.

    To use, you should have the ``openai`` python package installed, and the
    environment variable ``TELNYX_API_KEY`` set with your API key.
    Alternatively, you can use the telnyx_api_key keyword argument.

    Any parameters that are valid to be passed to the `openai.create` call can be passed
    in, even if not explicitly saved on this class.

    Example:
        .. code-block:: python

            from langchain_community.chat_models import ChatTelnyx
            chat = ChatTelnyx(model="moonshotai/Kimi-K2.6")
    """

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "telnyx-chat"

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"telnyx_api_key": "TELNYX_API_KEY"}

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return False

    telnyx_api_key: SecretStr = Field(default=SecretStr(""))
    """Telnyx API key."""
    model_name: str = Field(default=DEFAULT_MODEL, alias="model")
    """Model name to use."""
    telnyx_api_base: str = Field(default=DEFAULT_API_BASE)
    """Base URL path for API requests."""
    tiktoken_enabled: bool = False
    """Set this to False for non-OpenAI implementations."""

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: dict) -> Any:
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

        except ImportError as e:
            raise ImportError(
                "Could not import openai python package. "
                "Please install it with `pip install openai`.",
            ) from e
        try:
            if is_openai_v1():
                client_params = {
                    "api_key": values["telnyx_api_key"].get_secret_value(),
                    "base_url": values["telnyx_api_base"],
                }
                if not values.get("client"):
                    values["client"] = openai.OpenAI(**client_params).chat.completions
                if not values.get("async_client"):
                    values["async_client"] = openai.AsyncOpenAI(
                        **client_params
                    ).chat.completions
            else:
                values["openai_api_base"] = values["telnyx_api_base"]
                values["openai_api_key"] = values[
                    "telnyx_api_key"
                ].get_secret_value()
                values["client"] = openai.ChatCompletion
        except AttributeError as exc:
            raise ValueError(
                "`openai` has no `ChatCompletion` attribute, this is likely "
                "due to an old version of the openai package. Try upgrading it "
                "with `pip install --upgrade openai`.",
            ) from exc

        if "model_name" not in values.keys():
            values["model_name"] = DEFAULT_MODEL

        return values
