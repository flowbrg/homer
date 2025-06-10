# src/core/config.py

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Annotated, Optional, Type, TypeVar

from src.resources import prompts

from langchain_core.runnables import RunnableConfig, ensure_config

import os

@dataclass(kw_only=True)
class Configuration:
    """Configuration class for indexing and retrieval operations.

    This class defines the parameters needed for configuring the indexing and
    retrieval processes.
    """

    embedding_model: str = field(
        default="nomic-embed-text",
        metadata={
            "description": "Name of the embedding model to use. Must be a valid embedding model name."
        },
    )

    response_system_prompt: str = field(
        default=prompts.RESPONSE_SYSTEM_PROMPT,
        metadata={"description": "The system prompt used for generating responses."},
    )

    response_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="qwen3:0.6b",
        metadata={
            "description": "The language model used for generating responses. Must be a valid LLM name."
        },
    )

    query_system_prompt: str = field(
        default=prompts.QUERY_SYSTEM_PROMPT,
        metadata={
            "description": "The system prompt used for processing and refining queries."
        },
    )

    query_model: str = field(
        default="qwen3:0.6b",
        metadata={
            "description": "The language model used for processing and refining queries. Must be a valid LLM name."
        },
    )

    ollama_host: str = field(
        default=os.getenv("OLLAMA_HOST"),
        metadata={
            "description": "The host URL for the Ollama service. Must be a valid URL. Default is https://127.0.0.1:11434/"
        },
    )

    def asdict(self) -> dict[str, any]:
        """Convert the instance to a dictionary.

        Args:
            cls (Type[T]): The class itself.
            instance (T): The instance of the class.

        Returns:
            dict[str, any]: A dictionary representation of the instance.
        """
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_runnable_config(
        cls: Type[T], config: Optional[RunnableConfig] = None
    ) -> T:
        """Create an IndexConfiguration instance from a RunnableConfig object.

        Args:
            cls (Type[T]): The class itself.
            config (Optional[RunnableConfig]): The configuration object to use.

        Returns:
            T: An instance of IndexConfiguration with the specified configuration.
        """
        config = ensure_config(config)
        configurable = config.get("configurable") or {}
        _fields = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in configurable.items() if k in _fields})


T = TypeVar("T", bound=Configuration)