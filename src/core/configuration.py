# src/core/config.py

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Optional, Type, TypeVar

from src.resources import prompts
from src.env import OLLAMA_CLIENT

from langchain_core.runnables import RunnableConfig, ensure_config

import os
import requests

@dataclass(kw_only=True)
class Configuration:
    """Configuration class for indexing and retrieval operations.

    This class defines the parameters needed for configuring the indexing and
    retrieval processes.
    """

    # Report configuration
    number_of_parts: int = field(
        default = 5,
        metadata = {
            "description": "The number of parts in the report outline. Must be a positive integer."
        },
    )

    # Ollama configuration
    ollama_host: str = field(
        default = "http://127.0.0.1:11434/",
        metadata = {
            "description": "The host URL for the Ollama service. Must be a valid URL. Default is http://127.0.0.1:11434/"
        },
    )
    
    # Models
    embedding_model: str = field(
        default = "nomic-embed-text",
    )

    response_model: str = field(
        default = "gemma3:1b",   #"qwen3:0.6b",
    )

    query_model: str = field(
        default = "gemma3:1b",   #"qwen3:0.6b",
    )

    outline_model: str = field(
        default = "gemma3:1b",   #"qwen3:0.6b",
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


def _is_ollama_client_available(url: str) -> bool:
    import requests
    try:
        response = requests.get(url, timeout=2)
        return response.ok
    except requests.RequestException:
        return False

def load_config(cls: Optional[Type[T]] = Configuration) -> T:
    config = cls()
    if _is_ollama_client_available(OLLAMA_CLIENT):
        print(f"[info] {OLLAMA_CLIENT} available")
        config.ollama_host = OLLAMA_CLIENT
    return config