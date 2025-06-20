#DEPRECATED, use src/core/configuration.py instead

from pydantic import BaseModel

class BaseConfiguration(BaseModel):
    embedding_model: str = "nomic-embed-text"
    query_model: str = "qwen3:0.6b"
    response_model: str = "qwen3:0.6b"
    ollama_host: str = "http://localhost:11434"



    
