"""Manage the configuration of various retrievers.

This module provides functionality to create and manage retrievers for different
vector store backends, specifically Elasticsearch, Pinecone, and MongoDB.

The retrievers support filtering results by user_id to ensure data isolation between users.
"""

import os
from contextlib import contextmanager
from typing import Generator
from pathlib import Path

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever

## Encoder constructors


from langchain.embeddings.base import Embeddings

def make_text_encoder(model: str) -> Embeddings:   
    from langchain_ollama import OllamaEmbeddings
    return OllamaEmbeddings(model=model)

## Retriever constructors


@contextmanager
def make_retriever(
    embedding_model: Embeddings
) -> Generator[VectorStoreRetriever, None, None]:
    """Create a retriever for the agent, based on the current configuration."""
    import faiss
    from langchain_community.docstore.in_memory import InMemoryDocstore
    from langchain_community.vectorstores import FAISS

    vstore_path = Path(os.getenv("VECTORSTORE_PATH"))
    index = faiss.IndexHNSWFlat(len(embedding_model.embed_query("hello world")), 32)

    if not vstore_path.is_dir():
        vstore = FAISS(
            embedding_function = embedding_model,
            index=index,
            docstore = InMemoryDocstore(),
            index_to_docstore_id={},
        )
        vstore.save_local(vstore_path)
    
    vstore = FAISS.load_local(
        vstore_path,
        embedding_model,
        allow_dangerous_deserialization=True
    )

    yield vstore.as_retriever()
