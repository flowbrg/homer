"""Manage the configuration of various retrievers.

This module provides functionality to create and manage retrievers for different
vector store backends, specifically Elasticsearch, Pinecone, and MongoDB.

The retrievers support filtering results by user_id to ensure data isolation between users.
"""

from contextlib import contextmanager
from typing import Generator

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever

from src.resources.utils import VECTORSTORE_PATH


_COLLECTION = "HOMER"

@contextmanager
def make_retriever(
    embedding_model: Embeddings
) -> Generator[VectorStoreRetriever, None, None]:
    """Create a retriever for the agent, based on the current configuration."""
    from langchain_chroma import Chroma

    vector_store = Chroma(
        collection_name = _COLLECTION,
        embedding_function = embedding_model,
        persist_directory = VECTORSTORE_PATH,  # Where to save data locally, remove if not necessary
    )

    yield vector_store.as_retriever()


def get_existing_documents():
    return "not implemented yet"


def delete_documents(docs: str | list[str]):
    return "not implemented yet"