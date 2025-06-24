"""Manage the configuration of various retrievers.

This module provides functionality to create and manage retrievers for different
vector store backends, specifically Elasticsearch, Pinecone, and MongoDB.

The retrievers support filtering results by user_id to ensure data isolation between users.
"""

from contextlib import contextmanager
from typing import Generator

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever

from src.resources.utils import VECTORSTORE_DIR, get_chroma_client


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
        persist_directory = VECTORSTORE_DIR,  # Where to save data locally, remove if not necessary
    )

    yield vector_store.as_retriever()


def get_existing_documents() -> list[str]:
    """
    Get all unique document sources from the ChromaDB collection.
    
    Returns:
        list[str]: List of unique source file names
    """
    client=get_chroma_client()
    collection = client.get_or_create_collection(_COLLECTION)
    
    # Get all documents with their metadata
    results = collection.get(include=["metadatas"])
    
    # Extract unique sources from metadata
    sources = set()
    for metadata in results["metadatas"]:
        if metadata and "source" in metadata.keys():
            sources.add(metadata["source"])
    
    del client
    return list(sources)


def delete_documents(docs: str | list[str]):
    """
    Delete documents by source file name(s).
    
    Args:
        docs: Single document source name or list of source names to delete
    """
    client = get_chroma_client()
    collection = client.get_or_create_collection(_COLLECTION)
    
    # Convert single string to list for uniform processing
    if isinstance(docs, str):
        docs = [docs]
    
    for doc_source in docs:
        # Delete all documents with the specified source
        collection.delete(
            where={"source": doc_source}
        )
    del client