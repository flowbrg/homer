from src.env import *

######################################## connect to database ########################################

import sqlite3

from contextlib import contextmanager

def get_connection() -> sqlite3.Connection:
    return sqlite3.connect(':memory:', check_same_thread=False)

import aiosqlite

def aget_connection() -> aiosqlite.Connection:
    """Asynchronous version of get_connection."""
    return aiosqlite.connect(PERSISTENT_DIR)

######################################## connect to database ########################################

from chromadb import PersistentClient

def get_chroma_client() -> PersistentClient:
    return PersistentClient(path=VECTORSTORE_DIR)

######################################## format documents ########################################

from langchain_core.documents import Document
from typing import Optional

def _format_doc(doc: Document) -> str:
    """Format a single document as XML.

    Args:
        doc (Document): The document to format.

    Returns:
        str: The formatted document as an XML string.
    """
    metadata = doc.metadata or {}
    meta = "".join(f" {k}={v!r}" for k, v in metadata.items())
    if meta:
        meta = f" {meta}"

    return f"<document{meta}>\n{doc.page_content}\n</document>"

def format_docs(docs: Optional[list[Document]]) -> str:
    """Format a list of documents as XML.

    This function takes a list of Document objects and formats them into a single XML string.

    Args:
        docs (Optional[list[Document]]): A list of Document objects to format, or None.

    Returns:
        str: A string containing the formatted documents in XML format.

    Examples:
        >>> docs = [Document(page_content="Hello"), Document(page_content="World")]
        >>> print(format_docs(docs))
        <documents>
        <document>
        Hello
        </document>
        <document>
        World
        </document>
        </documents>

        >>> print(format_docs(None))
        <documents></documents>
    """
    if not docs:
        return "<documents></documents>"
    formatted = "\n".join(_format_doc(doc) for doc in docs)
    return f"""<documents>
{formatted}
</documents>"""

######################################## format messages ########################################

import re

from langchain_core.messages import AnyMessage, AIMessage
from langchain_core.messages.human import HumanMessage

def _format_message(message: AnyMessage) -> str:
    text = re.sub(r'<think>.*?</think>', '', message.content, flags=re.DOTALL)
    if isinstance(message, HumanMessage):
        flag = "HumanMessage"
    if isinstance(message, AIMessage):
        flag = "AIMessage"
    else:
        flag = "message"
    return f"<{flag}>\n{text}\n</{flag}>"

def format_messages(messages: Optional[list[AnyMessage]])-> str:
    if not messages:
        return "<messages></messages>"
    formatted = "\n".join(_format_message(message) for message in messages)
    return f"""<messages>
{formatted}
<messages>"""

######################################## format sources ########################################

from pathlib import Path

def format_sources_markdown(documents: Optional[list[Document]])-> str:
    """
    Convert a list of documents to a markdown list of unique sources.
    
    Args:
        documents: List of document objects with metadata attribute
        
    Returns:
        str: Markdown formatted list of unique sources, sorted alphabetically
    """
    if not documents:
        return "No sources available."
    
    # Extract unique sources
    sources = set()
    for document in documents:
        source = document.metadata.get("source", "unknown")
        sources.add(source)
    
    # Sort and format as markdown
    sorted_sources = sorted(sources)
    markdown_lines = [f"- {source}" for source in sorted_sources]
    
    return "\n".join(markdown_lines)

def format_sources(documents: Optional[list[Document]])-> str:
    """
    Convert a list of documents to a string of unique sources.
    
    Args:
        documents: List of document objects with metadata attribute
        
    Returns:
        str: formatted list of unique sources, sorted alphabetically
    """
    if not documents:
        return "No sources available."
    
    # Extract unique sources
    sources = set()
    for document in documents:
        source = document.metadata.get("source", "unknown")
        sources.add(Path(source).stem)
    
    # Sort and format as markdown
    sorted_sources = sorted(sources)
    sources_lines = [f"- {source}" for source in sorted_sources]
    
    return "\n".join(sources_lines)

######################################## Structured messages ########################################

from langchain_core.messages import AnyMessage

def get_message_text(msg: AnyMessage) -> str:
    """Get the text content of a message.

    This function extracts the text content from various message formats.

    Args:
        msg (AnyMessage): The message object to extract text from.

    Returns:
        str: The extracted text content of the message.

    Examples:
        >>> from langchain_core.messages import HumanMessage
        >>> get_message_text(HumanMessage(content="Hello"))
        'Hello'
        >>> get_message_text(HumanMessage(content={"text": "World"}))
        'World'
        >>> get_message_text(HumanMessage(content=[{"text": "Hello"}, " ", {"text": "World"}]))
        'Hello World'
    """
    content = msg.content
    if isinstance(content, str):
        return content
    elif isinstance(content, dict):
        return content.get("text", "")
    else:
        txts = [c if isinstance(c, str) else (c.get("text") or "") for c in content]
        return "".join(txts).strip()
    
######################################## Remove duplicates ########################################

def remove_duplicates(base: list[str], new: list[str]) -> list[str]:
    base_set = set(base)
    return [item for item in new if item not in base_set]