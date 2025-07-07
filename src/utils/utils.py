from src.constant import *

######################################## connect to database ########################################

import sqlite3

from contextlib import contextmanager

def get_connection() -> sqlite3.Connection:
    return sqlite3.connect(':memory:', check_same_thread=False)

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

from langchain_core.messages import AIMessage
from langchain_core.messages.human import HumanMessage


def ya_format_messages(messages=list[AnyMessage]):
    "yet another format messages function"
    
    if not messages:
        return []
    
    formated_list = []
    for m in messages:
        if isinstance(m, HumanMessage):
            formated_list.append(("human", m.content))
        elif isinstance(m, AIMessage):
            formated_list.append(("ai", m.content))
        else:
            formated_list.append(("system", m.content)) 
    return formated_list


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

######################################## Make document batch ########################################

from itertools import islice
from typing import Iterator, List

def make_document_batch(documents: List[Document], size: int = 100) -> List[List[Document]]:
    """
    Split a list of documents into batches of specified size.
    
    Args:
        documents: List of documents to batch
        size: Maximum size of each batch (default: 100)
        
    Returns:
        List of document batches
        
    Raises:
        ValueError: If size is less than 1
    """
    if size < 1:
        raise ValueError("Batch size must be at least 1")
    
    if not documents:
        return []
    
    # Convert to iterator for memory efficiency
    doc_iter = iter(documents)
    
    # Use islice to create batches efficiently
    batches = []
    while True:
        batch = list(islice(doc_iter, size))
        if not batch:
            break
        batches.append(batch)
    
    return batches


# Alternative generator version for memory efficiency with large datasets
def make_document_batch_generator(documents: List[Document], size: int = 100) -> Iterator[List[Document]]:
    """
    Generator version that yields batches one at a time.
    Memory efficient for very large document lists.
    """
    if size < 1:
        raise ValueError("Batch size must be at least 1")
    
    doc_iter = iter(documents)
    while True:
        batch = list(islice(doc_iter, size))
        if not batch:
            break
        yield batch


# One-liner alternative using list comprehension (less readable but very concise)
def make_document_batch_oneliner(documents: List[Document], size: int = 100) -> List[List[Document]]:
    """Concise one-liner version."""
    return [documents[i:i + size] for i in range(0, len(documents), size)]

######################################## Streamlit connection button state ########################################

from streamlit.runtime.state.session_state_proxy import SessionStateProxy
from src.constant import OLLAMA_CLIENT

def is_connected(session_state: SessionStateProxy) -> bool:
    if "baseConfig" not in session_state:
        raise Exception("config not loaded in the session state")
    elif session_state.baseConfig.ollama_host == OLLAMA_CLIENT:
        return True
    return False

######################################## Clean thinking part ########################################

import re

def extract_think_and_answer(text: str) -> tuple[Optional[str], str]:
    """
    Separates a string into two parts: the content within <think>...</think> tags
    and the remaining text (answer part).

    Args:
        text: The input string potentially containing <think>...</think> blocks.

    Returns:
        A tuple containing (thinking_part, answer_part).
        If no <think> tags are found, thinking_part will be an empty string.
    """
    think_match = re.search(r'<think>(.*?)</think>', text, flags=re.DOTALL | re.IGNORECASE)

    thinking_part = ""
    answer_part = text

    if think_match:
        thinking_part = think_match.group(1).strip()
        answer_part = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE).strip()
        return thinking_part, answer_part
    return None, text

######################################## Ensure path exists ########################################

def ensure_path(path_str: str):
    """Crée le répertoire parent si le chemin est un fichier, ou le répertoire lui-même."""
    from pathlib import Path
    path = Path(path_str)
    
    # Si le chemin se termine par '/' ou n'a pas d'extension, c'est un dossier
    if path_str.endswith('/') or not path.suffix:
        path.mkdir(parents=True, exist_ok=True)
    else:
        # C'est un fichier, créer le répertoire parent
        path.parent.mkdir(parents=True, exist_ok=True)

######################################## Combine system and user prompt ########################################

def combine_prompts(
    system: Optional[str],
    user: str,
) -> str:
    if not system:
        system=""
    return f"SYSTEM:\n {system}\n"+f"USER:\n {user}"