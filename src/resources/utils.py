# src/resources/utils.py

import os
import yaml



######################################## environment variables ########################################

OLLAMA_CLIENT= "http://127.0.0.1:11434/"
PERSISTENT_DATA = "./user_data/threads/threads.db"
VECTORSTORE_PATH= "./user_data/vectorstore/"

######################################## connect to database ########################################

import sqlite3

def get_connection() -> sqlite3.Connection:
    return sqlite3.connect(PERSISTENT_DATA, check_same_thread=False)

import aiosqlite

def aget_connection() -> aiosqlite.Connection:
    """Asynchronous version of get_connection."""
    return aiosqlite.connect(PERSISTENT_DATA)

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

######################################## Embedding model ########################################

from langchain.embeddings.base import Embeddings

#def load_embeddings(model_name: str)-> Embeddings:
#    from langchain_huggingface import HuggingFaceEmbeddings
#    return HuggingFaceEmbeddings(model_name="nomic-ai/nomic-embed-text-v2-moe",cache_folder="./cache", model_kwargs={"trust_remote_code":True})

def load_embedding_model(model: str, host: str = "http://localhost:11434") -> Embeddings:   
    from langchain_ollama import OllamaEmbeddings
    return OllamaEmbeddings(model=model,base_url=host)

######################################## Chat model ########################################

from langchain.chat_models.base import BaseChatModel

def load_chat_model(model: str, host: str = "http://localhost:11434")-> BaseChatModel:
    from langchain_ollama import ChatOllama
    return ChatOllama(model=model,base_url=host)


#def load_llm(model_name: str)-> BaseChatModel:
#    from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
#    from transformers.models.auto.tokenization_auto import AutoTokenizer

    # load the tokenizer and the model
#    tokenizer = AutoTokenizer.from_pretrained(model_name,cache_dir="./cache")


#    llm = HuggingFacePipeline.from_model_id(
#        model_id=model_name,
#        task="text-generation",
#        pipeline_kwargs=dict(
#            max_new_tokens=512,
#            do_sample=False,
#            repetition_penalty=1.03,
#        ),
#        model_kwargs=dict(
#            # https://huggingface.co/docs/transformers/main_classes/model
#            torch_dtype="auto",
#            device_map="cpu",
#            cache_dir="./cache",
#        )
#    )

#    return ChatHuggingFace(
#       llm=llm,
#        tokenizer=tokenizer,
#    )

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

######################################## Manage base yaml config ########################################

from src.core.configuration import Configuration, T  # Your dataclass config
from typing import Type

def load_config(cls: Type[T] = Configuration) -> T:
    """Load configuration from a YAML file into a dataclass instance."""
    config = cls()
    return config