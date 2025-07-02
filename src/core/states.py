import uuid
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, Optional, Sequence, Union

from langchain_core.documents import Document
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages


def reduce_docs(
    existing: Optional[Sequence[Document]],
    new: Union[
        Sequence[Document],
        Sequence[dict[str, Any]],
        Sequence[str],
        str,
        Literal["delete"],
    ],
) -> Sequence[Document]:
    """Reduce and process documents based on the input type."""
    if new == "delete":
        return []
    if isinstance(new, str):
        return [Document(page_content=new, metadata={"id": str(uuid.uuid4())})]
    if isinstance(new, list):
        coerced = []
        for item in new:
            if isinstance(item, str):
                coerced.append(
                    Document(page_content=item, metadata={"id": str(uuid.uuid4())})
                )
            elif isinstance(item, dict):
                coerced.append(Document(**item))
            else:
                coerced.append(item)
        return coerced
    return existing or []


@dataclass(kw_only=True)
class InputIndexState:
    """Represents the input state for document indexing.

    This class defines the structure of the input index state, which includes
    the path of the documents to be indexed.
    """

    path: str
    """The path to the files."""


@dataclass(kw_only=True)
class IndexState(InputIndexState):
    """Represents the state for document indexing.

    Attrs:
        docs (Sequence[Document]): a sequence of documents using a reducer
        to handle parallel modifications
    """

    docs: Annotated[Sequence[Document], reduce_docs]
    """A list of documents that the agent can index."""


@dataclass(kw_only=True)
class InputState:
    """Represents the input state for the agent.

    This class defines the structure of the input state, which includes
    the messages exchanged between the user and the agent. It serves as
    a restricted version of the full State, providing a narrower interface
    to the outside world compared to what is maintained internally.
    """

    messages: Annotated[Sequence[AnyMessage], add_messages]
    """Messages track the primary execution state of the agent.

    Typically accumulates a pattern of Human/AI/Human/AI messages; if
    you were to combine this template with a tool-calling ReAct agent pattern,
    it may look like this:

    1. HumanMessage - user input
    2. AIMessage with .tool_calls - agent picking tool(s) to use to collect
         information
    3. ToolMessage(s) - the responses (or errors) from the executed tools
    
        (... repeat steps 2 and 3 as needed ...)
    4. AIMessage without .tool_calls - agent responding in unstructured
        format to the user.

    5. HumanMessage - user responds with the next conversational turn.

        (... repeat steps 2-5 as needed ... )
    
    Merges two lists of messages, updating existing messages by ID.

    By default, this ensures the state is "append-only", unless the
    new message has the same ID as an existing message.

    Returns:
        A new list of messages with the messages from `right` merged into `left`.
        If a message in `right` has the same ID as a message in `left`, the
        message from `right` will replace the message from `left`."""



@dataclass(kw_only=True)
class RetrievalState(InputState):
    """The state of your graph / agent."""

    query: str = field(default_factory=str)
    """A list of search queries that the agent has generated."""

    retrieved_docs: list[Document] = field(default_factory=list)
    """Populated by the retriever. This is a list of documents that the agent can reference."""

    summary: str = field(default_factory=str)
    """A summary of the retrieved documents or the conversation history."""


def add_sections(
    existing: Sequence[dict[str, str]], 
    new: Union[Sequence[dict[str, str]], dict[str, str]]
) -> Sequence[dict[str, str]]:
    """Combine existing sections with new sections."""
    existing_list = list(existing) if existing else []
    
    if isinstance(new, dict):
        return existing_list + [new]
    elif isinstance(new, list):
        return existing_list + new
    return existing_list


@dataclass(kw_only=True)
class ReportState(InputState):
    """The state of your report graph / agent."""

    outlines: list[dict[str, str]] = field(default_factory=list)
    """A list of sections that the agent has generated for the report."""

    report: Annotated[list[dict[str, str]], add_sections] = field(default_factory=list[dict[str, str]])
    """The final report as a list of sections, where each section is a dictionary with keys like 'title' and 'content'."""

    retrieved_docs: list[Document] = field(default_factory=list)
    """Populated by the retriever. This is a list of documents that the agent can reference."""

    current_section_index: int = 0
    """Track which section is being processed."""