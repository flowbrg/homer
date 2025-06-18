
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, Optional, Sequence, Union

from langchain_core.documents import Document
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages

# Optional, the InputState is a restricted version of the State that is used to
# define a narrower interface to the outside world vs. what is maintained
# internally.

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


# This is the primary state of your agent, where you can store any information


@dataclass(kw_only=True)
class State(InputState):
    """The state of your graph / agent."""

    enhanced_query: list[str] = field(default_factory=list)
    """A list of search queries that the agent has generated."""

    retrieved_docs: list[Document] = field(default_factory=list)
    """Populated by the retriever. This is a list of documents that the agent can reference."""

    # Feel free to add additional attributes to your state as needed.
    # Common examples include retrieved documents, extracted entities, API connections, etc.


def add_sections(
    existing: Sequence[dict[str, str]], new: Sequence[dict[str, str]]
) -> Sequence[dict[str, str]]:
    """Combine existing sections with new sections.

    Args:
        existing (Sequence[Dict[str, str]]): The current list of sections in the state.
        new (Sequence[Dict[str, str]]): The new sections to be added.

    Returns:
        Sequence[Dict[str, str]]: A new list containing all sections from both input sequences.
    """
    return list(existing) + list(new)


@dataclass(kw_only=True)
class StateReport(InputState):
    """The state of your report graph / agent."""

    query: str = field(default_factory=list)
    """An improved search query that the agent has generated."""

    outlines: list[dict[str, str]] = field(default_factory=list)
    """A list of sections that the agent has generated for the report."""

    report: Annotated[list[dict[str, str]], add_sections] = field(default_factory=list)
    """The final report as a list of sections, where each section is a dictionary with keys like 'title' and 'content'."""

    retrieved_docs: list[Document] = field(default_factory=list)
    """Populated by the retriever. This is a list of documents that the agent can reference."""

    # Feel free to add additional attributes to your state as needed.
    # Common examples include retrieved documents, extracted entities, API connections, etc.