"""Main entrypoint for the conversational retrieval graph.

This module defines the core structure and functionality of the conversational
retrieval graph. It includes the main graph definition, state management,
and key functions for processing user inputs, generating queries, retrieving
relevant documents, and formulating responses.
"""

from datetime import datetime, timezone
from typing import cast, TypedDict
from pydantic import BaseModel

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.sqlite import SqliteSaver

from src.core.agents import retrieval, InputState, State, StateReport
from src.core.configuration import Configuration
from src.resources.utils import format_docs, get_message_text, load_chat_model, load_embedding_model, get_connection
from src.resources import prompts


# Define the structured output
class OutlineEntry(BaseModel):
    """Represents a single entry in the outline with a title and summary."""

    title: str
    summary: str


class Outline(BaseModel):
    """Represents an outline for a report with a query and multiple entries."""

    entries: list[OutlineEntry]


def improve_query(
    state: StateReport, *, config: RunnableConfig
    ) -> dict[str, list[str]]:
    """Generate a search query based on the user input and configuration.

    This function analyzes the messages in the state and generates an appropriate
    """
    configuration = Configuration.from_runnable_config(config)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompts.IMPROVE_QUERY_SYSTEM_PROMPT),
            ("human", "{message}"),
        ]
    )

    model = load_chat_model(model=configuration.query_model)

    message_value = prompt.invoke(
        {
            "message": state.messages,
            "system_time": datetime.now(tz=timezone.utc).isoformat(),
        },
        config,
    )

    improved_query = model.invoke(message_value, config)
    return {"query": get_message_text(improved_query)}

def generate_outline(
    state: StateReport, *, config: RunnableConfig
    ) -> list[dict[str, str]]:
    """Generate an outline based on the latest query in the state.

    This function uses the latest query from the state to generate an outline
    for a report. It utilizes a language model to create a structured outline
    with specific sections and summaries.

    Args:
        state (State): The current state containing queries and messages.
        config (RunnableConfig | None, optional): Configuration for the outline generation process.

    Returns:
        list[dict[str, str]]:
    """
    configuration = Configuration.from_runnable_config(config)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", configuration.outline_system_prompt),
            ("human", "{message}"),
        ]
    )

    retrieved_docs = format_docs(state.retrieved_docs)

    model = load_chat_model(model = configuration.outline_model).with_structured_output(Outline)

    message_value = prompt.invoke(
        {
            "message": state.query,
            "number_of_parts": configuration.number_of_parts,
            "context": retrieved_docs,
            "system_time": datetime.now(tz=timezone.utc).isoformat(),
        },
        config,
    )

    generated_outlines = cast(Outline, model.invoke(message_value, config))
    return {
        "outlines": generated_outlines.model_dump()["entries"],  # Convert Outline to a list of dicts
    }


def retrieve(
    state: StateReport, *, config: RunnableConfig
) -> dict[str, list[Document]]:
    """Retrieve documents based on the latest query in the state.

    This function takes the current state and configuration, uses the latest query
    from the state to retrieve relevant documents using the retriever, and returns
    the retrieved documents.

    Args:
        state (State): The current state containing queries and the retriever.
        config (RunnableConfig | None, optional): Configuration for the retrieval process.

    Returns:
        dict[str, list[Document]]: relevant documents
    """
    configuration = Configuration.from_runnable_config(config)
    
    if not state.outlines: # If no outlines are available, it means the generate_outline step has not been called yet.
        input = state.query
    else:
        current_section = state.outlines[0]
        input = "\n".join([current_section["title"], current_section["summary"]])

    with retrieval.make_retriever(embedding_model = load_embedding_model(model=configuration.embedding_model)) as retriever:
        response = retriever.invoke(input, config)
        
        return {
            "retrieved_docs": response
        }


def generate_section(
    state: StateReport, *, config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
    """Call the LLM powering our "agent"."""
    configuration = Configuration.from_runnable_config(config)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompts.SECTION_SYSTEM_PROMPT),
            ("human", "{message}"),
        ]
    )
    model = load_chat_model(model = configuration.response_model)
    retrieved_docs = format_docs(state.retrieved_docs)
    current_section = state.outlines.pop(0) # Get the first section to generate
    message_value = prompt.invoke(
        {
            "message": state.messages,
            "title": state.report[-1]["title"] if state.report else "",
            "summary": state.report[-1]["summary"] if state.report else "",
            "previous_sections_text": "\n".join(section.get("content","") for section in state.report) if state.report else "This is the first section. There are no previous sections.",
            "context": retrieved_docs,
            "system_time": datetime.now(tz=timezone.utc).isoformat(),
        },
        config,
    )
    response = model.invoke(message_value, config)
    current_section["content"] = get_message_text(response)
    # We return a list, because this will get added to the existing list
    return {"report": [current_section]}


def should_continue(state: StateReport, *, config: RunnableConfig):
    if len(state.outlines) == 0:
        return END
    else:
        return "retrieve"


builder = StateGraph(StateReport, input=InputState, config_schema=Configuration)

builder.add_node(improve_query)
builder.add_node("initial_retrieval", retrieve)
builder.add_node(generate_outline)
builder.add_node(retrieve)
builder.add_node(generate_section)
builder.add_edge("__start__", "improve_query")
builder.add_edge("improve_query", "initial_retrieval")
builder.add_edge("initial_retrieval", "generate_outline")
builder.add_edge("generate_outline", "retrieve")
builder.add_edge("retrieve", "generate_section")
builder.add_conditional_edges("generate_section",should_continue)

# Finally, we compile it!
# This compiles it into a graph you can invoke and deploy.
graph = builder.compile(
    interrupt_before=[],  # if you want to update the state before calling the tools
    interrupt_after=[],
)
graph.name = "ReportGraph"

def get_report_graph() -> CompiledStateGraph:
    return graph