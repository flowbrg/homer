"""Main entrypoint for the conversational retrieval graph.

This module defines the core structure and functionality of the conversational
retrieval graph. It includes the main graph definition, state management,
and key functions for processing user inputs, generating queries, retrieving
relevant documents, and formulating responses.
"""

from datetime import datetime, timezone
from typing import cast
from pydantic import BaseModel

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig

from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.sqlite import SqliteSaver

from src.core.agents import retrieval
from src.core.agents.states import InputState, State
from src.core.configuration import Configuration
from src.resources.utils import format_docs, format_messages, get_connection
from src.core.models import load_chat_model, load_embedding_model

# Define the function that calls the model

class SearchQuery(BaseModel):
    """Search the indexed documents for a query."""

    query: str


def generate_query(
    state: State, *, config: RunnableConfig
) -> dict[str, list[str]]:
    """Generate a search query based on the current state and configuration.

    This function analyzes the messages in the state and generates an appropriate
    search query. For the first message, it uses the user's input directly.
    For subsequent messages, it uses a language model to generate a refined query.

    Args:
        state (State): The current state containing messages and other information.
        config (RunnableConfig | None, optional): Configuration for the query generation process.

    Returns:
        dict[str, list[str]]: A dictionary with a 'queries' key containing a list of generated queries.

    Behavior:
        - If there's only one message (first user input), it uses that as the query.
        - For subsequent messages, it uses a language model to generate a refined query.
        - The function uses the configuration to set up the prompt and model for query generation.
    """
    
    configuration = Configuration.from_runnable_config(config)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", configuration.query_system_prompt),
            ("human", "{message}"),
        ]
    )

    model = load_chat_model(model = configuration.query_model).with_structured_output(
        SearchQuery
    )

    message_value = prompt.invoke(
        {
            "message": state.messages[-1],
            "system_time": datetime.now(tz=timezone.utc).isoformat(),
        },
        config,
    )

    generated = cast(SearchQuery, model.invoke(message_value, config))
    
    return {
        "enhanced_query": generated.query,
        "retrieved_docs": [],
    }


def retrieve(
    state: State, *, config: RunnableConfig
) -> dict[str, list[Document]]:
    """Retrieve documents based on the latest query in the state.

    This function takes the current state and configuration, uses the latest query
    from the state to retrieve relevant documents using the retriever, and returns
    the retrieved documents.

    Args:
        state (State): The current state containing queries and the retriever.
        config (RunnableConfig | None, optional): Configuration for the retrieval process.

    Returns:
        dict[str, list[Document]]: A dictionary with a single key "retrieved_docs"
        containing a list of retrieved Document objects.
    """
    configuration = Configuration.from_runnable_config(config)
    with retrieval.make_retriever(embedding_model = load_embedding_model(model=configuration.embedding_model)) as retriever:
        response = retriever.invoke(state.enhanced_query, config)
        return {
            "retrieved_docs": response
        }


def respond(
    state: State, *, config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
    """Call the LLM powering our "agent"."""
    configuration = Configuration.from_runnable_config(config)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", configuration.response_system_prompt),
            ("human", "{messages}"),
        ]
    )
    model = load_chat_model(model = configuration.response_model)

    retrieved_docs = format_docs(state.retrieved_docs)
    history = format_messages(state.messages)
    
    message_value = prompt.invoke(
        {
            "messages": state.messages,
            "context": retrieved_docs,
            "history": history,
            "system_time": datetime.now(tz=timezone.utc).isoformat(),
        },
        config,
    )
    response = model.invoke(message_value, config)
    # We return a list, because this will get added to the existing list
    return {
        "messages": [response],
        "retrieved_docs": [],
        "enhanced_query": "",
    }


def get_retrieval_graph() -> CompiledStateGraph:
    """
    Constructs and returns the retrieval graph.

    This LangGraph pipeline follows a 3-step retrieval process:

        InputState → generate_query → retrieve → respond → OutputState

    Graph Overview:
    ----------------
        [START]
           |
    ┌──────▼─────────┐
    │ generate_query │
    └──────┬─────────┘
           |
    ┌──────▼──────┐
    │  retrieve   │
    └──────┬──────┘
           |
    ┌──────▼─────┐
    │  respond   │
    └────────────┘

    Nodes:
        - `generate_query`: Transforms the input into a retrievable query.
        - `retrieve`: Fetches documents or context based on the generated query.
        - `respond`: Generates a response based on the retrieved content.

    Returns:
        A compiled `StateGraph` that can be invoked with appropriate state and configuration.

    Usage:
    ------
    To invoke the compiled graph, provide an initial state and a config dictionary:

        ```python
        from your_module import get_retrieval_graph, Configuration, State

        config = Configuration(...)  # your pydantic or dataclass config instance
        state = State(...)  # your initial input state

        graph = get_retrieval_graph()
        result = graph.invoke(state, {"configurable": config})
        ```

    Notes:
        - The configuration is passed using the `{"configurable": ...}` key, as required by LangGraph.
        - Ensure the `Configuration` object matches the schema expected by the graph (`config_schema`).
    """

    builder = StateGraph(State, input=InputState, config_schema=Configuration)

    builder.add_node(generate_query)
    builder.add_node(retrieve)
    builder.add_node(respond)
    builder.add_edge("__start__", "generate_query")
    builder.add_edge("generate_query", "retrieve")
    builder.add_edge("retrieve", "respond")

    # Finally, we compile it!
    # This compiles it into a graph you can invoke and deploy.
    memory = SqliteSaver(get_connection())
    graph = builder.compile(
        checkpointer=memory, # Use the SQLite checkpointer for memory
        interrupt_before=[],
        interrupt_after=[],
    )
    return graph