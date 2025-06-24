"""Main entrypoint for the conversational retrieval graph.

This module defines the core structure and functionality of the conversational
retrieval graph. It is used for a simple query to a LLM to generate a name for
the conversation. Only relevant in the case of persistent database for discussions
"""

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig

from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.core.states import InputState
from src.core.configuration import Configuration
from src.core.models import load_chat_model
from src.resources import prompts


def respond(
    state: InputState, *, config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
    """Call the LLM powering our "agent"."""
    configuration = Configuration.from_runnable_config(config)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompts.NAME_SYSTEM_PROMPT),
            ("human", "{messages}"),
        ]
    )
    model = load_chat_model(model = configuration.response_model)
    
    message_value = prompt.invoke(
        {
            "messages": state.messages,
        },
        config,
    )
    response = model.invoke(message_value, config)
    # We return a list, because this will get added to the existing list
    return {
        "messages": [response],
    }


def get_simple_query_graph() -> CompiledStateGraph:
    """
    Constructs and returns the retrieval graph.

    This LangGraph pipeline follows a 3-step retrieval process:

        InputState → respond → OutputState

    Graph Overview:
    ----------------
        [START]
           |
    ┌──────▼─────┐
    │  respond   │
    └────────────┘

    Nodes:
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

    builder = StateGraph(InputState, input=InputState, config_schema=Configuration)

    builder.add_node(respond)
    builder.add_edge("__start__", "respond")

    # Finally, we compile it!
    # This compiles it into a graph you can invoke and deploy.
    graph = builder.compile()

    return graph