"""Main entrypoint for the conversational retrieval graph.

This module defines the core structure and functionality of the conversational
retrieval graph. It includes the main graph definition, state management,
and key functions for processing user inputs, generating queries, retrieving
relevant documents, and formulating responses.
"""

#from datetime import datetime, timezone
from typing import cast
from pydantic import BaseModel

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.sqlite import SqliteSaver

from src.core import retrieval
from src.core.agents.states import InputState, RetrievalState
from src.core.configuration import Configuration
from src.resources.utils import format_docs, format_messages, format_sources, get_message_text, get_connection
from src.resources import prompts
from src.core.models import load_chat_model, load_embedding_model

# Define the function that calls the model

class SearchQuery(BaseModel):
    """Search the indexed documents for a query."""

    query: str


def generate_query(
    state: RetrievalState, *, config: RunnableConfig
) -> dict[str, list[str]]:
    """Generate a search query based on the current state and configuration.

    This function analyzes the messages in the state and generates an appropriate
    search query. For the first message, it uses the user's input directly.
    For subsequent messages, it uses a language model to generate a refined query.

    Args:
        state (RetrievalState): The current state containing messages and other information.
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
            ("system", prompts.IMPROVE_QUERY_SYSTEM_PROMPT),
            ("human", "{message}"),
        ]
    )

    model = load_chat_model(model = configuration.query_model, ).with_structured_output(
        SearchQuery
    )

    message_value = prompt.invoke(
        {
            "message": format_messages(state.messages[-1:]),
            "previous_messages": format_messages(state.messages[-3:-1]) if len(state.messages) >= 3 else "There were no previous messages.",
            #"system_time": datetime.now(tz=timezone.utc).isoformat(),
        },
        config,
    )

    generated = cast(SearchQuery, model.invoke(message_value, config))
    
    return {
        "query": generated.query,
        "retrieved_docs": "delete" if state.retrieved_docs else [],
    }


def retrieve(
    state: RetrievalState, *, config: RunnableConfig
) -> dict[str, list[Document]]:
    """Retrieve documents based on the latest query in the state.

    This function takes the current state and configuration, uses the latest query
    from the state to retrieve relevant documents using the retriever, and returns
    the retrieved documents.

    Args:
        state (RetrievalState): The current state containing queries and the retriever.
        config (RunnableConfig | None, optional): Configuration for the retrieval process.

    Returns:
        dict[str, list[Document]]: A dictionary with a single key "retrieved_docs"
        containing a list of retrieved Document objects.
    """
    configuration = Configuration.from_runnable_config(config)
    with retrieval.make_retriever(embedding_model = load_embedding_model(model=configuration.embedding_model)) as retriever:
        response = retriever.invoke(state.query, config)
        return {
            "retrieved_docs": response
        }


def respond(
    state: RetrievalState, *, config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
    """Call the LLM powering our "agent"."""
    configuration = Configuration.from_runnable_config(config)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompts.RESPONSE_SYSTEM_PROMPT),
            ("human", "{messages}"),
        ]
    )
    model = load_chat_model(model = configuration.response_model)
    
    nb_messages = len(state.messages)%6 # The amount of messages since last summary
    message_value = prompt.invoke(
        {
            "messages": format_messages(state.messages[-1:]),
            "context": format_docs(state.retrieved_docs),
            # Limit to last 6 messages, which were not included in the last summary
            "previous_messages": format_messages(state.messages[-nb_messages:-1]) if len(state.messages) >= 6 else "There were no previous messages.",
            "summary": state.summary if state.summary else "",
            #"system_time": datetime.now(tz=timezone.utc).isoformat(),
        },                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
        config,
    )
    response = model.invoke(message_value, config)
    # We return a list, because this will get added to the existing list

    print("[info] Sources retrievedfor current thread")
    print(format_sources(documents = state.retrieved_docs if state.retrieved_docs else []))

    return {
        "messages": [response],
        "retrieved_docs": [],
        "query": "",
    }


def summarize_conversation(
    state: RetrievalState, *, config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
    # First, we get any existing summary
    summary = state.summary if state.summary else ""
    # If we have a summary, we will extend it with the new messages

    configuration = Configuration.from_runnable_config(config)

    # Create our summarization prompt 
    if summary:
        
        # A summary already exists
        summary_system_prompt = f"""This is summary of the conversation to date:
        <summary>
        {summary}
        </summary>

        Extend the summary by taking into account the new messages above:
        """

    else:
        summary_system_prompt = "Create a summary of the conversation:"

    model = load_chat_model(model = configuration.query_model)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", summary_system_prompt),
            ("human", "{messages}"),
        ]
    )
    message_value = prompt.invoke(
        {
            "messages": state.messages[-6:], # Limit to last 6 messages
        },
        config,
    )

    # Add prompt to our history
    response = model.invoke(message_value, config)
    
    # Delete all but the 2 most recent messages
    return {"summary": response.content}

def should_summarize(
    state: RetrievalState, *, config: RunnableConfig
):
    if len(state.messages) % 6 == 0:
        return "summarize_conversation"
    return END

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

    builder = StateGraph(RetrievalState, input=InputState, config_schema=Configuration)

    builder.add_node(generate_query)
    builder.add_node(retrieve)
    builder.add_node(respond)
    builder.add_node(summarize_conversation)
    builder.add_edge("__start__", "generate_query")
    builder.add_edge("generate_query", "retrieve")
    builder.add_edge("retrieve", "respond")
    builder.add_conditional_edges("respond", should_summarize)

    # Finally, we compile it!
    # This compiles it into a graph you can invoke and deploy.
    memory = SqliteSaver(get_connection())
    graph = builder.compile(
        checkpointer=memory, # Use the SQLite checkpointer for memory
        interrupt_before=[],
        interrupt_after=[],
    )
    return graph