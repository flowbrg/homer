"""Main entrypoint for the conversational retrieval graph.

This module defines the core structure and functionality of the conversational
retrieval graph. It includes the main graph definition, state management,
and key functions for processing user inputs, generating queries, retrieving
relevant documents, and formulating responses.
"""
from src.utils.logging import setup_logging, get_logger
#setup_logging("INFO")  # or "DEBUG" for more detailed logs
# Initialize retrievalAgentLogger
retrievalAgentLogger = get_logger(__name__)

from typing import cast, Dict, List, Union
from pydantic import BaseModel

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.sqlite import SqliteSaver

from src.core import retrieval
from src.core.states import InputState, RetrievalState
from src.core.configuration import Configuration
from src.core.models import load_chat_model, load_embedding_model
from src.utils.utils import format_docs, format_messages, format_sources, get_connection, combine_prompts, ya_format_messages
from src.utils import prompts



class SearchQuery(BaseModel):
    """
    Pydantic model for structured search query output.
    
    This model ensures that the language model returns a properly formatted
    search query string when generating queries from conversation context.
    
    Attributes:
        query (str): The generated search query string optimized for document retrieval.
    """
    query: str


def rephrase_query(
    state: RetrievalState, *, config: RunnableConfig
) -> Dict[str, Union[str, List]]:
    """
    Generate an optimized search query based on conversation context.

    This function analyzes the current conversation state and generates a search query
    optimized for document retrieval. It uses a language model with structured output
    to create queries that effectively capture the user's information needs while
    considering conversation history.

    Args:
        state (RetrievalState): Current conversation state containing:
            - messages: List of conversation messages
            - retrieved_docs: Previously retrieved documents (optional)
            - Other state information for context
        config (RunnableConfig): Configuration containing:
            - query_model: Model identifier for query generation
            - ollama_host: Host URL for Ollama models (if applicable)
            - Other model and system configurations

    Returns:
        Dict[str, Union[str, List]]: Dictionary containing:
            - "query" (str): Generated search query string
            - "retrieved_docs" (List): Empty list or "delete" to clear previous docs

    Raises:
        ValueError: If configuration is invalid or missing required parameters
        Exception: If model loading or query generation fails

    Example:
        >>> state = RetrievalState(messages=[user_message, ai_message])
        >>> config = RunnableConfig(configurable=Configuration(...))
        >>> result = generate_query(state, config=config)
        >>> print(result["query"])  # "search query about user's question"

    Note:
        The function uses conversation history (last 3 messages) to provide context
        for better query generation, improving retrieval relevance.
    """
    retrievalAgentLogger.info("Starting query generation")
    
    try:
        # Get configuration
        configuration = Configuration.from_runnable_config(config)
        if not configuration:
            retrievalAgentLogger.error("Configuration not found in config")
            raise ValueError("Configuration is required for query generation")
        
        retrievalAgentLogger.debug(f"Using query model: {configuration.query_model}")

        # Load and configure model
        model = load_chat_model(
            model=configuration.query_model, 
            host=configuration.ollama_host
        ).with_structured_output(SearchQuery)
        
        # Prepare message context
        previous_messages = (
            format_messages(state.messages[-3:-1]) 
            if len(state.messages) >= 3 
            else "There were no previous messages."
        )

        # Create prompt
        system_prompt = prompts.REPHRASE_QUERY_SYSTEM_PROMPT.format(
            previous_messages= previous_messages,
        )
        user_prompt = state.messages[-1].content

        messages = [
            ("human", combine_prompts(system_prompt,user_prompt)),
        ]

        # Generate rephrased query
        generated = cast(SearchQuery, model.invoke(messages, config))
        
        retrievalAgentLogger.info(f"Generated query: '{generated.query}'")
        
        return {
            "query": generated.query,
            "retrieved_docs": "delete" if state.retrieved_docs else [],
        }
        
    except Exception as e:
        retrievalAgentLogger.error(f"Error in generate_query: {str(e)}")
        # Return a fallback query based on the last user message
        try:
            fallback_query = state.messages[-1].content if state.messages else "general search"
            retrievalAgentLogger.warning(f"Using fallback query: '{fallback_query}'")
            return {
                "query": fallback_query,
                "retrieved_docs": "delete" if state.retrieved_docs else [],
            }
        except Exception as fallback_error:
            retrievalAgentLogger.error(f"Fallback query generation failed: {str(fallback_error)}")
            raise e


def retrieve(
    state: RetrievalState, *, config: RunnableConfig
) -> Dict[str, List[Document]]:
    """
    Retrieve relevant documents based on the generated query.

    This function takes a search query from the state and retrieves the most relevant
    documents from the indexed document collection using vector similarity search.
    It uses the configured embedding model to encode the query and find matching documents.

    Args:
        state (RetrievalState): Current state containing:
            - query: The search query string to retrieve documents for
            - Other state information
        config (RunnableConfig): Configuration containing:
            - embedding_model: Model identifier for document embeddings
            - ollama_host: Host URL for Ollama models (if applicable)
            - Retrieval parameters (top_k, similarity threshold, etc.)

    Returns:
        Dict[str, List[Document]]: Dictionary containing:
            - "retrieved_docs": List of relevant Document objects with metadata

    Raises:
        ValueError: If configuration is invalid or query is empty
        Exception: If embedding model loading or document retrieval fails

    Example:
        >>> state = RetrievalState(query="machine learning algorithms")
        >>> config = RunnableConfig(configurable=Configuration(...))
        >>> result = retrieve(state, config=config)
        >>> print(len(result["retrieved_docs"]))  # Number of retrieved documents

    Note:
        The function uses a context manager for the retriever to ensure proper
        resource cleanup after document retrieval.
    """
    retrievalAgentLogger.info(f"Starting document retrieval for query: '{state.query}'")
    
    try:
        # Validate query
        if not state.query or not state.query.strip():
            retrievalAgentLogger.warning("Empty or whitespace-only query provided")
            return {"retrieved_docs": []}
        
        # Get configuration
        configuration = Configuration.from_runnable_config(config)
        if not configuration:
            retrievalAgentLogger.error("Configuration not found in config")
            raise ValueError("Configuration is required for document retrieval")
        
        retrievalAgentLogger.debug(f"Using embedding model: {configuration.embedding_model}")
        
        # Load embedding model
        embeddings = load_embedding_model(
            model=configuration.embedding_model, 
            host=configuration.ollama_host
        )
        
        # Retrieve documents
        with retrieval.make_retriever(embedding_model=embeddings) as retriever:
            response = retriever.invoke(state.query, config)
            
            if response:
                retrievalAgentLogger.info(f"Successfully retrieved {len(response)} documents")
                for doc in response:
                    retrievalAgentLogger.debug(f"Document: {doc.page_content} from {doc.metadata.get('source', 'unknown')}\n")
            else:
                retrievalAgentLogger.warning("No documents retrieved for the query")
            
            return {"retrieved_docs": response}
    
    except Exception as e:
        retrievalAgentLogger.error(f"Error in retrieve: {str(e)}")
        retrievalAgentLogger.warning("Returning empty document list due to retrieval error")
        return {"retrieved_docs": []}


def respond(
    state: RetrievalState, *, config: RunnableConfig
) -> Dict[str, Union[List[BaseMessage], List, str]]:
    """
    Generate a conversational response based on retrieved documents and chat history.

    This function creates a contextual response using the retrieved documents as context,
    considering the conversation history and any existing conversation summary. It handles
    message history efficiently by using summaries for older conversations.

    Args:
        state (RetrievalState): Current state containing:
            - messages: Conversation history
            - retrieved_docs: Documents retrieved for context
            - summary: Optional conversation summary for context
        config (RunnableConfig): Configuration containing:
            - response_model: Model identifier for response generation
            - ollama_host: Host URL for Ollama models (if applicable)
            - Response generation parameters

    Returns:
        Dict[str, Union[List[BaseMessage], List, str]]: Dictionary containing:
            - "messages": List with the generated response message
            - "retrieved_docs": Empty list (cleared after use)
            - "query": Empty string (cleared after use)

    Raises:
        ValueError: If configuration is invalid
        Exception: If model loading or response generation fails

    Example:
        >>> state = RetrievalState(
        ...     messages=[user_msg], 
        ...     retrieved_docs=[doc1, doc2],
        ...     summary="Previous conversation about AI"
        ... )
        >>> result = respond(state, config=config)
        >>> print(result["messages"][0].content)  # Generated response

    Side Effects:
        - Prints source information to console for debugging
        - Clears retrieved_docs and query from state after processing

    Note:
        Uses modulo arithmetic (len(messages) % 6) to determine which recent
        messages to include, working with the summarization cycle.
    """
    retrievalAgentLogger.info("Starting response generation")
    
    try:
        # Get configuration
        configuration = Configuration.from_runnable_config(config)
        if not configuration:
            retrievalAgentLogger.error("Configuration not found in config")
            raise ValueError("Configuration is required for response generation")
        
        retrievalAgentLogger.debug(f"Using response model: {configuration.response_model}")
        
        # Load model
        model = load_chat_model(
            model=configuration.response_model, 
            host=configuration.ollama_host
        )
        
        # Prepare context
        previous_messages = ya_format_messages(state.messages[-3:-1] if len(state.messages)>2 else [])
        context_docs = format_docs(state.retrieved_docs) if state.retrieved_docs else ""
        
        # Create prompt
        system_prompt = prompts.RESPONSE_SYSTEM_PROMPT.format(
            context = context_docs,
            summary = state.summary if state.summary else "",
        )

        messages = [
            ("system", system_prompt)
        ] + previous_messages + [
            ("human", state.messages[-1].content)
        ]

        # Generate response
        response = model.invoke(input=messages, config=config)
        
        retrievalAgentLogger.info("Response generated successfully")
        
        # Display sources for debugging
        if state.retrieved_docs:
            retrievalAgentLogger.info("Displaying source information")
            print("[info] Sources retrieved for current thread")
            print(format_sources(documents=state.retrieved_docs))
        else:
            retrievalAgentLogger.debug("No retrieved documents to display")

        return {
            "messages": [response],
            "retrieved_docs": [],
            "query": "",
        }
        
    except Exception as e:
        retrievalAgentLogger.error(f"Error in respond: {str(e)}")
        # Create a fallback response
        try:
            from langchain_core.messages import AIMessage
            fallback_response = AIMessage(content="I apologize, but I encountered an error while generating a response. Please try rephrasing your question.")
            retrievalAgentLogger.warning("Using fallback response due to error")
            return {
                "messages": [fallback_response],
                "retrieved_docs": [],
                "query": "",
            }
        except Exception as fallback_error:
            retrievalAgentLogger.error(f"Fallback response generation failed: {str(fallback_error)}")
            raise e


def summarize_conversation(
    state: RetrievalState, *, config: RunnableConfig
) -> Dict[str, str]:
    """
    Create or extend a conversation summary to manage long chat histories.

    This function generates a concise summary of recent conversation messages,
    either creating a new summary or extending an existing one. This helps
    maintain context while keeping prompt sizes manageable for long conversations.

    Args:
        state (RetrievalState): Current state containing:
            - messages: Conversation history to summarize
            - summary: Existing summary to extend (optional)
        config (RunnableConfig): Configuration containing:
            - query_model: Model identifier for summarization
            - ollama_host: Host URL for Ollama models (if applicable)

    Returns:
        Dict[str, str]: Dictionary containing:
            - "summary": Generated or updated conversation summary

    Raises:
        ValueError: If configuration is invalid
        Exception: If model loading or summarization fails

    Example:
        >>> state = RetrievalState(
        ...     messages=[msg1, msg2, msg3, msg4, msg5, msg6],
        ...     summary="Previous summary of earlier conversation"
        ... )
        >>> result = summarize_conversation(state, config=config)
        >>> print(result["summary"])  # Updated conversation summary

    Note:
        - Processes the last 6 messages to create/update the summary
        - Uses different prompts for creating new summaries vs. extending existing ones
        - Helps maintain conversation context while reducing prompt length
    """
    retrievalAgentLogger.info("Starting conversation summarization")
    
    try:
        # Get configuration
        configuration = Configuration.from_runnable_config(config)
        if not configuration:
            retrievalAgentLogger.error("Configuration not found in config")
            raise ValueError("Configuration is required for conversation summarization")
        
        # Get existing summary
        existing_summary = state.summary if state.summary else ""
        messages_to_summarize = ya_format_messages(state.messages[-6:])  # Last 6 messages
        
        retrievalAgentLogger.info(f"Summarizing {len(messages_to_summarize)} messages, "
                   f"existing summary: {'Yes' if existing_summary else 'No'}")
        
        # Determine prompt based on existing summary
        if existing_summary:
            summary_system_prompt = f"""This is summary of the conversation to date:
<summary>
{existing_summary}
</summary>

Extend the summary by taking into account the new messages:"""
            retrievalAgentLogger.debug("Extending existing summary")
        else:
            summary_system_prompt = "Create a summary of the conversation:"
            retrievalAgentLogger.debug("Creating new summary")

        # Load model
        model = load_chat_model(
            model=configuration.query_model, 
            host=configuration.ollama_host
        )
        
        # Create prompt
        messages = [
            ('system', summary_system_prompt)
        ] + messages_to_summarize

        
        response = model.invoke(messages, config)
        
        retrievalAgentLogger.info("Conversation summary generated successfully")
        retrievalAgentLogger.debug(f"Summary length: {len(response.content)} characters")
        
        return {"summary": response.content}
        
    except Exception as e:
        retrievalAgentLogger.error(f"Error in summarize_conversation: {str(e)}")
        # Return existing summary or empty string as fallback
        fallback_summary = state.summary if state.summary else ""
        retrievalAgentLogger.warning(f"Using fallback summary due to error: {'existing' if fallback_summary else 'empty'}")
        return {"summary": fallback_summary}


def should_summarize(
    state: RetrievalState, *, config: RunnableConfig
) -> str:
    """
    Determine whether conversation summarization should occur.

    This function implements the logic for deciding when to summarize the conversation
    based on message count. It triggers summarization every 6 messages to keep
    conversation context manageable while preserving important information.

    Args:
        state (RetrievalState): Current state containing conversation messages
        config (RunnableConfig): Configuration for the decision process

    Returns:
        str: Either "summarize_conversation" to trigger summarization,
             or END to continue without summarization

    Example:
        >>> state = RetrievalState(messages=[msg1, msg2, msg3, msg4, msg5, msg6])
        >>> should_summarize(state, config)  # Returns "summarize_conversation"
        >>> 
        >>> state = RetrievalState(messages=[msg1, msg2, msg3])
        >>> should_summarize(state, config)  # Returns END

    Note:
        Uses modulo arithmetic (len(messages) % 6 == 0) to trigger summarization
        every 6 messages, maintaining a consistent conversation management cycle.
    """
    message_count = len(state.messages)
    should_trigger = message_count % 6 == 0
    
    retrievalAgentLogger.info(f"Summarization check: {message_count} messages, "
               f"trigger: {'Yes' if should_trigger else 'No'}")
    
    if should_trigger:
        retrievalAgentLogger.info("Triggering conversation summarization")
        return "summarize_conversation"
    else:
        retrievalAgentLogger.debug("Continuing without summarization")
        return END


def get_retrieval_graph() -> CompiledStateGraph:
    """
    Construct and compile the conversational retrieval graph.

    This function creates a LangGraph StateGraph that implements a complete
    conversational retrieval system with memory management. The graph handles
    query generation, document retrieval, response generation, and conversation
    summarization in a coordinated workflow.

    Returns:
        CompiledStateGraph: A compiled graph ready for execution with:
            - SQLite checkpointer for conversation persistence
            - Proper state management and transitions
            - Error handling and recovery mechanisms

    Graph Architecture:
        ```
        [START] → rephrase_query → retrieve → respond → [should_summarize?]
                                                                ↓
                                                END ← summarize_conversation
        ```

    Nodes:
        - generate_query: Transforms user input into optimized search queries
        - retrieve: Fetches relevant documents using vector similarity
        - respond: Generates contextual responses using retrieved documents
        - summarize_conversation: Creates/updates conversation summaries

    Example:
        >>> graph = get_retrieval_graph()
        >>> config = {"configurable": Configuration(...)}
        >>> thread_config = {"configurable": {"thread_id": "user_123"}}
        >>> 
        >>> result = graph.invoke(
        ...     {"messages": [user_message]}, 
        ...     config={**config, **thread_config}
        ... )
        >>> print(result["messages"][-1].content)

    Features:
        - Conversation persistence with SQLite checkpointer
        - Automatic conversation summarization every 6 messages
        - Document retrieval with vector similarity search
        - Contextual response generation with chat history
        - Robust error handling and fallback mechanisms

    Raises:
        Exception: If graph compilation fails or required components are missing

    Note:
        The graph uses Configuration schema for type safety and includes
        comprehensive logging throughout the execution flow.
    """
    retrievalAgentLogger.info("Building conversational retrieval graph")
    
    try:
        # Create StateGraph with proper schemas
        builder = StateGraph(
            RetrievalState, 
            input_schema=InputState, 
            config_schema=Configuration
        )

        # Add nodes
        retrievalAgentLogger.debug("Adding graph nodes")
        builder.add_node(rephrase_query)
        builder.add_node(retrieve)
        builder.add_node(respond)
        builder.add_node(summarize_conversation)

        # Define edges
        retrievalAgentLogger.debug("Defining graph edges")
        builder.add_edge("__start__", "rephrase_query")
        builder.add_edge("rephrase_query", "retrieve")
        builder.add_edge("retrieve", "respond")
        builder.add_conditional_edges("respond", should_summarize)

        # Setup memory/checkpointer
        retrievalAgentLogger.debug("Setting up SQLite checkpointer")
        memory = SqliteSaver(get_connection())

        # Compile graph
        graph = builder.compile(
            checkpointer=memory,
            interrupt_before=[],
            interrupt_after=[],
        )
        
        retrievalAgentLogger.info("Successfully compiled conversational retrieval graph")
        return graph
        
    except Exception as e:
        retrievalAgentLogger.error(f"Error building retrieval graph: {str(e)}")
        raise