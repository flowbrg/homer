"""Main entrypoint for the conversational retrieval graph.

This module defines the core structure and functionality of the conversational
retrieval graph. It includes the main graph definition, state management,
and key functions for processing user inputs, generating queries, retrieving
relevant documents, and formulating responses.
"""


from utils.logging import get_logger
# Initialize logger
logger = get_logger(__name__)

from typing import cast, Dict, List, Union
from pydantic import BaseModel

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.sqlite import SqliteSaver

from core import retrieval
from core.states import InputState, RetrievalState
from core.configuration import Configuration
from core.models import load_chat_model, load_embedding_model
from utils.utils import (format_docs, format_messages, format_sources,
                         get_connection, combine_prompts)
from core import prompts



########################### Structured output class ##########################


class SearchQuery(BaseModel):
  """
  Pydantic model for structured search query output.
  
  This model ensures that the language model returns a properly formatted
  search query string when generating queries from conversation context.
  
  Attributes:
    query (str): The generated search query string optimized for document
                 retrieval.
  """
  query: str


###############################################################################
#                            Nodes initialization                             #
###############################################################################

############################# Rephrase query node #############################


def rephrase_query(
  state: RetrievalState, *, config: RunnableConfig
) -> Dict[str, Union[str, List]]:
  
  logger.info("Starting query generation")
  
  try:
    # Get configuration
    configuration = Configuration.from_runnable_config(config)
    if not configuration:
      logger.error("Configuration not found in config")
      raise ValueError("Configuration is required for query generation")
    
    logger.debug(f"Using query model: {configuration.query_model}")

    # Load and configure model
    model = load_chat_model(
      model=configuration.query_model, 
      host=configuration.ollama_host
    ).with_structured_output(SearchQuery)
    
    # Prepare message context
    previous_messages = "There were no previous messages."
    if len(state.messages) >= 3 :
      format_messages(state.messages[-3:-1]) 

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
    
    logger.info(f"Generated query: '{generated.query}'")
    
    return {
      "query": generated.query,
      "retrieved_docs": "delete" if state.retrieved_docs else [],
    }
    
  except Exception as e:
    logger.error(f"Error in generate_query: {str(e)}")
    # Return a fallback query based on the last user message
    try:
      fallback_query = "general search"
      if state.messages:
        state.messages[-1].content
      logger.warning(f"Using fallback query: '{fallback_query}'")
      return {
        "query": fallback_query,
        "retrieved_docs": "delete" if state.retrieved_docs else [],
      }
    except Exception as fallback_error:
      logger.error(f"Fallback query generation failed: {str(fallback_error)}")
      raise e


################################# Retrieve node ###############################


def retrieve(
  state: RetrievalState, *, config: RunnableConfig
) -> Dict[str, List[Document]]:

  logger.info(f"Starting document retrieval for query: '{state.query}'")
  
  try:
    # Validate query
    if not state.query or not state.query.strip():
      logger.warning("Empty or whitespace-only query provided")
      return {"retrieved_docs": []}
    
    # Get configuration
    configuration = Configuration.from_runnable_config(config)
    if not configuration:
      logger.error("Configuration not found in config")
      raise ValueError("Configuration is required for document retrieval")
    
    logger.debug(f"Using embedding model: {configuration.embedding_model}")
    
    # Load embedding model
    embeddings = load_embedding_model(
      model=configuration.embedding_model, 
      host=configuration.ollama_host
    )
    
    # Retrieve documents
    with retrieval.make_retriever(embedding_model=embeddings) as retriever:
      response = retriever.invoke(state.query, config)
      
      if response:
        logger.info(f"Successfully retrieved {len(response)} documents")
        for doc in response:
          logger.debug(
            f"Document: {doc.page_content[:500]}... from {doc.metadata.get('source', 'unknown')}\n")
      else:
        logger.warning("No documents retrieved for the query")
      
      return {"retrieved_docs": response}
  
  except Exception as e:
    logger.error(f"Error in retrieve: {str(e)}")
    logger.warning("Returning empty document list due to retrieval error")
    return {"retrieved_docs": []}


################################# Respond node ################################


def respond(
  state: RetrievalState, *, config: RunnableConfig
) -> Dict[str, Union[List[BaseMessage], List, str]]:
  
  logger.info("Starting response generation")
  
  try:
    # Get configuration
    configuration = Configuration.from_runnable_config(config)
    if not configuration:
      logger.error("Configuration not found in config")
      raise ValueError("Configuration is required for response generation")
    
    logger.debug(f"Using response model: {configuration.response_model}")
    
    # Load model
    model = load_chat_model(
      model=configuration.response_model, 
      host=configuration.ollama_host
    )
    
    # Prepare context
    context_docs = ""
    if state.retrieved_docs:
      context_docs = format_docs(state.retrieved_docs)
    
    # Create prompt
    system_prompt = prompts.RESPONSE_SYSTEM_PROMPT.format(
      context = context_docs,
      summary = state.summary if state.summary else "",
    )

    logger.debug(f"System prompt: {system_prompt[:1000]}...")

    messages = [
      ("system", system_prompt),
      ("human", state.messages[-1].content),
    ]

    # Generate response
    response = model.invoke(input=messages, config=config)
    
    logger.info("Response generated successfully")
    
    # Display sources for debugging
    if state.retrieved_docs:
      logger.debug("Displaying source information")
      print("=====Sources retrieved for current thread=====")
      print(format_sources(documents=state.retrieved_docs))
    else:
      logger.debug("No retrieved documents to display")

    return {
      "messages": [response],
      "retrieved_docs": [],
      "query": "",
    }
    
  except Exception as e:
    logger.error(f"Error in respond: {str(e)}")
    # Create a fallback response
    try:
      from langchain_core.messages import AIMessage
      fallback_response = AIMessage(content="""I apologize, but I encountered an error while generating a response. Please try rephrasing your question.""")
      logger.warning("Using fallback response due to error")
      return {
        "messages": [fallback_response],
        "retrieved_docs": [],
        "query": "",
      }
    except Exception as fallback_error:
      logger.error(
        f"Fallback response generation failed: {str(fallback_error)}")
      raise e


################################ Summarize node ###############################


def summarize_conversation(
  state: RetrievalState, *, config: RunnableConfig
) -> Dict[str, str]:
  
  logger.info("Starting conversation summarization")
  
  try:
    # Get configuration
    configuration = Configuration.from_runnable_config(config)
    if not configuration:
      logger.error("Configuration not found in config")
      raise ValueError(
        "Configuration is required for conversation summarization")
    
    # Get existing summary
    existing_summary = state.summary if state.summary else ""
    messages_to_summarize = state.messages[-6:]  # Last 6 messages
    
    logger.info(f"Summarizing {len(messages_to_summarize)} messages, "
           f"existing summary: {'Yes' if existing_summary else 'No'}")
    
    # Determine prompt based on existing summary
    if existing_summary:
      summary_system_prompt = f"""This is summary of the conversation to date:
<summary>
{existing_summary}
</summary>

Extend the summary by taking into account the new messages:"""
      logger.debug("Extending existing summary")
    else:
      summary_system_prompt = "Create a summary of the conversation:"
      logger.debug("Creating new summary")

    # Load model
    model = load_chat_model(
      model=configuration.query_model, 
      host=configuration.ollama_host
    )
    
    # Create prompt
    messages = [SystemMessage(content=summary_system_prompt)] + messages_to_summarize

    
    response = model.invoke(messages, config)
    
    logger.info("Conversation summary generated successfully")
    logger.debug(f"Summary length: {len(response.content)} characters")
    
    return {"summary": response.content}
    
  except Exception as e:
    logger.error(f"Error in summarize_conversation: {str(e)}")
    # Return existing summary or empty string as fallback
    fallback_summary = state.summary if state.summary else ""
    logger.warning(f"Using fallback summary due to error: {'existing' if fallback_summary else 'empty'}")
    return {"summary": fallback_summary}


###################### Should summarize conditional edge ######################


def should_summarize(
  state: RetrievalState, *, config: RunnableConfig
) -> str:
  
  message_count = len(state.messages)
  should_trigger = message_count % 6 == 0
  
  logger.info(f"Summarization check: {message_count} messages, "
         f"trigger: {'Yes' if should_trigger else 'No'}")
  
  if should_trigger:
    logger.info("Triggering conversation summarization")
    return "summarize_conversation"
  else:
    logger.debug("Continuing without summarization")
    return END


###############################################################################
#                               Graph compiler                                #
###############################################################################

def get_retrieval_graph() -> CompiledStateGraph:
  
  logger.info("Building conversational retrieval graph")
  
  try:
    # Create StateGraph with proper schemas
    builder = StateGraph(
      RetrievalState, 
      input_schema=InputState, 
      config_schema=Configuration
    )

    # Add nodes
    logger.debug("Adding graph nodes")
    builder.add_node(rephrase_query)
    builder.add_node(retrieve)
    builder.add_node(respond)
    builder.add_node(summarize_conversation)

    # Define edges
    logger.debug("Defining graph edges")
    builder.add_edge("__start__", "rephrase_query")
    builder.add_edge("rephrase_query", "retrieve")
    builder.add_edge("retrieve", "respond")
    builder.add_conditional_edges("respond", should_summarize)

    # Setup memory/checkpointer
    logger.debug("Setting up SQLite checkpointer")
    memory = SqliteSaver(get_connection())

    # Compile graph
    graph = builder.compile(
      checkpointer=memory,
      interrupt_before=[],
      interrupt_after=[],
    )
    
    logger.info("Successfully compiled conversational retrieval graph")
    return graph
    
  except Exception as e:
    logger.error(f"Error building retrieval graph: {str(e)}")
    raise