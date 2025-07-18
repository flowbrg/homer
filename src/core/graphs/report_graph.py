"""Main entrypoint for the report generation graph.

This module defines the core structure and functionality of the report generation
graph. It includes the main graph definition, state management, and key functions 
for processing user inputs, generating outlines, retrieving relevant documents for 
each section, synthesizing content, and reviewing/polishing the final report sections.

The report generation follows a structured workflow:
1. Initial document retrieval based on user query
2. Outline generation with configurable number of sections
3. Section-by-section processing:
   - Document retrieval for current section
   - Content synthesis from retrieved documents
   - Content review and polishing
4. Iterative processing until all sections are complete

The graph supports both technical and general writing styles, with appropriate
prompts and formatting for each style.
"""

from utils.logging import get_logger
# Initialize retrievalAgentLogger
logger = get_logger(__name__)

from pydantic import BaseModel
from typing import cast, Any

from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

from core.states import ReportState, InputState
from core.configuration import Configuration
from core import retrieval
from core.models import load_chat_model, load_embedding_model
from utils.utils import format_docs, get_message_text, combine_prompts
from core import prompts


######################################## Structured output class ########################################


class Outline(BaseModel):
    """
    Represents an outline for a report with multiple entries.
    
    This Pydantic model ensures structured output from the language model
    when generating report outlines. Each entry represents a major section
    or topic that will be covered in the final report.
    
    Attributes:
        entries (list[str]): List of outline entries/section titles for the report.
                           Each entry should be a clear, descriptive title that
                           guides the content generation for that section.
    
    Example:
        >>> outline = Outline(entries=[
        ...     "Introduction to Machine Learning",
        ...     "Supervised Learning Algorithms", 
        ...     "Unsupervised Learning Methods",
        ...     "Conclusion and Future Directions"
        ... ])
    """
    entries: list[str]


######################################## Initial retrieval node ########################################


def initial_retrieval(
    state: ReportState, *, config: RunnableConfig) -> dict[str, list[Document]]:
    """
    Retrieve documents based on the input state to create the outline.
    
    This function performs the initial document retrieval using the user's main query
    to gather relevant context that will inform the outline generation. It serves as
    the foundation for understanding what information is available for the report.

    Args:
        state (ReportState): Current report generation state containing:
            - messages: List of conversation messages with user query
            - Other state information for report generation context
        config (RunnableConfig): Configuration containing:
            - embedding_model: Model identifier for document embeddings
            - ollama_host: Host URL for Ollama models (if applicable)

    Returns:
        dict[str, list[Document]]: Dictionary containing:
            - "retrieved_docs": List of relevant Document objects that provide
              context for outline generation. Empty list if retrieval fails.

    Raises:
        Exception: If embedding model loading, retrieval system setup, or 
                  document retrieval fails. Errors are logged and handled gracefully.

    Example:
        >>> state = ReportState(messages=[user_query_message])
        >>> config = RunnableConfig(configurable=Configuration(...))
        >>> result = initial_retrieval(state, config=config)
        >>> print(len(result["retrieved_docs"]))  # Number of retrieved documents
    """
    logger.info("Starting initial document retrieval for outline generation")
    
    try:
        # Get configuration and validate
        configuration = Configuration.from_runnable_config(config)
        if not configuration:
            logger.error("Configuration not found in config")
            raise ValueError("Configuration is required for initial retrieval")
        
        # Extract main query from the last message
        main_query = get_message_text(state.messages[-1])
        logger.info(f"Processing main query: '{main_query}'")
        logger.debug(f"Using embedding model: {configuration.embedding_model}")
        
        # Setup retriever with embedding model
        with retrieval.make_retriever(
            embedding_model=load_embedding_model(model=configuration.embedding_model, host=configuration.ollama_host),   
        ) as retriever:
            logger.debug("Retriever initialized successfully")
            
            # Perform document retrieval
            response = retriever.invoke(main_query, config)
            
            if response:
                logger.info(f"Successfully retrieved {len(response)} documents for outline generation")
                for i, doc in enumerate(response):
                    logger.debug(f"Document {i+1}: {doc.page_content[:100]}... from {doc.metadata.get('source', 'unknown')}")
            else:
                logger.warning("No documents retrieved for initial query")
            
            return {"retrieved_docs": response}
            
    except Exception as e:
        logger.error(f"Error in initial_retrieval: {str(e)}")
        logger.warning("Returning empty document list due to retrieval error")
        return {"retrieved_docs": []}
    

######################################## Generate outline node ########################################


def generate_outline(
    state: ReportState, *, config: RunnableConfig) -> dict[str, Any]:
    """
    Generate an outline based on the user query and retrieved context.
    
    This function creates a structured outline for the report using the initially
    retrieved documents as context. It uses a language model with structured output
    to ensure the outline has the proper format and number of sections specified
    in the report configuration.

    Args:
        state (ReportState): Current report state containing:
            - messages: Conversation messages with the main user query
            - retrieved_docs: Documents retrieved in initial_retrieval step
        config (RunnableConfig): Configuration containing:
            - report_model: Model identifier for outline generation
            - ollama_host: Host URL for Ollama models (if applicable)
            - writing_style: Style preference ("technical" or "general")

    Returns:
        dict[str, Any]: Dictionary containing:
            - "outlines": List of section titles/topics for the report
            - "current_section_index": Index (0) for tracking current section
            - "report_header": Formatted header with report type and title

    Raises:
        Exception: If model loading, outline generation, or prompt formatting fails.
                  Errors are logged and fallback values are returned.

    Example:
        >>> state = ReportState(
        ...     messages=[query_msg], 
        ...     retrieved_docs=[doc1, doc2],
        ... )
        >>> result = generate_outline(state, config=config)
        >>> print(result["outlines"])  # ["Section 1", "Section 2", ...]
    """
    logger.info("Starting outline generation")
    
    try:
        # Get configuration and validate
        configuration = Configuration.from_runnable_config(config)
        if not configuration:
            logger.error("Configuration not found in config")
            raise ValueError("Configuration is required for outline generation")
        
        # Extract query and context information
        main_query = get_message_text(state.messages[-1])
        context_text = "\n\n".join([doc.page_content for doc in state.retrieved_docs])
        
        logger.info(f"Generating outline for query: '{main_query}'")
        logger.info(f"Using {len(state.retrieved_docs)} documents as context")
        logger.info(f"Target sections: {configuration.number_of_parts}, Style: {configuration.writing_style}")
        logger.debug(f"Using report model: {configuration.report_model}")
        logger.debug(f"Context length: {len(context_text)} characters")
        
        # Load model with structured output
        model = load_chat_model(model=configuration.report_model, host=configuration.ollama_host).with_structured_output(Outline)
        logger.debug("Model loaded successfully with structured output")
        
        # Create prompt with context and requirements
        system_prompt = prompts.OUTLINE_SYSTEM_PROMPT.format(
            context = format_docs(state.retrieved_docs),
            number_of_parts = configuration.number_of_parts
        )
        user_prompt = state.messages[-1].content
        messages = [
            ("human", combine_prompts(system=system_prompt, user=user_prompt))
        ]
        logger.debug("Prompt created and formatted")

        # Generate outline using structured output
        generated = cast(Outline, model.invoke(messages, config))
        logger.info(f"Successfully generated outline with {len(generated.entries)} sections")
        
        # Log generated outline entries
        for i, entry in enumerate(generated.entries, 1):
            logger.debug(f"Section {i}: {entry}")
        
        # Initialize report structure and determine writing style label
        style_label = "Technical Report" if configuration.writing_style == "technical" else "General Report"
        report_header = f"{style_label.upper()}\nTITLE: {main_query}\n\n"
        
        logger.info(f"Report header created: {style_label}")
        logger.info("Outline generation completed successfully")
        
        return {
            "outlines": generated.entries,
            "current_section_index": 0,
            "report_header": report_header,
        }
        
    except Exception as e:
        logger.error(f"Error in generate_outline: {str(e)}")
        logger.warning("Generating fallback outline due to error")
        
        # Create fallback outline and header
        fallback_outline = ["Report Section"]
        fallback_header = "TECHNICAL REPORT\nTITLE: Error in outline generation\n\n"
        
        logger.info("Fallback outline created with single section")
        
        return {
            "outlines": fallback_outline,
            "current_section_index": 0,
            "report_header": fallback_header,
            "writing_style": "technical"
        }


######################################## Retrieve for section node ########################################


def retrieve_for_section(
    state: ReportState, *, config: RunnableConfig) -> dict[str, list[Document]]:
    """
    Retrieve documents for the current section being processed.
    
    This function performs targeted document retrieval for the specific section
    currently being processed. It uses the section title as the query to find
    the most relevant documents for that particular part of the report.

    Args:
        state (ReportState): Current report state containing:
            - outlines: List of section titles generated in outline phase
            - current_section_index: Index of the section currently being processed
            - Other state information for context
        config (RunnableConfig): Configuration containing:
            - embedding_model: Model identifier for document embeddings
            - ollama_host: Host URL for Ollama models (if applicable)
            - Retrieval parameters and system configurations

    Returns:
        dict[str, list[Document]]: Dictionary containing:
            - "retrieved_docs": List of Document objects relevant to the current
              section. Empty list if retrieval fails or section is invalid.

    Raises:
        Exception: If embedding model loading, retriever setup, or document
                  retrieval fails. Errors are logged and handled gracefully.

    Example:
        >>> state = ReportState(
        ...     outlines=["Introduction", "Methods", "Results"],
        ...     current_section_index=1
        ... )
        >>> result = retrieve_for_section(state, config=config)
        >>> # Retrieves documents relevant to "Methods" section
    """
    logger.info(f"Starting document retrieval for section index: {state.current_section_index}")
    
    try:
        # Validate section state
        if not state.outlines:
            logger.warning("No outlines available for section retrieval")
            return {"retrieved_docs": []}
        
        if state.current_section_index >= len(state.outlines):
            logger.warning(f"Section index {state.current_section_index} exceeds outline length {len(state.outlines)}")
            return {"retrieved_docs": []}
        
        # Get configuration and current section
        configuration = Configuration.from_runnable_config(config)
        if not configuration:
            logger.error("Configuration not found in config")
            raise ValueError("Configuration is required for section retrieval")
        
        current_section = state.outlines[state.current_section_index]
        logger.info(f"Retrieving documents for section: '{current_section}'")
        logger.debug(f"Using embedding model: {configuration.embedding_model}")
        
        # Setup retriever and perform document retrieval
        with retrieval.make_retriever(
            embedding_model=load_embedding_model(model=configuration.embedding_model, host=configuration.ollama_host),   
        ) as retriever:
            logger.debug("Section retriever initialized successfully")
            
            # Retrieve documents using section title as query
            response = retriever.invoke(current_section, config)
            
            if response:
                logger.info(f"Successfully retrieved {len(response)} documents for section '{current_section}'")
                for i, doc in enumerate(response):
                    logger.debug(f"Section doc {i+1}: {doc.page_content[:100]}... from {doc.metadata.get('source', 'unknown')}")
            else:
                logger.warning(f"No documents retrieved for section '{current_section}'")
            
            return {"retrieved_docs": response}
            
    except Exception as e:
        logger.error(f"Error in retrieve_for_section: {str(e)}")
        logger.warning("Returning empty document list due to section retrieval error")
        return {"retrieved_docs": []}


######################################## Synthesize section node ########################################


def synthesize_section(
    state: ReportState, *, config: RunnableConfig) -> dict[str, Any]:
    """
    Synthesize raw section content from retrieved documents.
    
    This function generates the initial content for the current report section
    using the documents retrieved specifically for that section. It applies the
    appropriate writing style (technical or general) and creates comprehensive
    content that addresses the section topic using the available context.

    Args:
        state (ReportState): Current report state containing:
            - outlines: List of section titles
            - current_section_index: Index of section being processed
            - retrieved_docs: Documents retrieved for current section
            - messages: Original user query for context
        config (RunnableConfig): Configuration containing:
            - report_model: Model identifier for content generation
            - ollama_host: Host URL for Ollama models (if applicable)
            - Content generation parameters
            - writing_style: Style preference ("technical" or "general")

    Returns:
        dict[str, Any]: Dictionary containing:
            - "raw_section_content": Generated content for the current section.
              Empty string if synthesis fails or section is invalid.

    Raises:
        Exception: If model loading, prompt formatting, or content generation fails.
                  Errors are logged and handled with error message content.

    Example:
        >>> state = ReportState(
        ...     outlines=["Introduction", "Methods"],
        ...     current_section_index=0,
        ...     retrieved_docs=[doc1, doc2],
        ... )
        >>> result = synthesize_section(state, config=config)
        >>> print(result["raw_section_content"])  # Generated introduction content
    """
    logger.info(f"Starting content synthesis for section index: {state.current_section_index}")
    
    try:
        # Validate section state
        if not state.outlines:
            logger.warning("No outlines available for content synthesis")
            return {"raw_section_content": ""}

        if state.current_section_index >= len(state.outlines):
            logger.warning(f"Section index {state.current_section_index} exceeds outline length {len(state.outlines)}")
            return {"raw_section_content": ""}

        # Get configuration and section information
        configuration = Configuration.from_runnable_config(config)
        if not configuration:
            logger.error("Configuration not found in config")
            raise ValueError("Configuration is required for content synthesis")

        current_section = state.outlines[state.current_section_index]
        main_query = state.messages[-1].content
        docs_count = len(state.retrieved_docs) if state.retrieved_docs else 0
        
        logger.info(f"Synthesizing content for section: '{current_section}'")
        logger.info(f"Writing style: {configuration.writing_style}")
        logger.info(f"Using {docs_count} retrieved documents")
        logger.debug(f"Using report model: {configuration.report_model}")
        logger.debug(f"Main query context: '{main_query}'")

        # Load content generation model
        model = load_chat_model(model=configuration.report_model, host=configuration.ollama_host)
        logger.debug("Content synthesis model loaded successfully")

        # Select appropriate prompt based on writing style
        prompt = prompts.TECHNICAL_SECTION_SYSTEM_PROMPT if configuration.writing_style == "technical" else prompts.GENERAL_SECTION_SYSTEM_PROMPT
        logger.debug(f"Using {'technical' if configuration.writing_style == 'technical' else 'general'} section prompt")
        
        # Format prompt with context and section information
        formatted_prompt = prompt.format(
            context = format_docs(state.retrieved_docs),
            current_section = current_section,
            main_query = main_query
        )
        messages  = [
            ("human", formatted_prompt)
        ]
        logger.debug("Content synthesis prompt formatted successfully")

        # Generate section content
        response = model.invoke(messages, config)
        synthesized_content = get_message_text(response).strip()
        
        content_length = len(synthesized_content)
        logger.info(f"Successfully synthesized section content ({content_length} characters)")
        logger.debug(f"Content preview: {synthesized_content[:200]}...")

        return {"raw_section_content": synthesized_content}
   
    except Exception as e:
        logger.error(f"Error in synthesize_section: {str(e)}")
        error_content = f"Error synthesizing section: {str(e)}"
        logger.warning(f"Returning error content: {error_content}")
        return {"raw_section_content": error_content}


######################################## Review section node ########################################


def review_section(
    state: ReportState, *, config: RunnableConfig) -> dict[str, Any]:
    """
    Review and polish the raw section content using review logic.
    
    This function takes the raw content generated in synthesize_section and
    applies a review and polishing process to improve quality, coherence,
    and alignment with the overall report objectives. It creates the final
    section content and advances the processing to the next section.

    Args:
        state (ReportState): Current report state containing:
            - outlines: List of section titles
            - current_section_index: Index of section being reviewed
            - raw_section_content: Raw content generated for current section
            - messages: Original user query for context
        config (RunnableConfig): Configuration containing:
            - report_model: Model identifier for content review
            - ollama_host: Host URL for Ollama models (if applicable)

    Returns:
        dict[str, Any]: Dictionary containing:
            - "report": List with single dictionary containing:
                - "title": Section title
                - "content": Final polished section content
            - "current_section_index": Incremented index for next section

    Raises:
        Exception: If model loading, review processing, or content polishing fails.
                  Errors are logged and handled with error section content.

    Example:
        >>> state = ReportState(
        ...     outlines=["Introduction", "Methods"],
        ...     current_section_index=0,
        ...     raw_section_content="Draft introduction content..."
        ... )
        >>> result = review_section(state, config=config)
        >>> print(result["report"][0]["title"])    # "Introduction"  
        >>> print(result["report"][0]["content"])  # Polished content
    """
    logger.info(f"Starting section review for index: {state.current_section_index}")
    
    try:
        # Validate section state
        if not state.outlines:
            logger.warning("No outlines available for section review")
            return {}
        
        if state.current_section_index >= len(state.outlines):
            logger.warning(f"Section index {state.current_section_index} exceeds outline length {len(state.outlines)}")
            return {}
        
        # Get configuration and section information
        configuration = Configuration.from_runnable_config(config)
        if not configuration:
            logger.error("Configuration not found in config")
            raise ValueError("Configuration is required for section review")
        
        current_section = state.outlines[state.current_section_index]
        main_query = get_message_text(state.messages[-1])
        raw_content_length = len(state.raw_section_content) if state.raw_section_content else 0
        
        logger.info(f"Reviewing section: '{current_section}'")
        logger.info(f"Raw content length: {raw_content_length} characters")
        logger.debug(f"Using report model: {configuration.report_model}")
        logger.debug(f"Main query context: '{main_query}'")

        # Load review model
        model = load_chat_model(model=configuration.report_model, host=configuration.ollama_host)
        logger.debug("Section review model loaded successfully")
        
        context_docs = format_docs(state.retrieved_docs)
        logger.debug(f"Context formatted as: {context_docs[:200]}...")

        # Format review prompt with context
        prompt = prompts.REVIEW_SYSTEM_PROMPT.format(
            main_query = main_query,
            current_section = current_section,
            context = context_docs,
            draft_section = state.raw_section_content,
        )
        messages  =  [
            ("human", prompt)
        ]
        logger.debug("Review prompt formatted successfully")

        # Generate polished content
        response = model.invoke(messages, config)
        polished_content = get_message_text(response).strip()
        
        polished_length = len(polished_content)
        logger.info(f"Section review completed ({polished_length} characters)")
        logger.debug(f"Polished content preview: {polished_content[:200]}...")
        
        # Create final section structure
        final_section = {"title": current_section, "content": polished_content}
        next_index = state.current_section_index + 1
        
        logger.info(f"Section '{current_section}' completed, advancing to index {next_index}")
        logger.debug(f"Final section structure: title='{current_section}', content_length={polished_length}")
        
        return {
            "report": [final_section],
            "current_section_index": next_index
        }
        
    except Exception as e:
        logger.error(f"Error in review_section: {str(e)}")
        
        # Create error section with fallback information
        current_section_title = current_section if 'current_section' in locals() else "Error"
        error_content = f"Error reviewing section: {str(e)}"
        next_index = state.current_section_index + 1
        
        logger.warning(f"Creating error section '{current_section_title}', advancing to index {next_index}")
        
        return {
            "report": [{"title": current_section_title, "content": error_content}],
            "current_section_index": next_index
        }


######################################## Should continue conditional edge ########################################


def should_continue(state: ReportState, *, config: RunnableConfig):
    """
    Determine whether to continue generating sections or complete the report.
    
    This function implements the control logic for the iterative section processing
    workflow. It checks if there are more sections to process based on the current
    section index and the total number of outlined sections.

    Args:
        state (ReportState): Current report state containing:
            - outlines: List of section titles generated in outline phase
            - current_section_index: Index of the next section to process
        config (RunnableConfig): Configuration for the decision process
                                (not actively used but required for signature consistency)

    Returns:
        Langgraph node: Either "retrieve_for_section" to continue processing the next section,
             or END to signal completion of all sections

    Decision Logic:
        - If no outlines exist: END (complete report)
        - If current_section_index >= len(outlines): END (all sections processed)
        - Otherwise: "retrieve_for_section" (continue with next section)

    Example:
        >>> state = ReportState(
        ...     outlines=["Intro", "Methods", "Results"],
        ...     current_section_index=1
        ... )
        >>> should_continue(state, config)  # Returns "retrieve_for_section"
        >>> 
        >>> state.current_section_index = 3
        >>> should_continue(state, config)  # Returns END
    """
    # Log current state for debugging
    outline_count = len(state.outlines) if state.outlines else 0
    current_index = state.current_section_index
    
    logger.info(f"Continuation check: index {current_index}, total sections {outline_count}")
    
    # Determine continuation based on section availability
    if not state.outlines:
        logger.info("No outlines available - completing report")
        return END
    
    if current_index >= len(state.outlines):
        logger.info(f"All sections processed ({current_index}/{outline_count}) - completing report")
        return END
    else:
        next_section = state.outlines[current_index]
        logger.info(f"Continuing to next section ({current_index + 1}/{outline_count}): '{next_section}'")
        return "retrieve_for_section"


######################################## Graph compiler ########################################


def get_report_graph() -> CompiledStateGraph:
    """
    Build and compile the report generation graph.
    
    This function constructs a LangGraph StateGraph that implements a complete
    report generation system with structured workflow management. The graph handles
    outline generation, iterative section processing, content synthesis, and
    review in a coordinated pipeline.

    Returns:
        CompiledStateGraph: A compiled graph ready for execution with:
            - Proper state management and type safety
            - Sequential and conditional workflow transitions
            - Comprehensive error handling throughout the pipeline
            - Support for both technical and general writing styles

    Nodes:
        - initial_retrieval: Gathers documents for outline generation
        - generate_outline: Creates structured report outline with specified sections
        - retrieve_for_section: Fetches targeted documents for current section
        - synthesize_section: Generates raw content from retrieved documents
        - review_section: Polishes and finalizes section content

    Example:
        >>> graph = get_report_graph()
        >>> config = {"configurable": Configuration(...)}
        >>> 
        >>> result = graph.invoke(
        ...     {
        ...         "messages": [user_query],
        ...     }, 
        ...     config=config
        ... )
        >>> print(len(result["report"]))  # Number of completed sections
    """
    logger.info("Building report generation graph")
    
    try:
        # Create StateGraph with proper schemas
        logger.debug("Initializing StateGraph with ReportState and InputState schemas")
        builder = StateGraph(ReportState, input=InputState, config_schema=Configuration)

        # Add nodes following the mermaid diagram flow
        builder.add_node("initial_retrieval", initial_retrieval)
        builder.add_node("generate_outline", generate_outline)
        builder.add_node("retrieve_for_section", retrieve_for_section)
        builder.add_node("synthesize_section", synthesize_section)
        builder.add_node("review_section", review_section)

        # Add edges following the sequential and conditional flow
        builder.add_edge("__start__", "initial_retrieval")
        builder.add_edge("initial_retrieval", "generate_outline")
        builder.add_edge("generate_outline", "retrieve_for_section")
        builder.add_edge("retrieve_for_section", "synthesize_section")    
        builder.add_edge("synthesize_section", "review_section")
        builder.add_conditional_edges("review_section", should_continue)

        # Compile the graph 
        graph = builder.compile()

        logger.info("Successfully compiled report generation graph")
        
        return graph
        
    except Exception as e:
        logger.error(f"Error building report graph: {str(e)}")
        logger.error("Graph compilation failed - check configuration and dependencies")
        raise ValueError(f"Error compiling the report graph: {e}")