from pydantic import BaseModel
from typing import cast, Any
import logging

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

from src.core.states import ReportState, InputState
from src.core.configuration import Configuration
from src.core import retrieval
from src.resources.utils import format_docs, get_message_text, format_messages
from src.core.models import load_chat_model, load_embedding_model
from src.resources import prompts


# Set up logging
logger = logging.getLogger(__name__)


class OutlineEntry(BaseModel):
    """Represents a single entry in the outline with a title and summary."""
    title: str
    summary: str


class Outline(BaseModel):
    """Represents an outline for a report with multiple entries."""
    entries: list[OutlineEntry]


def generate_outline(
    state: ReportState, *, config: RunnableConfig
) -> dict[str, list[dict[str, str]]]:
    """Generate an outline based on the user query."""
    try:
        configuration = Configuration.from_runnable_config(config)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", prompts.OUTLINE_SYSTEM_PROMPT),
            ("human", "{query}")
        ])
        
        retrieved_docs = format_docs(state.retrieved_docs)
        
        model = load_chat_model(model=configuration.outline_model).with_structured_output(Outline)
        
        message_value = prompt.invoke({
            "query": state.messages[-1].content,
            "number_of_parts": configuration.number_of_parts,
            "context": retrieved_docs,
        }, config)
        
        generated_outlines = cast(Outline, model.invoke(message_value, config))
        return {
            "outlines": generated_outlines.model_dump()["entries"],
            "current_section_index": 0  # Reset the index
        }
    except Exception as e:
        logger.error(f"Error in generate_outline: {str(e)}")
        # Return a default outline on error
        return {
            "outlines": [{"title": "Report Section", "summary": "Unable to generate outline"}],
            "current_section_index": 0
        }

def retrieve(
    state: ReportState, *, config: RunnableConfig
) -> dict[str, list[Document]]:
    """Retrieve documents based on the current section or query."""
    try:
        configuration = Configuration.from_runnable_config(config)
        
        # Determine what to search for
        if state.outlines and state.current_section_index < len(state.outlines):
            current_section = state.outlines[state.current_section_index]
            input_query = "\n".join([current_section["title"], current_section["summary"]])
        else:
            input_query = get_message_text(state.messages[-1])
        
        with retrieval.make_retriever(
            embedding_model=load_embedding_model(model=configuration.embedding_model),   
        ) as retriever:
            response = retriever.invoke(input_query, config)
            return {"retrieved_docs": response}
    except Exception as e:
        logger.error(f"Error in retrieve: {str(e)}")
        return {"retrieved_docs": []}


def generate_section(
    state: ReportState, *, config: RunnableConfig
) -> dict[str, Any]:
    """Generate a section of the report."""
    try:
        configuration = Configuration.from_runnable_config(config)
        
        if not state.outlines or state.current_section_index >= len(state.outlines):
            return {}
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", prompts.SECTION_SYSTEM_PROMPT),
        ])
        
        model = load_chat_model(model=configuration.response_model)
        retrieved_docs = format_docs(state.retrieved_docs)
        
        # Get current section safely
        current_section = state.outlines[state.current_section_index].copy()
        
        message_value = prompt.invoke({
            "title": current_section["title"],
            "summary": current_section["summary"],
            "previous_sections_text": "\n".join(
                section.get("content", "") for section in state.report
            ) if state.report else "This is the first section. There are no previous sections.",
            "context": retrieved_docs,
        }, config)
        
        response = model.invoke(message_value, config)
        current_section["content"] = get_message_text(response)
        
        return {
            "report": [current_section],
            "current_section_index": state.current_section_index + 1
        }
    except Exception as e:
        logger.error(f"Error in generate_section: {str(e)}")
        return {
            "report": {"title": "Error", "summary": "", "content": f"Error generating section: {str(e)}"},
            "current_section_index": state.current_section_index + 1
        }


def should_continue(state: ReportState, *, config: RunnableConfig):
    """Determine whether to continue generating sections."""
    if not state.outlines or state.current_section_index >= len(state.outlines):
        return END
    else:
        return "retrieve"


def get_report_graph() -> CompiledStateGraph:
    """Build and compile the report generation graph."""
    builder = StateGraph(ReportState, input=InputState, config_schema=Configuration)
    
    # Add nodes
    builder.add_node("initial_retrieval", retrieve)
    builder.add_node("generate_outline", generate_outline)
    builder.add_node("retrieve", retrieve)
    builder.add_node("generate_section", generate_section)
    
    # Add edges
    builder.add_edge("__start__", "initial_retrieval")
    builder.add_edge("initial_retrieval", "generate_outline")
    builder.add_edge("generate_outline", "retrieve")
    builder.add_edge("retrieve", "generate_section")
    builder.add_conditional_edges("generate_section", should_continue)
    
    # Compile the graph
    graph = builder.compile(
        interrupt_before=[],
        interrupt_after=[],
    )
    graph.name = "ReportGraph"
    return graph
