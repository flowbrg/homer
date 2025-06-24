"""
All agents wrappers
"""


from langchain_core.messages.human import HumanMessage
from langchain_core.messages import AnyMessage

from src.core.configuration import Configuration
from src.core.retrieval_graph import get_retrieval_graph

from typing import Any

class RetrievalAgent:
    """
    Wrapper class for the retrieval agent.
    """
    def __init__(self):
        self._graph = get_retrieval_graph()


    def get_messages(
        self,
        configuration: Configuration,
        thread_id: int
    ) -> list[AnyMessage]:
        config = {"configurable": configuration.asdict() | {"thread_id": str(thread_id)}}
        graph_state = self._graph.get_state(config=config) # Output of get_state is a snapshot state tuple
        messages = graph_state.values["messages"] if "messages" in graph_state.values.keys() else []
        return messages

    def stream(
        self,
        query: str,
        configuration: Configuration,
        thread_id: int
    ) -> str | Any:
        """Invoke the retrieval graph with a query and thread ID.

        Args:
            query (dict[str, Any] | Any): The query to process.
            thread_id (int): The ID of the thread for context.

        Yields:
            str | Any: message chunks of the 'response' node.
        """
        config = {"configurable": configuration.asdict() | {"thread_id": thread_id}}
        for message_chunk, metadata in self._graph.stream(
            input={"messages":[HumanMessage(content=query)]},
            stream_mode="messages",
            config=config,
        ):
            if message_chunk.content and metadata["langgraph_node"] == "respond":
                yield message_chunk.content


from src.core.report_graph import get_report_graph

class ReportAgent:
    """
    Wrapper class for the report agent.
    """
    def __init__(self):
        self._graph = get_report_graph()
    
    def invoke(query: str):
        return "Report generation is not implemented yet."
    

from src.core.index_graph import get_index_graph

class IndexAgent:
    """
    Wrapper Class for the index graph.
    """
    def __init__(self):
        self._graph = get_index_graph()

    def invoke(self, path: str):
        self._graph.invoke(input={"path": path})