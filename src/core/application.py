# src/core/application.py
import asyncio

from langchain_core.messages.human import HumanMessage
from langchain_core.messages import AnyMessage

from src.core.database import DatabaseWrapper
from src.core.configuration import Configuration
from src.core.agents.retrival_graph import get_retrieval_graph

from typing import Any, Dict

class Application:
    """
    Main orchestrator class that loads a BaseConfiguration, initializes components, | {"thread_id": "1"}
    manages user context, and coordinates the lifecycle of the intelligent agent.
    """

    def __init__(self, config: Configuration):
        self._config = config
        self._database_wrapper = DatabaseWrapper()
        self._retrieval_graph = get_retrieval_graph()
        #self._report_graph = get_report_graph()

    def get_config(self):
        return self._config
    
    def set_config(self, config: Configuration):
        self._config = config

    def get_messages(
            self,
            thread_id: int) -> list[AnyMessage]:
        config = {"configurable": self._config.asdict() | {"thread_id": thread_id}}
        values = self._retrieval_graph.get_state(config=config)[0] # Output of get_state is a snapshot state tuple, [0] is the value of the state
        messages = values.get("messages","")
        return messages

    def invoke_retrieval_graph(
            self,
            query: str,
            thread_id: int) -> Dict[str, Any] | Any:
        """Invoke the retrieval graph with a query and thread ID.

        Args:
            query (dict[str, Any] | Any): The query to process.
            thread_id (int): The ID of the thread for context.

        Returns:
            Dict[str, Any] | Any: The result of the retrieval process.
        """
        config = {"configurable": self._config.asdict() | {"thread_id": thread_id}}
        return self._retrieval_graph.invoke(input=HumanMessage(query), config=config)

    def stream_retrieval_graph(
            self,
            query: str,
            thread_id: int) -> Dict[str, Any] | Any:
        config = {"configurable": self._config.asdict() | {"thread_id": thread_id}}
        for message_chunk, metadata in self._retrieval_graph.stream(input=HumanMessage(query), stream_mode="messages", config=config):
            if message_chunk.content and metadata["langgraph_node"] == "respond":
                yield message_chunk.content

    def invoke_report_graph(query: str):
        return "Report generation is not implemented yet."

    # Uncomment the following methods if you want to implement knowledge base population and clearing

    #def populate_knowledge(self):
    #    print("Populating the knowledge base...")
    #    update_database(config=self.config)
    #    print("Knowledge base update complete")
        

    #def clear_knowledge(self):
    #    clear(config=self.config)
