# src/core/application.py


from langchain_core.messages.human import HumanMessage
from langchain_core.messages import AnyMessage

from src.core.configuration import Configuration
from src.core import database as db
from src.core.agents.retrival_graph import get_retrieval_graph
from src.core.agents.simple_query_graph import get_simple_query_graph

from typing import Any, Dict

class Application:
    """
    Main orchestrator class that loads a BaseConfiguration, initializes components, 
    manages user context, and coordinates the lifecycle of the intelligent agent.
    """

    def __init__(self, config: Configuration):
        self._config = config
        self._retrieval_graph = get_retrieval_graph()
        self._simple_query_graph = get_simple_query_graph()
        #self._report_graph = get_report_graph()

    def get_config(self):
        return self._config
    
    def set_config(self, config: Configuration):
        self._config = config

    def get_config(self):
        return self._current_thread
    
    def set_current_thread(self, thread = int):
        self._current_thread = thread

    def get_messages(
            self,
            thread_id: int) -> list[AnyMessage]:
        config = {"configurable": self._config.asdict() | {"thread_id": str(thread_id)}}
        graph_state = self._retrieval_graph.get_state(config=config) # Output of get_state is a snapshot state tuple
        messages = graph_state.values["messages"] if "messages" in graph_state.values.keys() else []
        return messages

    def invoke_simple_query_graph(
            self,
            query: str) -> Dict[str, Any] | Any:
        """Invoke the retrieval graph with a query and thread ID.

        Args:
            query (dict[str, Any] | Any): The query to process.

        Returns:
            Dict[str, Any] | Any: The result of the retrieval process.
        """
        config = {"configurable": self._config.asdict()}
        output = self._simple_query_graph.invoke(input={"messages":[HumanMessage(content=query)]}, config=config)
        return output["messages"][-1].content

    def stream_retrieval_graph(
            self,
            query: str,
            thread_id: int) -> str | Any:
        """Invoke the retrieval graph with a query and thread ID.

        Args:
            query (dict[str, Any] | Any): The query to process.
            thread_id (int): The ID of the thread for context.

        Yields:
            str | Any: message chunks of the 'response' node.
        """
        config = {"configurable": self._config.asdict() | {"thread_id": thread_id}}
        for message_chunk, metadata in self._retrieval_graph.stream(input={"messages":[HumanMessage(content=query)]}, stream_mode="messages", config=config):
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
