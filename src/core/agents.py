"""
All agents wrappers
"""

from typing import Literal, Any, Dict, Optional

from langchain_core.messages.human import HumanMessage
from langchain_core.messages import AnyMessage
from langchain_core.runnables import RunnableConfig

from src.core.configuration import Configuration


######################################## Report Agent ########################################


from src.core.graphs.retrieval_graph import get_retrieval_graph


class RetrievalAgent:
    """
    Wrapper class for the retrieval agent.
    """
    def __init__(self):
        self._graph = get_retrieval_graph() # Compile the retrieval agent graph


    def get_messages(
        self,
        configuration: Configuration,
        thread_id: int
    ) -> list[AnyMessage]:
        config = {"configurable": configuration.asdict() | {"thread_id": str(thread_id)}}
        graph_state = self._graph.get_state(config=config) # Output of get_state is a snapshot state tuple
        messages = graph_state.values["messages"] if "messages" in graph_state.values.keys() else []
        # graph_state = (values= {"messages": ...
        return messages

    def stream(
        self,
        query: str,
        configuration: Configuration,
        thread_id: int
    ) -> str | Any:
        """
        Stream the retrieval graph with a query and thread ID.

        Args:
            query (str): The query to process.
            configuration (Configuration(dataclass)): The configuration holding the models, host url and other parameters
            thread_id (int): The ID of the thread for context.

        Yields:
            str | Any: Message chunks of the 'response' node.
        """
        config = {"configurable": configuration.asdict() | {"thread_id": thread_id}}
        input = {
            "messages":[HumanMessage(content=query)]
        }
        for message_chunk, metadata in self._graph.stream(
            input=input,
            stream_mode="messages",
            config=config,
        ):
            if message_chunk.content and metadata["langgraph_node"] == "respond":
                yield message_chunk.content


######################################## Report Agent ########################################


from src.core.graphs.index_graph import get_index_graph


class IndexAgent:
    """
    Wrapper Class for the index graph.
    """
    def __init__(self):
        self._graph = get_index_graph()

    def invoke(self, path: str, configuration: Configuration):
        self._graph.invoke(input={"path": path}, config = {"configurable": configuration.asdict()})


######################################## Report Agent ########################################


from src.core.graphs.report_graph import get_report_graph


class ReportAgent:
    """
    Wrapper Class for the Report graph.
    """
    def __init__(self):
        self._graph = get_report_graph()

    def invoke(
        self,
        query: str,
        writing_style: Optional[Literal["technical", "general"]],
        number_of_parts: Optional[int],
        configuration: Configuration,
    )-> Dict[str, Any]:
        """
        Invoke the report graph with a query and thread ID.

        Args:
            query (str): The query to process.
            writing_style (Optional[Literal["technical", "general"]]): The writing style of the report, either general or technical with a lot of details and precise values.
            number_of_parts (Optional[int]): The approximate number of parts wanted (LLMs are not deterministic so nothing can guarantee this exact number of parts, but it will most likely be a close value)
            configuration (Configuration(dataclass)): The configuration holding the models, host url and other parameters

        Returns:
            Dict[str, Any]: The output state of the agent.
        """
        config = {"configurable": configuration.asdict()}
        input = {
            "messages": query,
            "writing_style": writing_style,
            "number_of_parts": number_of_parts,
        }
        output = self._graph.invoke(input=input, config=config)
        return output["report"]