"""
All agents wrappers
"""

from typing import Literal, Any, Dict, Optional

from langchain_core.messages.human import HumanMessage
from langchain_core.messages import AnyMessage

from core.configuration import Configuration

from langgraph.graph.state import CompiledStateGraph


################################## BaseAgent ##################################


class BaseAgent:
  """
  Base class for all agents.
  """
  def __init__(self, graph: CompiledStateGraph):
    self._graph = graph


################################ Rerieval Agent ###############################


from core.graphs.retrieval_graph import get_retrieval_graph


class RetrievalAgent(BaseAgent):
  """
  Wrapper class for the retrieval agent.
  """
  def __init__(self):
    super().__init__(get_retrieval_graph()) # Compile the retrieval agent graph


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
      configuration (Configuration(dataclass)): The configuration holding the
        models, host url and other parameters.
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
      stream_mode="messages", # Stream the "messages" value of the graph state
      config=config,
    ):
      if message_chunk.content and metadata["langgraph_node"] == "respond":
        yield message_chunk.content


################################# Index Agent #################################


from core.graphs.index_graph import get_index_graph


class IndexAgent(BaseAgent):
  """
  Wrapper Class for the index graph.
  """
  def __init__(self):
    super().__init__(get_index_graph()) # Compile the retrieval agent graph

  def invoke(self, path: str, configuration: Configuration):
    self._graph.invoke(input={"path": path}, config = {"configurable": configuration.asdict()})


################################ Report Agent #################################


from core.graphs.report_graph import get_report_graph


class ReportAgent(BaseAgent):
  """
  Wrapper Class for the Report graph.
  """
  def __init__(self):
    super().__init__(get_report_graph())

  def invoke(
    self,
    query: str,
    configuration: Configuration,
  )-> Dict[str, Any]:
    """
    Invoke the report graph with a query and thread ID.

    Args:
      query (str): The query to process.
      writing_style (Optional[Literal["technical", "general"]]): The writing
        style of the report, either general or technical with a lot of details
        and precise values.
      number_of_parts (Optional[int]): The approximate number of parts wanted
        (LLMs are not deterministic so nothing can guarantee this exact number
        of parts, but it will most likely be a close value)
      configuration (Configuration(dataclass)): The configuration holding the
        models, host url and other parameters

    Returns:
      Dict[str, Any]: The output state of the agent.
    """
    config = {"configurable": configuration.asdict()}
    input = {
      "messages": query,
    }
    output = self._graph.invoke(input=input, config=config)
    return output["report"], output["report_header"]