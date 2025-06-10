# src/core/application.py

from src.core.db import DatabaseWrapper #, clear, update_database
from src.core.configuration import Configuration
from src.core.retrieval_agent.graph import get_retrieval_graph

class Application:
    """
    Main orchestrator class that loads a BaseConfiguration, initializes components,
    manages user context, and coordinates the lifecycle of the intelligent agent.
    """

    def __init__(self, config: Configuration):
        self._config = config
        self._database_wrapper = DatabaseWrapper()
        self._retrieval_graph = get_retrieval_graph()

    # Public methods
    def get_config(self):
        return self._config
    
    def set_config(self, config: Configuration):
        self._config = config

    def get_database_wrapper(self):
        return self._database_wrapper

    def retrieval_invoke(self, query: str):
        return self._retrieval_graph.invoke(input=query, config={"configurable": self._config.asdict()})

    def retrieval_stream(self, query: str):
        for t in self._retrieval_graph.stream(input=query, config={"configurable": self._config.asdict()}):
            yield t

    # Uncomment the following methods if you want to implement knowledge base population and clearing

    #def populate_knowledge(self):
    #    print("Populating the knowledge base...")
    #    update_database(config=self.config)
    #    print("Knowledge base update complete")
        

    #def clear_knowledge(self):
    #    clear(config=self.config)
