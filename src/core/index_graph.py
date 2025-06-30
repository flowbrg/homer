from typing import Optional
from pathlib import Path
from tqdm import tqdm

from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

from src.core import retrieval
from src.core.configuration import Configuration
from src.core.states import IndexState, InputIndexState
from src.core.models import load_embedding_model
from src.resources.utils import remove_duplicates, make_document_batch

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader


def parse_pdfs(
    state: InputIndexState, *, config: Optional[RunnableConfig] = None
) -> dict[str, str]:
    
    path = Path(state.path)
    if not path.is_dir():
        raise FileNotFoundError(f"Directory not found: {state.path}")

    documents = []
    
    pdf_files = remove_duplicates(
        base = retrieval.get_existing_documents(),
        new = [str(p) for p in list(path.glob("*.pdf"))]
    )
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512,
                                                       chunk_overlap=50,
                                                       length_function=len,
                                                       is_separator_regex=False)

    if not pdf_files:
       print("[info] No new PDF files")

    for pdf_file in tqdm(pdf_files, "Loading files..."):
        # Load the file into a Document object
        loader = PyMuPDFLoader(str(pdf_file))
        # Split the Document content into smaller chunks
        document = text_splitter.split_documents(loader.load())
        # Add them to the list of Documents
        documents.extend(document)

    return {"docs": documents}


def index_docs(
    state: IndexState, *, config: Optional[RunnableConfig] = None
) -> dict[str, str]:
    """Asynchronously index documents in the given state using the configured retriever.

    This function takes the documents from the state, ensures they have a user ID,
    adds them to the retriever's index, and then signals for the documents to be
    deleted from the state.

    Args:
        state (IndexState): The current state containing documents and retriever.
        config (Optional[RunnableConfig]): Configuration for the indexing process.
    """
    configuration = Configuration.from_runnable_config(config)

    if not configuration:
        raise ValueError("Configuration required to run index_docs.")
    
    documents_batch = make_document_batch(documents=state.docs)

    with retrieval.make_retriever(embedding_model=load_embedding_model(model=configuration.embedding_model)) as retriever:
        for batch in tqdm(documents_batch, "Adding document batch..."):
            retriever.add_documents(batch)
        
    return {"docs": "delete"}


def should_index(state: IndexState, *, config: RunnableConfig):
    if not state.docs:
        return END
    return "index_docs"



def get_index_graph() -> CompiledStateGraph:
    builder = StateGraph(IndexState, config_schema=Configuration)
    builder.add_node(parse_pdfs)
    builder.add_node(index_docs)
    builder.add_edge("__start__", "parse_pdfs")
    builder.add_conditional_edges("parse_pdfs", should_index)

    # Finally, we compile it!
    # This compiles it into a graph you can invoke and deploy.
    graph = builder.compile()
    graph.name = "IndexGraph"

    return graph
