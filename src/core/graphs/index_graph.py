from src.utils.logging import get_logger
# Configure logger
logger = get_logger(__name__)

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
from src.utils.utils import remove_duplicates, make_batch
from src.constant import OLLAMA_LOCALHOST

from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import PyMuPDFLoader
from src.parser import VisionLoader


############################## Parse PDF node ##############################

def parse_pdfs(
    state: InputIndexState, *, config: Optional[RunnableConfig] = None
) -> dict[str, str]:
    """
    Parse PDF files from a directory and split them into document chunks.
    
    This function scans a directory for PDF files, loads them using PyMuPDFLoader,
    splits the content into smaller chunks using RecursiveCharacterTextSplitter,
    and returns the processed documents. It filters out already processed files
    to avoid duplicates.
    
    Args:
        state (InputIndexState): State object containing the directory path to scan.
            Must have a 'path' attribute pointing to a valid directory.
        config (Optional[RunnableConfig], optional): Configuration for the parsing process.
            Defaults to None.
    
    Returns:
        dict[str, str]: Dictionary containing the processed documents under the 'docs' key.
            Returns {"docs": documents} where documents is a list of split document chunks.
    
    Raises:
        FileNotFoundError: If the specified directory path does not exist.
        
    Example:
        >>> state = InputIndexState(path="/path/to/pdfs")
        >>> result = parse_pdfs(state)
        >>> print(len(result["docs"]))  # Number of document chunks
    """
    logger.info(f"Starting PDF parsing from directory: {state.path}")
    
    try:

        configuration = Configuration.from_runnable_config(config)

        path = Path(state.path)
        if not path.is_dir():
            logger.error(f"Directory not found: {state.path}")
            raise FileNotFoundError(f"Directory not found: {state.path}")

        documents = []
        
        # Get new PDF files (excluding already processed ones)
        pdf_files = remove_duplicates(
            base=retrieval.get_existing_documents(),
            new=[str(p) for p in list(path.glob("*.pdf"))]
        )
        
        logger.info(f"Found {len(pdf_files)} new PDF files to process")
        
        #embeddings = load_embedding_model(model=Configuration.embedding_model, host=Configuration.ollama_host)

        # Configure text splitter
        #text_splitter = SemanticChunker(
        #    embeddings=embeddings,
        #    breakpoint_threshold_type="percentile",
        #    min_chunk_size=256,
        #)
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False
        )

        if not pdf_files:
            logger.info("No new PDF files to process")
            return {"docs": documents}

        # Process each PDF file
        for pdf_file in tqdm(pdf_files, desc="Loading files..."):
            try:
                logger.debug(f"Processing file: {pdf_file}")
                
                # Load the file into a Document object
                if configuration.ollama_host != OLLAMA_LOCALHOST:
                    logger.info("using server parser")
                    loader = VisionLoader(
                        file_path=str(pdf_file),
                        mode = 'single',
                        ollama_base_url= configuration.ollama_host,
                        ollama_model=configuration.vision_model,
                    )
                else:
                    logger.info("distant client not found, falling back to local parser")
                    loader = PyMuPDFLoader(
                        file_path=str(pdf_file),
                        extract_tables='markdown',
                        mode= "single"
                    )
                
                # Split the Document content into smaller chunks
                document = text_splitter.split_documents(loader.load())
                #ensure metadata
                document[0].metadata={"source": pdf_file}
                # Add them to the list of Documents
                documents.extend(document)
                
                logger.debug(f"Successfully processed {pdf_file}, created {len(document)} chunks")
                
            except Exception as e:
                logger.error(f"Failed to process file {pdf_file}: {str(e)}")
                continue

        logger.info(f"PDF parsing completed. Total document chunks created: {len(documents)}")
        return {"docs": documents}
    
    except Exception as e:
        logger.error(f"Error in parse_pdfs: {str(e)}")
        raise


############################## Index Document node ##############################


def index_docs(
    state: IndexState, *, config: Optional[RunnableConfig] = None
) -> dict[str, str]:
    """
    Index documents into the retrieval system using embeddings.
    
    This function takes documents from the state, processes them in batches,
    and adds them to the configured retriever's index using the specified
    embedding model. After successful indexing, it signals for documents
    to be removed from the state.
    
    Args:
        state (IndexState): The current state containing documents to be indexed.
            Must have a 'docs' attribute containing the list of documents.
        config (Optional[RunnableConfig], optional): Configuration containing
            embedding model settings and other indexing parameters. Defaults to None.
    
    Returns:
        dict[str, str]: Dictionary with 'docs' key set to 'delete' to signal
            that documents should be removed from state after indexing.
    
    Raises:
        ValueError: If configuration is not provided or invalid.
        Exception: If there are errors during document indexing process.
        
    Example:
        >>> state = IndexState(docs=[doc1, doc2, doc3])
        >>> config = RunnableConfig(embedding_model="text-embedding-ada-002")
        >>> result = index_docs(state, config=config)
        >>> print(result)  # {"docs": "delete"}
        
    Note:
        This function uses the configured embedding model to create vector
        representations of the documents for efficient similarity search.
    """
    logger.info("Starting document indexing process")
    
    try:
        # Get configuration
        configuration = Configuration.from_runnable_config(config)

        if not configuration:
            logger.error("Configuration required but not provided")
            raise ValueError("Configuration required to run index_docs.")
        
        logger.info(f"Using embedding model: {configuration.embedding_model}")
        
        # Prepare document batches
        documents_batch = make_batch(list=state.docs, size= 20)
        total_batches = len(documents_batch)
        total_documents = len(state.docs)
        
        logger.info(f"Processing {total_documents} documents in {total_batches} batches")

        # Index documents using the retriever
        with retrieval.make_retriever(
            embedding_model=load_embedding_model(model=configuration.embedding_model)
        ) as retriever:
            
            for i, batch in enumerate(tqdm(documents_batch, desc="Adding document batch..."), 1):
                try:
                    retriever.add_documents(batch)
                    logger.debug(f"Successfully indexed batch {i}/{total_batches} ({len(batch)} documents)")
                    
                except Exception as e:
                    logger.error(f"Failed to index batch {i}/{total_batches}: {str(e)}")
                    raise
        
        logger.info(f"Document indexing completed successfully. Indexed {total_documents} documents")
        return {"docs": "delete"}
    
    except Exception as e:
        logger.error(f"Error in index_docs: {str(e)}")
        raise


def should_index(state: IndexState, *, config: RunnableConfig) -> str:
    """
    Determine whether to proceed with document indexing based on state content.
    
    This function serves as a conditional edge in the state graph, deciding
    whether to continue to the indexing step or end the workflow based on
    whether there are documents to process.
    
    Args:
        state (IndexState): The current state containing documents to check.
            Must have a 'docs' attribute.
        config (RunnableConfig): Configuration for the decision process.
    
    Returns:
        str: Either "index_docs" to proceed with indexing, or END to terminate
            the workflow if no documents are present.
            
    Example:
        >>> state_with_docs = IndexState(docs=[doc1, doc2])
        >>> should_index(state_with_docs, config=config)  # Returns "index_docs"
        >>> 
        >>> empty_state = IndexState(docs=[])
        >>> should_index(empty_state, config=config)  # Returns END
        
    Note:
        This function is used as a conditional edge in the LangGraph workflow
        to control the flow based on the presence of documents.
    """
    if not state.docs:
        logger.info("No documents to index, ending workflow")
        return END
    
    logger.info(f"Found {len(state.docs)} documents, proceeding to indexing")
    return "index_docs"


def get_index_graph() -> CompiledStateGraph:
    """
    Create and compile the document indexing state graph.
    
    This function constructs a LangGraph StateGraph that defines the workflow
    for parsing PDF files and indexing them. The graph includes nodes for
    PDF parsing and document indexing, with conditional logic to determine
    whether indexing should proceed.
    
    Returns:
        CompiledStateGraph: A compiled state graph ready for execution.
            The graph includes:
            - parse_pdfs: Node for parsing PDF files from a directory
            - index_docs: Node for indexing documents into the retrieval system
            - Conditional edge: Determines whether to proceed with indexing
            
    Example:
        >>> graph = get_index_graph()
        >>> result = graph.invoke({"path": "/path/to/pdfs"}, config=config)
        >>> print(f"Indexing completed: {result}")
        
    Graph Flow:
        1. __start__ -> parse_pdfs: Always starts with PDF parsing
        2. parse_pdfs -> should_index: Conditional check for documents
        3. should_index -> index_docs OR END: Based on document availability
        
    Note:
        The graph uses the Configuration schema for type safety and
        validation of configuration parameters.
    """
    logger.info("Building document indexing graph")
    
    try:
        # Create the StateGraph with IndexState and Configuration schema
        builder = StateGraph(IndexState, config_schema=Configuration)
        
        # Add nodes to the graph
        builder.add_node(parse_pdfs)
        builder.add_node(index_docs)
        
        # Define edges
        builder.add_edge("__start__", "parse_pdfs")
        builder.add_conditional_edges("parse_pdfs", should_index)

        # Compile the graph
        graph = builder.compile()
        graph.name = "IndexGraph"
        
        logger.info("Successfully built and compiled IndexGraph")
        return graph
    
    except Exception as e:
        logger.error(f"Error building index graph: {str(e)}")
        raise