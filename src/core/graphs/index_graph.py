from utils.logging import get_logger
# Configure logger
logger = get_logger(__name__)

from typing import Optional
from pathlib import Path
from tqdm import tqdm

from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

from core import retrieval
from core.configuration import Configuration
from core.states import IndexState, InputIndexState
from core.models import load_embedding_model
from utils.utils import remove_duplicates, make_batch

from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import PyMuPDFLoader
from parser import VisionLoader

######################################## Parse PDF node ########################################

def parse_pdfs(
  state: InputIndexState, *, config: Optional[RunnableConfig] = None
) -> dict[str, str]:
  
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
    #  embeddings=embeddings,
    #  breakpoint_threshold_type="percentile",
    #  min_chunk_size=256,
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
        if configuration.ocr:
          logger.debug("using server parser")
          loader = VisionLoader(
            file_path=str(pdf_file),
            mode = 'single',
            ollama_base_url= configuration.ollama_host,
            ollama_model=configuration.vision_model,
          )
        else:
          logger.debug("distant client not found, falling back to local parser")
          loader = PyMuPDFLoader(
            file_path=str(pdf_file),
            extract_tables='markdown',
            mode= "single"
          )
        
        # Split the Document content into smaller chunks
        chunks = text_splitter.split_documents(loader.load())
        #ensure metadata
        for c in chunks:
          c.metadata={"source": pdf_file}
        # Add them to the list of Documents
        documents.extend(chunks)
        
        logger.debug(f"Successfully processed {pdf_file}, created {len(chunks)} chunks")
        
      except Exception as e:
        logger.error(f"Failed to process file {pdf_file}: {str(e)}")
        continue

    logger.info(f"PDF parsing completed. Total document chunks created: {len(documents)}")
    return {"docs": documents}
  
  except Exception as e:
    logger.error(f"Error in parse_pdfs: {str(e)}")
    raise


######################################## Index Document node ########################################


def index_docs(
  state: IndexState, *, config: Optional[RunnableConfig] = None
) -> dict[str, str]:
  
  logger.info("Starting document indexing process")
  
  try:
    # Get configuration
    configuration = Configuration.from_runnable_config(config)

    if not configuration:
      logger.error("Configuration required but not provided")
      raise ValueError("Configuration required to run index_docs.")
    
    logger.info(f"Using embedding model: {configuration.embedding_model}")
    
    # Prepare document batches
    documents_batch = make_batch(obj=state.docs, size= 20)
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


######################################## Should index conditional edge ########################################

def should_index(state: IndexState, *, config: RunnableConfig) -> str:
  
  if not state.docs:
    logger.info("No documents to index, ending workflow")
    return END
  
  logger.info(f"Found {len(state.docs)} documents, proceeding to indexing")
  return "index_docs"


######################################## Graph compiler ########################################


def get_index_graph() -> CompiledStateGraph:
  
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