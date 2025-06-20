from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pathlib import Path
from tqdm import tqdm

def iterate_markdown(path: str) -> list[Document]:
    """Searches for markdown files recursively from a given directory. Returns a list of Langchain Documents.

    Args:
        path (str):Parent directory of markdown files.
    
    Returns:
        list[Documents]:List of loaded Langchain documents.
    """
    documents = []
    markdown_files = list(Path(path).rglob("*.md"))

    for file_path in tqdm(markdown_files, "loading mardown files"):
        loader = UnstructuredMarkdownLoader(str(file_path))
        #text_splitter=SemanticChunker(embeddings=OllamaEmbeddings(model="nomic-embed-text"),
        #                              breakpoint_threshold_amount=0.7,
        #                              min_chunk_size=256)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=256,
                                                       chunk_overlap=50,
                                                       length_function=len,
                                                       is_separator_regex=False)
        chunks = loader.load_and_split(text_splitter=text_splitter)
        for d in chunks:
            d.metadata["source"] = Path(d.metadata["source"]).stem
            d.metadata["type"] = "Text"
        documents.extend(chunks)
    print("Markdown files loading complete")
    return documents