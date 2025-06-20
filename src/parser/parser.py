import os
import torch

from src.parser.loaders import iterate_image_caption, iterate_markdown, pdf_parser

from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from typing import Optional
from pathlib import Path

def parser(data_path: str, temp_path: Optional[str], **kwargs) -> list[Document]:
    """Preprocessing pipeline for pdf files. All files in data_path folder are analyzed using marker-pdf [https://github.com/VikParuchuri/marker].
    Files are turned into a markdown file and images are extracted alongside. Text within images is automatically put into the markdown file.
    
    Args:
        data_path (str): Directory in which the pdf files are stored.
        temp_path (str): Specific temporary directory for markdown files and images. Can be left blank.
        **kwargs: Additional keyword arguments.
            disable_clear_temp (bool): If You wish to keep the markdown and images files. Default is False (temp is cleared after execution).
            ignore_image_processing (bool): Ignore the image processing. Default is False.
            ignore_pdf_preprocessing (bool): Skip the PDF parsing step. Default is False.
    
    Returns:
        list[Documents]: List of loaded Langchain documents.
    """
    # Extract kwargs with defaults
    disable_clear_temp = kwargs.get('disable_clear_temp', False)
    ignore_image_processing = kwargs.get('ignore_image_processing', False)
    ignore_pdf_preprocessing = kwargs.get('ignore_pdf_preprocessing', False)
    
    # If no temp directory is given, create one inside the data folder
    if not temp_path:
        temp_path = os.path.join(data_path, 'temp')
    
    # Assert that if PDF preprocessing is ignored, temp_path must exist and not be empty
    if ignore_pdf_preprocessing:
        assert os.path.exists(temp_path), "When ignore_pdf_preprocessing is True, temp_path must exist"
        assert len(os.listdir(temp_path)) > 0, "When ignore_pdf_preprocessing is True, temp_path cannot be empty"
    
    documents = []
    
    # Only perform PDF parsing if not ignored
    if not ignore_pdf_preprocessing:
        pdf_parser(data_path, temp_path)  # Save pdfs as markdown and images into the temp directory
    torch.cuda.empty_cache()
    
    # Process images if not ignored
    if not ignore_image_processing:
        documents.extend(iterate_image_caption(temp_path))
    torch.cuda.empty_cache()
    
    documents.extend(iterate_markdown(temp_path))
    
    # Clear temp directory if not disabled
    if not disable_clear_temp:
        import shutil
        try:
            shutil.rmtree('./data/temp')
            print("temp directory cleared")
        except Exception as e:
            print(f"Error while clearing the temp directory: {e}")
    
    return documents

def simple_parser(data_path: str) -> list[Document]:
    """
    Parses a PDF file using PyMuPDF and returns a list of LangChain Document objects.

    Parameters:
        data_path (str): Path to a directory containing PDF files.

    Returns:
        list[Document]: A list of Document objects, one per page, from all PDFs.

    Raises:
        FileNotFoundError: If the specified PDF file does not exist.
        Exception: If there is an error during PDF parsing.
    
    Example:
        >>> docs = simple_parser("path/to/file.pdf")
        >>> print(docs[0].page_content)
    """
    path = Path(data_path)
    if not path.is_dir():
        raise FileNotFoundError(f"Directory not found: {data_path}")

    documents = []
    pdf_files = list(path.glob("*.pdf"))

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=256,
                                                       chunk_overlap=50,
                                                       length_function=len,
                                                       is_separator_regex=False)

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in directory: {data_path}")

    for pdf_file in pdf_files:
        loader = PyMuPDFLoader(str(pdf_file))
        documents.extend(loader.load())

    chunks = text_splitter.split_documents(documents)

    return chunks