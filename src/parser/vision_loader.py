from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from pathlib import Path
from typing import Literal, List

from parser import parser

class VisionLoader(BaseLoader):
    
    def __init__(self, 
                 file_path: str, 
                 ollama_model: str,
                 ollama_base_url: str,
                 mode: Literal['single','page'] = 'page',
                ):

        assert Path(file_path).suffix.lower() == '.pdf'
        
        super().__init__()

        self.file_path = file_path
        self.mode = mode
        self.pipeline = parser.PDFToMarkdownPipeline(
            ollama_model=ollama_model,
            ollama_base_url=ollama_base_url,
            enable_validation=False,
            dpi=400  # Higher resolution
        )
        

    def load(self) -> List[Document]:
        result = self.pipeline.convert_pdf(self.file_path)
        if self.mode == 'single':
            return [Document(page_content = "\n\n".join(result.pages))]
        return [Document(page_content = content) for content in result.pages]
    
    def lazy_load(self):
        return "not implemented yet"
    
    async def aload(self):
        return "not implemented yet"
    
    async def alazy_load(self):
        return "not implemented yet"
