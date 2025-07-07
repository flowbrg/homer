import fitz
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
import base64
import io
from PIL import Image
from src.logger_config import setup_logging, get_logger
from src.vision_parser.validation import TextValidator, ValidationResult  # Add this import

@dataclass
class ConversionResult:
    """Result of PDF conversion"""
    pages: List[str]  # Markdown content for each page
    metadata: Dict
    success: bool
    errors: List[str] = None
    validation_results: Optional[List[ValidationResult]] = None  # Add validation results

class VisionProcessor:
    """Simple vision processor for PDF content"""
    
    def __init__(self, model_name: str, base_url: str, temperature: float = 0.1):
        self.logger = get_logger(__name__)
        self.model_name = model_name
        self.base_url = base_url
        self.temperature = temperature
        self.chat_model = self._init_ollama(model_name, base_url)
        
        self.logger.info(f"VisionProcessor initialized with model: {model_name}")
    
    def _init_ollama(self, model_name: str, base_url: str) -> ChatOllama:
        """Initialize ChatOllama"""
        try:
            chat_model = ChatOllama(
                model=model_name,
                base_url=base_url,
                temperature=self.temperature
            )
            self.logger.info(f"Connected to Ollama at {base_url}")
            return chat_model
        except Exception as e:
            self.logger.error(f"Failed to initialize ChatOllama: {e}")
            raise ConnectionError(f"Failed to initialize ChatOllama: {e}")
    
    def process_page(self, image_base64: str) -> str:
        """Process page image and return markdown"""
        
        prompt = """Extract and convert all content from this PDF page to clean markdown format.

INSTRUCTIONS:
1. Extract all visible text accurately
2. Convert tables to proper markdown table format with | separators and --- headers
3. Convert mathematical formulas to LaTeX notation ($...$ for inline, $$...$$ for display)
4. Describe any diagrams, charts, or images with appropriate markdown formatting
5. Maintain document structure with proper headers (#, ##, ###)
6. Preserve lists and formatting
7. Ensure LaTeX formulas use correct syntax (e.g., P_i not P_i_i)

Return only the markdown content, no additional commentary."""

        try:
            content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
            ]
            
            message = HumanMessage(content=content)
            response = self.chat_model.invoke([message])
            
            self.logger.debug(f"Vision model response received, length: {len(response.content)}")
            return response.content
            
        except Exception as e:
            self.logger.error(f"Vision processing failed: {e}")
            raise RuntimeError(f"Vision processing failed: {e}")

class Utils:
    """Simple utility functions"""
    
    @staticmethod
    def optimize_image_for_vision(image_data: bytes, max_size: tuple = (1024, 1024)) -> str:
        """Prepare images for vision model processing"""
        try:
            img = Image.open(io.BytesIO(image_data))
            
            # Resize if too large while maintaining aspect ratio
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Convert to base64
            buffer = io.BytesIO()
            img.save(buffer, format='PNG', optimize=True)
            img_str = base64.b64encode(buffer.getvalue()).decode()
            
            return img_str
            
        except Exception as e:
            raise ValueError(f"Failed to optimize image: {e}")
    
    @staticmethod
    def extract_page_image(page: fitz.Page, dpi: int = 300) -> str:
        """Convert PDF page to optimized base64 image"""
        # Create transformation matrix for desired DPI
        zoom = dpi / 72
        matrix = fitz.Matrix(zoom, zoom)
        
        # Render page as image
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        img_data = pix.tobytes("png")
        pix = None
        
        return Utils.optimize_image_for_vision(img_data)

class PDFToMarkdownPipeline:
    """Simple vision-only PDF to Markdown pipeline with validation"""
    
    def __init__(self, ollama_model: str, ollama_base_url: str, dpi: int = 300, 
                 enable_validation: bool = True, validation_threshold: float = 0.65):
        """Initialize the pipeline"""
        self.logger = get_logger(__name__)
        self.dpi = dpi
        self.enable_validation = enable_validation
        
        # Initialize vision processor
        self.vision_processor = VisionProcessor(
            model_name=ollama_model,
            base_url=ollama_base_url
        )
        
        # Initialize validator if enabled
        if self.enable_validation:
            self.validator = TextValidator(overall_threshold=validation_threshold)
            self.logger.info(f"Text validation enabled with threshold: {validation_threshold}")
        
        # Pipeline metadata
        self.metadata = {
            "ollama_model": ollama_model,
            "ollama_base_url": ollama_base_url,
            "dpi": dpi,
            "strategy": "vision_only",
            "validation_enabled": enable_validation
        }
    
    def convert_pdf(self, pdf_path: str) -> ConversionResult:
        """Main conversion pipeline with validation"""
        try:
            pdf_path = Path(pdf_path)
            if not pdf_path.exists():
                return ConversionResult(
                    pages=[],
                    metadata=self.metadata,
                    success=False,
                    errors=[f"PDF file not found: {pdf_path}"]
                )
            
            # Open PDF document
            with fitz.open(str(pdf_path)) as doc:
                pages_markdown = []
                errors = []
                
                self.logger.info(f"Processing PDF with {doc.page_count} pages...")
                
                for page_num in range(doc.page_count):
                    try:
                        page = doc[page_num]
                        self.logger.info(f"Processing page {page_num + 1}/{doc.page_count}")
                        
                        # Convert page to image
                        page_image = Utils.extract_page_image(page, self.dpi)
                        
                        # Process with vision model
                        markdown = self.vision_processor.process_page(page_image)
                        pages_markdown.append(markdown)
                        
                    except Exception as e:
                        error_msg = f"Error processing page {page_num + 1}: {str(e)}"
                        errors.append(error_msg)
                        self.logger.warning(f"Warning: {error_msg}")
                        pages_markdown.append(f"<!-- Error processing page {page_num + 1}: {str(e)} -->")

            # Run validation if enabled
            validation_results = None
            if self.enable_validation and pages_markdown:
                self.logger.info("Running text validation...")
                validation_results = self.validator.validate_document(str(pdf_path), pages_markdown)
                
                # Log validation summary
                passed_count = sum(1 for r in validation_results if r.passed_threshold)
                avg_score = sum(r.validation_score for r in validation_results) / len(validation_results)
                self.logger.info(f"Validation complete: {passed_count}/{len(validation_results)} pages passed "
                               f"(avg score: {avg_score:.3f})")

            # Compile metadata
            result_metadata = {
                **self.metadata,
                "source": str(pdf_path),
                "total_pages": len(pages_markdown),
                "successful_pages": len(pages_markdown) - len(errors)
            }
            
            # Add validation metadata if available
            if validation_results:
                passed_validation = sum(1 for r in validation_results if r.passed_threshold)
                avg_validation_score = sum(r.validation_score for r in validation_results) / len(validation_results)
                result_metadata.update({
                    "validation": {
                        "pages_passed": passed_validation,
                        "total_pages": len(validation_results),
                        "pass_rate": passed_validation / len(validation_results),
                        "average_score": avg_validation_score
                    }
                })
            
            return ConversionResult(
                pages=pages_markdown,
                metadata=result_metadata,
                success=len(errors) == 0,
                errors=errors if errors else None,
                validation_results=validation_results
            )
            
        except Exception as e:
            return ConversionResult(
                pages=[],
                metadata=self.metadata,
                success=False,
                errors=[f"Pipeline error: {str(e)}"]
            )
    
    def save_results(self, result: ConversionResult, output_dir: str = "./output") -> List[Path]:
        """Save conversion results to files including validation report"""
        self.logger.info(f"Saving results to {output_dir}")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        saved_files = []
        
        # Save each page as separate markdown file
        for i, page_content in enumerate(result.pages):
            page_file = output_path / f"page_{i+1:03d}.md"
            page_file.write_text(page_content, encoding='utf-8')
            saved_files.append(page_file)
            self.logger.debug(f"Saved page {i+1} to {page_file}")
        
        # Save combined document
        combined_content = "\n\n---\n\n".join(result.pages)
        combined_file = output_path / "combined_document.md"
        combined_file.write_text(combined_content, encoding='utf-8')
        saved_files.append(combined_file)
        self.logger.debug(f"Saved combined document to {combined_file}")
        
        # Save metadata
        import json
        metadata_file = output_path / "conversion_metadata.json"
        
        with open(metadata_file, 'w') as f:
            json.dump(result.metadata, f, indent=2)
        saved_files.append(metadata_file)
        self.logger.debug(f"Saved metadata to {metadata_file}")
        
        # Save validation report if available
        if result.validation_results:
            validation_file = output_path / "validation_report.json"
            validation_data = []
            
            for i, val_result in enumerate(result.validation_results):
                validation_data.append({
                    "page": i + 1,
                    "passed": val_result.passed_threshold,
                    "score": val_result.validation_score,
                    "word_overlap": val_result.word_overlap_ratio,
                    "char_overlap": val_result.char_overlap_ratio,
                    "sentence_overlap": val_result.sentence_overlap_ratio,
                    "semantic_similarity": val_result.semantic_similarity,
                    "extracted_words": val_result.extracted_word_count,
                    "llm_words": val_result.llm_word_count,
                    "missing_words": list(val_result.missing_words),
                    "extra_content_ratio": val_result.extra_content_ratio
                })
            
            with open(validation_file, 'w') as f:
                json.dump({
                    "summary": result.metadata.get("validation", {}),
                    "page_results": validation_data
                }, f, indent=2)
            
            saved_files.append(validation_file)
            self.logger.debug(f"Saved validation report to {validation_file}")
        
        self.logger.info(f"Successfully saved {len(saved_files)} files to {output_dir}")
        
        return saved_files


# Convenience function for simple usage
def convert_pdf_to_markdown(pdf_path: str,
                                  ollama_model: str = "llama3.2-vision:11b", 
                                  ollama_base_url: str = "http://localhost:11434",
                                  output_dir: str = "./output",
                                  dpi: int = 300,
                                  log_level: str = "INFO",
                                  enable_validation: bool = True,
                                  validation_threshold: float = 0.6) -> ConversionResult:
    """
    Simple function to convert a PDF to markdown using vision-only approach with validation
    
    Args:
        pdf_path: Path to the PDF file
        ollama_model: Ollama model name for vision processing
        ollama_base_url: Ollama server URL
        output_dir: Directory to save output files
        dpi: Image resolution for processing
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        enable_validation: Whether to run text validation
        validation_threshold: Minimum validation score to pass (0-1)
    
    Returns:
        ConversionResult with pages, metadata, and validation results
    """
    # Set up logging
    logger = setup_logging(log_level=log_level)
    
    pipeline = PDFToMarkdownPipeline(
        ollama_model, 
        ollama_base_url, 
        dpi, 
        enable_validation=enable_validation,
        validation_threshold=validation_threshold
    )
    result = pipeline.convert_pdf(pdf_path)
    
    if result.success:
        saved_files = pipeline.save_results(result, output_dir)
        print(f"Conversion completed! Files saved to {output_dir}")
        for file in saved_files:
            print(f"  - {file}")
        
        # Print validation summary if available
        if result.validation_results:
            passed_count = sum(1 for r in result.validation_results if r.passed_threshold)
            total_pages = len(result.validation_results)
            avg_score = sum(r.validation_score for r in result.validation_results) / total_pages
            
            print(f"\nValidation Summary:")
            print(f"  ✅ Passed: {passed_count}/{total_pages} pages")
            print(f"  📊 Average score: {avg_score:.3f}")
            
            # Show pages that failed validation
            failed_pages = [i+1 for i, r in enumerate(result.validation_results) 
                          if not r.passed_threshold]
            if failed_pages:
                print(f"  ⚠️  Failed pages: {failed_pages}")
    else:
        print("Conversion failed:")
        for error in result.errors or []:
            print(f"  - {error}")
    
    return result