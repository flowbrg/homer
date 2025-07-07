import re
import fitz
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass
from difflib import SequenceMatcher
from collections import Counter
import unicodedata
from src.logger_config import get_logger

@dataclass
class ValidationResult:
    """Result of text validation comparison"""
    word_overlap_ratio: float          # Ratio of overlapping words
    char_overlap_ratio: float          # Ratio of overlapping characters
    sentence_overlap_ratio: float      # Ratio of overlapping sentences
    semantic_similarity: float         # Normalized edit distance
    extracted_word_count: int          # Words in PyMuPDF extraction
    llm_word_count: int                # Words in LLM output
    missing_words: Set[str]            # Important words missing from LLM
    extra_content_ratio: float         # How much extra content LLM added
    validation_score: float            # Overall validation score (0-1)
    passed_threshold: bool             # Whether it passed the threshold
    metadata: Dict                     # Additional validation info

class TextValidator:
    """Validates LLM output against PyMuPDF text extraction"""
    
    def __init__(self, 
                 word_threshold: float = 0.7,
                 char_threshold: float = 0.6,
                 sentence_threshold: float = 0.5,
                 overall_threshold: float = 0.65):
        """
        Initialize validator with thresholds
        
        Args:
            word_threshold: Minimum word overlap ratio (0-1)
            char_threshold: Minimum character overlap ratio (0-1) 
            sentence_threshold: Minimum sentence overlap ratio (0-1)
            overall_threshold: Minimum overall validation score (0-1)
        """
        self.logger = get_logger(__name__)
        self.word_threshold = word_threshold
        self.char_threshold = char_threshold
        self.sentence_threshold = sentence_threshold
        self.overall_threshold = overall_threshold
        
    def validate_page(self, page: fitz.Page, llm_markdown: str) -> ValidationResult:
        """Validate LLM output against PyMuPDF extraction for a single page"""
        
        # Extract text from PyMuPDF
        extracted_text = self._extract_clean_text(page)
        
        # Clean LLM markdown (remove markdown formatting, image descriptions)
        llm_text = self._clean_llm_output(llm_markdown)
        
        # Perform various overlap calculations
        word_overlap = self._calculate_word_overlap(extracted_text, llm_text)
        char_overlap = self._calculate_char_overlap(extracted_text, llm_text)
        sentence_overlap = self._calculate_sentence_overlap(extracted_text, llm_text)
        semantic_sim = self._calculate_semantic_similarity(extracted_text, llm_text)
        
        # Calculate word counts
        extracted_words = self._get_words(extracted_text)
        llm_words = self._get_words(llm_text)
        
        # Find missing important words
        missing_words = self._find_missing_important_words(extracted_words, llm_words)
        
        # Calculate extra content ratio
        extra_content = self._calculate_extra_content_ratio(extracted_text, llm_text)
        
        # Calculate overall validation score
        validation_score = self._calculate_validation_score(
            word_overlap, char_overlap, sentence_overlap, semantic_sim
        )
        
        # Check if it passes threshold
        passed = validation_score >= self.overall_threshold
        
        result = ValidationResult(
            word_overlap_ratio=word_overlap,
            char_overlap_ratio=char_overlap,
            sentence_overlap_ratio=sentence_overlap,
            semantic_similarity=semantic_sim,
            extracted_word_count=len(extracted_words),
            llm_word_count=len(llm_words),
            missing_words=missing_words,
            extra_content_ratio=extra_content,
            validation_score=validation_score,
            passed_threshold=passed,
            metadata={
                "extracted_text_length": len(extracted_text),
                "llm_text_length": len(llm_text),
                "thresholds": {
                    "word": self.word_threshold,
                    "char": self.char_threshold,
                    "sentence": self.sentence_threshold,
                    "overall": self.overall_threshold
                }
            }
        )
        
        self._log_validation_result(result)
        return result
    
    def _extract_clean_text(self, page: fitz.Page) -> str:
        """Extract and clean text from PyMuPDF"""
        text = page.get_text()
        
        # Basic cleaning
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = text.strip()
        
        # Remove common OCR artifacts
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\'\"\/\\\+\=\*\&\%\$\#\@]', '', text)
        
        return text
    
    def _clean_llm_output(self, markdown: str) -> str:
        """Clean LLM markdown output to extract just the text content"""
        
        # Remove markdown formatting
        text = re.sub(r'#+\s*', '', markdown)  # Headers
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Bold
        text = re.sub(r'\*(.*?)\*', r'\1', text)  # Italic
        text = re.sub(r'`(.*?)`', r'\1', text)  # Code
        text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)  # Links
        
        # Remove table formatting
        text = re.sub(r'\|', ' ', text)  # Table separators
        text = re.sub(r'^[-\s]+$', '', text, flags=re.MULTILINE)  # Table header separators
        
        # Remove LaTeX math (but keep the content)
        text = re.sub(r'\$\$(.*?)\$\$', r'\1', text, flags=re.DOTALL)  # Display math
        text = re.sub(r'\$(.*?)\$', r'\1', text)  # Inline math
        
        # Remove image descriptions (common patterns)
        text = re.sub(r'\*\*Diagram Description:\*\*.*?(?=\n\n|\n#|\Z)', '', text, flags=re.DOTALL)
        text = re.sub(r'Figure \d+[:\-].*?(?=\n\n|\n#|\Z)', '', text, flags=re.DOTALL)
        text = re.sub(r'!\[.*?\]\(.*?\)', '', text)  # Image tags
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def _get_words(self, text: str) -> List[str]:
        """Extract words from text, normalized"""
        # Convert to lowercase and extract words
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Remove very short words and numbers-only
        words = [w for w in words if len(w) > 2 and not w.isdigit()]
        
        return words
    
    def _calculate_word_overlap(self, extracted: str, llm_text: str) -> float:
        """Calculate word overlap ratio"""
        extracted_words = set(self._get_words(extracted))
        llm_words = set(self._get_words(llm_text))
        
        if not extracted_words:
            return 1.0 if not llm_words else 0.0
        
        overlap = len(extracted_words & llm_words)
        return overlap / len(extracted_words)
    
    def _calculate_char_overlap(self, extracted: str, llm_text: str) -> float:
        """Calculate character-level overlap using longest common subsequence"""
        if not extracted:
            return 1.0 if not llm_text else 0.0
        
        # Use difflib for sequence matching
        matcher = SequenceMatcher(None, extracted.lower(), llm_text.lower())
        matching_blocks = matcher.get_matching_blocks()
        
        total_matching = sum(block.size for block in matching_blocks)
        return total_matching / len(extracted)
    
    def _calculate_sentence_overlap(self, extracted: str, llm_text: str) -> float:
        """Calculate sentence-level overlap"""
        # Split into sentences
        extracted_sentences = self._get_sentences(extracted)
        llm_sentences = self._get_sentences(llm_text)
        
        if not extracted_sentences:
            return 1.0 if not llm_sentences else 0.0
        
        # Check how many extracted sentences have similar matches in LLM output
        matches = 0
        for ext_sent in extracted_sentences:
            ext_words = set(self._get_words(ext_sent))
            if len(ext_words) < 3:  # Skip very short sentences
                continue
                
            for llm_sent in llm_sentences:
                llm_words = set(self._get_words(llm_sent))
                if len(ext_words & llm_words) / len(ext_words) > 0.6:  # 60% word overlap
                    matches += 1
                    break
        
        return matches / len(extracted_sentences) if extracted_sentences else 0.0
    
    def _get_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def _calculate_semantic_similarity(self, extracted: str, llm_text: str) -> float:
        """Calculate semantic similarity using normalized edit distance"""
        if not extracted:
            return 1.0 if not llm_text else 0.0
            
        # Use SequenceMatcher for similarity
        matcher = SequenceMatcher(None, extracted.lower(), llm_text.lower())
        return matcher.ratio()
    
    def _find_missing_important_words(self, extracted_words: List[str], llm_words: List[str]) -> Set[str]:
        """Find important words that are missing from LLM output"""
        extracted_set = set(extracted_words)
        llm_set = set(llm_words)
        
        missing = extracted_set - llm_set
        
        # Filter for "important" words (longer words, not common stopwords)
        stopwords = {'the', 'and', 'are', 'for', 'this', 'that', 'with', 'from', 'they', 'have', 'been', 'will', 'their', 'said', 'each', 'which', 'what', 'there', 'more', 'can', 'may', 'also', 'some', 'time', 'very', 'when', 'much', 'new', 'two', 'way', 'who', 'its', 'now', 'find', 'long', 'down', 'day', 'did', 'get', 'come', 'made', 'part', 'over'}
        
        important_missing = {word for word in missing 
                           if len(word) > 4 and word not in stopwords}
        
        return important_missing
    
    def _calculate_extra_content_ratio(self, extracted: str, llm_text: str) -> float:
        """Calculate how much extra content the LLM added"""
        if not extracted:
            return 1.0 if llm_text else 0.0
        
        extra_length = max(0, len(llm_text) - len(extracted))
        return extra_length / len(extracted)
    
    def _calculate_validation_score(self, word_overlap: float, char_overlap: float, 
                                  sentence_overlap: float, semantic_sim: float) -> float:
        """Calculate weighted overall validation score"""
        
        # Weighted combination of different metrics
        weights = {
            'word': 0.4,      # Most important - content words should match
            'char': 0.2,      # Character-level accuracy
            'sentence': 0.3,  # Sentence-level structure
            'semantic': 0.1   # Overall similarity
        }
        
        score = (
            word_overlap * weights['word'] +
            char_overlap * weights['char'] +
            sentence_overlap * weights['sentence'] +
            semantic_sim * weights['semantic']
        )
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _log_validation_result(self, result: ValidationResult):
        """Log validation results"""
        status = "✅ PASSED" if result.passed_threshold else "❌ FAILED"
        
        self.logger.info(f"Text validation {status} (score: {result.validation_score:.3f})")
        self.logger.debug(f"  Word overlap: {result.word_overlap_ratio:.3f}")
        self.logger.debug(f"  Char overlap: {result.char_overlap_ratio:.3f}")
        self.logger.debug(f"  Sentence overlap: {result.sentence_overlap_ratio:.3f}")
        self.logger.debug(f"  Semantic similarity: {result.semantic_similarity:.3f}")
        
        if result.missing_words:
            self.logger.debug(f"  Missing important words: {list(result.missing_words)[:5]}")
        
        if result.extra_content_ratio > 0.5:
            self.logger.debug(f"  Extra content ratio: {result.extra_content_ratio:.3f}")

    def validate_document(self, pdf_path: str, pages_markdown: List[str]) -> List[ValidationResult]:
        """Validate entire document"""
        results = []
        
        with fitz.open(pdf_path) as doc:
            for page_num, markdown in enumerate(pages_markdown):
                if page_num < doc.page_count:
                    page = doc[page_num]
                    result = self.validate_page(page, markdown)
                    results.append(result)
        
        # Log summary
        passed_count = sum(1 for r in results if r.passed_threshold)
        avg_score = sum(r.validation_score for r in results) / len(results)
        
        self.logger.info(f"Document validation: {passed_count}/{len(results)} pages passed "
                        f"(avg score: {avg_score:.3f})")
        
        return results