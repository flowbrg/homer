""" Markdown to PDF Converter

Convert dictionaries of Markdown content to PDF using ReportLab.
Used in the report page of the streamlit app to turn the report
agent output into a downloadable PDF.
"""
from utils.logging import get_logger
# Initialize logger
logger = get_logger(__name__)

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.colors import HexColor
import re
from typing import List, Dict, Literal
import os


################################## Constants ##################################


# List of tuples used for flags conversion (pattern, replacement)
_CONVERSIONS = [
  # Bold: **text** or __text__
  (r'\*\*(.*?)\*\*', r'<b>\1</b>'),
  (r'__(.*?)__', r'<b>\1</b>'),
  
  # Italic: *text* or _text_ (avoid matching bold patterns)
  (r'(?<!\*)\*([^*\n]+?)\*(?!\*)', r'<i>\1</i>'),
  (r'(?<!_)_([^_\n]+?)_(?!_)', r'<i>\1</i>'),
  
  # Code: `text`
  (r'`([^`]+?)`', r'<font name="Courier" size="10">\1</font>'),
  
  # Strikethrough: ~~text~~
  (r'~~(.*?)~~', r'<strike>\1</strike>'),
  
  # Headers
  (r'^### (.*?)$', r'<font size="13"><b>\1</b></font><br/>', re.MULTILINE),
  (r'^## (.*?)$', r'<font size="15"><b>\1</b></font><br/>', re.MULTILINE),
  (r'^# (.*?)$', r'<font size="17"><b>\1</b></font><br/>', re.MULTILINE),
  
  # Links: [text](url) -> underlined text
  (r'\[([^\]]+?)\]\([^)]+?\)', r'<u>\1</u>'),
  
  # Line breaks
  (r'\n', '<br/>'),
]

_IGNORED_TOKENS = [
  '<|endoftext|>', '<|startoftext|>', '<pad>', '<unk>', '<s>', '</s>',
  '<mask>', '<cls>', '<sep>', '<|im_start|>', '<|im_end|>', '<text>',
  'assistant', 'human', '<think>', '</think>','</message>','</messages>',
  '<message>','<messages>','</endofturn>','<endofturn>','<startofturn>',
  '<startofturn>',
]


################################ Converter class ##############################


class MarkdownToPDF:
  """Convert list of dictionaries with Markdown content to PDF."""
  
  

  def __init__(self, page_size=letter):
    self.page_size = page_size
    self.styles = self._create_styles()
  
  def _create_styles(self) -> Dict[str, ParagraphStyle]:
    """Create custom paragraph styles."""
    base_styles = getSampleStyleSheet()
    
    return {
      'title': ParagraphStyle(
        'Title',
        parent=base_styles['Heading1'],
        fontSize=18,
        spaceAfter=16,
        spaceBefore=8,
        alignment=TA_CENTER,
        textColor=HexColor('#2C3E50'),
        fontName='Helvetica-Bold'
      ),
      'content': ParagraphStyle(
        'Content',
        parent=base_styles['Normal'],
        fontSize=11,
        spaceAfter=12,
        alignment=TA_JUSTIFY,
        textColor=HexColor('#34495E'),
        leading=14
      )
    }
  
  def _filter_ignored_tokens(self, text: str) -> str:
    """Remove ignored tokens from text."""
    if not text:
      return text

    for token in _IGNORED_TOKENS:
      text = text.replace(token, '')

    return text.strip()

  def _clean_think_tags(self, text: str) -> str:
    #In case the agent is running a reasoning model
    """Remove <think>...</think> blocks from text."""
    return re.sub(pattern=r'<think>.*?</think>',
            repl='',
            string=text,
            flags=re.DOTALL | re.IGNORECASE).strip()
  
  def _markdown_to_reportlab(self, text: str) -> str:
    """Convert Markdown to ReportLab markup."""
    if not text:
      return ""
    
    # Remove ignored tokens first
    text = self._filter_ignored_tokens(text)

    # Remove <think> tags first
    text = self._clean_think_tags(text)
    
    # Escape XML characters
    text = (text.replace('&', '&amp;')
           .replace('<', '&lt;')
           .replace('>', '&gt;'))
    
    for pattern, replacement, *flags in _CONVERSIONS:
      flag = flags[0] if flags else 0
      text = re.sub(pattern, replacement, text, flags=flag)
    
    return text
  
  def generate_pdf(self,
           data: List[Dict[str, str]],
          header: str = "",
           filename: str = "output.pdf",
           output_dir: str = "",) -> str:
    """
    Generate PDF from list of dictionaries.
    
    Args:
      data: List of dicts with 'title' and 'content' keys
      filename: Output PDF filename
      output_dir: Directory to save the PDF (optional, defaults to
            current directory)
      
    Returns:
      Absolute path to created PDF file
    """
    # Create full path
    if output_dir:
      os.makedirs(output_dir, exist_ok=True)
      filepath = os.path.join(output_dir, filename)
    else:
      filepath = filename
      
    doc = SimpleDocTemplate(
      filepath,
      pagesize=self.page_size,
      rightMargin=72,
      leftMargin=72,
      topMargin=72,
      bottomMargin=72
    )
    
    text = []
    text.append(Paragraph(header, self.styles['title']))
    text.append(Spacer(1, 18))
    logger.info(f"Added header: {header}")
    
    for i, item in enumerate(data):
      title = item.get('title', f'Section {i+1}')
      content = item.get('content', '')
      logger.info(f"Added section: {item.get('title', f'Section {i+1}')}")
      if not title and not content:
        continue
      
      # Add title
      if title:
        title_formatted = self._markdown_to_reportlab(title)
        text.append(Paragraph(title_formatted, self.styles['title']))
      
      # Add content
      if content:
        content_formatted = self._markdown_to_reportlab(content)
        if content_formatted:
          text.append(Paragraph(text=content_formatted, 
                      style=self.styles['content']))
      
      # Add spacing between sections (except after last item)
      if i < len(data) - 1:
        text.append(Spacer(1, 24))
    
    doc.build(text)
    return os.path.abspath(filepath)


############################### Interface method ##############################


def dict_to_pdf(data: List[Dict[str, str]],
        output_filename: str = "report.pdf", 
        output_dir: str = None, header: str ="") -> str:
  """
  Simple interface function to convert dictionaries to PDF.
  
  Args:
    data: List of dictionaries with 'title' and 'content' keys
    output_filename: Name of the output PDF file
    output_dir: Directory to save the PDF (optional, defaults to current
          directory)
    
  Returns:
    str: Path to the created PDF file
  """
  converter = MarkdownToPDF()
  logger.info("Starting the conversion")
  return converter.generate_pdf(data=data,
                  header=header,
                  filename=output_filename,
                  output_dir=output_dir)


################################ Example usage ################################


if __name__ == "__main__":
  sample_data = [
    {
      "title": "**Introduction** to Python",
      "content": """This is an **introduction** to Python programming.

<think>This think block should be removed from the PDF</think>

Python is a *high-level* programming language that's great for:
- Web development
- Data science  
- Machine learning

Here's some `code example`: `print("Hello, World!")`

## Getting Started

To start with Python, you need to ~~install~~ download it first.

Visit [Python.org](https://python.org) for more information."""
    },
    {
      "title": "Advanced *Topics*",
      "content": """Here we cover **advanced topics** in Python.

<think>Another think block that should disappear</think>

### Object-Oriented Programming

Python supports __object-oriented programming__ with classes and objects.

```python
class MyClass:
  pass
```

The above `code block` shows a simple class definition."""
    }
  ]

  pdf_path = dict_to_pdf(sample_data, "sample_output.pdf", "reports")
  print(f"PDF created successfully: {pdf_path}")