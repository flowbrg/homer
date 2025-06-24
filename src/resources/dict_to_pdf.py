from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
import re
from typing import List, Dict, Any
import os

def dict_to_pdf(data: List[Dict[str, str]], output_filename: str = "report.pdf", 
                page_size=letter, add_page_breaks: bool = False) -> str:
    """
    Convert a list of dictionaries to a formatted PDF file.
    
    Args:
        data: List of dictionaries containing 'title', 'summary', and 'content' keys
        output_filename: Name of the output PDF file
        page_size: Page size (letter, A4, etc.)
        add_page_breaks: Whether to add page breaks between sections
    
    Returns:
        str: Path to the created PDF file
    """
    
    # Create the PDF document
    doc = SimpleDocTemplate(output_filename, pagesize=page_size,
                          rightMargin=72, leftMargin=72,
                          topMargin=72, bottomMargin=18)
    
    # Get the default stylesheet and create custom styles
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        spaceAfter=12,
        alignment=TA_CENTER,
        textColor='#2C3E50'
    )
    
    summary_style = ParagraphStyle(
        'CustomSummary',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=8,
        leftIndent=20,
        rightIndent=20,
        alignment=TA_JUSTIFY,
        textColor='#34495E',
        fontName='Helvetica-Oblique'
    )
    
    content_style = ParagraphStyle(
        'CustomContent',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=12,
        alignment=TA_JUSTIFY,
        textColor='#2C3E50'
    )
    
    # Build the story (content elements)
    story = []
    
    for i, item in enumerate(data):
        # Extract data with fallback values
        title = item.get('title', f'Section {i+1}')
        summary = item.get('summary', '')
        content = item.get('content', '')
        
        # Clean content by removing <think> tags and their content
        content = clean_content(content)
        
        # Add title
        story.append(Paragraph(title, title_style))
        story.append(Spacer(1, 6))
        
        # Summary is skipped - not included in PDF
        
        # Add content if it exists
        if content.strip():
            # Split content into paragraphs for better formatting
            paragraphs = content.split('\n\n')
            for paragraph in paragraphs:
                if paragraph.strip():
                    story.append(Paragraph(paragraph.strip(), content_style))
                    story.append(Spacer(1, 6))
        
        # Add spacing between sections
        if i < len(data) - 1:  # Don't add extra space after last item
            story.append(Spacer(1, 20))
            if add_page_breaks:
                story.append(PageBreak())
    
    # Build the PDF
    doc.build(story)
    
    return os.path.abspath(output_filename)

def clean_content(content: str) -> str:
    """
    Clean content by removing <think> tags and their content, and other unwanted elements.
    
    Args:
        content: Raw content string
    
    Returns:
        str: Cleaned content
    """
    # Remove <think> blocks
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
    
    # Remove extra whitespace and normalize line breaks
    content = re.sub(r'\n\s*\n', '\n\n', content)
    content = content.strip()
    
    return content

def advanced_dict_to_pdf(data: List[Dict[str, Any]], output_filename: str = "advanced_report.pdf",
                        title: str = None, author: str = None, subject: str = None) -> str:
    """
    Advanced version with more customization options and metadata support.
    
    Args:
        data: List of dictionaries with report sections
        output_filename: Name of the output PDF file
        title: Document title (appears in PDF metadata and optionally as cover)
        author: Document author
        subject: Document subject
    
    Returns:
        str: Path to the created PDF file
    """
    
    doc = SimpleDocTemplate(output_filename, pagesize=letter,
                          rightMargin=72, leftMargin=72,
                          topMargin=72, bottomMargin=18,
                          title=title, author=author, subject=subject)
    
    styles = getSampleStyleSheet()
    
    # Enhanced custom styles
    doc_title_style = ParagraphStyle(
        'DocumentTitle',
        parent=styles['Title'],
        fontSize=20,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor='#1A237E'
    )
    
    section_title_style = ParagraphStyle(
        'SectionTitle',
        parent=styles['Heading1'],
        fontSize=14,
        spaceAfter=10,
        spaceBefore=15,
        textColor='#3F51B5',
        borderWidth=1,
        borderColor='#E3F2FD',
        borderPadding=5,
        backColor='#F5F5F5'
    )
    
    summary_style = ParagraphStyle(
        'Summary',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=10,
        leftIndent=15,
        rightIndent=15,
        alignment=TA_JUSTIFY,
        textColor='#424242',
        fontName='Helvetica-Oblique',
        backColor='#FAFAFA',
        borderWidth=0.5,
        borderColor='#E0E0E0',
        borderPadding=8
    )
    
    content_style = ParagraphStyle(
        'Content',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=12,
        alignment=TA_JUSTIFY,
        textColor='#212121'
    )
    
    story = []
    
    # Add document title if provided
    if title:
        story.append(Paragraph(title, doc_title_style))
        story.append(Spacer(1, 20))
    
    # Process each section
    for i, item in enumerate(data):
        section_title = item.get('title', f'Section {i+1}')
        summary = item.get('summary', '')
        content = item.get('content', '')
        
        # Clean and process content
        content = clean_content(content)
        
        # Add section title
        story.append(Paragraph(section_title, section_title_style))
        
        # Summary is skipped - not included in PDF
        
        # Add content
        if content.strip():
            paragraphs = content.split('\n\n')
            for paragraph in paragraphs:
                if paragraph.strip():
                    story.append(Paragraph(paragraph.strip(), content_style))
                    story.append(Spacer(1, 8))
        
        # Add section separator
        if i < len(data) - 1:
            story.append(Spacer(1, 25))
    
    doc.build(story)
    return os.path.abspath(output_filename)

# Example usage
#if __name__ == "__main__":
#    # Sample data similar to your format
#    sample_data = [
#        {
#            'title': 'Hello World!',
#            'summary': "This report outlines the significance of the 'Hello World!' message in programming and education.",
#            'content': '''<think>
#This is a sample think block that should be removed.
#</think>

#The report on "Hello World!" highlights its foundational role in programming and computational fields. As a universally recognized programming example, it symbolizes the principles of algorithmic structure, simplicity, and abstraction.

#The term emphasizes the foundational nature of code, enabling developers to conceptualize complex systems through foundational constructs. Its widespread application across disciplines underscores its utility in education and technical communication.'''
#        },
#        {
#            'title': 'Additional Keywords',
#            'summary': "The report includes key terms like 'programming', 'education', and 'language' to highlight its relevance.",
#            'content': 'The report on "Hello World!" emphasizes its foundational significance in programming and computational education. As a universally recognized example of algorithmic simplicity, it symbolizes the principles of abstraction, clarity, and foundational constructs.'
#        }
#    ]
    
#    # Basic usage
#    pdf_path = dict_to_pdf(sample_data, "hello_world_report.pdf")
#    print(f"Basic PDF created: {pdf_path}")
    
#    # Advanced usage
#    advanced_path = advanced_dict_to_pdf(
#        sample_data, 
#        "hello_world_advanced.pdf",
#        title="Hello World Analysis Report",
#        author="Report Generator",
#        subject="Programming Education"
#    )
#    print(f"Advanced PDF created: {advanced_path}")
