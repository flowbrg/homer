import fitz
import base64
import io
from PIL import Image

@staticmethod
def optimize_image_for_vision(image_data: bytes,
                              max_size: tuple = (1024, 1024)) -> str:
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
def extract_page_image(page: fitz.Page,
                       dpi: int = 300) -> str:
  """Convert PDF page to optimized base64 image"""
  # Create transformation matrix for desired DPI
  zoom = dpi / 72
  matrix = fitz.Matrix(zoom, zoom)
  
  # Render page as image
  pix = page.get_pixmap(matrix=matrix, alpha=False)
  img_data = pix.tobytes("png")
  pix = None
  
  return optimize_image_for_vision(img_data)