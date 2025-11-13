import logging
from typing import Optional


def setup_logging(level: str = "INFO",
          format_string: Optional[str] = None) -> None:
  """
  Setup global logging configuration for the application.
  
  Args:
    level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format_string (str, optional): Custom format string for log messages
  """
  if format_string is None:
    format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
  
  logging.basicConfig(
    level=getattr(logging, level.upper()),
    format=format_string,
    datefmt='%Y-%m-%d %H:%M:%S',
    force=True  # Override any existing configuration
  )


def get_logger(name: str) -> logging.Logger:
  """
  Get a logger instance for the specified module/class.
  
  Args:
    name (str): Name for the logger (typically __name__)
    
  Returns:
    logging.Logger: Configured logger instance
    
  Example:
    >>> logger = get_logger(__name__)
    >>> logger.info("This is a log message")
  """
  return logging.getLogger(name)
