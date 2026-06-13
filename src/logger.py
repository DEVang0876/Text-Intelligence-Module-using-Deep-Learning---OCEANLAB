import logging
from src.config import LOG_FILE_PATH

def setup_logger():
    """
    Sets up a logger to save logs to a file.
    
    Returns:
        logging.Logger: The configured logger.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    
    # Create handlers
    f_handler = logging.FileHandler(LOG_FILE_PATH)
    f_handler.setLevel(logging.INFO)
    
    # Create formatters and add it to handlers
    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    f_handler.setFormatter(f_format)
    
    # Add handlers to the logger
    logger.addHandler(f_handler)
    
    return logger

logger = setup_logger()
