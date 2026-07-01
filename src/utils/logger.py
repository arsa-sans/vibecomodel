import logging
import sys
import os
from datetime import datetime

def setup_logger(name: str, log_dir: str = "outputs/logs"):
    """Set up a professional logger that logs to both console and a file."""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Avoid duplicate handlers if logger is already set up
    if logger.hasHandlers():
        return logger
    
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File Handler
    log_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{name}.log"
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger
