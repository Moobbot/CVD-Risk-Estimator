import logging
import sys
import os
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from datetime import datetime
from config import LOG_CONFIG, IS_DEV

class CustomFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels"""
    grey = "\x1b[38;21m"
    blue = "\x1b[38;5;39m"
    yellow = "\x1b[38;5;226m"
    red = "\x1b[38;5;196m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    def __init__(self, fmt):
        super().__init__()
        self.fmt = fmt
        self.FORMATS = {
            logging.DEBUG: self.grey + self.fmt + self.reset,
            logging.INFO: self.blue + self.fmt + self.reset,
            logging.WARNING: self.yellow + self.fmt + self.reset,
            logging.ERROR: self.red + self.fmt + self.reset,
            logging.CRITICAL: self.bold_red + self.fmt + self.reset
        }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

def setup_logger(name, log_file=None):
    """
    Thiết lập logger với cấu hình nâng cao
    
    Parameters:
    -----------
    name: str
        Tên của logger
    log_file: str
        Đường dẫn file log (optional)
        
    Returns:
    --------
    logging.Logger: Logger đã được cấu hình
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_CONFIG["LEVEL"]))
    
    # Tránh duplicate logs
    if logger.handlers:
        return logger

    # Tạo formatter
    formatter = CustomFormatter(LOG_CONFIG["FORMAT"])
    
    # Console Handler với màu sắc
    if LOG_CONFIG["CONSOLE"]:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.stream.reconfigure(encoding='utf-8')
        logger.addHandler(console_handler)
    
    # File Handler với rotation
    if log_file or LOG_CONFIG["FILE"]:
        log_path = log_file or LOG_CONFIG["FILE"]
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        
        # Daily rotation
        file_handler = TimedRotatingFileHandler(
            log_path,
            when="midnight",
            interval=1,
            backupCount=LOG_CONFIG["BACKUP_COUNT"],
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Size-based rotation
        size_handler = RotatingFileHandler(
            log_path,
            maxBytes=LOG_CONFIG["MAX_BYTES"],
            backupCount=LOG_CONFIG["BACKUP_COUNT"],
            encoding='utf-8'
        )
        size_handler.setFormatter(formatter)
        logger.addHandler(size_handler)
    
    return logger

def log_message(logger, level, message, *args, **kwargs):
    """
    Ghi log với level tương ứng và xử lý môi trường
    
    Parameters:
    -----------
    logger: logging.Logger
        Logger instance
    level: str
        Level của log (debug, info, warning, error, critical)
    message: str
        Nội dung log
    *args, **kwargs: 
        Các tham số bổ sung
    """
    # Thêm timestamp và request ID nếu có
    if 'request_id' in kwargs:
        message = f"[{kwargs['request_id']}] {message}"
    
    if IS_DEV:
        # Trong môi trường dev, in ra console với màu sắc
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [{level.upper()}] {message}")
    
    # Ghi log theo level
    log_func = getattr(logger, level.lower())
    log_func(message, *args, **kwargs)

# Tạo logger mặc định
logger = setup_logger("cvd_risk") 