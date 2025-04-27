import logging
import sys
from config import LOG_CONFIG, IS_DEV

def setup_logger(name):
    """
    Thiết lập logger với cấu hình phù hợp cho môi trường
    
    Parameters:
    -----------
    name: str
        Tên của logger
        
    Returns:
    --------
    logging.Logger: Logger đã được cấu hình
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_CONFIG["LEVEL"]))
    
    # Tạo formatter
    formatter = logging.Formatter(LOG_CONFIG["FORMAT"])
    
    # Thêm handler cho console trong môi trường dev
    if LOG_CONFIG["CONSOLE"]:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Thêm handler cho file trong môi trường prod
    if LOG_CONFIG["FILE"]:
        file_handler = logging.FileHandler(LOG_CONFIG["FILE"])
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
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
    if IS_DEV:
        # Trong môi trường dev, in ra console
        print(f"[{level.upper()}] {message}")
    
    # Ghi log theo level
    log_func = getattr(logger, level.lower())
    log_func(message, *args, **kwargs)

# Tạo logger mặc định
logger = setup_logger("cvd_risk") 