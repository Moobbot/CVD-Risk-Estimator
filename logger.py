import logging
import sys
from logging.handlers import RotatingFileHandler
from datetime import datetime
from pathlib import Path
from config import LOG_CONFIG, IS_DEV, FOLDERS

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


class DateFileHandler(logging.handlers.RotatingFileHandler):
    """
    A custom file handler that creates log files with date-based filenames,
    organizes them in a directory structure by year and month,
    and supports log rotation for large files.
    """
    def __init__(self, base_dir, prefix="app", suffix="log", encoding='utf-8',
                 maxBytes=10*1024*1024, backupCount=5):
        self.base_dir = Path(base_dir)
        self.prefix = prefix
        self.suffix = suffix
        self.encoding = encoding
        self.maxBytes = maxBytes
        self.backupCount = backupCount

        # Create the initial log file
        self.update_filename()

        # Initialize the handler with the current filename
        super().__init__(
            self.filename,
            maxBytes=self.maxBytes,
            backupCount=self.backupCount,
            encoding=self.encoding
        )

        # Store the last date to check for date changes
        self.last_date = datetime.now().date()

    def update_filename(self):
        """Update the log filename based on the current date"""
        now = datetime.now()
        year_month_dir = self.base_dir / str(now.year) / f"{now.month:02d}"
        year_month_dir.mkdir(parents=True, exist_ok=True)

        # Format: prefix_YYYY-MM-DD.suffix
        date_str = now.strftime("%Y-%m-%d")
        self.filename = str(year_month_dir / f"{self.prefix}_{date_str}.{self.suffix}")

    def emit(self, record):
        """Check if the date has changed and update the log file if needed"""
        current_date = datetime.now().date()
        if current_date != self.last_date:
            # Close the current log file
            self.close()

            # Update the filename for the new date
            self.update_filename()

            # Re-open the handler with the new filename
            self.baseFilename = self.filename
            self._open()

            # Update the last date
            self.last_date = current_date

        # Call the parent class's emit method
        super().emit(record)

def setup_logger(name):
    """
    Set up a logger with advanced configuration

    Parameters:
    -----------
    name: str
        Name of the logger

    Returns:
    --------
    logging.Logger: Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_CONFIG["LEVEL"]))

    # Avoid duplicate logs
    if logger.handlers:
        return logger

    # Create formatter
    formatter = CustomFormatter(LOG_CONFIG["FORMAT"])

    # Console Handler with colors
    if LOG_CONFIG["CONSOLE"]:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.stream.reconfigure(encoding='utf-8')
        logger.addHandler(console_handler)

    # File Handler with date-based organization
    logs_dir = Path(FOLDERS["LOGS"])
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Create a DateFileHandler for daily logs with rotation capability
    date_handler = DateFileHandler(
        base_dir=logs_dir,
        prefix=f"{name}",
        suffix="log",
        encoding='utf-8',
        maxBytes=LOG_CONFIG["MAX_BYTES"],
        backupCount=LOG_CONFIG["BACKUP_COUNT"]
    )
    date_handler.setFormatter(formatter)
    logger.addHandler(date_handler)

    return logger

def log_message(logger, level, message, *args, **kwargs):
    """
    Log a message with the specified level

    Parameters:
    -----------
    logger: logging.Logger
        Logger instance
    level: str
        Log level (debug, info, warning, error, critical)
    message: str
        Log message content
    *args, **kwargs:
        Additional parameters for the log
    """
    # Add timestamp and request ID if available
    if 'request_id' in kwargs:
        message = f"[{kwargs['request_id']}] {message}"

    if IS_DEV:
        # In development environment, print to console with colors
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [{level.upper()}] {message}")

    # Log with the appropriate level
    log_func = getattr(logger, level.lower())
    log_func(message, *args, **kwargs)

# Default logger is created on demand by the modules that need it