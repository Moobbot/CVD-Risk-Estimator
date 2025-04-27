import os
from datetime import timedelta
import logging

# Environment Configuration
ENV = os.getenv("ENV", "dev")  # Default to dev environment
IS_DEV = ENV == "dev"
IS_STAGING = ENV == "staging"
IS_PROD = ENV == "prod"

# Environment-specific configurations
ENV_CONFIGS = {
    "dev": {
        "DEBUG": True,
        "LOG_LEVEL": "DEBUG",
        "CORS_ORIGINS": ["http://localhost:3000", "http://127.0.0.1:3000"],
        "RATE_LIMIT": "1000/hour",
        "CACHE_TIMEOUT": 300,  # 5 minutes
    },
    "staging": {
        "DEBUG": False,
        "LOG_LEVEL": "INFO",
        "CORS_ORIGINS": ["https://staging.example.com"],
        "RATE_LIMIT": "500/hour",
        "CACHE_TIMEOUT": 600,  # 10 minutes
    },
    "prod": {
        "DEBUG": False,
        "LOG_LEVEL": "WARNING",
        "CORS_ORIGINS": ["https://example.com"],
        "RATE_LIMIT": "100/hour",
        "CACHE_TIMEOUT": 1800,  # 30 minutes
    }
}

# Get current environment config
CURRENT_ENV_CONFIG = ENV_CONFIGS.get(ENV, ENV_CONFIGS["dev"])

# Server Configuration
HOST_CONNECT = "0.0.0.0"  # Chạy trên tất cả các địa chỉ IP
PORT_CONNECT = int(os.getenv("PORT", 8000))  # Port có thể được cấu hình qua biến môi trường

# API Configuration
API_TITLE = "CVD Risk Prediction API"
API_DESCRIPTION = "API for predicting cardiovascular disease risk from DICOM images"
API_VERSION = "1.0.0"
API_PREFIX = "/api/v1"

# Base Directory Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Folder Configuration
FOLDERS = {
    "UPLOAD": os.path.join(BASE_DIR, "uploads"),
    "RESULTS": os.path.join(BASE_DIR, "results"),
    "REPORTS": os.path.join(BASE_DIR, "reports"),
    "VISUALIZATIONS": os.path.join(BASE_DIR, "visualizations"),
    "DEBUG": os.path.join(BASE_DIR, "debug"),
    "MODEL": os.path.join(BASE_DIR, "checkpoint"),
    "DETECTOR": os.path.join(BASE_DIR, "detector"),
    "TEMP": os.path.join(BASE_DIR, "temp"),
    "LOGS": os.path.join(BASE_DIR, "logs"),
}

# Create necessary directories
for folder in FOLDERS.values():
    os.makedirs(folder, exist_ok=True)

# File Retention Configuration
FILE_RETENTION_DAYS = int(os.getenv("FILE_RETENTION_DAYS", 7))
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", 100 * 1024 * 1024))  # 100MB
ALLOWED_EXTENSIONS = {'.dcm'}

# Model Configuration
MODEL_CONFIG = {
    "CHECKPOINT_PATH": "NLST-Tri2DNet_True_0.0001_16-00700-encoder.ptm",  # Chỉ giữ tên file
    "MODEL_PATH": FOLDERS["MODEL"],  # Đường dẫn đến thư mục checkpoint
    "RETINANET_PATH": FOLDERS["DETECTOR"],  # Đường dẫn đến thư mục detector
    "BATCH_SIZE": int(os.getenv("BATCH_SIZE", 16)),
    "NUM_WORKERS": int(os.getenv("NUM_WORKERS", 4)),
    "DEVICE": os.getenv("DEVICE", "cuda" if os.getenv("CUDA_VISIBLE_DEVICES") else "cpu")
}

# Logging Configuration
LOG_CONFIG = {
    "FILE": os.path.join(FOLDERS["LOGS"], f"cvd_api_{ENV}.log"),
    "FORMAT": "%(asctime)s - %(levelname)s - [%(name)s] - %(message)s",
    "LEVEL": CURRENT_ENV_CONFIG["LOG_LEVEL"],
    "CONSOLE": IS_DEV,
    "FILE": not IS_DEV,
    "MAX_BYTES": 10 * 1024 * 1024,  # 10MB
    "BACKUP_COUNT": 5,
    "ROTATION": "midnight"
}

# Security Configuration
SECURITY_CONFIG = {
    "CORS_ORIGINS": CURRENT_ENV_CONFIG["CORS_ORIGINS"],
    "CORS_METHODS": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    "CORS_HEADERS": ["Content-Type", "Authorization", "X-Requested-With"],
    "RATE_LIMIT": CURRENT_ENV_CONFIG["RATE_LIMIT"],
    "JWT_SECRET": os.getenv("JWT_SECRET", "your-secret-key"),
    "JWT_ALGORITHM": "HS256",
    "JWT_EXPIRATION": 24 * 60 * 60,  # 24 hours
    "PASSWORD_SALT": os.getenv("PASSWORD_SALT", "your-salt"),
    "API_KEY_HEADER": "X-API-Key",
    "API_KEY": os.getenv("API_KEY", "your-api-key")
}

# Cache Configuration
CACHE_CONFIG = {
    "TYPE": "simple",
    "DEFAULT_TIMEOUT": CURRENT_ENV_CONFIG["CACHE_TIMEOUT"],
    "KEY_PREFIX": f"cvd_api_{ENV}_",
    "MAX_ENTRIES": 1000,
    "CLEANUP_INTERVAL": 3600  # 1 hour
}

# Error Messages
ERROR_MESSAGES = {
    "invalid_file": "Invalid file format. Only DICOM files are allowed.",
    "file_too_large": f"File size exceeds the maximum allowed size of {MAX_FILE_SIZE / (1024 * 1024)}MB.",
    "model_not_found": "Model checkpoint not found. Please ensure the model is properly installed.",
    "processing_error": "Error processing the DICOM files. Please check the file format and try again.",
    "server_error": "Internal server error. Please try again later.",
    "session_not_found": "Session not found. Please check the session ID.",
    "file_not_found": "File not found. Please check the file path.",
    "invalid_session": "Invalid session ID. Please check the session ID.",
    "invalid_request": "Invalid request. Please check the request parameters.",
    "unauthorized": "Unauthorized access. Please check your credentials.",
    "rate_limit_exceeded": "Rate limit exceeded. Please try again later.",
    "invalid_api_key": "Invalid API key. Please check your API key."
}

# Session Configuration
SESSION_CONFIG = {
    "EXPIRY_DAYS": FILE_RETENTION_DAYS,
    "CLEANUP_INTERVAL": 24 * 60 * 60,  # 24 hours
    "COOKIE_SECURE": not IS_DEV,
    "COOKIE_HTTPONLY": True,
    "COOKIE_SAMESITE": "Lax"
}

# API Response Configuration
RESPONSE_CONFIG = {
    "DEFAULT_TIMEOUT": 30,
    "MAX_RETRIES": 3,
    "CHUNK_SIZE": 8192,
    "COMPRESSION_LEVEL": 6,
    "CACHE_CONTROL": "no-cache" if IS_DEV else "max-age=3600"
}

# Cleanup Configuration
CLEANUP_CONFIG = {
    "ENABLED": not IS_DEV,
    "INTERVAL": 24 * 60 * 60,  # 24 hours
    "MAX_FILES": 1000,
    "MAX_SIZE": 10 * 1024 * 1024 * 1024,  # 10GB
    "RETENTION_DAYS": FILE_RETENTION_DAYS
}

# Performance Monitoring Configuration
MONITORING_CONFIG = {
    "ENABLED": not IS_DEV,
    "METRICS_INTERVAL": 60,  # 1 minute
    "LOG_LEVEL": "INFO",
    "ALERT_THRESHOLD": {
        "CPU_USAGE": 80,  # 80%
        "MEMORY_USAGE": 80,  # 80%
        "DISK_USAGE": 80,  # 80%
        "RESPONSE_TIME": 5  # 5 seconds
    }
}

# Initialize logging
logging.basicConfig(
    level=getattr(logging, LOG_CONFIG["LEVEL"]),
    format=LOG_CONFIG["FORMAT"],
    handlers=[
        logging.StreamHandler() if LOG_CONFIG["CONSOLE"] else logging.NullHandler(),
        logging.handlers.RotatingFileHandler(
            LOG_CONFIG["FILE"],
            maxBytes=LOG_CONFIG["MAX_BYTES"],
            backupCount=LOG_CONFIG["BACKUP_COUNT"]
        ) if LOG_CONFIG["FILE"] else logging.NullHandler()
    ]
) 