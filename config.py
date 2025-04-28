import os
from datetime import timedelta

# Environment Configuration
ENV = os.getenv("ENV", "dev")
IS_DEV = ENV == "dev"

# Server Configuration
HOST_CONNECT = "0.0.0.0"  # Chạy trên tất cả các địa chỉ IP
# Port có thể được cấu hình qua biến môi trường
PORT_CONNECT = int(os.getenv("PORT", 8000))

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
    "LOGS": os.path.join(BASE_DIR, "logs"),
}

FOLDERS_DETECTOR = "./detector"

# Create necessary directories
for folder in FOLDERS.values():
    os.makedirs(folder, exist_ok=True)

# File Configuration
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", 100 * 1024 * 1024))  # 100MB
ALLOWED_EXTENSIONS = {".dcm"}
FILE_RETENTION = int(os.getenv("FILE_RETENTION", 1))  # 1 hour


# Model Configuration
MODEL_CONFIG = {
    "CHECKPOINT_PATH": os.path.join(
        BASE_DIR, "checkpoint", "NLST-Tri2DNet_True_0.0001_16-00700-encoder.ptm"
    ),
    "MODEL_PATH": os.path.join(BASE_DIR, "checkpoint"),
    "RETINANET_PATH": os.path.join(BASE_DIR, "checkpoint", "retinanet_heart.pt"),
    "BATCH_SIZE": int(os.getenv("BATCH_SIZE", 16)),
    "DEVICE": os.getenv(
        "DEVICE", "cuda" if os.getenv("CUDA_VISIBLE_DEVICES") else "cpu"
    ),
    "DETECTION_METHODS": ["auto", "model", "simple"],
    "DEFAULT_DETECTION_METHOD": "auto",
    "VISUALIZE_DEFAULT": True,
}

# Logging Configuration
LOG_CONFIG = {
    "FILE": os.path.join(FOLDERS["LOGS"], f"cvd_api_{ENV}.log"),
    "FORMAT": "%(asctime)s - %(levelname)s - [%(name)s] - %(message)s",
    "LEVEL": "DEBUG" if IS_DEV else "INFO",
    "CONSOLE": IS_DEV,
    "MAX_BYTES": 10 * 1024 * 1024,  # 10MB
    "BACKUP_COUNT": 5,
}

# Error Messages
ERROR_MESSAGES = {
    "invalid_file": "Invalid file format. Only DICOM files are allowed.",
    "file_too_large": f"File size exceeds the maximum allowed size of {MAX_FILE_SIZE / (1024 * 1024)}MB.",
    "model_not_found": "Model checkpoint not found. Please ensure the model is properly installed.",
    "processing_error": "Error processing the DICOM files. Please check the file format and try again.",
    "server_error": "Internal server error. Please try again later.",
    "invalid_session": "Session has expired or does not exist.",
    "file_not_found": "Requested file not found.",
    "unauthorized": "Unauthorized access. Please provide a valid API key.",
}

# Cleanup Configuration
CLEANUP_CONFIG = {
    "ENABLED": True,
    "INTERVAL_HOURS": 24,  # Run cleanup every 24 hours
    "MAX_AGE_DAYS": 7,  # Keep files for 7 days
    "PATTERNS": {
        "UPLOAD": "*.dcm",
        "RESULTS": "*.zip",
        "REPORTS": "*.txt",
        "VISUALIZATIONS": "*.png",
        "DEBUG": "*.json",
    },
}

# Security Configuration
SECURITY_CONFIG = {
    "ALLOWED_IPS": ["127.0.0.1", "192.168.1.0/24", "10.0.0.0/8"],
    "SECURITY_HEADERS": {"ENABLED": True, "HSTS_MAX_AGE": 31536000},
    "CORS_ORIGINS": ["*"] if IS_DEV else ["https://example.com"],
    "CORS_METHODS": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    "CORS_HEADERS": ["*"],
}
