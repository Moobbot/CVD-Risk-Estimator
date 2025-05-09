import os
from datetime import timedelta
from pathlib import Path
import logging

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Get the base directory of the project
    BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
    # Load environment variables from .env file
    env_path = BASE_DIR / '.env'
    load_dotenv(dotenv_path=env_path)
    logging.info(f"Loaded environment variables from {env_path}")
except ImportError:
    logging.warning("python-dotenv package not installed. Using environment variables from system.")
except Exception as e:
    logging.warning(f"Failed to load .env file: {e}")

# Environment Configuration
ENV = os.getenv("ENV", "dev")
IS_DEV = ENV == "dev"

# Server Configuration
HOST_CONNECT = os.getenv("HOST_CONNECT", "0.0.0.0")  # Run on all IP addresses
PORT_CONNECT = int(os.getenv("PORT", 5556))  # Default port is 5556

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

os.makedirs(FOLDERS_DETECTOR, exist_ok=True)

# File Configuration
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", 100 * 1024 * 1024))  # 100MB
ALLOWED_EXTENSIONS = {".dcm"}
FILE_RETENTION = int(os.getenv("FILE_RETENTION", 1))  # 1 hour


# Model Configuration
MODEL_ITER = int(os.getenv("MODEL_ITER", 700))
MODEL_CONFIG = {
    "CHECKPOINT_PATH": os.path.join(
        BASE_DIR, "checkpoint", f"NLST-Tri2DNet_True_0.0001_16-00{MODEL_ITER}-encoder.ptm"
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
    "ITER": MODEL_ITER  # Model iteration checkpoint
}

# Logging Configuration
LOG_CONFIG = {
    "FILE": os.path.join(FOLDERS["LOGS"], f"cvd_api_{ENV}.log"),
    "FORMAT": "%(asctime)s - %(levelname)s - [%(name)s] - %(message)s",
    "LEVEL": os.getenv("LOG_LEVEL", "DEBUG" if IS_DEV else "INFO"),
    "CONSOLE": IS_DEV,
    "MAX_BYTES": int(os.getenv("LOG_MAX_BYTES", 10 * 1024 * 1024)),  # 10MB
    "BACKUP_COUNT": int(os.getenv("LOG_BACKUP_COUNT", 5)),
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
    "ENABLED": os.getenv("CLEANUP_ENABLED", "true").lower() in ("true", "1", "yes"),
    "INTERVAL_HOURS": int(os.getenv("CLEANUP_INTERVAL_HOURS", 3)),  # Run cleanup every 3 hours
    "MAX_AGE_DAYS": int(os.getenv("CLEANUP_MAX_AGE_DAYS", 1)),  # Keep files for 1 days
    "PATTERNS": {
        "UPLOAD": "*.dcm",
        "RESULTS": "*.zip",
        "REPORTS": "*.txt",
        "VISUALIZATIONS": "*.dcm", #png
        "DEBUG": "*.json",
    },
}

# Security Configuration
# Parse comma-separated list of allowed IPs
allowed_ips_str = os.getenv("ALLOWED_IPS", "127.0.0.1,192.168.1.0/24,10.0.0.0/8")
allowed_ips = [ip.strip() for ip in allowed_ips_str.split(",") if ip.strip()]

# Parse comma-separated list of allowed origins for CORS
cors_origins_str = os.getenv("CORS_ORIGINS", "*" if IS_DEV else "https://example.com")
if cors_origins_str == "*":
    cors_origins = ["*"]
else:
    cors_origins = [origin.strip() for origin in cors_origins_str.split(",") if origin.strip()]

SECURITY_CONFIG = {
    "ALLOWED_IPS": allowed_ips,
    "CORS_ORIGINS": cors_origins,
    "CORS_METHODS": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    "CORS_HEADERS": ["*"],
}
