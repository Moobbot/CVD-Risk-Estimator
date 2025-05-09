import os
from datetime import timedelta
from pathlib import Path
import logging

# Environment variables defined directly in config.py
# These are the default values that will be used if not set in environment or .env file

# Environment Configuration
# Options: dev, test, prod
ENV_DEFAULT = "dev"

# Server Configuration
HOST_CONNECT_DEFAULT = "0.0.0.0"
PORT_DEFAULT = 5556

# File Configuration
# 100MB in bytes
MAX_FILE_SIZE_DEFAULT = 104857600
# File retention period in hours
FILE_RETENTION_DEFAULT = 1

# Model Configuration
BATCH_SIZE_DEFAULT = 16
# Options: cuda, cpu
DEVICE_DEFAULT = "cuda"
# Set to empty to use CPU
CUDA_VISIBLE_DEVICES_DEFAULT = "0"
# Model iteration checkpoint to load
MODEL_ITER_DEFAULT = 700

# Logging Configuration
# Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_LEVEL_DEFAULT = "DEBUG"
# 10MB in bytes
LOG_MAX_BYTES_DEFAULT = 10485760
LOG_BACKUP_COUNT_DEFAULT = 5

# Cleanup Configuration
CLEANUP_ENABLED_DEFAULT = "true"
CLEANUP_INTERVAL_HOURS_DEFAULT = 3
CLEANUP_MAX_AGE_DAYS_DEFAULT = 1

# Security Configuration
# Comma-separated list of allowed IPs or CIDR notations
ALLOWED_IPS_DEFAULT = "127.0.0.1,192.168.1.0/24,10.0.0.0/8"
# Comma-separated list of allowed origins for CORS
CORS_ORIGINS_DEFAULT = "*"

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
ENV = os.getenv("ENV", ENV_DEFAULT)
IS_DEV = ENV == "dev"

# Server Configuration
HOST_CONNECT = os.getenv("HOST_CONNECT", HOST_CONNECT_DEFAULT)  # Run on all IP addresses
PORT_CONNECT = int(os.getenv("PORT", PORT_DEFAULT))  # Default port is 5556

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
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", MAX_FILE_SIZE_DEFAULT))  # 100MB
ALLOWED_EXTENSIONS = {".dcm"}
FILE_RETENTION = int(os.getenv("FILE_RETENTION", FILE_RETENTION_DEFAULT))  # 1 hour


# Model Configuration
MODEL_ITER = int(os.getenv("MODEL_ITER", MODEL_ITER_DEFAULT))

# Determine device based on environment variables and CUDA availability
import torch
CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE_ENV = os.getenv("DEVICE", DEVICE_DEFAULT)
CUDA_VISIBLE_DEVICES_ENV = os.getenv("CUDA_VISIBLE_DEVICES", CUDA_VISIBLE_DEVICES_DEFAULT)

# If CUDA is not available or DEVICE is explicitly set to 'cpu', use CPU
if not CUDA_AVAILABLE or DEVICE_ENV.lower() == "cpu" or not CUDA_VISIBLE_DEVICES_ENV:
    ACTUAL_DEVICE = "cpu"
    print(f"Using CPU for model inference. CUDA available: {CUDA_AVAILABLE}, DEVICE: {DEVICE_ENV}, CUDA_VISIBLE_DEVICES: {CUDA_VISIBLE_DEVICES_ENV}")
else:
    ACTUAL_DEVICE = "cuda"
    print(f"Using CUDA for model inference. CUDA available: {CUDA_AVAILABLE}, DEVICE: {DEVICE_ENV}, CUDA_VISIBLE_DEVICES: {CUDA_VISIBLE_DEVICES_ENV}")

MODEL_CONFIG = {
    "CHECKPOINT_PATH": os.path.join(
        BASE_DIR, "checkpoint", f"NLST-Tri2DNet_True_0.0001_16-00{MODEL_ITER}-encoder.ptm"
    ),
    "MODEL_PATH": os.path.join(BASE_DIR, "checkpoint"),
    "RETINANET_PATH": os.path.join(BASE_DIR, "checkpoint", "retinanet_heart.pt"),
    "BATCH_SIZE": int(os.getenv("BATCH_SIZE", BATCH_SIZE_DEFAULT)),
    "DEVICE": ACTUAL_DEVICE,
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
    "MAX_BYTES": int(os.getenv("LOG_MAX_BYTES", LOG_MAX_BYTES_DEFAULT)),  # 10MB
    "BACKUP_COUNT": int(os.getenv("LOG_BACKUP_COUNT", LOG_BACKUP_COUNT_DEFAULT)),
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
    "ENABLED": os.getenv("CLEANUP_ENABLED", CLEANUP_ENABLED_DEFAULT).lower() in ("true", "1", "yes"),
    "INTERVAL_HOURS": int(os.getenv("CLEANUP_INTERVAL_HOURS", CLEANUP_INTERVAL_HOURS_DEFAULT)),  # Run cleanup every 3 hours
    "MAX_AGE_DAYS": int(os.getenv("CLEANUP_MAX_AGE_DAYS", CLEANUP_MAX_AGE_DAYS_DEFAULT)),  # Keep files for 1 days
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
allowed_ips_str = os.getenv("ALLOWED_IPS", ALLOWED_IPS_DEFAULT)
allowed_ips = [ip.strip() for ip in allowed_ips_str.split(",") if ip.strip()]

# Parse comma-separated list of allowed origins for CORS
cors_origins_str = os.getenv("CORS_ORIGINS", CORS_ORIGINS_DEFAULT if IS_DEV else "https://example.com")
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
