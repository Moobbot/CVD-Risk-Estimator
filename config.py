import torch
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Environment Configuration
ENV_DEFAULT = "dev"
ENV = os.getenv("ENV", ENV_DEFAULT)
IS_DEV = ENV == "dev"

# Server Configuration
HOST_CONNECT_DEFAULT = "0.0.0.0"
PORT_CONNECT_DEFAULT = 5556
HOST_CONNECT = os.getenv("HOST_CONNECT", HOST_CONNECT_DEFAULT)
PORT_CONNECT = int(os.getenv("PORT", PORT_CONNECT_DEFAULT))

# Base Directory Configuration
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

# Folder Configuration
FOLDERS = {
    "UPLOAD": os.getenv("UPLOAD_FOLDER", os.path.join(BASE_DIR, "uploads")),
    "RESULTS": os.getenv("RESULTS_FOLDER", os.path.join(BASE_DIR, "results")),
    "LOGS": os.path.join(BASE_DIR, "logs"),
    "CLEANUP": os.getenv("CLEANUP_FOLDER", os.path.join(BASE_DIR, "cleanup_folder")),
    "DETECTOR": os.path.join(BASE_DIR, "detector"),
    "CHECKPOINT": os.path.join(BASE_DIR, "checkpoint"),
}

# print("FOLDERS CONFIG:", FOLDERS)

# Create necessary directories
for folder in FOLDERS.values():
    os.makedirs(folder, exist_ok=True)

# File Configuration
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", 104857600))  # 100MB
ALLOWED_EXTENSIONS = {".dcm"}
FILE_RETENTION = int(os.getenv("FILE_RETENTION", 1))  # 1 hour

# Model Configuration
MODEL_ITER = int(os.getenv("MODEL_ITER", 700))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 16))

# Determine device based on environment variables and CUDA availability
CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE_ENV = os.getenv("DEVICE", "cuda")
CUDA_VISIBLE_DEVICES_ENV = os.getenv("CUDA_VISIBLE_DEVICES", "0")

# If CUDA is not available or DEVICE is explicitly set to 'cpu', use CPU
if not CUDA_AVAILABLE or DEVICE_ENV.lower() == "cpu" or not CUDA_VISIBLE_DEVICES_ENV:
    ACTUAL_DEVICE = "cpu"
    print(
        f"Using CPU for model inference. CUDA available: {CUDA_AVAILABLE}, DEVICE: {DEVICE_ENV}, CUDA_VISIBLE_DEVICES: {CUDA_VISIBLE_DEVICES_ENV}"
    )
else:
    ACTUAL_DEVICE = "cuda"
    print(
        f"Using CUDA for model inference. CUDA available: {CUDA_AVAILABLE}, DEVICE: {DEVICE_ENV}, CUDA_VISIBLE_DEVICES: {CUDA_VISIBLE_DEVICES_ENV}"
    )

MODEL_CONFIG = {
    "CHECKPOINT_PATH": os.path.join(
        FOLDERS["CHECKPOINT"],
        f"NLST-Tri2DNet_True_0.0001_16-00{MODEL_ITER}-encoder.ptm",
    ),
    "MODEL_PATH": FOLDERS["CHECKPOINT"],
    "RETINANET_PATH": os.path.join(FOLDERS["CHECKPOINT"], "retinanet_heart.pt"),
    "BATCH_SIZE": BATCH_SIZE,
    "DEVICE": ACTUAL_DEVICE,
    "DETECTION_METHODS": ["auto", "model", "simple"],
    "DEFAULT_DETECTION_METHOD": "auto",
    "VISUALIZE_DEFAULT": True,
    "ITER": MODEL_ITER,
}

# API Configuration
API_CONFIG = {
    "TITLE": "CVD Risk Prediction API",
    "DESCRIPTION": "API for predicting cardiovascular disease risk from DICOM images",
    "VERSION": "1.0.0",
    "PREFIX": "/api/v1",
}

# Logging Configuration
LOG_CONFIG = {
    "FILE": os.path.join(FOLDERS["LOGS"], f"cvd_api_{ENV}.log"),
    "FORMAT": "%(asctime)s - %(levelname)s - [%(name)s] - %(message)s",
    "LEVEL": os.getenv("LOG_LEVEL", "DEBUG" if IS_DEV else "INFO"),
    "CONSOLE": IS_DEV,
    "MAX_BYTES": int(os.getenv("LOG_MAX_BYTES", 10485760)),  # 10MB
    "BACKUP_COUNT": int(os.getenv("LOG_BACKUP_COUNT", 5)),
}

# Error Messages
ERROR_MESSAGES = {
    "invalid_file": "Invalid file format. Only ZIP is allowed.",
    "model_not_found": "Model not found. Please check the model path.",
    "processing_error": "Error processing the file. Please try again.",
}

# Cleanup Configuration
CLEANUP_CONFIG = {
    "ENABLED": os.getenv("CLEANUP_ENABLED", "true").lower() in ("true", "1", "yes"),
    "INTERVAL_HOURS": int(os.getenv("CLEANUP_INTERVAL_HOURS", 3)),
    "MAX_AGE_DAYS": int(os.getenv("CLEANUP_MAX_AGE_DAYS", 1)),
    "PATTERNS": {
        "UPLOAD": "*.dcm",
        "RESULTS": "*.zip",
        "REPORTS": "*.txt",
        "VISUALIZATIONS": "*.dcm",
        "DEBUG": "*.json",
    },
}

# Security Configuration
allowed_ips_str = os.getenv("ALLOWED_IPS", "127.0.0.1,192.168.1.0/24,10.0.0.0/8")
allowed_ips = [ip.strip() for ip in allowed_ips_str.split(",") if ip.strip()]

cors_origins_str = os.getenv("CORS_ORIGINS", "*" if IS_DEV else "https://example.com")
if cors_origins_str == "*":
    cors_origins = ["*"]
else:
    cors_origins = [
        origin.strip() for origin in cors_origins_str.split(",") if origin.strip()
    ]

SECURITY_CONFIG = {
    "ALLOWED_IPS": allowed_ips,
    "CORS_ORIGINS": cors_origins,
    "CORS_METHODS": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    "CORS_HEADERS": ["*"],
}
