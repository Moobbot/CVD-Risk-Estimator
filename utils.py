import os
import shutil
import logging
import uuid
import hashlib
import json
import psutil
from datetime import datetime, timedelta
import socket
from typing import List, Dict, Any, Optional
import pydicom
from fastapi import UploadFile, HTTPException
from config import (
    FOLDERS,
    FILE_RETENTION_DAYS,
    MAX_FILE_SIZE,
    ALLOWED_EXTENSIONS,
    ERROR_MESSAGES,
    LOG_CONFIG,
    SESSION_CONFIG,
    CLEANUP_CONFIG,
    MONITORING_CONFIG,
    SECURITY_CONFIG
)

logger = logging.getLogger(__name__)

def get_local_ip() -> str:
    """Get local IP address"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

def get_request_id() -> str:
    """Generate a unique request ID"""
    return str(uuid.uuid4())

def get_file_hash(file_path: str) -> str:
    """Calculate SHA-256 hash of a file"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def cleanup_old_results(folders: List[str]) -> None:
    """Clean up old files in specified folders"""
    if not CLEANUP_CONFIG["ENABLED"]:
        return
        
    cutoff_date = datetime.now() - timedelta(days=CLEANUP_CONFIG["RETENTION_DAYS"])
    
    for folder in folders:
        if not os.path.exists(folder):
            continue
            
        # Check folder size
        total_size = 0
        file_count = 0
        
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                file_time = datetime.fromtimestamp(os.path.getctime(file_path))
                file_size = os.path.getsize(file_path)
                
                # Check if file is too old
                if file_time < cutoff_date:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        logger.info(f"Removed old file: {file_path}")
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                        logger.info(f"Removed old directory: {file_path}")
                else:
                    total_size += file_size
                    file_count += 1
                    
                # Check if folder exceeds limits
                if (file_count > CLEANUP_CONFIG["MAX_FILES"] or 
                    total_size > CLEANUP_CONFIG["MAX_SIZE"]):
                    # Remove oldest files
                    files = [(f, os.path.getctime(os.path.join(folder, f))) 
                            for f in os.listdir(folder)]
                    files.sort(key=lambda x: x[1])
                    
                    for f, _ in files[:file_count - CLEANUP_CONFIG["MAX_FILES"]]:
                        f_path = os.path.join(folder, f)
                        if os.path.isfile(f_path):
                            os.remove(f_path)
                            logger.info(f"Removed file due to size limit: {f_path}")
                        elif os.path.isdir(f_path):
                            shutil.rmtree(f_path)
                            logger.info(f"Removed directory due to size limit: {f_path}")
                            
            except Exception as e:
                logger.error(f"Error cleaning up {file_path}: {str(e)}")

def validate_dicom_file(file: UploadFile) -> bool:
    """Validate if the uploaded file is a valid DICOM file"""
    try:
        # Check file extension
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in ALLOWED_EXTENSIONS:
            logger.warning(f"Invalid file extension: {file_ext}")
            return False
            
        # Check file size
        if file.size > MAX_FILE_SIZE:
            logger.warning(f"File too large: {file.size} bytes")
            return False
            
        # Try to read DICOM file
        ds = pydicom.dcmread(file.file)
        file.file.seek(0)  # Reset file pointer
        
        # Validate required DICOM tags
        required_tags = ["PatientID", "StudyDate", "Modality"]
        for tag in required_tags:
            if not hasattr(ds, tag):
                logger.warning(f"Missing required DICOM tag: {tag}")
                return False
                
        return True
    except Exception as e:
        logger.error(f"Error validating DICOM file: {str(e)}")
        return False

def save_uploaded_file(file: UploadFile, folder: str) -> str:
    """Save uploaded file to specified folder"""
    try:
        # Create timestamp-based filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{file.filename}"
        file_path = os.path.join(folder, filename)
        
        # Save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Calculate file hash
        file_hash = get_file_hash(file_path)
        
        # Save metadata
        metadata = {
            "filename": filename,
            "original_name": file.filename,
            "size": file.size,
            "content_type": file.content_type,
            "hash": file_hash,
            "upload_time": datetime.now().isoformat()
        }
        
        metadata_path = f"{file_path}.meta"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Saved file: {file_path} with hash: {file_hash}")
        return file_path
    except Exception as e:
        logger.error(f"Error saving file {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=ERROR_MESSAGES["processing_error"])

def get_error_message(error_key: str) -> str:
    """Get error message from configuration"""
    return ERROR_MESSAGES.get(error_key, "An unknown error occurred.")

def setup_logging() -> None:
    """Setup logging configuration"""
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

def is_session_expired(session_id: str, expiry_days: int) -> bool:
    """Check if a session is expired"""
    session_dir = os.path.join(FOLDERS["UPLOAD"], session_id)
    if not os.path.exists(session_dir):
        return True
        
    session_time = datetime.fromtimestamp(os.path.getctime(session_dir))
    return datetime.now() - session_time > timedelta(days=expiry_days)

def get_system_metrics() -> Dict[str, Any]:
    """Get system performance metrics"""
    if not MONITORING_CONFIG["ENABLED"]:
        return {}
        
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        metrics = {
            "cpu_usage": cpu_percent,
            "memory_usage": memory.percent,
            "disk_usage": disk.percent,
            "timestamp": datetime.now().isoformat()
        }
        
        # Check for alerts
        alerts = []
        if cpu_percent > MONITORING_CONFIG["ALERT_THRESHOLD"]["CPU_USAGE"]:
            alerts.append(f"High CPU usage: {cpu_percent}%")
        if memory.percent > MONITORING_CONFIG["ALERT_THRESHOLD"]["MEMORY_USAGE"]:
            alerts.append(f"High memory usage: {memory.percent}%")
        if disk.percent > MONITORING_CONFIG["ALERT_THRESHOLD"]["DISK_USAGE"]:
            alerts.append(f"High disk usage: {disk.percent}%")
            
        if alerts:
            logger.warning("System alerts: " + ", ".join(alerts))
            
        return metrics
    except Exception as e:
        logger.error(f"Error getting system metrics: {str(e)}")
        return {}

def validate_api_key(api_key: Optional[str] = None) -> bool:
    """Validate API key"""
    if not api_key:
        return False
    return api_key == SECURITY_CONFIG["API_KEY"]

def get_dicom_metadata(file_path: str) -> Dict[str, Any]:
    """Extract metadata from DICOM file"""
    try:
        ds = pydicom.dcmread(file_path)
        metadata = {
            "PatientID": getattr(ds, "PatientID", "Unknown"),
            "PatientName": str(getattr(ds, "PatientName", "Unknown")),
            "StudyDate": getattr(ds, "StudyDate", "Unknown"),
            "Modality": getattr(ds, "Modality", "Unknown"),
            "PixelSpacing": getattr(ds, "PixelSpacing", [1, 1]),
            "SliceThickness": getattr(ds, "SliceThickness", 1),
            "ImageSize": [ds.Rows, ds.Columns],
            "BitsAllocated": getattr(ds, "BitsAllocated", 16),
            "BitsStored": getattr(ds, "BitsStored", 16),
            "HighBit": getattr(ds, "HighBit", 15),
            "PhotometricInterpretation": getattr(ds, "PhotometricInterpretation", "MONOCHROME2"),
            "RescaleSlope": getattr(ds, "RescaleSlope", 1),
            "RescaleIntercept": getattr(ds, "RescaleIntercept", 0)
        }
        return metadata
    except Exception as e:
        logger.error(f"Error extracting DICOM metadata: {str(e)}")
        return {} 