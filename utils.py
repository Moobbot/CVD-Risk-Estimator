# -*- coding: utf-8 -*-
# @Author  : chq_N
# @Time    : 2020/10/28

import logging
import os
import shutil
import uuid
import hashlib
import json
from datetime import datetime, timedelta
import socket
from typing import List, Dict, Any
import zipfile
import numpy as np
import pydicom
from fastapi import UploadFile, HTTPException
from fastapi.responses import JSONResponse
import psutil
from config import (
    FILE_RETENTION,
    FOLDERS,
    IS_DEV,
    MAX_FILE_SIZE,
    ALLOWED_EXTENSIONS,
    ERROR_MESSAGES,
)

logger = logging.getLogger(__name__)

import SimpleITK as sitk

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

def cleanup_old_files(folders: List[str], expiry_time=FILE_RETENTION) -> None:
    """Clean up old files in specified folders.

    Args:
        folders (List): List of folders to be checked.
        expiry_time (int): expiry period (hour). The default is 1 hour.
    """
    cutoff_date = datetime.now() - timedelta(hours=expiry_time)

    for folder in folders:
        if not os.path.exists(folder):
            continue

        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            # Skip .gitignore files
            if filename == ".gitignore":
                continue

            try:
                file_time = datetime.fromtimestamp(os.path.getctime(file_path))

                if file_time < cutoff_date:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        logger.info(f"Removed old file: {file_path}")
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                        logger.info(f"Removed old directory: {file_path}")
            except Exception as e:
                logger.error(f"Error cleaning up {file_path}: {str(e)}")


def process_attention_scores(cam_data, heart_indices, dicom_names):
    """Xử lý điểm chú ý để trả về định dạng giống Sybil"""
    attention_scores = []

    for idx, orig_idx in enumerate(heart_indices):
        if idx >= len(cam_data):
            break

        # Lấy slice tương ứng từ cam_data
        cam_slice = cam_data[idx]

        # Tính điểm chú ý (trung bình giá trị trong slice)
        score = float(np.mean(cam_slice))

        if score > 0:
            attention_scores.append(
                {
                    "file_name_pred": f"{orig_idx}_{dicom_names[orig_idx]}.png",
                    "attention_score": score,
                }
            )

    # Sắp xếp theo điểm chú ý giảm dần
    attention_scores.sort(key=lambda x: x["attention_score"], reverse=True)

    # Tạo kết quả
    result = {
        "attention_scores": attention_scores,
        "total_images": len(heart_indices),
        "returned_images": len(attention_scores),
    }

    return result


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
            "upload_time": datetime.now().isoformat(),
        }

        metadata_path = f"{file_path}.meta"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved file: {file_path} with hash: {file_hash}")
        return file_path
    except Exception as e:
        logger.error(f"Error saving file {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=ERROR_MESSAGES["processing_error"])


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
            "PhotometricInterpretation": getattr(
                ds, "PhotometricInterpretation", "MONOCHROME2"
            ),
            "RescaleSlope": getattr(ds, "RescaleSlope", 1),
            "RescaleIntercept": getattr(ds, "RescaleIntercept", 0),
        }
        return metadata
    except Exception as e:
        logger.error(f"Error extracting DICOM metadata: {str(e)}")
        return {}


def get_system_metrics() -> Dict[str, Any]:
    """Get system metrics"""
    try:
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)

        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent

        # Disk usage
        disk = psutil.disk_usage("/")
        disk_percent = disk.percent

        # Network I/O
        net_io = psutil.net_io_counters()

        return {
            "cpu_usage": cpu_percent,
            "memory_usage": memory_percent,
            "disk_usage": disk_percent,
            "network": {
                "bytes_sent": net_io.bytes_sent,
                "bytes_recv": net_io.bytes_recv,
                "packets_sent": net_io.packets_sent,
                "packets_recv": net_io.packets_recv,
            },
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Error getting system metrics: {str(e)}")
        return {
            "error": "Failed to get system metrics",
            "timestamp": datetime.now().isoformat(),
        }


def get_error_message(error_key: str) -> str:
    """Get error message from config"""
    return ERROR_MESSAGES.get(error_key, "Unknown error occurred")


def save_uploaded_zip(
    file: UploadFile, session_id: str, folder_save: str = FOLDERS["UPLOAD"]
) -> str:
    """Lưu file ZIP tải lên"""
    try:
        zip_path = os.path.join(folder_save, session_id)

        # Save file
        with open(zip_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Calculate file hash
        file_hash = get_file_hash(zip_path)

        # Save metadata
        metadata = {
            "session_id": session_id,
            "original_name": file.filename,
            "size": file.size,
            "content_type": file.content_type,
            "hash": file_hash,
            "upload_time": datetime.now().isoformat(),
        }

        metadata_path = f"{zip_path}.meta"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved ZIP file: {zip_path} with hash: {file_hash}")
        return zip_path
    except Exception as e:
        logger.error(f"Error saving ZIP file {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=ERROR_MESSAGES["processing_error"])


def extract_zip_file(
    zip_path: str, session_id: str, folder_save: str = FOLDERS["UPLOAD"]
) -> tuple:
    """Giải nén ZIP, kiểm tra thư mục con"""
    unzip_path = os.path.join(folder_save, session_id)
    os.makedirs(unzip_path, exist_ok=True)
    logger.info(f"Extracting ZIP to: {unzip_path}")

    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(unzip_path)
    except zipfile.BadZipFile:
        os.remove(zip_path)
        logger.error(f"Invalid ZIP file: {zip_path}")
        return (
            None,
            JSONResponse(
                content={"error": ERROR_MESSAGES["invalid_file"]}, status_code=400
            ),
            400,
        )

    os.remove(zip_path)

    # Nếu ZIP chỉ có 1 thư mục con, cập nhật lại đường dẫn
    subfolders = [
        f for f in os.listdir(unzip_path) if os.path.isdir(os.path.join(unzip_path, f))
    ]
    if len(subfolders) == 1:
        unzip_path = os.path.join(unzip_path, subfolders[0])
        logger.info(f"Updated unzip path to subfolder: {unzip_path}")

    return unzip_path, None, None


def get_valid_files(unzip_path: str) -> List[str]:
    """Lấy danh sách file hợp lệ (DICOM/PNG)"""
    valid_files = []
    try:
        for root, _, files in os.walk(unzip_path):
            for filename in files:
                if filename.lower().endswith((".dcm", ".png")):
                    file_path = os.path.join(root, filename)
                    valid_files.append(file_path)
                    logger.info(f"Found valid file: {file_path}")

        if not valid_files:
            logger.warning(f"No valid files found in: {unzip_path}")
    except Exception as e:
        logger.error(f"Error getting valid files from {unzip_path}: {str(e)}")
        raise HTTPException(status_code=500, detail=ERROR_MESSAGES["processing_error"])

    return valid_files


def create_zip_result(output_dir, session_id, folder_save=FOLDERS["RESULTS"]):
    """Nén ảnh dự đoán thành file ZIP"""
    result_zip_path = os.path.join(folder_save, f"{session_id}.zip")
    if IS_DEV == "dev":
        print(f"Creating zip file from {output_dir} to {result_zip_path}")
    else:
        logger.info(f"Creating ZIP file from {output_dir} to {result_zip_path}")

    with zipfile.ZipFile(result_zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(output_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, output_dir)
                zipf.write(file_path, arcname)

    if IS_DEV == "dev":
        print(f"Zip file size: {os.path.getsize(result_zip_path)} bytes")
    else:
        logger.info(f"ZIP file size: {os.path.getsize(result_zip_path)} bytes")
    return result_zip_path

def CT_resize(image, new_size=None, new_space=None, new_direction=None, new_org=None):
    if new_size is None:
        new_size = image.GetSize()
    if new_space is None:
        new_space = image.GetSpacing()
    if new_direction is None:
        new_direction = image.GetDirection()
    if new_org is None:
        new_org = image.GetOrigin()
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputDirection(new_direction)
    resampler.SetOutputSpacing(new_space)
    resampler.SetSize(new_size)
    resampler.SetOutputOrigin(new_org)
    resampler.SetInterpolator(sitk.sitkGaussian)
    resampler.SetDefaultPixelValue(
        sitk.GetArrayFromImage(image).min().astype('float'))
    return resampler.Execute(image)


def norm(input_array, norm_down, norm_up):
    input_array = input_array.astype('float32')
    normed_array = (input_array - norm_down) / (norm_up - norm_down)
    normed_array[normed_array > 1] = 1
    normed_array[normed_array < 0] = 0
    return normed_array


def visualize_data(npy_img):
    print(npy_img)