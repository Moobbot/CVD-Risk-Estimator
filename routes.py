import os
import zipfile
import uuid
import shutil
import traceback

import SimpleITK as sitk
from fastapi import APIRouter, File, UploadFile, Request
from fastapi.responses import JSONResponse, FileResponse

from config import FOLDERS, ERROR_MESSAGES
from logger import setup_logger
from utils import create_zip_result
from call_model import predict

# Cấu hình SimpleITK
sitk.ProcessObject.SetGlobalDefaultThreader("platform")

# Set up logger with date-based organization
logger = setup_logger("routes")

# Initialize router
router = APIRouter()

# Global variables for models (will be set by the main app)
heart_detector = None
model = None


@router.post("/api_predict")
async def api_predict(request: Request) -> JSONResponse:
    """
    API nhận session_id, truy cập folder đã giải nén sẵn, thực hiện dự đoán
    Args:
        request: Request object (body chứa session_id)
    Returns:
        JSONResponse: Kết quả dự đoán bao gồm điểm rủi ro và đường dẫn đến ảnh kết quả
    """
    logger.info("API predict (session_id) called")
    data = await request.json()
    session_id = data.get("session_id") if data else None
    if not session_id:
        return JSONResponse({"error": "Missing session_id"}, status_code=400)

    dicom_uuid_dir = os.path.join(FOLDERS["UPLOAD"], session_id)
    result_uuid_dir = os.path.join(FOLDERS["RESULTS"], session_id, "cvd")

    if not os.path.exists(dicom_uuid_dir):
        return JSONResponse({"error": f"Session folder not found: {dicom_uuid_dir}"}, status_code=404)

    os.makedirs(result_uuid_dir, exist_ok=True)

    # Kiểm tra các file sau khi giải nén
    valid_files = []
    for root, _, files in os.walk(dicom_uuid_dir):
        for filename in files:
            if filename.lower().endswith((".dcm", ".png")):
                valid_files.append(os.path.join(root, filename))
    if not valid_files:
        return JSONResponse({"error": "No valid files found in the session folder"}, status_code=400)

    logger.info(f"Found {len(valid_files)} valid files")

    # Tìm thư mục con chứa file DICOM (nếu có)
    dicom_dir = dicom_uuid_dir
    for root, _, files in os.walk(dicom_uuid_dir):
        if any(file.endswith(".dcm") for file in files):
            dicom_dir = root
            logger.info(f"Found directory containing DICOM files: {root}")
            break

    if model is None:
        return JSONResponse({"error": ERROR_MESSAGES["model_not_found"]}, status_code=500)

    try:
        pred_dict, attention_info, gif_path = predict(
            dicom_dir=dicom_dir,
            output_dir=result_uuid_dir,
            heart_detector=heart_detector,
            model=model,
            session_id=session_id,
            create_gif=True
        )

        response = {
            "session_id": session_id,
            "predictions": pred_dict["predictions"],
            "attention_info": attention_info,
            "message": "Prediction successful.",
        }
        logger.info(f"Prediction {session_id} successful.")
        return JSONResponse(response)
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse({"error": f"Processing error: {str(e)}"}, status_code=500)