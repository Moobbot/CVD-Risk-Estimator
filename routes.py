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
        return JSONResponse(
            {"error": f"Session folder not found: {dicom_uuid_dir}"}, status_code=404
        )

    os.makedirs(result_uuid_dir, exist_ok=True)

    # Kiểm tra các file sau khi giải nén
    valid_files = []
    for root, _, files in os.walk(dicom_uuid_dir):
        for filename in files:
            if filename.lower().endswith((".dcm", ".png")):
                valid_files.append(os.path.join(root, filename))
    if not valid_files:
        return JSONResponse(
            {"error": "No valid files found in the session folder"}, status_code=400
        )

    logger.info(f"Found {len(valid_files)} valid files")

    # Tìm thư mục con chứa file DICOM (nếu có)
    dicom_dir = dicom_uuid_dir
    for root, _, files in os.walk(dicom_uuid_dir):
        if any(file.endswith(".dcm") for file in files):
            dicom_dir = root
            logger.info(f"Found directory containing DICOM files: {root}")
            break

    if model is None:
        return JSONResponse(
            {"error": ERROR_MESSAGES["model_not_found"]}, status_code=500
        )

    try:
        pred_dict, attention_info, gif_path = predict(
            dicom_dir=dicom_dir,
            output_dir=result_uuid_dir,
            heart_detector=heart_detector,
            model=model,
            session_id=session_id,
            create_gif=True,
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


@router.post("/api_predict_zip")
async def api_predict_zip(
    request: Request, file: UploadFile = File(...)
) -> JSONResponse:
    """
    API nhận vào file ZIP chứa ảnh DICOM, chuyển đổi sang NIFTI và thực hiện dự đoán

    Args:
        request: Request object
        file: File ZIP chứa ảnh DICOM

    Returns:
        JSONResponse: Kết quả dự đoán bao gồm điểm rủi ro và đường dẫn đến ảnh kết quả
    """
    logger.info("API predict_zip called")

    # Kiểm tra định dạng file
    if not file or file.filename == "":
        return JSONResponse({"error": ERROR_MESSAGES["invalid_file"]}, status_code=400)

    if not file.filename.endswith(".zip"):
        return JSONResponse(
            {"error": "Invalid file format. Only ZIP is allowed."}, status_code=400
        )

    logger.info(f"File upload: {file.filename}")

    # Tạo UUID duy nhất cho mỗi request
    session_id = str(uuid.uuid4())
    logger.info(f"Session ID: {session_id}")

    # Tạo thư mục cho session này
    dicom_uuid_dir = os.path.join(FOLDERS["UPLOAD"], session_id)
    result_uuid_dir = os.path.join(FOLDERS["RESULTS"], session_id)

    try:
        # Tạo thư mục
        os.makedirs(dicom_uuid_dir, exist_ok=True)
        os.makedirs(result_uuid_dir, exist_ok=True)

        # Đường dẫn lưu file ZIP tạm thời
        zip_path = os.path.join(FOLDERS["UPLOAD"], f"{session_id}.zip")

        # Lưu file ZIP
        content = await file.read()
        with open(zip_path, "wb") as temp_zip:
            temp_zip.write(content)

        # Giải nén file ZIP
        try:
            logger.info(f"Extracting file {file.filename} to {dicom_uuid_dir}")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(dicom_uuid_dir)

            # Xóa file ZIP sau khi giải nén
            os.remove(zip_path)
        except zipfile.BadZipFile:
            os.remove(zip_path)
            return JSONResponse({"error": "Invalid ZIP file"}, status_code=400)

        # Kiểm tra các file sau khi giải nén
        valid_files = []
        for root, _, files in os.walk(dicom_uuid_dir):
            for filename in files:
                if filename.lower().endswith((".dcm", ".png")):
                    valid_files.append(os.path.join(root, filename))

        if not valid_files:
            shutil.rmtree(dicom_uuid_dir)  # Xóa thư mục rỗng
            return JSONResponse(
                {"error": "No valid files found in the ZIP archive"}, status_code=400
            )

        logger.info(f"Found {len(valid_files)} valid files")

        # Tìm thư mục con chứa file DICOM (nếu có)
        dicom_dir = dicom_uuid_dir
        for root, _, files in os.walk(dicom_uuid_dir):
            if any(file.endswith(".dcm") for file in files):
                dicom_dir = root
                logger.info(f"Found directory containing DICOM files: {root}")
                break

        # Check if models are available
        if model is None:
            return JSONResponse(
                {"error": ERROR_MESSAGES["model_not_found"]}, status_code=500
            )

        # Run prediction
        try:
            pred_dict, attention_info, gif_path = predict(
                dicom_dir=dicom_dir,
                output_dir=result_uuid_dir,
                heart_detector=heart_detector,
                model=model,
                session_id=session_id,
                create_gif=True,
            )

            # Kiểm tra thư mục kết quả có tồn tại và có ảnh không
            overlay_files = os.listdir(result_uuid_dir)
            if not overlay_files:
                return JSONResponse(
                    {"error": "No overlay images generated"}, status_code=500
                )

            logger.info(
                f"Found {len(overlay_files)} overlay images in {result_uuid_dir}"
            )

            # Nén kết quả thành file ZIP
            try:
                zip_path = create_zip_result(result_uuid_dir, session_id)
                logger.info(f"Created ZIP file at: {zip_path}")

                if not os.path.exists(zip_path) or os.path.getsize(zip_path) == 0:
                    return JSONResponse(
                        {"error": "Failed to create zip file"}, status_code=500
                    )

            except Exception as e:
                logger.error(f"Error creating ZIP file: {str(e)}")
                return JSONResponse(
                    {"error": f"Failed to create zip file: {str(e)}"}, status_code=500
                )

            # Tạo URL cho file ZIP và GIF
            base_url = str(request.base_url).rstrip("/")
            zip_download_link = f"{base_url}/download_zip/{session_id}"
            gif_download_link = (
                f"{base_url}/download_gif/{session_id}" if gif_path else None
            )

            # Tạo kết quả trả về
            response = {
                "session_id": session_id,
                "predictions": pred_dict["predictions"],
                "overlay_images": zip_download_link,
                "overlay_gif": gif_download_link,
                "attention_info": attention_info,
                "message": "Prediction successful.",
            }

            logger.info(f"Prediction {session_id} successful.")
            return JSONResponse(response)

        except Exception as e:
            logger.error(f"Error during processing: {str(e)}")
            logger.error(traceback.format_exc())
            return JSONResponse(
                {"error": f"Processing error: {str(e)}"}, status_code=500
            )

    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse({"error": f"Processing error: {str(e)}"}, status_code=500)


@router.get("/download_zip/{session_id}")
async def download_zip(session_id: str):
    """API để tải xuống file ZIP chứa ảnh overlay theo Session ID"""
    file_path = os.path.join(FOLDERS["RESULTS"], f"{session_id}.zip")
    if os.path.exists(file_path):
        logger.info(f"✅ File found: {file_path}, preparing download...")
        return FileResponse(file_path, filename=f"{session_id}_results.zip")

    logger.warning(f"⚠️ File not found: {file_path}")
    return JSONResponse(
        {"error": "File not found", "session_id": session_id}, status_code=404
    )


@router.get("/download_gif/{session_id}")
async def download_gif(session_id: str):
    """API để tải xuống file GIF chứa ảnh overlay theo Session ID"""
    # Đường dẫn mới: file GIF nằm trong thư mục session_id
    session_dir = os.path.join(FOLDERS["RESULTS"], session_id)
    file_path = os.path.join(session_dir, "results.gif")

    if os.path.exists(file_path):
        logger.info(f"✅ GIF file found: {file_path}, preparing download...")
        return FileResponse(
            file_path, filename=f"{session_id}_results.gif", media_type="image/gif"
        )

    # Kiểm tra đường dẫn cũ để tương thích ngược (nếu cần)
    old_path = os.path.join(FOLDERS["RESULTS"], f"{session_id}.gif")
    if os.path.exists(old_path):
        logger.info(f"✅ GIF file found at old path: {old_path}, preparing download...")
        return FileResponse(
            old_path, filename=f"{session_id}_results.gif", media_type="image/gif"
        )

    logger.warning(f"⚠️ GIF file not found: {file_path}")
    return JSONResponse(
        {"error": "GIF file not found", "session_id": session_id}, status_code=404
    )


@router.get("/preview/{session_id}/{filename}")
async def preview_file(session_id: str, filename: str):
    """API để xem trước ảnh overlay"""
    overlay_dir = os.path.join(FOLDERS["RESULTS"], session_id)
    file_path = os.path.join(overlay_dir, filename)

    if os.path.exists(file_path):
        logger.info(f"✅ Preview file: {file_path}")
        return FileResponse(file_path)

    logger.warning(f"⚠️ Preview file not found: {file_path}")
    return JSONResponse(
        {"error": "File not found", "session_id": session_id, "filename": filename},
        status_code=404,
    )
