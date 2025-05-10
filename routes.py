import os
import zipfile
import uuid
import shutil
import traceback

import SimpleITK as sitk
from fastapi import APIRouter, File, UploadFile, Request
from fastapi.responses import JSONResponse, FileResponse

from config import FOLDERS
from image import Image
from logger import setup_logger
from utils import process_attention_scores, create_zip_result

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
async def api_predict(request: Request, file: UploadFile = File(...)) -> JSONResponse:
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
        return JSONResponse({"error": "No selected file"}, status_code=400)

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
    overlay_dir = os.path.join(result_uuid_dir)

    try:
        # Tạo thư mục
        os.makedirs(dicom_uuid_dir, exist_ok=True)
        os.makedirs(result_uuid_dir, exist_ok=True)
        os.makedirs(overlay_dir, exist_ok=True)

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

        # Đọc ảnh DICOM và phát hiện tim
        try:
            logger.info(f"Reading DICOM images from directory: {dicom_dir}")
            img = Image(dicom_dir, heart_detector)

            # Phát hiện tim
            logger.info("Detecting heart...")
            if not img.detect_heart():
                logger.error("Could not detect heart in DICOM images")
                return JSONResponse(
                    {"error": "Could not detect heart in DICOM images"}, status_code=422
                )

            # Chuyển đổi sang đầu vào cho mô hình
            logger.info("Converting to model input...")
            network_input = img.to_network_input()

            # Check if model is available
            if model is None:
                return JSONResponse(
                    {"error": "CVD risk prediction model is not available. Please check server logs."},
                    status_code=500
                )

            # Dự đoán điểm rủi ro
            logger.info("Estimating cardiovascular risk...")
            try:
                pred_result = model.aug_transform(network_input)
                score = pred_result[1].item()
                logger.info(f"Risk score: {score}")
            except Exception as e:
                logger.error(f"Error during risk prediction: {str(e)}")
                logger.error(traceback.format_exc())
                return JSONResponse(
                    {"error": f"Error during risk prediction: {str(e)}"},
                    status_code=500
                )

            # Tính toán và lưu Grad-CAM
            logger.info("Calculating and saving Grad-CAM...")
            try:
                cam_data = model.grad_cam_visual(network_input)

                # Lưu ảnh Grad-CAM vào thư mục và tạo GIF trực tiếp từ bộ nhớ
                success, gif_path = img.save_grad_cam_on_original(cam_data, overlay_dir, create_gif=True, session_id=session_id)
                logger.info(f"Saved Grad-CAM to directory: {overlay_dir}")
            except Exception as e:
                logger.error(f"Error during Grad-CAM calculation: {str(e)}")
                logger.error(traceback.format_exc())
                return JSONResponse(
                    {"error": f"Error during visualization: {str(e)}"},
                    status_code=500
                )

            if not success:
                return JSONResponse(
                    {"error": "Failed to save overlay images"}, status_code=500
                )

            # Kiểm tra thư mục overlay có tồn tại và có ảnh không
            if not os.path.exists(overlay_dir):
                return JSONResponse(
                    {"error": "Overlay images folder not found"}, status_code=500
                )

            overlay_files = os.listdir(overlay_dir)
            if not overlay_files:
                return JSONResponse(
                    {"error": "No overlay images generated"}, status_code=500
                )

            logger.info(f"Found {len(overlay_files)} overlay images in {overlay_dir}")

            # Nén kết quả thành file ZIP
            try:
                zip_path = create_zip_result(overlay_dir, session_id)
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

            # Ghi log kết quả tạo GIF
            if gif_path:
                logger.info(f"Created GIF file at: {gif_path}")
            else:
                logger.warning("Could not create GIF file")
                # Thử tạo GIF từ file nếu tạo trực tiếp từ bộ nhớ không thành công
                try:
                    gif_path = img.create_gif_from_overlay_images(overlay_dir, session_id)
                    if gif_path:
                        logger.info(f"Created GIF file from disk at: {gif_path}")
                except Exception as e:
                    logger.error(f"Error creating GIF file from disk: {str(e)}")
                    # Không trả về lỗi, tiếp tục xử lý vì GIF là tính năng bổ sung

            # Tạo URL cho file ZIP và GIF
            base_url = str(request.base_url).rstrip("/")
            zip_download_link = f"{base_url}/download_zip/{session_id}"
            gif_download_link = f"{base_url}/download_gif/{session_id}" if gif_path else None

            # Xử lý điểm chú ý
            heart_indices = [i for i, val in enumerate(img.bbox_selected) if val == 1]
            attention_info = process_attention_scores(
                cam_data, heart_indices, img.dicom_names
            )

            # Tạo kết quả trả về theo định dạng của Sybil
            predictions = [{"score": float(score)}]

            response = {
                "session_id": session_id,
                "predictions": predictions,
                "overlay_images": zip_download_link,
                "overlay_gif": gif_download_link,
                "attention_info": attention_info,
                "message": "Prediction successful.",
            }

            # logger.info(f"Response: {response}")
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
        return FileResponse(file_path, filename=f"{session_id}_results.gif", media_type="image/gif")

    # Kiểm tra đường dẫn cũ để tương thích ngược (nếu cần)
    old_path = os.path.join(FOLDERS["RESULTS"], f"{session_id}.gif")
    if os.path.exists(old_path):
        logger.info(f"✅ GIF file found at old path: {old_path}, preparing download...")
        return FileResponse(old_path, filename=f"{session_id}_results.gif", media_type="image/gif")

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
