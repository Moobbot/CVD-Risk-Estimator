import os
import zipfile
import uuid
import shutil
import traceback
import csv
from typing import List, Dict, Any
from datetime import datetime

import SimpleITK as sitk
from fastapi import APIRouter, File, UploadFile, Request
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

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


class BatchProcessRequest(BaseModel):
    """Request model for batch processing"""
    folder_path: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "folder_path": "C:/path/to/folder/containing/subfolders"
            }
        }


@router.post(
    "/api_batch_process",
    summary="Batch Process Multiple Folders",
    description="""
    API xử lý batch: đọc tuần tự từng subfolder trong folder được chọn và thực hiện dự đoán CVD risk.
    
    **Cách sử dụng:**
    1. Chọn một folder chứa nhiều subfolder
    2. Mỗi subfolder chứa các file DICOM (.dcm)
    3. API sẽ xử lý từng subfolder tuần tự
    4. Kết quả được trả về dưới dạng file CSV
    
    **Cấu trúc folder:**
    ```
    your_folder/
    ├── subfolder1/
    │   ├── image1.dcm
    │   └── image2.dcm
    ├── subfolder2/
    │   └── ...
    └── subfolder3/
        └── ...
    ```
    
    **Output:**
    - Mỗi subfolder sẽ có 1 file CSV riêng chứa kết quả từng ảnh
    - Tất cả CSV files được nén thành 1 file ZIP
    - File ZIP cũng chứa 1 file summary CSV tổng hợp
    
    **CSV cho mỗi subfolder bao gồm:**
    - file_name: Tên file ảnh
    - attention_score: Điểm attention của ảnh
    - overall_cvd_score: Điểm CVD risk tổng thể của subfolder
    - subfolder_name: Tên subfolder
    - subfolder_path: Đường dẫn đầy đủ
    
    **Summary CSV bao gồm:**
    - subfolder_name: Tên subfolder
    - status: "success" hoặc "error"
    - overall_score: Điểm dự đoán CVD risk
    - total_images: Tổng số ảnh đã xử lý
    - returned_images: Số ảnh được trả về
    - error_message: Thông báo lỗi (nếu có)
    - csv_file: Tên file CSV tương ứng
    """,
    response_description="File ZIP chứa các CSV files (mỗi subfolder 1 CSV) và 1 file summary CSV",
    response_model=None,
    tags=["Batch Processing"]
)
async def api_batch_process(request: BatchProcessRequest):
    """
    API xử lý batch: đọc tuần tự từng subfolder trong folder được chọn và thực hiện dự đoán
    
    Args:
        request: Request object chứa folder_path (đường dẫn đến folder chứa các subfolder)
    
    Returns:
        FileResponse: File ZIP chứa các CSV files (mỗi subfolder 1 CSV với kết quả từng ảnh) 
                     và 1 file summary CSV tổng hợp
    """
    logger.info("API batch process called")
    
    folder_path = request.folder_path
    
    # Kiểm tra folder có tồn tại không
    if not os.path.exists(folder_path):
        return JSONResponse(
            {"error": f"Folder not found: {folder_path}"}, status_code=404
        )
    
    if not os.path.isdir(folder_path):
        return JSONResponse(
            {"error": f"Path is not a directory: {folder_path}"}, status_code=400
        )
    
    # Kiểm tra model có sẵn không
    if model is None:
        return JSONResponse(
            {"error": ERROR_MESSAGES["model_not_found"]}, status_code=500
        )
    
    # Lấy danh sách các subfolder
    subfolders = []
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isdir(item_path):
            subfolders.append(item_path)
    
    if not subfolders:
        return JSONResponse(
            {"error": "No subfolders found in the specified folder"}, status_code=400
        )
    
    logger.info(f"Found {len(subfolders)} subfolders to process")
    
    # Tạo thư mục kết quả cho batch processing
    batch_session_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    batch_result_dir = os.path.join(FOLDERS["RESULTS"], batch_session_id)
    os.makedirs(batch_result_dir, exist_ok=True)
    
    # Danh sách CSV files đã tạo
    csv_files = []
    summary_results = []
    
    # Xử lý từng subfolder tuần tự
    for idx, subfolder_path in enumerate(subfolders, 1):
        subfolder_name = os.path.basename(subfolder_path)
        logger.info(f"Processing subfolder {idx}/{len(subfolders)}: {subfolder_name}")
        
        try:
            # Kiểm tra file DICOM trong subfolder
            valid_files = []
            for root, _, files in os.walk(subfolder_path):
                for filename in files:
                    if filename.lower().endswith((".dcm", ".png")):
                        valid_files.append(os.path.join(root, filename))
            
            if not valid_files:
                logger.warning(f"No valid files found in subfolder: {subfolder_name}")
                # Tạo CSV rỗng cho subfolder này
                csv_filename = f"{subfolder_name}_results.csv"
                csv_path = os.path.join(batch_result_dir, csv_filename)
                with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
                    writer = csv.DictWriter(
                        csvfile,
                        fieldnames=["file_name", "attention_score", "status", "error_message"]
                    )
                    writer.writeheader()
                    writer.writerow({
                        "file_name": "",
                        "attention_score": "",
                        "status": "error",
                        "error_message": "No valid DICOM files found"
                    })
                csv_files.append(csv_path)
                summary_results.append({
                    "subfolder_name": subfolder_name,
                    "status": "error",
                    "error_message": "No valid DICOM files found",
                    "csv_file": csv_filename
                })
                continue
            
            # Tìm thư mục chứa file DICOM
            dicom_dir = subfolder_path
            for root, _, files in os.walk(subfolder_path):
                if any(file.endswith(".dcm") for file in files):
                    dicom_dir = root
                    break
            
            # Tạo thư mục kết quả cho subfolder này
            subfolder_result_dir = os.path.join(batch_result_dir, subfolder_name)
            os.makedirs(subfolder_result_dir, exist_ok=True)
            
            # Thực hiện dự đoán
            try:
                pred_dict, attention_info, gif_path = predict(
                    dicom_dir=dicom_dir,
                    output_dir=subfolder_result_dir,
                    heart_detector=heart_detector,
                    model=model,
                    session_id=f"{batch_session_id}_{subfolder_name}",
                    create_gif=False,  # Không tạo GIF cho batch processing để tiết kiệm thời gian
                )
                
                # Lấy điểm số tổng thể
                overall_score = pred_dict["predictions"][0]["score"] if pred_dict.get("predictions") else None
                
                # Tạo CSV cho subfolder này với thông tin từng ảnh
                csv_filename = f"{subfolder_name}_results.csv"
                csv_path = os.path.join(batch_result_dir, csv_filename)
                
                with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
                    fieldnames = [
                        "file_name",
                        "attention_score",
                        "overall_cvd_score",
                        "subfolder_name",
                        "subfolder_path"
                    ]
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    
                    # Ghi thông tin từng ảnh
                    attention_scores = attention_info.get("attention_scores", [])
                    if attention_scores:
                        for img_info in attention_scores:
                            writer.writerow({
                                "file_name": img_info.get("file_name_pred", ""),
                                "attention_score": img_info.get("attention_score", 0),
                                "overall_cvd_score": overall_score,
                                "subfolder_name": subfolder_name,
                                "subfolder_path": subfolder_path
                            })
                    else:
                        # Nếu không có attention scores, vẫn tạo CSV với thông tin tổng thể
                        writer.writerow({
                            "file_name": "N/A",
                            "attention_score": "",
                            "overall_cvd_score": overall_score,
                            "subfolder_name": subfolder_name,
                            "subfolder_path": subfolder_path
                        })
                
                csv_files.append(csv_path)
                summary_results.append({
                    "subfolder_name": subfolder_name,
                    "status": "success",
                    "overall_score": overall_score,
                    "total_images": attention_info.get("total_images", 0),
                    "returned_images": attention_info.get("returned_images", 0),
                    "csv_file": csv_filename
                })
                
                logger.info(f"Successfully processed subfolder: {subfolder_name}, score: {overall_score}, CSV created: {csv_filename}")
                
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Error processing subfolder {subfolder_name}: {error_msg}")
                logger.error(traceback.format_exc())
                
                # Tạo CSV với thông báo lỗi
                csv_filename = f"{subfolder_name}_results.csv"
                csv_path = os.path.join(batch_result_dir, csv_filename)
                with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
                    writer = csv.DictWriter(
                        csvfile,
                        fieldnames=["file_name", "attention_score", "status", "error_message"]
                    )
                    writer.writeheader()
                    writer.writerow({
                        "file_name": "",
                        "attention_score": "",
                        "status": "error",
                        "error_message": error_msg
                    })
                csv_files.append(csv_path)
                summary_results.append({
                    "subfolder_name": subfolder_name,
                    "status": "error",
                    "error_message": error_msg,
                    "csv_file": csv_filename
                })
                
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Unexpected error processing subfolder {subfolder_name}: {error_msg}")
            logger.error(traceback.format_exc())
            
            # Tạo CSV với thông báo lỗi
            csv_filename = f"{subfolder_name}_results.csv"
            csv_path = os.path.join(batch_result_dir, csv_filename)
            try:
                with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
                    writer = csv.DictWriter(
                        csvfile,
                        fieldnames=["file_name", "attention_score", "status", "error_message"]
                    )
                    writer.writeheader()
                    writer.writerow({
                        "file_name": "",
                        "attention_score": "",
                        "status": "error",
                        "error_message": error_msg
                    })
                csv_files.append(csv_path)
                summary_results.append({
                    "subfolder_name": subfolder_name,
                    "status": "error",
                    "error_message": error_msg,
                    "csv_file": csv_filename
                })
            except Exception as csv_error:
                logger.error(f"Error creating error CSV for {subfolder_name}: {csv_error}")
    
    # Tạo file ZIP chứa tất cả các CSV files
    zip_filename = f"{batch_session_id}_results.zip"
    zip_path = os.path.join(batch_result_dir, zip_filename)
    
    try:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for csv_file in csv_files:
                if os.path.exists(csv_file):
                    arcname = os.path.basename(csv_file)
                    zipf.write(csv_file, arcname)
                    logger.info(f"Added {arcname} to ZIP")
        
        # Tạo file summary CSV
        summary_csv_filename = f"{batch_session_id}_summary.csv"
        summary_csv_path = os.path.join(batch_result_dir, summary_csv_filename)
        with open(summary_csv_path, "w", newline="", encoding="utf-8") as csvfile:
            fieldnames = [
                "subfolder_name",
                "status",
                "overall_score",
                "total_images",
                "returned_images",
                "error_message",
                "csv_file"
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for result in summary_results:
                writer.writerow(result)
        
        # Thêm summary CSV vào ZIP
        with zipfile.ZipFile(zip_path, "a", zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(summary_csv_path, summary_csv_filename)
        
        logger.info(f"ZIP file created: {zip_path}")
        logger.info(f"Total processed: {len(summary_results)}, Successful: {sum(1 for r in summary_results if r['status'] == 'success')}, Failed: {sum(1 for r in summary_results if r['status'] == 'error')}")
        
        # Trả về file ZIP
        return FileResponse(
            zip_path,
            filename=zip_filename,
            media_type="application/zip",
        )
        
    except Exception as e:
        logger.error(f"Error creating ZIP file: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            {"error": f"Failed to create ZIP file: {str(e)}"}, status_code=500
        )
