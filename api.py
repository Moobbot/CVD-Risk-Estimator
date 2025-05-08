import os
import sys
import zipfile
import logging
import uuid
import shutil
import json
import base64
from typing import Dict, Any
import traceback
from datetime import datetime

import SimpleITK as sitk
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from config import FOLDERS, MODEL_CONFIG, API_TITLE, API_DESCRIPTION, API_VERSION
from tri_2d_net.init_model import init_model
from heart_detector import HeartDetector
from image import Image

# Cấu hình SimpleITK
sitk.ProcessObject.SetGlobalDefaultThreader("platform")

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(FOLDERS["LOGS"], "api.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Khởi tạo ứng dụng FastAPI
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION
)

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Phục vụ các file tĩnh từ thư mục kết quả
app.mount("/results", StaticFiles(directory=FOLDERS["RESULTS"]), name="results")

# Hàm tiện ích
def get_local_ip():
    """Lấy địa chỉ IP cục bộ của máy chủ"""
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 1))
        ip_address = s.getsockname()[0]
        s.close()
    except:
        ip_address = "127.0.0.1"
    return ip_address

def cleanup_old_results(folders, expiry_time=3600):
    """Xóa các thư mục kết quả cũ"""
    current_time = datetime.now().timestamp()
    for folder in folders:
        if os.path.exists(folder):
            for subfolder in os.listdir(folder):
                subfolder_path = os.path.join(folder, subfolder)
                if (
                    os.path.isdir(subfolder_path)
                    and (current_time - os.path.getmtime(subfolder_path)) > expiry_time
                ):
                    shutil.rmtree(subfolder_path)
                    logger.info(f"Đã xóa thư mục cũ: {subfolder_path}")

def create_zip_result(output_dir, session_id):
    """Nén thư mục kết quả thành file ZIP"""
    result_zip_path = os.path.join(FOLDERS["RESULTS"], f"{session_id}.zip")
    logger.info(f"Tạo file ZIP từ {output_dir} đến {result_zip_path}")

    with zipfile.ZipFile(result_zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(output_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, output_dir)
                zipf.write(file_path, arcname)

    logger.info(f"Kích thước file ZIP: {os.path.getsize(result_zip_path)} bytes")
    return result_zip_path

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
            attention_scores.append({
                "file_name_pred": f"pred_{dicom_names[orig_idx]}.png",
                "attention_score": score
            })

    # Sắp xếp theo điểm chú ý giảm dần
    attention_scores.sort(key=lambda x: x["attention_score"], reverse=True)

    # Tạo kết quả
    result = {
        "attention_scores": attention_scores,
        "total_images": len(heart_indices),
        "returned_images": len(attention_scores)
    }

    return result

def load_model():
    """
    Tải các mô hình cần thiết cho ứng dụng

    Returns:
        tuple: (heart_detector, model) - Các mô hình đã được tải
    """
    try:
        logger.info("Đang tải mô hình nhận diện tim...")
        heart_detector = HeartDetector()
        if not heart_detector.load_model():
            logger.warning("Không thể tải mô hình nhận diện tim, sẽ sử dụng phương pháp đơn giản")

        logger.info("Đang tải mô hình dự đoán rủi ro tim mạch...")
        m = init_model()
        m.load_model(MODEL_CONFIG["ITER"])
        logger.info("Đã tải mô hình thành công")
        return heart_detector, m
    except Exception as e:
        logger.error(f"Không thể tải mô hình: {str(e)}")
        logger.error(traceback.format_exc())
        raise RuntimeError(f"Không thể tải mô hình: {str(e)}")

# Dọn dẹp các thư mục kết quả cũ khi khởi động
cleanup_old_results([FOLDERS["UPLOAD"], FOLDERS["RESULTS"]])

# Tải mô hình khi khởi động ứng dụng
try:
    heart_detector, model = load_model()
except Exception as e:
    logger.error(f"Lỗi khi khởi tạo mô hình: {str(e)}")
    # Không raise exception ở đây để ứng dụng vẫn có thể khởi động


@app.post("/api_predict")
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
        return JSONResponse({"error": "Invalid file format. Only ZIP is allowed."}, status_code=400)

    logger.info(f"File upload: {file.filename}")

    # Tạo UUID duy nhất cho mỗi request
    session_id = str(uuid.uuid4())
    logger.info(f"Session ID: {session_id}")

    # Tạo thư mục cho session này
    dicom_uuid_dir = os.path.join(FOLDERS["UPLOAD"], session_id)
    result_uuid_dir = os.path.join(FOLDERS["RESULTS"], session_id)
    overlay_dir = os.path.join(result_uuid_dir, "serie_0")

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
            logger.info(f"Giải nén file {file.filename} vào {dicom_uuid_dir}")
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
            return JSONResponse({"error": "No valid files found in the ZIP archive"}, status_code=400)

        logger.info(f"Đã tìm thấy {len(valid_files)} file hợp lệ")

        # Tìm thư mục con chứa file DICOM (nếu có)
        dicom_dir = dicom_uuid_dir
        for root, dirs, files in os.walk(dicom_uuid_dir):
            if any(file.endswith('.dcm') for file in files):
                dicom_dir = root
                logger.info(f"Tìm thấy thư mục chứa file DICOM: {root}")
                break

        # Đọc ảnh DICOM và phát hiện tim
        try:
            logger.info(f"Đọc ảnh DICOM từ thư mục: {dicom_dir}")
            img = Image(dicom_dir, heart_detector)

            # Phát hiện tim
            logger.info("Đang phát hiện tim...")
            if not img.detect_heart():
                logger.error("Không thể phát hiện tim trong ảnh DICOM")
                return JSONResponse({"error": "Could not detect heart in DICOM images"}, status_code=422)

            # Chuyển đổi sang đầu vào cho mô hình
            logger.info("Chuyển đổi sang đầu vào cho mô hình...")
            network_input = img.to_network_input()

            # Dự đoán điểm rủi ro
            logger.info("Đang ước lượng rủi ro tim mạch...")
            score = model.aug_transform(network_input)[1].item()
            logger.info(f"Điểm rủi ro: {score}")

            # Tính toán và lưu Grad-CAM
            logger.info("Tính toán và lưu Grad-CAM...")
            cam_data = model.grad_cam_visual(network_input)

            # Lưu ảnh Grad-CAM vào thư mục serie_0
            img.save_grad_cam_on_original(cam_data, overlay_dir)
            logger.info(f"Đã lưu Grad-CAM vào thư mục: {overlay_dir}")

            # Kiểm tra thư mục overlay có tồn tại và có ảnh không
            if not os.path.exists(overlay_dir):
                return JSONResponse({"error": "Overlay images folder not found"}, status_code=500)

            overlay_files = os.listdir(overlay_dir)
            if not overlay_files:
                return JSONResponse({"error": "No overlay images generated"}, status_code=500)

            logger.info(f"Tìm thấy {len(overlay_files)} ảnh overlay trong {overlay_dir}")

            # Nén kết quả thành file ZIP
            try:
                zip_path = create_zip_result(overlay_dir, session_id)
                logger.info(f"Đã tạo file ZIP tại: {zip_path}")

                if not os.path.exists(zip_path) or os.path.getsize(zip_path) == 0:
                    return JSONResponse({"error": "Failed to create zip file"}, status_code=500)

            except Exception as e:
                logger.error(f"Lỗi khi tạo file ZIP: {str(e)}")
                return JSONResponse({"error": f"Failed to create zip file: {str(e)}"}, status_code=500)

            # Tạo URL cho file ZIP
            base_url = str(request.base_url).rstrip("/")
            zip_download_link = f"{base_url}/download_zip/{session_id}"

            # Xử lý điểm chú ý
            heart_indices = [i for i, val in enumerate(img.bbox_selected) if val == 1]
            attention_info = process_attention_scores(cam_data, heart_indices, img.dicom_names)

            # Tạo kết quả trả về theo định dạng của Sybil
            predictions = [{"score": float(score)}]

            response = {
                "session_id": session_id,
                "predictions": predictions,
                "overlay_images": zip_download_link,
                "attention_info": attention_info,
                "message": "Prediction successful."
            }

            logger.info(f"Response: {response}")
            return JSONResponse(response)

        except Exception as e:
            logger.error(f"Lỗi trong quá trình xử lý: {str(e)}")
            logger.error(traceback.format_exc())
            return JSONResponse({"error": f"Processing error: {str(e)}"}, status_code=500)

    except Exception as e:
        logger.error(f"Lỗi xử lý: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse({"error": f"Processing error: {str(e)}"}, status_code=500)


@app.get("/download_zip/{session_id}")
async def download_zip(session_id: str):
    """API để tải xuống file ZIP chứa ảnh overlay theo Session ID"""
    file_path = os.path.join(FOLDERS["RESULTS"], f"{session_id}.zip")
    if os.path.exists(file_path):
        logger.info(f"✅ Tìm thấy file: {file_path}, chuẩn bị tải xuống...")
        return FileResponse(file_path, filename=f"{session_id}_results.zip")

    logger.warning(f"⚠️ Không tìm thấy file: {file_path}")
    return JSONResponse(
        {"error": "File not found", "session_id": session_id},
        status_code=404
    )


@app.get("/preview/{session_id}/{filename}")
async def preview_file(session_id: str, filename: str):
    """API để xem trước ảnh overlay"""
    overlay_dir = os.path.join(FOLDERS["RESULTS"], session_id, "serie_0")
    file_path = os.path.join(overlay_dir, filename)

    if os.path.exists(file_path):
        logger.info(f"✅ Xem trước file: {file_path}")
        return FileResponse(file_path)

    logger.warning(f"⚠️ Không tìm thấy file xem trước: {file_path}")
    return JSONResponse(
        {"error": "File not found", "session_id": session_id, "filename": filename},
        status_code=404
    )


if __name__ == "__main__":
    import uvicorn
    from config import HOST_CONNECT, PORT_CONNECT

    LOCAL_IP = get_local_ip()
    print(f"Running on: http://127.0.0.1:{PORT_CONNECT} (localhost)")
    print(f"Running on: http://{LOCAL_IP}:{PORT_CONNECT} (local network)")

    # Chạy trên tất cả địa chỉ IP (0.0.0.0) để nhận cả localhost và IP cục bộ
    uvicorn.run("api:app", host=HOST_CONNECT, port=PORT_CONNECT, reload=True)
