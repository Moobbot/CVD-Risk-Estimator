from init_model import init_model
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import os
import zipfile
import dicom2nifti
import nibabel as nib
import numpy as np
from pathlib import Path
import torch
import logging
import uuid
import shutil
from typing import Dict, Any
from colab_support.image import Image
import SimpleITK as sitk
sitk.ProcessObject.SetGlobalDefaultThreader("platform")

# Cấu hình logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="DICOM to NIFTI Prediction API")

# Cấu hình thư mục lưu trữ
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
DICOM_DIR = os.path.join(BASE_DIR, "dicom_files")
NIFTI_DIR = os.path.join(BASE_DIR, "nifti_files")
ITER = 700  # Số lần lặp lại của mô hình (Checkpoint)

# Đảm bảo các thư mục tồn tại
for directory in [UPLOAD_DIR, DICOM_DIR, NIFTI_DIR]:
    os.makedirs(directory, exist_ok=True)


def load_model():
    try:
        m = init_model()
        m.load_model(ITER)
        logger.info("Đã tải mô hình thành công")
        return m
    except Exception as e:
        logger.error(f"Không thể tải mô hình: {str(e)}")
        raise RuntimeError(f"Không thể tải mô hình: {str(e)}")


# Tải mô hình khi khởi động ứng dụng
model = load_model()


@app.post("/predict/")
async def predict_from_dicom_zip(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    API nhận vào file ZIP chứa ảnh DICOM, chuyển đổi sang NIFTI và thực hiện dự đoán
    """
    if not file.filename.endswith('.zip'):
        raise HTTPException(
            status_code=400, detail="File phải có định dạng ZIP")

    try:
        # Tạo UUID duy nhất cho mỗi request
        process_id = str(uuid.uuid4())
        logger.info(f"Xử lý yêu cầu với ID: {process_id}")

        # Tạo thư mục cho UUID này
        dicom_uuid_dir = os.path.join(DICOM_DIR, process_id)
        nifti_uuid_dir = os.path.join(NIFTI_DIR, process_id)
        os.makedirs(dicom_uuid_dir, exist_ok=True)
        os.makedirs(nifti_uuid_dir, exist_ok=True)

        # Đường dẫn lưu file ZIP tạm thời
        zip_path = os.path.join(UPLOAD_DIR, f"{process_id}.zip")

        # Lưu file ZIP
        with open(zip_path, "wb") as temp_zip:
            content = await file.read()
            temp_zip.write(content)

        # Giải nén file ZIP vào thư mục DICOM theo UUID
        logger.info(f"Giải nén file {file.filename} vào {dicom_uuid_dir}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dicom_uuid_dir)

        # Chuyển đổi DICOM sang NIFTI
        logger.info(f"Chuyển đổi DICOM sang NIFTI với ID: {process_id}")
        try:
            dicom2nifti.convert_directory(dicom_uuid_dir, nifti_uuid_dir)
        except Exception as e:
            logger.error(f"Lỗi chuyển đổi DICOM sang NIFTI: {str(e)}")
            raise HTTPException(status_code=500,
                                detail=f"Không thể chuyển đổi DICOM sang NIFTI: {str(e)}")

        # Tìm file NIFTI đã được chuyển đổi
        nifti_files = list(Path(nifti_uuid_dir).glob("*.nii.gz"))
        if not nifti_files:
            nifti_files = list(Path(nifti_uuid_dir).glob("*.nii"))

        if not nifti_files:
            raise HTTPException(status_code=404,
                                detail="Không tìm thấy file NIFTI sau khi chuyển đổi")

        # Đổi tên file NIFTI theo UUID
        original_nifti = str(nifti_files[0])
        nifti_extension = ".nii.gz" if original_nifti.endswith(
            ".nii.gz") else ".nii"
        new_nifti_name = os.path.join(
            nifti_uuid_dir, f"{process_id}{nifti_extension}")

        # Đổi tên file
        shutil.move(original_nifti, new_nifti_name)
        logger.info(f"Đã đổi tên file NIFTI thành: {new_nifti_name}")

        # Xử lý file NIFTI và dự đoán
        logger.info(f"Đang xử lý file NIFTI: {new_nifti_name}")

        img = Image()
        img.load_nifti(new_nifti_name)
        img.detect_heart()
        score = model.aug_transform(img.to_network_input())[1]

        # Thêm thông tin về các file đã tạo vào kết quả
        result = {
            "score": score,
        }

        # Tùy chọn: Xóa file ZIP sau khi xử lý
        # os.remove(zip_path)

        return result

    except Exception as e:
        logger.error(f"Lỗi xử lý: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý: {str(e)}")


@app.get("/status/{process_id}")
async def check_status(process_id: str):
    """
    Kiểm tra trạng thái xử lý của một process_id cụ thể
    """
    dicom_path = os.path.join(DICOM_DIR, process_id)
    nifti_path = os.path.join(NIFTI_DIR, process_id)

    if not os.path.exists(dicom_path):
        return {"status": "not_found", "message": f"Không tìm thấy dữ liệu với ID: {process_id}"}

    nifti_files = list(Path(nifti_path).glob(f"{process_id}.*"))

    return {
        "status": "completed" if nifti_files else "processing",
        "process_id": process_id,
        "dicom_directory": dicom_path,
        "nifti_file": str(nifti_files[0]) if nifti_files else None
    }


@app.delete("/cleanup/{process_id}")
async def cleanup_files(process_id: str):
    """
    Xóa các file liên quan đến một process_id cụ thể
    """
    dicom_path = os.path.join(DICOM_DIR, process_id)
    nifti_path = os.path.join(NIFTI_DIR, process_id)
    zip_path = os.path.join(UPLOAD_DIR, f"{process_id}.zip")

    deleted = []

    if os.path.exists(dicom_path):
        shutil.rmtree(dicom_path)
        deleted.append("dicom_directory")

    if os.path.exists(nifti_path):
        shutil.rmtree(nifti_path)
        deleted.append("nifti_directory")

    if os.path.exists(zip_path):
        os.remove(zip_path)
        deleted.append("zip_file")

    if not deleted:
        return {"status": "not_found", "message": f"Không tìm thấy dữ liệu với ID: {process_id}"}

    return {
        "status": "success",
        "message": f"Đã xóa các file liên quan đến ID: {process_id}",
        "deleted": deleted
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
