from init_model import init_model
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import os
import zipfile
import logging
import uuid
from typing import Dict, Any
from heart_detector import HeartDetector
from image import Image
import SimpleITK as sitk
sitk.ProcessObject.SetGlobalDefaultThreader("platform")

# Cấu hình logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="DICOM to NIFTI Prediction API")

# Cấu hình thư mục lưu trữ
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DICOM_DIR = os.path.join(BASE_DIR, "dicom_files")
RESULT_DIR = os.path.join(BASE_DIR, "results")
ITER = 700  # Số lần lặp lại của mô hình (Checkpoint)

# Đảm bảo các thư mục tồn tại
for directory in [DICOM_DIR, RESULT_DIR]:
    os.makedirs(directory, exist_ok=True)


def load_model():
    try:
        logger.info("Đang tải mô hình nhận diện tim...")
        heart_detector = HeartDetector()
        logger.info("Đang tải mô hình nhận dự đoán rủi ro tim mạch...")
        m = init_model()
        m.load_model(ITER)
        logger.info("Đã tải mô hình thành công")
        return heart_detector, m
    except Exception as e:
        logger.error(f"Không thể tải mô hình: {str(e)}")
        raise RuntimeError(f"Không thể tải mô hình: {str(e)}")


# Tải mô hình khi khởi động ứng dụng
heart_detector, model = load_model()


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
        result_uuid_dir = os.path.join(RESULT_DIR, process_id)
        os.makedirs(dicom_uuid_dir, exist_ok=True)
        os.makedirs(result_uuid_dir, exist_ok=True)

        # Đường dẫn lưu file ZIP tạm thời
        zip_path = os.path.join(DICOM_DIR, file.filename)

        # Lưu file ZIP
        with open(zip_path, "wb") as temp_zip:
            content = await file.read()
            temp_zip.write(content)

        # Giải nén file ZIP vào thư mục DICOM theo UUID
        logger.info(f"Giải nén file {file.filename} vào {dicom_uuid_dir}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dicom_uuid_dir)

        os.remove(zip_path)

        img = Image(dicom_uuid_dir, heart_detector)
        logger.info("Đang phát hiện tim...")
        if not img.detect_heart():
            raise HTTPException(
                status_code=422, detail=f"Không thể phát hiện tim")
        
        # Chuyển đổi sang đầu vào cho mô hình
        network_input = img.to_network_input()
        logger.info("Đang ước lượng rủi ro tim mạch...")
        
        # Dự đoán điểm rủi ro
        score = model.aug_transform(network_input)[1].item()
        
        # Tính toán và lưu Grad-CAM trên vùng 128x128x128
        cam_data = model.grad_cam_visual(network_input)
        img.save_grad_cam_on_original(cam_data, result_uuid_dir)

        # Thêm thông tin về các file đã tạo vào kết quả
        result = {
            "score": score,
        }

        return result
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Lỗi xử lý: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
