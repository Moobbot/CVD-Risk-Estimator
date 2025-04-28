import os
import glob
from matplotlib import pyplot as plt
import numpy as np
import pydicom
import torch
from scipy.ndimage import zoom
import logging
from datetime import datetime

from config import ERROR_MESSAGES, IS_DEV


logger = logging.getLogger(__name__)


def load_dicom_series(dicom_dir):
    """
    Tải một loạt ảnh DICOM và chuyển đổi thành mảng 3D numpy

    Parameters:
    -----------
    dicom_dir: str
        Đường dẫn đến thư mục chứa các file DICOM

    Returns:
    --------
    numpy.ndarray: Mảng 3D chứa dữ liệu ảnh
    metadata: Dict chứa thông tin metadata
    """
    if IS_DEV:
        print(f"Đang đọc các file DICOM từ {dicom_dir}...")
    logger.info(f"Đang đọc các file DICOM từ {dicom_dir}...")

    # Lấy danh sách các file DICOM
    dicom_files = glob.glob(os.path.join(dicom_dir, "*.dcm"))
    if not dicom_files:
        raise ValueError(ERROR_MESSAGES["invalid_file"])

    # Sắp xếp theo instance number để đảm bảo thứ tự lớp
    dicom_files = sorted(dicom_files, key=lambda x: pydicom.dcmread(x).InstanceNumber)

    # Đọc file đầu tiên để lấy thông tin kích thước
    first_slice = pydicom.dcmread(dicom_files[0])
    rows = first_slice.Rows
    cols = first_slice.Columns

    # Tạo mảng để lưu dữ liệu
    img_array = np.zeros((len(dicom_files), rows, cols), dtype=np.float32)

    # Đọc tất cả các slice
    for idx, dicom_file in enumerate(dicom_files):
        ds = pydicom.dcmread(dicom_file)
        img_array[idx, :, :] = ds.pixel_array.astype(np.float32)

        # Rescale theo Rescale Slope và Intercept nếu có
        if hasattr(ds, "RescaleSlope") and hasattr(ds, "RescaleIntercept"):
            img_array[idx, :, :] = (
                img_array[idx, :, :] * ds.RescaleSlope + ds.RescaleIntercept
            )

    # Thu thập metadata từ file đầu tiên
    metadata = {
        "PatientID": (
            first_slice.PatientID if hasattr(first_slice, "PatientID") else "Unknown"
        ),
        "PatientName": (
            str(first_slice.PatientName)
            if hasattr(first_slice, "PatientName")
            else "Unknown"
        ),
        "StudyDate": (
            first_slice.StudyDate if hasattr(first_slice, "StudyDate") else "Unknown"
        ),
        "Modality": (
            first_slice.Modality if hasattr(first_slice, "Modality") else "Unknown"
        ),
        "PixelSpacing": (
            first_slice.PixelSpacing if hasattr(first_slice, "PixelSpacing") else [1, 1]
        ),
        "SliceThickness": (
            first_slice.SliceThickness if hasattr(first_slice, "SliceThickness") else 1
        ),
    }

    if IS_DEV:
        print(f"Đã đọc {len(dicom_files)} slices, kích thước: {img_array.shape}")
    logger.info(f"Đã đọc {len(dicom_files)} slices, kích thước: {img_array.shape}")
    return img_array, metadata


def preprocess_heart_ct(ct_volume, heart_bbox):
    """
    Tiền xử lý ảnh CT vùng tim để sẵn sàng cho mô hình Tri2D-Net

    Parameters:
    -----------
    ct_volume: numpy.ndarray
        Mảng 3D chứa dữ liệu ảnh CT
    heart_bbox: tuple
        Tọa độ (x_min, y_min, z_min, x_max, y_max, z_max) của vùng tim

    Returns:
    --------
    numpy.ndarray: Mảng đã được tiền xử lý, kích thước (3, 128, 128, 128)
    """
    if IS_DEV:
        print("Đang tiền xử lý ảnh CT vùng tim...")
    logger.info("Đang tiền xử lý ảnh CT vùng tim...")

    # Cắt vùng tim
    x_min, y_min, z_min, x_max, y_max, z_max = heart_bbox
    heart_ct = ct_volume[z_min:z_max, y_min:y_max, x_min:x_max]

    if IS_DEV:
        print(f"Kích thước vùng tim sau khi cắt: {heart_ct.shape}")
    logger.info(f"Kích thước vùng tim sau khi cắt: {heart_ct.shape}")

    # Chuẩn hóa giá trị HU vào khoảng [-300, 500]
    heart_ct = np.clip(heart_ct, -300, 500)
    heart_ct = (heart_ct - (-300)) / (500 - (-300))  # Chuẩn hóa về [0, 1]

    # Resize về kích thước 128x128x128
    target_shape = (128, 128, 128)
    if heart_ct.shape != target_shape:
        zoom_factors = [t / s for t, s in zip(target_shape, heart_ct.shape)]
        heart_ct = zoom(
            heart_ct, zoom_factors, order=1
        )  # order=1: linear interpolation

    if IS_DEV:
        print(f"Kích thước vùng tim sau khi resize: {heart_ct.shape}")
        print(f"Giá trị min: {heart_ct.min()}, max: {heart_ct.max()}")
    logger.info(f"Kích thước vùng tim sau khi resize: {heart_ct.shape}")
    logger.info(f"Giá trị min: {heart_ct.min()}, max: {heart_ct.max()}")

    # Tạo đầu vào 3 kênh cho mô hình Tri2D-Net
    heart_ct_3channel = np.stack([heart_ct, heart_ct, heart_ct], axis=0)

    if IS_DEV:
        print(f"Kích thước đầu vào sau khi chuẩn bị: {heart_ct_3channel.shape}")
    logger.info(f"Kích thước đầu vào sau khi chuẩn bị: {heart_ct_3channel.shape}")

    return heart_ct_3channel


def generate_report(metadata, risk_score):
    """
    Tạo báo cáo dự đoán nguy cơ CVD

    Parameters:
    -----------
    metadata: dict
        Thông tin metadata của ảnh CT
    risk_score: float
        Điểm nguy cơ CVD
    """
    try:
        # Lấy thông tin chi tiết về mức độ nguy cơ
        risk_details = get_risk_details(risk_score)

        # Tạo nội dung báo cáo
        report = f"""
        ========================================
        BÁO CÁO DỰ ĐOÁN NGUY CƠ BỆNH TIM MẠCH
        ========================================
        
        Thông tin bệnh nhân:
        - ID: {metadata.get('PatientID', 'N/A')}
        - Tên: {metadata.get('PatientName', 'N/A')}
        - Ngày chụp: {metadata.get('StudyDate', 'N/A')}
        
        Thông tin ảnh:
        - Dạng ảnh: {metadata.get('Modality', 'N/A')}
        - Độ dày lớp: {metadata.get('SliceThickness', 'N/A')} mm
        
        Kết quả dự đoán:
        - Điểm nguy cơ CVD: {risk_score:.5f} ({risk_details['risk_percentage']})
        - Mức độ nguy cơ: {risk_details['risk_level']}
        
        Khuyến nghị:"""

        # Thêm các khuyến nghị
        for recommendation in risk_details["recommendations"]:
            report += f"\n        - {recommendation}"

        report += f"""
        
        Thời gian dự đoán: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        Lưu ý: Kết quả này chỉ mang tính tham khảo và cần được bác sĩ chuyên khoa đánh giá.
        """

        # Lưu báo cáo
        with open("cvd_risk_report.txt", "w", encoding="utf-8") as f:
            f.write(report)

        if IS_DEV:
            print("Đã tạo báo cáo dự đoán nguy cơ bệnh tim mạch.")
        logger.info("Đã tạo báo cáo dự đoán nguy cơ bệnh tim mạch.")
    except Exception as e:
        if IS_DEV:
            print(f"Không thể tạo báo cáo: {e}")
        logger.error(f"Không thể tạo báo cáo: {e}")
        import traceback

        traceback.print_exc()


def visualize_results(ct_volume, heart_bbox, risk_score):
    """
    Trực quan hóa kết quả phát hiện và dự đoán

    Parameters:
    -----------
    ct_volume: numpy.ndarray
        Mảng 3D chứa dữ liệu ảnh CT
    heart_bbox: tuple
        Tọa độ (x_min, y_min, z_min, x_max, y_max, z_max) của vùng tim
    risk_score: float
        Điểm nguy cơ CVD
    """
    try:
        # Hiển thị một vài slice ở vùng tim
        x_min, y_min, z_min, x_max, y_max, z_max = heart_bbox
        z_center = (z_min + z_max) // 2

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Hiển thị slice trước, ở giữa và sau vùng tim
        for i, z_pos in enumerate([z_center - 10, z_center, z_center + 10]):
            if 0 <= z_pos < ct_volume.shape[0]:
                # Hiển thị ảnh CT gốc
                axes[i].imshow(ct_volume[z_pos], cmap="gray")

                # Vẽ bounding box
                rect = plt.Rectangle(
                    (x_min, y_min),
                    x_max - x_min,
                    y_max - y_min,
                    fill=False,
                    edgecolor="red",
                    linewidth=2,
                )
                axes[i].add_patch(rect)

                axes[i].set_title(f"Slice {z_pos}")
                axes[i].axis("off")

        plt.suptitle(f"Nguy cơ CVD: {risk_score:.5f}", fontsize=16)

        # Lưu và hiển thị
        plt.tight_layout()
        fig.savefig("cvd_risk_result.png", dpi=300, bbox_inches="tight")
        plt.close()

        if IS_DEV:
            print("Đã lưu hình ảnh kết quả vào 'cvd_risk_result.png'.")
        logger.info("Đã lưu hình ảnh kết quả vào 'cvd_risk_result.png'.")
    except Exception as e:
        if IS_DEV:
            print(f"Không thể hiển thị kết quả: {e}")
        logger.error(f"Không thể hiển thị kết quả: {e}")
        import traceback

        traceback.print_exc()


def get_risk_details(risk_score):
    """Phân tích chi tiết mức độ nguy cơ CVD"""
    details = {
        "risk_level": get_risk_level(risk_score),
        "risk_percentage": f"{risk_score * 100:.2f}%",
        "recommendations": [],
    }

    # Thêm các khuyến nghị dựa trên mức độ rủi ro
    if risk_score > 0.7:
        details["recommendations"].append("Khám chuyên khoa tim mạch gấp")
        details["recommendations"].append("Chụp mạch vành để đánh giá chi tiết")
    elif risk_score > 0.5:
        details["recommendations"].append("Khám chuyên khoa tim mạch")
        details["recommendations"].append("Theo dõi các yếu tố nguy cơ tim mạch")
    else:
        details["recommendations"].append("Tái khám định kỳ")
        details["recommendations"].append("Duy trì lối sống lành mạnh")

    return details


def get_risk_level(risk_score):
    """Chuyển đổi điểm nguy cơ thành mức độ nguy cơ"""
    if risk_score < 0.2:
        return "Thấp"
    elif risk_score < 0.5:
        return "Trung bình"
    else:
        return "Cao"


def save_heart_detection_images(ct_volume, heart_bbox, output_dir):
    """
    Lưu các ảnh phát hiện tim vào thư mục kết quả

    Parameters:
    -----------
    ct_volume: numpy.ndarray
        Mảng 3D chứa dữ liệu ảnh CT
    heart_bbox: tuple
        Tọa độ (x_min, y_min, z_min, x_max, y_max, z_max) của vùng tim
    output_dir: str
        Thư mục để lưu kết quả
    """
    os.makedirs(output_dir, exist_ok=True)

    # Lấy các slice ở giữa vùng tim
    z_min, z_max = heart_bbox[2], heart_bbox[5]
    z_center = (z_min + z_max) // 2

    # Lưu các slice cách nhau 10 đơn vị
    for i, z_pos in enumerate(range(z_center - 20, z_center + 21, 10)):
        if 0 <= z_pos < ct_volume.shape[0]:
            plt.figure(figsize=(10, 10))
            plt.imshow(ct_volume[z_pos], cmap="gray")

            # Vẽ bounding box
            x_min, y_min, x_max, y_max = (
                heart_bbox[0],
                heart_bbox[1],
                heart_bbox[3],
                heart_bbox[4],
            )
            plt.gca().add_patch(
                plt.Rectangle(
                    (x_min, y_min),
                    x_max - x_min,
                    y_max - y_min,
                    fill=False,
                    edgecolor="red",
                    linewidth=2,
                )
            )

            plt.title(f"Slice {z_pos}")
            plt.axis("off")
            plt.savefig(
                os.path.join(output_dir, f"heart_detection_slice_{z_pos}.png"),
                bbox_inches="tight",
                dpi=300,
            )
            plt.close()


def save_grad_cam_visualization(model, ct_volume, heart_bbox, output_dir):
    """
    Lưu ảnh Grad-CAM vào thư mục kết quả, không hiển thị ảnh.
    """
    import matplotlib

    matplotlib.use("Agg")  # Không dùng giao diện hiển thị
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    # Lấy ảnh tim đã chuẩn hóa, resize, 3 kênh
    heart_ct_3channel = preprocess_heart_ct(ct_volume, heart_bbox)  # (3, 128, 128, 128)
    if heart_ct_3channel.shape != (3, 128, 128, 128):
        raise ValueError(
            f"Heart CT shape phải là (3,128,128,128), hiện tại: {heart_ct_3channel.shape}"
        )

    # Thêm batch dimension
    heart_tensor = (
        torch.from_numpy(heart_ct_3channel).float().unsqueeze(0)
    )  # (1, 3, 128, 128, 128)

    plt.ioff()  # Không hiển thị ảnh

    # Gọi hàm grad_cam_visual (hàm này sẽ tự vẽ và lưu figure hiện tại)
    model.model.grad_cam_visual(heart_tensor)

    # Lưu ảnh Grad-CAM
    plt.savefig(
        os.path.join(output_dir, "grad_cam_visualization.png"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close("all")
