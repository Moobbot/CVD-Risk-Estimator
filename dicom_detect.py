#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CVD Risk Predictor using Tri2D-Net
----------------------------------
Script để đọc ảnh DICOM, tiền xử lý và dự đoán nguy cơ CVD.
"""

import os
import sys
import glob
import numpy as np
import pydicom
import torch
import SimpleITK as sitk
from scipy.ndimage import zoom
from torchvision.ops import nms
import argparse
import matplotlib.pyplot as plt
from datetime import datetime

# Cài đặt định nghĩa đường dẫn
MODEL_PATH = "./checkpoint"  # Thay đổi nếu cần
RETINANET_PATH = "./detector"  # Đường dẫn đến mô hình RetinaNet đã huấn luyện
CHECKPOINT_PATH = ".NLST-Tri2DNet_True_0.0001_16-00700-encoder.ptm"

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
    print(f"Đang đọc các file DICOM từ {dicom_dir}...")

    # Lấy danh sách các file DICOM
    dicom_files = glob.glob(os.path.join(dicom_dir, "*.dcm"))
    if not dicom_files:
        raise ValueError(f"Không tìm thấy file DICOM nào trong {dicom_dir}")

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

    print(f"Đã đọc {len(dicom_files)} slices, kích thước: {img_array.shape}")
    return img_array, metadata


class HeartDetector:
    """
    Class để phát hiện vùng tim từ ảnh CT
    """

    def __init__(self, model_path=RETINANET_PATH):
        """
        Khởi tạo detector với mô hình RetinaNet đã được huấn luyện
        """
        print("Khởi tạo detector vùng tim...")

        # Kiểm tra CUDA
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Sử dụng thiết bị: {self.device}")

        try:
            # Tải mô hình RetinaNet
            from detector.retinanet.model import resnet50

            self.model = resnet50(num_classes=1)
            self.model.load_state_dict(
                torch.load(
                    os.path.join(model_path, "retinanet_heart.pt"),
                    map_location=self.device,
                )
            )
            self.model.eval()
            self.model = self.model.to(self.device)
            print("Đã tải mô hình detector thành công.")
        except Exception as e:
            print(f"Lỗi khi tải mô hình detector: {e}")
            print("Sử dụng simple detector...")
            self.model = None

    def detect_heart_region(self, ct_volume):
        """
        Phát hiện vùng tim từ ảnh CT

        Parameters:
        -----------
        ct_volume: numpy.ndarray
            Mảng 3D chứa dữ liệu ảnh CT

        Returns:
        --------
        (x_min, y_min, z_min, x_max, y_max, z_max): tuple
            Tọa độ của vùng tim
        """
        print("Đang phát hiện vùng tim...")

        if self.model is None:
            # Nếu không có mô hình, sử dụng phương pháp simple detection
            return self._simple_heart_detection(ct_volume)

        # Xử lý từng slice để tìm vùng tim
        boxes_all = []
        scores_all = []

        # Chọn các slice chính giữa (thường chứa tim)
        start_idx = max(0, ct_volume.shape[0] // 3)
        end_idx = min(ct_volume.shape[0], 2 * ct_volume.shape[0] // 3)

        for i in range(start_idx, end_idx, 5):  # Xử lý slice theo step
            img_slice = ct_volume[i]

            # Chuẩn hóa ảnh về khoảng [0, 1]
            img_normalized = self._normalize_for_detection(img_slice)

            # Chuẩn bị input cho mô hình (3 kênh)
            img_input = np.stack(
                [img_normalized, img_normalized, img_normalized], axis=0
            )
            img_tensor = (
                torch.from_numpy(img_input).float().unsqueeze(0).to(self.device)
            )

            # Dự đoán
            with torch.no_grad():
                scores, boxes = self.model(img_tensor)

            # Lấy các box có confidence cao
            keep_indices = torch.where(scores[0] > 0.5)[0]
            if len(keep_indices) > 0:
                boxes_slice = boxes[0, keep_indices].cpu().numpy()
                scores_slice = scores[0, keep_indices].cpu().numpy()

                # Lưu lại cùng với thông tin slice hiện tại
                for box, score in zip(boxes_slice, scores_slice):
                    boxes_all.append([box[0], box[1], i, box[2], box[3], i])
                    scores_all.append(score)

        if not boxes_all:
            print(
                "Không phát hiện được vùng tim, sử dụng phương pháp simple detection."
            )
            return self._simple_heart_detection(ct_volume)

        # Chuyển sang numpy array
        boxes_all = np.array(boxes_all)
        scores_all = np.array(scores_all)

        # Tìm box có điểm cao nhất
        best_idx = np.argmax(scores_all)
        x_min, y_min, z_min, x_max, y_max, z_max = boxes_all[best_idx]

        # Mở rộng box để đảm bảo bắt toàn bộ tim
        width = x_max - x_min
        height = y_max - y_min
        depth = max(2, z_max - z_min)  # Đảm bảo có độ sâu

        x_min = max(0, x_min - width * 0.1)
        y_min = max(0, y_min - height * 0.1)
        z_min = max(0, z_min - depth)

        x_max = min(ct_volume.shape[2], x_max + width * 0.1)
        y_max = min(ct_volume.shape[1], y_max + height * 0.1)
        z_max = min(ct_volume.shape[0], z_max + depth)

        print(
            f"Đã phát hiện vùng tim: [{x_min:.1f}, {y_min:.1f}, {z_min:.1f}] - [{x_max:.1f}, {y_max:.1f}, {z_max:.1f}]"
        )
        return (int(x_min), int(y_min), int(z_min), int(x_max), int(y_max), int(z_max))

    def _simple_heart_detection(self, ct_volume):
        """
        Phương pháp phát hiện tim đơn giản dựa trên giá trị HU
        """
        # Tìm các điểm có giá trị HU trong khoảng tim (-50 đến 100)
        heart_mask = np.logical_and(ct_volume > -50, ct_volume < 100)

        # Tìm trung tâm ảnh
        center_z, center_y, center_x = np.array(ct_volume.shape) // 2

        # Xác định vùng tim dựa vào vị trí trung tâm
        z_size, y_size, x_size = ct_volume.shape

        # Giả định tim nằm ở trung tâm ảnh
        x_min = max(0, center_x - x_size // 4)
        y_min = max(0, center_y - y_size // 4)
        z_min = max(0, center_z - z_size // 4)

        x_max = min(ct_volume.shape[2], center_x + x_size // 4)
        y_max = min(ct_volume.shape[1], center_y + y_size // 4)
        z_max = min(ct_volume.shape[0], center_z + z_size // 4)

        print(
            f"Sử dụng phương pháp simple detection: [{x_min}, {y_min}, {z_min}] - [{x_max}, {y_max}, {z_max}]"
        )
        return (x_min, y_min, z_min, x_max, y_max, z_max)

    def _normalize_for_detection(self, img_slice):
        """
        Chuẩn hóa ảnh cho việc phát hiện
        """
        # Cắt giá trị HU trong khoảng phù hợp
        img_slice = np.clip(img_slice, -1000, 400)

        # Chuẩn hóa về khoảng [0, 1]
        img_normalized = (img_slice - (-1000)) / (400 - (-1000))

        return img_normalized


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
    numpy.ndarray: Mảng đã được tiền xử lý, kích thước (1, 128, 128, 128)
    """
    print("Đang tiền xử lý ảnh CT vùng tim...")

    # Cắt vùng tim
    x_min, y_min, z_min, x_max, y_max, z_max = heart_bbox
    heart_ct = ct_volume[z_min:z_max, y_min:y_max, x_min:x_max]

    print(f"Kích thước vùng tim sau khi cắt: {heart_ct.shape}")

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

    print(f"Kích thước vùng tim sau khi resize: {heart_ct.shape}")
    print(f"Giá trị min: {heart_ct.min()}, max: {heart_ct.max()}")

    # Thêm kênh cho phù hợp với input của mô hình
    heart_ct = heart_ct.reshape(1, *heart_ct.shape)

    return heart_ct


class Tri2DNetModel:
    """
    Class để load và sử dụng mô hình Tri2D-Net
    """

    def __init__(self, model_path=MODEL_PATH):
        """
        Khởi tạo và tải mô hình Tri2D-Net
        """
        print("Đang khởi tạo mô hình Tri2D-Net...")

        try:
            # Import model module
            sys.path.append("./")  # Đảm bảo có thể import từ thư mục hiện tại
            from init_model import init_model

            # Khởi tạo model
            self.model = init_model()

            # Tải checkpoint
            checkpoint_path = os.path.join(
                model_path, CHECKPOINT_PATH
            )
            self.model.encoder.load_state_dict(
                torch.load(
                    checkpoint_path,
                    map_location="cuda" if torch.cuda.is_available() else "cpu",
                )
            )
            self.model.encoder.eval()

            print("Đã tải mô hình Tri2D-Net thành công.")
        except Exception as e:
            print(f"Lỗi khi tải mô hình Tri2D-Net: {e}")
            self.model = None

    def predict_risk(self, processed_ct):
        """
        Dự đoán nguy cơ CVD từ ảnh CT đã tiền xử lý

        Parameters:
        -----------
        processed_ct: numpy.ndarray
            Mảng 4D chứa dữ liệu ảnh CT đã tiền xử lý

        Returns:
        --------
        float: Điểm nguy cơ CVD trong khoảng [0, 1]
        """
        if self.model is None:
            raise ValueError("Mô hình chưa được tải thành công.")

        print("Đang dự đoán nguy cơ CVD...")

        # Chuyển đổi sang tensor
        ct_tensor = torch.from_numpy(processed_ct).float()

        # Thực hiện dự đoán
        with torch.no_grad():
            # Phương thức aug_transform của mô hình áp dụng data augmentation
            # và lấy giá trị trung bình của các dự đoán
            pred_prob = self.model.aug_transform(ct_tensor)
            risk_score = pred_prob[1]  # Lấy xác suất của class 1 (có nguy cơ CVD)

        print(f"Điểm nguy cơ CVD đã dự đoán: {risk_score:.5f}")
        return risk_score

    def generate_heatmap(self, processed_ct):
        """
        Tạo bản đồ nhiệt để trực quan hóa các vùng quan trọng

        Parameters:
        -----------
        processed_ct: numpy.ndarray
            Mảng 4D chứa dữ liệu ảnh CT đã tiền xử lý

        Returns:
        --------
        numpy.ndarray: Mảng bản đồ nhiệt
        """
        if self.model is None:
            raise ValueError("Mô hình chưa được tải thành công.")

        print("Đang tạo bản đồ nhiệt...")

        # Chuyển đổi sang tensor
        ct_tensor = torch.from_numpy(processed_ct).float()

        try:
            # Sử dụng phương thức grad_cam_visual của mô hình
            self.model.grad_cam_visual(ct_tensor)
            print("Đã tạo bản đồ nhiệt.")
        except Exception as e:
            print(f"Không thể tạo bản đồ nhiệt: {e}")


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
        plt.show()

        print("Đã lưu hình ảnh kết quả vào 'cvd_risk_result.png'.")
    except Exception as e:
        print(f"Không thể hiển thị kết quả: {e}")


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
        - Điểm nguy cơ CVD: {risk_score:.5f}
        - Mức độ nguy cơ: {get_risk_level(risk_score)}
        
        Thời gian dự đoán: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        Lưu ý: Kết quả này chỉ mang tính tham khảo và cần được bác sĩ chuyên khoa đánh giá.
        """

        # Lưu báo cáo
        with open("cvd_risk_report.txt", "w", encoding="utf-8") as f:
            f.write(report)

        print("Đã tạo báo cáo dự đoán nguy cơ bệnh tim mạch.")
        print(report)
    except Exception as e:
        print(f"Không thể tạo báo cáo: {e}")


def get_risk_level(risk_score):
    """Chuyển đổi điểm nguy cơ thành mức độ nguy cơ"""
    if risk_score < 0.2:
        return "Thấp"
    elif risk_score < 0.5:
        return "Trung bình"
    else:
        return "Cao"


def main(dicom_dir, visualize=True):
    """
    Hàm chính để thực hiện toàn bộ quy trình

    Parameters:
    -----------
    dicom_dir: str
        Đường dẫn đến thư mục chứa các file DICOM
    visualize: bool
        Nếu True, hiển thị kết quả trực quan
    """
    try:
        # 1. Đọc ảnh DICOM
        ct_volume, metadata = load_dicom_series(dicom_dir)

        # 2. Phát hiện vùng tim
        heart_detector = HeartDetector()
        heart_bbox = heart_detector.detect_heart_region(ct_volume)

        # 3. Tiền xử lý ảnh CT
        processed_ct = preprocess_heart_ct(ct_volume, heart_bbox)

        # 4. Tải mô hình và dự đoán
        model = Tri2DNetModel()
        risk_score = model.predict_risk(processed_ct)

        # 5. Tạo báo cáo
        generate_report(metadata, risk_score)

        # 6. Trực quan hóa kết quả (nếu yêu cầu)
        if visualize:
            visualize_results(ct_volume, heart_bbox, risk_score)

            # Tạo bản đồ nhiệt
            try:
                model.generate_heatmap(processed_ct)
            except Exception as e:
                print(f"Không thể tạo bản đồ nhiệt: {e}")

        return risk_score

    except Exception as e:
        print(f"Lỗi trong quá trình xử lý: {e}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dự đoán nguy cơ CVD từ ảnh DICOM.")
    parser.add_argument(
        "dicom_dir", type=str, help="Đường dẫn đến thư mục chứa các file DICOM"
    )
    parser.add_argument(
        "--no-vis",
        dest="visualize",
        action="store_false",
        help="Không hiển thị kết quả trực quan",
    )
    parser.set_defaults(visualize=True)

    args = parser.parse_args()

    main(args.dicom_dir, args.visualize)
# python dicom_detect.py ./data/Tuong_20230828
    