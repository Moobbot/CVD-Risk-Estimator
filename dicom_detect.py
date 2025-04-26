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

sys.path.append("./detector")

# Cài đặt định nghĩa đường dẫn
MODEL_PATH = "./checkpoint"  # Thay đổi nếu cần
RETINANET_PATH = "./detector"  # Đường dẫn đến mô hình RetinaNet đã huấn luyện
CHECKPOINT_PATH = "NLST-Tri2DNet_True_0.0001_16-00700-encoder.ptm"
import logging

# Set up logging
logging.basicConfig(
    filename="cvd_detection.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


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


def inspect_model(model):
    """Print a summary of the model structure"""
    print(f"Model type: {type(model)}")

    # If it's a class with a forward method
    if hasattr(model, "forward"):
        try:
            # Check signature of forward method
            import inspect

            sig = inspect.signature(model.forward)
            print(f"Forward method signature: {sig}")
        except Exception as e:
            print(f"Couldn't inspect forward method: {e}")

    # List main attributes
    try:
        print("Model attributes:")
        for attr_name in dir(model):
            if not attr_name.startswith("_"):  # Skip private attributes
                attr = getattr(model, attr_name)
                if not callable(attr):
                    print(f"  {attr_name}: {type(attr)}")
    except Exception as e:
        print(f"Couldn't list attributes: {e}")


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
            model_file = os.path.join(model_path, "retinanet_heart.pt")
            if os.path.exists(model_file):
                self.model = torch.load(model_file, map_location=self.device)
                self.model.eval()
                print("Đã tải mô hình detector thành công.")
            else:
                print(f"Không tìm thấy file mô hình: {model_file}")
                self.model = None
        except Exception as e:
            print(f"Lỗi khi tải mô hình detector: {e}")
            print("Sử dụng simple detector...")
            self.model = None

    def _normalize_for_detection(self, img_slice):
        """
        Chuẩn hóa ảnh cho việc phát hiện
        """
        # Cắt giá trị HU trong khoảng phù hợp
        img_slice = np.clip(img_slice, -1000, 400)

        # Chuẩn hóa về khoảng [0, 1]
        img_normalized = (img_slice - (-1000)) / (400 - (-1000))

        return img_normalized

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

    def _improved_heart_detection(self, ct_volume):
        """
        Enhanced method for heart detection using image processing
        """
        from scipy import ndimage

        # Find the middle slice
        mid_slice = ct_volume.shape[0] // 2

        # Window the HU values to focus on soft tissue
        windowed = np.clip(ct_volume[mid_slice], -150, 250)
        normalized = (windowed - (-150)) / (250 - (-150))

        # Threshold to create a binary mask
        threshold = 0.5
        binary = (normalized > threshold).astype(np.uint8)

        # Find connected components
        labeled, num_features = ndimage.label(binary)

        # Find the largest connected component near the center
        center_y, center_x = np.array(binary.shape) // 2
        center_region = labeled[center_y, center_x]

        if (
            center_region == 0
        ):  # If center is not in any region, find the largest region
            sizes = ndimage.sum(binary, labeled, range(1, num_features + 1))
            largest_region = np.argmax(sizes) + 1
            mask = labeled == largest_region
        else:
            mask = labeled == center_region

        # Find bounding box of the region
        y_indices, x_indices = np.where(mask)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        x_min, x_max = np.min(x_indices), np.max(x_indices)

        # Determine Z range (depth) - typically 1/3 of slices centered at middle
        z_min = max(0, mid_slice - ct_volume.shape[0] // 6)
        z_max = min(ct_volume.shape[0], mid_slice + ct_volume.shape[0] // 6)

        # Expand the bounding box slightly
        x_min = max(0, x_min - 20)
        y_min = max(0, y_min - 20)
        x_max = min(ct_volume.shape[2] - 1, x_max + 20)
        y_max = min(ct_volume.shape[1] - 1, y_max + 20)

        return (int(x_min), int(y_min), int(z_min), int(x_max), int(y_max), int(z_max))

    def detect_heart_region(self, ct_volume):
        """
        Phát hiện vùng tim từ ảnh CT
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
                try:
                    # Try standard format first (returning scores, boxes)
                    output = self.model(img_tensor)

                    # Check the output format and adapt accordingly
                    if isinstance(output, tuple) and len(output) == 2:
                        scores, boxes = output
                    elif (
                        isinstance(output, dict)
                        and "scores" in output
                        and "boxes" in output
                    ):
                        scores, boxes = output["scores"], output["boxes"]
                    else:
                        print(
                            f"Unexpected model output format: {type(output)}. Skipping slice {i}."
                        )
                        continue

                    # Lấy các box có confidence cao
                    keep_indices = torch.where(scores[0] > 0.5)[0]
                    if len(keep_indices) > 0:
                        boxes_slice = boxes[0, keep_indices].cpu().numpy()
                        scores_slice = scores[0, keep_indices].cpu().numpy()

                        # Lưu lại cùng với thông tin slice hiện tại
                        for box, score in zip(boxes_slice, scores_slice):
                            boxes_all.append([box[0], box[1], i, box[2], box[3], i])
                            scores_all.append(score)
                except Exception as e:
                    print(f"Lỗi khi xử lý slice {i}: {e}")
                    continue

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

    # Tạo đầu vào 3 kênh cho mô hình Tri2D-Net
    # Mô hình yêu cầu input có hình dạng (channels, depth, height, width)
    heart_ct_3channel = np.stack([heart_ct, heart_ct, heart_ct], axis=0)

    print(f"Kích thước đầu vào sau khi chuẩn bị: {heart_ct_3channel.shape}")

    return heart_ct_3channel


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
            checkpoint_path = os.path.join(model_path, CHECKPOINT_PATH)
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
            Mảng đã được tiền xử lý, kích thước (3, 128, 128, 128)

        Returns:
        --------
        float: Điểm nguy cơ CVD trong khoảng [0, 1]
        """
        if self.model is None:
            raise ValueError("Mô hình chưa được tải thành công.")

        print("Đang dự đoán nguy cơ CVD...")

        # Chuyển đổi sang tensor
        ct_tensor = torch.from_numpy(processed_ct).float()

        # Kiểm tra kích thước đầu vào
        if (
            ct_tensor.dim() == 4
        ):  # Đầu vào có hình dạng (channels, depth, height, width)
            if ct_tensor.shape[0] < 2:  # Nếu chỉ có 1 kênh
                print("Input chỉ có 1 kênh, nhân bản thành 3 kênh...")
                ct_tensor = ct_tensor.repeat(3, 1, 1, 1)  # Nhân bản kênh

        print(f"Kích thước tensor đầu vào: {ct_tensor.shape}")

        # Thực hiện dự đoán
        with torch.no_grad():
            try:
                # Thử sử dụng phương thức aug_transform của mô hình
                pred_prob = self.model.aug_transform(ct_tensor)
                print(
                    f"pred_prob shape: {pred_prob.shape if hasattr(pred_prob, 'shape') else 'scalar'}"
                )
                print(f"pred_prob: {pred_prob}")

                # Xử lý kết quả tùy theo định dạng
                if isinstance(pred_prob, (int, float)):
                    risk_score = float(pred_prob)
                elif isinstance(pred_prob, torch.Tensor):
                    if pred_prob.numel() == 1:  # Chỉ có một giá trị
                        risk_score = pred_prob.item()
                    elif len(pred_prob.shape) == 0:  # Tensor scalar
                        risk_score = pred_prob.item()
                    elif pred_prob.shape[0] == 1:  # Một phần tử trong tensor
                        risk_score = float(pred_prob[0])
                    elif pred_prob.shape[0] >= 2:  # Nhiều phần tử
                        # Đây là trường hợp nhị phân, ta lấy xác suất của lớp dương (positive class)
                        risk_score = float(pred_prob[1])
                    else:
                        # Trường hợp không xác định, lấy giá trị trung bình
                        risk_score = float(pred_prob.mean())
                else:
                    print(
                        f"WARNING: Không nhận dạng được kiểu dữ liệu đầu ra: {type(pred_prob)}"
                    )
                    risk_score = 0.5

                # Đảm bảo risk_score nằm trong khoảng [0, 1]
                risk_score = max(0.0, min(1.0, float(risk_score)))

            except Exception as e:
                print(f"Lỗi trong quá trình dự đoán: {e}")
                import traceback

                traceback.print_exc()

                # Thử phương pháp dự đoán thay thế
                try:
                    print("Thử phương pháp dự đoán thay thế...")
                    # Sử dụng encoder trực tiếp nếu có thể
                    if hasattr(self.model, "encoder"):
                        features = self.model.encoder(ct_tensor)
                        risk_score = torch.sigmoid(features.mean()).item()
                    else:
                        risk_score = 0.5
                except Exception as e2:
                    print(f"Phương pháp thay thế cũng thất bại: {e2}")
                    risk_score = 0.5

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


def debug_model_output(self, processed_ct):
    """
    Hàm debug để in ra chi tiết về đầu ra của mô hình
    """
    print("\n=== DEBUG MODEL OUTPUT ===")

    if self.model is None:
        print("Model is None, cannot debug")
        return

    # Chuyển đổi sang tensor
    ct_tensor = torch.from_numpy(processed_ct).float()
    print(f"Input shape: {ct_tensor.shape}")

    # Đảm bảo có đủ số kênh
    if ct_tensor.shape[0] < 3:
        ct_tensor = ct_tensor.repeat(3, 1, 1, 1)
        print(f"Expanded to shape: {ct_tensor.shape}")

    # Kiểm tra các thuộc tính model
    print("Model attributes:")
    for attr_name in dir(self.model):
        if not attr_name.startswith("_"):  # Skip private attributes
            attr = getattr(self.model, attr_name)
            if not callable(attr):
                print(f"  {attr_name}: {type(attr)}")

    # Thực hiện dự đoán với nhiều cách khác nhau
    with torch.no_grad():
        try:
            # Try direct encoder access
            if hasattr(self.model, "encoder"):
                print("\nTrying direct encoder access:")
                features = self.model.encoder(ct_tensor)
                print(f"Type: {type(features)}")
                if isinstance(features, torch.Tensor):
                    print(f"Shape: {features.shape}")
                    print(f"Values: {features}")
        except Exception as e:
            print(f"Encoder access error: {e}")

        try:
            # Try forward pass
            if hasattr(self.model, "forward"):
                print("\nTrying forward pass:")
                output = self.model(ct_tensor)
                print(f"Type: {type(output)}")
                if isinstance(output, torch.Tensor):
                    print(f"Shape: {output.shape}")
                    print(f"Values: {output}")
        except Exception as e:
            print(f"Forward pass error: {e}")

        try:
            # Try aug_transform method
            if hasattr(self.model, "aug_transform"):
                print("\nTrying aug_transform method:")
                output = self.model.aug_transform(ct_tensor)
                print(f"Type: {type(output)}")
                if isinstance(output, torch.Tensor):
                    print(f"Shape: {output.shape}")
                    print(f"Values: {output}")
        except Exception as e:
            print(f"aug_transform error: {e}")
            import traceback

            traceback.print_exc()

    print("=== END DEBUG ===\n")


def main(dicom_dir, visualize=True, debug=False):
    """
    Hàm chính để thực hiện toàn bộ quy trình

    Parameters:
    -----------
    dicom_dir: str
        Đường dẫn đến thư mục chứa các file DICOM
    visualize: bool
        Nếu True, hiển thị kết quả trực quan
    debug: bool
        Nếu True, hiển thị thông tin debug
    """
    try:
        # 1. Đọc ảnh DICOM
        ct_volume, metadata = load_dicom_series(dicom_dir)
        logging.info(f"Đã đọc CT volume với kích thước {ct_volume.shape}")

        if debug:
            # Save a few sample slices
            mid_slice = ct_volume.shape[0] // 2
            for i, idx in enumerate([mid_slice - 10, mid_slice, mid_slice + 10]):
                if 0 <= idx < ct_volume.shape[0]:
                    plt.imsave(f"debug_slice_{idx}.png", ct_volume[idx], cmap="gray")

            print("CT volume shape:", ct_volume.shape)
            print("CT value range:", np.min(ct_volume), np.max(ct_volume))

        # 2. Phát hiện vùng tim
        heart_detector = HeartDetector()
        heart_bbox = heart_detector._simple_heart_detection(
            ct_volume
        )  # Use simple detector for now
        logging.info(f"Vùng tim: {heart_bbox}")

        # 3. Tiền xử lý ảnh CT - với 3 kênh
        processed_ct = preprocess_heart_ct(
            ct_volume, heart_bbox
        )  # Cập nhật hàm này để trả về 3 kênh
        logging.info(f"Kích thước sau tiền xử lý: {processed_ct.shape}")

        # 4. Tải mô hình và dự đoán
        model = Tri2DNetModel()

        if debug:
            # Run debug function
            model.debug_model_output(processed_ct)

        # Try the modified predict_risk method
        risk_score = model.predict_risk(processed_ct)
        logging.info(f"Điểm nguy cơ CVD: {risk_score:.5f}")

        # 5. Tạo báo cáo
        generate_report(metadata, risk_score)

        # 6. Trực quan hóa kết quả (nếu yêu cầu)
        if visualize:
            visualize_results(ct_volume, heart_bbox, risk_score)
            logging.info(f"Đã lưu kết quả trực quan vào 'cvd_risk_result.png'")

        return risk_score

    except Exception as e:
        print(f"Lỗi trong quá trình xử lý: {e}")
        logging.error(f"Lỗi: {e}")
        import traceback

        traceback.print_exc()
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
# python dicom_detect.py ""./dataset/Tuong_20230828"
