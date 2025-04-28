
import os
import numpy as np
import torch

from logger import logger,  log_message
from config import MODEL_CONFIG

class HeartDetector:
    """
    Class để phát hiện vùng tim từ ảnh CT
    """

    def __init__(self, model_path=MODEL_CONFIG["RETINANET_PATH"]):
        """
        Khởi tạo detector với mô hình RetinaNet đã được huấn luyện
        """
        log_message(logger, "info", "Khởi tạo detector vùng tim...")

        # Kiểm tra đường dẫn mô hình có tồn tại không
        if not os.path.exists(model_path):
            log_message(
                logger,
                "info",
                f"Thư mục mô hình {model_path} không tồn tại, tạo thư mục mới.",
            )
            try:
                os.makedirs(model_path, exist_ok=True)
            except Exception as e:
                log_message(logger, "error", f"Không thể tạo thư mục: {e}")

        # Kiểm tra CUDA
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log_message(logger, "info", f"Sử dụng thiết bị: {self.device}")

        try:
            # Tải mô hình RetinaNet
            if os.path.exists(model_path):
                log_message(logger, "info", f"Loading model at {model_path}...")
                # Thêm weights_only=True để tránh warning
                self.model = torch.load(model_path, map_location=self.device)
                self.model.eval()
                log_message(logger, "info", "Detector model loaded successfully.")
            else:
                log_message(logger, "warning", f"Model not found at: {model_path}")
        except Exception as e:
            log_message(logger, "error", f"Failed to load the detector model: {e}")
            log_message(logger, "info", "Use simple detector...")
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

        log_message(
            logger,
            "info",
            f"Sử dụng phương pháp simple detection: [{x_min}, {y_min}, {z_min}] - [{x_max}, {y_max}, {z_max}]",
        )
        return (x_min, y_min, z_min, x_max, y_max, z_max)

    def _process_detection_output(self, output):
        """
        Xử lý output từ mô hình detector
        """
        if isinstance(output, tuple) and len(output) >= 2:
            return output[0], output[1]
        elif isinstance(output, list) and len(output) >= 2:
            return output[0], output[1]
        elif isinstance(output, dict) and "scores" in output and "boxes" in output:
            return output["scores"], output["boxes"]
        return None, None

    def _process_scores_and_boxes(self, scores, boxes):
        """
        Xử lý scores và boxes từ mô hình detector
        """
        if isinstance(scores, (int, float)):
            if (
                isinstance(boxes, (list, tuple, np.ndarray, torch.Tensor))
                and len(boxes) >= 4
            ):
                return [boxes], [scores]
            return None, None

        if isinstance(scores, (np.ndarray, torch.Tensor)):
            if isinstance(scores, np.ndarray):
                keep_indices = np.where(scores > 0.3)[0]
            else:  # torch.Tensor
                if len(scores.shape) > 1:
                    keep_indices = torch.where(scores[0] > 0.3)[0]
                else:
                    keep_indices = torch.where(scores > 0.3)[0]
                keep_indices = keep_indices.cpu().numpy()

            if len(keep_indices) == 0:
                return None, None

            if isinstance(scores, np.ndarray):
                if len(scores.shape) > 1:
                    scores_slice = scores[0, keep_indices]
                else:
                    scores_slice = scores[keep_indices]
            else:  # torch.Tensor
                if len(scores.shape) > 1:
                    scores_slice = scores[0, keep_indices].cpu().numpy()
                else:
                    scores_slice = scores[keep_indices].cpu().numpy()

            if isinstance(boxes, np.ndarray):
                if len(boxes.shape) > 1:
                    if len(boxes.shape) > 2:
                        boxes_slice = boxes[0, keep_indices].copy()
                    else:
                        boxes_slice = boxes[keep_indices].copy()
                else:
                    boxes_slice = [boxes]
            elif isinstance(boxes, torch.Tensor):
                if len(boxes.shape) > 1:
                    if len(boxes.shape) > 2:
                        boxes_slice = boxes[0, keep_indices].cpu().numpy()
                    else:
                        boxes_slice = boxes[keep_indices].cpu().numpy()
                else:
                    boxes_slice = [boxes.cpu().numpy()]
            else:
                return None, None

            return boxes_slice, scores_slice

        return None, None

    def detect_heart_region(self, ct_volume):
        """
        Phát hiện vùng tim từ ảnh CT
        """
        log_message(logger, "info", "Đang phát hiện vùng tim...")

        if self.model is None:
            log_message(
                logger,
                "info",
                "Không tìm thấy mô hình detector, sử dụng phương pháp simple detection.",
            )
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
                    output = self.model(img_tensor)
                    scores, boxes = self._process_detection_output(output)
                    if scores is None or boxes is None:
                        continue

                    boxes_slice, scores_slice = self._process_scores_and_boxes(
                        scores, boxes
                    )
                    if boxes_slice is None or scores_slice is None:
                        continue

                    # Thêm boxes và scores vào danh sách
                    for box, score in zip(boxes_slice, scores_slice):
                        if isinstance(box, (list, tuple, np.ndarray)) and len(box) >= 4:
                            x1 = float(box[0])
                            y1 = float(box[1])
                            x2 = float(box[2])
                            y2 = float(box[3])

                            boxes_all.append([x1, y1, i, x2, y2, i])
                            scores_all.append(float(score))

                except Exception as e:
                    log_message(
                        logger, "error", f"Error processing detection results: {e}"
                    )
                    continue

        # If no heart region was detected, use simple detection
        if not boxes_all:
            log_message(
                logger,
                "info",
                "Không phát hiện được vùng tim, sử dụng phương pháp simple detection.",
            )
            return self._simple_heart_detection(ct_volume)

        # Process detection results
        try:
            boxes_all = np.array(boxes_all)
            scores_all = np.array(scores_all)

            # Get the box with the highest score
            best_idx = np.argmax(scores_all)
            x_min, y_min, z_min, x_max, y_max, z_max = boxes_all[best_idx]

            # Expand the box to ensure capturing the entire heart
            width = x_max - x_min
            height = y_max - y_min
            depth = max(2, z_max - z_min)  # Ensure minimal depth

            x_min = max(0, x_min - width * 0.1)
            y_min = max(0, y_min - height * 0.1)
            z_min = max(0, z_min - depth)

            x_max = min(ct_volume.shape[2], x_max + width * 0.1)
            y_max = min(ct_volume.shape[1], y_max + height * 0.1)
            z_max = min(ct_volume.shape[0], z_max + depth)

            log_message(
                logger,
                "info",
                f"Đã phát hiện vùng tim: [{x_min:.1f}, {y_min:.1f}, {z_min:.1f}] - [{x_max:.1f}, {y_max:.1f}, {z_max:.1f}]",
            )
            return (
                int(x_min),
                int(y_min),
                int(z_min),
                int(x_max),
                int(y_max),
                int(z_max),
            )
        except Exception as e:
            log_message(logger, "error", f"Error processing detection results: {e}")
            return self._simple_heart_detection(ct_volume)
