import os
import sys
import torch
import logging
import numpy as np
from config import (
    FOLDERS,
    MODEL_CONFIG,
    ERROR_MESSAGES,
    LOG_CONFIG,
    IS_DEV
)

# Thêm đường dẫn đến thư mục detector từ config
sys.path.append(FOLDERS["DETECTOR"])

logger = logging.getLogger(__name__)

class HeartDetector:
    """
    Class để phát hiện vùng tim từ ảnh CT
    """

    def __init__(self, model_path=MODEL_CONFIG["RETINANET_PATH"]):
        """
        Khởi tạo detector với mô hình RetinaNet đã được huấn luyện
        """
        if IS_DEV:
            print("Khởi tạo detector vùng tim...")
        logger.info("Khởi tạo detector vùng tim...")

        # Kiểm tra đường dẫn mô hình có tồn tại không
        if not os.path.exists(model_path):
            if IS_DEV:
                print(f"Thư mục mô hình {model_path} không tồn tại, tạo thư mục mới.")
            logger.info(f"Thư mục mô hình {model_path} không tồn tại, tạo thư mục mới.")
            try:
                os.makedirs(model_path, exist_ok=True)
            except Exception as e:
                logger.error(f"Không thể tạo thư mục: {e}")

        # Kiểm tra CUDA
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if IS_DEV:
            print(f"Sử dụng thiết bị: {self.device}")
        logger.info(f"Sử dụng thiết bị: {self.device}")

        try:
            # Tải mô hình RetinaNet
            model_file = os.path.join(model_path, "retinanet_heart.pt")
            if os.path.exists(model_file):
                if IS_DEV:
                    print(f"Đang tải mô hình từ {model_file}...")
                logger.info(f"Đang tải mô hình từ {model_file}...")
                # Thêm weights_only=True để tránh warning
                self.model = torch.load(model_file, map_location=self.device)
                self.model.eval()
                if IS_DEV:
                    print("Đã tải mô hình detector thành công.")
                logger.info("Đã tải mô hình detector thành công.")
            else:
                if IS_DEV:
                    print(f"Không tìm thấy file mô hình tại: {model_file}")
                logger.warning(f"Không tìm thấy file mô hình tại: {model_file}")
                # Thử tìm mô hình trong thư mục detector từ config
                alt_model_file = os.path.join(FOLDERS["DETECTOR"], "retinanet_heart.pt")
                if os.path.exists(alt_model_file):
                    if IS_DEV:
                        print(f"Tìm thấy mô hình tại đường dẫn thay thế: {alt_model_file}")
                    logger.info(f"Tìm thấy mô hình tại đường dẫn thay thế: {alt_model_file}")
                    self.model = torch.load(alt_model_file, map_location=self.device)
                    self.model.eval()
                    if IS_DEV:
                        print("Đã tải mô hình detector thành công.")
                    logger.info("Đã tải mô hình detector thành công.")
                else:
                    if IS_DEV:
                        print("Không tìm thấy mô hình detector.")
                    logger.warning("Không tìm thấy mô hình detector.")
                    self.model = None
        except Exception as e:
            if IS_DEV:
                print(f"Lỗi khi tải mô hình detector: {e}")
            logger.error(f"Lỗi khi tải mô hình detector: {e}")
            if IS_DEV:
                print("Sử dụng simple detector...")
            logger.info("Sử dụng simple detector...")
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

        if IS_DEV:
            print(
                f"Sử dụng phương pháp simple detection: [{x_min}, {y_min}, {z_min}] - [{x_max}, {y_max}, {z_max}]"
            )
        logger.info(
            f"Sử dụng phương pháp simple detection: [{x_min}, {y_min}, {z_min}] - [{x_max}, {y_max}, {z_max}]"
        )
        return (x_min, y_min, z_min, x_max, y_max, z_max)

    def detect_heart_region(self, ct_volume):
        """
        Phát hiện vùng tim từ ảnh CT
        """
        if IS_DEV:
            print("Đang phát hiện vùng tim...")
        logger.info("Đang phát hiện vùng tim...")

        if self.model is None:
            if IS_DEV:
                print(
                    "Không tìm thấy mô hình detector, sử dụng phương pháp simple detection."
                )
            logger.info(
                "Không tìm thấy mô hình detector, sử dụng phương pháp simple detection."
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
                    
                    # Xử lý output dựa vào kiểu
                    if isinstance(output, tuple):
                        if len(output) >= 2:
                            scores, boxes = output[0], output[1]
                    elif isinstance(output, list):
                        if len(output) >= 2:
                            scores, boxes = output[0], output[1]
                    elif isinstance(output, dict):
                        if "scores" in output and "boxes" in output:
                            scores = output["scores"]
                            boxes = output["boxes"]
                    else:
                        continue

                    # Xử lý scores và boxes
                    if isinstance(scores, (int, float)):
                        if isinstance(boxes, (list, tuple, np.ndarray, torch.Tensor)):
                            if len(boxes) >= 4:
                                boxes_slice = [boxes]
                                scores_slice = [scores]
                            else:
                                continue
                        else:
                            continue
                    elif isinstance(scores, (np.ndarray, torch.Tensor)):
                        if isinstance(scores, np.ndarray):
                            keep_indices = np.where(scores > 0.3)[0]
                            if len(keep_indices) > 0:
                                if len(scores.shape) > 1:
                                    scores_slice = scores[0, keep_indices]
                                else:
                                    scores_slice = scores[keep_indices]
                            else:
                                continue
                        else:  # torch.Tensor
                            if len(scores.shape) > 1:
                                keep_indices = torch.where(scores[0] > 0.3)[0]
                                if len(keep_indices) > 0:
                                    scores_slice = scores[0, keep_indices].cpu().numpy()
                                else:
                                    continue
                            else:
                                keep_indices = torch.where(scores > 0.3)[0]
                                if len(keep_indices) > 0:
                                    scores_slice = scores[keep_indices].cpu().numpy()
                                else:
                                    continue

                        # Lấy boxes tương ứng
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
                    if IS_DEV:
                        print(f"Error processing detection results: {e}")
                    logger.error(f"Error processing detection results: {e}")
                    continue

        # If no heart region was detected, use simple detection
        if not boxes_all:
            if IS_DEV:
                print(
                    "Không phát hiện được vùng tim, sử dụng phương pháp simple detection."
                )
            logger.info(
                "Không phát hiện được vùng tim, sử dụng phương pháp simple detection."
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

            if IS_DEV:
                print(
                    f"Đã phát hiện vùng tim: [{x_min:.1f}, {y_min:.1f}, {z_min:.1f}] - [{x_max:.1f}, {y_max:.1f}, {z_max:.1f}]"
                )
            logger.info(
                f"Đã phát hiện vùng tim: [{x_min:.1f}, {y_min:.1f}, {z_min:.1f}] - [{x_max:.1f}, {y_max:.1f}, {z_max:.1f}]"
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
            if IS_DEV:
                print(f"Error processing detection results: {e}")
            logger.error(f"Error processing detection results: {e}")
            return self._simple_heart_detection(ct_volume)

class Tri2DNetModel:
    """
    Class để load và sử dụng mô hình Tri2D-Net
    """

    def __init__(self, model_path=MODEL_CONFIG["MODEL_PATH"]):
        """
        Khởi tạo và tải mô hình Tri2D-Net
        """
        if IS_DEV:
            print("Đang khởi tạo mô hình Tri2D-Net...")
        logger.info("Đang khởi tạo mô hình Tri2D-Net...")

        try:
            # Import model module
            sys.path.append("./")  # Đảm bảo có thể import từ thư mục hiện tại
            from tri_2d_net.init_model import init_model  # Sử dụng init_model từ tri_2d_net

            # Khởi tạo model
            self.model = init_model()  # Sử dụng init_model để khởi tạo

            # Tải checkpoint
            checkpoint_path = os.path.join(model_path, MODEL_CONFIG["CHECKPOINT_PATH"])
            if not os.path.exists(checkpoint_path):
                if IS_DEV:
                    print(f"Không tìm thấy checkpoint tại: {checkpoint_path}")
                logger.warning(f"Không tìm thấy checkpoint tại: {checkpoint_path}")
                # Thử tìm trong thư mục hiện tại
                alt_checkpoint_path = MODEL_CONFIG["CHECKPOINT_PATH"]
                if os.path.exists(alt_checkpoint_path):
                    if IS_DEV:
                        print(f"Tìm thấy checkpoint tại đường dẫn thay thế: {alt_checkpoint_path}")
                    logger.info(f"Tìm thấy checkpoint tại đường dẫn thay thế: {alt_checkpoint_path}")
                    checkpoint_path = alt_checkpoint_path
                else:
                    raise ValueError(ERROR_MESSAGES["model_not_found"])

            # Tải state dict
            state_dict = torch.load(
                checkpoint_path,
                map_location="cuda" if torch.cuda.is_available() else "cpu",
            )
            
            # Kiểm tra và tải state dict
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            
            # Tải state dict vào model
            self.model.encoder.load_state_dict(state_dict)
            self.model.encoder.eval()

            if IS_DEV:
                print("Đã tải mô hình Tri2D-Net thành công.")
            logger.info("Đã tải mô hình Tri2D-Net thành công.")
        except Exception as e:
            if IS_DEV:
                print(f"Lỗi khi tải mô hình Tri2D-Net: {e}")
            logger.error(f"Lỗi khi tải mô hình Tri2D-Net: {e}")
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
            raise ValueError(ERROR_MESSAGES["model_not_found"])

        if IS_DEV:
            print("Đang dự đoán nguy cơ CVD...")
        logger.info("Đang dự đoán nguy cơ CVD...")

        # Chuyển đổi sang tensor
        ct_tensor = torch.from_numpy(processed_ct).float()

        # Kiểm tra kích thước đầu vào
        if ct_tensor.dim() == 4:
            if ct_tensor.shape[0] < 2:
                if IS_DEV:
                    print("Input chỉ có 1 kênh, nhân bản thành 3 kênh...")
                logger.info("Input chỉ có 1 kênh, nhân bản thành 3 kênh...")
                ct_tensor = ct_tensor.repeat(3, 1, 1, 1)

        if IS_DEV:
            print(f"Kích thước tensor đầu vào: {ct_tensor.shape}")
        logger.info(f"Kích thước tensor đầu vào: {ct_tensor.shape}")

        # Thực hiện dự đoán
        with torch.no_grad():
            try:
                # Gọi aug_transform của model
                pred_prob = self.model.aug_transform(ct_tensor)
                
                if IS_DEV:
                    print(
                        f"pred_prob shape: {pred_prob.shape if hasattr(pred_prob, 'shape') else 'scalar'}"
                    )
                    print(f"pred_prob: {pred_prob}")
                logger.info(
                    f"pred_prob shape: {pred_prob.shape if hasattr(pred_prob, 'shape') else 'scalar'}"
                )
                logger.info(f"pred_prob: {pred_prob}")

                # Lấy giá trị xác suất
                if isinstance(pred_prob, (int, float)):
                    risk_score = float(pred_prob)
                elif isinstance(pred_prob, torch.Tensor):
                    if pred_prob.numel() == 1:
                        risk_score = pred_prob.item()
                    elif len(pred_prob.shape) == 0:
                        risk_score = pred_prob.item()
                    elif pred_prob.shape[0] == 1:
                        risk_score = float(pred_prob[0])
                    elif pred_prob.shape[0] >= 2:
                        risk_score = float(pred_prob[1])
                    else:
                        risk_score = float(pred_prob.mean())
                elif isinstance(pred_prob, np.ndarray):
                    if pred_prob.size == 1:
                        risk_score = float(pred_prob.item())
                    elif pred_prob.shape[0] == 1:
                        risk_score = float(pred_prob[0])
                    elif pred_prob.shape[0] >= 2:
                        risk_score = float(pred_prob[1])
                    else:
                        risk_score = float(np.mean(pred_prob))
                else:
                    if IS_DEV:
                        print(
                            f"WARNING: Không nhận dạng được kiểu dữ liệu đầu ra: {type(pred_prob)}"
                        )
                    logger.warning(
                        f"WARNING: Không nhận dạng được kiểu dữ liệu đầu ra: {type(pred_prob)}"
                    )
                    risk_score = 0.5

                # Đảm bảo risk_score nằm trong khoảng [0, 1]
                risk_score = max(0.0, min(1.0, float(risk_score)))

            except Exception as e:
                if IS_DEV:
                    print(f"Lỗi trong quá trình dự đoán: {e}")
                logger.error(f"Lỗi trong quá trình dự đoán: {e}")
                import traceback
                traceback.print_exc()
                risk_score = 0.5

        if IS_DEV:
            print(f"Điểm nguy cơ CVD đã dự đoán: {risk_score:.5f}")
        logger.info(f"Điểm nguy cơ CVD đã dự đoán: {risk_score:.5f}")
        return risk_score 