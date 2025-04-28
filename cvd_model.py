import os
import sys
import torch
import numpy as np

from config import FOLDERS_DETECTOR, MODEL_CONFIG, ERROR_MESSAGES
from logger import logger, log_message

# Thêm đường dẫn đến thư mục detector từ config
sys.path.append(FOLDERS_DETECTOR)


class Tri2DNetModel:
    """
    Class để load và sử dụng mô hình Tri2D-Net
    """

    def __init__(self, model_path=MODEL_CONFIG["CHECKPOINT_PATH"]):
        """
        Khởi tạo và tải mô hình Tri2D-Net
        """
        log_message(logger, "info", "Đang khởi tạo mô hình Tri2D-Net...")

        try:
            # Import model module
            sys.path.append("./")  # Đảm bảo có thể import từ thư mục hiện tại
            from tri_2d_net.init_model import init_model

            # Khởi tạo model
            self.model = init_model()  # Sử dụng init_model để khởi tạo

            # Tải checkpoint
            if not os.path.exists(model_path):
                log_message(
                    logger,
                    "warning",
                    f"Không tìm thấy checkpoint tại: {model_path}",
                )

            # Tải state dict
            state_dict = torch.load(
                model_path,
                map_location="cuda" if torch.cuda.is_available() else "cpu",
            )

            # Kiểm tra và tải state dict
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]

            # Tải state dict vào model
            self.model.encoder.load_state_dict(state_dict)
            self.model.encoder.eval()

            log_message(logger, "info", "Đã tải mô hình Tri2D-Net thành công.")
        except Exception as e:
            log_message(logger, "error", f"Lỗi khi tải mô hình Tri2D-Net: {e}")
            self.model = None

    def _process_prediction_output(self, pred_prob):
        """
        Xử lý output từ mô hình dự đoán
        """
        if isinstance(pred_prob, (int, float)):
            return float(pred_prob)
        elif isinstance(pred_prob, torch.Tensor):
            if pred_prob.numel() == 1 or len(pred_prob.shape) == 0:
                return pred_prob.item()
            elif pred_prob.shape[0] == 1:
                return float(pred_prob[0])
            elif pred_prob.shape[0] >= 2:
                return float(pred_prob[1])
            return float(pred_prob.mean())
        elif isinstance(pred_prob, np.ndarray):
            if pred_prob.size == 1:
                return float(pred_prob.item())
            elif pred_prob.shape[0] == 1:
                return float(pred_prob[0])
            elif pred_prob.shape[0] >= 2:
                return float(pred_prob[1])
            return float(np.mean(pred_prob))
        else:
            log_message(
                logger,
                "warning",
                f"WARNING: Không nhận dạng được kiểu dữ liệu đầu ra: {type(pred_prob)}",
            )
            return 0.5

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

        log_message(logger, "info", "Đang dự đoán nguy cơ CVD...")

        # Chuyển đổi sang tensor
        ct_tensor = torch.from_numpy(processed_ct).float()

        # Kiểm tra kích thước đầu vào
        if ct_tensor.dim() == 4:
            if ct_tensor.shape[0] < 2:
                log_message(
                    logger, "info", "Input chỉ có 1 kênh, nhân bản thành 3 kênh..."
                )
                ct_tensor = ct_tensor.repeat(3, 1, 1, 1)

        log_message(logger, "info", f"Kích thước tensor đầu vào: {ct_tensor.shape}")

        # Thực hiện dự đoán
        with torch.no_grad():
            try:
                # Gọi aug_transform của model
                pred_prob = self.model.aug_transform(ct_tensor)

                log_message(
                    logger,
                    "info",
                    f"pred_prob shape: {pred_prob.shape if hasattr(pred_prob, 'shape') else 'scalar'}",
                )
                log_message(logger, "info", f"pred_prob: {pred_prob}")

                # Lấy giá trị xác suất
                risk_score = self._process_prediction_output(pred_prob)

                # Đảm bảo risk_score nằm trong khoảng [0, 1]
                risk_score = max(0.0, min(1.0, float(risk_score)))

            except Exception as e:
                log_message(logger, "error", f"Lỗi trong quá trình dự đoán: {e}")
                import traceback

                traceback.print_exc()
                risk_score = 0.5

        log_message(logger, "info", f"Điểm nguy cơ CVD đã dự đoán: {risk_score:.5f}")
        return risk_score
