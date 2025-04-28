#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CVD Risk Predictor using Tri2D-Net
----------------------------------
Script để đọc ảnh DICOM, tiền xử lý và dự đoán nguy cơ CVD.
"""

import os
import argparse
from matplotlib import pyplot as plt
import numpy as np

from logger import logger, log_message

from dicom_processor import (
    load_dicom_series,
    preprocess_heart_ct,
    generate_report,
    visualize_results,
)

from cvd_model import Tri2DNetModel
from heart_detector import HeartDetector

def debug_visualization(ct_volume, output_dir="debug"):
    """
    Hàm tiện ích để lưu ảnh debug
    """
    os.makedirs(output_dir, exist_ok=True)
    mid_slice = ct_volume.shape[0] // 2
    for i, idx in enumerate([mid_slice - 10, mid_slice, mid_slice + 10]):
        if 0 <= idx < ct_volume.shape[0]:
            plt.imsave(
                os.path.join(output_dir, f"debug_slice_{idx}.png"),
                ct_volume[idx],
                cmap="gray",
            )

def main(dicom_dir, visualize=True, detection_method="auto", debug=False):
    """
    Hàm chính để thực hiện toàn bộ quy trình

    Parameters:
    -----------
    dicom_dir: str
        Đường dẫn đến thư mục chứa các file DICOM
    visualize: bool
        Nếu True, hiển thị kết quả trực quan
    detection_method: str
        Phương pháp phát hiện tim: auto, model, simple
    debug: bool
        Nếu True, hiển thị thông tin debug
    """
    try:
        # 1. Đọc ảnh DICOM
        ct_volume, metadata = load_dicom_series(dicom_dir)
        log_message(logger, "info", f"Đã đọc CT volume với kích thước {ct_volume.shape}")

        # Debug logging
        if debug:
            debug_visualization(ct_volume)
            log_message(logger, "debug", f"CT volume shape: {ct_volume.shape}")
            log_message(logger, "debug", f"CT value range: {np.min(ct_volume)}, {np.max(ct_volume)}")

        # 2. Phát hiện vùng tim
        heart_detector = HeartDetector()
        heart_region = heart_detector.detect_heart_region(ct_volume)
        log_message(logger, "info", f"Đã phát hiện vùng tim: {heart_region}")

        # 3. Tiền xử lý ảnh tim
        processed_ct = preprocess_heart_ct(ct_volume, heart_region)
        log_message(logger, "info", f"Đã tiền xử lý ảnh tim, kích thước: {processed_ct.shape}")

        # 4. Dự đoán nguy cơ CVD
        model = Tri2DNetModel()
        risk_score = model.predict_risk(processed_ct)
        log_message(logger, "info", f"Điểm nguy cơ CVD: {risk_score:.5f}")

        # 5. Tạo báo cáo
        report = generate_report(metadata, risk_score)
        log_message(logger, "info", "Đã tạo báo cáo")

        # 6. Hiển thị kết quả
        if visualize:
            visualize_results(ct_volume, heart_region, risk_score)
            log_message(logger, "info", "Đã hiển thị kết quả trực quan")

        return report

    except Exception as e:
        log_message(logger, "error", f"Lỗi trong quá trình xử lý: {e}")
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
    parser.add_argument(
        "--detection-method",
        choices=["auto", "model", "simple"],
        default="auto",
        help="Phương pháp phát hiện tim: auto (tự động), model (chỉ dùng mô hình), simple (phương pháp đơn giản)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Bật chế độ debug",
    )
    parser.set_defaults(visualize=True)

    args = parser.parse_args()
    main(args.dicom_dir, args.visualize, args.detection_method, args.debug)

# python dicom_detect.py "./uploads/Tuong_20230828"
