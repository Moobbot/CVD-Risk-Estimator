#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CVD Risk Predictor using Tri2D-Net
----------------------------------
Script để đọc ảnh DICOM, tiền xử lý và dự đoán nguy cơ CVD.
"""

import os
import sys
import argparse
import logging
from datetime import datetime

from matplotlib import pyplot as plt
import numpy as np

from config import (
    FOLDERS,
    MODEL_CONFIG,
    ERROR_MESSAGES,
    LOG_CONFIG,
    IS_DEV
)

from dicom_processor import (
    load_dicom_series,
    preprocess_heart_ct,
    generate_report,
    visualize_results,
    get_risk_details
)

from model_processor import (
    HeartDetector,
    Tri2DNetModel
)

# Setup logging
logger = logging.getLogger(__name__)

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
        if IS_DEV:
            print(f"Đã đọc CT volume với kích thước {ct_volume.shape}")
        logger.info(f"Đã đọc CT volume với kích thước {ct_volume.shape}")

        # Debug logging
        if debug:
            mid_slice = ct_volume.shape[0] // 2
            for i, idx in enumerate([mid_slice - 10, mid_slice, mid_slice + 10]):
                if 0 <= idx < ct_volume.shape[0]:
                    plt.imsave(f"debug_slice_{idx}.png", ct_volume[idx], cmap="gray")
            if IS_DEV:
                print("CT volume shape:", ct_volume.shape)
                print("CT value range:", np.min(ct_volume), np.max(ct_volume))
            logger.debug(f"CT volume shape: {ct_volume.shape}")
            logger.debug(f"CT value range: {np.min(ct_volume)}, {np.max(ct_volume)}")

        # 2. Phát hiện vùng tim
        heart_detector = HeartDetector()

        # Lựa chọn phương pháp phát hiện tim
        if detection_method == "simple":
            if IS_DEV:
                print(
                    "Sử dụng phương pháp phát hiện tim đơn giản theo lựa chọn của người dùng."
                )
            logger.info(
                "Sử dụng phương pháp phát hiện tim đơn giản theo lựa chọn của người dùng."
            )
            heart_bbox = heart_detector._simple_heart_detection(ct_volume)
        elif detection_method == "model":
            if heart_detector.model is None:
                if IS_DEV:
                    print(
                        "Yêu cầu dùng mô hình nhưng mô hình không có sẵn. Sử dụng phương pháp đơn giản."
                    )
                logger.warning(
                    "Yêu cầu dùng mô hình nhưng mô hình không có sẵn. Sử dụng phương pháp đơn giản."
                )
                heart_bbox = heart_detector._simple_heart_detection(ct_volume)
            else:
                if IS_DEV:
                    print("Sử dụng mô hình phát hiện tim theo lựa chọn của người dùng.")
                logger.info("Sử dụng mô hình phát hiện tim theo lựa chọn của người dùng.")
                heart_bbox = heart_detector.detect_heart_region(ct_volume)
        else:  # auto
            if IS_DEV:
                print("Tự động lựa chọn phương pháp phát hiện tim tốt nhất.")
            logger.info("Tự động lựa chọn phương pháp phát hiện tim tốt nhất.")
            heart_bbox = heart_detector.detect_heart_region(ct_volume)

        # Debug detector
        if debug:
            heart_detector.debug_detection(ct_volume)

        if IS_DEV:
            print(f"Vùng tim: {heart_bbox}")
        logger.info(f"Vùng tim: {heart_bbox}")

        # 3. Tiền xử lý ảnh CT - với 3 kênh
        processed_ct = preprocess_heart_ct(ct_volume, heart_bbox)
        if IS_DEV:
            print(f"Kích thước sau tiền xử lý: {processed_ct.shape}")
        logger.info(f"Kích thước sau tiền xử lý: {processed_ct.shape}")

        # 4. Tải mô hình và dự đoán
        model = Tri2DNetModel()

        # Try the modified predict_risk method
        risk_score = model.predict_risk(processed_ct)
        if IS_DEV:
            print(f"Điểm nguy cơ CVD: {risk_score:.5f}")
        logger.info(f"Điểm nguy cơ CVD: {risk_score:.5f}")

        # Lấy thông tin chi tiết về mức độ nguy cơ
        risk_details = get_risk_details(risk_score)
        if IS_DEV:
            print(
                f"Điểm nguy cơ CVD: {risk_score:.5f} - Mức độ: {risk_details['risk_level']}"
            )
            print(f"Khuyến nghị: {', '.join(risk_details['recommendations'])}")
        logger.info(
            f"Điểm nguy cơ CVD: {risk_score:.5f} - Mức độ: {risk_details['risk_level']}"
        )
        logger.info(f"Khuyến nghị: {', '.join(risk_details['recommendations'])}")

        # 5. Tạo báo cáo
        generate_report(metadata, risk_score)

        # 6. Trực quan hóa kết quả (nếu yêu cầu)
        if visualize:
            visualize_results(ct_volume, heart_bbox, risk_score)
            if IS_DEV:
                print(f"Đã lưu kết quả trực quan vào 'cvd_risk_result.png'")
            logger.info(f"Đã lưu kết quả trực quan vào 'cvd_risk_result.png'")

        # 7. Lưu thông tin chi tiết dưới dạng JSON
        if debug:
            import json

            details = {
                "patient_id": metadata.get("PatientID", "N/A"),
                "risk_score": float(risk_score),
                "risk_details": risk_details,
                "prediction_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            with open("cvd_prediction_details.json", "w") as f:
                json.dump(details, f, indent=4)
            if IS_DEV:
                print("Đã lưu thông tin chi tiết vào 'cvd_prediction_details.json'")
            logger.info("Đã lưu thông tin chi tiết vào 'cvd_prediction_details.json'")

        return risk_score

    except Exception as e:
        if IS_DEV:
            print(f"Lỗi trong quá trình xử lý: {e}")
        logger.error(f"Lỗi: {e}")
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

    # Chọn phương pháp phát hiện tim
    detection_method = args.detection_method

    main(args.dicom_dir, args.visualize, detection_method, args.debug)

# python dicom_detect.py "./uploads/Tuong_20230828"
