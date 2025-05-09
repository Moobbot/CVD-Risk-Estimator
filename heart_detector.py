# -*- coding: utf-8 -*-
# @Author  : chq_N
# @Time    : 2019/12/10

from copy import deepcopy
import os
import logging
import sys

import cv2
import numpy as np
import torch

from config import MODEL_CONFIG

# Add detector directory to path to fix import issues
detector_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "detector")
if detector_path not in sys.path:
    sys.path.append(detector_path)


class HeartDetector:
    def __init__(self):
        self.model = None
        # Use the device from MODEL_CONFIG for consistency
        self.device = torch.device(MODEL_CONFIG["DEVICE"])
        print(f"Heart detector using device: {self.device}")

    def load_model(self):
        """Load the heart detection model"""
        try:
            model_path = MODEL_CONFIG["RETINANET_PATH"]
            if not os.path.exists(model_path):
                print(f"Heart detector model not found at: {model_path}")
                return False

            # Load model with explicit device mapping
            self.model = torch.load(model_path, map_location=self.device)
            self.model = self.model.to(self.device)
            self.model.eval()
            print("Heart detector model loaded successfully")
            return True
        except Exception as e:
            print(f"Failed to load heart detector model: {e}")
            import traceback
            traceback.print_exc()
            return False

    def draw_caption(self, image, box, caption):
        b = np.array(box).astype(int)
        cv2.putText(image, caption, (b[0], b[1] - 10),
                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
        cv2.putText(image, caption, (b[0], b[1] - 10),
                    cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

    def __visualize(self, pic, bbox, caption, selected):
        pic = (pic * 255).astype('uint8')
        self.draw_caption(pic, bbox, caption)
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        if selected:
            cv2.rectangle(pic, (x1, y1), (x2, y2), (0, 255, 0), 2)
        else:
            cv2.rectangle(pic, (x1, y1), (x2, y2), (0, 0, 255), 1)
        return pic

    def __calc_iou(self, a, b):
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        area_a = (a[2] - a[0]) * (a[3] - a[1])

        iw = min(a[2], b[2]) - max(a[0], b[0])
        ih = min(a[3], b[3]) - max(a[1], b[1])
        iw = np.clip(iw, 0, None)
        ih = np.clip(ih, 0, None)

        ua = area_a + area_b - iw * ih
        ua = np.clip(ua, 1e-8, None)
        intersection = iw * ih

        IoU = intersection / ua
        return IoU

    def __continue_smooth(self, bbox_selected):
        tmp_idx = [-1] * len(bbox_selected)
        tmp_len = [-1] * len(bbox_selected)
        max_len = 0
        r_idx = -1
        for i, t in enumerate(bbox_selected):
            if t == 0:
                continue
            elif i == 0 or tmp_idx[i - 1] == -1:
                tmp_idx[i] = i
                tmp_len[i] = 1
            else:
                tmp_idx[i] = tmp_idx[i - 1]
                tmp_len[i] = i - tmp_idx[i] + 1
            if max_len < tmp_len[i]:
                max_len = tmp_len[i]
                r_idx = i
        l_idx = tmp_idx[r_idx]
        smoothed_selected = np.zeros(len(bbox_selected))
        smoothed_selected[l_idx:r_idx + 1] = 1
        return smoothed_selected

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
        Phương pháp đơn giản để phát hiện vùng tim khi không có mô hình
        """
        # Lấy kích thước ảnh
        depth, height, width = ct_volume.shape

        # Tìm slice giữa
        mid_slice = depth // 2

        # Giả định vùng tim nằm ở giữa ảnh, chiếm khoảng 60% diện tích
        center_x, center_y = width // 2, height // 2
        heart_width = int(width * 0.6)
        heart_height = int(height * 0.6)

        x_min = max(0, center_x - heart_width // 2)
        y_min = max(0, center_y - heart_height // 2)
        x_max = min(width, center_x + heart_width // 2)
        y_max = min(height, center_y + heart_height // 2)

        # Giả định tim xuất hiện trong 60% số slice ở giữa
        z_min = max(0, depth // 2 - int(depth * 0.3))
        z_max = min(depth, depth // 2 + int(depth * 0.3))

        return [x_min, y_min, z_min, x_max, y_max, z_max]

    def detect(self, whole_img):
        """
        Phát hiện tim từ ảnh CT

        Args:
            whole_img: Ảnh CT đã được chuẩn hóa

        Returns:
            bbox_list: Danh sách bounding box cho mỗi slice
            bbox_selected: Danh sách chỉ định slice nào chứa tim
            visual_bbox: Danh sách ảnh đã được vẽ bounding box
        """
        if self.model is None:
            if not self.load_model():
                print("Không thể tải mô hình, sử dụng phương pháp đơn giản")
                # Trả về kết quả giả lập khi không có mô hình
                bbox = self._simple_heart_detection(whole_img)

                # Tạo danh sách bounding box giả lập
                depth, height, width = whole_img.shape
                x_min, y_min, z_min, x_max, y_max, z_max = bbox

                # Tạo bbox_list với kích thước bằng số lượng slice
                bbox_list = np.zeros((depth, 4))
                bbox_selected = np.zeros(depth)

                # Đánh dấu các slice chứa tim
                for i in range(z_min, z_max + 1):
                    if 0 <= i < depth:
                        bbox_list[i] = [x_min, y_min, x_max, y_max]
                        bbox_selected[i] = 1

                # Không tạo visual_bbox khi sử dụng phương pháp đơn giản
                return bbox_list, bbox_selected, None

        frame_num = whole_img.shape[0]
        bbox_list = list()
        bbox_selected = list()
        visual_bbox = list()

        try:
            for j in range(frame_num - 1, -1, -1):
                pic = np.tile(np.expand_dims(whole_img[j], axis=2), (1, 1, 3))
                # Ensure input is float32 before sending to device
                torch_pic = torch.from_numpy(pic).to(self.device).float()
                torch_pic = torch_pic.unsqueeze(0).permute(0, 3, 1, 2).contiguous()

                with torch.no_grad():
                    scores, classification, transformed_anchors = self.model(
                        torch_pic)
                    # Move tensors to CPU before converting to numpy
                    scores = scores.cpu().numpy()
                    transformed_anchors = transformed_anchors.cpu().numpy()
                    if scores.size == 0:
                        scores = np.asarray([0])
                        transformed_anchors = np.asarray([[0, 0, 0, 0]])
                    bbox_id = np.argmax(scores)
                    bbox = np.array(transformed_anchors[bbox_id, :])
                    bbox_list.append(bbox)

                    score = scores[bbox_id]
                    if score > 0.3:
                        selected = 1
                    elif np.sum(bbox_selected) <= 0:
                        selected = 0
                    elif bbox_selected[-1] == 1 and self.__calc_iou(
                            bbox_list[-2], bbox_list[-1]) > 0.8 and score > 0.07:
                        selected = 1
                    else:
                        selected = 0
                    bbox_selected.append(selected)

                    visual_bbox.append(
                        self.__visualize(
                            deepcopy(pic), bbox,
                            ': %.3f%%' % (score * 100),
                            selected))

            bbox_list = np.array(bbox_list)
            bbox_selected = self.__continue_smooth(bbox_selected)
            return bbox_list, bbox_selected, visual_bbox

        except Exception as e:
            print(f"Lỗi khi phát hiện tim: {e}")
            # Sử dụng phương pháp đơn giản khi có lỗi
            bbox = self._simple_heart_detection(whole_img)

            # Tạo danh sách bounding box giả lập
            depth, height, width = whole_img.shape
            x_min, y_min, z_min, x_max, y_max, z_max = bbox

            # Tạo bbox_list với kích thước bằng số lượng slice
            bbox_list = np.zeros((depth, 4))
            bbox_selected = np.zeros(depth)

            # Đánh dấu các slice chứa tim
            for i in range(z_min, z_max + 1):
                if 0 <= i < depth:
                    bbox_list[i] = [x_min, y_min, x_max, y_max]
                    bbox_selected[i] = 1

            return bbox_list, bbox_selected, None

    def debug_detection(self, ct_volume):
        """
        Hàm debug để kiểm tra quá trình phát hiện tim

        Args:
            ct_volume: Ảnh CT đầu vào
        """
        if self.model is None:
            print("Không thể debug: Mô hình chưa được tải")
            return

        # Chọn một số slice để debug
        depth = ct_volume.shape[0]
        slice_indices = [depth//4, depth//2, 3*depth//4]

        for i in slice_indices:
            if 0 <= i < ct_volume.shape[0]:
                print(f"Debug detection cho slice {i}")

                # Chuẩn bị input
                img_slice = ct_volume[i]
                img_normalized = self._normalize_for_detection(img_slice)

                # Chuẩn bị input cho mô hình
                pic = np.tile(np.expand_dims(img_normalized, axis=2), (1, 1, 3))
                torch_pic = torch.from_numpy(pic).to(self.device).float()
                torch_pic = torch_pic.unsqueeze(0).permute(0, 3, 1, 2).contiguous()

                # Lưu ảnh để kiểm tra trực quan
                plt.imsave(f"debug_detector_slice_{i}.png", img_normalized, cmap="gray")

                # Thực hiện inference
                try:
                    with torch.no_grad():
                        print(f"Input tensor shape: {torch_pic.shape}, device: {torch_pic.device}")
                        print(f"Model device: {next(self.model.parameters()).device}")

                        scores, classification, transformed_anchors = self.model(torch_pic)
                        print(f"Scores shape: {scores.shape}")
                        print(f"Classification shape: {classification.shape}")
                        print(f"Transformed anchors shape: {transformed_anchors.shape}")

                        # Lấy kết quả tốt nhất
                        if scores.size(0) > 0:
                            best_score_idx = torch.argmax(scores)
                            best_score = scores[best_score_idx].item()
                            best_bbox = transformed_anchors[best_score_idx].cpu().numpy()
                            print(f"Best score: {best_score:.4f}")
                            print(f"Best bbox: {best_bbox}")
                        else:
                            print("Không phát hiện được đối tượng nào")

                except Exception as e:
                    print(f"Lỗi khi thực hiện inference: {e}")

