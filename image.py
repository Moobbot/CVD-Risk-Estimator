import os

import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from skimage.transform import resize as imresize
import pydicom  # Thêm import pydicom

from bbox_cut import crop_w_bbox, parse_bbox
from utils import norm, CT_resize
from config import CLEANUP_CONFIG  # Import cấu hình


class Image:
    CT_AXIAL_SIZE = 512

    def __init__(self, dicom_directory, heart_detector=None):
        self.__load_dicom_series(dicom_directory)
        self.org_npy = None
        self.bbox = None
        self.bbox_selected = None
        self.visual_bbox = None
        self.detected_ct_img = None
        self.detected_npy = None
        self.heart_detector = heart_detector
        self.min_point = None
        self.max_point = None

    def __load_dicom_series(self, dicom_directory):
        """Load DICOM series from a specified directory"""
        reader = sitk.ImageSeriesReader()
        dicom_paths = reader.GetGDCMSeriesFileNames(dicom_directory)
        reader.SetFileNames(dicom_paths)
        self.dicom_names = [os.path.splitext(os.path.basename(path))[
            0] for path in dicom_paths[::-1]]
        self.org_ct_img = reader.Execute()

    def detect_heart(self):
        if self.heart_detector is None:
            # Sửa lỗi import tương đối
            from heart_detector import HeartDetector
            self.heart_detector = HeartDetector()
            self.heart_detector.load_model()

        try:
            # Resize org ct
            old_size = np.asarray(self.org_ct_img.GetSize()).astype('float')
            if min(old_size[0], old_size[1]) < 480 or max(old_size[0], old_size[1]) > 550:
                print('Resizing the image...')
                new_size = np.asarray([
                    Image.CT_AXIAL_SIZE, Image.CT_AXIAL_SIZE, old_size[-1]]
                ).astype('float')
                old_space = np.asarray(
                    self.org_ct_img.GetSpacing()).astype('float')
                new_space = old_space * old_size / new_size
                self.org_ct_img = CT_resize(
                    self.org_ct_img,
                    new_size=new_size.astype('int').tolist(),
                    new_space=new_space.tolist())
            self.org_npy = sitk.GetArrayFromImage(self.org_ct_img)
            self.org_npy = norm(self.org_npy, -500, 500)

            # detect heart
            self.bbox, self.bbox_selected, self.visual_bbox = self.heart_detector.detect(
                self.org_npy)

            if self.bbox is None or self.bbox_selected is None:
                print("Phát hiện tim không thành công, trả về False")
                return False

            # Save min and max points for mapping back to original image
            org_space = np.array(self.org_ct_img.GetSpacing())
            try:
                self.min_point, self.max_point = parse_bbox(
                    self.bbox, self.bbox_selected, self.org_ct_img.GetSize(), org_space)
            except Exception as e:
                print(f"Lỗi khi phân tích bbox: {e}")
                # Không return False ở đây, tiếp tục thử crop_w_bbox

            # Crop heart region
            self.detected_ct_img = crop_w_bbox(
                self.org_ct_img, self.bbox, self.bbox_selected)

            if self.detected_ct_img is None:
                print("Không thể cắt vùng tim, trả về False")
                return False

            # Lấy tên các file DICOM chứa tim
            self.detected_dicom_names = [v for f, v in zip(
                self.bbox_selected, self.dicom_names) if f == 1]

            # Chuyển đổi sang numpy array và chuẩn hóa
            self.detected_npy = sitk.GetArrayFromImage(self.detected_ct_img)
            self.detected_npy = norm(self.detected_npy, -300, 500)
            return True

        except Exception as e:
            print(f"Lỗi trong quá trình phát hiện tim: {e}")
            return False

    def save_visual_bbox(self, output_folder):
        os.makedirs(output_folder, exist_ok=True)
        for i, slice_img in enumerate(self.visual_bbox):
            output_path = os.path.join(
                output_folder, f"{i}_{self.dicom_names[i]}.png")
            plt.imsave(output_path, slice_img)

    def save_grad_cam_on_original(self, cam_data, output_folder):
        """
        Save grad-cam visualization overlaid on the original DICOM images

        Args:
            cam_data: The grad-cam heatmap array (128x128x128)
            output_folder: Folder to save the visualizations
        """
        if self.min_point is None or self.max_point is None:
            return False

        os.makedirs(output_folder, exist_ok=True)

        # Get the indices of slices that have the heart detected
        heart_indices = [i for i, val in enumerate(self.bbox_selected) if val == 1]

        # Color map for heatmap visualization
        color = cv2.COLORMAP_JET

        reversed_img = np.flip(self.org_npy, axis=0)

        for idx, orig_idx in enumerate(heart_indices):
            if idx >= len(cam_data):
                break

            # Get original image
            orig_img = reversed_img[orig_idx]

            # Convert to BGR for visualization
            orig_img_vis = np.tile(np.expand_dims(orig_img, axis=2), (1, 1, 3))
            orig_img_vis = (orig_img_vis * 255).astype('uint8')

            # Get the bounding box for this slice
            x1, y1, x2, y2 = [int(val) for val in self.bbox[orig_idx]]

            # Extract and resize the corresponding grad-cam slice to match the heart region size
            cam_slice = cam_data[idx]
            resized_cam = imresize(cam_slice, (y2-y1, x2-x1))

            # Apply colormap to create heatmap
            heatmap = cv2.applyColorMap(np.uint8(255 * resized_cam), color)

            # Create a blank heatmap of the original image size
            full_heatmap = np.zeros_like(orig_img_vis)

            # Place the resized heatmap on the original image location
            full_heatmap[y1:y2, x1:x2] = heatmap

            # Blend the heatmap with the original image
            blended = cv2.addWeighted(orig_img_vis, 0.7, full_heatmap, 0.3, 0)

            # Convert to RGB for saving with matplotlib
            blended = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)

            # Save the visualization
            output_path = os.path.join(
                output_folder, f"{orig_idx}_{self.dicom_names[orig_idx]}.png")
            plt.imsave(output_path, blended)

        return True

    def to_network_input(self):
        """
        Chuyển đổi dữ liệu đã phát hiện thành đầu vào cho mạng neural

        Returns:
            numpy.ndarray: Mảng đầu vào cho mạng neural
        """
        try:
            if self.detected_npy is None:
                raise ValueError("Chưa phát hiện vùng tim hoặc phát hiện không thành công")

            data = self.detected_npy

            # Tạo mặt nạ dựa trên ngưỡng giá trị
            mask = np.clip(
                (data > 0.1375).astype('float') * (data < 0.3375).astype('float')
                + (data > 0.5375).astype('float'), 0, 1)

            # Làm mịn mặt nạ bằng bộ lọc Gaussian
            mask = gaussian_filter(mask, sigma=3)

            # Tạo đầu vào cho mạng neural
            network_input = np.stack([data, data * mask]).astype('float32')
            return network_input

        except Exception as e:
            print(f"Lỗi khi tạo đầu vào cho mạng neural: {e}")
            raise

