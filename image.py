import os
import re

import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import cv2
import imageio
from scipy.ndimage import gaussian_filter
from skimage.transform import resize as imresize
import pydicom  # Thêm import pydicom

from bbox_cut import crop_w_bbox, parse_bbox
from utils import norm, CT_resize
from config import CLEANUP_CONFIG, FOLDERS  # Import cấu hình


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

    def save_grad_cam_on_original(self, cam_data, output_folder, create_gif=False, session_id=None):
        """
        Save grad-cam visualization overlaid on the original DICOM images
        and optionally create a GIF directly from the images in memory

        Args:
            cam_data: The grad-cam heatmap array (128x128x128)
            output_folder: Folder to save the visualizations
            create_gif: Whether to create a GIF from the images (default: False)
            session_id: Session ID for the GIF filename (required if create_gif=True)

        Returns:
            tuple: (bool, str) - (Success status, GIF path if created or None)
        """
        if self.min_point is None or self.max_point is None:
            return False, None

        if create_gif and session_id is None:
            raise ValueError("session_id is required when create_gif=True")

        os.makedirs(output_folder, exist_ok=True)

        # Get the indices of slices that have the heart detected
        heart_indices = [i for i, val in enumerate(self.bbox_selected) if val == 1]

        # Color map for heatmap visualization
        color = cv2.COLORMAP_JET

        reversed_img = np.flip(self.org_npy, axis=0)

        # List to store blended images for GIF creation
        blended_images = []
        image_indices = []

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

            # Store the blended image for GIF creation if requested
            if create_gif:
                blended_images.append(blended)
                image_indices.append(orig_idx)

        # Create GIF directly from memory if requested
        gif_path = None
        if create_gif and blended_images:
            gif_path = self.create_gif_from_images(blended_images, image_indices, session_id, output_folder)

        return True, gif_path

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

    def create_gif_from_images(self, images, indices, session_id, output_folder=None):
        """
        Tạo file GIF trực tiếp từ danh sách ảnh trong bộ nhớ

        Args:
            images: Danh sách các ảnh đã xử lý trong bộ nhớ
            indices: Danh sách các chỉ số tương ứng với mỗi ảnh
            session_id: ID của phiên làm việc
            output_folder: Thư mục để lưu file GIF, nếu None sẽ lưu vào thư mục session_id

        Returns:
            str: Đường dẫn đến file GIF đã tạo, hoặc None nếu không thành công
        """
        try:
            # Nếu không có output_folder, sử dụng thư mục session_id
            if output_folder is None:
                output_folder = os.path.join(FOLDERS["RESULTS"], session_id, "cvd")
                os.makedirs(output_folder, exist_ok=True)

            # Đường dẫn file GIF trong thư mục session_id
            gif_path = os.path.join(output_folder, "results.gif")

            if not images:
                return None

            # Sắp xếp ảnh theo thứ tự chỉ số
            sorted_images = [img for _, img in sorted(zip(indices, images))]

            # Lưu file GIF trực tiếp từ danh sách ảnh trong bộ nhớ
            imageio.mimsave(gif_path, sorted_images, duration=0.2, loop=0)

            return gif_path

        except Exception as e:
            print(f"Lỗi khi tạo GIF từ ảnh trong bộ nhớ: {e}")
            return None

    def create_gif_from_overlay_images(self, output_dir, session_id):
        """
        Tạo file GIF từ các ảnh overlay đã lưu trên đĩa

        Args:
            output_dir: Thư mục chứa các ảnh overlay
            session_id: ID của phiên làm việc

        Returns:
            str: Đường dẫn đến file GIF đã tạo, hoặc None nếu không thành công
        """
        try:
            # Lưu file GIF trong cùng thư mục với các ảnh overlay
            gif_path = os.path.join(output_dir, "results.gif")

            # Lấy danh sách các file ảnh PNG
            image_files = []
            for file in os.listdir(output_dir):
                if file.endswith(".png"):
                    image_files.append(os.path.join(output_dir, file))

            if not image_files:
                return None

            # Sắp xếp ảnh theo thứ tự số slice
            def extract_index(filename):
                match = re.search(r'^(\d+)_', os.path.basename(filename))
                if match:
                    return int(match.group(1))
                return 0

            image_files.sort(key=extract_index)

            # Đọc các ảnh và tạo GIF
            images = []
            for image_file in image_files:
                images.append(imageio.imread(image_file))

            # Lưu file GIF
            imageio.mimsave(gif_path, images, duration=0.2, loop=0)

            return gif_path

        except Exception as e:
            print(f"Lỗi khi tạo GIF từ file: {e}")
            return None

