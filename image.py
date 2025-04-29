import os

import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

# from google.colab import auth
# from google.colab import files
# from googleapiclient.discovery import build
# from googleapiclient.http import MediaIoBaseDownload

from bbox_cut import crop_w_bbox
from utils import norm, CT_resize


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
            from .heart_detector import HeartDetector
            self.heart_detector = HeartDetector()
            self.heart_detector.load_model()
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
        self.detected_ct_img = crop_w_bbox(
            self.org_ct_img, self.bbox, self.bbox_selected)
        if self.detected_ct_img is None:
            return False
        self.detected_dicom_names = [v for f, v in zip(
            self.bbox_selected, self.dicom_names) if f == 1]
        self.detected_npy = sitk.GetArrayFromImage(self.detected_ct_img)
        self.detected_npy = norm(self.detected_npy, -300, 500)
        return True

    def save_visual_bbox(self, output_folder):
        os.makedirs(output_folder, exist_ok=True)
        for i, slice_img in enumerate(self.visual_bbox):
            output_path = os.path.join(
                output_folder, f"{i}_{self.dicom_names[i]}.png")
            plt.imsave(output_path, slice_img)

    def to_network_input(self):
        data = self.detected_npy
        mask = np.clip(
            (data > 0.1375).astype('float') * (data < 0.3375).astype('float')
            + (data > 0.5375).astype('float'), 0, 1)
        mask = gaussian_filter(mask, sigma=3)
        network_input = np.stack([data, data * mask]).astype('float32')
        return network_input
