import io
import os.path as osp

import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid
from scipy.ndimage import gaussian_filter

from .bbox_cut import crop_w_bbox
from .heart_detect import detector
from .utils import norm, CT_resize


class Image:
    CT_AXIAL_SIZE = 512

    def __init__(self):
        self.org_ct_img = None
        self.org_npy = None
        self.bbox = None
        self.bbox_selected = None
        self.visual_bbox = None
        self.detected_ct_img = None
        self.detected_npy = None
        
    def load_nifti(self, file_name):
        self.org_ct_img = sitk.ReadImage(file_name)

    def detect_heart(self):
        # Resize org ct
        old_size = np.asarray(self.org_ct_img.GetSize()).astype('float')
        if min(old_size[0], old_size[1]) < 480 or max(old_size[0], old_size[1]) > 550:
            print('Resizing the image...')
            new_size = np.asarray([
                Image.CT_AXIAL_SIZE, Image.CT_AXIAL_SIZE, old_size[-1]]
            ).astype('float')
            old_space = np.asarray(self.org_ct_img.GetSpacing()).astype('float')
            new_space = old_space * old_size / new_size
            self.org_ct_img = CT_resize(
                self.org_ct_img,
                new_size=new_size.astype('int').tolist(),
                new_space=new_space.tolist())
        self.org_npy = sitk.GetArrayFromImage(self.org_ct_img)
        self.org_npy = norm(self.org_npy, -500, 500)
        # detect heart
        self.bbox, self.bbox_selected, self.visual_bbox = detector(self.org_npy)
        self.detected_ct_img = crop_w_bbox(
            self.org_ct_img, self.bbox, self.bbox_selected)
        if self.detected_ct_img is None:
            print('Fail to detect heart in the image. '
                  'Please manually crop the heart region.')
            return
        self.detected_npy = sitk.GetArrayFromImage(self.detected_ct_img)
        self.detected_npy = norm(self.detected_npy, -300, 500)

    def detect_visual(self):
        total_img_num = len(self.visual_bbox)
        fig = plt.figure(figsize=(15, 15))
        grid = ImageGrid(fig, 111, nrows_ncols=(8, 8), axes_pad=0.05)
        for i in range(64):
            grid[i].imshow(self.visual_bbox[i * int(total_img_num / 64)])
        plt.show()

    def to_network_input(self):
        data = self.detected_npy
        mask = np.clip(
            (data > 0.1375).astype('float') * (data < 0.3375).astype('float')
            + (data > 0.5375).astype('float'), 0, 1)
        mask = gaussian_filter(mask, sigma=3)
        network_input = np.stack([data, data * mask]).astype('float32')
        return network_input
