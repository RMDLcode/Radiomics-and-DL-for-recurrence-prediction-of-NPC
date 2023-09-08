from posixpath import join
from torch.utils.data import DataLoader
import os
import sys
import random
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset as dataset
from .transforms import RandomCrop, RandomFlip_Y, RandomFlip_X, Gamma, Spacial_simple, Compose

class Train_Dataset(dataset):
    def __init__(self, args):

        self.args = args
        self.filename_list = self.load_file_name_list(os.path.join(args.dataset_path, 'train_path_list.txt'))
        self.transforms = Compose([
                RandomFlip_Y(prob=0.5),
                RandomFlip_X(prob=0.5),
                RandomCrop(self.args.crop_size, self.args.inputshape),
                Spacial_simple(angle_x=180, angle_y=0, angle_z=0,
                               patch_size=(self.args.crop_size, self.args.inputshape, self.args.inputshape),
                               p_scale_per_sample=0.2, p_rotation_per_sample=0.2,
                               p_elastic_per_sample=0.2, Do_scale=True, Do_elastic_deform=False),
                Gamma(prob=0.3, gamma_range=(0.7, 1.4)),
            ])

    def __getitem__(self, index):
        ct_array = np.load(self.filename_list[index][0])
        seg_array = np.load(self.filename_list[index][1])
        ct_name = self.filename_list[index][0][-10:]

        ct_array = ct_array.astype(np.float32)
        seg_array = seg_array.astype(np.int32)

        z_patch = self.args.crop_size
        z_start = 0
        z_end = seg_array.shape[0]
        prob = random.uniform(0, 1)
        if prob <= 1.0:
            z_start = np.min(np.where(seg_array >= 1)[0])-24
            if z_start < 0:
                z_start = 0

            z_end = np.max(np.where(seg_array >= 1)[0])+24
            if z_end > seg_array.shape[0]:
                z_end = seg_array.shape[0]

            z_miners = z_end - z_start
            if z_miners <= z_patch:
              z_start_new = z_end - z_patch
              z_end_new = z_start + z_patch
              z_start = z_start_new
              z_end = z_end_new
            if z_start < 0:
                z_start = 0
            if z_end > seg_array.shape[0]:
                z_end = seg_array.shape[0]
            else:
              pass

        ct_array = ct_array[:, z_start:z_end, :, :]
        seg_array = seg_array[z_start:z_end, :, :]

        ct_array = torch.FloatTensor(ct_array)
        seg_array = torch.IntTensor(seg_array).unsqueeze(0)

        if self.transforms:
            ct_array, seg_array = self.transforms(ct_array, seg_array)

        return ct_array, seg_array.squeeze(0), ct_name

    def __len__(self):
        return len(self.filename_list)

    def load_file_name_list(self, file_path):
        file_name_list = []
        with open(file_path, 'r') as file_to_read:
            while True:
                lines = file_to_read.readline().strip()
                if not lines:
                    break
                file_name_list.append(lines.split())
        return file_name_list

if __name__ == "__main__":
    pass