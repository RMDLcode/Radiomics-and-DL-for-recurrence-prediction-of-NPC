from posixpath import join
from torch.utils.data import DataLoader
import os
import sys
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset as dataset
# from .transforms import RandomCrop, Compose


class Val_Dataset(dataset):
    def __init__(self, args):

        self.args = args
        self.filename_list = self.load_file_name_list(os.path.join(args.dataset_path, 'val_path_list.txt'))
        self.n_labels = args.n_labels
        self.cut_size = args.test_cut_size
        self.cut_stride = args.test_cut_stride

    def __getitem__(self, index):

        ct_array = np.load(self.filename_list[index][0])
        seg_array = np.load(self.filename_list[index][1])

        ct_name = self.filename_list[index][0][-10:]
        ct_array = ct_array.astype(np.float32)
        seg_array = seg_array.astype(np.int32)
        
        inputshape = self.args.inputshape
        y_start = torch.div((ct_array.shape[2] - inputshape), 2, rounding_mode='floor')
        x_start = torch.div((ct_array.shape[3] - inputshape), 2, rounding_mode='floor')

        #Center cut in the XY direction
        ct_array = ct_array[:,:,y_start:y_start+inputshape,x_start:x_start+inputshape]
        seg_array = seg_array[:,y_start:y_start+inputshape,x_start:x_start+inputshape]

        ct_array = torch.FloatTensor(ct_array)
        seg_array = torch.IntTensor(seg_array)

        self.ori_shape = ct_array.shape

        data_np = self.padding_img(ct_array, self.cut_size,self.cut_stride)
        self.padding_shape = data_np.shape

        data_np = self.extract_ordered_overlap(data_np, self.cut_size, self.cut_stride)

        #if self.transforms:
        #    ct_array, seg_array = self.transforms(ct_array, seg_array)

        self.result = None
        return data_np, seg_array, torch.IntTensor([self.padding_shape[1], self.cut_stride, self.ori_shape[1],self.ori_shape[2], self.ori_shape[3], self.n_labels]),ct_name
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
    

    def padding_img(self, img, size, stride):
        assert (len(img.shape) == 4)
        img_d, img_s, img_h, img_w = img.shape
        leftover_s = (img_s - size) % stride

        if (leftover_s != 0):
            s = img_s + (stride - leftover_s)
        else:
            s = img_s

        tmp_full_imgs = np.zeros((img_d, s, img_h, img_w),dtype=np.float32)
        tmp_full_imgs[:,:img_s,:,:] = img[:,:,:,:]
        #print("Padded images shape: " + str(tmp_full_imgs.shape)) 
        return tmp_full_imgs
    
    # Divide all the full_imgs in pacthes
    def extract_ordered_overlap(self, img, size, stride):
        img_d, img_s, img_h, img_w = img.shape
        #print(img.shape)
        assert (img_s - size) % stride == 0
        N_patches_img = (img_s - size) // stride + 1

        #print("Patches number of the image:{}".format(N_patches_img))
        patches = np.empty((N_patches_img, img_d, size, img_h, img_w), dtype=np.float32)

        for s in range(N_patches_img):
            patch = img[:,s * stride : s * stride + size,:,:]
            patches[s] = patch

        return patches
if __name__ == "__main__":
    pass