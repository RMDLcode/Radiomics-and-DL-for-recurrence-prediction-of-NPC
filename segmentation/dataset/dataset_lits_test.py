from torch._C import dtype
from utils.common import *
from scipy import ndimage
import numpy as np
from torchvision import transforms as T
import torch, os
from torch.utils.data import Dataset, DataLoader
from glob import glob
import math
import SimpleITK as sitk

class Img_DataSet(Dataset):
    def __init__(self, data_path, label_path, args):
        self.n_labels = args.n_labels
        self.cut_size = args.test_cut_size
        self.cut_stride = args.test_cut_stride

        self.data_np = np.load(data_path).astype(np.float32)
        self.label_np = np.load(label_path).astype(np.int32)
        # self.radiomics = np.load(radiomics_path).astype(np.float32)
        
        self.label_ori_shape = self.label_np.shape

        self.z_start =0
        self.z_finish = self.label_ori_shape[0]
        self.data_np = self.data_np[:,self.z_start:self.z_finish,:,:]
        self.label_np = self.label_np[self.z_start:self.z_finish,:,:]
        
        inputshape=args.inputshape
        #y_start = (self.data_np.shape[2] - inputshape)//2
        #x_start = (self.data_np.shape[3] - inputshape)//2
        y_start = torch.div((self.data_np.shape[2] - inputshape), 2, rounding_mode='floor')
        x_start = torch.div((self.data_np.shape[3] - inputshape), 2, rounding_mode='floor')

        self.data_np = self.data_np[:,:,y_start:y_start+inputshape,x_start:x_start+inputshape]
        self.label_np = self.label_np[:,y_start:y_start+inputshape,x_start:x_start+inputshape]
        #print(self.z_start, self.z_finish, self.label_ori_shape)

        
        self.ori_shape = self.data_np.shape

        self.data_np = self.padding_img(self.data_np, self.cut_size,self.cut_stride)
        self.padding_shape = self.data_np.shape

        self.data_np = self.extract_ordered_overlap(self.data_np, self.cut_size, self.cut_stride)

        if self.n_labels==2:
            self.label_np[self.label_np > 0] = 1
        self.label = torch.from_numpy(np.expand_dims(self.label_np,axis=0)).long()

        self.result = None

        
    def __getitem__(self, index):
        data = torch.from_numpy(self.data_np[index])
        data = torch.FloatTensor(data)

        return data, self.label_ori_shape, self.z_start, self.label_ori_shape[0] - self.z_finish
        # return data, self.label_ori_shape, self.radiomics, self.z_start, self.label_ori_shape[0] - self.z_finish

    def __len__(self):
        return len(self.data_np)

    def update_result(self, tensor):
        # tensor = tensor.detach().cpu() # shape: [N,class,s,h,w]
        # tensor_np = np.squeeze(tensor_np,axis=0)
        if self.result is not None:
            self.result = torch.cat((self.result, tensor), dim=0)
        else:
            self.result = tensor

    def recompone_result(self):

        patch_s = self.result.shape[2]
        N_patches_img = (self.padding_shape[1] - patch_s) // self.cut_stride + 1
        assert (self.result.shape[0] == N_patches_img)

        full_prob = torch.zeros((self.n_labels, self.padding_shape[1], self.ori_shape[2],self.ori_shape[3]))
        full_sum = torch.zeros((self.n_labels, self.padding_shape[1], self.ori_shape[2], self.ori_shape[3]))

        for s in range(N_patches_img):
            full_prob[:, s * self.cut_stride:s * self.cut_stride + patch_s] += self.result[s]
            full_sum[:, s * self.cut_stride:s * self.cut_stride + patch_s] += 1

        assert (torch.min(full_sum) >= 1.0)  # at least one
        final_avg = full_prob / full_sum
        assert (torch.max(final_avg) <= 1.0)  # max value for a pixel is 1.0
        assert (torch.min(final_avg) >= 0.0)  # min value for a pixel is 0.0
        img = final_avg[:, :self.ori_shape[1], :self.ori_shape[2], :self.ori_shape[3]]

        return img.unsqueeze(0)

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

    def extract_ordered_overlap(self, img, size, stride):
        img_d, img_s, img_h, img_w = img.shape
        #print(img.shape)
        assert (img_s - size) % stride == 0
        N_patches_img = (img_s - size) // stride + 1 

        #print("Patches number of the image:{}".format(N_patches_img))
        patches = np.empty((N_patches_img, img_d, size, img_h, img_w), dtype=np.float32)

        for s in range(N_patches_img):  # loop over the full images
            patch = img[:,s * stride : s * stride + size,:,:]
            patches[s] = patch

        return patches 
def Test_Datasets(dataset_path, args):
    data_list = sorted(glob(os.path.join(dataset_path, 'mr/*')))
    label_list = sorted(glob(os.path.join(dataset_path, 'label/*'))) 
    print("The number of test samples is: ", len(data_list))
    for datapath, labelpath in zip(data_list, label_list):
        print("\nStart Evaluate: ", datapath)
        yield Img_DataSet(datapath, labelpath, args=args), datapath.split('-')[-1]
