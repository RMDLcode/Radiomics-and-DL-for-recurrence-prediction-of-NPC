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
        self.n_labels = args.n_labels#类别数
        self.cut_size = args.test_cut_size#裁剪Z方向长度
        self.cut_stride = args.test_cut_stride#滑窗步长

        self.data_np = np.load(data_path).astype(np.float32)#图
        self.label_np = np.load(label_path).astype(np.int32)#标签
        # self.radiomics = np.load(radiomics_path).astype(np.float32)#影像组学特征
        
        self.label_ori_shape = self.label_np.shape#标签shape
        print('滑窗前的shape',self.data_np.shape)

        #可以手动减少需要滑的Z方向的长度，依然是中心滑窗的
        self.z_start =0
        self.z_finish = self.label_ori_shape[0]
        self.data_np = self.data_np[:,self.z_start:self.z_finish,:,:]
        self.label_np = self.label_np[self.z_start:self.z_finish,:,:]
        
        inputshape=args.inputshape#XY大小
        #y_start = (self.data_np.shape[2] - inputshape)//2
        #x_start = (self.data_np.shape[3] - inputshape)//2
        y_start = torch.div((self.data_np.shape[2] - inputshape), 2, rounding_mode='floor')#Y方向最小点
        x_start = torch.div((self.data_np.shape[3] - inputshape), 2, rounding_mode='floor')#X方向最小点

        #对XY方向做中心裁剪
        self.data_np = self.data_np[:,:,y_start:y_start+inputshape,x_start:x_start+inputshape]
        self.label_np = self.label_np[:,y_start:y_start+inputshape,x_start:x_start+inputshape]
        #print(self.z_start, self.z_finish, self.label_ori_shape)

        
        self.ori_shape = self.data_np.shape#裁剪后的shape
        print('裁剪后的shape', self.data_np.shape)

        self.data_np = self.padding_img(self.data_np, self.cut_size,self.cut_stride)#补0，用于Z方向中心滑窗
        self.padding_shape = self.data_np.shape#Z方向补0后的shape
        print('#Z方向补0后的shape', self.padding_shape)
        self.data_np = self.extract_ordered_overlap(self.data_np, self.cut_size, self.cut_stride)#一个用于存放所有滑窗图的矩阵

        if self.n_labels==2:#一般不会出现2分类的前景不为1这个问题.
            self.label_np[self.label_np > 0] = 1
        self.label = torch.from_numpy(np.expand_dims(self.label_np,axis=0)).long()#label加一维转torch
        
        # 预测结果保存
        self.result = None

        
    def __getitem__(self, index):
        data = torch.from_numpy(self.data_np[index])
        data = torch.FloatTensor(data)
        #图像，原图shape，Z方向起点，Z方向终点到原图Z方向长度的距离。
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

    def recompone_result(self):#直接滑窗图叠加除以2得到真图

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
        assert (len(img.shape) == 4)  # 要求是4D array
        img_d, img_s, img_h, img_w = img.shape#深度，zyx
        leftover_s = (img_s - size) % stride#中心滑窗需要补整，这个算余数

        if (leftover_s != 0):
            s = img_s + (stride - leftover_s)#在后面补Z方向
        else:
            s = img_s

        tmp_full_imgs = np.zeros((img_d, s, img_h, img_w),dtype=np.float32)#补全后的大的全0图
        tmp_full_imgs[:,:img_s,:,:] = img[:,:,:,:]#把原图的数据赋值给补全后的大的全0图
        #print("Padded images shape: " + str(tmp_full_imgs.shape))
        return tmp_full_imgs

    def extract_ordered_overlap(self, img, size, stride):
        img_d, img_s, img_h, img_w = img.shape#深度，zyx
        #print(img.shape)
        assert (img_s - size) % stride == 0#按理来说之前补0了，可以整除的
        N_patches_img = (img_s - size) // stride + 1 #总共要滑多少个patch

        #print("Patches number of the image:{}".format(N_patches_img))
        patches = np.empty((N_patches_img, img_d, size, img_h, img_w), dtype=np.float32)#一个用于存放所有滑窗图的矩阵

        for s in range(N_patches_img):  # loop over the full images
            patch = img[:,s * stride : s * stride + size,:,:]#Z方向滑窗
            patches[s] = patch#存进上面构建的大矩阵

        return patches #一个用于存放所有滑窗图的矩阵
def Test_Datasets(dataset_path, args):
    data_list = sorted(glob(os.path.join(dataset_path, 'mr/*')))#图路径glob
    label_list = sorted(glob(os.path.join(dataset_path, 'label/*'))) #标签路径glob
    # radiomics_list = sorted(glob(os.path.join(dataset_path, 'radiomics_norm/*')))#影像组学特征路径glob
    print("The number of test samples is: ", len(data_list))#样本数量
    for datapath, labelpath in zip(data_list, label_list):
        print("\nStart Evaluate: ", datapath)
        yield Img_DataSet(datapath, labelpath, args=args), datapath.split('-')[-1]
    # for datapath, labelpath, radiomicspath in zip(data_list, label_list, radiomics_list):
    #     print("\nStart Evaluate: ", datapath)
    #     yield Img_DataSet(datapath, labelpath, radiomicspath, args=args), datapath.split('-')[-1]
