# coding:utf-8
from torch.utils.data import Dataset
import torch
import numpy as np
import random
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.functional import normalize
from batchgenerators.augmentations.utils import create_zero_centered_coordinate_mesh, elastic_deform_coordinates, \
    interpolate_img, \
    rotate_coords_2d, rotate_coords_3d, scale_coords, resize_segmentation, resize_multichannel_image, \
    elastic_deform_coordinates_2
from batchgenerators.augmentations.crop_and_pad_augmentations import random_crop as random_crop_aug
from batchgenerators.augmentations.crop_and_pad_augmentations import center_crop as center_crop_aug
#----------------------data augment-------------------------------------------

class RandomCrop:
    def __init__(self, slices, inputshape):
        self.slices = slices
        self.inputshape = inputshape
    def _get_range(self, slices, crop_slices):
        if slices <= crop_slices:
            start = 0
        else:
            start = random.randint(0, slices - crop_slices)
        end = start + crop_slices
        if end > slices:
            end = slices
        return start, end

    def __call__(self, img, mask, inputshape=256):
        inputshape=self.inputshape
        ss, es = self._get_range(mask.size(1), self.slices)
        y_start = random.randint(0, img.size(2) - inputshape)
        x_start = random.randint(0, img.size(3) - inputshape)
        # print(self.shape, img.shape, mask.shape)

        #Create an empty matrix and complete the random cropping
        tmp_img = torch.zeros((img.size(0), self.slices, inputshape, inputshape))
        tmp_mask = torch.zeros((mask.size(0), self.slices, inputshape, inputshape))
        tmp_img[:,:es-ss,:,:] = img[:,ss:es,y_start:y_start+inputshape,x_start:x_start+inputshape]
        tmp_mask[:,:es-ss,:,:] = mask[:,ss:es,y_start:y_start+inputshape,x_start:x_start+inputshape]
        return tmp_img, tmp_mask

def rotate_coords_3d(coords, angle_x, angle_y, angle_z):
    rot_matrix = np.identity(len(coords))
    rot_matrix = create_matrix_rotation_x_3d(angle_x, rot_matrix)
    rot_matrix = create_matrix_rotation_y_3d(angle_y, rot_matrix)
    rot_matrix = create_matrix_rotation_z_3d(angle_z, rot_matrix)
    coords = np.dot(coords.reshape(len(coords), -1).transpose(), rot_matrix).transpose().reshape(coords.shape)
    return coords

def create_zero_centered_coordinate_mesh(shape):
    tmp = tuple([np.arange(i) for i in shape])
    coords = np.array(np.meshgrid(*tmp, indexing='ij')).astype(float)
    for d in range(len(shape)):
        coords[d] -= ((np.array(shape).astype(float) - 1) / 2.)[d]
    return coords

class Spacial_simple:
    def __init__(self, angle_x=15, angle_y=15, angle_z=15, patch_size=(32, 256, 256),
                 p_scale_per_sample=0.2, p_rotation_per_sample=0.2, p_elastic_per_sample=0.2,
                 Do_scale=True, Do_elastic_deform=False, alpha=(0., 200.), sigma=(9., 13.)):
        self.angle_x = angle_x
        self.angle_y = angle_y
        self.angle_z = angle_z
        self.Do_scale = Do_scale
        self.p_scale_per_sample = p_scale_per_sample
        self.p_rotation_per_sample = p_rotation_per_sample
        self.p_elastic_per_sample = p_elastic_per_sample
        self.do_elastic_deform = Do_elastic_deform
        self.alpha = alpha
        self.sigma = sigma
        self.patch_size = patch_size

    def __call__(self, data, seg):
        data = data.unsqueeze(0)
        seg = seg.unsqueeze(0)
        data = data.numpy()
        seg = seg.numpy()

        do_scale = self.Do_scale
        do_elastic_deform = self.do_elastic_deform
        p_scale_per_sample = self.p_scale_per_sample
        p_rotation_per_sample = self.p_rotation_per_sample
        p_el_per_sample = self.p_elastic_per_sample
        alpha = self.alpha
        sigma = self.sigma

        independent_scale_for_each_axis = False
        p_independent_scale_per_axis = 1
        scale = (0.7, 1.4)

        patch_center_dist_from_border = None
        random_crop = False
        patch_size = self.patch_size

        dim = len(patch_size)  

        patch_center_dist_from_border = dim * [patch_center_dist_from_border]
        data_result = np.zeros((data.shape[0], data.shape[1], patch_size[0], patch_size[1], patch_size[2]),
                               dtype=np.float32)
        seg_result = np.zeros((seg.shape[0], seg.shape[1], patch_size[0], patch_size[1], patch_size[2]),
                              dtype=np.float32)
        border_cval_data = 0
        border_cval_seg = 0

        for sample_id in range(data.shape[0]):

            a_x = np.random.uniform(-self.angle_x / 360 * 2. * np.pi, self.angle_x / 360 * 2. * np.pi)
            a_y = np.random.uniform(-self.angle_y / 360 * 2. * np.pi, self.angle_y / 360 * 2. * np.pi)
            a_z = np.random.uniform(-self.angle_z / 360 * 2. * np.pi, self.angle_z / 360 * 2. * np.pi)
            coords = create_zero_centered_coordinate_mesh(patch_size)

            if do_elastic_deform and np.random.uniform() < p_el_per_sample:
                a = np.random.uniform(alpha[0], alpha[1])
                s = np.random.uniform(sigma[0], sigma[1])
                coords = elastic_deform_coordinates(coords, a, s)

            if np.random.uniform() < p_rotation_per_sample:
                coords = rotate_coords_3d(coords, angle_x=a_x, angle_y=a_y, angle_z=a_z)

            if do_scale and np.random.uniform() < p_scale_per_sample:
                if independent_scale_for_each_axis and np.random.uniform() < p_independent_scale_per_axis:
                    sc = []
                    for _ in range(dim):
                        if np.random.random() < 0.5 and scale[0] < 1:
                            sc.append(np.random.uniform(scale[0], 1))
                        else:
                            sc.append(np.random.uniform(max(scale[0], 1), scale[1]))
                else:  #
                    if np.random.random() < 0.5 and scale[0] < 1:
                        sc = np.random.uniform(scale[0], 1)
                    else:
                        sc = np.random.uniform(max(scale[0], 1), scale[1])

                coords = scale_coords(coords, sc)

            coords_mean = coords.mean(axis=tuple(range(1, len(coords.shape))), keepdims=True)
            coords -= coords_mean

            for d in range(dim):
                ctr = data.shape[d + 2] / 2. - 0.5
                coords[d] += ctr
            for channel_id in range(data.shape[1]):
                data_result[sample_id, channel_id] = interpolate_img(data[sample_id, channel_id], coords, order=3,
                                                                     mode='nearest', cval=border_cval_data)
            if seg is not None:
                for channel_id in range(seg.shape[1]):
                    seg_result[sample_id, channel_id] = interpolate_img(seg[sample_id, channel_id], coords, order=0,
                                                                        mode='constant', cval=border_cval_seg,
                                                                        is_seg=True)

        data_result = torch.FloatTensor(data_result)
        seg_result = torch.IntTensor(seg_result)
        return data_result.squeeze(0), seg_result.squeeze(0)

class RandomFlip_Y:
    def __init__(self, prob=0.5):
        self.prob = prob

    def _flip(self, img, prob):
        if prob[0] <= self.prob:
            img = img.flip(2)
        return img

    def __call__(self, img, mask):
        prob = (random.uniform(0, 1), random.uniform(0, 1))
        return self._flip(img, prob), self._flip(mask, prob)

class RandomFlip_X:
    def __init__(self, prob=0.5):
        self.prob = prob

    def _flip(self, img, prob):
        if prob[1] <= self.prob:
            img = img.flip(3)
        return img

    def __call__(self, img, mask):
        prob = (random.uniform(0, 1), random.uniform(0, 1))
        return self._flip(img, prob), self._flip(mask, prob)

def create_matrix_rotation_x_3d(angle, matrix=None):
    rotation_x = np.array([[1, 0, 0],
                           [0, np.cos(angle), -np.sin(angle)],
                           [0, np.sin(angle), np.cos(angle)]])
    if matrix is None:
        return rotation_x

    return np.dot(matrix, rotation_x)


def create_matrix_rotation_y_3d(angle, matrix=None):
    rotation_y = np.array([[np.cos(angle), 0, np.sin(angle)],
                           [0, 1, 0],
                           [-np.sin(angle), 0, np.cos(angle)]])
    if matrix is None:
        return rotation_y

    return np.dot(matrix, rotation_y)


def create_matrix_rotation_z_3d(angle, matrix=None):
    rotation_z = np.array([[np.cos(angle), -np.sin(angle), 0],
                           [np.sin(angle), np.cos(angle), 0],
                           [0, 0, 1]])
    if matrix is None:
        return rotation_z

    return np.dot(matrix, rotation_z)


def augment_gamma(data_sample_tmp, gamma_range=(0.7, 1.5), invert_image=False, epsilon=1e-7, per_channel=False,
                  retain_stats = False):
    data_sample = data_sample_tmp.copy()
    if invert_image:
        data_sample = - data_sample

    if not per_channel:
        retain_stats_here = retain_stats() if callable(retain_stats) else retain_stats
        if retain_stats_here:
            mn = data_sample.mean()
            sd = data_sample.std()
        if np.random.random() < 0.5 and gamma_range[0] < 1:
            gamma = np.random.uniform(gamma_range[0], 1)
        else:
            gamma = np.random.uniform(max(gamma_range[0], 1), gamma_range[1])
        minm = data_sample.min()
        rnge = data_sample.max() - minm
        data_sample = np.power(((data_sample - minm) / float(rnge + epsilon)), gamma) * rnge + minm
        if retain_stats_here:
            data_sample = data_sample - data_sample.mean()
            data_sample = data_sample / (data_sample.std() + 1e-8) * sd
            data_sample = data_sample + mn
    else:
        for c in range(data_sample.shape[0]):
            retain_stats_here = retain_stats() if callable(retain_stats) else retain_stats
            if retain_stats_here:
                mn = data_sample[c].mean()
                sd = data_sample[c].std()
            if np.random.random() < 0.5 and gamma_range[0] < 1:
                gamma = np.random.uniform(gamma_range[0], 1)
            else:
                gamma = np.random.uniform(max(gamma_range[0], 1), gamma_range[1])
            minm = data_sample[c].min()
            rnge = data_sample[c].max() - minm
            data_sample[c] = np.power(((data_sample[c] - minm) / float(rnge + epsilon)), gamma) * float(rnge + epsilon) + minm
            if retain_stats_here:
                data_sample[c] = data_sample[c] - data_sample[c].mean()
                data_sample[c] = data_sample[c] / (data_sample[c].std() + 1e-8) * sd
                data_sample[c] = data_sample[c] + mn
    if invert_image:
        data_sample = - data_sample
    return data_sample

def Range01(image):
    max_tmp=np.max(image)
    min_tmp=np.min(image)
    image_new= (image-min_tmp)/(max_tmp-min_tmp)
    return image_new

class Gamma:
    def __init__(self, prob=0.3, gamma_range=(0.7, 1.5)):
        self.prob = prob
        self.gamma_range = gamma_range

    def _gamma(self, img):
        image_new = img
        for i in range(img.shape[0]):
            image_new[i] = torch.Tensor(
                augment_gamma(img[i], gamma_range=self.gamma_range, invert_image=False, epsilon=1e-7, per_channel=False,
                              retain_stats=False))
        return image_new

    def __call__(self, img, mask):
        new_img = img
        prob_tmp = random.uniform(0, 1)
        if prob_tmp <= self.prob:
            new_img = torch.Tensor(Range01(self._gamma(new_img.cpu().numpy()))).float()
        return new_img, mask

class ToTensor:
    def __init__(self):
        self.to_tensor = transforms.ToTensor()

    def __call__(self, img, mask):
        img = self.to_tensor(img)
        mask = torch.from_numpy(np.array(mask))
        return img, mask[None]


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img, mask):
        return normalize(img, self.mean, self.std, False), mask


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask





