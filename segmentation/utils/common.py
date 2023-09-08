import SimpleITK as sitk
import numpy as np
from scipy import ndimage
import torch, random
import math

def to_one_hot_3d(target, n_classes=2):  # shape = [batch, s, h, w] to onehot
    n, s, h, w = target.size()
    one_hot = torch.zeros(n, n_classes, s, h, w).scatter_(1, target.view(n, 1, s, h, w), 1)
    return one_hot

def get_one_hot(mask, N):
    for i in range(mask.size(0)):
      label = mask[i]
      size = list(label.size())
      label = label.view(-1)
      ones = torch.sparse.torch.eye(N)
      ones = ones.index_select(0, label)
      size.append(N)
      mask[i] = ones.view(*size)
      
    return mask

def random_crop_3d(img, label, crop_size):
    random_x_max = img.shape[0] - crop_size[0]
    random_y_max = img.shape[1] - crop_size[1]
    random_z_max = img.shape[2] - crop_size[2]

    if random_x_max < 0 or random_y_max < 0 or random_z_max < 0:
        return None

    x_random = random.randint(0, random_x_max)
    y_random = random.randint(0, random_y_max)
    z_random = random.randint(0, random_z_max)

    crop_img = img[x_random:x_random + crop_size[0], y_random:y_random + crop_size[1], z_random:z_random + crop_size[2]]
    crop_label = label[x_random:x_random + crop_size[0], y_random:y_random + crop_size[1],z_random:z_random + crop_size[2]]

    return crop_img, crop_label

def center_crop_3d(img, label, slice_num=16):
    if img.shape[0] < slice_num:
        return None
    left_x = img.shape[0]//2 - slice_num//2
    right_x = img.shape[0]//2 + slice_num//2

    crop_img = img[left_x:right_x]
    crop_label = label[left_x:right_x]
    return crop_img, crop_label

def load_file_name_list(file_path):
    file_name_list = []
    with open(file_path, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline().strip()
            if not lines:
                break
                pass
            file_name_list.append(lines)
            pass
    return file_name_list

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

def adjust_learning_rate(optimizer, epoch, args, num_loops=100):#Cosine decay
    if epoch >= num_loops:
        epoch = num_loops - 1
    value = (math.cos(epoch * math.pi / num_loops) + 1.0) * 0.5
    lr = value * (args.lr - args.lr/100) + args.lr/100
    #lr = args.lr * (0.1 ** (epoch // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_learning_rate_V2(optimizer, epoch, args, num_loops=100, exponent=0.9):#Polynomial decay
    lr = args.lr * (1 - epoch / num_loops)**exponent
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr