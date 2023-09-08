import glob
import os.path
import SimpleITK as sitk
import numpy as np
import random

def write_train_val_name_list(valid_rate=0.2, fixed_path = r'./reccurence_data/train'):
    data_name_list = os.listdir(os.path.join(fixed_path, "mr"))
    data_num = len(data_name_list)
    print('the fixed dataset total numbers of samples is :', data_num)
    random.shuffle(data_name_list)

    assert valid_rate < 1.0
    train_name_list = data_name_list[0:int(data_num*(1-valid_rate))]#Training set list
    val_name_list = data_name_list[int(data_num*(1-valid_rate)):int(data_num*((1-valid_rate) + valid_rate))]#Verification set list

    write_name_list(train_name_list, "train_path_list.txt", fixed_path)
    write_name_list(val_name_list, "val_path_list.txt", fixed_path)


def write_name_list(name_list, file_name, fixed_path):
    f = open(os.path.join(fixed_path, file_name), 'w')
    for name in name_list:
        ct_path = os.path.join(fixed_path, 'mr', name)
        seg_path = os.path.join(fixed_path, 'label', name)
        f.write(ct_path + ' ' + seg_path + ' ' + "\n")
    f.close()


write_train_val_name_list(valid_rate=0.2, fixed_path = r'./reccurence_data/train')