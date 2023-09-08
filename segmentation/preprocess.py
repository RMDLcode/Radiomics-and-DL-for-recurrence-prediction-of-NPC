import glob
import os.path
import SimpleITK as sitk
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
from scipy import ndimage

def center_crop(img, cropSize):
    img_tmp = img
    cropz = cropSize[0]
    cropy = cropSize[1]
    cropx = cropSize[2]
    z, y, x = img_tmp.shape
    if z < 61:
        cha = 61 - z
        buquan = np.zeros(shape=(cha, y, x))
        img_tmp = np.concatenate((img_tmp, buquan), axis=0)
        startz = 0
    else:
        startz = z-61
    z, y, x = img_tmp.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    # startz = z // 2 - (cropz // 2)

    img_tmp = img_tmp[startz:startz + cropz, starty:starty + cropy, startx:startx + cropx]

    return img_tmp

def get_meta(img):
    meta_data = []
    meta_data.append(img.GetOrigin())
    meta_data.append(img.GetSize())
    meta_data.append(img.GetSpacing())
    meta_data.append(img.GetDirection())
    # print(img.GetDirection())
    return meta_data

def set_meta(img,meta_data):
    if type(img ) == np.ndarray:
        img = sitk.GetImageFromArray(img)
    img.SetOrigin(meta_data[0])
    img.SetSpacing(meta_data[2])
    img.SetDirection(meta_data[3])
    return img

def get_z_score(image):
    shape = image.shape
    image_ = np.reshape(image, [-1, 1])
    stand = StandardScaler()
    image_z = stand.fit_transform(image_)
    image_z = np.reshape(image_z, shape)
    min_val = image_z.min()
    max_val = image_z.max()
    image_z = (image_z - min_val) / (max_val - min_val)
    return image_z.astype(np.float32)

def preprocess(output_path):
    # The original data folder path
    path_test = os.path.join(output_path, r'data_ori/*')
    name_test = glob.glob(path_test)
    cropSize = [60, 320, 320]
    for name in name_test:
        image_path = os.path.join(name, 'T1C_image.nii.gz')
        image_path2 = os.path.join(name, 'T1_image.nii.gz')
        #1 of the label is the GTV region, and 2 is the recurrence region within GTV
        label_path = os.path.join(name, 'T1C_label_GTV_Prim_Rec.nii.gz')

        meta_info = get_meta(sitk.ReadImage(image_path))
        image = sitk.GetArrayFromImage(sitk.ReadImage(image_path))
        image2 = sitk.GetArrayFromImage(sitk.ReadImage(image_path2))
        label = sitk.GetArrayFromImage(sitk.ReadImage(label_path))

        image = get_z_score(image)
        image2 = get_z_score(image2)
        # This is to crop the image smaller to speed up training processing
        image = center_crop(image, cropSize)
        image2 = center_crop(image2, cropSize)
        label = center_crop(label, cropSize)

        lab = label.copy()
        #Images outside GTV are set to 0
        image[lab == 0] = 0
        image2[lab == 0] = 0
        #Data dimension increase
        image = np.expand_dims(image, axis=0)
        image2 = np.expand_dims(image2, axis=0)

        label[label < 2] = 0
        label[label == 2] = 1

        mix_image = np.concatenate((image, image2), axis=0)

        #The last six digits of the ID are named and saved in npy format
        os.makedirs(os.path.join(output_path, r'data_tmp/mr'), exist_ok=True)
        os.makedirs(os.path.join(output_path, r'data_tmp/label'), exist_ok=True)
        np.save(os.path.join(os.path.join(output_path, r'data_tmp/mr'), name[-6:]), mix_image)
        np.save(os.path.join(os.path.join(output_path, r'data_tmp/label'), name[-6:]), label)

preprocess(output_path=r'D:')