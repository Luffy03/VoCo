
import argparse
import os
from functools import partial
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from utils.data_test import get_loader
from utils.utils import dice, resample_3d
from utils.utils import AverageMeter, distributed_all_gather

from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR
from monai.transforms import Activations, AsDiscrete, Compose
from monai.utils.enums import MetricReduction
import zipfile
import shutil
import SimpleITK as sitk
from tqdm import tqdm

from utils.utils import *
from PIL import Image


def norm(img):
    new_img = img.copy()
    new_img[img<-175] = 0
    new_img[img>250] = 250

    out = new_img/250
    out = (255*out).astype(np.uint8)
    return out


def check_size():
    data_path = "D:\data/amos22\imagesTr"
    pred_path = 'D:\data/amos22\labelsTr'
    view_path = './pred/view_tr'

    # data_path = "D:\data/amos22/imagesTs"
    # pred_path = './pred/test'
    # view_path = './pred/view_ts'

    check_dir(view_path)
    cmap = color_map()

    ls = os.listdir(pred_path)
    num = 0

    # for i in tqdm(ls):
    i = ls[0]
    # i = 'FLARETs_0031_0000.nii'

    img_path = os.path.join(data_path, i) # i[:-7]+'_0000.nii.gz'
    img_itk = sitk.ReadImage(img_path)
    img = sitk.GetArrayFromImage(img_itk)
    print(img_itk.GetSpacing(), img_itk.GetDirection())
    # img = np.flip(img, 1)
    # img = np.flip(img, 2)

    pred = os.path.join(pred_path, i)
    pred_itk = sitk.ReadImage(pred)
    pred = sitk.GetArrayFromImage(pred_itk)
    print(pred_itk.GetSpacing(), pred_itk.GetDirection())
    # pred = pred.transpose()

    print(img.shape, pred.shape)

    c, h, w = img.shape
    for j in range(c):
        im = img[j, :, :]
        pre = pred[j, :, :].astype(np.uint8)

        pre = Image.fromarray(pre, mode='P')
        pre.putpalette(cmap)

        im = norm(im)

        import cv2
        cv2.imwrite(view_path + '/' + str(j) + '_raw.png', im)
        pre.save(view_path + '/' + str(j) + '_pred.png')


def rename():
    pred_path = './pred/test'

    ls = os.listdir(pred_path)

    for i in ls:
        old_name = os.path.join(pred_path, i)
        new_name = os.path.join(pred_path, i[:-13] + '.nii.gz')
        os.rename(old_name, new_name)


def check_direction():
    data_path = "D:\data\FLARE22\imagesTr" # (-1, -1, 1)

    data_path = "D:\data/amos22\imagesTr" # (1, -1, 1)

    data_path = 'D:\data\BTCV\imagesTr'  # (1, 1, 1)

    ls = os.listdir(data_path)

    for i in tqdm(ls):

        img_path = os.path.join(data_path, i) # i[:-7]+'_0000.nii.gz'
        img_itk = sitk.ReadImage(img_path)
        print(img_itk.GetSpacing(), img_itk.GetDirection())


if __name__ == "__main__":
    check_direction()

