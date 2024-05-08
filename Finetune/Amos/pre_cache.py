# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
from monai.transforms import *
from monai.utils.enums import MetricReduction
from monai.handlers import StatsHandler, from_engine
from monai import data, transforms
from monai.data import *

# import resource
#
# rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
# resource.setrlimit(resource.RLIMIT_NOFILE, (8192, rlimit[1]))
# print('Setting resource limit:', str(resource.getrlimit(resource.RLIMIT_NOFILE)))

# os.environ['CUDA_VISIBLE_DEVICES'] = "2"
# os.environ['MASTER_ADDR'] = 'localhost'
# os.environ['MASTER_PORT'] = '28890'

parser = argparse.ArgumentParser(description="Swin UNETR segmentation pipeline")
parser.add_argument(
    "--pretrained_dir", default="./runs/logs_0.9129/", type=str, help="pretrained checkpoint directory"
)
parser.add_argument("--data_dir", default="D:\data/amos22", type=str, help="dataset directory")
parser.add_argument("--exp_name", default="logs_0.9129", type=str, help="experiment name")
parser.add_argument("--json_list", default="dataset_CT.json", type=str, help="dataset json file")
parser.add_argument(
    "--pretrained_model_name",
    default="model_best.pt",
    type=str,
    help="pretrained model name",
)
roi = 96
parser.add_argument("--use_normal_dataset", default=True, help="use monai Dataset class")
parser.add_argument("--feature_size", default=48, type=int, help="feature size")
parser.add_argument("--batch_size", default=1, type=int, help="number of batch size")
parser.add_argument("--sw_batch_size", default=4, type=int, help="number of sliding window batch size")
parser.add_argument("--infer_overlap", default=0.75, type=float, help="sliding window inference overlap")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=16, type=int, help="number of output channels")
parser.add_argument("--a_min", default=-175.0, type=float, help="a_min in ScaleIntensityRanged")
parser.add_argument("--a_max", default=250.0, type=float, help="a_max in ScaleIntensityRanged")
parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
parser.add_argument("--space_x", default=1.5, type=float, help="spacing in x direction")
parser.add_argument("--space_y", default=1.5, type=float, help="spacing in y direction")
parser.add_argument("--space_z", default=1.5, type=float, help="spacing in z direction")
parser.add_argument("--roi_x", default=roi, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=roi, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=roi, type=int, help="roi size in z direction")
parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
parser.add_argument("--distributed", action="store_true", help="start distributed training")
parser.add_argument("--workers", default=8, type=int, help="number of workers")
parser.add_argument("--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability")
parser.add_argument("--RandRotate90d_prob", default=0.2, type=float, help="RandRotate90d aug probability")
parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float, help="RandScaleIntensityd aug probability")
parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float, help="RandShiftIntensityd aug probability")
parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
parser.add_argument("--use_checkpoint", default=True, help="use gradient checkpointing to save memory")


def main():
    args = parser.parse_args()
    args.test_mode = True
    val_loader, test_transforms = get_loader(args)

    with torch.no_grad():
        for idx, batch_data in enumerate(val_loader):
            print(idx)
            # print(batch_data.keys())

            # img_name = batch_data["image_meta_dict"]["filename_or_obj"][0].split("/")[-1]

            # raw_data = np.load('./raw_data.npy', allow_pickle=True)
            # raw_data = raw_data.item()
            #
            # shapes, affines = raw_data['shape'], raw_data['affine']
            # shape, affine = shapes[img_name], affines[img_name]
            # h, w, d = shape
            # target_shape = (h, w, d)

            data = batch_data["image"]


if __name__ == "__main__":
    main()