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
from dataset.dataloader_bdmap import get_loader_Atlas
from utils.utils import dice, resample_3d
from utils.utils import AverageMeter, distributed_all_gather
from tqdm import tqdm
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR
from monai.transforms import Activations, AsDiscrete, Compose
from monai.utils.enums import MetricReduction
from utils.utils import *
import cv2
from PIL import Image

# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '28890'

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (8192, rlimit[1]))
print('Setting resource limit:', str(resource.getrlimit(resource.RLIMIT_NOFILE)))

parser = argparse.ArgumentParser(description="Swin UNETR segmentation pipeline")
parser.add_argument(
    "--pretrained_dir", default="./runs/logs_scratch_v2/", type=str, help="pretrained checkpoint directory"
)
parser.add_argument("--data_dir", default="/project/medimgfmod/CT/AbdomenAtlasMini1.0/", type=str, help="dataset directory")
parser.add_argument("--data_txt_path", default='./dataset/dataset_list', help="dataset json file")
parser.add_argument("--dataset_list", default=['AbdomenAtlas1.0'], help="dataset json file")

parser.add_argument("--pos", default=1, type=int, help="number of positive sample")
parser.add_argument("--neg", default=0, type=int, help="number of negative sample")

roi=96
parser.add_argument("--cache_dataset", default=False, help="use monai CACHE Dataset class")
parser.add_argument("--feature_size", default=48, type=int, help="feature size")
parser.add_argument("--batch_size", default=8, type=int, help="number of batch size")
parser.add_argument("--sw_batch_size", default=1, type=int, help="number of sliding window batch size")
parser.add_argument("--infer_overlap", default=0.75, type=float, help="sliding window inference overlap")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=14, type=int, help="number of output channels")
parser.add_argument("--a_min", default=-175.0, type=float, help="a_min in ScaleIntensityRanged")
parser.add_argument("--a_max", default=250.0, type=float, help="a_max in ScaleIntensityRanged")
parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
parser.add_argument("--space_x", default=1.5, type=float, help="spacing in x direction")
parser.add_argument("--space_y", default=1.5, type=float, help="spacing in y direction")
parser.add_argument("--space_z", default=2.0, type=float, help="spacing in z direction")
parser.add_argument("--roi_x", default=roi, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=roi, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=roi, type=int, help="roi size in z direction")
parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
parser.add_argument("--distributed", action="store_true", help="start distributed training")
parser.add_argument("--workers", default=16, type=int, help="number of workers")
parser.add_argument("--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability")
parser.add_argument("--RandRotate90d_prob", default=0.2, type=float, help="RandRotate90d aug probability")
parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float, help="RandScaleIntensityd aug probability")
parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float, help="RandShiftIntensityd aug probability")
parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
parser.add_argument("--use_checkpoint", default=True, help="use gradient checkpointing to save memory")

import warnings
warnings.filterwarnings('ignore')


def main():
    args = parser.parse_args()
    args.test_mode = True
    loader = get_loader_Atlas(args)

    # num = 0
    # vis_path = './vis/'
    # check_dir(vis_path)

    with torch.no_grad():
        for batch_data in tqdm(loader[0]):
            image, label = batch_data["image"], batch_data["label"]

            print(image.shape, label.shape, torch.unique(label))

            # img = image[0][0].data.cpu().numpy()
            # label = label[0][0].data.cpu().numpy()
            #
            # h, w, c = img.shape
            # cmap = color_map()
            #
            # for j in range(c):
            #     im = img[:, :, j]
            #     la = label[:, :, j]
            #
            #     if len(list(np.unique(la))) > 1:
            #         im = (255 * im).astype(np.uint8)
            #         la = Image.fromarray(la.astype(np.uint8), mode='P')
            #         la.putpalette(cmap)
            #         num += 1
            #
            #         cv2.imwrite(vis_path+str(num)+'_im.png', im)
            #         la.save(vis_path+str(num)+'_lab.png')


if __name__ == "__main__":
    main()
