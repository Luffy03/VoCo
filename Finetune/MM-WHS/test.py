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
from torch.cuda.amp import GradScaler, autocast
from utils.data_utils import get_loader
from utils.utils import dice, resample_3d
from utils.utils import AverageMeter, distributed_all_gather

from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR
from monai.transforms import Activations, AsDiscrete, Compose
from monai.utils.enums import MetricReduction

os.environ['CUDA_VISIBLE_DEVICES'] = "2"
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '28890'

parser = argparse.ArgumentParser(description="Swin UNETR segmentation pipeline")
parser.add_argument(
    "--pretrained_dir", default="./runs/logs_0.9054/", type=str, help="pretrained checkpoint directory"
)
parser.add_argument("--data_dir", default="/data/jiaxin/data/MM-WHS/ct_train/", type=str, help="dataset directory")
parser.add_argument("--exp_name", default="MMWHS", type=str, help="experiment name")
parser.add_argument("--json_list", default="./dataset_CT.json", type=str, help="dataset json file")
parser.add_argument(
    "--pretrained_model_name",
    default="model_0.9054.pt",
    type=str,
    help="pretrained model name",
)
roi=64
parser.add_argument("--feature_size", default=48, type=int, help="feature size")
parser.add_argument("--batch_size", default=1, type=int, help="number of batch size")
parser.add_argument("--sw_batch_size", default=8, type=int, help="number of sliding window batch size")
parser.add_argument("--infer_overlap", default=0.7, type=float, help="sliding window inference overlap")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=8, type=int, help="number of output channels")
parser.add_argument("--a_min", default=-1000.0, type=float, help="a_min in ScaleIntensityRanged")
parser.add_argument("--a_max", default=1000.0, type=float, help="a_max in ScaleIntensityRanged")
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
    output_directory = "./outputs/" + args.exp_name
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    val_loader = get_loader(args)[1]
    pretrained_dir = args.pretrained_dir
    model_name = args.pretrained_model_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_pth = os.path.join(pretrained_dir, model_name)
    model = SwinUNETR(
        img_size=(64, 64, 64),
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        feature_size=args.feature_size,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        use_checkpoint=args.use_checkpoint,
    )
    inf_size = [args.roi_x, args.roi_y, args.roi_z]
    model_inferer = partial(
        sliding_window_inference,
        roi_size=inf_size,
        sw_batch_size=args.sw_batch_size,
        predictor=model,
        overlap=args.infer_overlap,
    )

    model_dict = torch.load(pretrained_pth)["state_dict"]
    model.load_state_dict(model_dict, strict=True)
    model.eval()
    model.to(device)

    with torch.no_grad():
        all_dice = None
        num = np.zeros(7)
        dice_list_case = []
        for idx, batch_data in enumerate(val_loader):
            img_name = batch_data["image_meta_dict"]["filename_or_obj"][0].split("/")[-1]

            if isinstance(batch_data, list):
                data, target = batch_data
            else:
                data, target = batch_data["image"], batch_data["label"]
            data, target = data.cuda(), target.cuda()
            with autocast(enabled=False):
                if model_inferer is not None:
                    logits = model_inferer(data)
                else:
                    logits = model(data)
            if not logits.is_cuda:
                target = target.cpu()

            outputs = torch.argmax(logits, 1).cpu().numpy()
            outputs = outputs.astype(np.uint8)[0]
            val_labels = target.cpu().numpy()[0, 0, :, :, :]

            len_class = len(list(np.unique(val_labels))) - 1
            dice_list_sub = []
            for i in range(1, 8):
                # judge this class exist or not, ignore background
                num[i - 1] += (np.sum(val_labels == i) > 0).astype(np.uint8)
                organ_Dice = dice(outputs == i, val_labels == i)
                dice_list_sub.append(organ_Dice)

            mean_dice = np.sum(dice_list_sub) / len_class
            print("Mean Organ Dice: {}".format(mean_dice))

            # acc of each organ
            print("Organ Dice:", dice_list_sub)

            if all_dice is None:
                all_dice = (np.asarray(dice_list_sub)).copy()
            else:
                all_dice = all_dice + np.asarray(dice_list_sub)
            print("Organ Dice accumulate:", all_dice*100 / num)

            dice_list_case.append(mean_dice)
            print("Overall Mean Dice: {}".format(100*np.mean(dice_list_case)))

            # # save predict
            print(logits.shape)
            val_outputs = torch.argmax(logits, 1).cpu().numpy()
            np.save(os.path.join(output_directory, 'pre'+ img_name[9:13]+'.npy'), val_outputs.astype(np.uint8)[0])
            # save label
            val_labels = target.cpu().numpy()
            np.save(os.path.join(output_directory, 'label' + img_name[9:13] + '.npy'), val_labels.astype(np.uint8)[0][0])

            # save input
            img = data.cpu().numpy()
            img = img * 255
            print(np.max(img))
            np.save(os.path.join(output_directory, 'img' + img_name[9:13] + '.npy'), img.astype(np.uint8)[0][0])


if __name__ == "__main__":
    main()
