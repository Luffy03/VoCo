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
from collections.abc import Callable, Sequence
from torch.utils.data import Dataset as _TorchDataset
from torch.utils.data import Subset
import collections
import numpy as np
from monai.data import *
import pickle
from monai.transforms import *
from math import *


def get_loader_1k(args):
    splits1 = "/btcv.json"
    splits2 = "/dataset_TCIAcovid19_0.json"
    splits3 = "/dataset_LUNA16_0.json"
    # splits3 = "/dataset_HNSCC_0.json"
    # splits4 = "/dataset_TCIAcolon_v2_0.json"
    # splits5 = "/dataset_LIDC_0.json"
    list_dir = "./jsons"
    jsonlist1 = list_dir + splits1
    jsonlist2 = list_dir + splits2
    jsonlist3 = list_dir + splits3
    # jsonlist4 = list_dir + splits4
    # jsonlist5 = list_dir + splits5
    datadir1 = "/data/linshan/CTs/BTCV"
    datadir2 = "/data/linshan/CTs/TCIAcovid19"
    datadir3 = "/data/linshan/CTs/Luna16-jx"
    num_workers = 8
    datalist1 = load_decathlon_datalist(jsonlist1, False, "training", base_dir=datadir1)
    print("Dataset 1 BTCV: number of data: {}".format(len(datalist1)))
    new_datalist1 = []
    for item in datalist1:
        item_dict = {"image": item["image"]}
        new_datalist1.append(item_dict)

    datalist2 = load_decathlon_datalist(jsonlist2, False, "training", base_dir=datadir2)
    print("Dataset 2 Covid 19: number of data: {}".format(len(datalist2)))

    datalist3 = load_decathlon_datalist(jsonlist3, False, "training", base_dir=datadir3)
    print("Dataset 3 Luna: number of data: {}".format(len(datalist3)))
    new_datalist3 = []
    for item in datalist3:
        item_dict = {"image": item["image"]}
        new_datalist3.append(item_dict)

    vallist1 = load_decathlon_datalist(jsonlist1, False, "validation", base_dir=datadir1)
    vallist2 = load_decathlon_datalist(jsonlist2, False, "validation", base_dir=datadir2)
    vallist3 = load_decathlon_datalist(jsonlist3, False, "validation", base_dir=datadir3)
    # vallist4 = load_decathlon_datalist(jsonlist4, False, "validation", base_dir=datadir4)
    # vallist5 = load_decathlon_datalist(jsonlist5, False, "validation", base_dir=datadir5)
    datalist = new_datalist1 + datalist2 + new_datalist3  # + datalist4 + datalist5
    # datalist = new_datalist1
    val_files = vallist1 + vallist2 + vallist3  # + vallist4 + vallist5
    print("Dataset all training: number of data: {}".format(len(datalist)))
    print("Dataset all validation: number of data: {}".format(len(val_files)))

    train_transforms = Compose([LoadImaged(keys=["image"], image_only=True, dtype=np.int16),
                                EnsureChannelFirstd(keys=["image"]),
                                Orientationd(keys=["image"], axcodes="RAS"),
                                ScaleIntensityRanged(
                                    keys=["image"], a_min=args.a_min, a_max=args.a_max,
                                    b_min=args.b_min, b_max=args.b_max, clip=True),
                                SpatialPadd(keys="image", spatial_size=[args.roi_x, args.roi_y,
                                                                        args.roi_z]),
                                CropForegroundd(keys=["image"], source_key="image"),
                                SpatialCropd(keys=["image"], roi_start=[60, 80, 0],
                                             roi_end=[440, 380, 10000]),
                                Resized(keys=["image"], mode="trilinear", align_corners=True,
                                        spatial_size=(384, 384, 96)),

                                # Random
                                RandShiftIntensityd(keys="image", offsets=0.1, prob=0.0),
                                CropForegroundd(keys="image", source_key="image", select_fn=threshold),
                                Resized(keys="image", mode="bilinear", align_corners=True,
                                        spatial_size=(384, 384, 96)),

                                VoCoAugmentation(args, aug=True)
                                ])

    val_transforms = Compose([LoadImaged(keys=["image"], image_only=True, dtype=np.int16),
                              EnsureChannelFirstd(keys=["image"]),
                              Orientationd(keys=["image"], axcodes="RAS"),
                              ScaleIntensityRanged(
                                  keys=["image"], a_min=args.a_min, a_max=args.a_max,
                                  b_min=args.b_min, b_max=args.b_max, clip=True),
                              SpatialPadd(keys="image", spatial_size=[args.roi_x, args.roi_y,
                                                                      args.roi_z]),
                              CropForegroundd(keys=["image"], source_key="image"),
                              SpatialCropd(keys=["image"], roi_start=[60, 80, 0],
                                           roi_end=[440, 380, 10000]),
                              Resized(keys=["image"], mode="trilinear", align_corners=True,
                                      spatial_size=(384, 384, 96)),
                              VoCoAugmentation(args, aug=False)
                              ])

    if args.cache_dataset:
        print("Using MONAI Cache Dataset")
        train_ds = CacheDataset(data=datalist, transform=train_transforms,
                                cache_rate=0.5, num_workers=num_workers)
    elif args.smartcache_dataset:
        print("Using MONAI SmartCache Dataset")
        train_ds = SmartCacheDataset(
            data=datalist,
            transform=train_transforms,
            replace_rate=1.0,
            cache_num=2 * args.batch_size * args.sw_batch_size,
        )
    else:
        print("Using Persistent dataset")
        # train_ds = Dataset(data=datalist, transform=train_transforms)
        train_ds = PersistentDataset(data=datalist,
                                     transform=train_transforms,
                                     pickle_protocol=pickle.HIGHEST_PROTOCOL,
                                     cache_dir='/data/linshan/cache/1.5k')

    if args.distributed:
        train_sampler = DistributedSampler(dataset=train_ds, even_divisible=True, shuffle=True)
    else:
        train_sampler = None
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, num_workers=num_workers, sampler=train_sampler,
        drop_last=True, pin_memory=True
    )

    val_ds = PersistentDataset(data=val_files,
                               transform=val_transforms,
                               pickle_protocol=pickle.HIGHEST_PROTOCOL,
                               cache_dir='/data/linshan/cache/1.5k')
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=num_workers, shuffle=False, drop_last=True)

    return train_loader, val_loader


def random_split(ls):
    length = len(ls)
    train_ls = ls[:ceil(length * 0.9)]
    val_ls = ls[ceil(length * 0.9):]
    return train_ls, val_ls


def get_loader(args):
    splits1 = "/btcv.json"
    splits2 = "/dataset_TCIAcovid19_0.json"
    splits3 = "/dataset_LUNA16_0.json"
    splits4 = "/stoic21.json"
    splits5 = "/Totalsegmentator_dataset.json"
    splits6 = "/flare23.json"
    splits7 = "/HNSCC.json"

    list_dir = "./jsons/"
    jsonlist1 = list_dir + splits1
    jsonlist2 = list_dir + splits2
    jsonlist3 = list_dir + splits3
    jsonlist4 = list_dir + splits4
    jsonlist5 = list_dir + splits5
    jsonlist6 = list_dir + splits6
    jsonlist7 = list_dir + splits7

    datadir1 = "./data/BTCV"
    datadir2 = "./data/TCIAcovid19"
    datadir3 = "./data/Luna16-jx"
    datadir4 = "./data/stoic21"
    datadir5 = "./data/Totalsegmentator_dataset"
    datadir6 = "./data/Flare23"
    datadir7 = "./data/HNSCC_convert_v1"

    num_workers = 16
    datalist1 = load_decathlon_datalist(jsonlist1, False, "training", base_dir=datadir1)
    print("Dataset 1 BTCV: number of data: {}".format(len(datalist1)))
    new_datalist1 = []
    for item in datalist1:
        item_dict = {"image": item["image"]}
        new_datalist1.append(item_dict)

    datalist2 = load_decathlon_datalist(jsonlist2, False, "training", base_dir=datadir2)
    print("Dataset 2 Covid 19: number of data: {}".format(len(datalist2)))

    datalist3 = load_decathlon_datalist(jsonlist3, False, "training", base_dir=datadir3)
    print("Dataset 3 Luna: number of data: {}".format(len(datalist3)))
    new_datalist3 = []
    for item in datalist3:
        item_dict = {"image": item["image"]}
        new_datalist3.append(item_dict)

    datalist4 = load_decathlon_datalist(jsonlist4, False, "training", base_dir=datadir4)
    # datalist4, vallist4 = random_split(datalist4)
    print("Dataset 4 TCIA Colon: number of data: {}".format(len(datalist4)))

    datalist5 = load_decathlon_datalist(jsonlist5, False, "training", base_dir=datadir5)
    # datalist5, vallist5 = random_split(datalist5)
    print("Dataset 5 Totalsegmentator: number of data: {}".format(len(datalist5)))

    datalist6 = load_decathlon_datalist(jsonlist6, False, "training", base_dir=datadir6)
    # datalist6, vallist6 = random_split(datalist6)
    print("Dataset 6 Flare23: number of data: {}".format(len(datalist6)))

    datalist7 = load_decathlon_datalist(jsonlist7, False, "training", base_dir=datadir7)
    # datalist7, vallist7 = random_split(datalist7)
    print("Dataset 7 HNSCC: number of data: {}".format(len(datalist7)))

    vallist1 = load_decathlon_datalist(jsonlist1, False, "validation", base_dir=datadir1)
    vallist2 = load_decathlon_datalist(jsonlist2, False, "validation", base_dir=datadir2)
    vallist3 = load_decathlon_datalist(jsonlist3, False, "validation", base_dir=datadir3)

    datalist = new_datalist1 + datalist2 + new_datalist3 + datalist4 + datalist5 + datalist6 + datalist7
    val_files = vallist1 + vallist2 + vallist3  # + vallist4 + vallist5 + vallist6 + vallist7
    print("Dataset all training: number of data: {}".format(len(datalist)))
    print("Dataset all validation: number of data: {}".format(len(val_files)))

    train_transforms = Compose([LoadImaged(keys=["image"], image_only=True, dtype=np.int16),
                                EnsureChannelFirstd(keys=["image"]),
                                Orientationd(keys=["image"], axcodes="RAS"),
                                ScaleIntensityRanged(
                                    keys=["image"], a_min=args.a_min, a_max=args.a_max,
                                    b_min=args.b_min, b_max=args.b_max, clip=True),
                                CropForegroundd(keys="image", source_key="image", select_fn=threshold),
                                Resized(keys="image", mode="bilinear", align_corners=True,
                                        spatial_size=(384, 384, 96)),
                                VoCoAugmentation(args, aug=True)
                                ])

    if args.cache_dataset:
        print("Using MONAI Cache Dataset")
        train_ds = CacheDataset(data=datalist, transform=train_transforms,
                                cache_rate=0.5, num_workers=num_workers)
    elif args.smartcache_dataset:
        print("Using MONAI SmartCache Dataset")
        train_ds = SmartCacheDataset(
            data=datalist,
            transform=train_transforms,
            replace_rate=1.0,
            cache_num=2 * args.batch_size * args.sw_batch_size,
        )
    else:
        print("Using Persistent dataset")
        # train_ds = Dataset(data=datalist, transform=train_transforms)
        train_ds = PersistentDataset(data=datalist,
                                     transform=train_transforms,
                                     pickle_protocol=pickle.HIGHEST_PROTOCOL,
                                     cache_dir='/data/linshan/cache/10k')

    if args.distributed:
        train_sampler = DistributedSampler(dataset=train_ds, even_divisible=True, shuffle=True)
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, num_workers=num_workers, sampler=train_sampler, shuffle=True,
        drop_last=True, pin_memory=True
    )

    return train_loader


def threshold(x):
    # threshold at 0
    return x > 0.3


class VoCoAugmentation():
    def __init__(self, args, aug):
        self.args = args
        self.aug = aug

    def __call__(self, x_in):
        crops_trans = get_crop_transform(roi_small=self.args.roi_x, aug=self.aug)

        vanilla_trans, labels = get_vanilla_transform(num=self.args.sw_batch_size,
                                                      roi_small=self.args.roi_x, aug=self.aug)

        imgs = []
        for trans in vanilla_trans:
            img = trans(x_in)
            imgs.append(img)

        crops = []
        for trans in crops_trans:
            crop = trans(x_in)
            crops.append(crop)

        return imgs, labels, crops


def get_vanilla_transform(num=2, num_crops=4, roi_small=64, roi=96, max_roi=384, aug=False):
    vanilla_trans = []
    labels = []
    for i in range(num):
        center_x, center_y, label = get_position_label(roi=roi,
                                                       max_roi=max_roi,
                                                       num_crops=num_crops)
        if aug:
            trans = Compose([
                SpatialCropd(keys=['image'],
                             roi_center=[center_x, center_y, roi // 2],
                             roi_size=[roi, roi, roi]),
                Resized(keys=["image"], mode="bilinear", align_corners=True,
                        spatial_size=(roi_small, roi_small, roi_small)),
                RandFlipd(keys=["image"], prob=0.2, spatial_axis=0),
                RandFlipd(keys=["image"], prob=0.2, spatial_axis=1),
                RandFlipd(keys=["image"], prob=0.2, spatial_axis=2),
                RandRotate90d(keys=["image"], prob=0.2, max_k=3),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=0.1),
                ToTensord(keys=["image"])])
        else:
            trans = Compose([
                SpatialCropd(keys=['image'],
                             roi_center=[center_x, center_y, roi // 2],
                             roi_size=[roi, roi, roi]),
                Resized(keys=["image"], mode="bilinear", align_corners=True,
                        spatial_size=(roi_small, roi_small, roi_small)),
                ToTensord(keys=["image"])])

        vanilla_trans.append(trans)
        labels.append(label)

    labels = np.concatenate(labels, 0).reshape(num, num_crops * num_crops)

    return vanilla_trans, labels


def get_crop_transform(num=4, roi_small=64, roi=96, aug=False):
    voco_trans = []
    # not symmetric at axis x !!!
    for i in range(num):
        for j in range(num):
            center_x = (i + 1 / 2) * roi
            center_y = (j + 1 / 2) * roi
            center_z = roi // 2

            if aug:
                trans = Compose([
                    SpatialCropd(keys=['image'],
                                 roi_center=[center_x, center_y, center_z],
                                 roi_size=[roi, roi, roi]),
                    Resized(keys=["image"],
                            mode="bilinear",
                            align_corners=True,
                            spatial_size=(roi_small, roi_small, roi_small)
                            ),
                    Resized(keys=["image"], mode="bilinear", align_corners=True,
                            spatial_size=(roi_small, roi_small, roi_small)),
                    RandFlipd(keys=["image"], prob=0.2, spatial_axis=0),
                    RandFlipd(keys=["image"], prob=0.2, spatial_axis=1),
                    RandFlipd(keys=["image"], prob=0.2, spatial_axis=2),
                    RandRotate90d(keys=["image"], prob=0.2, max_k=3),
                    RandShiftIntensityd(keys="image", offsets=0.1, prob=0.1),
                    ToTensord(keys=["image"])],
                )
            else:
                trans = Compose([
                    SpatialCropd(keys=['image'],
                                 roi_center=[center_x, center_y, center_z],
                                 roi_size=[roi, roi, roi]),
                    Resized(keys=["image"],
                            mode="bilinear",
                            align_corners=True,
                            spatial_size=(roi_small, roi_small, roi_small)
                            ),
                    ToTensord(keys=["image"])],
                )

            voco_trans.append(trans)

    return voco_trans


def get_position_label(roi=96, base_roi=96, max_roi=384, num_crops=4):
    half = roi // 2
    center_x, center_y = np.random.randint(low=half, high=max_roi - half), \
        np.random.randint(low=half, high=max_roi - half)
    # center_x, center_y = np.random.randint(low=half, high=half+1), \
    #     np.random.randint(low=half, high=half+1)
    # center_x, center_y = roi + half, roi + half
    # print(center_x, center_y)

    x_min, x_max = center_x - half, center_x + half
    y_min, y_max = center_y - half, center_y + half

    total_area = roi * roi
    labels = []
    for i in range(num_crops):
        for j in range(num_crops):
            crop_x_min, crop_x_max = i * base_roi, (i + 1) * base_roi
            crop_y_min, crop_y_max = j * base_roi, (j + 1) * base_roi

            dx = min(crop_x_max, x_max) - max(crop_x_min, x_min)
            dy = min(crop_y_max, y_max) - max(crop_y_min, y_min)
            if dx <= 0 or dy <= 0:
                area = 0
            else:
                area = (dx * dy) / total_area
            labels.append(area)

    labels = np.asarray(labels).reshape(1, num_crops * num_crops)

    return center_x, center_y, labels


if __name__ == '__main__':
    center_x, center_y, labels = get_position_label()
    print(center_x, center_y, labels)