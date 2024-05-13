import torch
import os
from tqdm import tqdm
import numpy as np
from utils.utils import *
from PIL import Image
import matplotlib.pyplot as plt


def read(img, transpose=False):
    img = sitk.ReadImage(img)
    direction = img.GetDirection()
    origin = img.GetOrigin()
    Spacing = img.GetSpacing()

    img = sitk.GetArrayFromImage(img)
    if transpose:
        img = img.transpose(1, 2, 0)

    return img, direction, origin, Spacing


def vis():
    path = 'D:\data\cache\Atlas'
    ls = os.listdir(path)
    num = 0

    for i in ls:
        data = torch.load(os.path.join(path, i))
        img, lab = data['image'], data['label']
        print(img.shape, lab.shape)
        img = img[0].data.cpu().numpy()
        #
        # lab_bg = lab.sum(0).unsqueeze(0)
        #
        # la = lab.argmax(0).unsqueeze(0)
        # la += 1
        # la[lab_bg == 0] = 0
        #
        lab = lab[0].data.cpu().numpy()

        cls_set = list(np.unique(lab))
        print(cls_set)

        h, w, c = img.shape
        cmap = color_map()

        for j in range(c):
            im = img[:, :, j]
            la = lab[:, :, j]

            if len(list(np.unique(la))) > 5:
                im = (255 * im).astype(np.uint8)

                la = Image.fromarray(la.astype(np.uint8), mode='P')
                la.putpalette(cmap)
                num += 1

                fig, axs = plt.subplots(1, 2, figsize=(16, 5))
                axs[0].imshow(im, cmap='gray')
                axs[0].axis("off")

                axs[1].imshow(la)
                axs[1].axis("off")

                plt.tight_layout()
                plt.show()
                plt.close()


def check_original():
    path = 'D:\data\cache\Atlas\BDMAP_00000870/'
    img = read(path + 'ct.nii.gz', True)[0]
    gt = read(path + 'label.nii.gz', True)[0]

    label_path = path+'segmentations'
    organ_ls = ["aorta", "gall_bladder", "kidney_left", "kidney_right", "liver", "pancreas", "postcava", "spleen",
                "stomach"]

    lab = []
    for i in organ_ls:
        la = read(label_path + '/' + i + '.nii.gz', True)[0]
        la = np.expand_dims(la, 0)
        lab.append(la)

    labs = np.concatenate(lab, 0)

    print(img.shape, labs.shape)

    lab_bg = labs.sum(0)

    print(np.unique(labs.sum(0)))
    lab = labs.argmax(0)
    lab += 1
    lab[lab_bg == 0] = 0

    print(np.unique(lab))
    h, w, c = img.shape
    cmap = color_map()

    for j in range(c):
        im = img[:, :, j]
        la = lab[:, :, j]
        g = gt[:, :, j]

        if len(list(np.unique(la))) > 1:
            im = (255 * im).astype(np.uint8)

            la = Image.fromarray(la.astype(np.uint8), mode='P')
            la.putpalette(cmap)

            g = Image.fromarray(g.astype(np.uint8), mode='P')
            g.putpalette(cmap)

            fig, axs = plt.subplots(1, 3, figsize=(16, 5))
            axs[0].imshow(im, cmap='gray')
            axs[0].axis("off")

            axs[1].imshow(la)
            axs[1].axis("off")

            axs[2].imshow(g)
            axs[2].axis("off")

            plt.tight_layout()
            plt.show()
            plt.close()


def exe(path):
    root = '/project/medimgfmod/CT/AbdomenAtlasMini1.0/'
    path = root + path
    label_path = path + '/segmentations'

    organ_ls = ["aorta", "gall_bladder", "kidney_left", "kidney_right", "liver", "pancreas", "postcava", "spleen", "stomach"]

    lab = []
    for i in organ_ls:
        la, direction, origin, Spacing = read(label_path + '/' + i+'.nii.gz')
        la = np.expand_dims(la, 0)
        lab.append(la)

    labs = np.concatenate(lab, 0)
    lab_bg = labs.sum(0)

    lab = labs.argmax(0)
    lab += 1
    lab[lab_bg == 0] = 0

    new = sitk.GetImageFromArray(lab)
    new.SetDirection(direction)
    new.SetOrigin(origin)
    new.SetSpacing(Spacing)
    sitk.WriteImage(new, path + '/' + 'label.nii.gz')
    print('save:', path + '/' + 'label.nii.gz')


def trans_lab(path):
    organ_ls = ["aorta", "gall_bladder", "kidney_left", "kidney_right", "liver", "pancreas", "postcava", "spleen",
                "stomach"]
    lab = []
    for i in organ_ls:
        la = read(path + '/' + i + '.nii.gz', True)[0]
        la = np.expand_dims(la, 0)
        lab.append(la)

    labs = np.concatenate(lab, 0)

    lab_bg = labs.sum(0)
    lab = labs.argmax(0)
    lab += 1
    lab[lab_bg == 0] = 0
    return lab


def check_pred_vis():
    path = 'test_examples/AbdomenAtlasPredict/BDMAP_A0000002/predictions'
    path_temp = 'test_examples/AbdomenAtlasPredict_temp/BDMAP_A0000002/predictions'

    pred, pred_temp = trans_lab(path), trans_lab(path_temp)
    print(np.unique(pred), np.unique(pred_temp))

    h, w, c = pred.shape
    cmap = color_map()

    for j in range(c):
        la = pred[:, :, j]
        g = pred_temp[:, :, j]

        if len(list(np.unique(la))) > 5:

            la = Image.fromarray(la.astype(np.uint8), mode='P')
            la.putpalette(cmap)

            g = Image.fromarray(g.astype(np.uint8), mode='P')
            g.putpalette(cmap)

            fig, axs = plt.subplots(1, 2, figsize=(16, 5))

            axs[0].imshow(la)
            axs[0].axis("off")

            axs[1].imshow(g)
            axs[1].axis("off")

            plt.tight_layout()
            plt.show()
            plt.close()


def check_pred_acc():
    root = '/project/medimgfmod/CT/AbdomenAtlasMini1.0/'
    ls = os.listdir(root)

    num = np.zeros(9)
    from utils.utils import dice
    all_dice = None

    for i in ls:
        path = root + i
        label_path = path + '/segmentations'
        lab = trans_lab(label_path)

        pred_path = os.path.join('./test_examples/AbdomenAtlasPredict_train/' + i, 'predictions')
        pred = trans_lab(pred_path)

        dice_list_sub = []

        for i in range(1, 10):
            num[i - 1] += (np.sum(lab == i) > 0).astype(np.uint8)
            organ_Dice = dice(pred == i, lab == i)
            dice_list_sub.append(organ_Dice)

        if all_dice is None:
            all_dice = (np.asarray(dice_list_sub)).copy()
        else:
            all_dice = all_dice + np.asarray(dice_list_sub)
        print("Organ Dice accumulate:", (all_dice / num), (all_dice / num).mean())


if __name__=='__main__':
    # vis()
    # check_pred_acc()

    # the path to Atlas train
    path = '/project/medimgfmod/CT/AbdomenAtlasMini1.0/'
    ls = os.listdir(path)
    import multiprocessing
    with multiprocessing.Pool(20) as pool:
        pool.map(exe, ls, 1)
