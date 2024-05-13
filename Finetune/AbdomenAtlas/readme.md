# VoCo for AbdomenAtlas

<a href="https://arxiv.org/abs/2402.17300"><img src='https://img.shields.io/badge/arXiv-VoCo-red' alt='Paper PDF'></a>
<a href='https://huggingface.co/datasets/Luffy503/VoCo-10k/tree/main'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue'></a>

CVPR 2024 paper, [**"VoCo: A Simple-yet-Effective Volume Contrastive Learning Framework for 3D Medical Image Analysis"**](https://arxiv.org/abs/2402.17300)

Authors: Linshan Wu, <a href="https://scholar.google.com/citations?user=PfM5gucAAAAJ&hl=en">Jiaxin Zhuang</a>, and <a href="https://scholar.google.com/citations?hl=en&user=Z_t5DjwAAAAJ">Hao Chen</a>

Code for AbdomenAtlasMini1.0 Training and Inference.


## Usage
### Pre-training
Please refer to the official [VoCo repo](https://github.com/Luffy03/VoCo)

### Requirement
I have stored all the required checkpoints and running logs in the project. 
Our Segmentation Training codes are based on [MONAI](https://github.com/Project-MONAI/research-contributions). 
Please also refer to the requirements.txt.

### Training

First edit the data_path of AbdomenAtlasMini1.0 in 'train.sh'
```
data_dir=YOUR AbdomenAtlasMini1.0 PATH
```
Reading 9 label files is not efficient in training and we also find that there are some bugs in 
the originial [data_loader](https://github.com/MrGiovanni/SuPreM/blob/d8a948c96e56f2050109c3ce418bc4caa09420a5/supervised_pretraining/dataset/dataloader_bdmap.py#L147)
(the data of label is loaded but the meta_keys of labels are not loaded, thus the following transform will result in not corresponding image and labels. We provide '/preprocess/try_load.py' for visualization). Thus, we first merge all 9 label files in to one.
```
# preprocess, in exe function of check.py , path=YOUR AbdomenAtlasMini1.0 PATH
python check.py
# merge all 9 organ label files to one label.nii.gz
```

After pre-processing, Training implementation
```
# bash
sh train.sh
# Or using slurm
sbatch train.slurm
```

To accelerate training, we use 'PersistentDataset' to pre-cache data.
```
# in train.sh
cache_dataset=False
# Or with adequate space
cache_dataset=True
cache_dir=Your path to save cache
```

### Inference
First edit the test and prediction path of AbdomenAtlasMini1.0 in 'Atlas_test.sh'
```
test_data_path=Your path to AbdomenAtlasTest
save_prediction_path=Your path to save the prediction AbdomenAtlasTest
```

Inference implementation
```
# bash
sh Atlas_test.sh
```

Inference Visualization
```
# We provide check_pred_vis() function in check.py for you to visualize the predictions
python check.py
```

## Acknowledgement
We thank [MONAI](https://github.com/Project-MONAI/research-contributions) and [SuPreM](https://github.com/MrGiovanni/SuPreM) for part of their codes.
## Citation ✏️ 📄
If you find this repo useful for your research, please consider citing the paper as follows:

```
@inproceedings{VoCo,
  title={VoCo: A Simple-yet-Effective Volume Contrastive Learning Framework for 3D Medical Image Analysis},
  author={Wu, Linshan and Zhuang, Jiaxin and Chen, Hao},
  booktitle={IEEE Conf. Comput. Vis. Pattern Recog.},
  year={2024}
  }
```
