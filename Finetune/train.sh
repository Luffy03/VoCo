now=$(date +"%Y%m%d_%H%M%S")
logdir=runs/logs
mkdir -p $logdir

data_dir=/project/medimgfmod/CT/AbdomenAtlasMini1.0/
cache_dataset=False
cache_dir=/scratch/medimgfmod/CT/cache/Atlas

torchrun --master_port=21472 main.py \
    --data_dir $data_dir --cache_dataset $cache_dataset --cache_dir $cache_dir --logdir $logdir | tee $logdir/$now.txt