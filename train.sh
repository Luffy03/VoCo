now=$(date +"%Y%m%d_%H%M%S")
logdir=runs/logs_10k
mkdir -p $logdir

torchrun --master_port=28802 voco_train.py \
    --logdir $logdir | tee $logdir/$now.txt