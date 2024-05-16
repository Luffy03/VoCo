now=$(date +"%Y%m%d_%H%M%S")
logdir=runs/logs_swin_large_scratch
mkdir -p $logdir

torchrun --master_port=20482 main.py \
    --logdir $logdir | tee $logdir/$now.txt