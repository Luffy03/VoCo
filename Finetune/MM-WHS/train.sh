now=$(date +"%Y%m%d_%H%M%S")
logdir=runs/logs
mkdir -p $logdir

torchrun --master_port=21120 --max-restart=10 main.py \
    --logdir $logdir | tee $logdir/$now.txt