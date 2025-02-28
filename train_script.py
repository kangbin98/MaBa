import os
import torch
print(torch.cuda.device_count())

if __name__ == '__main__':
    task = f"python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    train.py \
    --name iira \
    --img_aug \
    --batch_size 64 \
    --MLM \
    --loss_names sdm+mlm+id \
    --dataset_name CUHK-PEDES \
    --root_dir data \
    --num_epoch 60"
    os.system(task)



