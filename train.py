import os

# Multiple gpus
os.system("CUDA_VISIBLE_DEVICES=0,1 bash ./mmsegmentation/tools/dist_train.sh rdrnet-s-simple_2xb6-120k_cityscapes-1024x1024.py 2 --work-dir ./weight/seg")

# Single gpu
# os.system("CUDA_VISIBLE_DEVICES=0 python ./mmsegmentation/tools/train.py rdrnet-s-simple_2xb6-120k_cityscapes-1024x1024.py --work-dir ./weight/seg")
