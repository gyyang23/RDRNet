# Reparameterizable Dual-Resolution Network for Real-time Semantic Segmentation ([arXiv](x))

By Guoyu Yang, Yuan Wang, Daming Shi*

This project is based on [mmsegmentation](https://github.com/open-mmlab/mmsegmentation).


# Highlight

<p align="left">
  <img src="figures/0.png" alt="overview-of-our-method" width="500"/></br>
  <span align="left">The trade-off between inference speed and accuracy for real-time semantic segmentation models on the Cityscapes test set.</span> 
</p>


# Experiment

## Environment
```
python==3.8.10
pytorch==1.12.1
torchvision==0.13.1
mmengine==0.7.3
mmcv==2.0.0
mmsegmentation==1.0.0
```

## Install
Please refer to [mmsegmentation](https://mmsegmentation.readthedocs.io/en/latest/get_started.html) for installation.

## Dataset
```
RDRNet
├── mmsegmentation
├── figures
├── data
│   ├── cityscapes
│   │   ├── leftImg8bit
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── gtFine
│   │   │   ├── train
│   │   │   ├── val
│   ├── VOCdevkit
│   │   ├── VOC2012
│   │   │   ├── JPEGImages
│   │   │   ├── SegmentationClass
│   │   │   ├── ImageSets
│   │   │   │   ├── Segmentation
├── rdrnet-s-simple_2xb6-120k_cityscapes-1024x1024.py
├── rdrnet-s_2xb6-120k_cityscapes-1024x1024.py
├── rdrnet-m_2xb6-120k_cityscapes-1024x1024.py
├── rdrnet-l_2xb6-120k_cityscapes-1024x1024.py
├── rdrnet-s-simple_2xb6-24400_voc2012-512x512.py
├── train.py
├── test.py
```

## Training
Single gpu for train:
```shell
CUDA_VISIBLE_DEVICES=0 python ./mmsegmentation/tools/train.py rdrnet-simple_2xb6-120k_cityscapes-1024x1024.py --work-dir ./weight/seg
```

Multiple gpus for train:
```shell
CUDA_VISIBLE_DEVICES=0,1 bash ./mmsegmentation/tools/dist_train.sh rdrnet-simple_2xb6-120k_cityscapes-1024x1024.py 2 --work-dir ./weight/seg
```

Train in pycharm: If you want to train in pycharm, you can run it in train.py.

see more details at [mmsegmentation](https://github.com/open-mmlab/mmsegmentation).

## Testing
```shell
CUDA_VISIBLE_DEVICES=0 python ./mmsegmentation/tools/test.py rdrnet-simple_2xb6-120k_cityscapes-1024x1024.py ./weight/seg/rdrnet_weight.pth
```

Test in pycharm: If you want to test in pycharm, you can run it in test.py.

see more details at [mmsegmentation](https://github.com/open-mmlab/mmsegmentation).

## Results on Cityscapes
|       Method       |  FPS  | Params (M) | GFLOPs | ImageNet |                                            val                                             | test |
|:------------------:|:-----:|:----------:|:------:|:--------:|:------------------------------------------------------------------------------------------:|:----:|
|     BiSeNetV1      | 65.9  |    13.3    |  118.0 | &#10003; |                                            74.4                                            | 73.6 |
|     BiSeNetV2      | 74.4  |    3.4     |  98.4  | &#10007; |                                            73.6                                            | 72.2 |
|   DDRNet-23-Slim   | 131.7 |    5.7     |  36.3  | &#10007; | [76.3](XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX) | 74.6 |
|     DDRNet-23      | 54.6  |    20.3    |  143.0 | &#10007; | [78.0](XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX) | 77.6 |
|     PIDNet-S       | 102.6 |    7.7     |  47.3  | &#10007; | [76.4](XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX) | 76.2 |
|     PIDNet-M       | 42.0  |    28.7    |  177.0 | &#10007; | [78.2](XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX) | 78.3 |
|     PIDNet-L       | 31.8  |    37.3    |  275.0 | &#10007; | [78.8](XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX) | 78.4 |
|   RDRNet-S-Simple  | 134.6 |    7.2     |  41.0  | &#10007; | [76.8](XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX) | 75.4 |
|     RDRNet-S       | 129.4 |    7.3     |  43.4  | &#10007; | [76.8](XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX) | 76.0 |
|     RDRNet-M       | 52.5  |    26.0    |  162.0 | &#10007; | [78.9](XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX) | 78.3 |
|     RDRNet-L       | 39.0  |    36.9    |  238.0 | &#10007; | [79.3](XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX) | 78.6 |

During the evaluation on the validation set, the training set was utilized for training. During the evaluation on the test set, both the training and validation sets were employed for training. When performing inference on RTX 3090, the image resolution used was 1024 &#10005; 2048.


# Citations

If you find RDRNet useful in your research, please consider citing:
```
@article{yang2023afpn,
  title={AFPN: Asymptotic Feature Pyramid Network for Object Detection},
  author={Yang, Guoyu and Lei, Jie and Zhu, Zhikuan and Cheng, Siyu and Feng, Zunlei and Liang, Ronghua},
  journal={arXiv preprint arXiv:2306.15988},
  year={2023}
}
```
