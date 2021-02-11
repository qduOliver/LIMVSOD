
# LIMVSOD
## A Novel Long-term Iterative Ming Scheme For Video Salient Object Detection

## Prerequisites
The training and testing experiments are conducted using PyTorch 1.1.0 with a single GeForce RTX 2080Ti GPU with 11GB Memory.
* Windows
* CUDA v10.1, cudnn v.7.5.0
* PyTorch 1.1.0
* torchvision

## Update
The training code has been uploaded
## Todo
Upload data preprocessing code
## Usage
1.Clone

git clone https://github.com/qduOliver/LIMVSOD.git

cd LIMVSOD/

2.Download the datasets

Download the following datasets and unzip them into your_data folder.
All datasets can be downloaded at this [data link](http://dpfan.net/news/).

* Davis
* Segtrack-v2
* Visal
* DAVSOD
* VOS

3.Download the pre-trained models

Download the following [pre-trained models](https://pan.baidu.com/s/1gZ4J85rfMWBM_Ozqey1O6A)(code:nncs) into pretmodel folder. 

3.Train
run train.py

4.Test
run test.py

## Data
Our saliency detection results can be downloaded on [BaiduCloud](https://pan.baidu.com/s/102o67mnMmKzHh2jnSYVJdA)(code:uj9v). 


Thanks to [CPD](https://github.com/wuzhe71/CPD), [PWC-net](https://github.com/sniklaus/pytorch-pwc) and [EfficientDet](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch)



