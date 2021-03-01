
# LIMVSOD
## A Novel Long-term Iterative Ming Scheme For Video Salient Object Detection

## Prerequisites
The training and testing experiments are conducted using PyTorch 1.1.0 with a single GeForce RTX 2080Ti GPU with 11GB Memory.
* Windows
* CUDA v10.1, cudnn v.7.5.0
* PyTorch 1.1.0
* torchvision

## Update
The training code will be uploaded
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

Download the following [pre-trained models](https://pan.baidu.com/s/1n6nvPP1MvBGqGo26I32beQ (code:uidl) into pretmodel folder. 

3.Train
run train.py

4.Test
run test.py

## Train
### Prepare Object Detection
Please download the object detection code according to this github link [EfficientDet](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch), we set threshold = 0.01, iou_threshold = 0.2. The maximum number of boxes is 10.
You can download it yourself according to the link and modify the relevant parameters, or you can directly use our modified code, the relevant code is in the OB folder.
### Prepare Motion Saliency
First use [PWC-net](https://github.com/sniklaus/pytorch-pwc) to generate the optical flow, and then use the davis training set of VSOD to fine-tune the [CPD](https://github.com/wuzhe71/CPD) model
### Prepare Classifier
Please cd Classifier， Then run 
### k-means
 

## Data
Our saliency detection results can be downloaded on [BaiduCloud](https://pan.baidu.com/s/1Nm-VLBMGbR9fmdSqQWAEQg (code:pnvp). 


Thanks to [CPD](https://github.com/wuzhe71/CPD), [PWC-net](https://github.com/sniklaus/pytorch-pwc) and [EfficientDet](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch)



