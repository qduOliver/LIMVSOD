# LIMVSOD
## A Novel Long-term Iterative Ming Scheme For Video Salient Object Detection

## Prerequisites
The training and testing experiments are conducted using PyTorch 1.1.0 with a single GeForce RTX 2080Ti GPU with 11GB Memory.
* Windows
* CUDA v10.1, cudnn v.7.5.0
* PyTorch 1.1.0
* torchvision

## Usage

### Train
1.Generate object Proposals

First, please download the object detection method EfficientDet, then modify the relevant parameters according to the content of the Preparatory work part, and finally rank all objects proposals according to objectness confidence, the maximum of 10 objects proposals.

2.Initialize/Updata the classifier label

We initialize saliency clusters and non-saliency clusters using thresholds. Then, We use two measurements to select a reliable training sample for the first iteration.
(1)the distance to the cluster's centroid (2)the motion saliency degree of each object proposal.
Please run Tools/Sample_filtering.py

3.Generate and assemble patch-level saliency map

Please train patch-level prediction models according to the tips in the paper. Then combine the saliency results at the patch level, and the combination code is the max_map.py file under Tools.

4.KFS(Key Frame Selection)

In order to select high-quality key frames, we take the following two steps：
(1) We compute the S-measure value between FS and MS. See the code Tools/s_measure_fs_ms.py
(2) Run Tools/KFS.py to filter key frame by S-measure value.

5.Online Fine-tuning

The selected key frame is treated as pseudoGT and fine-tuned to obtain the final prediction weight


### Test
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

4.Test
run cpd/test.py

## Preparatory work
### Prepare Object Detection
Please download the object detection code according to this github link [EfficientDet](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch), we set threshold = 0.01, iou_threshold = 0.2. The maximum number of boxes is 10.
You can download it yourself according to the link and modify the relevant parameters, or you can directly use our modified code, the relevant code is in the OB folder.
### Prepare Motion Saliency
First use [PWC-net](https://github.com/sniklaus/pytorch-pwc) to generate the optical flow, and then use the davis training set of VSOD to fine-tune the [CPD](https://github.com/wuzhe71/CPD) model
### Prepare Classifier
Please cd Classifier, Then run train_resnet18.py to train this model, run test_resnet18.py to produce prediction results.
### K-means
Please cd K-means, and run k-means.py 

## Data
Our saliency detection results can be downloaded on [BaiduCloud](https://pan.baidu.com/s/1Nm-VLBMGbR9fmdSqQWAEQg (code:pnvp). 


Thanks to [CPD](https://github.com/wuzhe71/CPD), [PWC-net](https://github.com/sniklaus/pytorch-pwc) and [EfficientDet](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch)



