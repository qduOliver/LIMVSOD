# LIMVSOD
## A Novel Long-term Iterative Ming Scheme For Video Salient Object Detection

## Prerequisites
The training and testing experiments are conducted using PyTorch 1.1.0 with a single GeForce RTX 2080Ti GPU with 11GB Memory.
* Windows
* CUDA v10.1, cudnn v.7.5.0
* PyTorch 1.1.0
* torchvision 0.4.0

## Usage


## Usage

### Train
The training process is :OB-->k-means-->Classifier-->Tools-->CPD
1.Generate object Proposals（OB）

First,  please download the object detection method EfficientDet, then modify the relevant parameters according to the content of the Preparatory work part, and finally rank all objects proposals according to objectness confidence, the maximum of 10 objects proposals.

2.Initialize/Updata the classifier label(k-means)

We initialize saliency clusters and non-saliency clusters using thresholds. Then, We use two measurements to select a reliable training sample for the first iteration.
(1)the distance to the cluster's centroid (2)the motion saliency degree of each object proposal.

3.Train the classifier(Classifier)
Gradually improve the credibility of the classifier by training the classifier to iteratively mine reliable data


3.Generate and assemble patch-level saliency map(Tools)
Please train patch-level prediction models according to the tips in the paper. Then combine the saliency results at the patch level, and the combination code is the max_map.py file under Tools.

4.KFS(Key Frame Selection)(Tools)

In order to select high-quality key frames, we take the following two steps：
(1) We compute the S-measure value between FS and MS. See the code Tools/Compute_S_Measure.py
(2) Run Tools/KFS_sen.py to filter key frame by S-measure value.

5.Online Fine-tuning(CPD)

The selected key frame is treated as pseudoGT and fine-tuned CPD to obtain the final prediction weight


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
run cpd/test_CPD.py

## Preparatory work
### Prepare Object Detection
Please download the object detection code according to this github link [EfficientDet](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch), we set threshold = 0.01, iou_threshold = 0.2. The maximum number of boxes is 10.
You can download it yourself according to the link and modify the relevant parameters, or you can directly use our modified code, the relevant code is in the OB folder.
### Prepare Motion Saliency
First use [PWC-net](https://github.com/sniklaus/pytorch-pwc) to generate the optical flow, and then use the davis training set of VSOD to fine-tune the [CPD](https://github.com/wuzhe71/CPD) model
### Prepare Classifier
Please cd Classifier, Then run train_resnet18.py to train this model, run test_resnet18.py to produce prediction results.
### K-means
Please cd K-means, and run 2k-means.py 

## Data
Our saliency detection results can be downloaded on [BaiduCloud](https://pan.baidu.com/s/1SG-8N_bYD58goOZvC0kzaQ (code:cclz). 


Thanks to [CPD](https://github.com/wuzhe71/CPD), [PWC-net](https://github.com/sniklaus/pytorch-pwc) and [EfficientDet](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch)

