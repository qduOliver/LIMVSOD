import torch
import shutil

from sklearn.cluster import KMeans
# from sklearn.externals import joblib
import joblib
import pandas as pd
import cfg

import os
from PIL import Image
from data.transform import get_test_transform, get_yuan_transform
from tqdm import tqdm
import torch.nn as nn
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# from models import resnet50
# from torchvision.models import resnet50
from models.efficientnet_pytorch import EfficientNet
import cv2
from sklearn.metrics.pairwise import euclidean_distances

# FBMS-59/FBMS-Test
# Segtrack-v2
# Visal
# davis_test
# Easy-35
# VOS_test_png_gt

model = EfficientNet.from_pretrained('efficientnet-b7')
for parameter in model.parameters():
    parameter.requires_grad = False
model.eval()

model.cuda()

# dataset_list =['Easy-35','VOS_test_png']
# davis_test
dataset_list = ['VOS_test_png']
for a in range(0, len(dataset_list)):

    dataset = dataset_list[a]
    # G:\spyder_workpeace_song\total\Yet-Another-EfficientDet-Pytorch-master\txt2patch4
    # path = r'D:\code\new_6\2_test_1\%s\adp_th_train_2_max_meanx2_nofix_noaug_adp\test_2_all_other_ms_mean\train\%s/'%(dataset,dataset)
    # path = r'G:\source\MyResult12\julei_8\julei_prop/%s'%dataset
    path = r'F:\source\Visal3/2julei_prop/%s' % dataset
    videos = os.listdir(path)
    for i in range(0, len(videos)):

        video = videos[i]
        print(video)
        total_path = path + '/%s/' % (video)
        if not os.path.exists(total_path):
            continue
        imgs = os.listdir(total_path)
        X_encoded = np.empty((len(imgs), 2560), dtype='float32')
        img_arr = np.empty((len(imgs), 300, 300, 3), dtype='float32')
        # img_arr = []
        img_arr_name = []
        img_shape_list = []
        for i in range(0, len(imgs)):
            img_name = imgs[i]
            img_path = total_path + '/' + img_name
            img = Image.open(img_path).convert('RGB')
            img_shape = img.size
            # print(type(img))
            img_shape_list.append(img_shape)
            img_input = get_test_transform(size=cfg.INPUT_SIZE)(img).unsqueeze(0)
            img_show = get_yuan_transform(size=cfg.INPUT_SIZE)(img).unsqueeze(0)
            img_arr[i, :] = np.transpose(img_show, [0, 2, 3, 1])
            # img_arr.append(img)
            # img_arr=np.array(img_arr)
            img_arr_name.append(img_name)
            if torch.cuda.is_available():
                img_input = img_input.cuda()
            with torch.no_grad():
                out = model.extract_features(img_input)
            # print(out.size())
            out2 = nn.AdaptiveAvgPool2d(1)(out)
            feature = out2.view(out.size(1), -1).squeeze(1)
            X_encoded[i, :] = feature.cpu()

        # 6，10
        n_clusters = 8

        km = KMeans(n_clusters=n_clusters, random_state=0)
        km.fit(X_encoded)
        center = km.cluster_centers_  # [5,2560]
        print(center)
        # best_center = center[4]

        # print('distance',distance)
        for hh in range(0, n_clusters):
            # plt.figure()
            cluster = hh
            rows, cols = 100, 100
            start = 0
            # 上面path= r'D:\code\new_6\2_test_1\%s\adp_th_train_2_max_meanx2_nofix_noaug_adp\test_2_all_other_ms_mean\train\%s/'%(dataset,dataset)
            save_path = r'F:\source\Visal3/3julei\%s/%s/%s/' % (dataset, video, hh)
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            distance = euclidean_distances(center[hh, np.newaxis], X_encoded, squared=True)

            labels = np.where(km.labels_ == cluster)[0][start:start + rows * cols]
            for i, label in enumerate(labels):
                # label = int(label)
                # print(label)
                img_shape = img_shape_list[label]
                img_name = img_arr_name[label]
                img = cv2.cvtColor(img_arr[label], cv2.COLOR_RGB2BGR)
                distance_yy = distance[0, label]
                print('distance_yy', distance_yy)
                # save_img_name =  '%.3ffir_%s'%(distance_yy,img_name)
                save_img_name = '%s_fir%.3f.jpg' % (img_name, distance_yy)
                # save_img_name = '%s' % (img_name)
                img = cv2.resize(img, img_shape)
                cv2.imwrite(save_path + save_img_name, img * 255)

