
# zhege\yon用于把ed_test和ms_test的两次帅选剩下来的 放到一起，用于测试
# -*- coding: utf-8 -*-
import os
import cv2
import  shutil

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
# VOS_test_png_gt 这个是第二次筛选之前的计算正负样本的到类中心的距离

model = EfficientNet.from_pretrained('efficientnet-b7')
for parameter in model.parameters():
    parameter.requires_grad = False
model.eval()
model.cuda()

# dataset_list =['Easy-35','VOS_test_png']
# davis_test
# dataset_list = ['davis_test']
dataset_list = [ 'Visal']
for a in range(0, len(dataset_list)):

    dataset = dataset_list[a]
    # 1
    # path = r'D:\code\new_6\2_test_1\%s\adp_th_train_2_max_meanx2_nofix_noaug_adp\test_2_all_other_ms_mean\train\%s/'%(dataset,dataset)
    path = r'F:\source\Visal17\16Sec_msTrain/%s' % dataset
    videos = os.listdir(path)
    for i in range(0, len(videos)):

        video = videos[i]
        print(video)
        total_path = path + '/%s/' % (video)
        if not os.path.exists(total_path):
            continue
        cal_list = os.listdir(total_path)
        for jj in range(0,len(cal_list)):
            cla = cal_list[jj]
            cla_path = total_path + cla
            imgs = os.listdir(cla_path)
            X_encoded = np.empty((len(imgs), 2560), dtype='float32')
            img_arr = np.empty((len(imgs), 300, 300, 3), dtype='float32')
            # img_arr = []
            img_arr_name = []
            img_shape_list = []
            for i in range(0, len(imgs)):
                img_name = imgs[i]
                img_path = cla_path + '/' + img_name
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
            n_clusters = 1

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
                save_path = r'F:\source\Visal17\17Sec_ed\%s/%s/%s/' % (dataset, video, cla)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                distance = euclidean_distances(center[hh, np.newaxis], X_encoded, squared=True)

                labels = np.where(km.labels_ == cluster)[0][start:start + rows * cols]
                for i, label in enumerate(labels):
                    # label = int(label)
                    # print(label)
                    img_shape = img_shape_list[label]
                    img_name = img_arr_name[label]
                    # '675.232_176_242_242_330_bbox_69.674_blackswan_00004.jpg_bird_0.012.jpg'
                    img = cv2.cvtColor(img_arr[label], cv2.COLOR_RGB2BGR)
                    distance_yy = distance[0, label]
                    print('distance_yy', distance_yy)
                    save_img_name = '%s_thi%.3f.jpg' % (img_name[:-4], distance_yy)
                    # save_img_name = '%s' % (img_name)
                    img = cv2.resize(img, img_shape)
                    cv2.imwrite(save_path + save_img_name, img * 255)














# # dataset = 'Segtrack-v2'
# dataset_list = ['davis_test']
# for aa in range(0, len(dataset_list)):
#     dataset = dataset_list[aa]
#     path0 = r'F:\source\Visal\julei0_1_train/%s/' % dataset
#     save_path = r'F:\source\Visal\julei0_1_secondTest/%s/' % dataset
#     videos0 = os.listdir(path0)
#
#     for i in range(0, len(videos0)):
#         video = videos0[i]
#         img0_path = path0 + video
#         imgs = os.listdir(img0_path)
#         for j in range(0, len(imgs)):
#             img = imgs[j]
#             if 'BB_0' in img:
#                 save_path0 = save_path + video + '/0/'
#                 if not os.path.exists(save_path0):
#                     os.makedirs(save_path0)
#
#                 shutil.copy(img0_path + '/' + img, save_path0)
#


        # cla_path = img0_path + '/' + cla
            # imgs = os.listdir(cla_path)
            # save_path0 = save_path + video + '/' + '%s/' % cla
            # if not os.path.exists(save_path0):
            #     os.makedirs(save_path0)
            # for j in range(0, len(imgs)):
            #     img = imgs[j]
            #     shutil.copy(img0_path + '/' + '%s/' % cla + img, save_path0)
