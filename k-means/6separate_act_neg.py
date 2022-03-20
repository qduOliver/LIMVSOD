# -*- coding: utf-8 -*-
import os
import cv2
import  shutil
# 这个是把正负样本，放到 0 和 1 文件夹中
# Segtrack-v2

dataset_list = ['VOS_test_png']
for sj in range(0, len(dataset_list)):

         dataset = dataset_list[sj]
         path0 = r'F:\source\Visal3\5julei_ms-\sal/%s/' % dataset
         path1 = r'F:\source\Visal3\5julei_ms+\sal/%s/' % dataset
         save_path = r'F:\source\Visal3\6julei+-_to_01/%s/' % dataset
         videos0 = os.listdir(path0)
         # diyige大的for是把负样本放到 0 文件中
         for i in range(0, len(videos0)):
             video = videos0[i]
             img0_path = path0 + video
             imgs = os.listdir(img0_path)
             save_path0 = save_path + video + '/' + '0'
             if not os.path.exists(save_path0):
                 os.makedirs(save_path0)

             for j in range(0, len(imgs)):
                 img = imgs[j]
                 shutil.copy(img0_path + '/' + img, save_path0)
         videos1 = os.listdir(path1)
         for j in range(0, len(videos1)):
             video = videos1[j]
             img1_path = path1 + video
             imgs1 = os.listdir(img1_path)
             save_path1 = save_path + video + '/' + '1'
             if not os.path.exists(save_path1):
                 os.makedirs(save_path1)

             for j in range(0, len(imgs1)):
                 img = imgs1[j]
                 shutil.copy(img1_path + '/' + img, save_path1)

