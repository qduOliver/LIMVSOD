
# zhege\yon用于把ed_test和ms_test的两次帅选剩下来的 放到一起，用于测试
# -*- coding: utf-8 -*-
import os
import cv2
import  shutil

# dataset = 'Segtrack-v2'
dataset_list = [ 'Visal']
for aa in range(0, len(dataset_list)):
    dataset = dataset_list[aa]
    path0 = r'F:\source\Visal17\9julei0_1_ed_test/%s/' % dataset
    path1 = r'F:\source\Visal17\7julei0_1ms_test/%s/' % dataset
    save_path = r'F:\source\Visal17\12julei0_1_test/%s/' % dataset
    videos0 = os.listdir(path0)

    for i in range(0, len(videos0)):
        video = videos0[i]
        img0_path = path0 + video
        clas = os.listdir(img0_path)
        for j in range(0, len(clas)):
            cla = clas[j]
            cla_path = img0_path + '/' + cla
            imgs = os.listdir(cla_path)
            save_path0 = save_path + video + '/' + '%s/' % cla
            if not os.path.exists(save_path0):
                os.makedirs(save_path0)
            for j in range(0, len(imgs)):
                img = imgs[j]
                shutil.copy(img0_path + '/' + '%s/' % cla + img, save_path0)
    videos1 = os.listdir(path1)
    for jj in range(0, len(videos1)):
        video = videos1[jj]
        img1_path = path1 + video
        clas1 = os.listdir(img1_path)
        for kk in range(0, len(clas1)):
            cla1 = clas1[kk]
            cla1_path = img1_path + '/' + cla1
            imgs1 = os.listdir(cla1_path)

            save_path1 = save_path + video + '/' + '%s/' % cla1
            if not os.path.exists(save_path1):
                os.makedirs(save_path1)
            else:
                print(video, cla1)

            for j in range(0, len(imgs1)):
                img1 = imgs1[j]
                shutil.copy(img1_path + '/%s/' % cla1 + img1, save_path1)

