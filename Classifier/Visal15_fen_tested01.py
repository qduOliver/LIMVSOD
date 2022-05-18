



# -*- coding: utf-8 -*-
import os
import numpy as np
import shutil

# 为了把分类器分好的结果，分别存到0 1 文件夹中


# test_csv_julei60_1_test best_acc =
import os
import cv2
import  shutil
# FBMS-59/FBMS-Test
# Segtrack-v2
# Visal
# davis_test
# Easy-35
# VOS_test_png   VOS_test_png_gt
dataset_list = [ 'Visal']
for aa in range(0, len(dataset_list)):
    dataset = dataset_list[aa]

    path0 = r'F:\source\Visal17\14first_test_result/%s/' % (dataset)

    save_path = r'F:\source\Visal17\15first_resultTo0_1/%s/' % dataset
    videos0 = os.listdir(path0)
    for i in range(0, len(videos0)):
        video = videos0[i]
        img0_path = path0 + video
        imgs = os.listdir(img0_path)
        c = 0
        k = 0
        for j in range(0, len(imgs)):
            img = imgs[j]
            w = img.split('T1')[1][1]
            # print(w)
            if w == '0':
                c =c +1
                save_path0 = save_path + video + '/%s/' % '0'
                if not os.path.exists(save_path0):
                    os.makedirs(save_path0)
                shutil.copy(img0_path + '/' + img, save_path0)
            if w == '1':
                save_path1 = save_path + video + '/%s/' % '1'
                if not os.path.exists(save_path1):
                    os.makedirs(save_path1)
                shutil.copy(img0_path + '/' + img, save_path1)

            if (w != '0') & (w != '1'):
                print(img)
        print('c:',c)



# dataset_list = ['davis_test']
# for sj in range(0, len(dataset_list)):
#     dataset = dataset_list[sj]
#     total_path = r'F:\source\Visal\julei0_1_train/%s/' % (dataset)
#     main_path = r'F:\source\Visal\add_num/%s/'%dataset
#
#     folders_list = os.listdir(total_path)
#
#     # clas_list 就是0 1 样本的地址了
#
#     for j in range(0, len(folders_list)):
#         folder_name = folders_list[j]
#         imgs_path = total_path + '/' + folder_name
#         imgs = os.listdir(imgs_path)
#         C = 0
#         for k in range(0, len(imgs)):
#             img_name = imgs[k]
#             if 'BB_1'in img_name:
#                 main_name = img_name.split('JL')[0]
#                 main_name_path = main_path + folder_name + '/' + main_name
#                 if os.path.exists(main_name_path):
#                     new_name = main_name + 'BB.jpg'
#                     new_path = main_path + folder_name + '/' + new_name
#                     C = C + 1
#                     os.renames(main_name_path, new_path)
#         print('%s,%s' % (folder_name, C))










