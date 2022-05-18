import os
from skimage import data, filters
import numpy as np
import cv2
import matplotlib.pyplot as plt

dataset_list = [ 'Visal']
pt = ''
for a in range(0, len(dataset_list)):
    dataset = dataset_list[a]
    path_total = r'F:\source\Visal17\36Smeasure_result\11/%s/%s/' % (pt,dataset)

    videos = os.listdir(path_total)
    for i in range(0, len(videos)):
        video = videos[i]

        select_save_path_end_bi = r'F:\source\Visal17\36Smeasure_result\KFS_novalue/%s/%s/%s/' % (pt,dataset, video)
        select_save_path_value = r'F:\source\Visal17\36Smeasure_result\KFS_value/%s/%s/%s/' % (pt,dataset, video)

        if not os.path.exists(select_save_path_end_bi):
            os.makedirs(select_save_path_end_bi)

        if not os.path.exists(select_save_path_value):
            os.makedirs(select_save_path_value)

        imgs_path = path_total + video
        imgs = os.listdir(imgs_path)

        img_list = []
        img_name_list = []
        e_value_list_max = []
        # e_value_list_mean=[]

        for j in range(0, len(imgs)):
            img = imgs[j]

            # img_name = img.split('.')[0][:-4]
            img_name = img.split('=')[0]
            sm_max = img.split('=')[1][:-4]
            # img_sm_max = '0.%s' % (sm_max)
            if sm_max == 'nan':
                # print(img)
                # continue
                img_sm_max = float(0)

            img_sm_max = float(sm_max)

            img_tu = plt.imread(imgs_path + '/' + img)

            # print(img_name)
            # print(img_sm_max)
            # print(img_sm_mean)

            x = 5
            if dataset == 'VOS_test_png':
                x = 25
            # print('x', x)
            if len(e_value_list_max) == x:
                s_five_max = max(e_value_list_max)
                s_five_max_index = e_value_list_max.index(s_five_max)
                max_name = img_name_list[s_five_max_index]

                max_img = img_list[s_five_max_index]
                max_max_e_value = e_value_list_max[s_five_max_index]
                print('max_max_e_value', max_max_e_value)
                print('max_name', max_name)

                if max_max_e_value > 0.0:
                    thresh_fg = filters.threshold_otsu(max_img)  # 返回一个阈值,这是二值化的一个值
                    fg_ssav_ms_bi = (max_img >= thresh_fg) * 1.0  # 根据阈值
                    fg_ssav_ms_bi = fg_ssav_ms_bi.astype(np.float32)

                    cv2.imwrite(select_save_path_end_bi + '/' + max_name + '.png', fg_ssav_ms_bi * 255)
                    cv2.imwrite(select_save_path_value + '/' + max_name + '_max%.3f.png' % (max_max_e_value),
                                fg_ssav_ms_bi * 255)

                img_list = []
                img_name_list = []
                e_value_list_max = []
                e_value_list_mean = []

                img_list.append(img_tu)
                img_name_list.append(img_name)
                # e_value_list_mean.append(img_sm_mean)
                e_value_list_max.append(img_sm_max)
            else:

                img_list.append(img_tu)
                img_name_list.append(img_name)
                e_value_list_max.append(img_sm_max)
