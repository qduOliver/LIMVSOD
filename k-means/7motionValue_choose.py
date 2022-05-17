



# -*- coding: utf-8 -*-
import os
import numpy as np
import shutil
# 这个是根据ms的值，进行第一次正负样本的帅选 吱吱吱吱吱吱吱吱吱吱吱吱吱吱吱吱吱吱吱
# Easy-35
# dataset = 'Segtrack-v2'
# total_apth = r'D:\code\new_6\1_ms_rank_class\fix_thre\add_gt\%s/'%(dataset)
dataset_list = ['VOS_test_png']
for sj in range(0, len(dataset_list)):
    dataset = dataset_list[sj]
    total_apth = r'F:\source\Visal3\6julei+-_to_01/%s/' % (dataset)
    #              F:\SJ\new\5_ms_selcect\ms_mean_01\fix
    # total_apth = r'D:\code\new_6\2_test_1\%s\adp_th_train_2_max_meanx2_nofix_noaug_adp\test_2_all_other_ms_mean\train_ed\%s/'%(dataset,dataset)
    video_list = os.listdir(total_apth)
    # swan级别

    for i in range(0, len(video_list)):

        vidoe_name = video_list[i]
        clas_path = total_apth + vidoe_name
        clas_list = os.listdir(clas_path)
        # clas_list 就是0 1 样本的地址了

        for j in range(0, len(clas_list)):
            cla_name = clas_list[j]
            imgs_path = clas_path + '/' + cla_name
            imgs = os.listdir(imgs_path)
            # imgs就是具体到正负样本的所有图片了

            msv_list = []
            for k in range(0, len(imgs)):
                img_name = imgs[k]
                # img_name就是具体某个聚类的某个图片了
                # 192.577_424_99_626_473_bbox_blackswan_00023.jpg_bird_0.406，jpg
                msv = float(img_name.split('mv')[1][:-4])
                # msv就是motion salientcy value，但是这个msv_list怎么是对所有聚类的图片进行motion saliency进行排序。
                # print('msv', msv)
                msv_list.append(msv)

            ave = np.mean(np.array(msv_list))
            # 这个np。mean（）是求数组msv_list的平均值的，也就是求聚类的motion salientcy 的平均值ave
            print('##############  ave  #################', vidoe_name, cla_name, ave)

            save_path_train = r'F:\source\Visal3\7julei0_1ms_train/%s/%s/%s/' % (
            dataset, vidoe_name, cla_name)
            save_path_test = r'F:\source\Visal3\7julei0_1ms_test/%s/%s/%s/' % (dataset, vidoe_name, cla_name)
            if not os.path.exists(save_path_train):
                os.makedirs(save_path_train)
            if not os.path.exists(save_path_test):
                os.makedirs(save_path_test)

            for k in range(0, len(imgs)):
                img_name = imgs[k]
                msv = float(img_name.split('mv')[1][:-4])

                if cla_name == '0':
                    if msv <= 1*ave:
                        shutil.copy(imgs_path + '/' + img_name, save_path_train + '/' + img_name)
                    else:
                        shutil.copy(imgs_path + '/' + img_name, save_path_test + '/' + img_name)

                if cla_name == '1':

                    if msv <=1*ave:
                        shutil.copy(imgs_path + '/' + img_name, save_path_test + '/' + img_name)
                    else:
                        shutil.copy(imgs_path + '/' + img_name, save_path_train + '/' + img_name)
            #




