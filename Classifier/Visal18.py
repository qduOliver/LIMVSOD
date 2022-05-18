import os
import csv




# -*- coding: utf-8 -*-
import os
import numpy as np
import shutil
# 这个是根据dao类中心的的值，进行第er次正负样本的帅选 吱吱吱吱吱吱吱吱吱吱吱吱吱吱吱吱吱吱吱
# Easy-35
dataset_list = [ 'Visal']
for a in range(0, len(dataset_list)):

    dataset = dataset_list[a]
    # dataset = 'davis_test'
    # total_apth = r'D:\code\new_6\1_ms_rank_class\fix_thre\add_gt\%s/'%(dataset)
    total_apth = r'F:\source\Visal17\17Sec_ed/%s/' % (dataset)
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
                msv = float(img_name.split('_thi')[1][:-4])
                # msv就是motion salientcy value，但是这个msv_list怎么是对所有聚类的图片进行motion saliency进行排序。
                # print('msv', msv)
                msv_list.append(msv)

            ave = np.mean(np.array(msv_list))
            # 这个np。mean（）是求数组msv_list的平均值的，也就是求聚类的motion salientcy 的平均值ave
            print('##############  ave  #################', vidoe_name, cla_name, ave)

            save_path_train = r'F:\source\Visal17/18Sec_edTrain/%s/%s/%s/' % (dataset, vidoe_name, cla_name)
            save_path_test = r'F:\source\Visal17\18Sec_edTest/%s/%s/%s/' % (dataset, vidoe_name, cla_name)
            if not os.path.exists(save_path_train):
                os.makedirs(save_path_train)
            if not os.path.exists(save_path_test):
                os.makedirs(save_path_test)
            cc = 0
            for k in range(0, len(imgs)):
                img_name = imgs[k]
                msv = float(img_name.split('_thi')[1][:-4])

                if cla_name == '0':
                    if msv <= 1* ave:
                        #默认是1

                        shutil.copy(imgs_path + '/' + img_name, save_path_train + '/' + img_name)
                    else:
                        shutil.copy(imgs_path + '/' + img_name, save_path_test + '/' + img_name)

                if cla_name == '1':

                    if msv <= 1* ave:
                        # 默认是1

                        shutil.copy(imgs_path + '/' + img_name, save_path_train + '/' + img_name)
                    else:
                        shutil.copy(imgs_path + '/' + img_name, save_path_test + '/' + img_name)
            #










# dataset_list = ['davis_test']
# for aa in range(0, len(dataset_list)):
#     dataset = dataset_list[aa]
#     # PATH=r'G:\spyder_workpeace_song\total\Yet-Another-EfficientDet-Pytorch-master\txt2patch4/%s/'%(dataset)
#     PATH = r'F:\source\Visal\julei0_1_secondTest/%s/' % (dataset)
#
#     videos = os.listdir(PATH)
#     for i in range(0, len(videos)):
#         video = videos[i]
#         cla_path = PATH + video
#         clas = os.listdir(cla_path)
#         num = 0
#         csv_path = r'F:\source\Visal/test_csv_julei8_Second/%s/' % (dataset)
#         # csv_path = './csv_patch_test_adp/%s/test_all/' % (dataset)
#
#         if not os.path.exists(csv_path):
#             os.makedirs(csv_path)
#
#         with open(csv_path + '/%s.csv' % (video), 'a', newline="", encoding='utf-8-sig') as csvfile:
#             writer = csv.writer(csvfile)
#             #        string = ''',path,classes'''
#             #        string=string.strip('''''')
#             # string=eval(string)
#             path = 'path'
#             classes = 'classes'
#             writer.writerow([path, classes])
#
#             for cl in range(0, len(clas)):
#                 cla = clas[cl]
#                 img_path = cla_path + '/' + cla
#                 imgs = os.listdir(img_path)
#
#                 for j in range(0, len(imgs)):
#                     img = imgs[j]
#                     img_name = img
#                     print(img_name)
#                     #        img_label_reg=img[-9:-4]
#                     #        nan=img[-7:-4]
#                     img_label_cla = cla[0]
#
#                     #        img_label_reg=0.000
#                     #        img_label_cla=img[-5:-4]
#                     print(img_label_cla)
#
#                     flo_path = img_path + '/%s' % (img_name)
#
#                     writer.writerow([num, flo_path, img_label_cla])
#                     num = num + 1