



# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
import shutil




# -*- coding: utf-8 -*-
import os
import numpy as np
import shutil
# 这个是根据ms的值，进行第er次正负样本的帅选 吱吱吱吱吱吱吱吱吱吱吱吱吱吱吱吱吱吱吱
# Easy-35
dataset_list = ['Visal']
for aa in range(0, len(dataset_list)):
    dataset = dataset_list[aa]
# total_apth = r'D:\code\new_6\1_ms_rank_class\fix_thre\add_gt\%s/'%(dataset)
    total_apth = r'F:\source\Visal17\15first_resultTo0_1/%s/' % (dataset)
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
                sec = '_sec'
                # bird_of_paradise_00000.jpg_227_65_512_323_pro0.086_mv0.515_t3_1.jpg
                # 0.000_blackswan_00039.jpg_0_330_123_358.jpgJF72.457.jpgmv0.000.jpg_sec116.823
                if sec in img_name:

                    msv = float(img_name.split('mv')[1].split('.jpg')[0])
                else:

                    msv = float(img_name.split('mv')[1].split('_T1')[0])

                msv_list.append(msv)
            print(len(msv_list))

            ave = np.mean(np.array(msv_list))
            # 这个np。mean（）是求数组msv_list的平均值的，也就是求聚类的motion salientcy 的平均值ave
            print('##############  ave  #################', vidoe_name, cla_name, ave)

            save_path_train = r'F:\source\Visal17\16Sec_msTrain/%s/%s/%s/' % (dataset, vidoe_name, cla_name)
            save_path_test = r'F:\source\Visal17\16Sec_msTest/%s/%s/%s/' % (dataset, vidoe_name, cla_name)
            if not os.path.exists(save_path_train):
                os.makedirs(save_path_train)
            if not os.path.exists(save_path_test):
                os.makedirs(save_path_test)

            for k in range(0, len(imgs)):
                img_name = imgs[k]
                sec = '_sec'
                # bird_of_paradise_00000.jpg_227_65_512_323_pro0.086_mv0.515_t3_1.jpg
                # bird_of_paradise_00002.jpg_227_66_347_320_pro0.683_mv405.306.jpg_sec142.452_t3_1.jpg
                if sec in img_name:

                    msv = float(img_name.split('mv')[1].split('.jpg')[0])
                else:

                    msv = float(img_name.split('mv')[1].split('_T1')[0])

                if cla_name == '0':
                    if msv <= ave:
                        shutil.copy(imgs_path + '/' + img_name, save_path_train + '/' + img_name)
                    else:
                        shutil.copy(imgs_path + '/' + img_name, save_path_test + '/' + img_name)

                if cla_name == '1':

                    if msv >= ave:
                        shutil.copy(imgs_path + '/' + img_name, save_path_train + '/' + img_name)
                    else:
                        shutil.copy(imgs_path + '/' + img_name, save_path_test + '/' + img_name)










# dataset_list = ['davis_test']
# shuzu = np.zeros(400)
# for sj in range(0, len(dataset_list)):
#     dataset = dataset_list[sj]
#     total_path = r'F:\source\Visal\add_num/%s/' % (dataset)
#
#     folders_list = os.listdir(total_path)
#
#
#
#
#     for j in range(0, len(folders_list)):
#         folder_name = folders_list[j]
#         imgs_path = total_path + '/' + folder_name
#         imgs = os.listdir(imgs_path)
#         imgsCount = len(imgs)
#         folderArr = np.zeros(imgsCount)
#
#         # imgList = os.listdir(r"./" + directory_name)
#         imgs.sort(key=lambda x: int(x.split('.')[0]))  # 按照数字顺序排列图片名
#         # print(imgs)  # 打印出来看看效果，不想看也可以删掉这句
#         if imgsCount < 400:
#
#             imgsArr = np.zeros(400)
#         else:
#             imgsArr = np.zeros(imgsCount)
#         c = 0
#
#         for k in range(0, len(imgs)):
#
#             index = len(imgs) - k - 1
#             imgName = imgs[index]
#             if ('AA' in imgName):
#                 imgsArr[k] = 1
#                 c = c + 1
#         tempArr = np.resize(imgsArr, (400))
#         shuzu = shuzu + tempArr
#         print(folder_name, c)
#
#     mat = shuzu.reshape(20, 20)
#
#     plt.matshow(mat, cmap=plt.cm.Blues)
#     plt.show()
#



