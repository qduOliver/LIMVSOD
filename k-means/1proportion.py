import math
import os
import cv2
import numpy as np

# FBMS-59/FBMS-Test
# Segtrack-v2
# Visal
# davis_test
# Easy-35
# VOS_test_png_gt
# 这个是用来计算patch的白色像素点个数占frame像素点个数的比重。
dataset_list = ['VOS_test_png']

for a in range(0, len(dataset_list)):
    dataset = dataset_list[a]
    dataset_path = r'.\txt2patch4/%s/' % (dataset)

    folders = os.listdir(dataset_path)


    for cc in range(0, len(folders)):
        folder = folders[cc]
        # ms_total_path = r'G:\source\MyResult\S_ms/%s/' % dataset
        ms_total_path = r'F:\source\CPD_pwc_11yearresult/%s/' % dataset
        gt_total_path = r'D:\source\VSOD_dataset/%s/' % (dataset)
        rgb_total_path = r'D:\source\VSOD_dataset/%s/' % (dataset)
        folder_path = dataset_path + folder
        imgs = os.listdir(folder_path)
        cla_ave_list = []
        for aaa in range(0, len(imgs)):
            img = imgs[aaa]
            cla_path = folder_path + '/' + img
            save_path = r'F:\source\Visal3/2julei_prop/%s/%s/' % (dataset, folder)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            w = img.split('_')
            if folder == 'bird_of_paradise':
                # 0.000_bird_of_paradise_00017.jpg_556_25_638_201.jpg

                png_name = w[1] + '_' + w[2] + '_' + w[3] + '_' + w[4][:-4] + '.png'
                x1 = int(w[5])
                y1 = int(w[6])
                x2 = int(w[7])
                y2 = int(w[8][:-4])
                # obj = w[4]
                # score = w[5]
                # cla_name = int(cla)
            elif folder == 'snow_leopards':
                png_name = w[1] + '_' + w[2] + '_' + w[3][:-4] + '.png'
                # 0.000_snow_leopards_00006.jpg_337_281_364_287.jpg
                x1 = int(w[4])
                y1 = int(w[5])
                x2 = int(w[6])
                y2 = int(w[7][:-4])

            elif dataset == 'Easy-35':
                png_name = w[1]
                # img_name : ''aeroplane_00034.png'
                x1 = int(w[2])
                y1 = int(w[3])
                x2 = int(w[4])
                y2 = int(w[5][:-4])




            else:
                # 'birdfall_00002.jpg_bear_0.019_z_78_76_95_110.jpg_fir42.133.jpg'
                # patch_name = w[2] + '_' + w[3]


                png_name = w[0] + '_' + w[1][:-4] + '.png'
                # img_name : ''aeroplane_00034.png'
                x1 = int(w[5])
                y1 = int(w[6])
                x2 = int(w[7])
                y2 = int(w[8][:-4])
                # obj = w[2]
                # score = w[3]
                # cla_name = int(cla)


            ms_path = ms_total_path + folder + '/' + png_name
            if dataset == 'Easy-35':
                rgb_path = rgb_total_path + folder + '/Imgs/' + png_name
            elif dataset == 'VOS_test_png':
                rgb_path = rgb_total_path + folder + '/Imgs/' + png_name
            else:
                rgb_path = rgb_total_path + folder + '/Imgs/' + png_name[:-4] +'.jpg'

            if not os.path.exists(ms_path):
                print('%s not exist!!!!'%ms_path)
                continue
            ms = cv2.imread(ms_path)
            rgb = cv2.imread(rgb_path)
            w = rgb.shape[1]
            h = rgb.shape[0]

            ms = cv2.resize(ms, (w, h), interpolation=cv2.INTER_LINEAR)

            cropped_ms = ms[y1:y2, x1:x2]
            # cropped就是 ndarray：(317,162,3)，其实y2-y1 = 317，x2 - x1 = 162
            cropped_rgb = rgb[y1:y2, x1:x2]

            bbox_w = x2 - x1
            bbox_h = y2 - y1
            s = bbox_w * bbox_h

            patch_com = cropped_ms.sum()
            frame_com = ms.sum()
            # com就是patch中所有点的和

            if frame_com == 0:
                print('%sframe sum is 0'%ms)
                continue
            proportion= patch_com/frame_com
            # aa = save_path + '/' + '%s_pro%.3f.jpg' % (img,proportion)
            aa = save_path + '/' + '%s_%s_%s_%s_%s_pro%.3f.jpg' % (png_name, x1, y1, x2, y2, proportion)
            cv2.imwrite(aa, cropped_rgb)















        
