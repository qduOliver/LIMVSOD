import os
from PIL import Image
import matplotlib.pyplot  as plt
import cv2
import numpy as np
# 0.000_283_170_317_187_bbox_0516.png_cake_0.106_gt_1_t2_1。png
# zhege 应该是patch_total_path里的patch命名形式
# 0_245_24_267_bbox_blackswan_00024
# 这个是我们的
dataset_list = [ 'Visal']
# dataset_list = ['VOS']
pt = ''
for sj in range(0, len(dataset_list)):

    dataset = dataset_list[sj]
    # patch_total_path = r'G:\source\MyResult11\julei_8S\test_all_patchTo01\14pt/%s/' % (dataset)
    patch_total_path = r'F:\source\Visal17\35all_patch_saliency/%s/%s/' % (pt,dataset)
    gt_path = r'D:\source\VSOD_dataset/%s/' % (dataset)

    # if dataset == 'VOS_test_png':
    #     dataset_gt = 'VOS_test_png'
    #     gt_path = r'D:\dataset\2020_result\video_u2net_results/%s/' % (dataset_gt)
    videos = os.listdir(patch_total_path)
    for i in range(0, len(videos)):
        video = videos[i]
        imgs_path = patch_total_path + video + '/'
        gts_path = gt_path + video + '/Imgs/'
        imgs = os.listdir(imgs_path)

        for j in range(0, len(imgs)):
            img_name = imgs[j]

            if dataset == 'Easy-35':
                # 0390.png_54_38_560_353_pro0.996_mv242.693.jpg_sec67.554
                w2 = img_name.split('.jpg')


                rgb_name = w2[0].split('_')[0]
                png_name = rgb_name[:-4] + '.png'
            elif dataset =='VOS':
                # 3_00000.png_193_7_618_432_pro0.999_mv246.099.jpg_sec24.714
                w2 = img_name.split('.png')

                # rgb_name11 = w2[0].split('_')[1:]
                rgb_name11 = w2[0]
                rgb_name = rgb_name11 + '.png'
                png_name = rgb_name11 + '.png'
            else:
                #'aeroplane_00001.jpg_15_74_282_172_pro0.956_mv164.504.jpg_sec1.544.png'
                w2 = img_name.split('.jpg')

                # rgb_name11 = w2[0].split('_')[1:]
                rgb_name11 = w2[0]
                rgb_name = rgb_name11 + '.jpg'
                png_name = rgb_name[:-4] + '.png'

            if os.path.exists(gts_path + rgb_name):

                gt = Image.open(gts_path + rgb_name)
                w = gt.size[0]
                h = gt.size[1]
                save_path = r'F:\source\Visal17\35all_patch_paste\%s\%s/%s/' % (pt,dataset, video)
                mask = Image.new('RGB', (w, h), 'black')

                patch = Image.open(imgs_path + '/' + img_name)

                if dataset == 'Easy-35':
                    ww = w2[0].split('_')

                    ## 'blackswan_00000.jpg_139_72_510_388_pro0.983.jpg_fir0.830.jpgmv255.268.jpg_sec3.077.png'
                    x1 = int(float(ww[4]))
                    y1 = int(float(ww[5]))
                    x2 = int(float(ww[6]))
                    y2 = int(float(ww[7]))
                else:
                    # 110_00089.png_bird_0.189_z_385_193_436_312_t5_1.png
                    ww = w2[1].split('_')
                    x1 = int(float(ww[4]))
                    y1 = int(float(ww[5]))
                    x2 = int(float(ww[6]))
                    y2 = int(float(ww[7]))



                mask.paste(patch, (x1, y1))

                if os.path.exists(save_path + png_name):
                    mask_1 = Image.open(save_path + png_name)
                    mask = np.array(mask)
                    mask_1 = np.array(mask_1)
                    mask = cv2.add(mask, mask_1)
                    # mask = (mask_1 + mask)/2
                    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
                    mask = Image.fromarray(np.uint8(mask * 255))

                if not os.path.exists(save_path):
                    os.makedirs(save_path)


                mask.save(save_path + png_name)
