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
# 这个是用来计算patch的白色像素点个数占frame像素点个数的比重。如果比重太低，就降低该box在初始正负样本时的数值，让它作用降低。
dataset_list = ['Segtrack-v2', 'Visal']

for a in range(0, len(dataset_list)):
    dataset = dataset_list[a]
    dataset_path = r'G:\spyder_workpeace_song\total\Yet-Another-EfficientDet-Pytorch-master\txt2patch6_yolov4/%s/' % (dataset)

    folders = os.listdir(dataset_path)


    for cc in range(0, len(folders)):
        folder = folders[cc]
        # ms_total_path = r'G:\source\MyResult\spy_flo2rgb2ms/%s/' % dataset
        ms_total_path = r'F:\source\CPD_pwc_11yearresult/%s/' % dataset
        gt_total_path = r'D:\source\VSOD_dataset/%s/' % (dataset)
        rgb_total_path = r'D:\source\VSOD_dataset/%s/' % (dataset)
        folder_path = dataset_path + folder
        imgs = os.listdir(folder_path)
        cla_ave_list = []
        for aaa in range(0, len(imgs)):
            img = imgs[aaa]
            cla_path = folder_path + '/' + img
            save_path = r'F:\source\Visal17/2julei_prop/%s/%s/' % (dataset, folder)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            w = img.split('_')
            if folder == 'bird_of_paradise':
                # 0.000_bird_of_paradise_00017.jpg_556_25_638_201.jpg

                png_name = w[0] + '_' + w[1] + '_' + w[2] + '_' + w[3][:-4] + '.png'
                x1 = int(w[7])
                y1 = int(w[8])
                x2 = int(w[9])
                y2 = int(w[10][:-4])
                # obj = w[4]
                # score = w[5]
                # cla_name = int(cla)
            elif folder == 'snow_leopards':
                png_name = w[0] + '_' + w[1] + '_' + w[2][:-4] + '.png'
                # 0.000_snow_leopards_00006.jpg_337_281_364_287.jpg
                x1 = int(w[6])
                y1 = int(w[7])
                x2 = int(w[8])
                y2 = int(w[9][:-4])

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















        #     for j in range(0, class_num):
        #         # 这个for循环就是用来计算 聚类的motion value的
        #
        #         if cla_name == j:
        #             if not os.path.exists(save_path + '%d' % (j)):
        #                 os.makedirs(save_path + '%d' % (j))
        #             aa = save_path + '%d' % (j) + '/' + '%s_%s_%s_%s_%s_bbox_%s_mv%.3f.jpg' % (
        #                 img_name, x1, y1, x2, y2, patch_name, motion_value)
        #             cv2.imwrite(aa, cropped_rgb)
        #             # 上一层遍历每一个patch，
        #             # 这个for循环就是看每一个patch的上面的数据命名，然后存储相应的rgb patch。前面的if语句就是判断该rgb patch
        #             # 属于哪个聚类的，最后挨个把patch的motion value 加起来得到类的motion value
        #             #  f.write('%s,%s,%s,%s,%s,%s,%s,%s'%(img_name,x1,y1,x2,y2,obj,score,cla)+'\n')
        #             # 'cla_{}_sum'.format(j)='cla_{}_sum'.format(j)+motion_v
        #             # 'cla_{}_num'.format(j)='cla_{}_num'.format(j)+1
        #             # print('********************' + str(motion_value))
        #             exec('cla_{}_sum = cla_{}_sum + {}'.format(j, j, motion_value))
        #             exec('cla_{}_num = cla_{}_num + {}'.format(j, j, 1))
        #
        #             # print(‘cla_0_num’，cla_0_num)
        #             # 'cla_{}_sum = cla_{}_sum + {}'.format(j,j,motion_v)
        #
        # class_0_ave = cla_0_sum / cla_0_num
        # new_name = '0_%.3f' % (class_0_ave)
        # os.rename(save_path + '0', save_path + new_name)
        # cla_ave_list.append(class_0_ave)
        # print('class_0_ave', class_0_ave)
        #
        # class_1_ave = cla_1_sum / cla_1_num
        # new_name = '1_%.3f' % (class_1_ave)
        # os.rename(save_path + '1', save_path + new_name)
        # cla_ave_list.append(class_1_ave)
        # print('class_1_ave', class_1_ave)
        #
        # class_2_ave = cla_2_sum / cla_2_num
        # new_name = '2_%.3f' % (class_2_ave)
        # os.rename(save_path + '2', save_path + new_name)
        # cla_ave_list.append(class_2_ave)
        # print('class_2_ave', class_2_ave)
        #
        # class_3_ave = cla_3_sum / cla_3_num
        # new_name = '3_%.3f' % (class_3_ave)
        # os.rename(save_path + '3', save_path + new_name)
        # cla_ave_list.append(class_3_ave)
        # print('class_3_ave', class_3_ave)
        #
        # class_4_ave = cla_4_sum / cla_4_num
        # new_name = '4_%.3f' % (class_4_ave)
        # os.rename(save_path + '4', save_path + new_name)
        # cla_ave_list.append(class_4_ave)
        # print('class_4_ave', class_4_ave)
        #
        # class_5_ave = cla_5_sum / cla_5_num
        # new_name = '5_%.3f' % (class_5_ave)
        # os.rename(save_path + '5', save_path + new_name)
        # cla_ave_list.append(class_5_ave)
        # print('class_5_ave', class_5_ave)
        #
        # class_6_ave = cla_6_sum / cla_6_num
        # new_name = '6_%.3f' % (class_6_ave)
        # os.rename(save_path + '6', save_path + new_name)
        # cla_ave_list.append(class_6_ave)
        # print('class_6_ave', class_6_ave)
        #
        # class_7_ave = cla_7_sum / cla_7_num
        # new_name = '7_%.3f' % (class_7_ave)
        # os.rename(save_path + '7', save_path + new_name)
        # cla_ave_list.append(class_7_ave)
        # print('class_7_ave', class_7_ave)
        #
        # x = np.array(cla_ave_list)
        # print(cla_ave_list)
        # x_new = np.argsort(-x)  # 按升序排列
        # print('x_new', x_new)



