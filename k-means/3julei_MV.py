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
dataset_list = ['VOS_test_png']
for a in range(0, len(dataset_list)):
    dataset = dataset_list[a]
    dataset_path = r'F:\source\Visal3\3julei/%s/' % (dataset)

    # txt_path = blackswan.txt 等等
    folders = os.listdir(dataset_path)

    # folders:['blackswan.txt', 'bmx-trees.txt', 'breakdance.txt', 'camel.txt', 'car-roundabout.txt', 'car-shadow.txt', 'cows.txt', 'dance-twirl.txt', 'dog.txt', 'drift-chicane.txt', 'drift-straight.txt', 'goat.txt', 'horsejump-high.txt', 'kite-surf.txt', 'libby.txt', 'motocross-jump.txt', 'paragliding-launch.txt', 'parkour.txt', 'scooter-black.txt', 'soapbox.txt']
    class_num = 8

    for cc in range(0, len(folders)):
        folder = folders[cc]
        # folder：'blackswan'
        print('folder', folder)
        # single_txt_path= txt_path + folder + '.txt'
        # single_txt_path:'F:\\SJ\\new\\4_k-means\\top13_txt_8_new_2/davis_test/blackswan.txt'
        # ms_total_path = r'F:\source\CPD_premodelvgg49_PWCresult/%s/' % dataset
        ms_total_path = r'F:\source\CPD_pwc_11yearresult/%s/' % dataset
        # ms_total_path是motion saliency
        gt_total_path = r'D:\source\VSOD_dataset/%s/' % (dataset)
        # gt_total_path = r'D:/dataset/2020_result/video_u2net_results/Visal/'
        # rgb_total_path = r'D:\dataset\video/%s/VOS_test_png_gt/' % (dataset)
        rgb_total_path = r'D:\source\VSOD_dataset/%s/' % (dataset)
        single_txt_path = dataset_path + folder

        clas = os.listdir(single_txt_path)

        # ['blackswan_00000.jpg_bird_0.647.jpg,409,70,571,387,bird,0.647, 0\n', 'blackswan_00006.jpg_giraffe_0.317.jpg,429,63,574,476,giraffe,0.317, 0\n', 'blackswan_00007.jpg_bird_0.293.jpg,427,64,576,474,bird,0.293, 0\n', 'blackswan_00010.jpg_giraffe_0.312.jpg,436,65,580,469,giraffe,0.312, 0\n', 'blackswan_00011.jpg_bird_0.264.jpg,414,68,577,473,bird,0.264, 0\n', 'blackswan_00013.jpg_bird_0.271.jpg,440,69,560,465,bird,0.271, 0\n', 'blackswan_00014.jpg_bird_0.401.jpg,448,81,558,372,bird,0.401, 0\n', 'blackswan_00015.jpg_bird_0.246.jpg,445,81,576,419,bird,0.246, 0\n', 'blackswan_00017.jpg_bird_0.342.jpg,427,92,568,469,bird,0.342, 0\n', 'blackswan_00018.jpg_bird_0.333.jpg,443,98,574,383,bird,0.333, 0\n', 'blackswan_00019.jpg_bird_0.418.jpg,423,100,564,451,bird,0.418, 0\n', 'blackswan_00020.jpg_bird_0.382.jpg,432,102,542,436,bird,0.382, 0\n', 'blackswan_00021.jpg_bird_0.298.jpg,389,102,570,474,bird,0.298, 0\n', 'blackswan_00022.jpg_bird_0.497.jpg,402,102,620,474,bird,0.497, 0\n', 'blackswan_00023.jpg_bird_0.406.jpg,424,99,626,473,bird,0.406, 0\n', 'blackswan_00026.jpg_bird_0.194.jpg,430,104,526,357,bird,0.194, 0\n', 'blackswan_00027.jpg_bird_0.250.jpg,409,100,544,458,bird,0.250, 0\n', 'blackswan_00029.jpg_bird_0.322.jpg,425,98,518,346,bird,0.322, 0\n', 'blackswan_00030.jpg_bird_0.531.jpg,385,97,543,474,bird,0.531, 0\n', 'blackswan_00031.jpg_bird_0.505.jpg,360,96,539,475,bird,0.505, 0\n', 'blackswan_00032.jpg_bird_0.316.jpg,399,90,539,467,bird,0.316, 0\n', 'blackswan_00033.jpg_bird_0.351.jpg,424,96,517,340,bird,0.351, 0\n', 'blackswan_00034.jpg_bird_0.449.jpg,413,97,579,464,bird,0.449, 0\n', 'blackswan_00035.jpg_bird_0.420.jpg,417,96,546,449,bird,0.420, 0\n', 'blackswan_00036.jpg_bird_0.369.jpg,425,98,517,337,bird,0.369, 0\n', 'blackswan_00037.jpg_bird_0.350.jpg,420,97,553,435,bird,0.350, 0\n', 'blackswan_00038.jpg_bird_0.696.jpg,424,101,585,408,bird,0.696, 0\n', 'blackswan_00040.jpg_bird_0.558.jpg,424,103,559,414,bird,0.558, 0\n', 'blackswan_00041.jpg_bird_0.410.jpg,417,96,560,459,bird,0.410, 0\n', 'blackswan_00042.jpg_bird_0.397.jpg,408,99,559,477,bird,0.397, 0\n', 'blackswan_00043.jpg_bird_0.309.jpg,382,93,568,469,bird,0.309, 0\n', 'blackswan_00044.jpg_bird_0.336.jpg,333,88,621,469,bird,0.336, 0\n', 'blackswan_00045.jpg_bird_0.378.jpg,385,91,569,466,bird,0.378, 0\n', 'blackswan_00046.jpg_bird_0.198.jpg,401,88,563,454,bird,0.198, 0\n', 'blackswan_00047.jpg_bird_0.337.jpg,430,91,521,344,bird,0.337, 0\n', 'blackswan_00048.jpg_bird_0.302.jpg,428,86,518,331,bird,0.302, 0\n', 'blackswan_00000.jpg_bird_0.023.jpg,656,147,673,159,bird,0.023, 1\n', 'blackswan_00000.jpg_bird_0.036.jpg,673,147,684,163,bird,0.036, 1\n', 'blackswan_00000.jpg_bird_0.069.jpg,637,153,661,163,bird,0.069, 1\n', 'blackswan_00001.jpg_bird_0.016.jpg,198,193,214,204,bird,0.016, 1\n', 'blackswan_00001.jpg_bird_0.016.jpg,674,141,691,157,bird,0.016, 1\n', 'blackswan_00001.jpg_bird_0.029.jpg,631,153,643,161,bird,0.029, 1\n', 'blackswan_00002.jpg_bird_0.025.jpg,624,153,635,160,bird,0.025, 1\n', 'blackswan_00003.jpg_bird_0.017.jpg,180,192,200,204,bird,0.017, 1\n', 'blackswan_00003.jpg_bird_0.017.jpg,618,152,630,160,bird,0.017, 1\n', 'blackswan_00004.jpg_bird_0.011.jpg,610,152,622,160,bird,0.011, 1\n', 'blackswan_00005.jpg_bird_0.012.jpg,605,153,617,160,bird,0.012, 1\n', 'blackswan_00005.jpg_bird_0.012.jpg,146,243,173,254,bird,0.012, 1\n', 'blackswan_00006.jpg_bird_0.012.jpg,600,152,612,161,bird,0.012, 1\n', 'blackswan_00007.jpg_bird_0.014.jpg,593,153,604,161,bird,0.014, 1\n', 'blackswan_00009.jpg_bird_0.030.jpg,579,155,590,162,bird,0.030, 1\n', 'blackswan_00014.jpg_bird_0.011.jpg,574,162,591,181,bird,0.011, 1\n', 'blackswan_00042.jpg_bird_0.015.jpg,113,249,133,264,bird,0.015, 1\n', 'blackswan_00000.jpg_bird_0.025.jpg,578,165,586,176,bird,0.025, 2\n', 'blackswan_00000.jpg_bird_0.058.jpg,683,142,699,160,bird,0.058, 2\n', 'blackswan_00000.jpg_bird_0.058.jpg,30,62,741,469,bird,0.058, 2\n', 'blackswan_00001.jpg_person_0.038.jpg,675,138,690,158,person,0.038, 2\n', 'blackswan_00002.jpg_bird_0.021.jpg,667,141,684,158,bird,0.021, 2\n', 'blackswan_00003.jpg_person_0.027.jpg,662,138,677,158,person,0.027, 2\n', 'blackswan_00004.jpg_person_0.025.jpg,656,139,671,157,person,0.025, 2\n', 'blackswan_00005.jpg_person_0.023.jpg,649,137,664,158,person,0.023, 2\n', 'blackswan_00006.jpg_person_0.025.jpg,644,136,658,158,person,0.025, 2\n', 'blackswan_00007.jpg_person_0.024.jpg,637,135,651,159,person,0.024, 2\n', 'blackswan_00009.jpg_person_0.022.jpg,623,140,638,160,person,0.022, 2\n', 'blackswan_00010.jpg_person_0.014.jpg,617,141,631,162,person,0.014, 2\n', 'blackswan_00011.jpg_bird_0.018.jpg,609,144,623,166,bird,0.018, 2\n', 'blackswan_00012.jpg_bird_0.013.jpg,601,146,615,169,bird,0.013, 2\n', 'blackswan_00013.jpg_bird_0.035.jpg,595,152,608,172,bird,0.035, 2\n', 'blackswan_00014.jpg_bird_0.044.jpg,586,157,601,178,bird,0.044, 2\n', 'blackswan_00015.jpg_bird_0.047.jpg,578,167,592,185,bird,0.047, 2\n', 'blackswan_00016.jpg_bird_0.052.jpg,569,168,582,192,bird,0.052, 2\n', 'blackswan_00033.jpg_bird_0.056.jpg,126,241,147,256,bird,0.056, 3\n', 'blackswan_00034.jpg_bird_0.030.jpg,117,243,138,256,bird,0.030, 3\n', 'blackswan_00035.jpg_bird_0.043.jpg,108,243,127,257,bird,0.043, 3\n', 'blackswan_00036.jpg_bird_0.016.jpg,99,244,118,258,bird,0.016, 3\n', 'blackswan_00037.jpg_bird_0.038.jpg,90,245,110,260,bird,0.038, 3\n', 'blackswan_00038.jpg_bird_0.029.jpg,82,246,101,261,bird,0.029, 3\n', 'blackswan_00039.jpg_bird_0.058.jpg,73,247,93,262,bird,0.058, 3\n', 'blackswan_00040.jpg_bird_0.040.jpg,65,248,85,262,bird,0.040, 3\n', 'blackswan_00041.jpg_bird_0.027.jpg,57,249,77,263,bird,0.027, 3\n', 'blackswan_00042.jpg_bird_0.054.jpg,50,250,69,263,bird,0.054, 3\n', 'blackswan_00043.jpg_bird_0.031.jpg,42,250,60,263,bird,0.031, 3\n', 'blackswan_00044.jpg_bird_0.040.jpg,34,249,54,263,bird,0.040, 3\n', 'blackswan_00045.jpg_bird_0.032.jpg,27,249,46,263,bird,0.032, 3\n', 'blackswan_00046.jpg_bird_0.024.jpg,19,249,37,262,bird,0.024, 3\n', 'blackswan_00047.jpg_bird_0.030.jpg,10,249,28,263,bird,0.030, 3\n', 'blackswan_00048.jpg_bird_0.042.jpg,1,249,21,265,bird,0.042, 3\n', 'blackswan_00000.jpg_bird_0.885.jpg,139,72,510,388,bird,0.885, 4\n', 'blackswan_00001.jpg_bird_0.289.jpg,130,68,573,462,bird,0.289, 4\n', 'blackswan_00001.jpg_bird_0.888.jpg,140,70,512,385,bird,0.888, 4\n', 'blackswan_00002.jpg_bird_0.312.jpg,143,64,561,462,bird,0.312, 4\n', 'blackswan_00002.jpg_bird_0.868.jpg,144,68,515,383,bird,0.868, 4\n', 'blackswan_00003.jpg_bird_0.438.jpg,127,65,561,472,bird,0.438, 4\n', 'blackswan_00003.jpg_bird_0.880.jpg,147,66,519,381,bird,0.880, 4\n', 'blackswan_00004.jpg_bird_0.447.jpg,152,65,572,471,bird,0.447, 4\n', 'blackswan_00004.jpg_bird_0.874.jpg,148,65,520,381,bird,0.874, 4\n', 'blackswan_00005.jpg_bird_0.445.jpg,137,64,564,471,bird,0.445, 4\n', 'blackswan_00005.jpg_bird_0.872.jpg,153,65,523,379,bird,0.872, 4\n', 'blackswan_00006.jpg_bird_0.854.jpg,152,64,527,379,bird,0.854, 4\n', 'blackswan_00007.jpg_bird_0.863.jpg,158,66,531,379,bird,0.863, 4\n'...
        for c in range(0, class_num):
            exec('cla_{}_sum = 0'.format(c))
            exec('cla_{}_num = 0'.format(c))

        # print(cla_0_num)
        cla_ave_list = []
        for aaa in range(0, len(clas)):
            cla = clas[aaa]
            cla_path = single_txt_path + '/' + cla
            print('***************************' + str(cla))
            imgs = os.listdir(cla_path)

            save_path = r'F:\source\Visal3/4julei_mv/%s/%s/' % (dataset, folder)
            # 这个save_path是新建的，而且这里的folder就是类别名称，比如blackswan
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            for i in range(0, len(imgs)):
                lin = imgs[i]
                # 'bird_of_paradise.jpg_409_70_571_387_pro0.326.jpg_fir2.900.jpg'
                #  bird_of_paradise_00017.jpg_556_25_638_201_pro0.000.jpg_fir3.627.jpg
                w = lin.split('_')
                if folder == 'bird_of_paradise':
                    patch_name = w[8][:-4]
                    prop = w[8][3:8]
                    img_name = w[0] + '_' + w[1] + '_' + w[2] + '_' + w[3]
                    png_name = img_name[:-4] + '.png'
                    x1 = int(w[4])
                    y1 = int(w[5])
                    x2 = int(w[6])
                    y2 = int(w[7])
                    # obj = w[4]
                    # score = w[5]
                    cla_name = int(cla)
                elif folder == 'snow_leopards':
                    patch_name = w[7][:-4]
                    # snow_leopards_00013.jpg_382_281_412_288_pro0.000.jpg_fir3.918
                    prop = w[7][3:8]
                    img_name = w[0] + '_' + w[1] + '_' + w[2]
                    png_name = img_name[:-4] + '.png'
                    # img_name : ''aeroplane_00034.png'
                    x1 = int(w[3])
                    y1 = int(w[4])
                    x2 = int(w[5])
                    y2 = int(w[6])
                    cla_name = int(cla)
                elif dataset =='Easy-35':
                    patch_name = w[5][:-4]
                    # 0398.png_79_308_154_352_pro0.083.jpg_fir68.241
                    prop = w[5][3:8]
                    img_name = w[0]
                    png_name = img_name[:-4] + '.png'
                    # img_name : ''aeroplane_00034.png'
                    x1 = int(w[1])
                    y1 = int(w[2])
                    x2 = int(w[3])
                    y2 = int(w[4])
                    cla_name = int(cla)

                else:
                    # 'blackswan_00000.jpg_409_70_571_387_pro0.326.jpg_fir2.900.jpg'
                    patch_name = w[6][:-4]
                    # # patch_name = ''person_0.183''
                    prop = w[6][3:8]
                    img_name = w[0] + '_' + w[1]
                    png_name = img_name[:-4] + '.png'
                    # img_name : ''aeroplane_00034.png'
                    x1 = int(w[2])
                    y1 = int(w[3])
                    x2 = int(w[4])
                    y2 = int(w[5])
                    # obj = w[2]
                    # score = w[3]
                    cla_name = int(cla)

                # 这个cla是看属于哪个聚类的
                # print('cla',cla)
                print('img_name', img_name)
                ms_path = ms_total_path + folder + '/' + png_name
                # ms_total_path = r'D:\source\flo_result_fine\flo_vgg_w_49/%s/'%(dataset)
                # gt_path = gt_total_path+folder+'/ground-truth/'+img_name
                # gt_path = gt_total_path+folder+'/GT_object_level/'+img_name
                if dataset == 'Easy-35':
                    gt_path = gt_total_path + folder + '/' + 'GT_object_level/' + png_name
                    rgb_path = rgb_total_path + folder + '/Imgs/' + img_name[:-4] + '.png'
                elif dataset == 'VOS_test_png':
                    gt_path = gt_total_path + folder + '/' + 'ground-truth/' + png_name
                    rgb_path = rgb_total_path + folder + '/Imgs/' + img_name[:-4] + '.png'
                else:
                    gt_path = gt_total_path + folder + '/' + 'ground-truth/' + png_name
                    rgb_path = rgb_total_path + folder + '/Imgs/' + img_name[:-4] + '.jpg'
                # ms_path: 'D:\\source\\flo_result_fine\\flo_vgg_w_49/davis_test/blackswan/blackswan_00000.png'
                if not os.path.exists(ms_path):
                    continue
                ms = cv2.imread(ms_path)
                # gt = cv2.imread(gt_path)
                rgb = cv2.imread(rgb_path)
                # 都是{ndarray（480，854，3）}
                w = rgb.shape[1]
                h = rgb.shape[0]

                ms = cv2.resize(ms, (w, h), interpolation=cv2.INTER_LINEAR)

                cropped_ms = ms[y1:y2, x1:x2]
                # cropped就是 ndarray：(317,162,3)，其实y2-y1 = 317，x2 - x1 = 162
                cropped_rgb = rgb[y1:y2, x1:x2]

                bbox_w = x2 - x1
                bbox_h = y2 - y1
                s = bbox_w * bbox_h
                print('s', s)
                com = cropped_ms.sum()
                # com = 12602097。crop。sum()就是把数组中所有的数加起来。
                if s == 0:
                    continue
                proportion = float(prop)


                motion_value = com / s
                if proportion < 0.25:
                    motion_value = motion_value/10
                # temp = w * h
                # temp = s / temp
                # temp = math.exp(-temp * 5)
                # motion_value = com * temp
                # com就是patch数组加起来，s是ptach的大小，他们相除就是patch的motion_value

                for j in range(0, class_num):
                    # 这个for循环就是用来计算 聚类的motion value的

                    if cla_name == j:
                        if not os.path.exists(save_path + '%d' % (j)):
                            os.makedirs(save_path + '%d' % (j))
                        aa = save_path + '%d' % (j) + '/' + '%s_%s_%s_%s_%s_%s_mv%.3f.jpg' % (
                            img_name, x1, y1, x2, y2,patch_name,motion_value)
                        cv2.imwrite(aa, cropped_rgb)
                        # 上一层遍历每一个patch，
                        # 这个for循环就是看每一个patch的上面的数据命名，然后存储相应的rgb patch。前面的if语句就是判断该rgb patch
                        # 属于哪个聚类的，最后挨个把patch的motion value 加起来得到类的motion value
                        #  f.write('%s,%s,%s,%s,%s,%s,%s,%s'%(img_name,x1,y1,x2,y2,obj,score,cla)+'\n')
                        # 'cla_{}_sum'.format(j)='cla_{}_sum'.format(j)+motion_v
                        # 'cla_{}_num'.format(j)='cla_{}_num'.format(j)+1
                        # print('********************' + str(motion_value))
                        exec('cla_{}_sum = cla_{}_sum + {}'.format(j, j, motion_value))
                        exec('cla_{}_num = cla_{}_num + {}'.format(j, j, 1))

                        # print(‘cla_0_num’，cla_0_num)
                        # 'cla_{}_sum = cla_{}_sum + {}'.format(j,j,motion_v)

        class_0_ave = cla_0_sum / cla_0_num
        new_name = '0_%.3f' % (class_0_ave)
        os.rename(save_path + '0', save_path + new_name)
        cla_ave_list.append(class_0_ave)
        print('class_0_ave', class_0_ave)

        class_1_ave = cla_1_sum / cla_1_num
        new_name = '1_%.3f' % (class_1_ave)
        os.rename(save_path + '1', save_path + new_name)
        cla_ave_list.append(class_1_ave)
        print('class_1_ave', class_1_ave)

        class_2_ave = cla_2_sum / cla_2_num
        new_name = '2_%.3f' % (class_2_ave)
        os.rename(save_path + '2', save_path + new_name)
        cla_ave_list.append(class_2_ave)
        print('class_2_ave', class_2_ave)

        class_3_ave = cla_3_sum / cla_3_num
        new_name = '3_%.3f' % (class_3_ave)
        os.rename(save_path + '3', save_path + new_name)
        cla_ave_list.append(class_3_ave)
        print('class_3_ave', class_3_ave)

        class_4_ave = cla_4_sum / cla_4_num
        new_name = '4_%.3f' % (class_4_ave)
        os.rename(save_path + '4', save_path + new_name)
        cla_ave_list.append(class_4_ave)
        print('class_4_ave', class_4_ave)

        class_5_ave = cla_5_sum / cla_5_num
        new_name = '5_%.3f' % (class_5_ave)
        os.rename(save_path + '5', save_path + new_name)
        cla_ave_list.append(class_5_ave)
        print('class_5_ave', class_5_ave)

        class_6_ave = cla_6_sum / cla_6_num
        new_name = '6_%.3f' % (class_6_ave)
        os.rename(save_path + '6', save_path + new_name)
        cla_ave_list.append(class_6_ave)
        print('class_6_ave', class_6_ave)

        class_7_ave = cla_7_sum / cla_7_num
        new_name = '7_%.3f' % (class_7_ave)
        os.rename(save_path + '7', save_path + new_name)
        cla_ave_list.append(class_7_ave)
        print('class_7_ave', class_7_ave)

        x = np.array(cla_ave_list)
        print(cla_ave_list)
        x_new = np.argsort(-x)  # 按升序排列
        print('x_new', x_new)


