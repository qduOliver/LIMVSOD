import random
import os
import shutil
from evalutor_my_bbox import Eval_thread
import cv2
import matplotlib.pyplot as plt

from skimage import data, filters

import numpy as np

datasets = [ 'Visal']
# datasets = ['VOS_test_png']
for d in range(0,len(datasets)):
    pt = ''
    dataset = datasets[d]
    total_ms_path = 'F:\source\CPD_pwc_11yearresult/%s/'%dataset
    total_patch_path = r'F:\source\Visal17\35all_patch_paste\%s/%s/'%(pt,dataset)
    total_path_save = r'F:\source\Visal17\36Smeasure_result\11/%s/%s/'%(pt,dataset)
    total_txt_path = r'F:\source\Visal17\36Smeasure_result\22/%s/%s/'%(pt,dataset)
    folders = os.listdir(total_ms_path)

    for i in range(0, len(folders)):
        folder = folders[i]
        img_path = total_ms_path + folder + '/'
        txt_folder_path = total_txt_path + folder + '/'
        if os.path.exists(txt_folder_path):
            print('exist')
        else:
            os.makedirs(txt_folder_path)
        # 到folder哪一级了
        if not os.path.exists(total_path_save + folder):
            os.makedirs(total_path_save + folder)
        else:
            print("exist")
        imgs = os.listdir(img_path)
        txt_path = txt_folder_path + '%s.txt' % folder
        txt = open(txt_path,'a')
        for a in range(0, len(imgs)):

            #        img=str(video)+'_'+'%05d'%(a[i])+'.png'
            img = imgs[a]


            ms_path = img_path + img
            patch_path = total_patch_path + folder + '/' + img
            if not os.path.exists(patch_path):
                print("不存在名叫 %s 图片"%img)
                continue

            # aa = plt.imread(file_path)
            # plt.imshow(aa*255)
            # plt.show()
            # 怎么显出出来的颜色变了啊 不是黑白变绿黑了
            # bb = plt.imread(gt_path)
            # plt.imshow(bb)sssssssssssssssssssssssssssssssssss
            # plt.show()
            # 'G:\\source\\MyResult\\dataset_ms/Segtrack-v2/birdfall/birdfall_00000.png'
            img_tu = plt.imread(ms_path)

            cs = plt.imread(patch_path)
            cs1 = cs[:,:,1]


            s_meature = Eval_thread(cs1, img_tu, True)
            sm_value = round(s_meature.run(), 3)
            # print(sm_value)

            cv2.imwrite(total_path_save + folder + '/' + img[:-4] + '=' + str(sm_value) + '.png', cs * 255)
            txt.write(total_path_save + folder + '/' + img + '  score : %s'%sm_value + '\n')

            thresh_end = filters.threshold_otsu(img_tu)  # 返回一个阈值
            cc = (img_tu >= thresh_end)*1.0

            cc = cc.astype(np.float)
        txt.close()
            #下面这个代码，是用来保存，大于某个阈值的图片
            # if sm_value > 0.90:
            #
            #     if not os.path.exists(path_select + folder):
            #         os.makedirs(path_select + folder)
            #
            #     cv2.imwrite(path_select + folder + '/' + img, img_tu_bi * 255)


