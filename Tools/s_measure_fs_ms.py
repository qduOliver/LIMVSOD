# -*- coding: utf-8 -*-
import random
import os
import shutil
from evalutor_my_bbox import Eval_thread
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from skimage import data,filters

import numpy as np

dataset_list = ['davis_test','Visal','Easy-35','VOS_test_png']

for sj in range(0,len(dataset_list)):

    dataset = dataset_list[sj]
    #
    fs_path = r'D:\code\new_5\7_paste\calss\no_full/%s/'%(dataset)
    ms_path = r'D:\dataset\flo_s\CPD_ms\720_flo_CPD_VGG_fine_davis2000all_lr4_b10_49_bi/%s/'%(dataset)

    save_path = r'D:\code\new_5\7_paste\class_no_full_smeature_v/%s/'%(dataset)

    videos = os.listdir(fs_path)

    for i in range(0,len(videos)):
        #GT_object_level
        #ground-truth
        video=videos[i]
        pris_path = fs_path+video
        mss_path = ms_path+video
        pris=os.listdir(pris_path)

        if not os.path.exists(save_path+video):
            os.makedirs(save_path+video)

        length=len(pris)
        for a in range(0,length-1):

            pri=pris[a]

            pri_img_path=pris_path+'/'+pri
            ms_img_path=mss_path+'/'+pri

            if os.path.exists(pri_img_path):

                gt=cv2.imread(pri_img_path,0)
                gt = gt.astype(np.float32)
                gt=gt/255

                img_1=cv2.imread(ms_img_path,0)
                img_1 = img_1.astype(np.float32)
                img_1=img_1/255

                shape=gt.shape
                width=shape[1]
                hight=shape[0]

                img_1 =cv2.resize(img_1,(width,hight))

                w=Eval_thread(gt,img_1,True)
                w=round(w.run(),3)
                print(w)

                cv2.imwrite(save_path+video+'/'+pri[:-4]+'_'+str('%.3f'%(w))+'.png',gt*255)
