# -*- coding: utf-8 -*-
import os
import numpy as np
import shutil

dataset = 'VOS_test_png'
total_apth = r'D:\code\new_6\2_test_1\%s\adp_th_train_2_max_meanx2_nofix_noaug_adp\test_2_all_others_result\test_pre_01\sal/'%(dataset)

video_list = os.listdir(total_apth)

for i in range(0,len(video_list)):
    
    vidoe_name = video_list[i]
    clas_path = total_apth + vidoe_name
    clas_list = os.listdir(clas_path)
    
    for j in range(0,len(clas_list)):
        cla_name = clas_list[j]
        imgs_path = clas_path + '/' + cla_name
        imgs = os.listdir(imgs_path)
        
        msv_list = []
        for k in range(0,len(imgs)):
            img_name = imgs[k]
            msv = float(img_name.split('_')[0])
            print('msv',msv)
            msv_list.append(msv)
        
        ave = np.mean(np.array(msv_list))                                                                 
        print('##############  ave  #################',ave)
        
        new_save_path_train = r'D:\code\new_6\2_test_1\%s\adp_th_train_2_max_meanx2_nofix_noaug_adp\test_2_all_other_ms_mean/train/%s/%s/%s/'%(dataset,dataset,vidoe_name,cla_name)
        new_save_path_test = r'D:\code\new_6\2_test_1\%s\adp_th_train_2_max_meanx2_nofix_noaug_adp\test_2_all_other_ms_mean/test/%s/%s/%s/'%(dataset,dataset,vidoe_name,cla_name)
        

        
        if not os.path.exists(new_save_path_train):
            os.makedirs(new_save_path_train)
            
        if not os.path.exists(new_save_path_test):
            os.makedirs(new_save_path_test)
        
        for k in range(0,len(imgs)):
            img_name = imgs[k]
            msv = float(img_name.split('_')[0])
            
            if cla_name=='0':
                if  msv<=ave :
                    shutil.copy(imgs_path+'/'+img_name,new_save_path_train+'/'+img_name)
                else:
                    shutil.copy(imgs_path+'/'+img_name,new_save_path_test+'/'+img_name)
                    
            if cla_name=='1':
                
                if  msv>=ave :
                    shutil.copy(imgs_path+'/'+img_name,new_save_path_train+'/'+img_name)
                else:
                    shutil.copy(imgs_path+'/'+img_name,new_save_path_test+'/'+img_name)
            
        

        
