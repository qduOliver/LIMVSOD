import os
from shutil import copyfile

# FBMS-59/FBMS-Test
# Segtrack-v2
# Visal
# davis_test
# Easy-35
# VOS_test_png_gt

dataset_List = ['Segtrack-v2','Visal','Easy-35','VOS_test_png']

for sj in range(0,len(dataset_List)):

    dataset = dataset_List[sj]
    path_t=r'D:/code/new/3_cut/M7_0.01_0.2_top13/cut/%s'%(dataset)
    folders=os.listdir(path_t)
    for i in range(0,len(folders)):
        folder=folders[i]
        from_path=path_t+'/'+folder
        imgs=os.listdir(from_path)
        savePath=r'%s/%s/%s'%('D:/code/new/3_cut/M7_0.01_0.2_top13/cut_con_1/',dataset,folder)
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        for j in range(0,len(imgs)):
            img_name=imgs[j]
            print(img_name)
            cuts=os.listdir(from_path+'/'+img_name)
            for k in range(0,len(cuts)):
                cut_name=cuts[k]
                copyfile(from_path+'/'+img_name+'/'+cut_name,savePath+'/'+img_name+'_'+cut_name) # 2