import os
import numpy as np
import shutil

# davis_test
# Segtrack-v2
# Visal
# Easy-35
# VOS_test_png   1.8

#dataset_list = ['Segtrack-v2','Visal','Easy-35','VOS_test_png']
# for jj in range(10,20):
#      ff = str(jj)

dataset_list = ['VOS_test_png']
# dataset_list = ['tryData']
for sj in range(0, len(dataset_list)):

         dataset = dataset_list[sj]
         # total_path = r'D:/code/new/5_ms_selcect/top13_8_patch_rgb_v_mean_new/%s'%(dataset)

         total_path = r'F:\source\Visal3/4julei_mv/%s' % (dataset)
         videos = os.listdir(total_path)
         for i in range(0, len(videos)):
             video = videos[i]
             print('video', video)
             clas_path = total_path + '/' + video
             cls_list = os.listdir(clas_path)
             cls_val_list = []
             for j in range(0, len(cls_list)):
                 cls = cls_list[j]
                 cla_val = float(cls.split('_')[1])
                 cls_val_list.append(cla_val)
             # cls_val_list.sort()
             # print(cls_val_list)]
             x = np.array(cls_val_list)
             index_list = np.argsort(-x)
             save_list = []
             del_list = []
             max_val = cls_val_list[index_list[0]]
             show_list = []

             sort_list = []
             for sj in range(0, len(index_list)):
                 # print(cls_val_list[index_list[sj]])
                 show_list.append(cls_val_list[index_list[sj]])
                 #这个show_list就是class—value从大到小排好序的

                 cur_val = cls_val_list[index_list[sj]]
                 if cur_val == 0.000:
                     continue

                 if max_val / cur_val < 2:
                     #  默认是2
                     max_val = cur_val
                     save_list.append(max_val)

                 # ratio = max_val / cur_val
                 # max_val = cur_val
                 # save_list.append(round(ratio,3))

             print('save_list', save_list)
             print('index_list', index_list)

             # print(show_list)
             # print(save_list)
             #
             index_list_save = []
             for a in range(0, len(save_list)):
                 index_list_save.append(index_list[a])

             print(index_list_save)
             for b in range(0, len(index_list_save)):

                 cls = cls_list[index_list_save[b]]
                 print(cls)
                 old_path = clas_path + '/' + cls


                 # new_path1 = r'D:\code\new_6\1_ms_rank_class\fix_thre/%s/2.0/cluster/%s/%s'%(dataset,video,cls)
                 # new_path2 = r'D:\code\new_6\1_ms_rank_class\fix_thre/%s/2.0/sal/%s/'%(dataset,video)
                 new_path1 = r'F:\source\Visal3/5julei_ms+/cluster/%s/%s/%s' % (dataset, video, cls)
                 new_path2 = r'F:\source\Visal3\5julei_ms+/sal/%s/%s/' % (dataset, video)

                 if not os.path.exists(new_path1):
                     os.makedirs(new_path1)

                 if not os.path.exists(new_path2):
                     os.makedirs(new_path2)

                 imgs = os.listdir(old_path)

                 for img in range(0, len(imgs)):
                     img_name = imgs[img]
                     shutil.copy(old_path + '/' + img_name, new_path1 + '/' + img_name)
                     shutil.copy(old_path + '/' + img_name, new_path2 + '/' + img_name)
