# -*- coding: utf-8 -*-
import os
import shutil

# davis_test
# Segtrack-v2
# Visal
# Easy-35
# VOS_test_png

dataset_list = ['VOS_test_png']
for sj in range(0, len(dataset_list)):

    dataset = dataset_list[sj]
    total_path = r'F:\source\Visal3/4julei_mv/%s/' % (dataset)
    videos = os.listdir(total_path)
    for i in range(0, len(videos)):
        video_name = videos[i]
        print(video_name)
        cluster_path = total_path + video_name
        clusters = os.listdir(cluster_path)
        for j in range(0, len(clusters)):
            cluster_name = clusters[j]
            clas_path = r'F:\source\Visal3\5julei_ms+\cluster/%s/%s/%s' % (dataset, video_name, cluster_name)

            if not os.path.exists(clas_path):
                save_path1 = r'F:\source\Visal3\5julei_ms-\cluster/%s/%s/%s' % (
                dataset, video_name, cluster_name)
                save_path2 = r'F:\source\Visal3\5julei_ms-\sal/%s/%s/' % (dataset, video_name)
                if not os.path.exists(save_path1):
                    os.makedirs(save_path1)
                if not os.path.exists(save_path2):
                    os.makedirs(save_path2)

                img_path = cluster_path + '/' + cluster_name
                imgs = os.listdir(img_path)
                for k in range(0, len(imgs)):
                    img_name = imgs[k]
                    shutil.copy(img_path + '/' + img_name, save_path1 + '/' + img_name)
                    shutil.copy(img_path + '/' + img_name, save_path2 + '/' + img_name)
