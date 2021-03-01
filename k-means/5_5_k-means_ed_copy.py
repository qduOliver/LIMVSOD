import torch
from sklearn.cluster import KMeans
import cfg

import os
from PIL import Image
from data.transform import get_test_transform,get_yuan_transform
import torch.nn as nn
import numpy as np
from models.efficientnet_pytorch import EfficientNet
import cv2
from sklearn.metrics.pairwise import euclidean_distances

# FBMS-59/FBMS-Test
# Segtrack-v2
# Visal                               
# davis_test
# Easy-35
# VOS_test_png_gt  

model = EfficientNet.from_pretrained('efficientnet-b7')
for parameter in model.parameters():
    parameter.requires_grad = False
model.eval()
model.cuda()

# dataset_list =['Easy-35','VOS_test_png']
# davis_test
dataset_list =['davis_test']
for a in range(0,len(dataset_list)):
    
    dataset = dataset_list[a]
    # 1
    path = r'D:\code\new_6\1_ms_rank_class\ms_adp_meanx2\train\%s/'%(dataset)
    videos = os.listdir(path)
    for i in range(0,len(videos)):
        
        video = videos[i]
        print(video)
        total_path=path+'/%s/1/'%(video)

        imgs = os.listdir(total_path)
        X_encoded = np.empty((len(imgs), 2560), dtype='float32')
        img_arr=np.empty((len(imgs),300,300,3), dtype='float32')
        # img_arr = []
        img_arr_name=[]
        img_shape_list = []
        for i in range(0,len(imgs)):
            img_name=imgs[i]
            img_path=total_path+'/'+img_name
            img = Image.open(img_path).convert('RGB')
            img_shape = img.size
            # print(type(img))
            img_shape_list.append(img_shape)
            img_input = get_test_transform(size=cfg.INPUT_SIZE)(img).unsqueeze(0)
            img_show = get_yuan_transform(size=cfg.INPUT_SIZE)(img).unsqueeze(0)
            img_arr[i,:]=np.transpose(img_show,[0,2,3,1])
            # img_arr.append(img)
            # img_arr=np.array(img_arr)
            img_arr_name.append(img_name)
            if torch.cuda.is_available():
                img_input = img_input.cuda()
            with torch.no_grad():
                out = model.extract_features(img_input)
            # print(out.size())
            out2 = nn.AdaptiveAvgPool2d(1)(out)
            feature = out2.view(out.size(1), -1).squeeze(1)
            X_encoded[i,:]=feature.cpu()
            
        n_clusters = 8
        
        km = KMeans(n_clusters=n_clusters,random_state=0)
        km.fit(X_encoded)
        center=km.cluster_centers_     #[5,2560]
        # print(center)
        # best_center = center[4]
        
        # print('distance',distance)
        for hh in range(0,n_clusters):
            # plt.figure()
            cluster = hh
            rows, cols = 100, 100
            start = 0
            
            save_path = r'D:\code\new_6\1_ms_rank_class\ms_adp_meanx2\train_ed\%s/%s/1/'%(dataset,video)
            if not os.path.exists(save_path):
                    os.makedirs(save_path)
                    
            distance=euclidean_distances(center[hh, np.newaxis], X_encoded, squared=True)
            
            labels = np.where(km.labels_==cluster)[0][start:start+rows*cols]
            for i, label in enumerate(labels):
                # label = int(label)
                # print(label)
                img_shape = img_shape_list[label]
                img_name=img_arr_name[label]
                img = cv2.cvtColor(img_arr[label], cv2.COLOR_RGB2BGR) 
                distance_yy = distance[0,label]
                print('distance_yy',distance_yy)
                save_img_name =  '%.3f_%s'%(distance_yy,img_name)
                img=cv2.resize(img,img_shape)
                cv2.imwrite(save_path+save_img_name,img*255)