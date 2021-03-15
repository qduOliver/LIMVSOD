import os
from PIL import Image
import matplotlib.pyplot  as plt
import cv2
import numpy as np

dataset_list = ['Segtrack-v2','davis_test','Visal','Easy-35','VOS_test_png']
for sj in range(0,len(dataset_list)):

    dataset = dataset_list[sj]
    patch_total_path = r'D:\code\new_5\6_bboxtoseg\seg/%s/'%(dataset)
    gt_path = r'D:\dataset\2020_result\video_u2net_results/%s/'%(dataset)
    
    if dataset == 'VOS_test_png':
        dataset_gt = 'VOS_test_png'
        gt_path = r'D:\dataset\2020_result\video_u2net_results/%s/'%(dataset_gt)

    # gt_path = r'D:\dataset\2020_result\video_u2net_results/%s/'%(dataset)
    videos =os.listdir(patch_total_path)
    for i in range(0,len(videos)):
        video = videos[i]
        imgs_path = patch_total_path+video
        # gts_path = gt_path+video+'/ground-truth/'
        gts_path = gt_path+video+'/'
        # gts_path = gt_path+video+'/GT_object_level/'
        imgs = os.listdir(imgs_path)

        for j in range(0,len(imgs)):
            img_name = imgs[j]
            w1 = img_name.split('bbox')[0]
            w2 = img_name.split('bbox')[1]
            gt_name = w2.split('.png')[0][1:]+'.png'
            # gt_name = w2.split('.jpg')[0][1:]
            print(gt_name)

            if os.path.exists(gts_path+gt_name):

                gt = Image.open(gts_path+gt_name)
                w = gt.size[0]
                h = gt.size[1]

                save_path = r'D:\code\new_5\6_bboxtoseg\seg_max\%s/%s/'%(dataset,video)
                mask = Image.new('RGB', (w,h),'black')

                patch = Image.open(imgs_path+'/'+img_name)

                ww = w1.split('_')

                ms = ww[0]
                x1 = int(float(ww[1]))
                y1 = int(float(ww[2]))
                x2 = int(float(ww[3]))
                y2 = int(float(ww[4]))

                if len(ww) == 7:
                    ms = ww[1]
                    x1 = int(ww[2])
                    y1 = int(ww[3])
                    x2 = int(ww[4])
                    y2 = int(ww[5])

                mask.paste(patch,(x1,y1))

                if os.path.exists(save_path+gt_name):

                    mask_1 = Image.open(save_path+gt_name)
                    # mask = np.array(mask) + np.array(mask_1)
                    mask = np.array(mask)
                    mask_1 = np.array(mask_1)
                    mask = cv2.add(mask,mask_1)
                    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
                    mask = Image.fromarray(np.uint8(mask*255))

                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                mask.save(save_path + gt_name)
        
