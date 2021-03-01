import numpy as np
import cv2
import random
from utils.utils import preprocess, invert_affine, postprocess, STANDARD_COLORS, standard_to_bgr, get_index_label, plot_one_box
import os

obj_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', '', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
            'cow', 'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack', 'umbrella', '', '', 'handbag', 'tie',
            'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', '', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
            'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
            'cake', 'chair', 'couch', 'potted plant', 'bed', '', 'dining table', '', '', 'toilet', '', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', '', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush']

color_list = standard_to_bgr(STANDARD_COLORS)

# 先选取得分最高的框，与剩下的比，去掉重复的框。
#然后从剩下的框中选概率最大的，比较概率值
def nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 5]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]   #从大到小排序
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        print(order[1:])
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep

# thresh_in=0.2 
# thresh_out=0.2
def score_nms_1 (dets, thresh_in,thresh_out):
    
    scores = dets[:, 5]
    order = scores.argsort()[::-1]   #从大到小排序
    best_index=order[0]
    score_best = dets[best_index]
    x1_b = score_best[0]
    y1_b = score_best[1]
    x2_b = score_best[2]
    y2_b = score_best[3]
    score_b = score_best[5]
    
    keep=[]
    keep.append(best_index)
    for i in range(1,len(dets)):
        index =order[i]
        score = dets[index]
        x1 = score[0]
        y1 = score[1]
        x2 = score[2]
        y2 = score[3]
        score_i = score[5]
        
        xx1 = np.maximum(x1, x1_b)
        yy1 = np.maximum(y1, y1_b)
        xx2 = np.minimum(x2, x2_b)
        yy2 = np.minimum(y2, y2_b)
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        
        if inter==0:
            keep.append(index)
        if x1_b<x1 and y1_b<y1 and x2_b>x2 and  y2_b>y2:
            if score_i>score_b*thresh_in:
                keep.append(index)
        elif x1_b<x1 and y1_b<y1 and x2_b<x2 and  y2_b<y2:
            if score_i>score_b*thresh_in:
                keep.append(index)
        else:
            if score_i>score_b*thresh_out:
                keep.append(index)
            else:
                print('no')
    return keep

def score_nms_2(dets, thresh_in,thresh_out):
    scores = dets[:, 5]
    order = scores.argsort()[::-1]   #从大到小排序
    best_index=order[1]
    score_best = dets[best_index]
    x1_b = score_best[0]
    y1_b = score_best[1]
    x2_b = score_best[2]
    y2_b = score_best[3]
    score_b = score_best[5]
    
    keep=[]
    keep.append(best_index)
    keep.append(order[0])
    for i in range(2,len(dets)):
        index =order[i]
        score = dets[index]
        x1 = score[0]
        y1 = score[1]
        x2 = score[2]
        y2 = score[3]
        score_i = score[5]
        
        xx1 = np.maximum(x1, x1_b)
        yy1 = np.maximum(y1, y1_b)
        xx2 = np.minimum(x2, x2_b)
        yy2 = np.minimum(y2, y2_b)
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        
        if inter==0:
            keep.append(index)
        if x1_b<x1 and y1_b<y1 and x2_b>x2 and  y2_b>y2:
            if score_i>score_b*thresh_in:
                keep.append(index)
        elif x1_b<x1 and y1_b<y1 and x2_b<x2 and  y2_b<y2:
            if score_i>score_b*thresh_in:
                keep.append(index)
        else:
            if score_i>score_b*thresh_out:
                keep.append(index)
            else:
                print('no')
    return keep
    
# FBMS-59/FBMS-Test
# Segtrack-v2
# Visal
# davis_test
# Easy-35
# VOS_test_png

dataset ='Easy-35'
path_total = r'D:\code\pycharm_workpeace\Yet-Another-EfficientDet-Pytorch-master\txt\new_1_total/%s/'%(dataset)
folders=os.listdir(path_total)
for a in range(0,len(folders)):
    folder = folders[a]
    print('folder',folder)
    imgs=os.listdir(path_total+'/'+folder)
    for b in range(0,len(imgs)):
        img_name= imgs[b]
        print('img_name',img_name)
        txt_path = r'%s\%s\%s'%(path_total,folder,img_name)
        f=open(txt_path)
        bboxs=f.readlines()
        dets=[]
        thresh=0.8
        for i in range(0,len(bboxs)):
            bbox=bboxs[i]
            w=bbox.split(',')
            num=w[0]
            x1=int(w[1])
            y1=int(w[2])
            x2=int(w[3])
            y2=int(w[4])
            if x1==x2 or y1==y2:
                continue;
                
            class_id= w[5]   
            obj=w[6]
            score=float(w[-1])
            det =[x1,y1,x2,y2,class_id,score]
            dets.append(det)
        dets=np.array(dets,dtype=float)
        print('dets',dets.shape)

# img=cv2.imread(r'F:\dataset\Segtrack-v2\bmx\Imgs/bmx_00035.jpg') 
# img_cp=img.copy()
# for box in dets.tolist():pa't'h
#     x1,y1,x2,y2,calss_id,score=int(box[0]),int(box[1]),int(box[2]),int(box[3]),box[-1]
#     y_text=int(random.uniform(y1, y2))
#     obj=obj_list[calss_id]
#     plot_one_box(img_cp, [x1, y1, x2, y2], label=obj,score=score,color=color_list[class_id])
# cv2.imwrite('./nms'+'/'+'bmx_00035_0.jpg', img_cp)

        rtn_box=nms(dets,thresh)	#0.3为faster-rcnn中配置文件的默认值
        cls_1_dets=dets[rtn_box, :]
        score_box_1=score_nms_1(cls_1_dets,0.2,0.2)
        cls_2_dets=cls_1_dets[score_box_1,:]
        cls_dets=cls_2_dets
        if len(cls_2_dets)>2:
            score_box_2=score_nms_2(cls_2_dets,0.2,0.2)
            cls_dets=cls_2_dets[score_box_2,:]
        print ("nms box:", cls_dets)
        save_path='./nms_total/m7_0.01_0.2/img/0.7_0.2_0.2/%s/%s'%(dataset,folder)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_path_txt = './nms_total/m7_0.01_0.2/txt/0.7_0.2_0.2//%s/%s'%(dataset,folder)
        if not os.path.exists(save_path_txt):
            os.makedirs(save_path_txt)
        f=open(save_path_txt+'/'+img_name,'a')
        img_path=r'D:\dataset\video\%s/%s\Imgs/%s'%(dataset,folder,img_name[:-4])
        # print(img_name[:-4])
        # img_path = r'D:\dataset\video\%s\VOS_test_png_gt/%s/Imgs/%s' % (dataset, folder, img_name[:-8]+'.png')
        img=cv2.imread(img_path) 
        img_cp=img.copy()
        
        for k in range(0,len(cls_dets.tolist())):
            box=cls_dets.tolist()[k]
            x1,y1,x2,y2,calss_id,score=int(box[0]),int(box[1]),int(box[2]),int(box[3]),box[4],box[-1]
            y_text=int(random.uniform(y1, y2))
            print('calss_id',calss_id)
            obj=obj_list[int(calss_id)]
            plot_one_box(img_cp, [x1, y1, x2, y2], label=obj,score=score,color=color_list[int(class_id)])
            f.write('%s,%s,%s,%s,%s,%s,%s,%s'%(str(k),x1,y1,x2,y2,class_id,obj,score)+'\n')

        f.close()
        cv2.imwrite(save_path+'/'+img_name[:-4], img_cp)