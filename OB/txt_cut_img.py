import os
import cv2

# FBMS-59/FBMS-Test
# Segtrack-v2
# Visal
# davis_test
# Easy-35
# VOS_test_png   VOS_test_png_gt

dataset='VOS_test_png'
# img_total_path = r'D:\dataset\video/%s/VOS_test_png_gt/'%(dataset)
img_total_path = r'D:\dataset\video/%s/'%(dataset)
txt_total_path = r'D:\code\pycharm_workpeace\Yet-Another-EfficientDet-Pytorch-master\txt\new_1_total/%s/'%(dataset)
folers = os.listdir(txt_total_path)
for i in range(0,len(txt_total_path)):
    folder=folers[i]
    imgs=os.listdir(img_total_path+folder+'/Imgs/')
    
    for k in range(0,len(imgs)):
        img_name=imgs[k]
        print(img_name)
        # txt_path=txt_total_path+folder+'/'+img_name+'.txt'
        txt_path = txt_total_path + folder + '/' + img_name[:-4] + '.jpg' + '.txt'
        if os.path.exists(txt_path):
            f=open(txt_path)
            fmany=f.readlines()
            for j in range(0,len(fmany)):
                line = fmany[j]
                w=line.split(',')
                num=w[0]
                x1=int(w[1])
                y1=int(w[2])
                x2=int(w[3])
                y2=int(w[4])
                obj=w[6]
                score=float(w[7])
                n=img_total_path+folder+'/Imgs/'+img_name
                img = cv2.imread(n)
                if y1==y2 or x1==x2:
                    continue
                cropped = img[y1:y2,x1:x2]
                if len(cropped)==0:
                    continue

                save_path=r'D:\code\new\3_cut\M7_0.01_0.2_top13\cut/%s/%s/%s'%(dataset,folder,img_name)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                cv2.imwrite("%s/%s_%.03f.jpg"%(save_path,obj,score), cropped)