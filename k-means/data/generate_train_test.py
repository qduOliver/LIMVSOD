import os

path = r'G:\song\Yet-Another-EfficientDet-Pytorch-master\nms\M7_0.01_0.00001_top5\0.5\cut_con_1\Segtrack-V2\birdfall/'
imgs = os.listdir(path)
f=open('../txt/birdfall.txt','a')
for i in range(0,len(imgs)):
    img_name=imgs[i]
    img_path=path+'/'+img_name
    # label=img_name.split('_')[0]
    AA=img_name.split('.jpg')[1]
    print(AA)
    label=AA.split('_')[1]
    print(label)
    f.write(img_path+','+label+'\n')
f.close()