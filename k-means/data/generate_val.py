import os

path = r'G:\song\Yet-Another-EfficientDet-Pytorch-master\nms\M7_0.01_0.00001_top5\0.5\cut_con_1\Segtrack-V2\birdfall/'
imgs = os.listdir(path)
if not os.path.exists('../txt/birdfall/train/'):
    os.makedirs('../txt/birdfall/train/')
    
if not os.path.exists('../txt/birdfall/val/'):
    os.makedirs('../txt/birdfall/val/')
train=open('../txt/birdfall/train/train_%d.txt'%(1),'a')
for i in range(0,len(imgs)):
    f=open('../txt/birdfall/val/val_%d.txt'%(i+1),'a')
    img_name=imgs[i]
    img_path=path+'/'+img_name
    # label=img_name.split('_')[0]
    AA=img_name.split('.jpg')[1]
    print(AA)
    label=AA.split('_')[1]
    print(label)
    f.write(img_path+','+label+'\n')
    train.write(img_path+','+label+'\n')
    f.close()