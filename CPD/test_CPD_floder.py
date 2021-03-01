import torch
import torch.nn.functional as F
import numpy as np
import pdb, os, argparse
from scipy import misc
import  cv2
import matplotlib.pyplot as plt
import time
import os
import torchvision
from data import get_loader_my
from model.Two_stream_pri_1 import CPD_VGG

from model.CPD_models import CPD_VGG,CPD
from data import test_dataset_my
from torch.autograd import Variable

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--is_ResNet', type=bool, default=False, help='VGG or ResNet backbone')
opt = parser.parse_args()

dataset_list=['davis_test']

for a in range(0,len(dataset_list)):
    
    dataset_name=dataset_list[a]
    dataset_path=r'D:\dataset\video/%s/'%(dataset_name)

    test_folders=os.listdir(dataset_path)
    
    for i in range(0,len(test_folders)):
        
        dataset = test_folders[i]
        
        model = CPD_VGG() 
    
        static_model=torch.load(r'D:\code\new_6\8_online\model_weight\adp_th_train_2_max_meanx2_nofix_noaug_adp_test_2_10\%s\txt_ob_patch_online\adp_th_train_2_max_meanx2_nofix_noaug_adp_test_2/%s/%s/vgg_10.pth'%(dataset_name,dataset_name,dataset))
        model.load_state_dict(static_model)  
        
        model.cuda()
        model.eval()
        
        if opt.is_ResNet:
            save_path = './flo_result_yuan(inputbn_res)/davis_test/' + dataset + '/'
        else:
            if dataset_name=='davis_test':
                dataset_name='davis_test'
                
            if dataset_name=='VOS_test':
                dataset_name='VOS_test'
                
            save_path = r'D:\code\new_6\8_online\model_result/adp_th_train_2_max_meanx2_nofix_noaug_adp_test_2_10/%s/10/%s/'%(dataset_name,dataset)
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        image_root = dataset_path + dataset +'/Imgs/'
        test_loader = test_dataset_my(image_root, opt.testsize)
        names=[]
        runtime_total=0
        
        for i in range(test_loader.size):
            
            image, name ,index,shape= test_loader.load_data()
            image = image.cuda()
    #     
            star=time.time()
            _, ress = model(image)
            end=time.time()
            runtime=end-star
            runtime_total=runtime_total+runtime
            print('runtime',runtime)
            h=shape[0]
            w=shape[1]
            
            res = F.upsample(ress, size=[h,w], mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            cv2.imwrite(save_path+name, res*255)	
            end=time.time()
        
        runtime_f= runtime_total/test_loader.size 
        print(runtime_f)
    