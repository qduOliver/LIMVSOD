import torch
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import pdb, os, argparse
from datetime import datetime
from utils import clip_gradient, adjust_lr

from model.CPD_models import CPD
from model.Two_stream_pri_1 import CPD_VGG
from data import get_loader_txt_pri,get_loader_txt
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import time

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=11, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batchsize', type=int, default=2, help='training batch size')
parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--is_ResNet', type=bool, default=False, help='VGG or ResNet backbone')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')
opt = parser.parse_args()

print('Learning Rate: {} ResNet: {} '.format(opt.lr, opt.is_ResNet))
# build models
if opt.is_ResNet:
    model = CPD()
else:
    model_total= CPD_VGG()
    
#rgb_pre_train=torch.load(r'premodel/cpd_fine_davis_aug/fuse_pri_3.pth',map_location='cuda:0')
rgb_pre_train=torch.load(r'premodel/CPD.pth',map_location='cuda:0')
model_dict = model_total.state_dict()
pretrained_dict = {k: v for k, v in rgb_pre_train.items() if k in model_dict}
#print(pretrained_dict.k)
model_dict.update(pretrained_dict)
model_total.load_state_dict(model_dict)
model_total.cuda()

params = model_total.parameters()
optimizer = torch.optim.Adam(params, opt.lr)
CE = torch.nn.BCEWithLogitsLoss()

#def train(train_loader, model1,model2,model_fuse, optimizer, epoch):
def train(train_loader, model_total,optimizer, epoch,dataset_name,video):
    model_total.train()

    for i, pack in enumerate(train_loader, start=0):
        # print('i',i)
        optimizer.zero_grad()
        images,gts = pack
       
        images = Variable(images)
        gts = Variable(gts)
        
        images = images.cuda()
        gts = gts.cuda()
        att_s,d= model_total(images)
        
#        print('gts',gts.shape)
        loss1 = CE(att_s, gts)
        loss2 = CE(d, gts)
        
        loss = loss1 + loss2
        loss.backward()

        clip_gradient(optimizer, opt.clip)
        optimizer.step()

#        if i % 10 == 0 or i == len(train_loader):
        print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss1: {:.4f} Loss2: {:0.4f} '.
              format(datetime.now(), epoch, opt.epoch, i, len(train_loader), loss1.data, loss2.data))
#        
    if opt.is_ResNet:
        save_path = 'models/CPD_Resnet/'
        
    else:
        save_path = r'D:\code\new_6\8_online/model_weight/adp_th_train_2_max_meanx2_nofix_noaug_adp_test_2_10/%s/%s/'%(dataset_name,video)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if (epoch) % 10 == 0:
        torch.save(model_total.state_dict(), save_path + 'vgg_%d.pth' % epoch)
print("Let's go!")
#writer = SummaryWriter(logdir='log/'+"_"+str(time.time()))
######################################################################video
#dataset_list = ['davis_test','Segtrack-v2','Visal','Easy-35','VOS_test_png']
dataset_list = ['davis_test']
for aa in range(0,len(dataset_list)):
    
    dataset_name = dataset_list[aa]
    path_txt_total = './txt_ob_patch_online/adp_th_train_2_max_meanx2_nofix_noaug_adp_test_2/%s/'%(dataset_name)
    txt_list = os.listdir(path_txt_total)
    
    for sj in range(0,len(txt_list)):
        txt_name = txt_list[sj]
        print(txt_name)
        path_txt = path_txt_total+txt_name
        
        for epoch in range(1, opt.epoch):
            adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
            train_loader_txt=get_loader_txt(path_txt, batchsize=opt.batchsize, trainsize=opt.trainsize)
            train(train_loader_txt, model_total, optimizer, epoch,dataset_name,path_txt[:-4])