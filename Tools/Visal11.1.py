import torch
import torch.nn.functional as F
import numpy as np
import pdb, os, argparse
from scipy import misc
import cv2
import matplotlib.pyplot as plt
import time
import os
import torchvision
from total.cpd_new.data import get_loader_my

from total.cpd_new.model.CPD_models import CPD
# from model.CPD_ResNet_models import CPD_ResNet
from total.cpd_new.data import test_dataset_my
from torch.autograd import Variable

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--is_ResNet', type=bool, default=False, help='VGG or ResNet backbone')
opt = parser.parse_args()

# davis_test
# Segtrack-v2
# Visal
# Easy-35
# VOS_test_png

# dataset_list = ['Segtrack-v2','Easy-35']
dataset_list = [ 'Visal']
pt = ''
for sj in range(0, len(dataset_list)):

    dataset_end = dataset_list[sj]
    dataset_path = r'F:\source\Visal17\35test_all_resultTo0_1/%s/%s/' % (pt,dataset_end)
    # dataset_path = r'G:\source\MyResult11\julei_8S\All_patchrgb2frame/%s/'%dataset_end
    model = CPD()

    static_model = torch.load(r'F:\SJ\NEW_6\8_online\model_weight\seelct_10k_davis_patch_pre_e5_b10/vgg_19.pth')
    #    static_model = torch.load(r'./premodel/CPD.pth')
    model.load_state_dict(static_model)

    model.cuda()
    model.eval()

    test_folders = os.listdir(dataset_path)

    for i in range(0, len(test_folders)):
        dataset = test_folders[i]
        if opt.is_ResNet:
            save_path = r''
            # save_path = r'D:\code\new\8_patch_seg\res18_test_train_all_1_nofix_19\select_10k_davis_train_pre_e5\ms_rank_img_spatial_all_0.7/top13/19epoch/Segtrack-v2\1/' + dataset + '/'
        else:
            # save_path = r'D:\code\new_6\5_bboxtoseg\seelct_10k_davis_patch_pre_e5_b10/'+dataset_end+'/calss2/' + dataset + '/'
            save_path = r'F:\source\Visal17\35all_patch_saliency/%s/%s/%s/' % (pt,dataset_end, dataset)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        image_root = dataset_path + dataset + '/1/'
        if not os.path.exists(image_root):
            print('%s不存在'%image_root)
            continue
        test_loader = test_dataset_my(image_root, opt.testsize)
        names = []
        for i in range(test_loader.size):
            image, name, index, shape = test_loader.load_data()

            image = image.cuda()
            #
            star = time.time()
            a_s, d_s = model(image)
            end = time.time()
            runtime = end - star
            # print('runtime',runtime)
            #            print('res',ress.shape)
            #            res=torchvision.utils.make_grid(res).numpy()
            h = shape[0]
            w = shape[1]

            # mask=mask.unsqueeze(0)
            res = F.upsample(d_s, size=[h, w], mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            #                plt.imshow(res)
            #                plt.show()
            #                print('res',res.shape)
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            #                print('res',res)
            #                plt.imshow(res)
            #                plt.show()
            #                misc.imsave(save_path+names[i], res)
            #                print(names[i+h-4])
            cv2.imwrite(save_path + name, res * 255)
            end = time.time()

#    train_loader = get_loader_my(image_root, batchsize=5, trainsize=352)
#    for i, pack in enumerate(train_loader, start=1):
#
#        images, gts = pack
#        #[5,3,352,352]
#        print('images',images.shape)
#        images = Variable(images)
#        images = images.cuda()
#        _, ress = model(images)
##        end=time.time()
##        runtime=end-star
##        print('runtime',runtime)
#        for j in range(5):
#            res = F.upsample(ress[j], size=[540,680], mode='bilinear', align_corners=False)
#            res = res.sigmoid().data.cpu().numpy().squeeze()
#
#            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
#            print('res',res)
##            misc.imsave(save_path+name, res)
#            cv2.imwrite(save_path+ int(i+j)+'.png', res*255)
#            end=time.time()