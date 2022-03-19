import time
import torch
from torch.backends import cudnn
from matplotlib import colors
import os

from backbone import EfficientDetBackbone
import cv2
import numpy as np

from efficientdet.utils import BBoxTransform, ClipBoxes
from utils import preprocess, invert_affine, postprocess, STANDARD_COLORS, standard_to_bgr, get_index_label, \
    plot_one_box

compound_coef = 7
force_input_size = None  # set None to use default size

# replace this part with your project's anchor config
anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]

threshold = 0.01
iou_threshold = 0.2

use_cuda = True
use_float16 = False
cudnn.fastest = True
cudnn.benchmark = True


# FBMS-59/FBMS-Test

def display(preds, imgs, folder, name, imshow=True, imwrite=False):
    txt_path = './txt/top10/%s/' % (dataset) + folder
    if not os.path.exists(txt_path):
        os.makedirs(txt_path)
    for i in range(len(imgs)):
        print('i', i)
        if len(preds[i]['rois']) == 0:
            continue

        f = open(txt_path + '/' + name + '.txt', 'a')
        ra = min(len(preds[i]['rois']),10)
        for j in range(ra):
            x1, y1, x2, y2 = preds[i]['rois'][j].astype(np.int)
            obj = obj_list[preds[i]['class_ids'][j]]
            score = float(preds[i]['scores'][j])
            plot_one_box(imgs[i], [x1, y1, x2, y2], label=obj, score=score,
                         color=color_list[get_index_label(obj, obj_list)])
            class_id = preds[i]['class_ids'][j]
            f.write('%s,%s,%s,%s,%s,%s,%s,%s' % (str(j), x1, y1, x2, y2, class_id, obj, score) + '\n')

        if imshow:
            cv2.imshow('img', imgs[i])
            cv2.waitKey(0)

        if imwrite:
            save_path = './test/top10/%s/%s' % (dataset, folder)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            cv2.imwrite(save_path + '/' + name, imgs[i])


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
# tf bilinear interpolation is different from any other's, just make do
input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size

# davis_test
# Segtrack-v2
# Visal
# Easy-35
# VOS_test_png_gt_1

dataset_list = ['VOS_test_png']
for ss in range(0, len(dataset_list)):
    dataset = dataset_list[ss]

    total_path = r'D:/source/VSOD_dataset/%s/' % (dataset)
 
    folers = os.listdir(total_path)
    for k in range(0, len(folers)):
        folder = folers[k]

        img_path = r'%s/%s/Imgs/' % (total_path, folder)

        imgs = os.listdir(img_path)
        for i in range(0, len(imgs)):
            img_name = imgs[i]
            img_p = img_path + img_name
            # img_p是具体的某个图片
            # print('img_p',img_p)

            ori_imgs, framed_imgs, framed_metas, imgname = preprocess(img_p, img_name=img_name, max_size=input_size)

            if use_cuda:
                x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
            else:
                x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

            x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)
            # compound_coef 1536
            # num_classes 90
            # anchor_ratios [(1,1),(1.4,0.7),(0.7,1.4),(1,1.25,1.58))
            # scales
            print(compound_coef, len(obj_list), anchor_ratios, anchor_scales)
            model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                                         ratios=anchor_ratios, scales=anchor_scales)
            model.load_state_dict(torch.load(f'weights/efficientdet-d{compound_coef}.pth'))
            # model.requires_grad_(False)
            model.eval()

            if use_cuda:
                model = model.cuda()
            if use_float16:
                model = model.half()

            with torch.no_grad():
                features, regression, classification, anchors = model(x)

                regressBoxes = BBoxTransform()
                clipBoxes = ClipBoxes()

                # print('features',features.shape)
                # print('regression',regression.shape)
                # print('classification',classification.shape)
                # print('anchors',anchors.shape)

                out = postprocess(x,
                                  anchors, regression, classification,
                                  regressBoxes, clipBoxes,
                                  threshold, iou_threshold)

            print('out_value', out)
            out = invert_affine(framed_metas, out)
            display(out, ori_imgs, folder, imgname, imshow=False, imwrite=True)

            print('running speed test...')
            with torch.no_grad():
                print('test1: model inferring and postprocessing')
                print('inferring image for 10 times...')
                t1 = time.time()
                for _ in range(10):
                    _, regression, classification, anchors = model(x)

                    out = postprocess(x,
                                      anchors, regression, classification,
                                      regressBoxes, clipBoxes,
                                      threshold, iou_threshold)
                    out = invert_affine(framed_metas, out)

                t2 = time.time()
                tact_time = (t2 - t1) / 10
                print(f'{tact_time} seconds, {1 / tact_time} FPS, @batch_size 1')

            # uncomment this if you want a extreme fps test
            # print('test2: model inferring only')
            # print('inferring images for batch_size 32 for 10 times...')
            # t1 = time.time()
            # x = torch.cat([x] * 32, 0)
            # for _ in range(10):
            #     _, regression, classification, anchors = model(x)
        #
        # t2 = time.time()
        # tact_time = (t2 - t1) / 10
        # print(f'{tact_time} seconds, {32 / tact_time} FPS, @batch_size 32')
