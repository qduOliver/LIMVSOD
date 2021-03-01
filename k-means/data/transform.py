# -*- coding:utf-8 -*-
# @time :2020.02.09
# @IDE : pycharm
# @author :lxztju
# @github : https://github.com/lxztju


import random
import math
import torch
import numpy as np
import cv2

from PIL import Image, ImageOps, ImageFilter
from torchvision import transforms

class Resize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        # padding
        ratio = self.size[0] / self.size[1]
        w, h = img.size
        if w / h < ratio:
            t = int(h * ratio)
            w_padding = (t - w) // 2
            img = img.crop((-w_padding, 0, w+w_padding, h))
        else:
            t = int(w / ratio)
            h_padding = (t - h) // 2
            img = img.crop((0, -h_padding, w, h+h_padding))

        img = img.resize(self.size, self.interpolation)

        return img

def resize(self, img,trainsize):
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR)
        else:
            return img

def img_pad_0(img, inp_dim):
    '''resize image with unchanged aspect ratio using padding'''
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w / img_w, h / img_h))
    new_h = int(img_h * min(w / img_w, h / img_h))
    resized_image = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)
    canvas[(h - new_h) // 2:(h - new_h) // 2 + new_h, (w - new_w) // 2:(w - new_w) // 2 + new_w, :] = resized_image
    return canvas
        
class RandomRotate(object):
    def __init__(self, degree, p=0.5):
        self.degree = degree
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            rotate_degree = random.uniform(-1*self.degree, self.degree)
            img = img.rotate(rotate_degree, Image.BILINEAR)
        return img

class RandomGaussianBlur(object):
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, img):
        if random.random() < self.p:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))
        return img



mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
def get_train_transform(mean=mean, std=std, size=0):
    train_transform = transforms.Compose([
        Resize((int(size * (256 / 224)), int(size * (256 / 224)))),
        transforms.RandomCrop(size),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        RandomRotate(15, 0.3),
        # RandomGaussianBlur(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return train_transform

def get_test_transform(mean=mean, std=std, size=0):
    return transforms.Compose([
        transforms.Resize((size, size)),
        # transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
def get_yuan_transform(mean=mean, std=std, size=0):
    return transforms.Compose([
        transforms.Resize((size, size)),
        # transforms.CenterCrop(size),
        transforms.ToTensor(),
        # transforms.Normalize(mean=mean, std=std),
    ])

def tta_test_transform(mean=mean, std=std, size=0):
    return transforms.Compose([
        Resize((int(size * (256 / 224)), int(size * (256 / 224)))),
        # transforms.CenterCrop(size),
        transforms.RandomCrop(size),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        RandomRotate(15, 0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
