#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 11:53:54 2019

@author: gary
"""
from PIL import Image, ImageFilter
import numpy as np
import torch
import random
import torchvision.transforms.functional as TF
from torchvision import transforms

#An object wrapper to provide methods to for common image transform activites
class ImageTransformTools(object):
    def __init__(self):
        
        return 
        
    def imresize(self, im, size, interp='bilinear'):
        if interp == 'nearest':
            resample = Image.NEAREST
        elif interp == 'bilinear':
            resample = Image.BILINEAR
        elif interp == 'bicubic':
            resample = Image.BICUBIC
        else:
            raise Exception('resample method undefined!')
        return im.resize(size, resample)
    
    def imnormalize(self,im, norm_transform):         
        # 0-255 to 0-1
        im = np.float32(np.array(im)) / 255.
        im = im.transpose((2, 0, 1))
        im = norm_transform(torch.from_numpy(im.copy()))
       
        return im

    def horizonal_flip(self, img, segm=None):
        if random.random() > 0.5:
            img = TF.hflip(img)
            if segm is not None:
                segm = TF.hflip(segm)
                return img, segm
            else:
                return img
        else:
            if segm is not None:
                return img, segm
            else:
                return img
            
    def random_crop(self,size, img, segm=None):
        
        #Check image is not smaller than crop size 
        if img.size[0] < size[0]:
            size_w = img.size[0]
        else:
            size_w = size[0]
        if img.size[1] < size[1]:
            size_h = img.size[1]
        else:
            size_h = size[1]
        
        
        if random.random() > 0.5: 
            # Random crop
            i, j, h, w = transforms.RandomCrop.get_params(
                    img, output_size=(size_h,size_w))
            img = TF.crop(img, i, j, h, w)
            #img.save("test.png")
            if segm is not None:
                segm = TF.crop(img, i, j, h, w)
                return img, segm
            else:
                return img
        else:
            if segm is not None:
                return img, segm
            else:
                return img
            
    def guassian_blur(self, img, radius):
        radius = np.random.uniform(radius[0], radius[1])
        return img.filter(ImageFilter.GaussianBlur(radius))
    
    # Round x to the nearest multiple of p and x' >= x
    def round2nearest_multiple(self, x, p):
        return ((x - 1) // p + 1) * p
                
    def label_rgb_to_grayscale(self, segm, class_names, class_rgb_values, pil=True):
        segm = np.array(segm)
        new_segm = np.zeros((segm.shape[0], segm.shape[1]))
        
        for i, slab in enumerate(class_rgb_values):  
            color = (int(slab[0]), int(slab[1]), int(slab[2]))
            y,x = np.where(np.all(segm == color, axis=-1))
            new_segm[y,x] = i
        
        if pil:
            new_segm = Image.fromarray(np.uint8(new_segm))
        
        return new_segm
    
    def get_edge_map(self,img):
       return img.filter(ImageFilter.FIND_EDGES).convert('L')
       
    def edge_map_to_numpy(self,img, resize):
       img = self.imresize(img,resize)
       img = np.array(img).astype('float64')
       img[0,:] = 0
       img[:,0] = 0
       img[:,img.shape[1]-1] = 0  
       img[img.shape[0]-1,:] = 0
       img = (img / 255.) + 1
       
       return img 
