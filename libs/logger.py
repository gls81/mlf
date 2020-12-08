# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 17:43:19 2020

@author: Gary
"""
import datetime
import os
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
#import cv2

class trainingLogger(object):
    def __init__(self, dataset_name, logs_dir="logs", img_dir="vis", add_time_stamp=False):
        
        self.dataset_name = dataset_name
        
        if add_time_stamp:
            time_stamp = datetime.datetime.now()
            time_stamp = time_stamp.strftime("%m_%d_%y_%H_%M_%S")
            self.log_dir = logs_dir + "__" + time_stamp
            self.img_dir = img_dir + "__" + time_stamp
        else: 
           self.log_dir = logs_dir
           self.img_dir = os.path.join(img_dir,self.dataset_name)
    
        
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
        self.frames = []
        
        if not os.path.isdir(self.img_dir):
            os.makedirs(self.img_dir)
        
    def logImage(self, img,segmap,pred,stage, train=True):
        images_to_stack = []
        
        images_to_stack.append(TF.to_tensor(img))
        images_to_stack.append(TF.to_tensor(segmap))
        images_to_stack.append(TF.to_tensor(pred))
        stack = torch.stack(images_to_stack)
        
        if train:
            self.writer.add_images('Image/train',stack ,stage)
        else:
            self.writer.add_images('Image/validation',stack ,stage)  
        
        
    def saveImage(self,img,pred,name,seg=True,blend=False,concat=False):
        # aggregate images and save
        
        if seg:
            seg_path = os.path.join(self.img_dir, "segmentation")
            if not os.path.isdir(seg_path):
                os.makedirs(seg_path)
            
            seg_image = Image.fromarray(pred)
            seg_image.save(os.path.join(seg_path, (str(name).zfill(4) + ".png")))
            
        if blend:
            blend_path = os.path.join(self.img_dir, "blend")
            if not os.path.isdir(blend_path):
                os.makedirs(blend_path)
            seg_image = Image.fromarray(pred)
            im_vis = Image.blend(img,seg_image, alpha=0.5)
            im_vis.save(os.path.join(blend_path, (str(name).zfill(4) + ".png")))
            
        if concat:
            concat_path = os.path.join(self.img_dir, "concat")
            if not os.path.isdir(concat_path):
                os.makedirs(concat_path)
            im_vis = np.concatenate((img, pred), axis=1)
            Image.fromarray(im_vis).save(
                os.path.join(concat_path, (str(name).zfill(4) + ".png")))
        
        self.frames.append(np.array(im_vis))
        
        return
            
    def saveVideo(self, source="blend"):
        
        image_folder = os.path.join(self.img_dir, "blend")
        
        images_list = glob.glob(image_folder +'/*.png')
        test = Image.open(images_list[0])
        size = test.size
        output = os.path.join(self.img_dir, (source + '.avi'))
        fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
        out = cv2.VideoWriter(output,fourcc, 5, size)
        for i in range(len(images_list)):
            img = Image.open(images_list[i])
            img_rgb = np.asarray(img)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
            out.write(img_bgr)
        out.release()

        # # aggregate images and save
        # size = (1280,720)
        # output = 'test.avi'
        # fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
        # #fourcc = cv2.VideoWriter_fourcc('M','S','V','C')
        # #fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
        # out = cv2.VideoWriter(output,fourcc, 5, size)
        # for i in range(len(self.frames)):
        #     out.write(self.frames[i])
        # out.release()

    def logScalar(self,value,text,stage):  
         self.writer.add_scalar(text, value, stage)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg
