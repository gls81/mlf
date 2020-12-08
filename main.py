#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 10:57:33 2019

@author: gary
    """
import argparse
import torch
from torch.utils.data import DataLoader
import os
import random
import datetime
import csv

from libs.config_defaults import _C as cfg
from libs.data.dataset_builder import dataset_factory
from libs.train import train, evaluate, test
from libs.logger import trainingLogger
from libs.model_utils import check_model,build_model, checkpoint


def get_unique_name(cfg):
  
    #Build save details from config
    separator = '-'
    model_name = separator.join(cfg.MODEL.components)
    
    if len(cfg.DATASET.train) == 1:
        unique_name = cfg.DATASET.train[0] + "-" + cfg.NAME
    else:
        seperator = '_'
        unique_name = seperator.join(cfg.DATASET.train) + "-" + cfg.NAME 

    cfg.DIR.logs =  os.path.join(cfg.DIR.parent, cfg.DIR.logs, "-".join([model_name,unique_name]))
    
    cfg.DIR.output =  os.path.join(cfg.DIR.parent, cfg.DIR.output, model_name, unique_name)
    cfg.DIR.ckpts =  os.path.join(cfg.DIR.output,cfg.DIR.ckpts)
    cfg.DIR.val_metrics =  os.path.join(cfg.DIR.parent, cfg.DIR.output,cfg.DIR.val_metrics)
    cfg.DIR.vis =  os.path.join(cfg.DIR.output,cfg.DIR.vis)
    return cfg


####################################################################################################################
# Main Code
####################################################################################################################

parser = argparse.ArgumentParser(description="PyTorch Semantic Segmentation Training")
parser.add_argument(
    "--cfg",
    default="test_seg.yaml",
    metavar="FILE",
    help="path to config file",
    type=str,
)
parser.add_argument(
    
    "opts",
    help="Modify config options using the command-line",
    default=None,
    nargs=argparse.REMAINDER,
)

args = parser.parse_args()
args.cfg = os.path.join("data","config", args.cfg)

#Load Config from file and extra command line
cfg.merge_from_file(args.cfg) #From yacs 
cfg.merge_from_list(args.opts)

#Set parameters dependant upon mode
if cfg.MODE.upper() == "TRAIN":
    cfg.PARAMS = cfg.TRAIN
elif cfg.MODE.upper() == "EVAL":
    cfg.PARAMS = cfg.VAL
elif cfg.MODE.upper() == "TEST":
    cfg.PARAMS = cfg.TEST
     
cfg = get_unique_name(cfg)

# Output directory for experimnet
if not os.path.isdir(cfg.DIR.output):
   os.makedirs(cfg.DIR.output)
with open(os.path.join(cfg.DIR.output, 'config.yaml'), 'w') as f:
    f.write("{}".format(cfg))
if not os.path.isdir(cfg.DIR.ckpts):
   os.makedirs(cfg.DIR.ckpts)

#In cases classes are ignored 
if cfg.TRAIN.criterion_ignore_index != -100:
    cfg.DATASET.shift_class_labels = abs(cfg.TRAIN.criterion_ignore_index)
else:
    cfg.DATASET.shift_class_labels = 0
    
cfg = check_model(cfg)
    
    

if cfg.MODE.upper() == "TRAIN":
    
    dataset, collator = dataset_factory(cfg.DATASET.train,cfg.MODE.upper(),cfg.DIR.datasets,cfg.DATASET,trainvis=True)
    dataloader_train = DataLoader(dataset["TRAIN"], batch_size=cfg.TRAIN.batch_size_per_gpu, shuffle=True, num_workers=0, collate_fn=collator["TRAIN"])
    dataloader_vis = DataLoader(dataset["TRAINVIS"], batch_size=1, shuffle=False, num_workers=0, collate_fn=collator["TRAINVIS"])
    cfg.DATASET.num_class = dataset["TRAIN"].class_num
    
    
    random.seed(cfg.TRAIN.seed)
    torch.manual_seed(cfg.TRAIN.seed)
    #Setup logging
    history = {'train': {'epoch': [], 'loss': [], 'acc': []}} 
    
    #time_stamp = datetime.datetime.now()
    #time_stamp = time_stamp.strftime("%m_%d_%y_%H_%M_%S")
    #log_folder = os.getcwd() +"/logs/"+ dataset_name + "_" + time_stamp
    logger = trainingLogger(dataset["TRAIN"].name,logs_dir=cfg.DIR.logs, img_dir=cfg.DIR.vis)
    model, optimizers, nets = build_model(cfg)
    
    cfg.TRAIN.max_iters = cfg.TRAIN.epoch_iters * cfg.TRAIN.num_epoch
    cfg.TRAIN.running_lr = [None] * len(optimizers)
    for i in range(len(optimizers)):
        cfg.TRAIN.running_lr[i] = cfg.TRAIN.lr[i]

    for epoch in range(cfg.TRAIN.checkpoint, cfg.TRAIN.num_epoch):
        train(model, dataloader_train, optimizers, logger, epoch+1, cfg, vis_data=dataloader_vis)
        # checkpointing
        checkpoint(nets, history, cfg, epoch+1)
        
  
if cfg.MODE.upper() == "EVAL":
    for ds in cfg.DATASET.val:
        dataset, collator = dataset_factory(ds,cfg.MODE.upper(),cfg.DIR.datasets,cfg.DATASET)
        dataloader = DataLoader(dataset["VAL"], batch_size=1, shuffle=False, num_workers=0, collate_fn=collator["VAL"])
        cfg.DATASET.num_class = dataset["VAL"].class_num
        logger = trainingLogger(dataset["VAL"].name,logs_dir=cfg.DIR.logs, img_dir=cfg.DIR.vis)
        model, optimizers, nets = build_model(cfg)
        results = evaluate(model, dataloader, logger, cfg)
        
        RESULTS2 = results["IOU"].tolist()

        with open('testres.csv','w', newline='') as result_file:
            wr = csv.writer(result_file, dialect='excel')
            wr.writerows(RESULTS2)
     
if cfg.MODE.upper() == "TEST":   
    for ds in cfg.DATASET.test:
        dataset, collator = dataset_factory(ds,cfg.MODE.upper(),cfg.DIR.datasets,cfg.DATASET)
        dataloader = DataLoader(dataset["TEST"], batch_size=1, shuffle=False, num_workers=0, collate_fn=collator["TEST"])
        cfg.DATASET.num_class = dataset["TEST"].class_num
        logger = trainingLogger(dataset["TEST"].name,logs_dir=cfg.DIR.logs, img_dir=cfg.DIR.vis)
        model, optimizers, nets = build_model(cfg)
        test(model, dataloader, logger, cfg)




       
        

