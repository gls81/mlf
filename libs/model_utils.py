# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 10:31:27 2020

@author: Gary
"""

import os
import torch
import torch.optim as optim
import torch.nn as nn
from libs.models.semantic_segmentation_models import ModelBuilder, SegmentationModule, SegmentationSingleModule
from libs.data.segmentation_data import SegmentationTrainDataset, SegmentationValidationDataset, SegmentationTestDataset
from libs.data.collators import SegmentationTrainCollator, SegmentationValidationCollator, SegmentationTestCollator

def check_model(cfg):
    
    #To determine if the model type is in fitting with the given components
    if cfg.MODEL.type == "SS":
        if len(cfg.MODEL.components) == 2: 
            assert cfg.MODEL.components_type == ["encoder","decoder"], "Model components types are not compatible!"
            if not cfg.MODEL.components[1].endswith('deepsup') and not cfg.MODE.upper() == "TRAIN":
                cfg.TRAIN.deep_sup_scale = None
        elif len(cfg.MODEL.components) == 1:
            assert cfg.MODEL.components_type == ["encoder-decoder"], "Model components types are not compatible!"
            
        cfg.DATASET.dataset_classes = {"TRAIN": SegmentationTrainDataset, "VAL": SegmentationValidationDataset, "TEST": SegmentationTestDataset, "TRAINVIS": SegmentationValidationDataset}
        cfg.DATASET.collator_classes = {"TRAIN": SegmentationTrainCollator, "VAL": SegmentationValidationCollator, "TEST": SegmentationTestCollator, "TRAINVIS": SegmentationValidationCollator}
            
            
            
    if cfg.MODEL.type == "GAN":
        assert len(cfg.MODEL.components) <=2, "Requres at least two components in the GAN model!"       
        if len(cfg.MODEL.components) == 2: 
            assert cfg.MODEL.components_type == ["discriminator","generator"], "Model components types are not compatible!"
                
        cfg.DATASET.dataset_classes = {"TRAIN": GANTrainDataset, "VAL": GANValidationDataset, "TEST": GANTestDataset, "TRAINVIS": GANValidationDataset}
        cfg.DATASET.collator_classes = {"TRAIN": GANTrainCollator, "VAL": GANValidationCollator, "TEST": GANTestCollator, "TRAINVIS": GANValidationCollator}
            
            
    return cfg
            

def get_criterion(name, ignore_index=-100):
    if name.upper() == "NLLLOSS":
        return nn.NLLLoss(ignore_index=ignore_index), False
    if name.upper() == "BCELOG":
        return nn.BCEWithLogitsLoss(), True
    
def group_weight(module):
    group_decay = []
    group_no_decay = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.conv._ConvNd):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.batchnorm._BatchNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)

    assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
    groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
    return groups

def create_optimizers_old(nets, cfg):
    
    if len(nets) == 2:
        (net, crit) = nets
        # params_to_update = net.parameters()
        # print("Params to learn:")

        # for name,param in net.named_parameters():
        #     if param.requires_grad == True:
        #         print("\t",name)
    
        optimizer = torch.optim.SGD(
            group_weight(net),
            lr=cfg.TRAIN.lr_encoder,
            momentum=cfg.TRAIN.beta1,
            weight_decay=cfg.TRAIN.weight_decay)
        
        return (optimizer)

    elif len(nets) == 3:
        (net_encoder, net_decoder, crit) = nets


        optimizer_encoder = torch.optim.SGD(
            group_weight(net_encoder),
            lr=cfg.TRAIN.lr_encoder,
            momentum=cfg.TRAIN.beta1,
            weight_decay=cfg.TRAIN.weight_decay)
        optimizer_decoder = torch.optim.SGD(
            group_weight(net_decoder),
            lr=cfg.TRAIN.lr_decoder,
            momentum=cfg.TRAIN.beta1,
            weight_decay=cfg.TRAIN.weight_decay)
        return (optimizer_encoder, optimizer_decoder)
    
    
def create_optimizers(net, cfg, index):
    
    optimizer = torch.optim.SGD(
            group_weight(net),
            lr=cfg.TRAIN.lr[index],
            momentum=cfg.TRAIN.beta1,
            weight_decay=cfg.TRAIN.weight_decay)
        
    return optimizer



def checkpoint(nets, history, cfg, epoch):
    print('Saving checkpoints...')
    
    (components, crit) = nets
 
    #torch.save(history,'{}/history_epoch_{}.pth'.format(cfg.DIR, epoch)) 
    for i, com in enumerate(components):
        print(i)
        torch.save(com.state_dict(),'{}/{}_{}_epoch_{}.pth'.format(cfg.DIR.ckpts,i,cfg.MODEL.components_type[i], epoch))
        
    #     dict_encoder = net_encoder.state_dict()


    #         dict_encoder,
    #         '{}/encoder_epoch_{}.pth'.format(cfg.DIR.ckpts, epoch))
    #     torch.save( 
    #         dict_decoder,
    #         '{}/decoder_epoch_{}.pth'.format(cfg.DIR.ckpts, epoch))
    
    # if len(nets) == 2:
    #     (net_encoder, crit) = nets
    #     dict_encoder = net_encoder.state_dict()
    #     #torch.save(
    #     #    history,
    #     #    '{}/history_epoch_{}.pth'.format(cfg.DIR, epoch))
    #     torch.save(
    #         dict_encoder,
    #         '{}/encoder_epoch_{}.pth'.format(cfg.DIR.ckpts, epoch))


    # elif len(nets) == 3:
    #     (net_encoder, net_decoder, crit) = nets
    #     dict_encoder = net_encoder.state_dict()
    #     dict_decoder = net_decoder.state_dict()
    #     #torch.save(
    #     #    history,
    #     #    '{}/history_epoch_{}.pth'.format(cfg.DIR, epoch)) 
    #     torch.save(
    #         dict_encoder,
    #         '{}/encoder_epoch_{}.pth'.format(cfg.DIR.ckpts, epoch))
    #     torch.save( 
    #         dict_decoder,
    #         '{}/decoder_epoch_{}.pth'.format(cfg.DIR.ckpts, epoch))
        
    return
    
    
def load_model(cfg):
    #Initalise the model to do this the best way would be to use a common interface build model
    #If we have the same encoder and decoder the model is a single arch else it is two modules, may need a slicker way to do this i.e parts or something
    if cfg.MODEL.arch_encoder == cfg.MODEL.arch_decoder:
        model, optimizers, nets = load_single_model(cfg)
    else:
        model, optimizers, nets = load_encoder_decoder_model(cfg)

    return model, optimizers, nets

def build_model(cfg):
    component_classes = {"encoder": ModelBuilder.build_encoder, "decoder": ModelBuilder.build_decoder}
    total_components = len(cfg.MODEL.components)
    components = [None] * total_components
    optimizers = [None] * total_components
    
    
    criterion, one_hot = get_criterion(cfg.TRAIN.criterion, ignore_index=cfg.TRAIN.criterion_ignore_index)
    
    for i, component in enumerate(cfg.MODEL.components):
        cfg.MODEL.components_weights[i] = get_weights_path(cfg, i)
        components[i] = component_classes[cfg.MODEL.components_type[i]](arch=component,fc_dim=cfg.MODEL.components_fc_dim[0],weights=cfg.MODEL.components_weights[i])
        
        if cfg.MODE.upper() == "TRAIN":
            optimizers[i] = create_optimizers(components[i], cfg, i)
    
    nets = (components, criterion)
    
    if total_components == 2:
        model =  SegmentationModule(components[0], components[1], criterion, cfg.TRAIN.deep_sup_scale)
    
    model.cuda()
    
    return model, optimizers, nets




def get_weights_path(cfg, index):
    if cfg.PARAMS.checkpoint > 0:
        weights_path = os.path.join(cfg.DIR.ckpts, '{}_{}_epoch_{}.pth'.format(index,cfg.MODEL.components_type[index],cfg.PARAMS.checkpoint))
        assert os.path.exists(weights_path), "checkpoint does not exitst!"
        return weights_path
    else: # Train from scratch or a pretrained model
        return ""


def load_single_model(cfg):
    if cfg.MODE.upper() == "TRAIN":
        if cfg.TRAIN.start_epoch > 0:
            cfg.MODEL.weights_encoder = os.path.join(
                cfg.DIR.ckpts, 'encoder_epoch_{}.pth'.format(cfg.TRAIN.start_epoch))
            assert os.path.exists(cfg.MODEL.weights_encoder), "checkpoint does not exitst!"
        softmax=False
    elif cfg.MODE.upper() == "VAL":
        cfg.MODEL.weights_encoder = os.path.join(
            cfg.DIR.ckpts, 'encoder_' + cfg.VAL.checkpoint)
        softmax=False
    elif cfg.MODE.upper() == "TEST":
        cfg.MODEL.weights_encoder = os.path.join(
            cfg.DIR.ckpts, 'encoder_' + cfg.TEST.checkpoint)
        softmax=True
    
    criterion, one_hot = get_criterion(cfg.TRAIN.criterion, ignore_index=cfg.TRAIN.criterion_ignore_index)
    
  
    net_all = ModelBuilder.build_single(
        arch=cfg.MODEL.arch_decoder,
        fc_dim=cfg.MODEL.fc_dim,
        num_class=cfg.DATASET.num_class,
        weights=cfg.MODEL.weights_decoder,
        use_softmax=softmax)
  
    
    nets = (net_all, criterion)
    
    if cfg.MODE.upper() == "TRAIN":
        optimizers = create_optimizers(nets, cfg)
    else:
        optimizers = None
    

    model = SegmentationSingleModule(
            net_all, criterion, one_hot=one_hot)
        
    model.cuda()
    
    return model, optimizers, nets


 
def load_encoder_decoder_model(cfg):
    if cfg.MODE.upper() == "TRAIN":
        if cfg.TRAIN.start_epoch > 0:
            cfg.MODEL.weights_encoder = os.path.join(
                cfg.DIR.ckpts, 'encoder_epoch_{}.pth'.format(cfg.TRAIN.start_epoch))
            cfg.MODEL.weights_decoder = os.path.join(
                cfg.DIR.ckpts, 'decoder_epoch_{}.pth'.format(cfg.TRAIN.start_epoch))
            assert os.path.exists(cfg.MODEL.weights_encoder) and \
                os.path.exists(cfg.MODEL.weights_decoder), "checkpoint does not exitst!"
        softmax=False
    elif cfg.MODE.upper() == "VAL":
        cfg.MODEL.weights_encoder = os.path.join(
            cfg.DIR.ckpts, 'encoder_' + cfg.VAL.checkpoint)
        cfg.MODEL.weights_decoder = os.path.join(
            cfg.DIR.ckpts, 'decoder_' + cfg.VAL.checkpoint)
        softmax=False
    elif cfg.MODE.upper() == "TEST":
        cfg.MODEL.weights_encoder = os.path.join(
            cfg.DIR.ckpts, 'encoder_' + cfg.TEST.checkpoint)
        cfg.MODEL.weights_decoder = os.path.join(
            cfg.DIR.ckpts, 'decoder_' + cfg.TEST.checkpoint)
        softmax=True
    
  
    net_encoder = ModelBuilder.build_encoder(
        arch=cfg.MODEL.arch_encoder,
        fc_dim=cfg.MODEL.fc_dim,
        weights=cfg.MODEL.weights_encoder)
    net_decoder = ModelBuilder.build_decoder(
        arch=cfg.MODEL.arch_decoder,
        fc_dim=cfg.MODEL.fc_dim,
        num_class=cfg.DATASET.num_class,
        weights=cfg.MODEL.weights_decoder,
        use_softmax=softmax)

    criterion, one_hot = get_criterion(cfg.TRAIN.criterion, ignore_index=cfg.TRAIN.criterion_ignore_index)
  
    
    if cfg.MODE.upper() == "TRAIN":
        optimizers = create_optimizers(nets, cfg)
    else:
        optimizers = None
    
    if cfg.MODEL.arch_decoder.endswith('deepsup') and cfg.MODE.upper() == "TRAIN":
        model =  (
            net_encoder, net_decoder, criterion, cfg.TRAIN.deep_sup_scale)
    else:
        model = SegmentationModule(
            net_encoder, net_decoder, criterion)
        
    model.cuda()
    
    return model, optimizers, nets