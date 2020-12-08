from __future__ import print_function
from __future__ import division
import torch
import time
from tqdm import tqdm

from libs.logger import AverageMeter
from libs.segmentation_utils import accuracy, intersectionAndUnion


def adjust_learning_rate(optimizers, cur_iter, cfg):
    
    scale_running_lr = ((1. - float(cur_iter) / cfg.TRAIN.max_iters) ** cfg.TRAIN.lr_pow)
    
    for i, optimizer in enumerate(optimizers):
        cfg.TRAIN.running_lr[i] = cfg.TRAIN.lr[i] * scale_running_lr   
        for param_group in optimizer.param_groups:
            param_group['lr'] = cfg.TRAIN.running_lr[i]
    
    
    # if isinstance(optimizers, list): 
    #     scale_running_lr = ((1. - float(cur_iter) / cfg.TRAIN.max_iters) ** cfg.TRAIN.lr_pow)
    #     cfg.TRAIN.running_lr[0] = cfg.TRAIN.lr[0] * scale_running_lr
    #     cfg.TRAIN.running_lr[1] = cfg.TRAIN.lr[1] * scale_running_lr

    #     (optimizer_encoder, optimizer_decoder) = optimizers
    #     for param_group in optimizer_encoder.param_groups:
    #         param_group['lr'] = cfg.TRAIN.running_lr_encoder
    #     for param_group in optimizer_decoder.param_groups:
    #         param_group['lr'] = cfg.TRAIN.running_lr_decoder
    # else:
    #     scale_running_lr = ((1. - float(cur_iter) / cfg.TRAIN.max_iters) ** cfg.TRAIN.lr_pow)
    #     cfg.TRAIN.running_lr_encoder = cfg.TRAIN.lr[0] * scale_running_lr
    #     for param_group in optimizers.param_groups:
    #         param_group['lr'] = cfg.TRAIN.running_lr[0]
        
        
def train(model, data, optimizers, logger, epoch, cfg, vis_data=None): 
    #batch_time = AverageMeter()
    #data_time = AverageMeter()
    ave_total_loss = AverageMeter()
    ave_acc = AverageMeter()
    
    epoch_time = time.time()
    
    if vis_data is not None:
        model.eval()
        with torch.no_grad():
            for i, (inputs, pil_image, labels, seg_image) in enumerate(vis_data):
                #img, segmap, inp = data.dataset.__loaddata__(1, rgb=True,test=True)
                segSize = (pil_image[0].size[1], pil_image[0].size[0])
                pred = model(inputs, segSize=segSize)
                pred = data.dataset.transform.__reverse__(pred)
                pred_color = data.dataset.__tomap__(pred.squeeze(), shift_index=False)
                logger.logImage(pil_image[0], seg_image[0], pred_color, epoch,  train=False)
        
    model.train(not cfg.TRAIN.fix_bn)
    pbar = tqdm(total=len(data))
    
    for i, inputs in enumerate(data):
        # load a batch of data
        model.zero_grad()
        
        # adjust learning rate
        cur_iter = i + (epoch - 1) * cfg.TRAIN.epoch_iters
        adjust_learning_rate(optimizers, cur_iter, cfg)

        # forward pass
        loss, acc = model(inputs[0].cuda(), inputs[1].cuda())#,edge=inputs[2].cuda())
        loss = loss.mean()
        acc = acc.mean()
        
        # Backward
        loss.backward()
        
#        if isinstance(optimizers, tuple): 
        for optimizer in optimizers:
            optimizer.step()
#        else:
#            optimizers.step()

        ave_total_loss.update(loss.data.item())
        ave_acc.update(acc.data.item()*100)
        pbar.update(1)

         
    epoch_time = time.time() - epoch_time
    print('Epoch complete in {:.0f}m {:.0f}s'.format(epoch_time // 60, epoch_time % 60)) 
    logger.logScalar(ave_total_loss.average(), "train/loss", epoch) 
    logger.logScalar(ave_acc.average(),"train/accuracy", epoch)
    
    
def evaluate(model, data, logger, cfg):
    acc_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    time_meter = AverageMeter()

    
    model.eval()
    with torch.no_grad():
        pbar = tqdm(total=len(data))
    
        for i, (inputs, pil_image, labels, seg_image) in enumerate(data):
            #img, segmap, inp = data.dataset.__loaddata__(1, rgb=True,test=True)
            segSize = (pil_image[0].size[1], pil_image[0].size[0])
            tic = time.perf_counter()
            pred = model(inputs, segSize=segSize)
            time_meter.update(time.perf_counter() - tic)
            pred_color = data.dataset.__tomap__(pred.squeeze(), shift_index=False)
            logger.saveImage(pil_image[0], pred_color, i, blend=True, concat=True)
            logger.logImage(pil_image[0], seg_image[0], pred_color, i,  train=False)
            

            # calculate accuracy
            acc, pix = accuracy(pred, labels[0])
            intersection, union = intersectionAndUnion(pred,  labels[0], cfg.DATASET.num_class)
            acc_meter.update(acc, pix)
            intersection_meter.update(intersection)
            union_meter.update(union)

            pbar.update(1)
    #logger.logScalar(ave_total_loss.average(), "val/loss", epoch)
    logger.logScalar(acc_meter.average(),"val/accuracy", 1)
    # summary
    print(acc_meter.average())
    iou = intersection_meter.sum / (union_meter.sum + 1e-10)
    for i, _iou in enumerate(iou):
        print('class [{}], IoU: {:.4f}'.format(data.dataset.class_names[i], _iou))
    
    print('[Eval Summary]:')
    print('Mean IoU: {:.4f}, Accuracy: {:.2f}%, Inference Time: {:.4f}s'
          .format(iou.mean(), acc_meter.average()*100, time_meter.average()))

    
    return {"IOU" : iou, "PIXEL": acc_meter.average()}
    
    
def test(model, data, logger, cfg):

    model.eval()
    with torch.no_grad():
        pbar = tqdm(total=len(data))
    
        for i, (inputs, pil_image) in enumerate(data):
            #img, segmap, inp = data.dataset.__loaddata__(1, rgb=True,test=True)
            segSize = (pil_image[0].size[1], pil_image[0].size[0])
            pred = model(inputs, segSize=segSize)
            pred_color = data.dataset.__tomap__(pred.squeeze(), shift_index=False)
            logger.saveImage(pil_image[0], pred_color, i, blend=True, concat=True)


            pbar.update(1)
    #logger.saveVideo("test")
    