#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 12:15:18 2019

@author: gary
"""
import numpy as np
from PIL import Image

def tensor_convertor(x):

    if x.is_cuda:
        x = x.cpu()
    if x.requires_grad:
        x = x.detach()

    return x.numpy()


def seg_pred_to_image(seg, label_values):
    #one_hot_label = one_hot_label.float()
    seg = tensor_convertor(seg)

    one_hot = reverse_one_hot(seg)
    image = colour_code_segmentation(one_hot, label_values)

    return image, one_hot


def one_hot_it(label, label_values):
    if label.mode == 'RGB':
        semantic_map = one_hot_it_rgb(label, label_values) 
    elif label.mode == 'L':
        semantic_map = one_hot_it_gs(label, label_values) 
            
    return semantic_map

def one_hot_it_rgb(label, label_values):
    semantic_map = []
    for colour in label_values:
        # colour_map = np.full((label.shape[0], label.shape[1], label.shape[2]), colour, dtype=int)
        equality = np.equal(label, colour)
        class_map = np.all(equality, axis = -1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)
    # print("Time 2 = ", time.time() - st)
    return semantic_map
    
    
def one_hot_it_gs(label, label_values):
    semantic_map = []
    for ind, name in enumerate(label_values):
        # colour_map = np.full((label.shape[0], label.shape[1], label.shape[2]), colour, dtype=int)
        class_map = np.equal(label, ind)
        #class_map = np.all(equality, axis = -1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)
    # print("Time 2 = ", time.time() - st)
    return semantic_map

def reverse_one_hot(image):
    """
    Transform a 2D array in one-hot format (depth is num_classes),
    to a 2D array with only 1 channel, where each pixel value is
    the classified class key.

    # Arguments
    image: The one-hot format image 
        
    # Returns
    A 2D array with the same width and hieght as the input, but
    with a depth size of 1, where each pixel value is the classified 
    class key.
    """
    x = np.argmax(image, axis = -1)
    return x

def colour_code_segmentation(image,label_values):
    """
    Given a 1-channel array of class keys, colour code the segmentation results.
        
    # Arguments
        image: single channel array where each value represents the class key.
        label_values
        
    # Returns
        Colour coded image for segmentation visualization
    """
    colour_codes = np.array(label_values)
    x = colour_codes[image.astype(int)]
    return x
    #return TF.to_tensor(x)

def unique(ar, return_index=False, return_inverse=False, return_counts=False):
    ar = np.asanyarray(ar).flatten()

    optional_indices = return_index or return_inverse
    optional_returns = optional_indices or return_counts

    if ar.size == 0:
        if not optional_returns:
            ret = ar
        else:
            ret = (ar,)
            if return_index:
                ret += (np.empty(0, np.bool),)
            if return_inverse:
                ret += (np.empty(0, np.bool),)
            if return_counts:
                ret += (np.empty(0, np.intp),)
        return ret
    if optional_indices:
        perm = ar.argsort(kind='mergesort' if return_index else 'quicksort')
        aux = ar[perm]
    else:
        ar.sort()
        aux = ar
    flag = np.concatenate(([True], aux[1:] != aux[:-1]))

    if not optional_returns:
        ret = aux[flag]
    else:
        ret = (aux[flag],)
        if return_index:
            ret += (perm[flag],)
        if return_inverse:
            iflag = np.cumsum(flag) - 1
            inv_idx = np.empty(ar.shape, dtype=np.intp)
            inv_idx[perm] = iflag
            ret += (inv_idx,)
        if return_counts:
            idx = np.concatenate(np.nonzero(flag) + ([ar.size],))
            ret += (np.diff(idx),)
    return ret

def colorEncode(labelmap, colors, class_names, mode='RGB', legend=False):
    labelmap = labelmap.astype('int')
    labelmap_rgb = np.zeros((labelmap.shape[0], labelmap.shape[1], 3),
                            dtype=np.uint8)
    for label in unique(labelmap):
        if label < 0 or label > len(colors):
            continue
        labelmap_rgb += (labelmap == label)[:, :, np.newaxis] * \
            np.tile(colors[label],
                    (labelmap.shape[0], labelmap.shape[1], 1))
        #if legend:
        #    leg_image = np.zeros((50,50,3)).astype(np.uint8)
        #    leg_image[:,:,0] = colors[label][0]
         #   leg_image[:,:,1] = colors[label][1]
          #  leg_image[:,:,2] = colors[label][2]
         #   test = Image.fromarray(leg_image)
         #   save_name = class_names[label] + ".png"
         #   test.save(save_name)
            #print(class_names[label])

    if mode == 'BGR':
        return labelmap_rgb[:, :, ::-1]
    else:
        return labelmap_rgb
    
###############################
# Metrics
###############################
        
def accuracy(preds, label):
    valid = (label >= 0)
    acc_sum = (valid * (preds == label)).sum()
    valid_sum = valid.sum()
    acc = float(acc_sum) / (valid_sum + 1e-10)
    return acc, valid_sum


def intersectionAndUnion(imPred, imLab, numClass):
    imPred = np.asarray(imPred).copy()
    imLab = np.asarray(imLab).copy()

    imPred += 1
    imLab += 1
    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    imPred = imPred * (imLab > 0)

    # Compute area intersection:
    intersection = imPred * (imPred == imLab)
    (area_intersection, _) = np.histogram(
        intersection, bins=numClass, range=(1, numClass))

    # Compute area union:
    (area_pred, _) = np.histogram(imPred, bins=numClass, range=(1, numClass))
    (area_lab, _) = np.histogram(imLab, bins=numClass, range=(1, numClass))
    area_union = area_pred + area_lab - area_intersection

    return (area_intersection, area_union)