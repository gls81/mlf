from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()
_C.NAME = "default"
_C.MODE = "TEST"
_C.NOTE = "This is the basic default value for the notes string"


# -----------------------------------------------------------------------------
# Directories
# -----------------------------------------------------------------------------

_C.DIR = CN()

_C.DIR.parent = "data"

#Name of the defualt output folder
_C.DIR.output = "output"

#Output folder subfolders
#Trained model checkpoint files
_C.DIR.ckpts = "ckpts"
#Saved metrics generated from validation 
_C.DIR.val_metrics = "results"
#Images saved to disk from output of models
_C.DIR.vis = "vis"


#Tensorboard log files
_C.DIR.logs = "logs"
_C.DIR.pretrained = "files"
#Main directory of datasets 
_C.DIR.datasets = "datasets"


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASET = CN()

#Name of the datasets to be applied 
_C.DATASET.train = ["ADEChallengeData2016"]
_C.DATASET.val = ["ADEChallengeData2016"]
_C.DATASET.test = ["TreeBags"]
_C.DATASET.classes = "ADEChallengeData2016"

# defining dataset transform options
# multiscale train/test, size of short edge (int or tuple)
_C.DATASET.custom_collator = False
_C.DATASET.imgInputSizes = [300, 375, 450, 525, 600]

# is input reshaped to square
_C.DATASET.imgInputIsSquare = True
# maximum input image size of long edge
_C.DATASET.imgMaxSize = 1000
# maxmimum downsampling rate of the network
_C.DATASET.padding_constant = 8
# downsampling rate of the segmentation label
_C.DATASET.segm_downsampling_rate = 8
# randomly horizontally flip images when train/test
_C.DATASET.random_hozizontal_flip = True
_C.DATASET.gussian_blur = []
_C.DATASET.random_crop = False
_C.DATASET.normalize_mean= [0.485, 0.456, 0.406]
_C.DATASET.normalize_std=[0.229, 0.224, 0.225]
_C.DATASET.one_hot = False

#Return a edge map of the RGB image
_C.DATASET.edge_map=False


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.type = "SS"

_C.MODEL.components = ["resnet50dilated","ppm_deepsup"]
_C.MODEL.components_type = ["encoder","decoder"]
_C.MODEL.components_weights = ["",""]
_C.MODEL.components_fc_dim = [2048]
_C.MODEL.components_input_dim = [] #Used for GAN generator input 
# architecture of net_encoder
#_C.MODEL.arch_encoder = "resnet50dilated"
# architecture of net_decoder
#_C.MODEL.arch_decoder = "ppm_deepsup"
# weights to finetune net_encoder
#_C.MODEL.weights_encoder = ""
# weights to finetune net_decoder
#_C.MODEL.weights_decoder = ""
# number of feature channels between encoder and decoder
#_C.MODEL.fc_dim = 2048

# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.batch_size_per_gpu = 2
# epochs to train for
_C.TRAIN.num_epoch = 20
# epoch to start training. useful if continue from a checkpoint
_C.TRAIN.checkpoint = 0
# iterations of each epoch (irrelevant to batch size)
_C.TRAIN.epoch_iters = 5000

# criterion to be used in training
_C.TRAIN.criterion = "NLLLOSS"
# Allows for addtional crit option to ingorne index. 
#This is used in many segmentationmodels to ingnore an index of background or padding
_C.TRAIN.criterion_ignore_index = -100

_C.TRAIN.optim = "SGD"
_C.TRAIN.lr = [0.02, 0.02]
# power in poly to drop LR
_C.TRAIN.lr_pow = 0.9
# momentum for sgd, beta1 for adam
_C.TRAIN.beta1 = 0.9
# weights regularizer
_C.TRAIN.weight_decay = 1e-4
# the weighting of deep supervision loss
_C.TRAIN.deep_sup_scale = 0.4
# fix bn params, only under finetuning
_C.TRAIN.fix_bn = False
# number of data loading workers
_C.TRAIN.workers = 16

# frequency to display
_C.TRAIN.disp_iter = 20
# manual seed
_C.TRAIN.seed = 304
# log visual output via tensorboard
_C.TRAIN.visualize = True
# Number of images from the training set to display per epoch
# For N images N random images from the training dataset will chosen at the begging and then logged as each training epoch
_C.TRAIN.visualise_sample_num = 1

_C.TRAIN.softmax = False


# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------
_C.VAL = CN()
# currently only supports 1
_C.VAL.batch_size = 1
# output visualization during validation
_C.VAL.visualize = False
# 0 denotes the output of all predictions if visualize is set to true 
_C.VAL.visualise_sample_num = 0
# the checkpoint to evaluate on
_C.VAL.checkpoint = 20
_C.VAL.softmax = False

# -----------------------------------------------------------------------------
# Testing
# -----------------------------------------------------------------------------
_C.TEST = CN()
# currently only supports 1
_C.TEST.batch_size = 1
# the checkpoint to test on
_C.TEST.checkpoint = 20
# folder to output visualization results
_C.TEST.result = "./"
_C.TEST.visualize = True
_C.TEST.softmax = True

