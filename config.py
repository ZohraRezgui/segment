import os

import torch
from easydict import EasyDict as edict

config = edict()

# Input paths (change these to your local paths)
config.data_root = "/home/zohra/pythonCode/data_machnet"  # parent folder containing train1, train2, test
config.checkpoint_dir = "./checkpoint" # where models will be saved


# train params
config.num_classes = 10 # number of classes 9 classes + backgorund
config.img_size = 512 # images will be resized to this
config.batch_size = 6 # batch size
config.lr = 1e-4 # learning rate
config.epochs = 10 #  number of epochs
config.combined_loss = "focal" # focal or ce for focal loss or cross entropy loss to be used in combination with dice loss
config.unfreeze_last_block = True # unfreeze the last blocks of mobilenet
config.use_augmentation = True # use augmentation
# Device
config.device = "cuda" if torch.cuda.is_available() else "cpu"

# output paths
config.save_pth = "model_6.pth" #the name of the model saved during training/loaded during evaluation
config.res_dir = "./results" # directory where evaluation results will be saved

