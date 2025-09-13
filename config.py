import os

import edict
import torch
from easydict import EasyDict as edict

config = edict()

# Paths (change these to your local paths)
config.data_root = "/home/zohra/pythonCode/data_machnet"  # parent folder containing train1, train2, test
config.checkpoint_dir = "./checkpoint"
config.save_pth = "best_model_celoss_diceloss"
# Model / training
config.num_classes = 10  #
config.img_size = 512 # (H, W) - adjust for your GPU
config.batch_size = 8
config.lr = 1e-4
config.epochs = 10

# Device
config.device = "cuda" if torch.cuda.is_available() else "cpu"
