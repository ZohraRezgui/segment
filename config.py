import os

import torch
from easydict import EasyDict as edict

config = edict()

# Paths (change these to your local paths)
config.data_root = "/home/zohra/pythonCode/data_machnet"  # parent folder containing train1, train2, test
config.checkpoint_dir = "./checkpoint"
# Model / training
config.num_classes = 10  #
config.img_size = 512 # (H, W) - adjust for your GPU
config.batch_size = 6
config.lr = 1e-4#1e-4
config.epochs = 10
config.combined_loss = "focal"
config.unfreeze_last_block = True
config.save_pth = f"best_model_{config.combined_loss}_diceloss_augment_lastblockres.pth"
config.use_augmentation = True
# Device
config.device = "cuda" if torch.cuda.is_available() else "cpu"
