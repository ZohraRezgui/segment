
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large


class SurgicalSegmentationModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.model = deeplabv3_mobilenet_v3_large(pretrained=True)
        self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        return self.model(x)["out"]

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # inputs: [B, C, H, W] (logits)
        # targets: [B, H, W] (class indices)
        inputs = F.softmax(inputs, dim=1)  # probs
        targets_onehot = F.one_hot(targets, num_classes=inputs.shape[1])  # [B,H,W,C]
        targets_onehot = targets_onehot.permute(0, 3, 1, 2).float()  # [B,C,H,W]

        dims = (0, 2, 3)
        intersection = torch.sum(inputs * targets_onehot, dims)
        cardinality = torch.sum(inputs + targets_onehot, dims)
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

        return 1 - dice.mean()
