
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import (
    DeepLabV3_MobileNet_V3_Large_Weights,
    deeplabv3_mobilenet_v3_large,
)


class SurgicalSegmentationModel(nn.Module):
    def __init__(self, num_classes=10, unfreeze_last_block=False):
        """
        unfreeze_last_layers: int, number of last blocks to unfreeze in backbone
        """
        super().__init__()

        weights = DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1
        self.model = deeplabv3_mobilenet_v3_large(weights=weights)


        # Replace classifier for custom number of classes
        self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)

        # Freeze all backbone parameters initially
        for param in self.model.backbone.parameters():
            param.requires_grad = False

        # Optionally unfreeze last block
        if unfreeze_last_block :
            for param in self.model.backbone["16"].parameters():
                param.requires_grad = True
            for param in self.model.backbone["15"].parameters():
                param.requires_grad = True
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


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)  # predicted probability of true class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss
