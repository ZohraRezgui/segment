import gc
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import config as cfg
from dataset import SARRARP50Dataset
from model import DiceLoss, FocalLoss, SurgicalSegmentationModel

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    # torch.backends.cudnn.benchmark = True




def batch_sanity_check(preds, masks, batch_idx=None):
    """
    Quick sanity check for a batch of predictions and targets.

    Args:
        preds (torch.Tensor): model predictions (B, H, W), already argmaxed
        masks (torch.Tensor): ground-truth masks (B, H, W)
        batch_idx (int, optional): batch index for printing
    """
    unique_pred = torch.unique(preds)
    unique_target = torch.unique(masks)

    prefix = f"Batch {batch_idx}: " if batch_idx is not None else ""
    print(
        f"{prefix}Predicted classes = {unique_pred.tolist()}, "
        f"Ground truth classes = {unique_target.tolist()}"
    )




def calculate_iou(pred, target, num_classes):
    ious = []
    pred = pred.cpu().int()
    target = target.cpu().int()

    for cls in range(1, num_classes):
        target_cls = target == cls
        if target_cls.sum() == 0:
            continue

        pred_cls = pred == cls
        intersection = (pred_cls & target_cls).sum().item()
        union = (pred_cls | target_cls).sum().item()

        if union > 0:
            ious.append(intersection / union)

    return float(np.mean(ious)) if ious else 0.0

def train_epoch(model, dataloader, criterion1,criterion2, optimizer, device):
    model.train()
    total_loss = 0
    total_iou = 0

    pbar = tqdm(dataloader, desc="Training")
    for images, masks in pbar:
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        ce_loss = criterion1(outputs, masks)
        dice_loss = criterion2(outputs, masks)
        loss = ce_loss + dice_loss
        loss.backward()
        optimizer.step()

        # Calculate IoU
        with torch.no_grad():
            preds = torch.argmax(outputs, dim=1)
            # batch_sanity_check(preds, masks, batch_idx=0)
            iou = calculate_iou(preds, masks, model.model.classifier[4].out_channels)

        total_loss += loss.item()
        total_iou += iou

        pbar.set_postfix({"Loss": f"{loss.item():.4f}", "IoU": f"{iou:.4f}"})

    return total_loss / len(dataloader), total_iou / len(dataloader)


def validate_epoch(model, dataloader, criterion1, criterion2, device):
    model.eval()
    total_loss = 0
    total_iou = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for images, masks in pbar:
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            ce_loss = criterion1(outputs, masks)
            dice_loss = criterion2(outputs, masks)
            loss = ce_loss + dice_loss

            preds = torch.argmax(outputs, dim=1)
            iou = calculate_iou(preds, masks, model.model.classifier[4].out_channels)

            total_loss += loss.item()
            total_iou += iou

            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "IoU": f"{iou:.4f}"})

    return total_loss / len(dataloader), total_iou / len(dataloader)


def main():


    # Device
    device = cfg.device
    print(f"Using device: {device}")

    # Datasets
    train_dataset = SARRARP50Dataset(
        cfg.data_root, split="train", image_size=cfg.img_size, num_classes=cfg.num_classes, use_augmentation=cfg.use_augmentation
    )
    test_dataset = SARRARP50Dataset(
        cfg.data_root, split="test", image_size=cfg.img_size, num_classes=cfg.num_classes
    )

    # Data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=6
    )
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=0)


    # Model, loss, optimizer
    model = SurgicalSegmentationModel(cfg.num_classes, cfg.unfreeze_last_block).to(device)


    if cfg.combined_loss == "focal":
        criterion = FocalLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    dice_loss = DiceLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', threshold = 0.001,factor=0.5, patience=2, verbose=True)
    # Training loop
    best_iou = 0

    for epoch in range(cfg.epochs):
        print(f"\nEpoch {epoch + 1}/{cfg.epochs}")

        # Train
        train_loss, train_iou = train_epoch(
            model, train_loader, criterion, dice_loss, optimizer, device
        )

        # Validate
        val_loss, val_iou = validate_epoch(model, test_loader, criterion, dice_loss, device)
        scheduler.step(val_loss)
        print(f"Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}")

        # Save best model
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(
                model.state_dict(),
                os.path.join(cfg.checkpoint_dir, cfg.save_pth),
            )
            print(f"New best model saved! IoU: {best_iou:.4f}")

    print(f"\nTraining completed. Best IoU: {best_iou:.4f}")





if __name__ == "__main__":

    main()
