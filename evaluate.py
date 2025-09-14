import csv
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import config as cfg
from dataset import SARRARP50Dataset
from model import SurgicalSegmentationModel


def compute_iou_per_class(pred, target, num_classes, ignore_background=True):
    """Compute IoU per class for a single prediction/target pair."""
    ious = np.zeros(num_classes, dtype=np.float64)
    counts = np.zeros(num_classes, dtype=np.int64)

    for cls in range(num_classes):
        if ignore_background and cls == 0:
            continue
        pred_cls = pred == cls
        target_cls = target == cls

        union = np.logical_or(pred_cls, target_cls).sum()
        if union == 0:
            continue
        inter = np.logical_and(pred_cls, target_cls).sum()

        ious[cls] += inter / union
        counts[cls] += 1

    return ious, counts


def evaluate(model, dataloader, num_classes, device):
    model.eval()

    video_ious = {}  # per-video IoUs
    class_ious = np.zeros(num_classes, dtype=np.float64)
    class_counts = np.zeros(num_classes, dtype=np.int64)

    with torch.no_grad():
        for images, masks, video_ids in tqdm(dataloader, desc="Evaluating"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            masks = masks.cpu().numpy()

            for i in range(len(preds)):
                pred = preds[i]
                target = masks[i]
                video_id = video_ids[i]

                # per class IoUs
                ious, counts = compute_iou_per_class(pred, target, num_classes)
                class_ious += ious
                class_counts += counts

                # mean IoU for this frame
                frame_ious = [iou for iou, c in zip(ious, counts) if c > 0]
                frame_mean_iou = np.mean(frame_ious) if frame_ious else 0

                if video_id not in video_ious:
                    video_ious[video_id] = []
                video_ious[video_id].append(frame_mean_iou)

    # average per class IoUs
    per_class_iou = {
        cls: (class_ious[cls] / class_counts[cls] if class_counts[cls] > 0 else 0.0)
        for cls in range(num_classes)
    }

    # average per video
    per_video_iou = {
        vid: float(np.mean(iou_list)) for vid, iou_list in video_ious.items()
    }

    return per_class_iou, per_video_iou

def save_results(per_class_iou, per_video_iou, out_csv="iou_results.csv"):
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)

        # Write per-class results
        writer.writerow(["Per-Class IoU"])
        writer.writerow(["Class", "IoU"])
        for cls, iou in per_class_iou.items():
            writer.writerow([cls, f"{iou:.4f}"])
        writer.writerow([])

        # Write per-video results
        writer.writerow(["Per-Video Mean IoU "])
        writer.writerow(["Video", "Mean IoU"])
        for vid, miou in per_video_iou.items():
            writer.writerow([vid, f"{miou:.4f}"])


if __name__ == "__main__":
    # DATA_ROOT = "/home/zohra/pythonCode/data_machnet"
    # NUM_CLASSES = 10
    # IMAGE_SIZE = 512
    # MODEL_PATH = "best_model_ce_dice.pth"

    device = cfg.device

    # dataset + dataloader
    test_dataset = SARRARP50Dataset(
        cfg.data_root,
        split="test",
        image_size=cfg.img_size,
        num_classes=cfg.num_classes,
        return_video_id=True,
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    # load model
    model = SurgicalSegmentationModel(cfg.num_classes)
    model.load_state_dict(torch.load(os.path.join(cfg.checkpoint_dir, cfg.save_pth), map_location=device))
    model = model.to(device)

    per_class_iou, per_video_iou = evaluate(model, test_loader, cfg.num_classes, device)

    print("\n=== Per-Class IoU ===")
    for cls, iou in per_class_iou.items():
        print(f"Class {cls}: {iou:.4f}")

    print("\n=== Per-Video Mean IoU ===")
    for vid, miou in per_video_iou.items():
        print(f"{vid}: {miou:.4f}")

    save_results(per_class_iou, per_video_iou, out_csv=f"iou_results_{os.path.splitext(cfg.save_pth)[0]}.csv")
