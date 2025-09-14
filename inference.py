import argparse
import os

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

from model import SurgicalSegmentationModel


def run_inference_on_image(
    image_path, model_path, num_classes=10, image_size=512, device=None
):


    model = SurgicalSegmentationModel(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

    return pred


def run_inference_on_folder(
    folder_path, model_path, output_folder, num_classes=10, image_size=512
):
    image_files = sorted(
        [f for f in os.listdir(folder_path) if f.endswith(".png")]
    )
    for img_file in tqdm(image_files, desc="Running inference"):
        img_path = os.path.join(folder_path, img_file)
        video_name = os.path.basename(os.path.dirname(folder_path))
        save_dir = os.path.join(output_folder, video_name)
        os.makedirs(save_dir ,exist_ok=True)

        pred = run_inference_on_image(img_path, model_path, num_classes, image_size)
        # Save grayscale prediction
        pred_img = Image.fromarray(pred.astype(np.uint8))
        pred_img = pred_img.resize((1920, 1080), resample=Image.NEAREST)
        pred_img.save(os.path.join(save_dir, img_file))


def main():
    parser = argparse.ArgumentParser(description="Run surgical segmentation inference")
    parser.add_argument(
        "--model", type=str, required=True, help="Path to the trained model (.pt/.pth)"
    )
    parser.add_argument(
        "--image", type=str, default=None, help="Path to a single image"
    )
    parser.add_argument(
        "--folder", type=str, default=None, help="Path to a folder of images"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output path/folder for predictions"
    )
    parser.add_argument(
        "--num_classes", type=int, default=10, help="Number of segmentation classes"
    )
    parser.add_argument(
        "--image_size", type=int, default=512, help="Resize images to this size"
    )
    args = parser.parse_args()

    if args.image is None and args.folder is None:
        raise ValueError("Either --image or --folder must be provided.")

    if args.image:
        pred = run_inference_on_image(
            args.image, args.model, args.num_classes, args.image_size
        )
        os.makedirs(args.output, exist_ok=True)
        base_name = os.path.basename(args.image)
        pred_img = Image.fromarray(pred.astype(np.uint8))
        pred_img = pred_img.resize((1920, 1080), resample=Image.NEAREST)

        pred_img.save(os.path.join(args.output, base_name))
        print(f"Prediction saved to {os.path.join(args.output, base_name)}")
    elif args.folder:
        run_inference_on_folder(
            args.folder, args.model, args.output, args.num_classes, args.image_size
        )
        print(f"Predictions saved to folder {args.output}")


if __name__ == "__main__":
    main()
