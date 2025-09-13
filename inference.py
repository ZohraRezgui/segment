import os

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

from config import config as cfg
from model import SurgicalSegmentationModel


def inference(image_path, model_path, num_classes=10, image_size=512):
    """Run inference on a single image"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = SurgicalSegmentationModel(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    input_tensor = transform(image).unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

    # Save prediction
    pred_image = Image.fromarray((pred * 255 // num_classes).astype(np.uint8))
    pred_image.save("prediction.png")
    # print(f"Prediction saved to prediction.png")

    return pred
