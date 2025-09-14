import argparse
import os

import numpy as np
from PIL import Image

# Foreground colors for classes 1-9 (RGB)
CLASS_COLORS = {
    1: (173, 216, 230),  # light blue
    2: (0, 128, 0),  # green
    3: (0, 0, 255),  # blue
    4: (255, 255, 0),  # yellow
    5: (165, 42, 42),  # brown
    6: (128, 0, 128),  # purple
    7: (255, 165, 0),  # orange
    8: (255, 0, 0),  # red
    9: (128, 128, 128),  # grey
}




def colorize_mask(mask, opaque=True):
    """Convert grayscale mask to RGB(A) image with specified class colors."""
    h, w = mask.shape
    if opaque:
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)
        for cls, color in CLASS_COLORS.items():
            color_mask[mask == cls] = color
        return Image.fromarray(color_mask)
    else:
        # For overlay: make background transparent
        color_mask = np.zeros((h, w, 4), dtype=np.uint8)  # RGBA
        for cls, color in CLASS_COLORS.items():
            color_mask[mask == cls, :3] = color
            color_mask[mask == cls, 3] = 128  # semi-transparent alpha
        return Image.fromarray(color_mask, mode="RGBA")


def overlay_on_image(image, color_mask):
    """Overlay RGBA color mask on image."""
    return Image.alpha_composite(image.convert("RGBA"), color_mask)

def generate_overlay_gif(
    image_folder, mask_folder, output_path, alpha=0.5, duration=0.2, max_frames=20):
    """Generate GIF with semi-transparent masks over original frames."""
    frames = []
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(".png")])
    if max_frames is not None:
        image_files = image_files[:max_frames]
    for img_file in image_files:
        img_path = os.path.join(image_folder, img_file)

        mask_path = os.path.join(mask_folder,img_file)

        image = Image.open(img_path).convert("RGB")
        mask = np.array(Image.open(mask_path).convert('L'))
        color_mask = colorize_mask(mask, opaque=False)

        overlay = overlay_on_image(image, color_mask)
        frames.append(overlay)

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(duration * 1000),
        loop=0,
    )


def save_colored_masks(mask_folder, output_folder):
    """Save prediction or ground truth masks as colored images with opaque foreground and black background."""
    os.makedirs(output_folder, exist_ok=True)
    mask_files = sorted([f for f in os.listdir(mask_folder) if f.endswith(".png")])
    for mask_file in mask_files:
        mask_path = os.path.join(mask_folder, mask_file)
        mask = np.array(Image.open(mask_path))
        color_mask = colorize_mask(mask, opaque=True)
        color_mask.save(os.path.join(output_folder, mask_file))


def main():
    parser = argparse.ArgumentParser(description="Visualization for segmentation masks")
    parser.add_argument(
        "--mode",
        choices=["gif", "mask"],
        required=True,
        help="Choose operation: generate GIF or save colored masks",
    )
    parser.add_argument(
        "--image_folder",
        type=str,
        help="Folder containing original images (required for GIF)",
    )
    parser.add_argument(
        "--mask_folder", type=str, required=True, help="Folder containing masks"
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="Output file/folder path"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.5, help="Transparency for overlay GIF"
    )
    parser.add_argument(
        "--duration", type=float, default=0.2, help="Frame duration (s) for GIF"
    )

    args = parser.parse_args()

    if args.mode == "gif":

        if not args.image_folder:
            parser.error("image_folder is required for GIF generation")
        generate_overlay_gif(
            image_folder=args.image_folder,
            mask_folder=args.mask_folder,
            output_path=args.output_path,
            alpha=args.alpha,
            duration=args.duration,
        )
        print(f"Overlay GIF saved to {args.output_path}")

    elif args.mode == "mask":
        save_colored_masks(mask_folder=args.mask_folder, output_folder=args.output_path)
        print(f"Colored masks saved to {args.output_path}")


if __name__ == "__main__":
    main()
