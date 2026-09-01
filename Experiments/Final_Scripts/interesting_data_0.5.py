import os
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN

# Check GPU availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)

input_dir = "/home/justincb/projects/def-primath/shared_data/datasets/bluesky_interesting"
output_dir = "/scratch/justincb/bluesky_faces_cleaned_50_percent"
target_size = (256, 256)

os.makedirs(output_dir, exist_ok=True)

# Fast, GPU-native face detector
detector = MTCNN(keep_all=True, device=device)

def extract_largest_face(image_path, base_name): # Take the largest face from each image to prevent multiple faces in the same image from imbalancing the dataset.
    img = Image.open(image_path).convert("RGB")
    boxes, _ = detector.detect(img)

    if boxes is None or len(boxes) == 0:
        return  # no faces → skip

    # Pick largest face by area
    largest = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    x1, y1, x2, y2 = largest

    # Expand bounding box by a margin (e.g., 80%)
    margin = 0.5
    w = x2 - x1
    h = y2 - y1
    x1 = max(0, int(x1 - w * margin / 2))
    y1 = max(0, int(y1 - h * margin / 2))
    x2 = min(img.width, int(x2 + w * margin / 2))
    y2 = min(img.height, int(y2 + h * margin / 2))

    crop = img.crop((x1, y1, x2, y2))
    crop = crop.resize(target_size, Image.LANCZOS)

    out_name = f"{base_name}_face.jpg"
    out_path = os.path.join(output_dir, out_name)

    if os.path.exists(out_path):
        os.remove(out_path)

    crop.save(out_path, "JPEG", quality=95, subsampling=2)

count = 0
print("Starting face extraction...")

for filename in os.listdir(input_dir):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        count += 1
        if count % 100 == 0:
            print(f"Processed {count} images...")

        full_path = os.path.join(input_dir, filename)
        base = os.path.splitext(filename)[0]

        extract_largest_face(full_path, base)

print("Done! Largest faces extracted and resized.")