import base64
import io
import json
import re
import urllib.request
import os
import cv2
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    raise SystemExit("Pillow is required: pip install pillow")

from typhoon_test import ocr_image,MODEL,EXTENSIONS
from step_2 import enhance_gray_sharp

# CROP_DIR = "crop"
GRAY_DIR = "gray"
IMAGE_DIR   = "ids"
OUTPUT_FILE = "test-pipe-pre.json"
# EXTENSIONS  = {".jpg", ".jpeg", ".png"}
# MODEL       = "scb10x/typhoon-ocr1.5-3b:latest"

def main():

    # Ensure dirs
    image_dir = Path(IMAGE_DIR)
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory '{IMAGE_DIR}' not found.")

    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in EXTENSIONS)

    # Check if it exists and is a folder
    if not Path(GRAY_DIR).is_dir():
        Path(GRAY_DIR).mkdir()

    if not images:
        print(f"No images found in '{IMAGE_DIR}'.")
        return
    
    print(f"Found {len(images)} image(s). Preprocessing image to '{GRAY_DIR}'...\n")

    for i, img_path in enumerate(images, 1):
        print(f"  [{i}/{len(images)}] {img_path.name} ... ", end="", flush=True)
        # print(f"  [{i}/{len(images)}] {img_path.name} ... ", end="")
        img = cv2.imread(img_path)
        gray = enhance_gray_sharp(img)
        newname = re.sub("_crop", "", img_path.stem)+"_gray_sharp"+img_path.suffix
        cv2.imwrite(Path(GRAY_DIR) / newname,gray)
        print(f"→  saved to {str(Path(GRAY_DIR) / newname)}")

    image_dir=Path(GRAY_DIR)
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory '{GRAY_DIR}' not found.")

    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in EXTENSIONS)

    print(f"Found {len(images)} image(s). Running OCR with {MODEL}...\n")

    all_results = {}

    # Process each image
    for i, img_path in enumerate(images, 1):
        print(f"  [{i}/{len(images)}] {img_path.name} ... ", end="", flush=True)

        fields, _ = ocr_image(str(img_path))
        all_results[img_path.name] = fields

        preview_full = json.dumps(fields, ensure_ascii=False)
        preview = preview_full[:100]
        print(f"→  {preview}{'...' if len(preview_full) > 100 else ''}")

    # Write results to output file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        json.dump(all_results, out, ensure_ascii=False, indent=2)

    print(f"\nDone. Results saved to '{OUTPUT_FILE}'.")


if __name__ == "__main__":
    main()