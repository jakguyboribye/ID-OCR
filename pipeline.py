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

def save_results(all_results: dict, output_file: str) -> None:
    """
    Write results to a temp file, then atomically rename it over the real
    output file. A crash/kill mid-write can never leave OUTPUT_FILE
    half-written or corrupted -- the rename either fully happens or doesn't.
    """
    tmp_path = f"{output_file}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as out:
        json.dump(all_results, out, ensure_ascii=False, indent=2)
    Path(tmp_path).replace(output_file)


def load_existing_results(output_file: str) -> dict:
    """Load whatever was saved from a previous (possibly interrupted) run."""
    path = Path(output_file)
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        # Corrupt/partial file from an old run -- start fresh rather than crash
        return {}


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
        # cv2.imread requires a plain str path, not a Path object -- passing
        # a Path directly raises "TypeError: Can't convert object to 'str'
        # for 'filename'" on some OpenCV/Python builds.
        img = cv2.imread(str(img_path))
        if img is None:
            # imread doesn't raise on a bad/corrupt/unreadable file, it just
            # returns None, which would otherwise crash inside
            # enhance_gray_sharp with a confusing error. Fail loudly here
            # instead and move on to the next image.
            print(f"→  SKIPPED (cv2 could not read this file, possibly corrupt or unsupported format)")
            continue
        gray = enhance_gray_sharp(img)
        newname = re.sub("_crop", "", img_path.stem)+"_gray_sharp"+img_path.suffix
        cv2.imwrite(str(Path(GRAY_DIR) / newname), gray)
        print(f"→  saved to {str(Path(GRAY_DIR) / newname)}")

    image_dir=Path(GRAY_DIR)
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory '{GRAY_DIR}' not found.")

    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in EXTENSIONS)

    print(f"Found {len(images)} image(s). Running OCR with {MODEL}...\n")

    all_results = load_existing_results(OUTPUT_FILE)
    if all_results:
        done = {k for k, v in all_results.items() if isinstance(v, dict) and "_error" not in v}
        print(f"Resuming: {len(done)} image(s) already have results in '{OUTPUT_FILE}'.")

    # Process each image
    try:
        for i, img_path in enumerate(images, 1):
            name = img_path.name

            if name in all_results and isinstance(all_results[name], dict) \
                    and "_error" not in all_results[name]:
                print(f"  [{i}/{len(images)}] {name} ... skipped (already done)")
                continue

            print(f"  [{i}/{len(images)}] {name} ... ", end="", flush=True)

            try:
                fields, _ = ocr_image(str(img_path))
            except Exception as e:
                # Don't let one bad image kill the whole batch
                fields = {"_error": f"Unhandled error: {e}", "raw_ocr_text": ""}

            all_results[name] = fields

            preview_full = json.dumps(fields, ensure_ascii=False)
            preview = preview_full[:100]
            print(f"→  {preview}{'...' if len(preview_full) > 100 else ''}")

            # Save after every image so a crash/interrupt never loses progress
            save_results(all_results, OUTPUT_FILE)

    except KeyboardInterrupt:
        print("\nInterrupted by user (Ctrl+C) -- progress already saved up to the last image.")

    print(f"\nDone. Results saved to '{OUTPUT_FILE}'.")


if __name__ == "__main__":
    main()