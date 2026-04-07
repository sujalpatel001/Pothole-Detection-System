"""
Downloads the pretrained YOLOv8n pothole-segmentation model (best.pt)
directly from HuggingFace into the project directory.
"""

import urllib.request
import os
from pathlib import Path

MODEL_URL = (
    "https://huggingface.co/keremberke/yolov8n-pothole-segmentation"
    "/resolve/main/best.pt"
)
SAVE_PATH = Path("best.pt")

def download_with_progress(url: str, dest: Path):
    print(f"\n  Downloading: {url}")
    print(f"  Saving to  : {dest}\n")

    def reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(downloaded / total_size * 100, 100)
            bar = "█" * int(pct // 2) + "░" * (50 - int(pct // 2))
            mb_done = downloaded / 1_048_576
            mb_total = total_size / 1_048_576
            print(f"\r  [{bar}] {pct:5.1f}%  {mb_done:.1f}/{mb_total:.1f} MB", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=reporthook)
    print(f"\n\n  ✅ Download complete! Saved as '{dest}'\n")

if __name__ == "__main__":
    if SAVE_PATH.exists():
        print(f"  ✅ '{SAVE_PATH}' already exists — skipping download.")
    else:
        download_with_progress(MODEL_URL, SAVE_PATH)
        size_mb = SAVE_PATH.stat().st_size / 1_048_576
        print(f"  File size : {size_mb:.1f} MB")
