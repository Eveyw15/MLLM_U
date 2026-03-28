"""
download_coco_for_subset.py

Download or verify COCO images needed for the first N samples of the
UnLOK-VQA evaluation subset.

Usage:
    python download_coco_for_subset.py \
        --data_path /path/to/zsre_mend_eval.json \
        --coco_root /path/to/coco2017 \
        --n 50

The script tries train2017 first, then val2017, and reports any failures.
"""

import argparse
import json
import urllib.request
import urllib.error
from pathlib import Path


COCO_SPLITS = ["train2017", "val2017"]
COCO_BASE_URL = "http://images.cocodataset.org"


def parse_args():
    p = argparse.ArgumentParser(description="Download COCO images for UnLOK-VQA subset")
    p.add_argument("--data_path", required=True,
                   help="Path to zsre_mend_eval.json")
    p.add_argument("--coco_root", required=True,
                   help="Root directory for COCO images (will contain train2017/, val2017/)")
    p.add_argument("--n", type=int, default=50,
                   help="Number of samples to process (default: 50)")
    return p.parse_args()


def find_image(coco_root: Path, image_id: int):
    """Return path if image already exists locally, else None."""
    fn = f"{image_id:012d}.jpg"
    for split in COCO_SPLITS:
        p = coco_root / split / fn
        if p.exists():
            return p
    return None


def download_image(coco_root: Path, image_id: int):
    """Try to download image from train2017 then val2017. Return (path, split) or (None, None)."""
    fn = f"{image_id:012d}.jpg"
    for split in COCO_SPLITS:
        dest = coco_root / split / fn
        url = f"{COCO_BASE_URL}/{split}/{fn}"
        try:
            urllib.request.urlretrieve(url, str(dest))
            return dest, split
        except urllib.error.HTTPError:
            pass
        except Exception as e:
            print(f"  Warning: unexpected error downloading {url}: {e}")
    return None, None


def main():
    args = parse_args()

    data_path = Path(args.data_path)
    coco_root = Path(args.coco_root)
    n = args.n

    assert data_path.exists(), f"Data file not found: {data_path}"

    # Create directories
    for split in COCO_SPLITS:
        (coco_root / split).mkdir(parents=True, exist_ok=True)

    with open(data_path) as f:
        data = json.load(f)

    subset = data[:n]
    print(f"Processing {len(subset)} samples from {data_path}")

    already_present = 0
    downloaded = 0
    failed_ids = []

    for ex in subset:
        image_id = ex["image_id"]

        existing = find_image(coco_root, image_id)
        if existing:
            already_present += 1
            continue

        path, split = download_image(coco_root, image_id)
        if path:
            downloaded += 1
        else:
            failed_ids.append(image_id)

    total_ok = already_present + downloaded
    print(f"\n=== Image availability ===")
    print(f"Already present : {already_present}")
    print(f"Downloaded now  : {downloaded}")
    print(f"Total OK        : {total_ok} / {len(subset)}")
    print(f"Failed          : {len(failed_ids)}")
    if failed_ids:
        print(f"Failed IDs      : {failed_ids[:20]}{'...' if len(failed_ids) > 20 else ''}")

    # Final verification pass
    print("\n=== Verification ===")
    missing = []
    for ex in subset:
        if find_image(coco_root, ex["image_id"]) is None:
            missing.append(ex["image_id"])
    if missing:
        print(f"Still missing {len(missing)} images: {missing[:10]}")
    else:
        print(f"All {len(subset)} images verified OK.")

    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
