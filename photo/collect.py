"""Collect licensed iNaturalist images and record their provenance."""
import argparse
import csv
import hashlib
import io
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

import requests
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "dataset"
API = "https://api.inaturalist.org/v1/observations"
TAXA = {
    "Iris setosa": ("setosa", 64),
    "Iris versicolor": ("versicolor", 64),
    "Iris virginica": ("virginica", 64),
    "Iris pseudacorus": ("other_iris", 24),
    "Iris germanica": ("other_iris", 24),
    "Iris sibirica": ("other_iris", 24),
    "Iris cristata": ("other_iris", 24),
    "Hemerocallis fulva": ("not_iris", 22),
    "Gladiolus communis": ("not_iris", 22),
    "Lilium lancifolium": ("not_iris", 22),
    "Tradescantia ohiensis": ("not_iris", 22),
    "Crocus vernus": ("not_iris", 22),
    "Taraxacum officinale": ("not_iris", 18),
    "Quercus robur": ("not_iris", 16),
}
LICENSES = {"cc0", "cc-by"}
WEIGHTS = ("https://storage.googleapis.com/tensorflow/keras-applications/efficientnet_v2/"
           "efficientnetv2-b0_notop.h5")


def fetch(url, **kwargs):
    for attempt in range(3):
        try:
            response = requests.get(url, timeout=(8, 14), **kwargs)
            response.raise_for_status()
            return response
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def collect(taxon, label, limit):
    candidates = []
    seen_observers = set()
    for page in (1, 2):
        response = fetch(API, params={
            "taxon_name": taxon, "quality_grade": "research", "photo_license": "cc0,cc-by",
            "photos": "true", "per_page": 120, "page": page,
            "order_by": "votes", "order": "desc",
        })
        for observation in response.json()["results"]:
            if observation.get("taxon", {}).get("name") != taxon:
                continue
            user = observation["user"]["id"]
            if user in seen_observers:
                continue
            for photo in observation["photos"]:
                if photo["license_code"] not in LICENSES or not photo.get("url"):
                    continue
                url = photo["url"].replace("/square.", "/medium.")
                if urlparse(url).hostname not in {
                    "inaturalist-open-data.s3.amazonaws.com", "static.inaturalist.org",
                }:
                    continue
                seen_observers.add(user)
                partition = int(hashlib.sha256(f"iris-demo-v1:{user}".encode()).hexdigest()[:8], 16) % 100
                split = "test" if partition < 20 else "validation" if partition < 38 else "train"
                candidates.append({
                    "filename": f"{photo['id']}.jpg", "species": taxon, "label": label,
                    "split": split, "observation": f"https://www.inaturalist.org/observations/{observation['id']}",
                    "photo_url": url, "license": photo["license_code"], "attribution": photo.get("attribution") or "",
                })
                break
            if len(candidates) >= int(limit * 1.55):
                return candidates
    return candidates


def save_image(row):
    path = CACHE / row["filename"]
    if path.exists():
        return row
    try:
        image = Image.open(io.BytesIO(fetch(row["photo_url"]).content))
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((512, 512))
        image.save(path, quality=92)
        return row
    except (requests.RequestException, OSError, ValueError):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="Select a new sample and replace the manifest.")
    args = parser.parse_args()
    CACHE.mkdir(exist_ok=True)
    weights = CACHE / "efficientnetv2_b0.h5"
    if not weights.exists():
        weights.write_bytes(fetch(WEIGHTS).content)
    manifest = ROOT / "image_sources.csv"
    if manifest.exists() and not args.refresh:
        with manifest.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        with ThreadPoolExecutor(max_workers=8) as pool:
            missing = [row["filename"] for row, downloaded in zip(rows, pool.map(save_image, rows))
                       if downloaded is None]
        if missing:
            raise RuntimeError(f"Could not retrieve {len(missing)} photos: {', '.join(missing[:5])}")
        print(f"Restored {len(rows)} photos from the fixed manifest")
        return

    known_files = set()
    rows = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for taxon, (label, limit) in TAXA.items():
            candidates = [row for row in collect(taxon, label, limit) if row["filename"] not in known_files]
            quotas = {"train": round(limit * .62), "validation": round(limit * .18), "test": round(limit * .20)}
            groups = {split: [row for row in candidates if row["split"] == split][:count + 3]
                      for split, count in quotas.items()}
            kept = []
            for split, group in groups.items():
                successful = [row for row in pool.map(save_image, group) if row]
                kept.extend(successful[:quotas[split]])
            rows.extend(kept)
            known_files.update(row["filename"] for row in kept)
            print(taxon, len(kept), flush=True)
    with (ROOT / "image_sources.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
