"""Download a bounded, source-disjoint OpenFake validation sample."""

from __future__ import annotations

import argparse
import csv
import json
import ssl
import shutil
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import certifi

API = "https://datasets-server.huggingface.co/rows?dataset=ComplexDataLab%2FOpenFake&config=core&split=test"
REDDIT_API = "https://datasets-server.huggingface.co/rows?dataset=ComplexDataLab%2FOpenFake&config=reddit&split=test"
TLS = ssl.create_default_context(cafile=certifi.where())
SOURCES = (
    ("calibration", "calibration", "fake", {"z-image-turbo", "gpt-image-1.5"}, 100),
    ("calibration", "calibration", "real", {"imagenet"}, 100),
    ("test", "calibration", "fake", {"flux.2-klein-9b", "midjourney-7", "illustrious"}, 100),
    ("test", "calibration", "real", {"docci"}, 100),
    ("final", "calibration", "fake", {"gpt-image-2", "ernie-image", "ernie-image-turbo",
                                  "seedream-v5.0", "nano-banana-pro", "sora-2",
                                  "veo-3", "recraft-v3", "wan-video-2.5"}, 100),
)


def request_json(url: str) -> dict:
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=60, context=TLS) as response:
                return json.load(response)
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError


def download(item: tuple[str, Path]) -> None:
    url, path = item
    if path.is_file() and path.stat().st_size:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=90, context=TLS) as response:
        path.write_bytes(response.read())


def sample(root: Path, rwfs_real: Path) -> None:
    manifest = []
    downloads = []
    counts = Counter()
    for offset in range(0, 5000, 100):
        payload = request_json(f"{API}&offset={offset}&length=100")
        for entry in payload["rows"]:
            row = entry["row"]
            if row["image"]["width"] * row["image"]["height"] > 20_000_000:
                continue
            rule = next(((folder, split, label, models, count) for folder, split, label, models, count in SOURCES
                         if counts[(folder, label)] < count and row["label"] == label and row["model"] in models), None)
            if rule is None:
                continue
            folder, split, label, _, _ = rule
            model = row["model"]
            index = entry["row_idx"]
            relative = Path(folder) / label / model / f"{index}.jpg"
            downloads.append((row["image"]["src"], root / relative))
            manifest.append({"path": relative.as_posix(), "label": label, "split": split,
                             "group": f"{model}:{index}", "source": model})
            counts[(folder, label)] += 1
        if all(counts[(folder, label)] == count for folder, _, label, _, count in SOURCES):
            break
    missing = [(split, label, count - counts[(folder, label)]) for folder, split, label, _, count in SOURCES
               if counts[(folder, label)] < count]
    if missing:
        raise ValueError(f"OpenFake sample quotas were not met: {missing}")
    real_images = sorted(path for path in rwfs_real.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})[:100]
    if len(real_images) < 100:
        raise ValueError("RWFS needs at least 100 real images for the final test.")
    for index, source in enumerate(real_images):
        relative = Path("final") / "real" / "rwfs" / f"{index:03d}{source.suffix.lower()}"
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        manifest.append({"path": relative.as_posix(), "label": "real", "split": "calibration",
                         "group": f"rwfs:{source.name}", "source": "rwfs-celeb"})
    reddit_counts = Counter()
    for offset in range(0, 5000, 100):
        payload = request_json(f"{REDDIT_API}&offset={offset}&length=100")
        for entry in payload["rows"]:
            row = entry["row"]
            label = row["label"]
            if (row["type"] != "image" or label not in {"real", "fake"} or reddit_counts[label] >= 100
                    or row["image"]["width"] * row["image"]["height"] > 20_000_000):
                continue
            index = entry["row_idx"]
            relative = Path("reddit") / label / f"{index}.jpg"
            downloads.append((row["image"]["src"], root / relative))
            manifest.append({"path": relative.as_posix(), "label": label, "split": "test",
                             "group": f"reddit:{index}", "source": "reddit"})
            reddit_counts[label] += 1
        if reddit_counts == {"real": 100, "fake": 100}:
            break
    if reddit_counts != {"real": 100, "fake": 100}:
        raise ValueError(f"OpenFake Reddit sample quotas were not met: {dict(reddit_counts)}")
    with ThreadPoolExecutor(max_workers=8) as pool:
        for number, _ in enumerate(pool.map(download, downloads), 1):
            if number % 25 == 0:
                print(f"Downloaded {number}/{len(downloads)}", flush=True)
    with (root / "manifest.csv").open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=("path", "label", "split", "group", "source"))
        writer.writeheader()
        writer.writerows(manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("datasets/OpenFake-sample"))
    parser.add_argument("--rwfs-real", type=Path,
                        default=Path("datasets/RWFS/real-20250420T081907Z-001/real"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sample(args.out, args.rwfs_real)


if __name__ == "__main__":
    main()
