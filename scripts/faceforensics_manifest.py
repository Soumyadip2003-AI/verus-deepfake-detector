"""Create a Verity validation manifest from an approved FaceForensics++ video download."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from urllib.request import urlopen

SPLIT_URL = "https://raw.githubusercontent.com/ondyari/FaceForensics/master/dataset/splits/{name}.json"
METHODS = ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures")
FIELDS = ("path", "label", "split", "group", "slice", "method")


def read_split(name: str, directory: Path) -> list[tuple[str, str]]:
    path = directory / f"{name}.json"
    if not path.is_file():
        directory.mkdir(parents=True, exist_ok=True)
        try:
            with urlopen(SPLIT_URL.format(name=name), timeout=20) as response:
                path.write_bytes(response.read())
        except OSError as exc:
            raise ValueError(f"Could not fetch official {name}.json; place it in {directory}") from exc
    pairs = json.loads(path.read_text())
    if not isinstance(pairs, list) or not all(
        isinstance(pair, list) and len(pair) == 2 and
        all(isinstance(item, str) and item.isdigit() and len(item) == 3 for item in pair)
        for pair in pairs
    ):
        raise ValueError(f"Invalid FaceForensics++ split: {path}")
    ids = [item for pair in pairs for item in pair]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Repeated source ID in {path}")
    return [tuple(pair) for pair in pairs]


def build_manifest(root: Path, split_dir: Path, methods: tuple[str, ...], output: Path) -> int:
    if not methods or len(methods) != len(set(methods)) or set(methods) - set(METHODS):
        raise ValueError("Choose one or more distinct FaceForensics++ methods.")
    pairs = {"calibration": read_split("val", split_dir), "test": read_split("test", split_dir)}
    ids = {split: {item for pair in values for item in pair} for split, values in pairs.items()}
    train_ids = {item for pair in read_split("train", split_dir) for item in pair}
    if ids["calibration"] & ids["test"]:
        raise ValueError("Official validation and test source IDs overlap.")
    if (ids["calibration"] | ids["test"]) & train_ids:
        raise ValueError("Evaluation source IDs overlap the FaceForensics++ training split.")
    rows = []
    missing = []
    for split, values in pairs.items():
        for first, second in values:
            for target, source in ((first, second), (second, first)):
                original = Path("original_sequences/youtube/c23/videos") / f"{target}.mp4"
                candidates = [(original, "real", "original")]
                candidates.extend(
                    (Path("manipulated_sequences") / method / "c23/videos" / f"{target}_{source}.mp4",
                     "fake", method) for method in methods
                )
                for relative, label, method in candidates:
                    if not (root / relative).is_file():
                        missing.append(str(relative))
                    else:
                        rows.append({"path": relative.as_posix(), "label": label, "split": split,
                                     "group": target, "slice": "c23-video", "method": method})
    if missing:
        examples = "\n".join(missing[:8])
        raise ValueError(f"The approved FaceForensics++ download is incomplete ({len(missing)} missing files).\n{examples}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Directory containing original_sequences and manipulated_sequences")
    parser.add_argument("output", type=Path, help="Manifest CSV to create")
    parser.add_argument("--split-dir", type=Path, default=Path("evaluation/faceforensics_splits"))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    args = parser.parse_args()
    try:
        count = build_manifest(args.root, args.split_dir, tuple(args.methods), args.output)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1) from None
    print(f"Wrote {count} labeled videos to {args.output}")


if __name__ == "__main__":
    main()
