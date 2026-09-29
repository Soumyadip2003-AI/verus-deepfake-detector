"""Score and validate the still-image detector with abstaining thresholds."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from backend.photo_detector import CHECKPOINT, PhotoDetector, pipeline_hashes, sha256
from scripts.validate import auroc, classify, metrics, rows_from, wilson_upper

MIN_PER_CLASS = 100
CALIBRATION_ERROR_UPPER = 0.03
TEST_ERROR_UPPER = 0.10
MIN_COVERAGE = 0.20
VERSION = 1
FIELDS = ("path", "label", "split", "group", "source", "fake_score",
          "model_sha256", "photo_pipeline_sha256", "photo_config_sha256")


def score(manifest: Path, root: Path, output: Path) -> None:
    if manifest.resolve() == output.resolve():
        raise ValueError("Score output must differ from the manifest.")
    rows = rows_from(manifest, {"path", "label", "split", "group"})
    root = root.resolve()
    paths = [(root / row["path"]).resolve() for row in rows]
    if any(not path.is_relative_to(root) or not path.is_file() for path in paths):
        raise ValueError("The manifest contains an invalid image path.")
    hashes = {"model_sha256": sha256(CHECKPOINT), **pipeline_hashes()}
    cached = {}
    if output.is_file():
        with output.open(newline="") as source:
            for old in csv.DictReader(source):
                if all(old.get(key) == value for key, value in hashes.items()):
                    cached[old["path"]] = old["fake_score"]
    detector = None
    for index, (row, path) in enumerate(zip(rows, paths), 1):
        if row["path"] in cached:
            row["fake_score"] = cached[row["path"]]
        else:
            if detector is None:
                previous = os.environ.get("VERITY_DEVICE")
                os.environ["VERITY_DEVICE"] = "cpu"
                try:
                    detector = PhotoDetector()
                finally:
                    if previous is None:
                        os.environ.pop("VERITY_DEVICE", None)
                    else:
                        os.environ["VERITY_DEVICE"] = previous
            row["fake_score"] = str(detector.analyze(path)["fake_score"])
        row["source"] = row.get("source", "")
        row.update(hashes)
        if index % 25 == 0:
            print(f"Scored {index}/{len(rows)}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_scores(path: Path) -> list[dict]:
    rows = rows_from(path, set(FIELDS) - {"source"})
    expected = {"model_sha256": sha256(CHECKPOINT), **pipeline_hashes()}
    for row in rows:
        if any(row[key] != value for key, value in expected.items()):
            raise ValueError("Scores were produced with a different photo model or pipeline.")
        row["score"] = float(row["fake_score"])
        if not math.isfinite(row["score"]) or not 0 <= row["score"] <= 1:
            raise ValueError("A photo score is invalid.")
    return rows


def calibrate(scores: Path, report_path: Path, thresholds_path: Path) -> bool:
    if len({p.resolve() for p in (scores, report_path, thresholds_path)}) != 3:
        raise ValueError("Scores, report, and thresholds paths must differ.")
    report_path.unlink(missing_ok=True)
    thresholds_path.unlink(missing_ok=True)
    rows = load_scores(scores)
    problems = []
    for split in ("calibration", "test"):
        for label in ("real", "fake"):
            count = sum(r["split"] == split and r["label"] == label for r in rows)
            if count < MIN_PER_CLASS:
                problems.append(f"{split}/{label}: need {MIN_PER_CLASS} images; found {count}")
    calibration = [r for r in rows if r["split"] == "calibration"]
    test = [r for r in rows if r["split"] == "test"]
    candidates = sorted({0.0, 1.0, *(r["score"] for r in calibration)})
    cal_real = [r for r in calibration if r["label"] == "real"]
    cal_fake = [r for r in calibration if r["label"] == "fake"]
    real_choices = [t for t in candidates if wilson_upper(sum(r["score"] <= t for r in cal_fake), len(cal_fake)) <= CALIBRATION_ERROR_UPPER
                    and sum(r["score"] <= t for r in cal_real) / len(cal_real) >= MIN_COVERAGE]
    fake_choices = [t for t in candidates if wilson_upper(sum(r["score"] >= t for r in cal_real), len(cal_real)) <= CALIBRATION_ERROR_UPPER
                    and sum(r["score"] >= t for r in cal_fake) / len(cal_fake) >= MIN_COVERAGE]
    if not real_choices or not fake_choices or max(real_choices) >= min(fake_choices):
        problems.append("No thresholds meet the calibration error and coverage requirements.")
        report = {"passed": False, "problems": problems, "auroc": auroc(rows)}
    else:
        real_max, fake_min = max(real_choices), min(fake_choices)
        cal_stats, test_stats = metrics(calibration, real_max, fake_min), metrics(test, real_max, fake_min)
        for name in ("false_real_upper_95", "false_fake_upper_95"):
            if test_stats[name] > TEST_ERROR_UPPER:
                problems.append(f"Held-out {name} exceeds {TEST_ERROR_UPPER}.")
        for name in ("real_coverage", "fake_coverage"):
            if test_stats[name] < MIN_COVERAGE:
                problems.append(f"Held-out {name} is below {MIN_COVERAGE}.")
        by_source = defaultdict(list)
        for row in test:
            by_source[row.get("source", "unknown")].append(row)
        report = {"passed": not problems, "problems": problems, "real_max": real_max,
                  "fake_min": fake_min, "auroc": auroc(test), "calibration": cal_stats,
                  "test": test_stats,
                  "test_sources": {name: metrics(part, real_max, fake_min) for name, part in by_source.items()}}
        if not problems:
            thresholds_path.parent.mkdir(parents=True, exist_ok=True)
            thresholds_path.write_text(json.dumps({"version": VERSION, "real_max": real_max,
                "fake_min": fake_min, "model_sha256": sha256(CHECKPOINT), **pipeline_hashes(),
                "validated_at": datetime.now(timezone.utc).isoformat(),
                "target_error_upper_95": TEST_ERROR_UPPER, "calibration_error_upper_95": CALIBRATION_ERROR_UPPER,
                "minimum_coverage": MIN_COVERAGE,
                "calibration": cal_stats, "test": test_stats}, indent=2) + "\n")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report["passed"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    scorer = commands.add_parser("score")
    scorer.add_argument("manifest", type=Path)
    scorer.add_argument("output", type=Path)
    scorer.add_argument("--root", type=Path, default=Path("datasets/OpenFake-sample"))
    checker = commands.add_parser("calibrate")
    checker.add_argument("scores", type=Path)
    checker.add_argument("report", type=Path)
    checker.add_argument("thresholds", type=Path)
    args = parser.parse_args()
    if args.command == "score":
        score(args.manifest, args.root, args.output)
    elif not calibrate(args.scores, args.report, args.thresholds):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
