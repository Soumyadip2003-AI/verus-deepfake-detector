"""Score labeled media, calibrate abstaining thresholds, and test held-out data."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from backend.app import CHECKPOINT, FACE_MODEL, THRESHOLD_VERSION, Detector, classify, pipeline_hashes, sha256

MIN_PER_CLASS = 100
MIN_GROUPS_PER_CLASS = 100
MAX_ERROR_UPPER = 0.05
MIN_COVERAGE = 0.20
MAX_NO_FACE_RATE = 0.10
FIELDS = ("path", "label", "split", "group", "slice", "method", "fake_score", "frames_with_faces",
          "model_sha256", "face_model_sha256", "pipeline_sha256", "clip_config_sha256",
          "clip_preprocessor_sha256")
SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".mp4", ".webm", ".mov"}


def rows_from(path: Path, required: set[str]) -> list[dict[str, str]]:
    with path.open(newline="") as source:
        reader = csv.DictReader(source)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path} needs columns: {', '.join(sorted(required))}")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"{path} has duplicate columns.")
        rows = list(reader)
    if not rows:
        raise ValueError("The manifest is empty.")
    groups: dict[str, str] = {}
    paths = set()
    for row in rows:
        if any(row[field] is None for field in required):
            raise ValueError(f"{path} contains an incomplete row.")
        if row["label"] not in {"real", "fake"} or row["split"] not in {"calibration", "test"}:
            raise ValueError("Labels must be real/fake and splits must be calibration/test.")
        if not row["group"] or not row["path"] or row["path"] in paths:
            raise ValueError("Every media path and source group must be nonempty; paths must be unique.")
        paths.add(row["path"])
        if row["group"] in groups and groups[row["group"]] != row["split"]:
            raise ValueError(f"Source group {row['group']} appears in both splits.")
        groups[row["group"]] = row["split"]
    return rows


def score_media(manifest: Path, root: Path, output: Path) -> None:
    destination = output.resolve()
    if destination == manifest.resolve():
        raise ValueError("Score output must differ from the input manifest.")
    rows = rows_from(manifest, {"path", "label", "split", "group"})
    dataset_root = root.resolve()
    media_paths = [(root / row["path"]).resolve() for row in rows]
    if destination in media_paths:
        raise ValueError("Score output must differ from input media.")
    output.unlink(missing_ok=True)
    seen = set()
    for row, media in zip(rows, media_paths):
        if not media.is_relative_to(dataset_root) or not media.is_file() or media.suffix.lower() not in SUFFIXES:
            raise ValueError(f"Invalid media path: {row['path']}")
        if media in seen:
            raise ValueError(f"Duplicate media file: {row['path']}")
        seen.add(media)
    previous_device = os.environ.get("VERITY_DEVICE")
    os.environ["VERITY_DEVICE"] = "cpu"
    try:
        detector = Detector()
    finally:
        if previous_device is None:
            os.environ.pop("VERITY_DEVICE", None)
        else:
            os.environ["VERITY_DEVICE"] = previous_device
    model_hash, face_hash = sha256(CHECKPOINT), sha256(FACE_MODEL)
    pipeline = pipeline_hashes()
    for index, (row, media) in enumerate(zip(rows, media_paths), 1):
        result = detector.analyze(media, media.suffix.lower() in {".mp4", ".webm", ".mov"})
        row["fake_score"] = "" if result["fake_score"] is None else str(result["fake_score"])
        row["frames_with_faces"] = str(result["frames_with_faces"])
        row["slice"] = row.get("slice", "")
        row["method"] = row.get("method", "")
        row["model_sha256"], row["face_model_sha256"] = model_hash, face_hash
        row.update(pipeline)
        if index % 25 == 0 or index == len(rows):
            print(f"Scored {index}/{len(rows)}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def wilson_upper(errors: int, total: int) -> float:
    """95% one-sided Wilson upper bound for a binomial error rate."""
    if total == 0:
        return 1.0
    z = 1.645
    p = errors / total
    a = z * z / total
    return (p + a / 2 + z * math.sqrt(p * (1 - p) / total + a / (4 * total))) / (1 + a)


def scored_rows(path: Path) -> list[dict]:
    rows = rows_from(path, set(FIELDS) - {"slice", "method"})
    expected = {"model_sha256": sha256(CHECKPOINT), "face_model_sha256": sha256(FACE_MODEL),
                **pipeline_hashes()}
    for row in rows:
        if any(row[key] != value for key, value in expected.items()):
            raise ValueError("Scores were produced with different model files or inference code.")
        raw = row["fake_score"]
        row["score"] = None if raw == "" else float(raw)
        if row["score"] is not None and (not math.isfinite(row["score"]) or not 0 <= row["score"] <= 1):
            raise ValueError(f"Invalid score for {row['path']}")
        try:
            frames_with_faces = int(row["frames_with_faces"])
        except ValueError as exc:
            raise ValueError(f"Invalid face count for {row['path']}") from exc
        if frames_with_faces < 0 or (row["score"] is None) != (frames_with_faces == 0):
            raise ValueError(f"Score and face count disagree for {row['path']}")
    return rows


def class_rows(rows: list[dict], split: str, label: str) -> list[dict]:
    return [r for r in rows if r["split"] == split and r["label"] == label and r["score"] is not None]


def metrics(rows: list[dict], real_max: float, fake_min: float) -> dict:
    counts = Counter(r["label"] for r in rows)
    detectable = [r for r in rows if r["score"] is not None]
    real = [r for r in detectable if r["label"] == "real"]
    fake = [r for r in detectable if r["label"] == "fake"]
    predicted = Counter((r["label"], classify(r["score"], real_max, fake_min)) for r in detectable)
    false_real = predicted[("fake", "no_strong_signal")]
    false_fake = predicted[("real", "likely_manipulated")]
    real_groups = {r["group"] for r in real}
    fake_groups = {r["group"] for r in fake}
    false_real_groups = {r["group"] for r in fake if classify(r["score"], real_max, fake_min) == "no_strong_signal"}
    false_fake_groups = {r["group"] for r in real if classify(r["score"], real_max, fake_min) == "likely_manipulated"}
    real_coverage = predicted[("real", "no_strong_signal")] / counts["real"] if counts["real"] else 0
    fake_coverage = predicted[("fake", "likely_manipulated")] / counts["fake"] if counts["fake"] else 0
    return {
        "media": len(rows), "labels": dict(counts), "no_face": len(rows) - len(detectable),
        "detectable_real": len(real), "detectable_fake": len(fake),
        "source_groups_real": len(real_groups), "source_groups_fake": len(fake_groups),
        "false_real": false_real, "false_fake": false_fake,
        "false_real_source_groups": len(false_real_groups),
        "false_fake_source_groups": len(false_fake_groups),
        "false_real_upper_95": wilson_upper(len(false_real_groups), len(fake_groups)),
        "false_fake_upper_95": wilson_upper(len(false_fake_groups), len(real_groups)),
        "real_coverage": real_coverage, "fake_coverage": fake_coverage,
        "inconclusive": sum(classify(r["score"], real_max, fake_min) == "inconclusive" for r in rows),
    }


def calibrate(scores: Path, report_path: Path, thresholds_path: Path) -> bool:
    if len({path.resolve() for path in (scores, report_path, thresholds_path)}) != 3:
        raise ValueError("Scores, report, and thresholds paths must differ.")
    thresholds_path.unlink(missing_ok=True)
    report_path.unlink(missing_ok=True)
    rows = scored_rows(scores)
    totals = Counter((row["split"], row["label"]) for row in rows)
    cal_real, cal_fake = class_rows(rows, "calibration", "real"), class_rows(rows, "calibration", "fake")
    test_real, test_fake = class_rows(rows, "test", "real"), class_rows(rows, "test", "fake")
    problems = []
    for split, label, subset in (("calibration", "real", cal_real), ("calibration", "fake", cal_fake),
                                 ("test", "real", test_real), ("test", "fake", test_fake)):
        if len(subset) < MIN_PER_CLASS:
            problems.append(f"{split}/{label}: need {MIN_PER_CLASS} media with detectable faces; found {len(subset)}")
        if len({r["group"] for r in subset}) < MIN_GROUPS_PER_CLASS:
            problems.append(f"{split}/{label}: need {MIN_GROUPS_PER_CLASS} distinct source groups")
        total = sum(r["split"] == split and r["label"] == label for r in rows)
        if total and 1 - len(subset) / total > MAX_NO_FACE_RATE:
            problems.append(f"{split}/{label}: more than {MAX_NO_FACE_RATE:.0%} had no detectable face")
    if not all((cal_real, cal_fake, test_real, test_fake)):
        report = {"passed": False, "problems": problems}
    else:
        candidates = sorted({0.0, 1.0, *(r["score"] for r in cal_real + cal_fake)})
        real_choices = [t for t in candidates if
                        wilson_upper(len({r["group"] for r in cal_fake if r["score"] <= t}),
                                     len({r["group"] for r in cal_fake})) <= MAX_ERROR_UPPER
                        and sum(r["score"] <= t for r in cal_real) / totals[("calibration", "real")] >= MIN_COVERAGE]
        fake_choices = [t for t in candidates if
                        wilson_upper(len({r["group"] for r in cal_real if r["score"] >= t}),
                                     len({r["group"] for r in cal_real})) <= MAX_ERROR_UPPER
                        and sum(r["score"] >= t for r in cal_fake) / totals[("calibration", "fake")] >= MIN_COVERAGE]
        if not real_choices or not fake_choices or max(real_choices) >= min(fake_choices):
            problems.append("No thresholds meet the calibration error and coverage requirements.")
            report = {"passed": False, "problems": problems}
        else:
            real_max, fake_min = max(real_choices), min(fake_choices)
            calibration = metrics([r for r in rows if r["split"] == "calibration"], real_max, fake_min)
            test = metrics([r for r in rows if r["split"] == "test"], real_max, fake_min)
            for name in ("false_real_upper_95", "false_fake_upper_95"):
                if test[name] > MAX_ERROR_UPPER:
                    problems.append(f"Held-out {name} exceeds {MAX_ERROR_UPPER}.")
            for name in ("real_coverage", "fake_coverage"):
                if test[name] < MIN_COVERAGE:
                    problems.append(f"Held-out {name} is below {MIN_COVERAGE}.")
            slices = defaultdict(list)
            methods = defaultdict(list)
            for row in rows:
                if row["split"] == "test" and row.get("slice"):
                    slices[row["slice"]].append(row)
                if row["split"] == "test" and row["label"] == "fake" and row.get("method"):
                    methods[row["method"]].append(row)
            method_reports = {}
            for name, subset in methods.items():
                stats = metrics(subset, real_max, fake_min)
                method_reports[name] = {key: stats[key] for key in
                                        ("media", "no_face", "source_groups_fake", "false_real_source_groups",
                                         "false_real_upper_95", "fake_coverage")}
                if stats["source_groups_fake"] < MIN_GROUPS_PER_CLASS:
                    problems.append(f"Held-out {name}: too few source groups.")
                if stats["no_face"] / stats["media"] > MAX_NO_FACE_RATE:
                    problems.append(f"Held-out {name}: too many files without a detectable face.")
                if stats["false_real_upper_95"] > MAX_ERROR_UPPER or stats["fake_coverage"] < MIN_COVERAGE:
                    problems.append(f"Held-out {name}: error or coverage requirement failed.")
            report = {"passed": not problems, "problems": problems, "real_max": real_max,
                      "fake_min": fake_min, "calibration": calibration, "test": test,
                      "test_slices": {name: metrics(subset, real_max, fake_min) for name, subset in slices.items()},
                      "test_methods": method_reports}
            if not problems:
                config = {"version": THRESHOLD_VERSION, "real_max": real_max, "fake_min": fake_min,
                          "model_sha256": sha256(CHECKPOINT), "face_model_sha256": sha256(FACE_MODEL),
                          **pipeline_hashes(),
                          "validated_at": datetime.now(timezone.utc).isoformat(),
                          "target_error_upper_95": MAX_ERROR_UPPER, "minimum_coverage": MIN_COVERAGE,
                          "calibration": calibration, "test": test}
                thresholds_path.parent.mkdir(parents=True, exist_ok=True)
                thresholds_path.write_text(json.dumps(config, indent=2) + "\n")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report["passed"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    scorer = commands.add_parser("score", help="Run the detector on a labeled media manifest")
    scorer.add_argument("manifest", type=Path)
    scorer.add_argument("output", type=Path)
    scorer.add_argument("--root", type=Path, default=Path("datasets"))
    calibrator = commands.add_parser("calibrate", help="Select thresholds and check a held-out test split")
    calibrator.add_argument("scores", type=Path)
    calibrator.add_argument("report", type=Path)
    calibrator.add_argument("thresholds", type=Path)
    args = parser.parse_args()
    if args.command == "score":
        score_media(args.manifest, args.root, args.output)
    elif not calibrate(args.scores, args.report, args.thresholds):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
