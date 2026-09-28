"""Download a reproducible, paired HiDF video sample using ZIP range requests.

The sample is for exploratory test reporting, not threshold calibration.
Dataset: https://zenodo.org/records/16140829 (CC BY-NC 4.0).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import random
import re
import ssl
import struct
import time
import urllib.error
import urllib.request
import zlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


RECORD = "https://zenodo.org/api/records/16140829/files"
ARCHIVES = {
    "Fake-vid.zip": (715291222, "708890ccaedc3a6cccee0c1d3e64a711"),
    "Real-vid.zip": (842885152, "ee53b75850a7f54b45d61a78085eda0a"),
}
METADATA_MD5 = "36357e2e43ee9155e314c051c6eb45a8"
SYSTEM_CERTS = Path("/etc/ssl/cert.pem")
TLS_CONTEXT = ssl.create_default_context(cafile=str(SYSTEM_CERTS) if SYSTEM_CERTS.is_file() else None)


@dataclass(frozen=True)
class Member:
    name: str
    offset: int
    stop: int
    compressed_size: int
    size: int
    crc: int
    method: int


def get(url: str, start: int | None = None, stop: int | None = None) -> bytes:
    headers = {"User-Agent": "Verity-HiDF-evaluation/1.0"}
    if start is not None:
        headers["Range"] = f"bytes={start}-{stop}"
    for attempt in range(10):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers),
                                        context=TLS_CONTEXT, timeout=45) as response:
                expected = stop - start + 1 if start is not None else 2_000_000
                if start is not None:
                    wanted = f"bytes {start}-{stop}/"
                    if response.status != 206 or not response.headers.get("Content-Range", "").startswith(wanted):
                        raise ValueError(f"Server did not honor ZIP range {start}-{stop}")
                data = response.read(expected + 1)
                if len(data) > expected or (start is not None and len(data) != expected):
                    raise ValueError("Unexpected download size")
                return data
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504) or attempt == 9:
                raise
            try:
                delay = float(exc.headers.get("Retry-After", ""))
            except ValueError:
                delay = 2**attempt
            time.sleep(min(60, max(1, delay)))
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, ssl.SSLCertVerificationError) or attempt == 9:
                raise
            time.sleep(min(30, 2**attempt))
    raise RuntimeError("Download retries exhausted")


def zip_index(archive: str) -> dict[str, Member]:
    length, _ = ARCHIVES[archive]
    url = f"{RECORD}/{archive}/content"
    tail_start = length - 65557
    tail = get(url, tail_start, length - 1)
    end = tail.rfind(b"PK\x05\x06")
    if end < 0:
        raise ValueError(f"Missing ZIP directory: {archive}")
    _, disk, cd_disk, on_disk, count, cd_size, cd_offset, comment = struct.unpack_from("<4s4H2IH", tail, end)
    if disk or cd_disk or on_disk != count or count == 0xffff or cd_offset + cd_size + 22 + comment != length:
        raise ValueError(f"Unsupported or changed ZIP directory: {archive}")
    directory = get(url, cd_offset, cd_offset + cd_size - 1)
    entries = []
    pos = 0
    while pos < len(directory):
        header = struct.unpack_from("<4s6H3I5H2I", directory, pos)
        if header[0] != b"PK\x01\x02" or header[3] or header[13]:
            raise ValueError(f"Unsupported ZIP entry: {archive}")
        name_len, extra_len, comment_len = header[10:13]
        name = directory[pos + 46:pos + 46 + name_len].decode("utf-8")
        entries.append((name, header[16], header[8], header[9], header[7], header[4]))
        pos += 46 + name_len + extra_len + comment_len
    if pos != cd_size or len(entries) != count:
        raise ValueError(f"Damaged ZIP directory: {archive}")
    entries.sort(key=lambda row: row[1])
    indexed = {}
    for index, (name, offset, compressed, size, crc, method) in enumerate(entries):
        next_offset = entries[index + 1][1] if index + 1 < len(entries) else cd_offset
        if next_offset - offset < 30 + compressed or method not in (0, 8):
            raise ValueError(f"Damaged ZIP entry: {name}")
        indexed[name] = Member(name, offset, next_offset - 1, compressed, size, crc, method)
    return indexed


def metadata() -> dict[str, dict[str, str]]:
    data = get(f"{RECORD}/metadata.csv/content")
    if hashlib.md5(data).hexdigest() != METADATA_MD5:
        raise ValueError("HiDF metadata checksum changed")
    rows = csv.DictReader(io.StringIO(data.decode("utf-8-sig")))
    return {row["ID"]: row for row in rows if row["Img/Vid"] == "Vid" and row["Base/Swap"] == "Base"}


def media_members(indexed: dict[str, Member], kind: str) -> dict[str, Member]:
    matches = {}
    pattern = rf"{kind}/([cf]\d{{3,5}})(?:_([cf]\d{{5}}))?\.mp4"
    for name, member in indexed.items():
        match = re.fullmatch(pattern, name)
        if match:
            if kind == "Fake-vid" and match[2] is None:
                raise ValueError(f"Unexpected fake filename: {name}")
            if match[1] in matches:
                raise ValueError(f"Duplicate source ID in {kind}: {match[1]}")
            matches[match[1]] = member
        elif name != f"{kind}/":
            raise ValueError(f"Unexpected ZIP path: {name}")
    return matches


def extract(archive: str, member: Member, root: Path) -> None:
    # Output names came from the strict archive-path pattern in media_members.
    destination = root / member.name
    if not destination.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"ZIP path escapes output directory: {member.name}")
    if destination.is_file():
        existing = destination.read_bytes()
        if len(existing) == member.size and zlib.crc32(existing) == member.crc:
            return
    data = get(f"{RECORD}/{archive}/content", member.offset, member.stop)
    local = struct.unpack_from("<4s5H3I2H", data)
    if local[0] != b"PK\x03\x04" or local[2] or local[3] != member.method:
        raise ValueError(f"Invalid local ZIP header: {member.name}")
    name_len, extra_len = local[9:11]
    if data[30:30 + name_len].decode("utf-8") != member.name:
        raise ValueError(f"ZIP filename mismatch: {member.name}")
    start = 30 + name_len + extra_len
    compressed = data[start:start + member.compressed_size]
    if len(compressed) != member.compressed_size:
        raise ValueError(f"Truncated ZIP entry: {member.name}")
    plain = zlib.decompress(compressed, -15) if member.method == 8 else compressed
    if len(plain) != member.size or zlib.crc32(plain) != member.crc:
        raise ValueError(f"ZIP CRC mismatch: {member.name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    temporary.write_bytes(plain)
    os.replace(temporary, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--out", type=Path, default=Path("datasets/HiDF"))
    parser.add_argument("--plan", action="store_true", help="inspect selection without downloading videos")
    args = parser.parse_args()
    if args.pairs < 1 or args.workers < 1:
        parser.error("pairs and workers must be positive")
    fake = media_members(zip_index("Fake-vid.zip"), "Fake-vid")
    real = media_members(zip_index("Real-vid.zip"), "Real-vid")
    facts = metadata()
    if set(fake) != set(real) or set(fake) != set(facts):
        raise ValueError("HiDF real/fake/metadata source IDs differ")
    if args.pairs > len(fake):
        parser.error(f"HiDF has only {len(fake)} pairs")
    rng = random.Random(args.seed)
    sources = {prefix: sorted(base for base in fake if base.startswith(prefix)) for prefix in ("c", "f")}
    f_count = round(args.pairs * len(sources["f"]) / len(fake))
    c_count = args.pairs - f_count
    selected = rng.sample(sources["c"], c_count) + rng.sample(sources["f"], f_count)
    rng.shuffle(selected)
    targets = [fake[base].name.rsplit("/", 1)[-1].split("_", 1)[1][:-4] for base in selected]
    reused_targets = sum(count - 1 for count in Counter(targets).values())
    size = sum(real[base].compressed_size + fake[base].compressed_size for base in selected)
    print(f"HiDF paired sample: {len(selected)} sources, {len(sources['c'])} c and {len(sources['f'])} f in release")
    print(f"Selected c={c_count}, f={f_count}; repeated target IDs={reused_targets}; download={size / 1e6:.1f} MB")
    if args.plan:
        return
    jobs = [("Real-vid.zip", real[base]) for base in selected] + [("Fake-vid.zip", fake[base]) for base in selected]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(extract, archive, member, args.out) for archive, member in jobs]
        for done, future in enumerate(as_completed(futures), 1):
            future.result()
            if done % 50 == 0 or done == len(jobs):
                print(f"Verified {done}/{len(jobs)} videos", flush=True)
    manifest = args.out / "manifest.csv"
    with manifest.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=("path", "label", "split", "group", "slice", "method"))
        writer.writeheader()
        for base in selected:
            for label, member in (("real", real[base]), ("fake", fake[base])):
                writer.writerow({"path": member.name, "label": label, "split": "test", "group": base,
                                 "slice": f"source_{base[0]}", "method": "face_swap" if label == "fake" else ""})
    provenance = {"source": "https://zenodo.org/records/16140829", "seed": args.seed, "pairs": len(selected),
                  "selected_source_counts": {"c": c_count, "f": f_count}, "reused_target_ids": reused_targets,
                  "verified_metadata_md5": METADATA_MD5,
                  "published_archive_md5": {name: md5 for name, (_, md5) in ARCHIVES.items()},
                  "purpose": "exploratory test summary; no threshold calibration"}
    (args.out / "selection.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Wrote {manifest}")


if __name__ == "__main__":
    main()
