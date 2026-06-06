"""RDD2022 -> YOLO dataset with per-country splits and the data files for the Norway experiments.

RDD2022 (Arya et al., 2022) has 47k road photos from Japan, India, Czech Republic, Norway,
United States and China, annotated with four damage types. The Norwegian part (E14 Trondheim -
Sundsvall and county roads in Innlandet) was collected by Statens vegvesen. Official test
labels are not public, so the annotated ``train`` folder of each country is split into
train/val/test in contiguous blocks of file names (neighbouring frames stay together).

    python -m road_damage.datasets.rdd2022 --raw data/RDD2022 --out data/rdd2022_yolo

Output: images/ and labels/ (symlinks + YOLO txt), <country>_<split>.txt image lists and data/*.yaml.
The lists sit next to images/ because Ultralytics resolves "./" entries relative to the list file.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

CLASSES = ("D00", "D10", "D20", "D40")
CLASS_NAMES = ("longitudinal crack", "transverse crack", "alligator crack", "pothole")
COUNTRIES = ("Czech", "India", "Japan", "Norway", "United_States", "China_MotorBike", "China_Drone")
SPLITS = {"train": 0.7, "val": 0.15, "test": 0.15}
NORWAY_FRACTIONS = (0.1, 0.25, 0.5, 1.0)


def parse_voc(xml_path: Path) -> tuple[int, int, list[tuple[int, float, float, float, float]]]:
    """(width, height, [(class, cx, cy, w, h) normalised]) keeping only the four damage classes."""
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    width, height = int(float(size.findtext("width"))), int(float(size.findtext("height")))
    boxes = []
    for obj in root.iter("object"):
        name = (obj.findtext("name") or "").strip()
        if name not in CLASSES:
            continue
        bb = obj.find("bndbox")
        x0, y0, x1, y1 = (float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        x0, x1 = max(0.0, min(x0, x1)), min(float(width), max(x0, x1))
        y0, y1 = max(0.0, min(y0, y1)), min(float(height), max(y0, y1))
        if x1 - x0 < 1 or y1 - y0 < 1:
            continue
        boxes.append(
            (
                CLASSES.index(name),
                (x0 + x1) / 2 / width,
                (y0 + y1) / 2 / height,
                (x1 - x0) / width,
                (y1 - y0) / height,
            )
        )
    return width, height, boxes


def find_country_dirs(raw: Path) -> dict[str, Path]:
    """<country>/train folders anywhere below ``raw`` (the archive nests them differently by release)."""
    found = {}
    for country in COUNTRIES:
        for candidate in raw.rglob(country):
            if (candidate / "train" / "annotations" / "xmls").is_dir():
                found[country] = candidate / "train"
                break
    return found


def block_split(n: int, seed: int = 0) -> np.ndarray:
    """Split sorted files into 20 contiguous blocks, then assign whole blocks to splits."""
    blocks = np.array_split(np.arange(n), 20)
    order = np.random.default_rng(seed).permutation(len(blocks))
    split = np.empty(n, dtype=object)
    cut_test, cut_val = round(20 * SPLITS["test"]), round(20 * (SPLITS["test"] + SPLITS["val"]))
    for rank, b in enumerate(order):
        split[blocks[b]] = "test" if rank < cut_test else ("val" if rank < cut_val else "train")
    return split


def write_yaml(
    path: Path, root: Path, train: list[str], val: list[str], test: list[str] | None = None
) -> None:
    lines = [f"path: {root.resolve()}", f"train: {json.dumps(train)}", f"val: {json.dumps(val)}"]
    if test:
        lines.append(f"test: {json.dumps(test)}")
    lines.append("names:")
    lines += [f"  {i}: {c}" for i, c in enumerate(CLASSES)]
    path.write_text("\n".join(lines) + "\n")


def convert(raw: Path, out: Path, seed: int = 0) -> dict:
    countries = find_country_dirs(raw)
    if not countries:
        raise FileNotFoundError(f"no <country>/train/annotations/xmls under {raw}")
    stats = {}
    for d in ("images", "labels", "data"):
        (out / d).mkdir(parents=True, exist_ok=True)
    for country, train_dir in sorted(countries.items()):
        xmls = sorted((train_dir / "annotations" / "xmls").glob("*.xml"))
        split = block_split(len(xmls), seed)
        lists = {s: [] for s in SPLITS}
        counts = np.zeros(len(CLASSES), int)
        for xml, s in zip(xmls, split, strict=True):
            image = train_dir / "images" / f"{xml.stem}.jpg"
            if not image.exists():
                continue
            _, _, boxes = parse_voc(xml)
            target = out / "images" / f"{xml.stem}.jpg"
            if not target.exists():
                os.symlink(image.resolve(), target)
            (out / "labels" / f"{xml.stem}.txt").write_text(
                "".join(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n" for c, cx, cy, w, h in boxes)
            )
            lists[s].append(f"./images/{xml.stem}.jpg")
            for c, *_ in boxes:
                counts[c] += 1
        for s, items in lists.items():
            (out / f"{country}_{s}.txt").write_text("\n".join(items) + "\n")
        stats[country] = {
            "images": {s: len(v) for s, v in lists.items()},
            "boxes": dict(zip(CLASSES, counts.tolist(), strict=True)),
        }
        log.info("%s: %s", country, stats[country])

    def lst(country: str, s: str) -> str:
        return f"{country}_{s}.txt"

    present = sorted(countries)
    others = [c for c in present if c != "Norway"]
    data = out / "data"
    write_yaml(
        data / "all.yaml",
        out,
        [lst(c, "train") for c in present],
        [lst(c, "val") for c in present],
        [lst(c, "test") for c in present],
    )
    write_yaml(
        data / "no_norway.yaml", out, [lst(c, "train") for c in others], [lst(c, "val") for c in others]
    )
    for country in present:  # evaluation on one country's held-out test block
        write_yaml(data / f"test_{country}.yaml", out, [lst(country, "train")], [lst(country, "test")])
    if "Norway" in countries:
        norway_train = (out / lst("Norway", "train")).read_text().split()
        rng = np.random.default_rng(seed)
        for fraction in NORWAY_FRACTIONS:  # fine-tuning with a growing share of Norwegian data
            subset = sorted(rng.permutation(norway_train)[: max(1, int(len(norway_train) * fraction))])
            name = f"Norway_train_{int(fraction * 100)}"
            (out / f"{name}.txt").write_text("\n".join(subset) + "\n")
            write_yaml(
                data / f"norway_{int(fraction * 100)}.yaml", out, [f"{name}.txt"], [lst("Norway", "val")]
            )
        write_replay(out, present, seed)
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    return stats


def write_replay(out: Path, countries: list[str], seed: int = 0) -> Path:
    """All Norwegian training photos plus as many replayed from the other countries.

    Fine-tuning on Norway alone forgets the other countries; mixing their photos back in keeps
    them. Validation covers every country, so checkpoint selection also watches for forgetting."""
    others = [c for c in countries if c != "Norway"]
    norway = (out / "Norway_train.txt").read_text().split()
    pool = sorted(line for c in others for line in (out / f"{c}_train.txt").read_text().split())
    replay = sorted(np.random.default_rng(seed).permutation(pool)[: len(norway)])
    (out / "Others_replay.txt").write_text("\n".join(replay) + "\n")
    path = out / "data" / "norway_replay.yaml"
    write_yaml(path, out, ["Norway_train.txt", "Others_replay.txt"], [f"{c}_val.txt" for c in countries])
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--raw", type=Path, help="extracted RDD2022 folder")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--replay-only", action="store_true", help="only (re)write the replay split")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.replay_only:
        countries = sorted(p.name.removesuffix("_train.txt") for p in args.out.glob("*_train.txt"))
        print(write_replay(args.out, countries))
        return
    if args.raw is None:
        parser.error("--raw is required unless --replay-only")
    stats = convert(args.raw, args.out)
    print(json.dumps({c: s["images"] for c, s in stats.items()}, indent=2))


if __name__ == "__main__":
    main()
