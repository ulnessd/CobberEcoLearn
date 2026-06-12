#!/usr/bin/env python3
"""
generate_bloom_dataset.py

Create a synthetic training dataset for CobberEcoBloom.

This script generates a balanced set of synthetic "drone-like" lake-surface
images with different levels of algal bloom coverage. It writes the images,
a supervisor CSV, and a short dataset summary.

The five classes are:

    clear_water
    mostly_clear
    field_check
    mostly_bloom
    dense_bloom

Typical usage:
    python generate_bloom_dataset.py 2000

Optional examples:
    python generate_bloom_dataset.py 8000 --size 192
    python generate_bloom_dataset.py 16000 --workers 20 --seed 123
    python generate_bloom_dataset.py 2000 --outroot Datasets

Dependencies:
    pip install numpy opencv-python
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


# ------------------------------------------------------------
# Class definitions
# ------------------------------------------------------------
CATEGORIES: Dict[str, Tuple[float, float]] = {
    "clear_water": (0.00, 0.04),
    "mostly_clear": (0.06, 0.15),
    "field_check": (0.25, 0.75),
    "mostly_bloom": (0.85, 0.94),
    "dense_bloom": (0.96, 1.00),
}

CATEGORY_ORDER = [
    "clear_water",
    "mostly_clear",
    "field_check",
    "mostly_bloom",
    "dense_bloom",
]


# ------------------------------------------------------------
# Image generation helpers
# ------------------------------------------------------------
def make_water_background(size: int, rng: np.random.Generator) -> np.ndarray:
    """Create a synthetic RGB water background."""
    y = np.linspace(0, 1, size)[:, None]
    x = np.linspace(0, 1, size)[None, :]

    base_r = 22 + 8 * y + 4 * np.sin(2 * np.pi * (x * 2.2 + y * 0.4))
    base_g = 92 + 20 * y + 8 * np.sin(2 * np.pi * (x * 1.4 - y * 0.7))
    base_b = 145 + 35 * (1 - y) + 8 * np.sin(2 * np.pi * (x * 1.8 + y * 1.1))

    img = np.dstack([base_r, base_g, base_b]).astype(np.float32)
    img += rng.normal(0, 5, (size, size, 3))

    # gentle wind-like streaks
    for _ in range(int(rng.integers(8, 18))):
        yy = int(rng.integers(0, size))
        x0 = int(rng.integers(0, size // 2))
        x1 = int(rng.integers(size // 2, size))
        color = (35, 120, 170)
        thickness = int(rng.integers(1, 3))
        cv2.line(img, (x0, yy), (x1, yy + int(rng.integers(-8, 8))), color, thickness)

    return np.clip(img, 0, 255).astype(np.uint8)


def generate_blob_mask(size: int, target_coverage: float, rng: np.random.Generator) -> np.ndarray:
    """Generate a patchy bloom mask with approximately the requested coverage."""
    if target_coverage <= 0.005:
        return np.zeros((size, size), dtype=np.uint8)

    if target_coverage >= 0.98:
        base = np.ones((size, size), dtype=np.float32)
        for _ in range(int(rng.integers(5, 14))):
            center = (int(rng.integers(0, size)), int(rng.integers(0, size)))
            axes = (int(rng.integers(8, 30)), int(rng.integers(8, 30)))
            angle = float(rng.uniform(0, 180))
            cv2.ellipse(base, center, axes, angle, 0, 360, 0.0, -1)
        base = cv2.GaussianBlur(base, (0, 0), sigmaX=5, sigmaY=5)
        threshold = np.quantile(base, 1.0 - target_coverage)
        return (base > threshold).astype(np.uint8)

    field = np.zeros((size, size), dtype=np.float32)
    n_blobs = int(np.interp(target_coverage, [0.02, 0.95], [3, 42]))
    n_blobs = max(1, n_blobs)

    for _ in range(n_blobs):
        center = (int(rng.integers(0, size)), int(rng.integers(0, size)))
        min_axis = int(np.interp(target_coverage, [0.02, 0.95], [7, 16]))
        max_axis = int(np.interp(target_coverage, [0.02, 0.95], [22, 52]))
        axes = (
            int(rng.integers(min_axis, max_axis + 1)),
            int(rng.integers(min_axis, max_axis + 1)),
        )
        angle = float(rng.uniform(0, 180))
        intensity = float(rng.uniform(0.5, 1.0))
        cv2.ellipse(field, center, axes, angle, 0, 360, intensity, -1)

    texture = rng.normal(0, 1, (size, size)).astype(np.float32)
    texture = cv2.GaussianBlur(texture, (0, 0), sigmaX=7, sigmaY=7)
    field = cv2.GaussianBlur(field, (0, 0), sigmaX=5, sigmaY=5)
    field = field + 0.45 * texture

    threshold = np.quantile(field, 1.0 - target_coverage)
    mask = (field > threshold).astype(np.uint8)

    mask_blur = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigmaX=1.1, sigmaY=1.1)
    mask = (mask_blur > 0.35).astype(np.uint8)
    return mask


def add_bloom_to_water(water: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Overlay green bloom patches on water."""
    img = water.astype(np.float32).copy()

    h, w = mask.shape
    bloom_color = np.zeros((h, w, 3), dtype=np.float32)
    bloom_color[:, :, 0] = rng.normal(65, 14, (h, w))
    bloom_color[:, :, 1] = rng.normal(150, 28, (h, w))
    bloom_color[:, :, 2] = rng.normal(55, 16, (h, w))

    streak = rng.normal(0, 1, (h, w)).astype(np.float32)
    streak = cv2.GaussianBlur(streak, (0, 0), sigmaX=8, sigmaY=2)
    bloom_color[:, :, 1] += 20 * streak
    bloom_color[:, :, 2] += 10 * streak

    alpha = (mask.astype(np.float32) * rng.uniform(0.62, 0.82))[:, :, None]
    img = (1 - alpha) * img + alpha * bloom_color

    # occasional shoreline edge for a little extra variety
    if rng.random() < 0.35:
        side = rng.choice(["top", "bottom", "left", "right"])
        shore_color = np.array([55, 75, 38], dtype=np.float32)
        thickness = int(rng.integers(8, 24))
        if side == "top":
            img[:thickness, :, :] = 0.65 * img[:thickness, :, :] + 0.35 * shore_color
        elif side == "bottom":
            img[-thickness:, :, :] = 0.65 * img[-thickness:, :, :] + 0.35 * shore_color
        elif side == "left":
            img[:, :thickness, :] = 0.65 * img[:, :thickness, :] + 0.35 * shore_color
        else:
            img[:, -thickness:, :] = 0.65 * img[:, -thickness:, :] + 0.35 * shore_color

    if rng.random() < 0.7:
        img = cv2.GaussianBlur(img, (3, 3), sigmaX=0.5)

    return np.clip(img, 0, 255).astype(np.uint8)


def make_bloom_image(size: int, coverage: float, seed: int):
    rng = np.random.default_rng(seed)
    water = make_water_background(size, rng)
    mask = generate_blob_mask(size, coverage, rng)
    actual_coverage = float(mask.mean())
    img = add_bloom_to_water(water, mask, rng)
    return img, actual_coverage


# ------------------------------------------------------------
# Dataset building helpers
# ------------------------------------------------------------
@dataclass
class Task:
    label: str
    index_in_class: int
    requested_coverage: float
    seed: int
    size: int
    outpath: str


def choose_worker_count(user_requested: int | None) -> int:
    if user_requested is not None:
        return max(1, int(user_requested))
    # Leave one core free if possible
    n = cpu_count() or 1
    return max(1, n - 1)


def counts_by_class(total_images: int, class_names: List[str]) -> Dict[str, int]:
    base = total_images // len(class_names)
    remainder = total_images % len(class_names)
    counts = {name: base for name in class_names}
    for i in range(remainder):
        counts[class_names[i]] += 1
    return counts


def make_tasks(total_images: int, size: int, seed: int, image_root: Path) -> List[Task]:
    rng = random.Random(seed)
    counts = counts_by_class(total_images, CATEGORY_ORDER)

    tasks: List[Task] = []
    for label in CATEGORY_ORDER:
        lo, hi = CATEGORIES[label]
        class_dir = image_root / label
        class_dir.mkdir(parents=True, exist_ok=True)

        for i in range(1, counts[label] + 1):
            requested = rng.uniform(lo, hi)
            image_seed = rng.randint(0, 2_000_000_000)
            filename = f"{label}_{i:06d}.png"
            outpath = class_dir / filename
            tasks.append(
                Task(
                    label=label,
                    index_in_class=i,
                    requested_coverage=requested,
                    seed=image_seed,
                    size=size,
                    outpath=str(outpath),
                )
            )

    # Shuffle task order so different classes are interleaved across workers
    rng.shuffle(tasks)
    return tasks


def worker_generate(task: Task):
    img_rgb, actual_coverage = make_bloom_image(task.size, task.requested_coverage, task.seed)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(task.outpath, img_bgr)
    if not ok:
        raise RuntimeError(f"Could not write image: {task.outpath}")

    return {
        "filename": Path(task.outpath).name,
        "relative_path": str(Path(task.outpath).relative_to(Path(task.outpath).parents[1])),
        "label": task.label,
        "requested_coverage": round(float(task.requested_coverage), 6),
        "actual_mask_coverage": round(float(actual_coverage), 6),
        "seed": int(task.seed),
    }


def assign_splits(records: List[dict], train_frac: float, val_frac: float, test_frac: float, seed: int) -> List[dict]:
    if not math.isclose(train_frac + val_frac + test_frac, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("train_frac + val_frac + test_frac must equal 1.0")

    rng = random.Random(seed)
    by_class: Dict[str, List[dict]] = {name: [] for name in CATEGORY_ORDER}
    for rec in records:
        by_class[rec["label"]].append(rec)

    output = []
    for label in CATEGORY_ORDER:
        items = by_class[label]
        rng.shuffle(items)

        n = len(items)
        n_train = int(round(train_frac * n))
        n_val = int(round(val_frac * n))
        # make sure total is exact
        n_test = n - n_train - n_val

        # safety adjustments if rounding causes weirdness
        if n_test < 0:
            n_test = 0
            n_val = n - n_train
        if n_val < 0:
            n_val = 0
            n_train = n

        for idx, rec in enumerate(items):
            if idx < n_train:
                rec["split"] = "train"
            elif idx < n_train + n_val:
                rec["split"] = "val"
            else:
                rec["split"] = "test"
            output.append(rec)

    return output


def write_supervisor_csv(records: List[dict], csv_path: Path):
    fieldnames = [
        "filename",
        "relative_path",
        "label",
        "requested_coverage",
        "actual_mask_coverage",
        "seed",
        "split",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow(rec)


def summarize_records(records: List[dict]) -> str:
    lines = []
    total = len(records)
    lines.append(f"Total images: {total}")
    lines.append("")

    by_label = Counter(rec["label"] for rec in records)
    by_split = Counter(rec["split"] for rec in records)
    lines.append("Counts by class:")
    for label in CATEGORY_ORDER:
        lines.append(f"  {label:14s} {by_label[label]}")
    lines.append("")

    lines.append("Counts by split:")
    for split in ["train", "val", "test"]:
        lines.append(f"  {split:14s} {by_split[split]}")
    lines.append("")

    lines.append("Counts by class and split:")
    for label in CATEGORY_ORDER:
        split_counts = Counter(rec["split"] for rec in records if rec["label"] == label)
        lines.append(
            f"  {label:14s} train={split_counts['train']:5d}   "
            f"val={split_counts['val']:5d}   test={split_counts['test']:5d}"
        )
    lines.append("")

    lines.append("Average actual bloom coverage by class:")
    for label in CATEGORY_ORDER:
        vals = [float(rec["actual_mask_coverage"]) for rec in records if rec["label"] == label]
        mean_val = sum(vals) / len(vals)
        lines.append(f"  {label:14s} {mean_val:.4f}")

    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a synthetic CobberEcoBloom dataset."
    )
    parser.add_argument(
        "total_images",
        type=int,
        help="Total number of images to generate (e.g. 2000, 8000, 16000).",
    )
    parser.add_argument(
        "--outroot",
        default=".",
        help="Root directory in which the dataset folder will be created.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=128,
        help="Image size in pixels (square images). Default: 128",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=17,
        help="Random seed. Default: 17",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes. Default: CPU count minus one.",
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.70,
        help="Training fraction. Default: 0.70",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.15,
        help="Validation fraction. Default: 0.15",
    )
    parser.add_argument(
        "--test-frac",
        type=float,
        default=0.15,
        help="Test fraction. Default: 0.15",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.total_images <= 0:
        raise SystemExit("total_images must be positive.")
    if args.size < 32:
        raise SystemExit("size should be at least 32 pixels.")

    workers = choose_worker_count(args.workers)

    dataset_name = f"BloomData_{args.total_images}"
    outroot = Path(args.outroot).resolve()
    dataset_dir = outroot / dataset_name
    images_dir = dataset_dir / "images"

    dataset_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("CobberEcoBloom dataset generation")
    print("=" * 72)
    print(f"Dataset name:        {dataset_name}")
    print(f"Output directory:    {dataset_dir}")
    print(f"Total images:        {args.total_images}")
    print(f"Image size:          {args.size} x {args.size}")
    print(f"Worker processes:    {workers}")
    print(f"Random seed:         {args.seed}")
    print(f"Split fractions:     train={args.train_frac:.2f}, val={args.val_frac:.2f}, test={args.test_frac:.2f}")
    print("Class plan:")

    class_counts = counts_by_class(args.total_images, CATEGORY_ORDER)
    for label in CATEGORY_ORDER:
        print(f"  {label:14s} {class_counts[label]}")
    print("-" * 72)

    t0 = time.perf_counter()

    task_build_start = time.perf_counter()
    tasks = make_tasks(args.total_images, args.size, args.seed, images_dir)
    task_build_end = time.perf_counter()

    gen_start = time.perf_counter()
    with Pool(processes=workers) as pool:
        records = pool.map(worker_generate, tasks)
    gen_end = time.perf_counter()

    split_start = time.perf_counter()
    records = assign_splits(records, args.train_frac, args.val_frac, args.test_frac, args.seed)
    records.sort(key=lambda r: (r["label"], r["filename"]))
    split_end = time.perf_counter()

    csv_start = time.perf_counter()
    csv_path = dataset_dir / "bloom_supervisor.csv"
    write_supervisor_csv(records, csv_path)

    summary_path = dataset_dir / "dataset_summary.txt"
    summary_text = summarize_records(records)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text + "\n")
    csv_end = time.perf_counter()

    t1 = time.perf_counter()

    total_seconds = t1 - t0
    build_seconds = task_build_end - task_build_start
    gen_seconds = gen_end - gen_start
    split_seconds = split_end - split_start
    csv_seconds = csv_end - csv_start

    images_per_second = args.total_images / gen_seconds if gen_seconds > 0 else float("inf")
    ms_per_image = 1000.0 * gen_seconds / args.total_images if args.total_images > 0 else 0.0

    print("Done.")
    print("-" * 72)
    print(f"Supervisor CSV:      {csv_path}")
    print(f"Summary file:        {summary_path}")
    print("")
    print("Timing / performance")
    print(f"  Build task list:   {build_seconds:10.3f} s")
    print(f"  Generate images:   {gen_seconds:10.3f} s")
    print(f"  Assign splits:     {split_seconds:10.3f} s")
    print(f"  Write CSV/summary: {csv_seconds:10.3f} s")
    print(f"  Total wall time:   {total_seconds:10.3f} s")
    print(f"  Images / second:   {images_per_second:10.2f}")
    print(f"  ms / image:        {ms_per_image:10.2f}")
    print("")
    print(summary_text)
    print("=" * 72)

    return 0


if __name__ == "__main__":
    # Needed for Windows multiprocessing under PyInstaller / normal execution
    try:
        import multiprocessing as mp
        mp.freeze_support()
    except Exception:
        pass
    raise SystemExit(main())
