#!/usr/bin/env python3
"""
generate_bloom_examples.py

Generate small synthetic "drone-like" lake-surface images for the first
CobberEcoBloom prototype.

This script makes 2 examples of each category by default:

    clear_water
    mostly_clear
    field_check
    mostly_bloom
    dense_bloom

The images are intentionally synthetic. They are not meant to be realistic
remote-sensing products. They are teaching images designed to preserve the
basic visual idea: blue water background plus green algal-bloom coverage.

Dependencies:
    pip install numpy opencv-python

Run:
    python generate_bloom_examples.py

Optional:
    python generate_bloom_examples.py --outdir bloom_examples --examples-per-class 4 --size 256 --seed 123
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np


CATEGORIES: Dict[str, Tuple[float, float]] = {
    "clear_water": (0.00, 0.04),
    "mostly_clear": (0.06, 0.15),
    "field_check": (0.25, 0.75),
    "mostly_bloom": (0.85, 0.94),
    "dense_bloom": (0.96, 1.00),
}


def make_water_background(size: int, rng: np.random.Generator) -> np.ndarray:
    y = np.linspace(0, 1, size)[:, None]
    x = np.linspace(0, 1, size)[None, :]

    base_r = 22 + 8 * y + 4 * np.sin(2 * np.pi * (x * 2.2 + y * 0.4))
    base_g = 92 + 20 * y + 8 * np.sin(2 * np.pi * (x * 1.4 - y * 0.7))
    base_b = 145 + 35 * (1 - y) + 8 * np.sin(2 * np.pi * (x * 1.8 + y * 1.1))

    img = np.dstack([base_r, base_g, base_b]).astype(np.float32)
    img += rng.normal(0, 5, (size, size, 3))

    for _ in range(int(rng.integers(8, 18))):
        yy = int(rng.integers(0, size))
        x0 = int(rng.integers(0, size // 2))
        x1 = int(rng.integers(size // 2, size))
        color = (35, 120, 170)
        thickness = int(rng.integers(1, 3))
        cv2.line(img, (x0, yy), (x1, yy + int(rng.integers(-8, 8))), color, thickness)

    return np.clip(img, 0, 255).astype(np.uint8)


def generate_blob_mask(size: int, target_coverage: float, rng: np.random.Generator) -> np.ndarray:
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate CobberEcoBloom synthetic example images.")
    parser.add_argument("--outdir", default="bloom_examples", help="Output directory.")
    parser.add_argument("--examples-per-class", type=int, default=2, help="Number of images per category.")
    parser.add_argument("--size", type=int, default=256, help="Image width/height in pixels.")
    parser.add_argument("--seed", type=int, default=17, help="Random seed.")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    manifest_rows = []

    print(f"Writing examples to: {outdir.resolve()}")
    print(f"Image size: {args.size} x {args.size}")
    print(f"Examples per category: {args.examples_per_class}\n")

    for category, (lo, hi) in CATEGORIES.items():
        category_dir = outdir / category
        category_dir.mkdir(parents=True, exist_ok=True)

        for i in range(1, args.examples_per_class + 1):
            requested_coverage = rng.uniform(lo, hi)
            image_seed = rng.randint(0, 2_000_000_000)

            img_rgb, actual_coverage = make_bloom_image(args.size, requested_coverage, image_seed)

            filename = f"{category}_{i:02d}.png"
            outpath = category_dir / filename

            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(outpath), img_bgr)

            manifest_rows.append(
                {
                    "filename": str(outpath),
                    "category": category,
                    "requested_coverage": f"{requested_coverage:.4f}",
                    "actual_mask_coverage": f"{actual_coverage:.4f}",
                    "seed": image_seed,
                }
            )

            print(
                f"{category:14s} example {i}: "
                f"requested={requested_coverage:.3f}, actual={actual_coverage:.3f}, "
                f"file={outpath}"
            )

    manifest_path = outdir / "manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["filename", "category", "requested_coverage", "actual_mask_coverage", "seed"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\nDone. Manifest written to: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
