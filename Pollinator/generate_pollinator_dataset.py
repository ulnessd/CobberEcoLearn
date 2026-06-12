#!/usr/bin/env python3
"""
Generate a synthetic pollinator-plant interaction dataset for a K-means chapter.

The design intentionally mirrors the spirit of the hydrogen-bond chapter:
- a simplified, field-inspired dataset
- three student-facing clustering features
- hidden structure that K-means can uncover without labels

Main clustering features:
    1. Dwell_Time_s
    2. Return_Regularity
    3. Contact_Fit

Additional visualization helpers are also generated:
    - Landing_X, Landing_Y
    - Landing_Radius, Landing_Angle_deg
    - Interaction_Mode (hidden truth; not for students)

Usage examples
--------------
python generate_pollinator_dataset.py
python generate_pollinator_dataset.py --n 84000 --output pollinator_interactions_dataset.csv
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PlantType:
    name: str
    reward: float              # 0-1
    hotspot: Tuple[float, float]
    landing_spread: float      # smaller -> more focused landing pattern


@dataclass(frozen=True)
class PollinatorType:
    name: str
    constancy: float           # 0-1 baseline tendency to revisit same flower type
    contact_bias: float        # additive modifier to contact fit
    dwell_bias: float          # additive modifier to dwell time


PLANTS: Dict[str, PlantType] = {
    "Composite Daisy": PlantType("Composite Daisy", reward=0.65, hotspot=(0.20, 0.10), landing_spread=0.23),
    "Deep Tube": PlantType("Deep Tube", reward=0.80, hotspot=(0.05, -0.25), landing_spread=0.18),
    "Bell Flower": PlantType("Bell Flower", reward=0.72, hotspot=(-0.10, -0.20), landing_spread=0.20),
    "Flat Cluster": PlantType("Flat Cluster", reward=0.60, hotspot=(0.00, 0.05), landing_spread=0.28),
    "Pea Flower": PlantType("Pea Flower", reward=0.78, hotspot=(0.22, -0.08), landing_spread=0.19),
    "Star Flower": PlantType("Star Flower", reward=0.58, hotspot=(-0.15, 0.12), landing_spread=0.26),
}


POLLINATORS: Dict[str, PollinatorType] = {
    "Bumblebee": PollinatorType("Bumblebee", constancy=0.78, contact_bias=0.08, dwell_bias=1.0),
    "Honeybee": PollinatorType("Honeybee", constancy=0.68, contact_bias=0.02, dwell_bias=0.4),
    "Solitary Bee": PollinatorType("Solitary Bee", constancy=0.62, contact_bias=0.05, dwell_bias=0.1),
    "Hoverfly": PollinatorType("Hoverfly", constancy=0.42, contact_bias=-0.06, dwell_bias=-0.4),
    "Butterfly": PollinatorType("Butterfly", constancy=0.50, contact_bias=-0.02, dwell_bias=0.6),
    "Beetle": PollinatorType("Beetle", constancy=0.35, contact_bias=-0.10, dwell_bias=0.9),
}


# Soft ecological preference matrix: 0 = weak fit, 1 = strong fit.
# These are intentionally stylized but ecologically plausible functional-group tendencies.
PREFERENCE = pd.DataFrame(
    {
        "Composite Daisy": [0.70, 0.72, 0.78, 0.88, 0.55, 0.65],
        "Deep Tube":       [0.85, 0.55, 0.30, 0.18, 0.82, 0.12],
        "Bell Flower":     [0.80, 0.58, 0.42, 0.28, 0.60, 0.22],
        "Flat Cluster":    [0.66, 0.74, 0.72, 0.90, 0.58, 0.62],
        "Pea Flower":      [0.92, 0.70, 0.36, 0.15, 0.30, 0.16],
        "Star Flower":     [0.62, 0.69, 0.80, 0.76, 0.46, 0.58],
    },
    index=["Bumblebee", "Honeybee", "Solitary Bee", "Hoverfly", "Butterfly", "Beetle"],
)


def clamp(x: np.ndarray | float, low: float, high: float):
    return np.minimum(np.maximum(x, low), high)


def choose_mode(pref: float, rng: np.random.Generator) -> str:
    """Pick a hidden interaction mode from compatibility.

    effective: high-value pollination-like visit
    exploratory: moderate visit, some useful contacts
    incidental: brief/poorly aligned visit
    nectar_steal: longer visit but weak contact with pollen-transfer zone
    """
    if pref >= 0.78:
        probs = [0.58, 0.22, 0.10, 0.10]
    elif pref >= 0.55:
        probs = [0.24, 0.40, 0.22, 0.14]
    elif pref >= 0.35:
        probs = [0.08, 0.34, 0.40, 0.18]
    else:
        probs = [0.02, 0.16, 0.60, 0.22]
    return rng.choice(["effective", "exploratory", "incidental", "nectar_steal"], p=probs)


def sample_features(
    plant: PlantType,
    pollinator: PollinatorType,
    pref: float,
    mode: str,
    rng: np.random.Generator,
) -> dict:
    # Contact fit is the key ecological analogue of directional geometry.
    if mode == "effective":
        contact_mu = 0.68 + 0.28 * pref + pollinator.contact_bias
        return_mu = 0.35 + 0.45 * pollinator.constancy + 0.18 * pref
        dwell_mu = 3.2 + 6.4 * plant.reward + 3.0 * pref + pollinator.dwell_bias
        x_mu, y_mu = plant.hotspot
        spread = plant.landing_spread
    elif mode == "exploratory":
        contact_mu = 0.34 + 0.30 * pref + 0.5 * pollinator.contact_bias
        return_mu = 0.18 + 0.35 * pollinator.constancy + 0.18 * pref
        dwell_mu = 1.8 + 3.6 * plant.reward + 1.7 * pref + 0.5 * pollinator.dwell_bias
        x_mu = plant.hotspot[0] + rng.normal(0.0, 0.14)
        y_mu = plant.hotspot[1] + rng.normal(0.0, 0.14)
        spread = plant.landing_spread * 1.75
    elif mode == "nectar_steal":
        contact_mu = 0.08 + 0.16 * pref + 0.3 * pollinator.contact_bias
        return_mu = 0.10 + 0.20 * pollinator.constancy + 0.12 * pref
        dwell_mu = 2.6 + 4.5 * plant.reward + 1.1 * pref + 0.9 * pollinator.dwell_bias
        # Long visit, but displaced from the contact hotspot.
        x_mu = plant.hotspot[0] + rng.choice([-0.45, 0.45])
        y_mu = plant.hotspot[1] + rng.choice([-0.35, 0.35])
        spread = plant.landing_spread * 1.25
    else:  # incidental
        contact_mu = 0.04 + 0.14 * pref + 0.25 * pollinator.contact_bias
        return_mu = 0.05 + 0.16 * pollinator.constancy + 0.10 * pref
        dwell_mu = 0.6 + 1.8 * plant.reward + 0.8 * pref + 0.2 * pollinator.dwell_bias
        x_mu = rng.uniform(-0.95, 0.95)
        y_mu = rng.uniform(-0.95, 0.95)
        spread = plant.landing_spread * 2.8

    contact_fit = float(clamp(rng.normal(contact_mu, 0.08), 0.0, 1.0))
    return_reg = float(clamp(rng.normal(return_mu, 0.09), 0.0, 1.0))
    dwell = float(max(0.3, rng.normal(dwell_mu, 1.15)))

    landing_x = float(clamp(rng.normal(x_mu, spread), -1.25, 1.25))
    landing_y = float(clamp(rng.normal(y_mu, spread), -1.25, 1.25))
    landing_radius = float(np.sqrt(landing_x**2 + landing_y**2))
    landing_angle = float((np.degrees(np.arctan2(landing_y, landing_x)) + 360.0) % 360.0)

    return {
        "Dwell_Time_s": round(dwell, 3),
        "Return_Regularity": round(return_reg, 3),
        "Contact_Fit": round(contact_fit, 3),
        "Landing_X": round(landing_x, 3),
        "Landing_Y": round(landing_y, 3),
        "Landing_Radius": round(landing_radius, 3),
        "Landing_Angle_deg": round(landing_angle, 3),
    }


def generate_dataset(n_rows: int = 84_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    plant_names = list(PLANTS.keys())
    pollinator_names = list(POLLINATORS.keys())

    rows = []
    for _ in range(n_rows):
        plant_name = rng.choice(plant_names)
        pollinator_name = rng.choice(pollinator_names)

        plant = PLANTS[plant_name]
        pollinator = POLLINATORS[pollinator_name]
        pref = float(PREFERENCE.loc[pollinator_name, plant_name])
        mode = choose_mode(pref, rng)
        feat = sample_features(plant, pollinator, pref, mode, rng)

        rows.append(
            {
                "Plant_Type": plant_name,
                "Pollinator_Type": pollinator_name,
                "Preference_Score": round(pref, 3),
                "Interaction_Mode": mode,
                **feat,
            }
        )

    df = pd.DataFrame(rows)

    # Optional convenience columns if you want to preserve a familiar interface
    # while adapting the existing app.
    df["Interaction_Pair"] = df["Plant_Type"] + " :: " + df["Pollinator_Type"]

    return df


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a synthetic pollinator-plant interaction dataset.")
    parser.add_argument("--n", type=int, default=84_000, help="Number of rows to generate (default: 84000).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument(
        "--output",
        type=str,
        default="pollinator_interactions_dataset.csv",
        help="Output CSV filename (default: pollinator_interactions_dataset.csv).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    df = generate_dataset(n_rows=args.n, seed=args.seed)
    df.to_csv(args.output, index=False)

    print(f"Wrote {len(df):,} rows to {args.output}")
    print("\nPlant types:")
    for p in PLANTS:
        print(f"  - {p}")
    print("\nPollinator types:")
    for p in POLLINATORS:
        print(f"  - {p}")
    print("\nPreview:")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
