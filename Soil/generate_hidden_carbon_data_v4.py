#!/usr/bin/env python3
"""
generate_hidden_carbon_data_v4.py

Generate a CobberLand-style synthetic ecology dataset for Chapter 16:

    Hidden Carbon — estimating belowground carbon from easy field measurements.

v4 design goal:
    Make the training data visually and pedagogically closer to the original
    CobberLand pattern.

Compared with v3:
    - The wetland/saturated branch has an inverted/stronger curvature so the
      low-moisture end of the green branch dives downward.
    - The prairie/mesic branch has a stronger linear component while retaining
      curvature.
    - The high end of the prairie/mesic branch overlaps with or exceeds the
      low end of the wetland/saturated branch.
    - The two branches still represent different ecological regimes.

Ecological story:
    The landscape is a fictional prairie-wetland restoration area.

    Easy field measurements:
        soil moisture
        surface elevation
        distance to water
        vegetation height
        canopy cover
        greenness index
        plant wetness score

    Hard measurement:
        belowground carbon stock, Mg C/ha

Scientific rationale:
    Belowground carbon can plausibly follow different response functions in
    different hydrologic regimes. Dry/mesic prairie-like sites may accumulate
    carbon through dense root systems and productivity. Saturated wetland-like
    sites may accumulate carbon through slowed decomposition, but recently
    saturated or transition-edge wetland sites may have lower carbon than
    productive mesic prairie sites.

Outputs:
    hidden_carbon_training.csv
    hidden_carbon_new_sites.csv
    hidden_carbon_training_plot.png
    hidden_carbon_model_benchmark.csv

Basic use:
    python generate_hidden_carbon_data_v4.py

Optional:
    python generate_hidden_carbon_data_v4.py --n 1000 --seed 42 --no-show
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


FEATURE_COLUMNS = [
    "soil_moisture_frac",
    "surface_elevation_cm",
    "distance_to_water_m",
    "vegetation_height_cm",
    "canopy_cover_pct",
    "greenness_index",
    "plant_wetness_score",
]

TARGET_COLUMN = "belowground_carbon_Mg_ha"


def make_latent_moisture(
    n: int,
    rng: np.random.Generator,
    fixed_for_new_sites: bool = False,
) -> np.ndarray:
    """
    Generate latent moisture with a deliberate training gap.
    """
    if fixed_for_new_sites and n == 10:
        return np.array([0.10, 0.18, 0.31, 0.41, 0.47, 0.53, 0.60, 0.72, 0.86, 0.95])

    n_dry = n // 2
    n_wet = n - n_dry

    dry = rng.uniform(0.08, 0.42, n_dry)
    wet = rng.uniform(0.58, 0.96, n_wet)

    moisture = np.concatenate([dry, wet])
    rng.shuffle(moisture)
    return moisture


def regime_from_moisture(m: np.ndarray) -> np.ndarray:
    regime = np.empty(len(m), dtype=object)
    regime[m < 0.43] = "prairie_mesic"
    regime[(m >= 0.43) & (m <= 0.57)] = "transition_gap"
    regime[m > 0.57] = "wetland_saturated"
    return regime


def make_easy_features(m: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    """
    Generate easy field measurements.

    These variables are helpful but imperfect proxies. They should not make the
    carbon target trivial.
    """
    n = len(m)
    is_wetland = m > 0.50

    x_dry = np.clip((m - 0.08) / (0.42 - 0.08), 0, 1)
    x_wet = np.clip((m - 0.58) / (0.96 - 0.58), 0, 1)

    soil_moisture_frac = np.clip(m + rng.normal(0, 0.020, n), 0.02, 0.99)

    surface_elevation_cm = (
        122
        - 78 * m
        + 8 * np.sin(5 * np.pi * m)
        + rng.normal(0, 11, n)
    )

    distance_to_water_m = (
        105 * (1 - m)
        + 10 * np.sin(4 * np.pi * m + 0.4)
        + rng.normal(0, 15, n)
    )
    distance_to_water_m = np.clip(distance_to_water_m, 0, None)

    vegetation_height_cm = np.where(
        is_wetland,
        38 + 70 * x_wet + 18 * x_wet**2 + rng.normal(0, 13, n),
        14 + 65 * x_dry + 22 * x_dry**2 + rng.normal(0, 13, n),
    )
    vegetation_height_cm = np.clip(vegetation_height_cm, 4, 160)

    canopy_cover_pct = np.where(
        is_wetland,
        42 + 50 * x_wet + rng.normal(0, 11, n),
        18 + 42 * x_dry + 16 * np.sin(np.pi * x_dry) + rng.normal(0, 11, n),
    )
    canopy_cover_pct = np.clip(canopy_cover_pct, 0, 100)

    greenness_index = np.where(
        is_wetland,
        0.40 + 0.50 * x_wet + rng.normal(0, 0.060, n),
        0.16 + 0.48 * x_dry + 0.08 * np.sin(2 * np.pi * x_dry) + rng.normal(0, 0.060, n),
    )
    greenness_index = np.clip(greenness_index, 0.05, 0.95)

    plant_wetness_score = np.where(
        is_wetland,
        3.0 + 1.7 * x_wet + rng.normal(0, 0.48, n),
        1.0 + 2.0 * x_dry + rng.normal(0, 0.48, n),
    )
    plant_wetness_score = np.round(np.clip(plant_wetness_score, 1, 5), 1)

    return pd.DataFrame({
        "soil_moisture_frac": np.round(soil_moisture_frac, 4),
        "surface_elevation_cm": np.round(surface_elevation_cm, 2),
        "distance_to_water_m": np.round(distance_to_water_m, 2),
        "vegetation_height_cm": np.round(vegetation_height_cm, 2),
        "canopy_cover_pct": np.round(canopy_cover_pct, 2),
        "greenness_index": np.round(greenness_index, 4),
        "plant_wetness_score": np.round(plant_wetness_score, 1),
    })


def make_belowground_carbon(
    m: np.ndarray,
    features: pd.DataFrame,
    rng: np.random.Generator,
    noise_scale: float = 3.8,
) -> np.ndarray:
    """
    Generate belowground carbon with two response branches.

    The branch shapes are intentionally pedagogical.

    Orange / prairie_mesic branch:
        Stronger linear component than v3, but still curved.
        High-end orange values should approach or exceed low-end green values.

    Green / wetland_saturated branch:
        The low side dives downward, then bends upward strongly. This produces a
        curve that is hard for one global linear regression plane to represent.
    """
    is_wetland = m > 0.50

    x_dry = np.clip((m - 0.08) / (0.42 - 0.08), 0, 1)
    x_wet = np.clip((m - 0.58) / (0.96 - 0.58), 0, 1)

    cover = features["canopy_cover_pct"].to_numpy()
    height = features["vegetation_height_cm"].to_numpy()
    green = features["greenness_index"].to_numpy()

    # Prairie/mesic branch:
    # Stronger linear rise, with curvature and a mid-branch root/productivity bump.
    # Approximate range: low ~6-15, high ~60-75, depending on noise and features.
    prairie_carbon = (
        5
        + 43 * x_dry
        + 12 * x_dry**2
        + 8 * np.sin(np.pi * x_dry)
        + 0.055 * height
        + 0.045 * cover
        + rng.normal(0, noise_scale, len(m))
    )

    # Wetland/saturated branch:
    # Low end is intentionally pulled down. The - term near x=0 creates the
    # "dive downward" behavior; the quadratic/cubic terms lift the high end.
    # Approximate range: low ~35-50, high ~170-210.
    wetland_carbon = (
        48
        - 22 * (1 - x_wet) ** 2
        + 28 * x_wet
        + 105 * x_wet**2
        + 24 * x_wet**3
        + 7 * np.sin(2.4 * np.pi * x_wet)
        + 5.0 * green
        + rng.normal(0, 1.7 * noise_scale, len(m))
    )

    carbon = np.where(is_wetland, wetland_carbon, prairie_carbon)
    return np.round(np.clip(carbon, 1, None), 2)


def make_sites(
    n: int,
    rng: np.random.Generator,
    fixed_for_new_sites: bool = False,
    start_label: str = "S",
    noise_scale: float = 3.8,
) -> pd.DataFrame:
    m = make_latent_moisture(n, rng, fixed_for_new_sites=fixed_for_new_sites)
    features = make_easy_features(m, rng)
    carbon = make_belowground_carbon(m, features, rng, noise_scale=noise_scale)

    df = features.copy()
    df.insert(0, "site_id", [f"{start_label}{i + 1:04d}" for i in range(n)])
    df.insert(1, "regime", regime_from_moisture(m))
    df[TARGET_COLUMN] = carbon

    return df


def plot_training_data(df: pd.DataFrame, outpath: Path, show: bool = True) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    color_map = {
        "prairie_mesic": "tab:orange",
        "wetland_saturated": "tab:green",
        "transition_gap": "tab:gray",
    }

    for regime, sub in df.groupby("regime"):
        axes[0, 0].scatter(
            sub["soil_moisture_frac"],
            sub[TARGET_COLUMN],
            s=14,
            alpha=0.62,
            label=regime,
            color=color_map.get(regime, None),
        )
    axes[0, 0].axvspan(0.43, 0.57, color="gray", alpha=0.12, label="training gap")
    axes[0, 0].set_xlabel("Observed soil moisture fraction")
    axes[0, 0].set_ylabel("Belowground carbon (Mg C/ha)")
    axes[0, 0].set_title("v4: stronger orange rise, low green dive, regime gap")
    axes[0, 0].grid(True, alpha=0.25)
    axes[0, 0].legend(fontsize=8)

    for regime, sub in df.groupby("regime"):
        axes[0, 1].scatter(
            sub["greenness_index"],
            sub[TARGET_COLUMN],
            s=14,
            alpha=0.62,
            label=regime,
            color=color_map.get(regime, None),
        )
    axes[0, 1].set_xlabel("Greenness index")
    axes[0, 1].set_ylabel("Belowground carbon (Mg C/ha)")
    axes[0, 1].set_title("Proxy variable does not fully resolve the branches")
    axes[0, 1].grid(True, alpha=0.25)

    for regime, sub in df.groupby("regime"):
        axes[1, 0].scatter(
            sub["vegetation_height_cm"],
            sub[TARGET_COLUMN],
            s=14,
            alpha=0.62,
            label=regime,
            color=color_map.get(regime, None),
        )
    axes[1, 0].set_xlabel("Vegetation height (cm)")
    axes[1, 0].set_ylabel("Belowground carbon (Mg C/ha)")
    axes[1, 0].set_title("Carbon vs. vegetation height")
    axes[1, 0].grid(True, alpha=0.25)

    for regime, sub in df.groupby("regime"):
        axes[1, 1].hist(
            sub[TARGET_COLUMN],
            bins=28,
            alpha=0.55,
            label=regime,
            color=color_map.get(regime, None),
        )
    axes[1, 1].set_xlabel("Belowground carbon (Mg C/ha)")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].set_title("High orange overlaps low green")
    axes[1, 1].grid(True, alpha=0.25)
    axes[1, 1].legend(fontsize=8)

    fig.suptitle("Hidden Carbon v4 synthetic dataset quality-control plot", fontsize=15)
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)

    if show:
        plt.show()
    else:
        plt.close(fig)


def benchmark_models(df: pd.DataFrame, outpath: Path) -> pd.DataFrame:
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LinearRegression
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.svm import SVR
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    except Exception:
        return pd.DataFrame()

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        train_size=0.8,
        random_state=42,
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42, max_depth=8),
        "Random Forest": RandomForestRegressor(random_state=42, n_estimators=160),
        "Support Vector Machine": make_pipeline(StandardScaler(), SVR(C=20.0, gamma="scale", epsilon=4.0)),
        "k-Nearest Neighbors": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=7)),
    }

    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        rows.append({
            "model": name,
            "MAE_Mg_C_ha": mean_absolute_error(y_test, pred),
            "MSE": mean_squared_error(y_test, pred),
            "R2": r2_score(y_test, pred),
        })

    bench = pd.DataFrame(rows).sort_values("MAE_Mg_C_ha")
    bench.to_csv(outpath, index=False)
    return bench


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate v4 synthetic hidden-carbon ecology data."
    )
    parser.add_argument("--n", type=int, default=1000, help="Number of training points. Default: 1000")
    parser.add_argument("--test-n", type=int, default=10, help="Number of new-site points. Default: 10")
    parser.add_argument("--seed", type=int, default=42, help="Random seed. Default: 42")
    parser.add_argument("--noise", type=float, default=3.8, help="Noise scale. Default: 3.8")
    parser.add_argument("--outdir", default=".", help="Output directory. Default: current directory")
    parser.add_argument("--no-show", action="store_true", help="Save the QC plot but do not display it.")
    parser.add_argument("--no-benchmark", action="store_true", help="Skip quick model benchmark CSV.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.n < 100:
        raise SystemExit("Please use at least 100 training points.")
    if args.test_n <= 0:
        raise SystemExit("--test-n must be positive.")

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    train_df = make_sites(
        args.n,
        rng,
        fixed_for_new_sites=False,
        start_label="S",
        noise_scale=args.noise,
    )

    test_df = make_sites(
        args.test_n,
        rng,
        fixed_for_new_sites=(args.test_n == 10),
        start_label="NEW",
        noise_scale=args.noise,
    )
    test_df["site_id"] = [f"NEW{i + 1:02d}" for i in range(len(test_df))]

    train_path = outdir / "hidden_carbon_training.csv"
    test_path = outdir / "hidden_carbon_new_sites.csv"
    plot_path = outdir / "hidden_carbon_training_plot.png"
    bench_path = outdir / "hidden_carbon_model_benchmark.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    plot_training_data(train_df, plot_path, show=not args.no_show)

    bench = pd.DataFrame()
    if not args.no_benchmark:
        bench = benchmark_models(train_df, bench_path)

    print("=" * 76)
    print("Hidden Carbon v4 synthetic dataset generated")
    print("=" * 76)
    print(f"Training rows:       {len(train_df)}")
    print(f"New-site rows:       {len(test_df)}")
    print(f"Random seed:         {args.seed}")
    print(f"Noise scale:         {args.noise}")
    print(f"Training CSV:        {train_path}")
    print(f"New-sites CSV:       {test_path}")
    print(f"QC plot:             {plot_path}")
    if not args.no_benchmark:
        print(f"Benchmark CSV:       {bench_path}")
    print("")
    print("Training set summary:")
    print(train_df[[
        "soil_moisture_frac",
        "vegetation_height_cm",
        "canopy_cover_pct",
        "greenness_index",
        TARGET_COLUMN,
    ]].describe().round(3))
    print("")
    print("Regime counts:")
    print(train_df["regime"].value_counts())
    print("")
    print("New-site challenge points:")
    print(test_df[["site_id", "regime", "soil_moisture_frac", TARGET_COLUMN]].to_string(index=False))
    if not bench.empty:
        print("")
        print("Quick model benchmark on 80/20 train/test split:")
        print(bench.round(4).to_string(index=False))
        print("")
        print("Pedagogical check:")
        print("  • Linear regression should be visibly biased.")
        print("  • High orange values should overlap or exceed low green values.")
        print("  • Flexible models may still perform similarly on random test splits.")
    print("=" * 76)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
