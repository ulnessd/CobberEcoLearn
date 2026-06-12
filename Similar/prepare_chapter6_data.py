#!/usr/bin/env python3
"""
prepare_chapter6_data.py

Prepare app-ready datasets for Chapter 6 of the Ecology machine learning book.

This script reads real ecological datasets:

    1. Palmer Penguins
       Expected files:
           penguins.csv
           penguins_raw.csv   optional

    2. Portal Project Teaching Database
       Expected files:
           surveys.csv
           species.csv
           plots.csv

It writes cleaned, classroom-ready CSV files to:

    EcoData/Ready/

Main outputs:

    penguin_similarity_ready.csv
        Individual penguins with clean physical measurements.

    portal_species_similarity_ready.csv
        Species-level summary features derived from Portal survey records.

    portal_plot_community_ready.csv
        Plot-level community matrix using common species counts.

    portal_plot_year_community_ready.csv
        Plot-year community matrix for possible later extensions.

    portal_species_key_ready.csv
        Key linking species_id codes to names and taxa.

    portal_plot_metadata_ready.csv
        Plot metadata plus simple community summaries.

    chapter6_data_summary.txt
        Human-readable summary of what was created.

    license_and_citation_notes.txt
        Short attribution and license notes.

Important note:
    Portal species_id includes values like "NA". Pandas normally treats "NA"
    as missing data. This script deliberately disables default NA parsing for
    Portal files so that species codes are preserved correctly.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------


def clean_column_name(name: str) -> str:
    """Convert a column name to a simple snake_case form."""
    name = name.strip()
    name = name.replace("(", "").replace(")", "")
    name = name.replace("/", "_")
    name = re.sub(r"[^A-Za-z0-9]+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_").lower()


def safe_name(text: str) -> str:
    """Make a string safe for use inside a column name."""
    text = str(text).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def find_file(base_dir: Path, filename: str, preferred_paths: Iterable[Path]) -> Path:
    """
    Find a file using preferred paths first, then recursive search.

    Parameters
    ----------
    base_dir:
        Project root.
    filename:
        File name to find.
    preferred_paths:
        Candidate paths relative to base_dir.
    """
    for rel_path in preferred_paths:
        candidate = base_dir / rel_path
        if candidate.exists():
            return candidate

    matches = [
        p for p in base_dir.rglob(filename)
        if "Ready" not in p.parts and p.is_file()
    ]

    if matches:
        return matches[0]

    preferred_text = "\n".join(f"    {base_dir / p}" for p in preferred_paths)
    raise FileNotFoundError(
        f"Could not find {filename}.\n"
        f"Looked first in:\n{preferred_text}\n"
        f"Then searched recursively under:\n    {base_dir}"
    )


def shannon_diversity(counts: Iterable[float]) -> float:
    """
    Compute Shannon diversity H = -sum p_i ln(p_i) for nonzero counts.
    """
    arr = np.asarray(list(counts), dtype=float)
    arr = arr[np.isfinite(arr)]
    arr = arr[arr > 0]

    total = arr.sum()
    if total <= 0:
        return 0.0

    p = arr / total
    return float(-(p * np.log(p)).sum())


def mode_or_missing(series: pd.Series):
    """Return the first mode of a series, or np.nan if no mode exists."""
    s = series.dropna()
    if s.empty:
        return np.nan
    modes = s.mode()
    if modes.empty:
        return np.nan
    return modes.iloc[0]


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------
# Palmer Penguins preparation
# ---------------------------------------------------------------------


def read_penguins(penguins_path: Path) -> pd.DataFrame:
    """
    Read Palmer penguin data.

    Handles the clean palmerpenguins CSV and, with limited support,
    the raw CSV format.
    """
    df = pd.read_csv(penguins_path)
    original_columns = list(df.columns)
    df.columns = [clean_column_name(c) for c in df.columns]

    # Clean palmerpenguins format:
    # species,island,bill_length_mm,bill_depth_mm,flipper_length_mm,body_mass_g,sex,year
    if "bill_length_mm" in df.columns:
        out = df.copy()

    # Raw Palmer format:
    # Culmen Length (mm), Culmen Depth (mm), Flipper Length (mm), Body Mass (g), etc.
    elif "culmen_length_mm" in df.columns:
        rename_map = {
            "culmen_length_mm": "bill_length_mm",
            "culmen_depth_mm": "bill_depth_mm",
            "flipper_length_mm": "flipper_length_mm",
            "body_mass_g": "body_mass_g",
        }
        out = df.rename(columns=rename_map).copy()

        if "species" in out.columns:
            # Raw species names are long. Keep the first word as a simple label.
            out["species"] = out["species"].astype(str).str.split().str[0]

    else:
        raise ValueError(
            "Penguin file does not look like either the clean or raw Palmer Penguins CSV.\n"
            f"Columns found were:\n{original_columns}"
        )

    required = [
        "species",
        "island",
        "bill_length_mm",
        "bill_depth_mm",
        "flipper_length_mm",
        "body_mass_g",
    ]

    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Penguin file is missing required columns: {missing}")

    keep_columns = [
        c for c in [
            "species",
            "island",
            "bill_length_mm",
            "bill_depth_mm",
            "flipper_length_mm",
            "body_mass_g",
            "sex",
            "year",
        ]
        if c in out.columns
    ]

    out = out[keep_columns].copy()

    numeric_columns = [
        "bill_length_mm",
        "bill_depth_mm",
        "flipper_length_mm",
        "body_mass_g",
    ]

    for col in numeric_columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=numeric_columns + ["species", "island"]).copy()

    if "sex" not in out.columns:
        out["sex"] = "unknown"
    else:
        out["sex"] = (
            out["sex"]
            .astype("string")
            .str.strip()
            .str.lower()
            .replace({"nan": pd.NA, "": pd.NA, ".": pd.NA})
            .fillna("unknown")
        )

    if "year" not in out.columns:
        out["year"] = pd.NA
    else:
        out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")

    out["species"] = out["species"].astype(str).str.strip()
    out["island"] = out["island"].astype(str).str.strip()

    out = out.reset_index(drop=True)
    out.insert(0, "penguin_id", [f"P{i:03d}" for i in range(1, len(out) + 1)])

    # Order columns explicitly.
    out = out[
        [
            "penguin_id",
            "species",
            "island",
            "sex",
            "year",
            "bill_length_mm",
            "bill_depth_mm",
            "flipper_length_mm",
            "body_mass_g",
        ]
    ]

    return out


# ---------------------------------------------------------------------
# Portal preparation
# ---------------------------------------------------------------------


def read_portal_csv(path: Path) -> pd.DataFrame:
    """
    Read a Portal CSV while preserving species codes such as 'NA'.

    By default pandas treats 'NA' as missing. That would damage the
    Portal species_id column, so default NA parsing is disabled.
    """
    return pd.read_csv(
        path,
        keep_default_na=False,
        na_values=["", " ", "NaN", "nan", "NULL", "null"],
    )


def read_portal_tables(
    surveys_path: Path,
    species_path: Path,
    plots_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    surveys = read_portal_csv(surveys_path)
    species = read_portal_csv(species_path)
    plots = read_portal_csv(plots_path)

    surveys.columns = [clean_column_name(c) for c in surveys.columns]
    species.columns = [clean_column_name(c) for c in species.columns]
    plots.columns = [clean_column_name(c) for c in plots.columns]

    required_surveys = [
        "record_id",
        "month",
        "day",
        "year",
        "plot_id",
        "species_id",
        "sex",
        "hindfoot_length",
        "weight",
    ]
    required_species = ["species_id", "genus", "species", "taxa"]
    required_plots = ["plot_id", "plot_type"]

    for table_name, df, required in [
        ("surveys.csv", surveys, required_surveys),
        ("species.csv", species, required_species),
        ("plots.csv", plots, required_plots),
    ]:
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"{table_name} is missing required columns: {missing}")

    # Basic type cleanup.
    for col in ["record_id", "month", "day", "year", "plot_id", "hindfoot_length", "weight"]:
        surveys[col] = pd.to_numeric(surveys[col], errors="coerce")

    surveys["plot_id"] = surveys["plot_id"].astype("Int64")
    plots["plot_id"] = pd.to_numeric(plots["plot_id"], errors="coerce").astype("Int64")

    surveys["species_id"] = surveys["species_id"].astype("string").str.strip()
    species["species_id"] = species["species_id"].astype("string").str.strip()

    surveys["sex"] = (
        surveys["sex"]
        .astype("string")
        .str.strip()
        .str.upper()
        .replace({"": pd.NA})
    )

    for col in ["genus", "species", "taxa"]:
        species[col] = species[col].astype("string").str.strip()

    plots["plot_type"] = plots["plot_type"].astype("string").str.strip()

    return surveys, species, plots


def merge_portal_tables(
    surveys: pd.DataFrame,
    species: pd.DataFrame,
    plots: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge Portal surveys with species and plot metadata.
    """
    data = surveys.copy()

    # Drop records without a usable species_id.
    data = data.dropna(subset=["species_id"])
    data = data[data["species_id"].astype(str).str.len() > 0].copy()

    data = data.merge(species, on="species_id", how="left")
    data = data.merge(plots, on="plot_id", how="left")

    data["species_name"] = (
        data["genus"].fillna("").astype(str).str.strip()
        + " "
        + data["species"].fillna("").astype(str).str.strip()
    ).str.strip()

    data.loc[data["species_name"] == "", "species_name"] = data.loc[
        data["species_name"] == "", "species_id"
    ]

    return data


def build_portal_species_key(data: pd.DataFrame) -> pd.DataFrame:
    key = (
        data[["species_id", "genus", "species", "species_name", "taxa"]]
        .drop_duplicates()
        .sort_values(["taxa", "species_name", "species_id"], na_position="last")
        .reset_index(drop=True)
    )
    return key


def build_portal_species_similarity(
    data: pd.DataFrame,
    min_observations: int,
    min_measurement_records: int,
) -> pd.DataFrame:
    """
    Build species-level feature table for similarity comparisons.

    The output intentionally uses real derived features but filters to
    species with enough records to be classroom-friendly.
    """
    rows = []

    for species_id, g in data.groupby("species_id", dropna=True):
        g = g.copy()

        n_obs = len(g)
        weights = pd.to_numeric(g["weight"], errors="coerce")
        hindfeet = pd.to_numeric(g["hindfoot_length"], errors="coerce")

        sex = g["sex"].astype("string")
        n_female = int((sex == "F").sum())
        n_male = int((sex == "M").sum())
        n_known_sex = n_female + n_male

        if n_known_sex > 0:
            frac_female = n_female / n_known_sex
        else:
            frac_female = np.nan

        years = pd.to_numeric(g["year"], errors="coerce").dropna()
        months = pd.to_numeric(g["month"], errors="coerce").dropna()

        row = {
            "species_id": species_id,
            "species_name": mode_or_missing(g["species_name"]),
            "genus": mode_or_missing(g["genus"]),
            "species": mode_or_missing(g["species"]),
            "taxa": mode_or_missing(g["taxa"]),
            "observations": int(n_obs),
            "weight_records": int(weights.notna().sum()),
            "hindfoot_records": int(hindfeet.notna().sum()),
            "mean_weight_g": float(weights.mean()) if weights.notna().any() else np.nan,
            "median_weight_g": float(weights.median()) if weights.notna().any() else np.nan,
            "mean_hindfoot_length_mm": float(hindfeet.mean()) if hindfeet.notna().any() else np.nan,
            "median_hindfoot_length_mm": float(hindfeet.median()) if hindfeet.notna().any() else np.nan,
            "female_fraction": float(frac_female) if np.isfinite(frac_female) else np.nan,
            "n_plots_observed": int(g["plot_id"].nunique(dropna=True)),
            "n_plot_types_observed": int(g["plot_type"].nunique(dropna=True)),
            "n_years_observed": int(years.nunique()) if not years.empty else 0,
            "first_year_observed": int(years.min()) if not years.empty else pd.NA,
            "last_year_observed": int(years.max()) if not years.empty else pd.NA,
            "seasonal_spread_months": int(months.nunique()) if not months.empty else 0,
            "peak_month": int(mode_or_missing(months)) if not months.empty else pd.NA,
        }

        rows.append(row)

    out = pd.DataFrame(rows)

    if out.empty:
        return out

    # Add plot-type fractions.
    plot_type_counts = pd.crosstab(data["species_id"], data["plot_type"])
    plot_type_fracs = plot_type_counts.div(plot_type_counts.sum(axis=1), axis=0)

    plot_type_fracs = plot_type_fracs.rename(
        columns={c: f"frac_plot_type_{safe_name(c)}" for c in plot_type_fracs.columns}
    )

    out = out.merge(
        plot_type_fracs.reset_index(),
        on="species_id",
        how="left",
    )

    # Add taxa one-hot flags. These can be useful for the app, but the app
    # can also ignore them if we want students to focus only on numeric features.
    taxa_dummies = pd.get_dummies(out["taxa"], prefix="taxa")
    taxa_dummies.columns = [safe_name(c) for c in taxa_dummies.columns]
    out = pd.concat([out, taxa_dummies], axis=1)

    # Classroom-friendly filter:
    #   enough observations overall,
    #   enough physical measurement records.
    out = out[
        (out["observations"] >= min_observations)
        & (out["weight_records"] >= min_measurement_records)
        & (out["hindfoot_records"] >= min_measurement_records)
    ].copy()

    # Fill plot-type fractions with zero when a species was never found in that plot type.
    frac_cols = [c for c in out.columns if c.startswith("frac_plot_type_")]
    out[frac_cols] = out[frac_cols].fillna(0.0)

    # Leave biological measurement missing values as missing. With the filter above,
    # the important ones should usually be complete or nearly complete.
    out = out.sort_values(["taxa", "species_name", "species_id"]).reset_index(drop=True)

    return out


def build_portal_community_tables(
    data: pd.DataFrame,
    plots: pd.DataFrame,
    top_n_species: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build plot-level and plot-year-level community tables.

    Counts are based on survey records. For the app, these count columns can
    be used directly, converted to presence/absence, or converted to relative
    abundance.
    """
    # Keep records with plot and species.
    d = data.dropna(subset=["plot_id", "species_id"]).copy()
    d["plot_id"] = d["plot_id"].astype(int)

    species_counts = d["species_id"].value_counts()
    top_species_ids = list(species_counts.head(top_n_species).index)

    # Plot-level full matrix, then select top species for readability.
    plot_counts_all = pd.crosstab(d["plot_id"], d["species_id"])

    plot_counts_top = plot_counts_all.reindex(columns=top_species_ids, fill_value=0)
    plot_counts_top = plot_counts_top.rename(
        columns={sid: f"count_{safe_name(sid)}" for sid in plot_counts_top.columns}
    )

    plot_summary = pd.DataFrame(index=plot_counts_all.index)
    plot_summary["total_captures"] = plot_counts_all.sum(axis=1)
    plot_summary["species_richness"] = (plot_counts_all > 0).sum(axis=1)
    plot_summary["shannon_diversity"] = plot_counts_all.apply(shannon_diversity, axis=1)

    plot_table = pd.concat([plot_summary, plot_counts_top], axis=1).reset_index()
    plot_table = plot_table.rename(columns={"index": "plot_id"})

    plot_table["plot_id"] = pd.to_numeric(plot_table["plot_id"], errors="coerce").astype("Int64")
    plot_metadata = plots.copy()
    plot_metadata["plot_id"] = pd.to_numeric(plot_metadata["plot_id"], errors="coerce").astype("Int64")

    plot_table = plot_metadata.merge(plot_table, on="plot_id", how="left")
    count_cols = [c for c in plot_table.columns if c.startswith("count_")]
    summary_cols = ["total_captures", "species_richness", "shannon_diversity"]

    plot_table[count_cols] = plot_table[count_cols].fillna(0).astype(int)
    plot_table["total_captures"] = plot_table["total_captures"].fillna(0).astype(int)
    plot_table["species_richness"] = plot_table["species_richness"].fillna(0).astype(int)
    plot_table["shannon_diversity"] = plot_table["shannon_diversity"].fillna(0.0)

    # Plot-year table.
    plot_year_counts_all = pd.crosstab(
        [d["plot_id"], d["year"]],
        d["species_id"],
    )

    plot_year_counts_top = plot_year_counts_all.reindex(columns=top_species_ids, fill_value=0)
    plot_year_counts_top = plot_year_counts_top.rename(
        columns={sid: f"count_{safe_name(sid)}" for sid in plot_year_counts_top.columns}
    )

    plot_year_summary = pd.DataFrame(index=plot_year_counts_all.index)
    plot_year_summary["total_captures"] = plot_year_counts_all.sum(axis=1)
    plot_year_summary["species_richness"] = (plot_year_counts_all > 0).sum(axis=1)
    plot_year_summary["shannon_diversity"] = plot_year_counts_all.apply(shannon_diversity, axis=1)

    plot_year_table = pd.concat([plot_year_summary, plot_year_counts_top], axis=1).reset_index()

    plot_year_table["plot_id"] = pd.to_numeric(
        plot_year_table["plot_id"], errors="coerce"
    ).astype("Int64")
    plot_year_table["year"] = pd.to_numeric(
        plot_year_table["year"], errors="coerce"
    ).astype("Int64")

    plot_year_table = plot_metadata.merge(plot_year_table, on="plot_id", how="right")

    plot_year_count_cols = [c for c in plot_year_table.columns if c.startswith("count_")]
    plot_year_table[plot_year_count_cols] = plot_year_table[plot_year_count_cols].fillna(0).astype(int)

    # Species key for the community count columns.
    species_key = (
        d[d["species_id"].isin(top_species_ids)]
        [["species_id", "species_name", "genus", "species", "taxa"]]
        .drop_duplicates()
        .copy()
    )
    species_key["community_count_column"] = species_key["species_id"].apply(
        lambda sid: f"count_{safe_name(sid)}"
    )
    species_key["total_records_in_surveys"] = species_key["species_id"].map(species_counts).astype(int)
    species_key = species_key.sort_values("total_records_in_surveys", ascending=False).reset_index(drop=True)

    plot_metadata_ready = plot_table[
        ["plot_id", "plot_type", "total_captures", "species_richness", "shannon_diversity"]
    ].copy()

    return plot_table, plot_year_table, species_key, plot_metadata_ready


# ---------------------------------------------------------------------
# License notes and summary
# ---------------------------------------------------------------------


def make_license_notes() -> str:
    return """Chapter 6 Dataset License and Citation Notes

Palmer Penguins
---------------
Dataset: Palmer Penguins
Source: palmerpenguins project by Allison Horst, Alison Hill, and Kristen Gorman.
License: CC0.
Suggested attribution:
    Data were obtained from the palmerpenguins dataset, originally collected
    by Dr. Kristen Gorman and the Palmer Station, Antarctica LTER.

Portal Project Teaching Database
--------------------------------
Dataset: Portal Project Teaching Database
Source: Weecology / Data Carpentry teaching dataset derived from the Portal Project.
License: CC0.
Suggested attribution:
    Data were obtained from the Portal Project Teaching Database, a simplified
    teaching version of long-term ecological data from the Portal Project.

Teaching Note
-------------
The files in EcoData/Ready are derived, classroom-ready versions of the original
datasets. They are intended for teaching similarity, feature choice, and ecological
parameter space. They should not be treated as the full research-grade versions of
the original datasets.
"""


def make_summary(
    penguin_path: Path,
    surveys_path: Path,
    species_path: Path,
    plots_path: Path,
    penguins_ready: pd.DataFrame,
    portal_species_ready: pd.DataFrame,
    portal_plot_ready: pd.DataFrame,
    portal_plot_year_ready: pd.DataFrame,
    community_species_key: pd.DataFrame,
    args: argparse.Namespace,
) -> str:
    lines = []

    lines.append("Chapter 6 Data Preparation Summary")
    lines.append("=" * 40)
    lines.append("")
    lines.append("Input files")
    lines.append("-----------")
    lines.append(f"Penguins: {penguin_path}")
    lines.append(f"Portal surveys: {surveys_path}")
    lines.append(f"Portal species: {species_path}")
    lines.append(f"Portal plots: {plots_path}")
    lines.append("")

    lines.append("Settings")
    lines.append("--------")
    lines.append(f"Minimum species observations: {args.min_species_observations}")
    lines.append(f"Minimum measurement records: {args.min_measurement_records}")
    lines.append(f"Top community species: {args.top_community_species}")
    lines.append("")

    lines.append("Outputs")
    lines.append("-------")
    lines.append(f"penguin_similarity_ready.csv: {len(penguins_ready)} rows")
    lines.append(
        "    Species counts: "
        + ", ".join(
            f"{k}={v}" for k, v in penguins_ready["species"].value_counts().items()
        )
    )
    lines.append("")
    lines.append(f"portal_species_similarity_ready.csv: {len(portal_species_ready)} rows")
    if not portal_species_ready.empty:
        lines.append(
            "    Taxa counts: "
            + ", ".join(
                f"{k}={v}" for k, v in portal_species_ready["taxa"].value_counts().items()
            )
        )
    lines.append("")
    lines.append(f"portal_plot_community_ready.csv: {len(portal_plot_ready)} rows")
    lines.append(f"portal_plot_year_community_ready.csv: {len(portal_plot_year_ready)} rows")
    lines.append(f"portal_species_key_ready.csv: {len(community_species_key)} rows")
    lines.append("")

    lines.append("Community species used")
    lines.append("----------------------")
    for _, row in community_species_key.iterrows():
        lines.append(
            f"{row['species_id']}: {row['species_name']} "
            f"({row['taxa']}), column={row['community_count_column']}, "
            f"records={row['total_records_in_surveys']}"
        )

    lines.append("")
    lines.append("Notes")
    lines.append("-----")
    lines.append(
        "Portal species_id values were read with pandas default NA parsing disabled, "
        "so species codes such as 'NA' are preserved."
    )
    lines.append(
        "The ready files are intended for teaching. They are cleaned and simplified "
        "versions of real data, not replacements for the original research datasets."
    )

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Chapter 6 ecology similarity datasets."
    )

    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path.cwd(),
        help="Project root directory. Default: current working directory.",
    )

    parser.add_argument(
        "--min-species-observations",
        type=int,
        default=30,
        help="Minimum Portal observations required for species similarity table.",
    )

    parser.add_argument(
        "--min-measurement-records",
        type=int,
        default=10,
        help="Minimum weight and hindfoot records required for species similarity table.",
    )

    parser.add_argument(
        "--top-community-species",
        type=int,
        default=12,
        help="Number of most common Portal species to include in community matrices.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir.resolve()

    eco_data_dir = base_dir / "EcoData"
    ready_dir = eco_data_dir / "Ready"
    ensure_dir(ready_dir)

    print(f"Project base directory: {base_dir}")
    print(f"Ready output directory: {ready_dir}")

    # Locate input files.
    penguins_path = find_file(
        base_dir,
        "penguins.csv",
        [
            Path("EcoData/Palmer/penguins.csv"),
            Path("EcoData/Penguins/penguins.csv"),
            Path("EcoData/penguins.csv"),
            Path("penguins.csv"),
        ],
    )

    surveys_path = find_file(
        base_dir,
        "surveys.csv",
        [
            Path("EcoData/Portal/surveys.csv"),
            Path("Portal/surveys.csv"),
            Path("surveys.csv"),
        ],
    )

    species_path = find_file(
        base_dir,
        "species.csv",
        [
            Path("EcoData/Portal/species.csv"),
            Path("Portal/species.csv"),
            Path("species.csv"),
        ],
    )

    plots_path = find_file(
        base_dir,
        "plots.csv",
        [
            Path("EcoData/Portal/plots.csv"),
            Path("Portal/plots.csv"),
            Path("plots.csv"),
        ],
    )

    print("\nFound input files:")
    print(f"  Penguins: {penguins_path}")
    print(f"  Portal surveys: {surveys_path}")
    print(f"  Portal species: {species_path}")
    print(f"  Portal plots: {plots_path}")

    # Prepare Penguins.
    print("\nPreparing Palmer Penguins data...")
    penguins_ready = read_penguins(penguins_path)
    penguins_out = ready_dir / "penguin_similarity_ready.csv"
    penguins_ready.to_csv(penguins_out, index=False)
    print(f"  Wrote {penguins_out} ({len(penguins_ready)} rows)")

    # Prepare Portal.
    print("\nPreparing Portal data...")
    surveys, species, plots = read_portal_tables(surveys_path, species_path, plots_path)
    portal_data = merge_portal_tables(surveys, species, plots)

    portal_species_all_key = build_portal_species_key(portal_data)
    portal_species_all_key_out = ready_dir / "portal_all_species_key_ready.csv"
    portal_species_all_key.to_csv(portal_species_all_key_out, index=False)
    print(f"  Wrote {portal_species_all_key_out} ({len(portal_species_all_key)} rows)")

    portal_species_ready = build_portal_species_similarity(
        portal_data,
        min_observations=args.min_species_observations,
        min_measurement_records=args.min_measurement_records,
    )
    portal_species_out = ready_dir / "portal_species_similarity_ready.csv"
    portal_species_ready.to_csv(portal_species_out, index=False)
    print(f"  Wrote {portal_species_out} ({len(portal_species_ready)} rows)")

    (
        portal_plot_ready,
        portal_plot_year_ready,
        community_species_key,
        portal_plot_metadata_ready,
    ) = build_portal_community_tables(
        portal_data,
        plots,
        top_n_species=args.top_community_species,
    )

    portal_plot_out = ready_dir / "portal_plot_community_ready.csv"
    portal_plot_year_out = ready_dir / "portal_plot_year_community_ready.csv"
    community_species_key_out = ready_dir / "portal_species_key_ready.csv"
    portal_plot_metadata_out = ready_dir / "portal_plot_metadata_ready.csv"

    portal_plot_ready.to_csv(portal_plot_out, index=False)
    portal_plot_year_ready.to_csv(portal_plot_year_out, index=False)
    community_species_key.to_csv(community_species_key_out, index=False)
    portal_plot_metadata_ready.to_csv(portal_plot_metadata_out, index=False)

    print(f"  Wrote {portal_plot_out} ({len(portal_plot_ready)} rows)")
    print(f"  Wrote {portal_plot_year_out} ({len(portal_plot_year_ready)} rows)")
    print(f"  Wrote {community_species_key_out} ({len(community_species_key)} rows)")
    print(f"  Wrote {portal_plot_metadata_out} ({len(portal_plot_metadata_ready)} rows)")

    # Notes and summary.
    license_out = ready_dir / "license_and_citation_notes.txt"
    write_text(license_out, make_license_notes())
    print(f"  Wrote {license_out}")

    summary = make_summary(
        penguin_path=penguins_path,
        surveys_path=surveys_path,
        species_path=species_path,
        plots_path=plots_path,
        penguins_ready=penguins_ready,
        portal_species_ready=portal_species_ready,
        portal_plot_ready=portal_plot_ready,
        portal_plot_year_ready=portal_plot_year_ready,
        community_species_key=community_species_key,
        args=args,
    )

    summary_out = ready_dir / "chapter6_data_summary.txt"
    write_text(summary_out, summary)
    print(f"  Wrote {summary_out}")

    print("\nDone. Inspect the ready files with:")
    print(f"  ls -lh {ready_dir}")
    print(f"  head -n 5 {penguins_out}")
    print(f"  head -n 5 {portal_species_out}")
    print(f"  head -n 5 {portal_plot_out}")


if __name__ == "__main__":
    main()