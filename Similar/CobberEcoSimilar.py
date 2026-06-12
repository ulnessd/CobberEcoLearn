#!/usr/bin/env python3
"""
CobberEcoSimilar.py

Ecological Similarity Explorer for Chapter 6 of the Ecology ML book.

Version 3 changes
-----------------
* More horizontal desktop layout: controls on the left, plots on the right.
* Removed the sample-size and focal-object controls to simplify the interface.
* Penguin mode keeps the grouped views only (no individual sample mode).
* The ecological-space plot now uses actual ecological variables for the axes.
* Clicking a similarity-matrix cell opens a comparison dialog.
* For penguin comparisons, the dialog shows representative species images.

Expected data layout, relative to this file or the working directory:

    EcoData/
        Ready/
            penguin_similarity_ready.csv
            portal_species_similarity_ready.csv
            portal_plot_community_ready.csv
            portal_plot_year_community_ready.csv
            portal_species_key_ready.csv
            license_and_citation_notes.txt
            chapter6_data_summary.txt

Dependencies:
    pip install pandas numpy bokeh PyQt6 PyQt6-WebEngine
"""

from __future__ import annotations

import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont, QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from PyQt6.QtWebEngineCore import QWebEnginePage
from PyQt6.QtWebEngineWidgets import QWebEngineView

from bokeh.embed import file_html
from bokeh.layouts import column, row
from bokeh.models import (
    BasicTicker,
    ColorBar,
    ColumnDataSource,
    CustomJS,
    Div,
    HoverTool,
    LinearColorMapper,
    TapTool,
)
from bokeh.palettes import Category10, Category20, Viridis256
from bokeh.plotting import figure
from bokeh.resources import CDN

APP_TITLE = "CobberEcoSimilar"

PENGUIN_IMAGE_FILES = {
    "adelie": "Adelie.png",
    "chinstrap": "Chinstrap.png",
    "gentoo": "Gentoo.png",
}

FEATURE_LABELS: Dict[str, str] = {
    "bill_length_mm": "Bill length (mm)",
    "bill_depth_mm": "Bill depth (mm)",
    "flipper_length_mm": "Flipper length (mm)",
    "body_mass_g": "Body mass (g)",
    "observations": "Observation count",
    "log_observations": "Log observation count",
    "weight_records": "Weight records",
    "hindfoot_records": "Hindfoot records",
    "mean_weight_g": "Mean weight (g)",
    "median_weight_g": "Median weight (g)",
    "mean_hindfoot_length_mm": "Mean hindfoot length (mm)",
    "median_hindfoot_length_mm": "Median hindfoot length (mm)",
    "female_fraction": "Female fraction",
    "n_plots_observed": "Number of plots observed",
    "n_plot_types_observed": "Number of plot types observed",
    "n_years_observed": "Number of years observed",
    "seasonal_spread_months": "Seasonal spread (months)",
    "peak_month": "Peak month",
    "frac_plot_type_control": "Fraction in Control plots",
    "frac_plot_type_long_term_krat_exclosure": "Fraction in long-term krat exclosures",
    "frac_plot_type_rodent_exclosure": "Fraction in rodent exclosures",
    "frac_plot_type_short_term_krat_exclosure": "Fraction in short-term krat exclosures",
    "frac_plot_type_spectab_exclosure": "Fraction in Spectab exclosures",
    "total_captures": "Total captures",
    "species_richness": "Species richness",
    "shannon_diversity": "Shannon diversity",
}

COUNT_NAME_HINTS: Dict[str, str] = {
    "count_dm": "Dipodomys merriami",
    "count_pp": "Chaetodipus penicillatus",
    "count_do": "Dipodomys ordii",
    "count_pb": "Chaetodipus baileyi",
    "count_rm": "Reithrodontomys megalotis",
    "count_ds": "Dipodomys spectabilis",
    "count_ot": "Onychomys torridus",
    "count_pf": "Perognathus flavus",
    "count_pe": "Peromyscus eremicus",
    "count_nl": "Neotoma albigula",
    "count_ol": "Onychomys leucogaster",
    "count_pm": "Peromyscus maniculatus",
}

PENGUIN_FEATURE_PRESETS = {
    "All physical traits": [
        "bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"
    ],
    "Bill shape": ["bill_length_mm", "bill_depth_mm"],
    "Body size": ["flipper_length_mm", "body_mass_g"],
    "Mass only": ["body_mass_g"],
}

PORTAL_SPECIES_FEATURE_PRESETS = {
    "Body traits": ["mean_weight_g", "mean_hindfoot_length_mm"],
    "Ecological breadth": [
        "log_observations", "n_plots_observed", "n_plot_types_observed",
        "n_years_observed", "seasonal_spread_months"
    ],
    "Habitat / treatment use": [
        "frac_plot_type_control", "frac_plot_type_long_term_krat_exclosure",
        "frac_plot_type_rodent_exclosure", "frac_plot_type_short_term_krat_exclosure",
        "frac_plot_type_spectab_exclosure"
    ],
    "Full species profile": [
        "mean_weight_g", "mean_hindfoot_length_mm", "female_fraction",
        "log_observations", "n_plots_observed", "n_plot_types_observed",
        "n_years_observed", "seasonal_spread_months",
        "frac_plot_type_control", "frac_plot_type_long_term_krat_exclosure",
        "frac_plot_type_rodent_exclosure", "frac_plot_type_short_term_krat_exclosure",
        "frac_plot_type_spectab_exclosure"
    ],
}

COMMUNITY_FEATURE_PRESETS = {
    "Presence / absence (Jaccard)": [],
    "Abundance (Bray-Curtis)": [],
    "Relative abundance profile": [],
    "Summary ecology": ["total_captures", "species_richness", "shannon_diversity"],
}

DATASET_CHOICES = [
    "Palmer Penguins",
    "Portal Species",
    "Portal Plot Communities",
    "Portal Plot-Year Communities",
]

PENGUIN_OBJECT_LEVELS = [
    "Species averages",
    "Species + island averages",
    "Species + sex averages",
]

PORTAL_OBJECT_LEVELS = {
    "Portal Species": ["All prepared species"],
    "Portal Plot Communities": ["All plots"],
    "Portal Plot-Year Communities": ["Plots in selected year"],
}


@dataclass
class PreparedDataset:
    df: pd.DataFrame
    labels: List[str]
    short_labels: List[str]
    groups: List[str]
    metadata_columns: List[str]
    default_features: List[str]
    all_features: List[str]
    title: str
    object_description: str
    dataset_type: str


@dataclass
class SimilarityResult:
    sim: np.ndarray
    method_name: str
    feature_columns: List[str]
    feature_labels: List[str]
    notes: str
    plot_x: np.ndarray
    plot_y: np.ndarray
    plot_x_label: str
    plot_y_label: str


@dataclass
class RenderContext:
    prepared: PreparedDataset
    result: SimilarityResult


def app_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def candidate_roots() -> List[Path]:
    roots: List[Path] = []
    for p in [Path.cwd(), app_dir(), app_dir().parent, Path.cwd().parent]:
        if p not in roots:
            roots.append(p)
    return roots


def find_ready_dir() -> Optional[Path]:
    for root in candidate_roots():
        for candidate in [
            root / "EcoData" / "Ready",
            root / "Ready",
            root / "DataSets_Models" / "EcoData" / "Ready",
        ]:
            if candidate.exists() and candidate.is_dir():
                return candidate
    return None


def nice_feature_name(col: str) -> str:
    if col in FEATURE_LABELS:
        return FEATURE_LABELS[col]
    if col in COUNT_NAME_HINTS:
        return f"{COUNT_NAME_HINTS[col]} ({col.replace('count_', '').upper()})"
    if col.startswith("count_"):
        return f"Species count: {col.replace('count_', '').upper()}"
    return col.replace("_", " ").title()


def safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def format_value(x) -> str:
    if pd.isna(x):
        return ""
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    try:
        xf = float(x)
        if abs(xf) >= 1000:
            return f"{xf:,.0f}"
        if abs(xf) >= 10:
            return f"{xf:.2f}"
        return f"{xf:.3f}"
    except Exception:
        return str(x)


def make_unique_short_labels(labels: Sequence[str], preferred: Optional[Sequence[str]] = None) -> List[str]:
    preferred = list(preferred) if preferred is not None else list(labels)
    seen: Dict[str, int] = {}
    out: List[str] = []
    for item in preferred:
        base = str(item)
        if len(base) > 22:
            base = base[:19] + "..."
        if base not in seen:
            seen[base] = 1
            out.append(base)
        else:
            seen[base] += 1
            out.append(f"{base}_{seen[base]}")
    return out


def color_cycle(n: int) -> List[str]:
    return list(Category10[10]) if n <= 10 else list(Category20[20])


def group_colors(groups: Sequence[str]) -> List[str]:
    unique = list(dict.fromkeys(str(g) for g in groups))
    palette = color_cycle(max(1, len(unique)))
    mapping = {g: palette[i % len(palette)] for i, g in enumerate(unique)}
    return [mapping[str(g)] for g in groups]


def zscore_array(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    m = np.nanmean(x)
    s = np.nanstd(x)
    if not np.isfinite(s) or s <= 1e-12:
        s = 1.0
    return (x - m) / s


def choose_top_variable_features(df: pd.DataFrame, features: Sequence[str], use_relative: bool = False) -> List[str]:
    cols = [c for c in features if c in df.columns]
    if not cols:
        return []
    x = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if use_relative:
        totals = x.sum(axis=1, keepdims=True)
        totals = np.where(totals > 0, totals, 1.0)
        x = x / totals
    vars_ = np.nanvar(x, axis=0)
    order = np.argsort(vars_)[::-1]
    chosen = [cols[i] for i in order[:2]]
    if len(chosen) == 1:
        chosen.append(chosen[0])
    return chosen


class EcoDataManager:
    def __init__(self, ready_dir: Optional[Path] = None):
        self.ready_dir = ready_dir or find_ready_dir()
        self._cache: Dict[str, pd.DataFrame] = {}

    def set_ready_dir(self, ready_dir: Path) -> None:
        self.ready_dir = ready_dir
        self._cache.clear()

    def require_ready_dir(self) -> Path:
        if self.ready_dir is None:
            raise FileNotFoundError(
                "Could not find EcoData/Ready. Use File > Choose Ready Data Folder, or run this app from the project root."
            )
        return self.ready_dir

    def read_csv(self, filename: str) -> pd.DataFrame:
        if filename in self._cache:
            return self._cache[filename].copy()
        path = self.require_ready_dir() / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing required data file: {path}")
        df = pd.read_csv(path)
        self._cache[filename] = df.copy()
        return df

    def read_text_if_exists(self, filename: str) -> str:
        try:
            path = self.require_ready_dir() / filename
        except FileNotFoundError:
            return ""
        if path.exists():
            return path.read_text(encoding="utf-8", errors="replace")
        return ""

    def year_values(self) -> List[int]:
        df = self.read_csv("portal_plot_year_community_ready.csv")
        years = pd.to_numeric(df["year"], errors="coerce").dropna().astype(int).unique()
        return sorted(years.tolist())

    def community_count_columns(self, df: pd.DataFrame) -> List[str]:
        return [c for c in df.columns if c.startswith("count_")]

    def prepare_penguins(self, object_level: str) -> PreparedDataset:
        raw = self.read_csv("penguin_similarity_ready.csv")
        numeric = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]

        if object_level == "Species averages":
            gcols = ["species"]
            df = raw.groupby(gcols, dropna=False)[numeric].mean().reset_index()
            df["n_penguins"] = raw.groupby(gcols, dropna=False).size().values
            labels = df["species"].astype(str).tolist()
            short = labels
            groups = labels
            meta = ["species", "n_penguins"]
            desc = "Each row is the average physical profile for one penguin species."

        elif object_level == "Species + island averages":
            gcols = ["species", "island"]
            df = raw.groupby(gcols, dropna=False)[numeric].mean().reset_index()
            df["n_penguins"] = raw.groupby(gcols, dropna=False).size().values
            labels = (df["species"].astype(str) + " on " + df["island"].astype(str)).tolist()
            short = (df["species"].str[:3].astype(str) + "-" + df["island"].str[:3].astype(str)).tolist()
            groups = df["species"].astype(str).tolist()
            meta = ["species", "island", "n_penguins"]
            desc = "Each row is a species-by-island average."

        elif object_level == "Species + sex averages":
            temp = raw[raw["sex"].isin(["male", "female"])].copy()
            gcols = ["species", "sex"]
            df = temp.groupby(gcols, dropna=False)[numeric].mean().reset_index()
            df["n_penguins"] = temp.groupby(gcols, dropna=False).size().values
            labels = (df["species"].astype(str) + " " + df["sex"].astype(str)).tolist()
            short = (df["species"].str[:3].astype(str) + "-" + df["sex"].str[:1].str.upper()).tolist()
            groups = df["species"].astype(str).tolist()
            meta = ["species", "sex", "n_penguins"]
            desc = "Each row is a species-by-sex average."
        else:
            raise ValueError(f"Unknown penguin object level: {object_level}")

        all_features = [c for c in numeric if c in df.columns]
        return PreparedDataset(
            df=df.reset_index(drop=True),
            labels=labels,
            short_labels=make_unique_short_labels(labels, short),
            groups=groups,
            metadata_columns=[c for c in meta if c in df.columns],
            default_features=list(PENGUIN_FEATURE_PRESETS["All physical traits"]),
            all_features=all_features,
            title=f"Palmer Penguins: {object_level}",
            object_description=desc,
            dataset_type="penguins",
        )

    def prepare_portal_species(self) -> PreparedDataset:
        df = self.read_csv("portal_species_similarity_ready.csv").copy()
        if "log_observations" not in df.columns and "observations" in df.columns:
            obs = pd.to_numeric(df["observations"], errors="coerce").fillna(0)
            df["log_observations"] = np.log10(obs + 1)

        labels = df["species_name"].astype(str).tolist()
        short = df["species_id"].astype(str).tolist()
        groups = df["genus"].astype(str).tolist()
        meta = ["species_id", "species_name", "genus", "species", "taxa", "observations"]
        all_features = [c for c in df.columns if c not in meta]

        return PreparedDataset(
            df=df.reset_index(drop=True),
            labels=labels,
            short_labels=make_unique_short_labels(labels, short),
            groups=groups,
            metadata_columns=[c for c in meta if c in df.columns],
            default_features=list(PORTAL_SPECIES_FEATURE_PRESETS["Full species profile"]),
            all_features=all_features,
            title="Portal Species",
            object_description="Each row is one Portal species summarized by real ecological and body-trait features.",
            dataset_type="portal_species",
        )

    def prepare_portal_plots(self) -> PreparedDataset:
        df = self.read_csv("portal_plot_community_ready.csv")
        return self._prepare_portal_community_common(df, "Portal Plot Communities", "Each row is one Portal plot summarized as a community.", include_year=False)

    def prepare_portal_plot_years(self, year: int) -> PreparedDataset:
        df = self.read_csv("portal_plot_year_community_ready.csv")
        df = df[pd.to_numeric(df["year"], errors="coerce") == int(year)].copy()
        return self._prepare_portal_community_common(df, f"Portal Plot-Year Communities: {year}", f"Each row is one Portal plot in {year}.", include_year=True)

    def _prepare_portal_community_common(self, df: pd.DataFrame, title: str, object_description: str, include_year: bool) -> PreparedDataset:
        df = df.copy().sort_values("plot_id").reset_index(drop=True)
        count_cols = self.community_count_columns(df)
        labels = ("Plot " + df["plot_id"].astype(str) + " (" + df["plot_type"].astype(str) + ")").tolist()
        if include_year:
            labels = [f"{lab}, {int(yr)}" for lab, yr in zip(labels, pd.to_numeric(df["year"], errors="coerce").fillna(0))]
        short = [f"P{int(pid)}" for pid in pd.to_numeric(df["plot_id"], errors="coerce").fillna(0)]
        if include_year:
            short = [f"{s}-{int(y)}" for s, y in zip(short, pd.to_numeric(df["year"], errors="coerce").fillna(0))]
        groups = df["plot_type"].astype(str).tolist()
        meta = ["plot_id", "plot_type", "year", "total_captures", "species_richness", "shannon_diversity"]
        all_features = [c for c in count_cols + ["total_captures", "species_richness", "shannon_diversity"] if c in df.columns]
        return PreparedDataset(
            df=df.reset_index(drop=True),
            labels=labels,
            short_labels=make_unique_short_labels(labels, short),
            groups=groups,
            metadata_columns=[c for c in meta if c in df.columns],
            default_features=count_cols,
            all_features=all_features,
            title=title,
            object_description=object_description,
            dataset_type="community",
        )

    def build_prepared(self, dataset_choice: str, object_level: str, year: Optional[int]) -> PreparedDataset:
        if dataset_choice == "Palmer Penguins":
            return self.prepare_penguins(object_level)
        if dataset_choice == "Portal Species":
            return self.prepare_portal_species()
        if dataset_choice == "Portal Plot Communities":
            return self.prepare_portal_plots()
        if dataset_choice == "Portal Plot-Year Communities":
            years = self.year_values()
            if year is None:
                year = years[0]
            return self.prepare_portal_plot_years(year)
        raise ValueError(f"Unknown dataset choice: {dataset_choice}")


def pairwise_euclidean(x: np.ndarray) -> np.ndarray:
    diff = x[:, None, :] - x[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def distance_to_similarity(dist: np.ndarray) -> np.ndarray:
    if dist.size == 0:
        return dist
    max_dist = float(np.nanmax(dist))
    if not np.isfinite(max_dist) or max_dist <= 1e-12:
        return np.ones_like(dist, dtype=float)
    return np.clip(1.0 - dist / max_dist, 0.0, 1.0)


class SimilarityEngine:
    @staticmethod
    def numeric_similarity(df: pd.DataFrame, features: Sequence[str]) -> SimilarityResult:
        features = [c for c in features if c in df.columns]
        if not features:
            raise ValueError("No numeric features selected.")
        x = df[features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        med = np.nanmedian(x, axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        inds = np.where(~np.isfinite(x))
        if len(inds[0]) > 0:
            x[inds] = np.take(med, inds[1])
        means = x.mean(axis=0)
        stds = x.std(axis=0, ddof=0)
        stds = np.where(stds > 1e-12, stds, 1.0)
        z = (x - means) / stds
        dist = pairwise_euclidean(z)
        sim = distance_to_similarity(dist)

        if len(features) >= 2:
            px = z[:, 0]
            py = z[:, 1]
            xlab = nice_feature_name(features[0])
            ylab = nice_feature_name(features[1])
        else:
            px = z[:, 0]
            py = np.zeros(len(df))
            xlab = nice_feature_name(features[0])
            ylab = "Reference line"

        return SimilarityResult(
            sim=sim,
            method_name="Standardized Euclidean similarity",
            feature_columns=list(features),
            feature_labels=[nice_feature_name(c) for c in features],
            notes=(
                "The selected measurements were standardized first, so each chosen feature has equal footing. "
                "The ecological-space plot uses the first two selected variables as the axes."
            ),
            plot_x=np.asarray(px, dtype=float),
            plot_y=np.asarray(py, dtype=float),
            plot_x_label=xlab,
            plot_y_label=ylab,
        )

    @staticmethod
    def jaccard_similarity(df: pd.DataFrame, features: Sequence[str]) -> SimilarityResult:
        features = [c for c in features if c in df.columns and c.startswith("count_")]
        if not features:
            raise ValueError("No species count columns selected for Jaccard similarity.")
        counts = df[features].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(dtype=float)
        binary = counts > 0
        n = binary.shape[0]
        sim = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(i, n):
                union = np.logical_or(binary[i], binary[j]).sum()
                inter = np.logical_and(binary[i], binary[j]).sum()
                value = 1.0 if union == 0 else inter / union
                sim[i, j] = sim[j, i] = value
        binary_df = pd.DataFrame(binary, columns=features)
        chosen = choose_top_variable_features(binary_df, features)
        px = zscore_array(binary_df[chosen[0]].astype(float).to_numpy())
        py = zscore_array(binary_df[chosen[1]].astype(float).to_numpy())
        return SimilarityResult(
            sim=sim,
            method_name="Jaccard similarity on presence/absence",
            feature_columns=list(features),
            feature_labels=[nice_feature_name(c) for c in features],
            notes=(
                "Counts were converted to present/absent first. The ecological-space plot uses the two species columns "
                "with the greatest variation in presence/absence."
            ),
            plot_x=px,
            plot_y=py,
            plot_x_label=f"Presence of {nice_feature_name(chosen[0])}",
            plot_y_label=f"Presence of {nice_feature_name(chosen[1])}",
        )

    @staticmethod
    def bray_curtis_similarity(df: pd.DataFrame, features: Sequence[str]) -> SimilarityResult:
        features = [c for c in features if c in df.columns and c.startswith("count_")]
        if not features:
            raise ValueError("No species count columns selected for Bray-Curtis similarity.")
        x = df[features].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(dtype=float)
        n = x.shape[0]
        sim = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(i, n):
                denom = np.sum(x[i] + x[j])
                value = 1.0 if denom <= 0 else 1.0 - np.sum(np.abs(x[i] - x[j])) / denom
                sim[i, j] = sim[j, i] = max(0.0, min(1.0, value))
        temp_df = pd.DataFrame(x, columns=features)
        chosen = choose_top_variable_features(temp_df, features)
        px = zscore_array(temp_df[chosen[0]].to_numpy(dtype=float))
        py = zscore_array(temp_df[chosen[1]].to_numpy(dtype=float))
        return SimilarityResult(
            sim=sim,
            method_name="Bray-Curtis abundance similarity",
            feature_columns=list(features),
            feature_labels=[nice_feature_name(c) for c in features],
            notes=(
                "This uses abundance information. The ecological-space plot uses the two count columns with the greatest variation."
            ),
            plot_x=px,
            plot_y=py,
            plot_x_label=nice_feature_name(chosen[0]),
            plot_y_label=nice_feature_name(chosen[1]),
        )

    @staticmethod
    def relative_abundance_similarity(df: pd.DataFrame, features: Sequence[str]) -> SimilarityResult:
        features = [c for c in features if c in df.columns and c.startswith("count_")]
        if not features:
            raise ValueError("No species count columns selected for relative abundance similarity.")
        x = df[features].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(dtype=float)
        totals = x.sum(axis=1, keepdims=True)
        totals = np.where(totals > 0, totals, 1.0)
        props = x / totals
        dist = pairwise_euclidean(props)
        sim = distance_to_similarity(dist)
        temp_df = pd.DataFrame(props, columns=features)
        chosen = choose_top_variable_features(temp_df, features)
        px = temp_df[chosen[0]].to_numpy(dtype=float)
        py = temp_df[chosen[1]].to_numpy(dtype=float)
        return SimilarityResult(
            sim=sim,
            method_name="Relative abundance profile similarity",
            feature_columns=list(features),
            feature_labels=[nice_feature_name(c) for c in features],
            notes=(
                "Counts were converted to relative abundances first. The ecological-space plot uses the two species with the most variable relative abundance."
            ),
            plot_x=px,
            plot_y=py,
            plot_x_label=f"Relative abundance of {nice_feature_name(chosen[0])}",
            plot_y_label=f"Relative abundance of {nice_feature_name(chosen[1])}",
        )

    @staticmethod
    def compute(prepared: PreparedDataset, preset_name: str, selected_features: Sequence[str]) -> SimilarityResult:
        if preset_name == "Presence / absence (Jaccard)":
            return SimilarityEngine.jaccard_similarity(prepared.df, selected_features)
        if preset_name == "Abundance (Bray-Curtis)":
            return SimilarityEngine.bray_curtis_similarity(prepared.df, selected_features)
        if preset_name == "Relative abundance profile":
            return SimilarityEngine.relative_abundance_similarity(prepared.df, selected_features)
        return SimilarityEngine.numeric_similarity(prepared.df, selected_features)


def render_dashboard(prepared: PreparedDataset, result: SimilarityResult) -> str:
    title_div = Div(
        text=f"""
        <div style=\"font-family:Lato,Arial,sans-serif;\">
          <h1 style=\"margin-bottom:0.15em;\">{APP_TITLE}</h1>
          <h2 style=\"margin-top:0;color:#6c1d45;\">{prepared.title}</h2>
          <p style=\"font-size:12pt; max-width:1100px;\">
            <b>Similarity question:</b> {result.method_name}.<br>
            <b>Objects:</b> {prepared.object_description}<br>
            <b>Features used:</b> {', '.join(result.feature_labels)}
          </p>
          <p style=\"font-size:11pt; max-width:1100px; color:#333;\">{result.notes}</p>
        </div>
        """,
        sizing_mode="stretch_width",
    )
    heatmap = create_similarity_heatmap(prepared, result)
    scatter = create_embedding_plot(prepared, result)
    foot = Div(
        text="""
        <div style=\"font-family:Lato,Arial,sans-serif;font-size:10pt;color:#555;margin-top:8px;\">
        Tip: click a cell in the similarity matrix to compare the two selected objects feature by feature.
        In penguin mode the comparison dialog also shows representative species images.
        </div>
        """,
        sizing_mode="stretch_width",
    )
    layout = column(title_div, row(heatmap, scatter, sizing_mode="stretch_width"), foot, sizing_mode="stretch_width")
    return file_html(layout, CDN, APP_TITLE)


def create_similarity_heatmap(prepared: PreparedDataset, result: SimilarityResult):
    sim = result.sim
    labels = prepared.short_labels
    n = len(labels)
    rows = []
    for i in range(n):
        for j in range(n):
            rows.append({
                "x": labels[j],
                "y": labels[i],
                "row_i": i,
                "col_i": j,
                "label_a": prepared.labels[i],
                "label_b": prepared.labels[j],
                "similarity": sim[i, j],
            })
    source = ColumnDataSource(pd.DataFrame(rows))
    mapper = LinearColorMapper(palette=Viridis256, low=0.0, high=1.0)
    height = min(760, max(430, 22 * n + 180))
    width = min(860, max(500, 20 * n + 230))
    p = figure(
        title="Similarity matrix",
        x_range=labels,
        y_range=list(reversed(labels)),
        tools="pan,wheel_zoom,box_zoom,reset,save,tap",
        toolbar_location="above",
        width=width,
        height=height,
    )
    p.title.text_font_size = "15pt"
    p.rect(
        x="x", y="y", width=1, height=1, source=source,
        fill_color={"field": "similarity", "transform": mapper},
        line_color="white", line_alpha=0.35,
    )
    p.add_tools(HoverTool(tooltips=[("Object A", "@label_a"), ("Object B", "@label_b"), ("Similarity", "@similarity{0.000}")]))
    p.xaxis.major_label_orientation = 1.1
    p.xaxis.major_label_text_font_size = "9pt" if n <= 24 else "7pt"
    p.yaxis.major_label_text_font_size = "9pt" if n <= 24 else "7pt"
    p.grid.grid_line_color = None
    color_bar = ColorBar(color_mapper=mapper, ticker=BasicTicker(desired_num_ticks=6), label_standoff=8)
    p.add_layout(color_bar, "right")
    cb = CustomJS(args=dict(source=source), code="""
        const inds = cb_obj.indices;
        if (inds.length > 0) {
            const k = inds[0];
            alert("eco-pair://" + source.data.row_i[k] + "::" + source.data.col_i[k]);
            cb_obj.indices = [];
        }
    """)
    source.selected.js_on_change("indices", cb)
    return p


def create_embedding_plot(prepared: PreparedDataset, result: SimilarityResult):
    groups = [str(g) for g in prepared.groups]
    colors = group_colors(groups)
    source = ColumnDataSource(dict(
        x=result.plot_x,
        y=result.plot_y,
        label=prepared.labels,
        short=prepared.short_labels,
        group=groups,
        color=colors,
    ))
    p = figure(
        title="Ecological space view",
        tools="pan,wheel_zoom,box_zoom,reset,save",
        toolbar_location="above",
        width=540,
        height=430,
    )
    p.title.text_font_size = "15pt"
    p.scatter(x="x", y="y", size=11, source=source, color="color", alpha=0.82, legend_field="group")
    p.add_tools(HoverTool(tooltips=[("Object", "@label"), ("Group", "@group"), (result.plot_x_label, "@x{0.000}"), (result.plot_y_label, "@y{0.000}")]))
    p.xaxis.axis_label = result.plot_x_label
    p.yaxis.axis_label = result.plot_y_label
    p.legend.location = "top_right"
    p.legend.click_policy = "hide"
    p.legend.label_text_font_size = "8pt"
    return p


def species_key_from_row(row: pd.Series) -> Optional[str]:
    if "species" in row.index:
        s = str(row["species"]).strip().lower()
        if "adelie" in s:
            return "adelie"
        if "chinstrap" in s:
            return "chinstrap"
        if "gentoo" in s:
            return "gentoo"
    label = " ".join(str(v).lower() for v in row.values[:3])
    for key in ["adelie", "chinstrap", "gentoo"]:
        if key in label:
            return key
    return None


def find_penguin_image_path(species_key: Optional[str]) -> Optional[Path]:
    """
    Find the representative penguin image in the local Palmer data folder.

    Expected normal location:
        EcoData/Palmer/Adelie.png
        EcoData/Palmer/Chinstrap.png
        EcoData/Palmer/Gentoo.png
    """
    if not species_key:
        return None

    filename = PENGUIN_IMAGE_FILES.get(species_key)
    if not filename:
        return None

    candidate_dirs = []

    # Standard project-root-style locations.
    for root in candidate_roots():
        candidate_dirs.extend([
            root / "EcoData" / "Palmer",
            root / "Palmer",
            root,
        ])

    # If the app already found EcoData/Ready, then EcoData/Palmer is adjacent.
    ready = find_ready_dir()
    if ready is not None:
        candidate_dirs.append(ready.parent / "Palmer")

    for folder in candidate_dirs:
        candidate = folder / filename
        if candidate.exists():
            return candidate

    return None


def pixmap_for_penguin_species(
    species_key: Optional[str],
    max_w: int = 240,
    max_h: int = 280,
) -> Optional[QPixmap]:
    path = find_penguin_image_path(species_key)
    if path is None:
        return None

    pix = QPixmap(str(path))
    if pix.isNull():
        return None

    return pix.scaled(
        max_w,
        max_h,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )

class PairComparisonDialog(QDialog):
    def __init__(self, context: RenderContext, i: int, j: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Pair comparison")
        self.resize(980, 720)
        self.context = context
        self.i = i
        self.j = j
        self._build()

    def _build(self) -> None:
        prepared = self.context.prepared
        result = self.context.result
        df = prepared.df
        sim_value = result.sim[self.i, self.j]
        layout = QVBoxLayout(self)
        header = QLabel(
            f"<b>{prepared.labels[self.i]}</b><br>compared with<br><b>{prepared.labels[self.j]}</b>"
            f"<br><br>Similarity = <b>{sim_value:0.3f}</b> using {result.method_name}"
        )
        header.setWordWrap(True)
        layout.addWidget(header)

        if prepared.dataset_type == "penguins":
            image_row = QHBoxLayout()
            for idx, tag in [(self.i, "Object A"), (self.j, "Object B")]:
                box = QGroupBox(tag + " image")
                box_layout = QVBoxLayout(box)
                species_key = species_key_from_row(df.iloc[idx])
                lab = QLabel(df.iloc[idx].get("species", ""))
                lab.setAlignment(Qt.AlignmentFlag.AlignCenter)
                pix = pixmap_for_penguin_species(species_key)
                img_label = QLabel()
                img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                if pix is not None:
                    img_label.setPixmap(pix)
                else:
                    img_label.setText("No image available")
                box_layout.addWidget(lab)
                box_layout.addWidget(img_label)
                image_row.addWidget(box)
            layout.addLayout(image_row)

        meta_cols = [c for c in prepared.metadata_columns if c in df.columns]
        if meta_cols:
            meta_box = QGroupBox("Metadata")
            meta_layout = QVBoxLayout(meta_box)
            meta_table = QTableWidget(len(meta_cols), 3)
            meta_table.setHorizontalHeaderLabels(["Field", "Object A", "Object B"])
            for r, col in enumerate(meta_cols):
                meta_table.setItem(r, 0, QTableWidgetItem(nice_feature_name(col)))
                meta_table.setItem(r, 1, QTableWidgetItem(format_value(df.iloc[self.i][col])))
                meta_table.setItem(r, 2, QTableWidgetItem(format_value(df.iloc[self.j][col])))
            meta_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            meta_table.verticalHeader().setVisible(False)
            meta_layout.addWidget(meta_table)
            layout.addWidget(meta_box)

        feature_cols = [c for c in result.feature_columns if c in df.columns]
        feat_box = QGroupBox("Features used in this similarity calculation")
        feat_layout = QVBoxLayout(feat_box)
        feat_table = QTableWidget(len(feature_cols), 4)
        feat_table.setHorizontalHeaderLabels(["Feature", "Object A", "Object B", "Difference (A - B)"])
        for r, col in enumerate(feature_cols):
            a = safe_float(df.iloc[self.i][col])
            b = safe_float(df.iloc[self.j][col])
            diff = a - b if np.isfinite(a) and np.isfinite(b) else np.nan
            feat_table.setItem(r, 0, QTableWidgetItem(nice_feature_name(col)))
            feat_table.setItem(r, 1, QTableWidgetItem(format_value(a)))
            feat_table.setItem(r, 2, QTableWidgetItem(format_value(b)))
            feat_table.setItem(r, 3, QTableWidgetItem(format_value(diff)))
        feat_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        feat_table.verticalHeader().setVisible(False)
        feat_layout.addWidget(feat_table)
        layout.addWidget(feat_box)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn, alignment=Qt.AlignmentFlag.AlignRight)


class InterceptingPage(QWebEnginePage):
    def __init__(self, parent):
        super().__init__(parent)
        self.main_window = parent

    def javaScriptAlert(self, origin, msg):
        if isinstance(msg, str) and msg.startswith("eco-pair://"):
            try:
                payload = msg.replace("eco-pair://", "")
                a, b = payload.split("::")
                self.main_window.show_pair_comparison(int(a), int(b))
            except Exception as exc:
                QMessageBox.warning(self.main_window, "Pair comparison error", str(exc))
        else:
            super().javaScriptAlert(origin, msg)


class CobberEcoSimilarApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1450, 920)
        self.cobber_maroon = QColor(108, 29, 69)
        self.cobber_gold = QColor(234, 170, 0)
        self.setFont(QFont("Lato", 10))

        self.data = EcoDataManager()
        self.current_prepared: Optional[PreparedDataset] = None
        self.current_context: Optional[RenderContext] = None
        self._updating = False

        self._build_menu()
        self._build_ui()
        self.setStatusBar(self._create_status_bar())
        self.refresh_controls()
        self.show_welcome()

    def _build_menu(self) -> None:
        menu = self.menuBar()
        file_menu = menu.addMenu("File")
        choose_action = file_menu.addAction("Choose Ready Data Folder...")
        choose_action.triggered.connect(self.choose_ready_folder)
        reload_action = file_menu.addAction("Reload Data")
        reload_action.triggered.connect(self.reload_data)
        file_menu.addSeparator()
        quit_action = file_menu.addAction("Quit")
        quit_action.triggered.connect(self.close)

        help_menu = menu.addMenu("Help")
        about_action = help_menu.addAction("About the Data")
        about_action.triggered.connect(self.show_data_notes)
        explain_action = help_menu.addAction("Explain Current Mode")
        explain_action.triggered.connect(self.explain_current_mode)

    def _build_ui(self) -> None:
        central = QWidget()
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(10)

        sidebar = QWidget()
        side_layout = QVBoxLayout(sidebar)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.setSpacing(10)
        side_layout.addWidget(self._create_similarity_controls(), stretch=0)
        side_layout.addWidget(self._create_feature_box(), stretch=0)
        side_layout.addStretch(1)
        sidebar.setMaximumWidth(430)

        self.web = QWebEngineView()
        self.web.setPage(InterceptingPage(self))

        main_layout.addWidget(sidebar, stretch=0)
        main_layout.addWidget(self.web, stretch=1)
        self.setCentralWidget(central)

    def _create_similarity_controls(self) -> QWidget:
        chooser_box = QGroupBox("Similarity setup")
        grid = QGridLayout(chooser_box)

        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems(DATASET_CHOICES)
        self.dataset_combo.currentTextChanged.connect(self.refresh_controls)

        self.object_combo = QComboBox()
        self.object_combo.currentTextChanged.connect(self.refresh_controls)

        self.preset_combo = QComboBox()
        self.preset_combo.currentTextChanged.connect(self.apply_preset_to_features)

        self.year_combo = QComboBox()
        self.year_combo.currentTextChanged.connect(self.refresh_controls)

        grid.addWidget(QLabel("Dataset:"), 0, 0)
        grid.addWidget(self.dataset_combo, 0, 1)
        grid.addWidget(QLabel("Object level:"), 1, 0)
        grid.addWidget(self.object_combo, 1, 1)
        grid.addWidget(QLabel("Similarity question:"), 2, 0)
        grid.addWidget(self.preset_combo, 2, 1)
        grid.addWidget(QLabel("Year:"), 3, 0)
        grid.addWidget(self.year_combo, 3, 1)

        button_row = QHBoxLayout()
        self.generate_btn = QPushButton("Generate Similarity")
        self.generate_btn.setStyleSheet(
            "QPushButton { font-size: 12pt; padding: 6px 12px; background-color:#6c1d45; color:white; border-radius:4px; }"
            "QPushButton:disabled { background-color:#999999; }"
        )
        self.generate_btn.clicked.connect(self.generate_similarity)
        self.reset_features_btn = QPushButton("Reset Features")
        self.reset_features_btn.clicked.connect(self.apply_preset_to_features)
        self.explain_btn = QPushButton("Explain")
        self.explain_btn.clicked.connect(self.explain_current_mode)
        button_row.addWidget(self.generate_btn)
        button_row.addWidget(self.reset_features_btn)
        button_row.addWidget(self.explain_btn)
        grid.addLayout(button_row, 4, 0, 1, 2)
        return chooser_box

    def _create_feature_box(self) -> QWidget:
        feature_box = QGroupBox("Features included")
        feature_layout = QVBoxLayout(feature_box)
        self.feature_list = QListWidget()
        self.feature_list.setMinimumWidth(380)
        self.feature_list.setMaximumHeight(330)
        feature_layout.addWidget(self.feature_list)
        feature_note = QLabel(
            'Check or uncheck features to change what "similar" means. For community modes, the count features are species columns.'
        )
        feature_note.setWordWrap(True)
        feature_note.setStyleSheet("color:#555; font-weight:normal;")
        feature_layout.addWidget(feature_note)
        return feature_box

    def _create_status_bar(self) -> QStatusBar:
        bar = QStatusBar()
        if self.data.ready_dir:
            bar.showMessage(f"Ready. Data folder: {self.data.ready_dir}")
        else:
            bar.showMessage("Ready. EcoData/Ready folder not found yet.")
        notes_button = QPushButton("Data Notes")
        notes_button.clicked.connect(self.show_data_notes)
        bar.addPermanentWidget(notes_button)
        return bar

    def refresh_controls(self) -> None:
        if self._updating:
            return
        self._updating = True
        try:
            dataset = self.dataset_combo.currentText()
            current_object = self.object_combo.currentText()
            current_year = self.year_combo.currentText()
            current_preset = self.preset_combo.currentText()

            self.object_combo.blockSignals(True)
            self.object_combo.clear()
            if dataset == "Palmer Penguins":
                self.object_combo.addItems(PENGUIN_OBJECT_LEVELS)
            else:
                self.object_combo.addItems(PORTAL_OBJECT_LEVELS.get(dataset, []))
            idx = max(0, self.object_combo.findText(current_object)) if current_object else 0
            self.object_combo.setCurrentIndex(idx if idx >= 0 else 0)
            self.object_combo.blockSignals(False)

            self.year_combo.blockSignals(True)
            self.year_combo.clear()
            years = self.data.year_values() if dataset == "Portal Plot-Year Communities" else []
            for y in years:
                self.year_combo.addItem(str(y))

            self.year_combo.setEnabled(dataset == "Portal Plot-Year Communities")

            if dataset == "Portal Plot-Year Communities" and current_year:
                year_idx = self.year_combo.findText(current_year)
                if year_idx >= 0:
                    self.year_combo.setCurrentIndex(year_idx)
                elif self.year_combo.count() > 0:
                    self.year_combo.setCurrentIndex(0)

            self.year_combo.blockSignals(False)

            self.preset_combo.blockSignals(True)
            self.preset_combo.clear()
            if dataset == "Palmer Penguins":
                presets = list(PENGUIN_FEATURE_PRESETS.keys())
            elif dataset == "Portal Species":
                presets = list(PORTAL_SPECIES_FEATURE_PRESETS.keys())
            else:
                presets = list(COMMUNITY_FEATURE_PRESETS.keys())
            self.preset_combo.addItems(presets)

            preset_idx = self.preset_combo.findText(current_preset)
            if preset_idx >= 0:
                self.preset_combo.setCurrentIndex(preset_idx)

            self.preset_combo.blockSignals(False)

            try:
                year = int(self.year_combo.currentText()) if self.year_combo.isEnabled() and self.year_combo.currentText() else None
                prepared = self.data.build_prepared(dataset_choice=dataset, object_level=self.object_combo.currentText(), year=year)
                self.current_prepared = prepared
                self.populate_feature_list()
            except Exception:
                self.current_prepared = None
                self.feature_list.clear()
        finally:
            self._updating = False

    def populate_feature_list(self) -> None:
        self.feature_list.clear()
        prepared = self.current_prepared
        if prepared is None:
            return
        if prepared.title.startswith("Palmer Penguins"):
            default = list(PENGUIN_FEATURE_PRESETS.get(self.preset_combo.currentText(), prepared.default_features))
        elif prepared.title.startswith("Portal Species"):
            default = list(PORTAL_SPECIES_FEATURE_PRESETS.get(self.preset_combo.currentText(), prepared.default_features))
        else:
            preset = self.preset_combo.currentText()
            if preset == "Summary ecology":
                default = list(COMMUNITY_FEATURE_PRESETS[preset])
            else:
                default = [c for c in prepared.all_features if c.startswith("count_")]
        for col in prepared.all_features:
            item = QListWidgetItem(nice_feature_name(col))
            item.setData(Qt.ItemDataRole.UserRole, col)
            flags = item.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled
            item.setFlags(flags)
            item.setCheckState(Qt.CheckState.Checked if col in default else Qt.CheckState.Unchecked)
            self.feature_list.addItem(item)

    def apply_preset_to_features(self) -> None:
        if self.current_prepared is None:
            return
        self.populate_feature_list()

    def selected_feature_columns(self) -> List[str]:
        cols: List[str] = []
        for i in range(self.feature_list.count()):
            item = self.feature_list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                col = item.data(Qt.ItemDataRole.UserRole)
                if col:
                    cols.append(col)
        return cols

    def choose_ready_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Choose Ready data folder", str(app_dir()))
        if folder:
            self.data.set_ready_dir(Path(folder))
            self.statusBar().showMessage(f"Data folder set to: {folder}")
            self.refresh_controls()
            self.show_welcome()

    def reload_data(self) -> None:
        self.data._cache.clear()
        self.refresh_controls()
        self.show_welcome()
        self.statusBar().showMessage("Data reloaded.")

    def show_welcome(self) -> None:
        html = """
        <html><body style="font-family:Lato,Arial,sans-serif; margin:20px;">
        <h1 style="color:#6c1d45;">CobberEcoSimilar</h1>
        <p>This app explores ecological similarity.</p>
        <ol>
          <li>Choose a dataset and a similarity question.</li>
          <li>Select which features define similarity.</li>
          <li>Click <b>Generate Similarity</b>.</li>
          <li>Click a matrix cell to compare two objects in detail.</li>
        </ol>
        <p>Version 3 simplifies the controls and uses real ecological variables for the ecological-space axes.</p>
        </body></html>
        """
        self.web.setHtml(html)

    def explain_current_mode(self) -> None:
        dataset = self.dataset_combo.currentText()
        preset = self.preset_combo.currentText()
        mapping = {
            "Palmer Penguins": "Use penguin physical measurements to see how similarity changes when you choose different body traits.",
            "Portal Species": "Use real species-level ecological features from the Portal Project to compare ecological profiles.",
            "Portal Plot Communities": "Compare plots as ecological communities using presence/absence, abundance, or summary ecology.",
            "Portal Plot-Year Communities": "Compare plots within a selected year to see how similar their communities look in that year.",
        }
        extra = {
            "Presence / absence (Jaccard)": "This asks whether the same species are present, not whether their abundances match.",
            "Abundance (Bray-Curtis)": "This asks whether the abundances of the species are similar.",
            "Relative abundance profile": "This asks whether community composition is similar after accounting for differences in total captures.",
        }
        QMessageBox.information(self, "Explain current mode", mapping.get(dataset, "") + "\n\n" + extra.get(preset, ""))

    def show_data_notes(self) -> None:
        summary = self.data.read_text_if_exists("chapter6_data_summary.txt")
        license_text = self.data.read_text_if_exists("license_and_citation_notes.txt")
        text = summary or "No summary file found."
        if license_text:
            text += "\n\n" + license_text
        dlg = QDialog(self)
        dlg.setWindowTitle("About the data")
        dlg.resize(760, 640)
        lay = QVBoxLayout(dlg)
        lab = QLabel("<b>Data summary and notes</b>")
        lab.setWordWrap(True)
        lay.addWidget(lab)
        note = QLabel(text)
        note.setWordWrap(True)
        note.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        lay.addWidget(note)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        lay.addWidget(close_btn, alignment=Qt.AlignmentFlag.AlignRight)
        dlg.exec()

    def generate_similarity(self) -> None:
        try:
            dataset = self.dataset_combo.currentText()
            year = int(self.year_combo.currentText()) if self.year_combo.isEnabled() and self.year_combo.currentText() else None
            prepared = self.data.build_prepared(dataset_choice=dataset, object_level=self.object_combo.currentText(), year=year)
            selected_features = self.selected_feature_columns()
            if not selected_features:
                if dataset == "Palmer Penguins":
                    selected_features = list(PENGUIN_FEATURE_PRESETS.get(self.preset_combo.currentText(), prepared.default_features))
                elif dataset == "Portal Species":
                    selected_features = list(PORTAL_SPECIES_FEATURE_PRESETS.get(self.preset_combo.currentText(), prepared.default_features))
                else:
                    if self.preset_combo.currentText() == "Summary ecology":
                        selected_features = list(COMMUNITY_FEATURE_PRESETS["Summary ecology"])
                    else:
                        selected_features = [c for c in prepared.all_features if c.startswith("count_")]

            result = SimilarityEngine.compute(prepared, self.preset_combo.currentText(), selected_features)
            self.current_context = RenderContext(prepared=prepared, result=result)
            html = render_dashboard(prepared, result)
            self.web.setHtml(html)
            self.statusBar().showMessage(f"Generated similarity for {prepared.title} using {result.method_name}.")
        except Exception as exc:
            traceback.print_exc()
            QMessageBox.critical(self, "Generation error", f"Could not generate similarity.\n\n{exc}")

    def show_pair_comparison(self, i: int, j: int) -> None:
        if self.current_context is None:
            return
        dlg = PairComparisonDialog(self.current_context, i, j, parent=self)
        dlg.exec()


def apply_app_stylesheet(app: QApplication) -> None:
    app.setStyleSheet("""
        QWidget { color: #222222; background-color: #ffffff; }
        QMainWindow, QDialog { background-color: #ffffff; }
        QMenuBar, QMenu { background-color: #ffffff; color: #222222; }
        QMenuBar::item:selected, QMenu::item:selected { background-color: #6c1d45; color: #ffffff; }
        QGroupBox {
            color: #222222;
            font-weight: bold;
            border: 1px solid #d6d6d6;
            border-radius: 5px;
            margin-top: 8px;
            padding-top: 10px;
            background-color: #fafafa;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 4px 0 4px;
            color: #6c1d45;
            background-color: #fafafa;
        }
        QLabel { color: #222222; background-color: transparent; }
        QComboBox {
            background-color: #ffffff;
            color: #111111;
            border: 1px solid #9a9a9a;
            border-radius: 3px;
            padding: 3px 6px;
            min-height: 24px;
            selection-background-color: #6c1d45;
            selection-color: #ffffff;
        }
        QComboBox:disabled { background-color: #eeeeee; color: #555555; }
        QComboBox QAbstractItemView {
            background-color: #ffffff;
            color: #111111;
            selection-background-color: #6c1d45;
            selection-color: #ffffff;
            border: 1px solid #9a9a9a;
        }
        QListWidget, QTableWidget {
            background-color: #ffffff;
            color: #111111;
            alternate-background-color: #f4f4f4;
            selection-background-color: #6c1d45;
            selection-color: #ffffff;
            border: 1px solid #cfcfcf;
        }
        QHeaderView::section {
            background-color: #eeeeee;
            color: #222222;
            padding: 4px;
            border: 1px solid #cccccc;
        }
        QPushButton {
            background-color: #f7f7f7;
            color: #111111;
            border: 1px solid #9a9a9a;
            border-radius: 4px;
            padding: 5px 10px;
        }
        QPushButton:hover { background-color: #eeeeee; }
        QPushButton:pressed { background-color: #dddddd; }
        QStatusBar { background-color: #f7f7f7; color: #222222; }
    """)


def main() -> None:
    app = QApplication(sys.argv)
    apply_app_stylesheet(app)
    win = CobberEcoSimilarApp()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
