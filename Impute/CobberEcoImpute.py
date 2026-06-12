#!/usr/bin/env python3
"""
CobberEcoImpute.py

A guided PyQt6 application for exploring imputation with weekly snow-cover data.
Designed for the Ecology ML book as the ecology analogue of CobberImpute.

Expected data location, relative to this script or the working directory:

    EcoData/Snow/Snowcoverdata_Rutgers.txt

The app also searches the current directory and a few nearby folders. The input
file is the Rutgers weekly snow-cover table with columns:

    Row, Year, Week, N. Hemisphere, Eurasia, N. America, N. America (no Greenland)

Core teaching structure:
    1. Inspect observed and missing data.
    2. Create artificial test gaps by hiding known data.
    3. Impute the gaps with transparent methods.
    4. Compare imputed values with hidden true values.
    5. Impute real missing entries where no answer key exists.

Dependencies:
    pip install pandas numpy matplotlib scikit-learn PyQt6
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

from PyQt6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PyQt6.QtGui import QAction, QBrush, QColor, QFont
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QStatusBar,
    QTableView,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as Canvas
from matplotlib.figure import Figure

APP_TITLE = "CobberEcoImpute"
REGIONS = ["N. Hemisphere", "Eurasia", "N. America", "N. America (no Greenland)"]


# ---------------------------------------------------------------------
# File/data helpers
# ---------------------------------------------------------------------


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


def find_snow_file() -> Optional[Path]:
    candidate = app_dir() / "Snowcoverdata_Rutgers.txt"
    return candidate if candidate.exists() else None


def decimal_time(year: pd.Series, week: pd.Series) -> pd.Series:
    return year.astype(float) + (week.astype(float) - 1.0) / 52.0


def read_snow_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    df.columns = [c.strip() for c in df.columns]
    required = ["Year", "Week"] + REGIONS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Snow file missing expected columns: {missing}")
    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Year", "Week"]).copy()
    df["Year"] = df["Year"].astype(int)
    df["Week"] = df["Week"].astype(int)
    return df


def wide_to_long_with_grid(wide: pd.DataFrame) -> pd.DataFrame:
    obs = wide.melt(
        id_vars=["Year", "Week"],
        value_vars=REGIONS,
        var_name="Region",
        value_name="SnowCover",
    )
    obs["SnowCover"] = pd.to_numeric(obs["SnowCover"], errors="coerce")

    min_year = int(obs["Year"].min())
    max_year = int(obs["Year"].max())
    grid = pd.MultiIndex.from_product(
        [range(min_year, max_year + 1), range(1, 53), REGIONS],
        names=["Year", "Week", "Region"],
    ).to_frame(index=False)
    full = grid.merge(obs, on=["Year", "Week", "Region"], how="left")
    full["DecimalYear"] = decimal_time(full["Year"], full["Week"])
    full["OriginalMissing"] = full["SnowCover"].isna()
    full["TestMissing"] = False
    full["Imputed"] = False
    full["ImputedValue"] = np.nan
    full["WorkingValue"] = full["SnowCover"]
    full["Method"] = ""
    return full.sort_values(["Year", "Week", "Region"]).reset_index(drop=True)


# ---------------------------------------------------------------------
# Qt dataframe model
# ---------------------------------------------------------------------


class DataFrameModel(QAbstractTableModel):
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self._df = df

    def rowCount(self, parent=QModelIndex()):
        return len(self._df)

    def columnCount(self, parent=QModelIndex()):
        return self._df.shape[1]

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        val = self._df.iat[index.row(), index.column()]
        col = self._df.columns[index.column()]
        if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole):
            if pd.isna(val):
                return ""
            if isinstance(val, (float, np.floating)):
                return f"{val:.3f}"
            return str(val)
        if role == Qt.ItemDataRole.ForegroundRole:
            if pd.isna(val):
                return QBrush(QColor("red"))
            if col in ("TestMissing", "OriginalMissing", "Imputed") and bool(val):
                return QBrush(QColor(108, 29, 69))
        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return str(self._df.columns[section])
        return str(self._df.index[section])


# ---------------------------------------------------------------------
# Imputation engine
# ---------------------------------------------------------------------


@dataclass
class ImputeResult:
    df: pd.DataFrame
    method: str
    target_mask: pd.Series
    metrics: Dict[str, float]
    comparison: pd.DataFrame


class SnowImputer:
    @staticmethod
    def overall_mean(df: pd.DataFrame, target_mask: pd.Series, region: str) -> pd.Series:
        train = df[(df["Region"] == region) & df["WorkingValue"].notna()]
        mean_val = float(train["WorkingValue"].mean())
        return pd.Series(mean_val, index=df.index[target_mask])

    @staticmethod
    def week_average(df: pd.DataFrame, target_mask: pd.Series, region: str) -> pd.Series:
        train = df[(df["Region"] == region) & df["WorkingValue"].notna()]
        week_means = train.groupby("Week")["WorkingValue"].mean()
        overall = train["WorkingValue"].mean()
        weeks = df.loc[target_mask, "Week"]
        vals = weeks.map(week_means).fillna(overall).astype(float)
        vals.index = df.index[target_mask]
        return vals

    @staticmethod
    def neighboring_year_average(df: pd.DataFrame, target_mask: pd.Series, region: str, window: int) -> pd.Series:
        train = df[(df["Region"] == region) & df["WorkingValue"].notna()].copy()
        week_means = train.groupby("Week")["WorkingValue"].mean()
        overall = train["WorkingValue"].mean()
        out: Dict[int, float] = {}
        for idx, row in df.loc[target_mask].iterrows():
            y = int(row["Year"])
            w = int(row["Week"])
            nearby = train[(train["Week"] == w) & (train["Year"].between(y - window, y + window))]
            if len(nearby) > 0:
                out[idx] = float(nearby["WorkingValue"].mean())
            elif w in week_means.index:
                out[idx] = float(week_means.loc[w])
            else:
                out[idx] = float(overall)
        return pd.Series(out)

    @staticmethod
    def linear_interpolation(df: pd.DataFrame, target_mask: pd.Series, region: str) -> pd.Series:
        sub = df[df["Region"] == region].sort_values("DecimalYear").copy()
        # Interpolate across the entire time series. This can fill full-year gaps
        # by borrowing from surrounding weeks/years.
        interp = sub.set_index("DecimalYear")["WorkingValue"].interpolate(
            method="index", limit_direction="both"
        )
        out: Dict[int, float] = {}
        for idx, row in df.loc[target_mask].iterrows():
            key = float(row["DecimalYear"])
            # There should be an exact key because DecimalYear was calculated consistently.
            out[idx] = float(interp.loc[key])
        return pd.Series(out)

    @staticmethod
    def knn_similar_years(df: pd.DataFrame, target_mask: pd.Series, region: str, k: int) -> pd.Series:
        # Build a year x week matrix for the selected region.
        sub = df[df["Region"] == region].copy()
        matrix = sub.pivot_table(index="Year", columns="Week", values="WorkingValue", aggfunc="mean")
        target_rows = df.loc[target_mask, ["Year", "Week"]]
        out: Dict[int, float] = {}
        fallback = SnowImputer.week_average(df, target_mask, region)

        for idx, row in target_rows.iterrows():
            year = int(row["Year"])
            week = int(row["Week"])
            if year not in matrix.index or week not in matrix.columns:
                out[idx] = float(fallback.loc[idx])
                continue

            target_vec = matrix.loc[year].copy()
            observed_weeks = target_vec[target_vec.notna()].index.tolist()
            # If the target year has too little remaining information, fall back.
            if len(observed_weeks) < 6:
                out[idx] = float(fallback.loc[idx])
                continue

            train = matrix[matrix.index != year].copy()
            train = train[train[week].notna()]
            if train.empty:
                out[idx] = float(fallback.loc[idx])
                continue

            use_weeks = [w for w in observed_weeks if w in train.columns]
            if len(use_weeks) < 6:
                out[idx] = float(fallback.loc[idx])
                continue

            x_train = train[use_weeks].copy()
            y_train = train[week].copy()
            # Fill remaining holes in training features by weekly means.
            x_train = x_train.apply(lambda col: col.fillna(col.mean()), axis=0)
            target_x = target_vec[use_weeks].to_frame().T
            scaler = StandardScaler()
            x_scaled = scaler.fit_transform(x_train)
            target_scaled = scaler.transform(target_x)
            kk = max(1, min(int(k), len(x_train)))
            model = KNeighborsRegressor(n_neighbors=kk, weights="distance")
            model.fit(x_scaled, y_train)
            out[idx] = float(model.predict(target_scaled)[0])
        return pd.Series(out)

    @staticmethod
    def impute(
        df: pd.DataFrame,
        target_mask: pd.Series,
        region: str,
        method: str,
        neighbor_window: int = 5,
        k: int = 5,
    ) -> ImputeResult:
        if not target_mask.any():
            return ImputeResult(df.copy(), method, target_mask, {}, pd.DataFrame())

        if method == "Overall mean":
            preds = SnowImputer.overall_mean(df, target_mask, region)
        elif method == "Week-of-year average":
            preds = SnowImputer.week_average(df, target_mask, region)
        elif method == "Neighboring-year average":
            preds = SnowImputer.neighboring_year_average(df, target_mask, region, neighbor_window)
        elif method == "Linear interpolation":
            preds = SnowImputer.linear_interpolation(df, target_mask, region)
        elif method == "KNN similar years":
            preds = SnowImputer.knn_similar_years(df, target_mask, region, k)
        else:
            raise ValueError(f"Unknown imputation method: {method}")

        out = df.copy()
        out.loc[preds.index, "ImputedValue"] = preds.values
        out.loc[preds.index, "WorkingValue"] = preds.values
        out.loc[preds.index, "Imputed"] = True
        out.loc[preds.index, "Method"] = method

        comparison = out.loc[preds.index, ["Year", "Week", "Region", "SnowCover", "WorkingValue", "OriginalMissing", "TestMissing"]].copy()
        comparison = comparison.rename(columns={"SnowCover": "TrueValue", "WorkingValue": "ImputedValue"})
        known = comparison["TrueValue"].notna()
        metrics: Dict[str, float] = {}
        if known.any():
            err = comparison.loc[known, "ImputedValue"] - comparison.loc[known, "TrueValue"]
            metrics = {
                "MAE": float(np.mean(np.abs(err))),
                "RMSE": float(np.sqrt(np.mean(err**2))),
                "Bias": float(np.mean(err)),
                "MaxAbsError": float(np.max(np.abs(err))),
                "N": int(known.sum()),
            }
            comparison.loc[known, "Error"] = err
        return ImputeResult(out, method, target_mask, metrics, comparison)


# ---------------------------------------------------------------------
# Plot widget helper
# ---------------------------------------------------------------------


class PlotCanvas(Canvas):
    def __init__(self, width=6, height=4):
        self.fig = Figure(figsize=(width, height), dpi=100)
        super().__init__(self.fig)

    def clear_with_message(self, message: str) -> None:
        self.fig.clf()
        ax = self.fig.add_subplot(111)
        ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
        ax.set_axis_off()
        self.fig.tight_layout()
        self.draw()


# ---------------------------------------------------------------------
# Main pages
# ---------------------------------------------------------------------


class InspectPage(QWidget):
    def __init__(self, main: "CobberEcoImputeApp"):
        super().__init__(main)
        self.main = main
        layout = QVBoxLayout(self)

        controls = QHBoxLayout()
        self.region_combo = QComboBox()
        self.region_combo.addItems(REGIONS)
        self.year_combo = QComboBox()
        self.refresh_years()
        self.region_combo.currentTextChanged.connect(self.refresh)
        self.year_combo.currentTextChanged.connect(self.refresh)
        controls.addWidget(QLabel("Region:"))
        controls.addWidget(self.region_combo)
        controls.addWidget(QLabel("Selected year:"))
        controls.addWidget(self.year_combo)
        controls.addStretch()
        layout.addLayout(controls)

        self.report = QTextEdit(readOnly=True)
        self.report.setMaximumHeight(115)
        self.table = QTableView()
        self.plot = PlotCanvas(width=8, height=4)
        self.year_plot = PlotCanvas(width=8, height=3)

        layout.addWidget(QLabel("Missing-data report"))
        layout.addWidget(self.report)
        layout.addWidget(self.plot, stretch=2)
        layout.addWidget(self.year_plot, stretch=2)
        layout.addWidget(QLabel("Data preview"))
        layout.addWidget(self.table, stretch=2)
        self.refresh()

    def refresh_years(self) -> None:
        self.year_combo.blockSignals(True)
        self.year_combo.clear()
        if self.main.working_df is not None:
            years = sorted(self.main.working_df["Year"].unique())
            self.year_combo.addItems([str(y) for y in years])
            if "2020" in [str(y) for y in years]:
                self.year_combo.setCurrentText("2020")
        self.year_combo.blockSignals(False)

    def refresh(self) -> None:
        df = self.main.working_df
        if df is None:
            return
        region = self.region_combo.currentText() or REGIONS[0]
        year = int(self.year_combo.currentText()) if self.year_combo.currentText() else int(df["Year"].max())
        sub = df[df["Region"] == region].copy()
        real_missing = int(sub["OriginalMissing"].sum())
        observed = int(sub["SnowCover"].notna().sum())
        test_missing = int(sub["TestMissing"].sum())
        imputed = int(sub["Imputed"].sum())
        missing_years = sorted(sub.loc[sub["OriginalMissing"], "Year"].unique().tolist())
        self.report.setPlainText(
            f"Region: {region}\n"
            f"Observed entries in source file: {observed}\n"
            f"Grid entries without source values: {real_missing}\n"
            f"Artificial test-missing entries currently hidden: {test_missing}\n"
            f"Currently imputed entries: {imputed}\n"
            f"Years with real missing grid entries: {missing_years[:12]}{'...' if len(missing_years) > 12 else ''}"
        )
        preview = sub[["Year", "Week", "Region", "SnowCover", "WorkingValue", "OriginalMissing", "TestMissing", "Imputed"]].head(300)
        self.table.setModel(DataFrameModel(preview.reset_index(drop=True)))
        self._plot_full(sub, region)
        self._plot_year(sub[sub["Year"] == year], region, year)

    def _plot_full(self, sub: pd.DataFrame, region: str) -> None:
        self.plot.fig.clf()
        ax = self.plot.fig.add_subplot(111)
        observed = sub[sub["SnowCover"].notna()]
        ax.plot(observed["DecimalYear"], observed["SnowCover"], linewidth=0.9)
        test = sub[sub["TestMissing"]]
        if not test.empty:
            ax.scatter(test["DecimalYear"], test["SnowCover"], marker="x", s=28, label="Hidden true values")
        real = sub[sub["OriginalMissing"]]
        if not real.empty:
            ax.scatter(real["DecimalYear"], np.zeros(len(real)), marker="|", s=12, label="Real missing grid entries")
        ax.set_title(f"Weekly snow cover record: {region}")
        ax.set_xlabel("Year")
        ax.set_ylabel("Snow cover")
        if not test.empty or not real.empty:
            ax.legend(fontsize=8)
        self.plot.fig.tight_layout()
        self.plot.draw()

    def _plot_year(self, sub: pd.DataFrame, region: str, year: int) -> None:
        self.year_plot.fig.clf()
        ax = self.year_plot.fig.add_subplot(111)
        ax.plot(sub["Week"], sub["SnowCover"], marker="o", label="Source value")
        missing = sub[sub["WorkingValue"].isna()]
        if not missing.empty:
            ax.scatter(missing["Week"], [0] * len(missing), marker="x", s=45, label="Missing in working data")
        imputed = sub[sub["Imputed"]]
        if not imputed.empty:
            ax.scatter(imputed["Week"], imputed["WorkingValue"], marker="s", s=45, label="Imputed")
        ax.set_title(f"Seasonal curve for {region}, {year}")
        ax.set_xlabel("Week")
        ax.set_ylabel("Snow cover")
        ax.set_xlim(1, 52)
        ax.legend(fontsize=8)
        self.year_plot.fig.tight_layout()
        self.year_plot.draw()


class TestGapPage(QWidget):
    def __init__(self, main: "CobberEcoImputeApp"):
        super().__init__(main)
        self.main = main
        layout = QVBoxLayout(self)
        controls_box = QGroupBox("Create an artificial test gap")
        form = QFormLayout(controls_box)
        self.region_combo = QComboBox(); self.region_combo.addItems(REGIONS)
        self.year_combo = QComboBox()
        self.pattern_combo = QComboBox()
        self.pattern_combo.addItems(["Full year", "Winter weeks", "Melt-season weeks", "Continuous block", "Random weeks"])
        self.start_week = QSpinBox(); self.start_week.setRange(1, 52); self.start_week.setValue(10)
        self.end_week = QSpinBox(); self.end_week.setRange(1, 52); self.end_week.setValue(20)
        self.random_n = QSpinBox(); self.random_n.setRange(1, 52); self.random_n.setValue(12)
        self.seed = QSpinBox(); self.seed.setRange(0, 999999); self.seed.setValue(7)
        self.create_btn = QPushButton("Create Test Gap")
        self.clear_btn = QPushButton("Clear Test Gap")
        button_row = QHBoxLayout(); button_row.addWidget(self.create_btn); button_row.addWidget(self.clear_btn)
        form.addRow("Region:", self.region_combo)
        form.addRow("Year to hide:", self.year_combo)
        form.addRow("Gap pattern:", self.pattern_combo)
        form.addRow("Start week:", self.start_week)
        form.addRow("End week:", self.end_week)
        form.addRow("Random weeks to hide:", self.random_n)
        form.addRow("Random seed:", self.seed)
        form.addRow(button_row)
        layout.addWidget(controls_box)
        self.summary = QTextEdit(readOnly=True); self.summary.setMaximumHeight(90)
        layout.addWidget(self.summary)
        self.plot = PlotCanvas(width=9, height=5)
        layout.addWidget(self.plot, stretch=1)
        self.create_btn.clicked.connect(self.create_gap)
        self.clear_btn.clicked.connect(self.clear_gap)
        self.pattern_combo.currentTextChanged.connect(self.update_control_states)
        self.refresh_years()
        self.update_control_states()
        self.refresh()

    def refresh_years(self) -> None:
        self.year_combo.clear()
        if self.main.working_df is not None:
            years = sorted(self.main.working_df["Year"].unique())
            self.year_combo.addItems([str(y) for y in years if y < max(years)])
            if self.year_combo.findText("2020") >= 0:
                self.year_combo.setCurrentText("2020")

    def update_control_states(self) -> None:
        pattern = self.pattern_combo.currentText()
        block = pattern == "Continuous block"
        rand = pattern == "Random weeks"
        self.start_week.setEnabled(block)
        self.end_week.setEnabled(block)
        self.random_n.setEnabled(rand)
        self.seed.setEnabled(rand)

    def selected_weeks(self) -> List[int]:
        pattern = self.pattern_combo.currentText()
        if pattern == "Full year":
            return list(range(1, 53))
        if pattern == "Winter weeks":
            return list(range(1, 14)) + list(range(48, 53))
        if pattern == "Melt-season weeks":
            return list(range(14, 27))
        if pattern == "Continuous block":
            a, b = sorted([self.start_week.value(), self.end_week.value()])
            return list(range(a, b + 1))
        if pattern == "Random weeks":
            rng = np.random.default_rng(self.seed.value())
            n = min(self.random_n.value(), 52)
            return sorted(rng.choice(np.arange(1, 53), size=n, replace=False).tolist())
        return []

    def create_gap(self) -> None:
        df = self.main.working_df
        if df is None:
            return
        region = self.region_combo.currentText()

        if not self.year_combo.currentText():
            QMessageBox.warning(self, "No year selected", "No year is selected for the test gap.")
            return

        year = int(self.year_combo.currentText())
        weeks = self.selected_weeks()
        mask = (df["Region"] == region) & (df["Year"] == year) & (df["Week"].isin(weeks)) & df["SnowCover"].notna()
        if not mask.any():
            QMessageBox.warning(self, "No observed values", "The selected gap does not contain observed values to hide.")
            return
        # Reset previous test gap first to keep the prototype simple and clear.
        self.main.clear_test_gap()
        self.main.truth_df = df.loc[mask, ["Year", "Week", "Region", "SnowCover"]].copy()
        self.main.truth_df = self.main.truth_df.rename(columns={"SnowCover": "TrueValue"}).reset_index(drop=True)
        self.main.working_df.loc[mask, "WorkingValue"] = np.nan
        self.main.working_df.loc[mask, "TestMissing"] = True
        self.main.working_df.loc[mask, "Imputed"] = False
        self.main.working_df.loc[mask, "ImputedValue"] = np.nan
        self.main.working_df.loc[mask, "Method"] = ""
        self.main.last_result = None
        self.main.test_gap_info = {"region": region, "year": year, "weeks": weeks, "pattern": self.pattern_combo.currentText()}
        self.main.refresh_all_pages()
        self.main.statusBar().showMessage(f"Created test gap: {region}, {year}, {len(self.main.truth_df)} hidden values.")

    def clear_gap(self) -> None:
        self.main.clear_test_gap()
        self.main.refresh_all_pages()
        self.main.statusBar().showMessage("Cleared artificial test gap.")

    def refresh(self) -> None:
        info = self.main.test_gap_info
        if info:
            self.summary.setPlainText(
                f"Current test gap: {info['region']}, {info['year']}, {info['pattern']}; "
                f"{len(info['weeks'])} selected weeks. Hidden true values are stored internally."
            )
        else:
            self.summary.setPlainText("No artificial test gap has been created yet.")
        self._plot()

    def _plot(self) -> None:
        self.plot.fig.clf()
        ax = self.plot.fig.add_subplot(111)
        df = self.main.working_df
        if df is None:
            self.plot.clear_with_message("No data loaded.")
            return
        region = self.region_combo.currentText() or REGIONS[0]
        if self.year_combo.currentText():
            year = int(self.year_combo.currentText())
        else:
            year = int(df["Year"].max())
        sub = df[(df["Region"] == region) & (df["Year"] == year)].copy()
        ax.plot(sub["Week"], sub["SnowCover"], marker="o", label="True/source curve")
        hidden = sub[sub["TestMissing"]]
        if not hidden.empty:
            ax.scatter(hidden["Week"], hidden["SnowCover"], marker="x", s=75, label="Hidden from working data")
        visible = sub[sub["WorkingValue"].notna()]
        ax.plot(visible["Week"], visible["WorkingValue"], linewidth=2, alpha=0.75, label="Working data visible to imputer")
        ax.set_title(f"Test-gap view: {region}, {year}")
        ax.set_xlabel("Week")
        ax.set_ylabel("Snow cover")
        ax.set_xlim(1, 52)
        ax.legend(fontsize=8)
        self.plot.fig.tight_layout()
        self.plot.draw()


class ImputeEvaluatePage(QWidget):
    def __init__(self, main: "CobberEcoImputeApp"):
        super().__init__(main)
        self.main = main
        layout = QVBoxLayout(self)
        controls = QGroupBox("Impute the current artificial test gap")
        grid = QGridLayout(controls)
        self.method_combo = QComboBox()
        self.method_combo.addItems(["Overall mean", "Week-of-year average", "Neighboring-year average", "Linear interpolation", "KNN similar years"])
        self.window_spin = QSpinBox(); self.window_spin.setRange(1, 30); self.window_spin.setValue(5)
        self.k_spin = QSpinBox(); self.k_spin.setRange(1, 20); self.k_spin.setValue(5)
        self.run_btn = QPushButton("Run Imputation")
        grid.addWidget(QLabel("Method:"), 0, 0); grid.addWidget(self.method_combo, 0, 1)
        grid.addWidget(QLabel("Neighbor-year window:"), 1, 0); grid.addWidget(self.window_spin, 1, 1)
        grid.addWidget(QLabel("K for KNN:"), 2, 0); grid.addWidget(self.k_spin, 2, 1)
        grid.addWidget(self.run_btn, 3, 0, 1, 2)
        layout.addWidget(controls)
        self.metrics = QLabel("Create a test gap, then run an imputation method.")
        self.metrics.setWordWrap(True)
        layout.addWidget(self.metrics)
        plots = QHBoxLayout()
        self.overlay_plot = PlotCanvas(width=6, height=4)
        self.error_plot = PlotCanvas(width=6, height=4)
        plots.addWidget(self.overlay_plot)
        plots.addWidget(self.error_plot)
        layout.addLayout(plots, stretch=1)
        self.quality_plot = PlotCanvas(width=8, height=3.5)
        layout.addWidget(self.quality_plot, stretch=1)
        self.table = QTableView()
        layout.addWidget(QLabel("Comparison table"))
        layout.addWidget(self.table, stretch=1)
        self.run_btn.clicked.connect(self.run_impute)
        self.method_combo.currentTextChanged.connect(self.update_param_states)
        self.update_param_states()
        self.refresh()

    def update_param_states(self) -> None:
        method = self.method_combo.currentText()
        self.window_spin.setEnabled(method == "Neighboring-year average")
        self.k_spin.setEnabled(method == "KNN similar years")

    def run_impute(self) -> None:
        info = self.main.test_gap_info
        if not info or self.main.working_df is None:
            QMessageBox.warning(self, "No test gap", "Create a test gap first.")
            return
        target_mask = (
                (self.main.working_df["Region"] == info["region"])
                & self.main.working_df["TestMissing"]
        )

        # Reset the artificial test gap before each new method.
        # Otherwise the next method may accidentally use the previous method's
        # imputed values as training/neighbor/interpolation data.
        self.main.working_df.loc[target_mask, "WorkingValue"] = np.nan
        self.main.working_df.loc[target_mask, "Imputed"] = False
        self.main.working_df.loc[target_mask, "ImputedValue"] = np.nan
        self.main.working_df.loc[target_mask, "Method"] = ""

        result = SnowImputer.impute(
            self.main.working_df,
            target_mask,
            region=info["region"],
            method=self.method_combo.currentText(),
            neighbor_window=self.window_spin.value(),
            k=self.k_spin.value(),
        )
        self.main.working_df = result.df
        self.main.last_result = result
        self.main.refresh_all_pages(skip=self)
        self.refresh()
        self.main.statusBar().showMessage(f"Imputed test gap using {result.method}.")

    def refresh(self) -> None:
        result = self.main.last_result
        if result is None or result.comparison.empty:
            self.metrics.setText("Create a test gap, then run an imputation method.")
            self.overlay_plot.clear_with_message("No imputation result yet.")
            self.error_plot.clear_with_message("No error plot yet.")
            self.quality_plot.clear_with_message("No prediction-quality plot yet.")
            self.table.setModel(DataFrameModel(pd.DataFrame()))
            return
        m = result.metrics
        if m:
            self.metrics.setText(
                f"Method: {result.method} | N = {m['N']} | MAE = {m['MAE']:.3f} | "
                f"RMSE = {m['RMSE']:.3f} | Bias = {m['Bias']:.3f} | Max abs error = {m['MaxAbsError']:.3f}"
            )
        else:
            self.metrics.setText("No hidden true values were available for scoring.")
        self.table.setModel(DataFrameModel(result.comparison.reset_index(drop=True)))
        self._plot_overlay(result)
        self._plot_error(result)
        self._plot_quality(result)

    def _plot_overlay(self, result: ImputeResult) -> None:
        self.overlay_plot.fig.clf()
        ax = self.overlay_plot.fig.add_subplot(111)
        comp = result.comparison.sort_values("Week")
        ax.plot(comp["Week"], comp["TrueValue"], marker="o", label="Hidden true value")
        ax.plot(comp["Week"], comp["ImputedValue"], marker="s", label="Imputed value")
        ax.set_title("Hidden truth vs imputation")
        ax.set_xlabel("Week")
        ax.set_ylabel("Snow cover")
        ax.legend(fontsize=8)
        self.overlay_plot.fig.tight_layout()
        self.overlay_plot.draw()

    def _plot_error(self, result: ImputeResult) -> None:
        self.error_plot.fig.clf()
        ax = self.error_plot.fig.add_subplot(111)
        comp = result.comparison.dropna(subset=["Error"]).sort_values("Week")
        ax.axhline(0, linestyle="--", linewidth=1)
        ax.bar(comp["Week"].astype(int), comp["Error"])
        ax.set_title("Error by week")
        ax.set_xlabel("Week")
        ax.set_ylabel("Imputed - true")
        self.error_plot.fig.tight_layout()
        self.error_plot.draw()

    def _plot_quality(self, result: ImputeResult) -> None:
        self.quality_plot.fig.clf()
        ax = self.quality_plot.fig.add_subplot(111)
        comp = result.comparison.dropna(subset=["TrueValue", "ImputedValue"])
        ax.scatter(comp["TrueValue"], comp["ImputedValue"], alpha=0.75)
        if not comp.empty:
            lo = min(comp["TrueValue"].min(), comp["ImputedValue"].min())
            hi = max(comp["TrueValue"].max(), comp["ImputedValue"].max())
            ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
        ax.set_title("Prediction quality")
        ax.set_xlabel("True value")
        ax.set_ylabel("Imputed value")
        self.quality_plot.fig.tight_layout()
        self.quality_plot.draw()


class RealMissingPage(QWidget):
    def __init__(self, main: "CobberEcoImputeApp"):
        super().__init__(main)
        self.main = main
        layout = QVBoxLayout(self)
        controls = QGroupBox("Impute real missing entries")
        form = QFormLayout(controls)
        self.region_combo = QComboBox(); self.region_combo.addItems(REGIONS)
        self.year_combo = QComboBox()
        self.method_combo = QComboBox()
        self.method_combo.addItems(["Overall mean", "Week-of-year average", "Neighboring-year average", "Linear interpolation", "KNN similar years"])
        self.window_spin = QSpinBox(); self.window_spin.setRange(1, 30); self.window_spin.setValue(5)
        self.k_spin = QSpinBox(); self.k_spin.setRange(1, 20); self.k_spin.setValue(5)
        self.run_btn = QPushButton("Impute Real Missing Values")
        form.addRow("Region:", self.region_combo)
        form.addRow("Year:", self.year_combo)
        form.addRow("Method:", self.method_combo)
        form.addRow("Neighbor-year window:", self.window_spin)
        form.addRow("K for KNN:", self.k_spin)
        form.addRow(self.run_btn)
        layout.addWidget(controls)
        self.warning = QLabel(
            "Real missing values have no hidden answer key. You can fill them and compare methods, but the app cannot tell you which answer is correct."
        )
        self.warning.setWordWrap(True)
        layout.addWidget(self.warning)
        self.plot = PlotCanvas(width=9, height=4)
        layout.addWidget(self.plot, stretch=1)
        self.table = QTableView()
        layout.addWidget(QLabel("Real-missing imputation table"))
        layout.addWidget(self.table, stretch=1)
        self.region_combo.currentTextChanged.connect(self.refresh_years)
        self.year_combo.currentTextChanged.connect(self.refresh)
        self.method_combo.currentTextChanged.connect(self.update_param_states)
        self.run_btn.clicked.connect(self.run_real_impute)

        self.refresh_years()

    def update_param_states(self) -> None:
        method = self.method_combo.currentText()
        self.window_spin.setEnabled(method == "Neighboring-year average")
        self.k_spin.setEnabled(method == "KNN similar years")

    def refresh_years(self) -> None:
        if self.main.working_df is None:
            return
        region = self.region_combo.currentText() or REGIONS[0]
        df = self.main.working_df
        max_year = int(df["Year"].max())
        years = sorted(df[(df["Region"] == region) & (df["OriginalMissing"]) & (df["Year"] < max_year)]["Year"].unique())
        current = self.year_combo.currentText()
        self.year_combo.blockSignals(True)
        self.year_combo.clear()
        self.year_combo.addItems([str(y) for y in years])
        if current and self.year_combo.findText(current) >= 0:
            self.year_combo.setCurrentText(current)
        self.year_combo.blockSignals(False)
        self.refresh()

    def run_real_impute(self) -> None:
        if self.main.working_df is None or not self.year_combo.currentText():
            QMessageBox.warning(self, "No real missing year", "No real missing entries are available for this selection.")
            return
        region = self.region_combo.currentText()
        year = int(self.year_combo.currentText())
        target_mask = (
            (self.main.working_df["Region"] == region)
            & (self.main.working_df["Year"] == year)
            & (self.main.working_df["OriginalMissing"])
        )
        result = SnowImputer.impute(
            self.main.working_df,
            target_mask,
            region=region,
            method=self.method_combo.currentText(),
            neighbor_window=self.window_spin.value(),
            k=self.k_spin.value(),
        )
        self.main.working_df = result.df
        self.main.real_result = result
        self.main.refresh_all_pages(skip=self)
        self.refresh()
        self.main.statusBar().showMessage(f"Imputed real missing entries for {region}, {year} using {result.method}.")

    def refresh(self) -> None:
        df = self.main.working_df
        if df is None or not self.year_combo.currentText():
            self.plot.clear_with_message("No real missing entries found, or no year selected.")
            self.table.setModel(DataFrameModel(pd.DataFrame()))
            return
        region = self.region_combo.currentText() or REGIONS[0]
        year = int(self.year_combo.currentText())
        sub = df[(df["Region"] == region) & (df["Year"] == year)].copy()
        missing = sub[sub["OriginalMissing"]]
        self.table.setModel(DataFrameModel(missing[["Year", "Week", "Region", "WorkingValue", "OriginalMissing", "Imputed", "Method"]].reset_index(drop=True)))
        self.plot.fig.clf()
        ax = self.plot.fig.add_subplot(111)
        observed = sub[sub["SnowCover"].notna()]
        ax.plot(observed["Week"], observed["SnowCover"], marker="o", label="Observed")
        if not missing.empty:
            ax.scatter(missing["Week"], [0] * len(missing), marker="x", s=50, label="Real missing")
        imputed = sub[sub["OriginalMissing"] & sub["Imputed"]]
        if not imputed.empty:
            ax.scatter(imputed["Week"], imputed["WorkingValue"], marker="s", s=55, label="Imputed")
        ax.set_title(f"Real missing data: {region}, {year}")
        ax.set_xlabel("Week")
        ax.set_ylabel("Snow cover")
        ax.set_xlim(1, 52)
        ax.legend(fontsize=8)
        self.plot.fig.tight_layout()
        self.plot.draw()


class ExportPage(QWidget):
    def __init__(self, main: "CobberEcoImputeApp"):
        super().__init__(main)
        self.main = main
        layout = QVBoxLayout(self)
        self.save_working_btn = QPushButton("Save working/imputed data CSV")
        self.save_truth_btn = QPushButton("Save hidden truth CSV for test gap")
        self.save_report_btn = QPushButton("Save imputation report CSV")
        layout.addWidget(self.save_working_btn)
        layout.addWidget(self.save_truth_btn)
        layout.addWidget(self.save_report_btn)
        layout.addStretch()
        self.save_working_btn.clicked.connect(self.save_working)
        self.save_truth_btn.clicked.connect(self.save_truth)
        self.save_report_btn.clicked.connect(self.save_report)

    def save_working(self) -> None:
        if self.main.working_df is None:
            return
        fname, _ = QFileDialog.getSaveFileName(self, "Save working data", "snowcover_working_imputed.csv", "CSV files (*.csv)")
        if fname:
            self.main.working_df.to_csv(fname, index=False)
            self.main.statusBar().showMessage(f"Saved {fname}")

    def save_truth(self) -> None:
        if self.main.truth_df is None or self.main.truth_df.empty:
            QMessageBox.warning(self, "No hidden truth", "No artificial test gap has been created yet.")
            return
        fname, _ = QFileDialog.getSaveFileName(self, "Save hidden truth", "snowcover_hidden_truth.csv", "CSV files (*.csv)")
        if fname:
            self.main.truth_df.to_csv(fname, index=False)
            self.main.statusBar().showMessage(f"Saved {fname}")

    def save_report(self) -> None:
        rows = []
        for label, result in [("test_gap", self.main.last_result), ("real_gap", self.main.real_result)]:
            if result is None:
                continue
            row = {"mode": label, "method": result.method}
            row.update(result.metrics)
            rows.append(row)
        if not rows:
            QMessageBox.warning(self, "No report", "Run an imputation first.")
            return
        report = pd.DataFrame(rows)
        fname, _ = QFileDialog.getSaveFileName(self, "Save report", "snowcover_imputation_report.csv", "CSV files (*.csv)")
        if fname:
            report.to_csv(fname, index=False)
            self.main.statusBar().showMessage(f"Saved {fname}")


# ---------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------


class CobberEcoImputeApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1420, 770)
        self.setFont(QFont("Lato", 10))
        self.cobber_maroon = QColor(108, 29, 69)
        self.cobber_gold = QColor(234, 170, 0)

        self.source_path: Optional[Path] = None
        self.original_wide: Optional[pd.DataFrame] = None
        self.working_df: Optional[pd.DataFrame] = None
        self.truth_df: Optional[pd.DataFrame] = None
        self.last_result: Optional[ImputeResult] = None
        self.real_result: Optional[ImputeResult] = None
        self.test_gap_info: Optional[Dict] = None

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        self.setStatusBar(QStatusBar())
        self._build_menu()
        self.load_default_data()
        self._build_pages()
        self.statusBar().showMessage(f"Ready. Loaded {self.source_path}" if self.source_path else "Ready.")

    def _build_menu(self) -> None:
        menu = self.menuBar()
        file_menu = menu.addMenu("File")
        open_action = QAction("Open Rutgers snow file...", self)
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)
        reload_action = QAction("Reload original data", self)
        reload_action.triggered.connect(self.reload_original)
        file_menu.addAction(reload_action)
        file_menu.addSeparator()
        quit_action = QAction("Quit", self)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

    def load_default_data(self) -> None:
        path = find_snow_file()
        if path is None:
            self.statusBar().showMessage("No default snow-cover file found. Use File > Open.")
            return
        self.load_data(path)

    def load_data(self, path: Path) -> None:
        wide = read_snow_file(path)
        self.source_path = path
        self.original_wide = wide
        self.working_df = wide_to_long_with_grid(wide)
        self.truth_df = pd.DataFrame()
        self.last_result = None
        self.real_result = None
        self.test_gap_info = None

    def open_file(self) -> None:
        fname, _ = QFileDialog.getOpenFileName(self, "Open Rutgers snow file", str(app_dir()), "Text/CSV files (*.txt *.csv);;All files (*.*)")
        if not fname:
            return
        try:
            self.load_data(Path(fname))
            self._build_pages()
            self.statusBar().showMessage(f"Loaded {fname}")
        except Exception as exc:
            QMessageBox.critical(self, "Load error", f"Could not load file:\n{exc}")

    def reload_original(self) -> None:
        if self.source_path is None:
            return
        self.load_data(self.source_path)
        self._build_pages()
        self.statusBar().showMessage("Reloaded original data.")

    def _build_pages(self) -> None:
        self.tabs.clear()
        self.inspect_page = InspectPage(self)
        self.test_page = TestGapPage(self)
        self.impute_page = ImputeEvaluatePage(self)
        self.real_page = RealMissingPage(self)
        self.export_page = ExportPage(self)
        self.tabs.addTab(self.inspect_page, "Inspect Data")
        self.tabs.addTab(self.test_page, "Create Test Gap")
        self.tabs.addTab(self.impute_page, "Impute / Evaluate")
        self.tabs.addTab(self.real_page, "Real Missing Data")
        self.tabs.addTab(self.export_page, "Save / Export")

    def refresh_all_pages(self, skip: Optional[QWidget] = None) -> None:
        for page in [getattr(self, "inspect_page", None), getattr(self, "test_page", None), getattr(self, "impute_page", None), getattr(self, "real_page", None)]:
            if page is not None and page is not skip and hasattr(page, "refresh"):
                page.refresh()

    def clear_test_gap(self) -> None:
        if self.working_df is None:
            return
        mask = self.working_df["TestMissing"]
        self.working_df.loc[mask, "WorkingValue"] = self.working_df.loc[mask, "SnowCover"]
        self.working_df.loc[mask, "TestMissing"] = False
        self.working_df.loc[mask, "Imputed"] = False
        self.working_df.loc[mask, "ImputedValue"] = np.nan
        self.working_df.loc[mask, "Method"] = ""
        self.truth_df = pd.DataFrame()
        self.last_result = None
        self.test_gap_info = None


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
        QComboBox, QSpinBox {
            background-color: #ffffff;
            color: #111111;
            border: 1px solid #9a9a9a;
            border-radius: 3px;
            padding: 3px 6px;
            min-height: 24px;
            selection-background-color: #6c1d45;
            selection-color: #ffffff;
        }
        QComboBox:disabled, QSpinBox:disabled { background-color: #eeeeee; color: #555555; }
        QComboBox QAbstractItemView {
            background-color: #ffffff;
            color: #111111;
            selection-background-color: #6c1d45;
            selection-color: #ffffff;
            border: 1px solid #9a9a9a;
        }
        QTableView, QTextEdit {
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
            padding: 6px 10px;
        }
        QPushButton:hover { background-color: #eeeeee; }
        QPushButton:pressed { background-color: #dddddd; }
        QStatusBar { background-color: #f7f7f7; color: #222222; }
    """)


def main() -> None:
    app = QApplication(sys.argv)
    apply_app_stylesheet(app)
    win = CobberEcoImputeApp()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
