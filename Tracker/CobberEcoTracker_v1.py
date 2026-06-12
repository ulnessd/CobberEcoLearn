# CobberEcoTracker_v1.py
# A PyQt6 exploratory movement-ecology app for Mike's large fish dataset.
#
# Goal:
#   Help students inspect individual fish movement biographies, examine
#   canal-distance transitions, engineer fish-level movement fingerprints,
#   and discover candidate movement types by clustering.
#
# Dependencies:
#   pip install PyQt6 pandas numpy matplotlib scikit-learn openpyxl
#
# Run:
#   python CobberEcoTracker_v1.py
#
# Expected data file:
#   Put LargeFish_MockDataset.xlsx in the same directory as this script,
#   or use the "Load Large Fish Dataset" button.

from __future__ import annotations

import sys
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QColor
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QPushButton,
    QFileDialog,
    QMessageBox,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QComboBox,
    QSpinBox,
    QGroupBox,
    QTextEdit,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QSplitter,
    QCheckBox,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


def app_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


APP_ROOT = app_root()


# ---------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------
class MplCanvas(FigureCanvas):
    def __init__(self, width: float = 7.2, height: float = 5.0, dpi: int = 100):
        self.figure = Figure(figsize=(width, height), dpi=dpi, tight_layout=True)
        super().__init__(self.figure)


def category_colors(categories: List[str]) -> Dict[str, str]:
    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
        "#bcbd22", "#17becf",
    ]
    uniq = list(dict.fromkeys([str(c) for c in categories]))
    return {cat: palette[i % len(palette)] for i, cat in enumerate(uniq)}


def format_float(x, digits=2, missing="—"):
    try:
        if pd.isna(x):
            return missing
        return f"{float(x):.{digits}f}"
    except Exception:
        return missing


# ---------------------------------------------------------------------
# Data processing
# ---------------------------------------------------------------------
@dataclass
class FishData:
    raw: pd.DataFrame
    clean: pd.DataFrame
    transitions: pd.DataFrame
    fingerprints: pd.DataFrame


class FishDataProcessor:
    REQUIRED_COLUMNS = [
        "Fish",
        "Habitat",
        "TimeBetweenRelocations(Days)",
        "DistanceMoved",
        "DistanceMoved(m per day)",
        "DistanceFromCanal",
        "Before/After",
        "Control/Impact",
    ]

    NUMERIC_COLUMNS = [
        "TimeBetweenRelocations(Days)",
        "DistanceMoved",
        "DistanceMoved(m per day)",
        "DistanceFromCanal",
        "ActualDepthOverPrev30Days",
        "DepthChangeOverPrev30Days",
        "Photoperiod(minutes)",
        "MaximumTempOverPrev30Days©",
    ]

    @classmethod
    def load_excel(cls, path: Path) -> FishData:
        df = pd.read_excel(path)
        return cls.process(df)

    @classmethod
    def process(cls, df: pd.DataFrame) -> FishData:
        raw = df.copy()

        missing = [c for c in cls.REQUIRED_COLUMNS if c not in raw.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        clean = raw.copy()
        clean["_row_order"] = np.arange(len(clean))

        # Normalize key categorical columns.
        clean["Fish"] = clean["Fish"].astype(int)
        clean["Habitat"] = clean["Habitat"].astype(str).str.strip().str.lower()
        clean["Before/After"] = clean["Before/After"].astype(str).str.strip()
        clean["Control/Impact"] = clean["Control/Impact"].astype(str).str.strip()

        # Convert numeric-looking fields to numeric. Some cells contain "." or text.
        for col in cls.NUMERIC_COLUMNS:
            if col in clean.columns:
                clean[col] = pd.to_numeric(clean[col], errors="coerce")

        # Friendly aliases for plotting and logic.
        clean["move_rate_m_per_day"] = clean["DistanceMoved(m per day)"]
        clean["distance_from_canal"] = clean["DistanceFromCanal"]
        clean["time_between_days"] = clean["TimeBetweenRelocations(Days)"]
        clean["distance_moved_m"] = clean["DistanceMoved"]
        clean["relocation_index"] = clean.groupby("Fish").cumcount()

        transitions = cls._build_transitions(clean)
        fingerprints = cls._build_fingerprints(clean, transitions)

        return FishData(raw=raw, clean=clean, transitions=transitions, fingerprints=fingerprints)

    @classmethod
    def _build_transitions(cls, clean: pd.DataFrame) -> pd.DataFrame:
        records = []

        for fish_id, group in clean.sort_values(["Fish", "_row_order"]).groupby("Fish"):
            g = group.reset_index(drop=True)
            if len(g) < 2:
                continue

            for i in range(len(g) - 1):
                current = g.iloc[i]
                nxt = g.iloc[i + 1]

                d0 = current.get("distance_from_canal", np.nan)
                d1 = nxt.get("distance_from_canal", np.nan)
                delta_d = d1 - d0 if pd.notna(d0) and pd.notna(d1) else np.nan

                records.append({
                    "Fish": fish_id,
                    "transition_index": i,
                    "from_row": int(current["_row_order"]),
                    "to_row": int(nxt["_row_order"]),
                    "from_habitat": current["Habitat"],
                    "to_habitat": nxt["Habitat"],
                    "habitat_transition": f"{current['Habitat']}→{nxt['Habitat']}",
                    "from_before_after": current["Before/After"],
                    "to_before_after": nxt["Before/After"],
                    "from_control_impact": current["Control/Impact"],
                    "to_control_impact": nxt["Control/Impact"],
                    "time_between_days": nxt.get("time_between_days", np.nan),
                    "distance_moved_m": nxt.get("distance_moved_m", np.nan),
                    "move_rate_m_per_day": nxt.get("move_rate_m_per_day", np.nan),
                    "distance_from_canal_current": d0,
                    "distance_from_canal_next": d1,
                    "delta_distance_from_canal": delta_d,
                    "abs_delta_distance_from_canal": abs(delta_d) if pd.notna(delta_d) else np.nan,
                    "depth_prev30": nxt.get("ActualDepthOverPrev30Days", np.nan),
                    "depth_change_prev30": nxt.get("DepthChangeOverPrev30Days", np.nan),
                    "photoperiod_minutes": nxt.get("Photoperiod(minutes)", np.nan),
                    "max_temp_prev30": nxt.get("MaximumTempOverPrev30Days©", np.nan),
                })

        return pd.DataFrame.from_records(records)

    @classmethod
    def _build_fingerprints(cls, clean: pd.DataFrame, transitions: pd.DataFrame) -> pd.DataFrame:
        records = []

        for fish_id, group in clean.sort_values(["Fish", "_row_order"]).groupby("Fish"):
            g = group.copy()
            t = transitions[transitions["Fish"] == fish_id].copy() if not transitions.empty else pd.DataFrame()

            n_obs = len(g)
            n_trans = len(t)
            habitats = g["Habitat"].dropna().astype(str)
            majority_habitat = habitats.value_counts().idxmax() if len(habitats) else "unknown"

            percent_canal = 100.0 * (habitats == "canal").mean() if len(habitats) else np.nan
            percent_marsh = 100.0 * (habitats == "marsh").mean() if len(habitats) else np.nan

            habitat_switches = 0
            if len(habitats) >= 2:
                habitat_switches = int((habitats.iloc[1:].values != habitats.iloc[:-1].values).sum())

            ci = g["Control/Impact"].dropna().astype(str)
            percent_into_out = 100.0 * ci.isin(["INTO", "OUT"]).mean() if len(ci) else np.nan

            ba = g["Before/After"].dropna().astype(str)
            percent_after = 100.0 * (ba == "A").mean() if len(ba) else np.nan

            move_rate = g["move_rate_m_per_day"]
            dist_canal = g["distance_from_canal"]
            dist_moved = g["distance_moved_m"]

            rec = {
                "Fish": fish_id,
                "n_observations": n_obs,
                "n_transitions": n_trans,
                "majority_habitat": majority_habitat,
                "percent_canal": percent_canal,
                "percent_marsh": percent_marsh,
                "habitat_switches": habitat_switches,
                "switch_rate": habitat_switches / max(1, n_trans),
                "percent_into_out": percent_into_out,
                "percent_after": percent_after,
                "mean_move_rate": move_rate.mean(skipna=True),
                "median_move_rate": move_rate.median(skipna=True),
                "max_move_rate": move_rate.max(skipna=True),
                "mean_distance_moved": dist_moved.mean(skipna=True),
                "total_distance_moved": dist_moved.sum(skipna=True),
                "mean_distance_from_canal": dist_canal.mean(skipna=True),
                "median_distance_from_canal": dist_canal.median(skipna=True),
                "max_distance_from_canal": dist_canal.max(skipna=True),
                "sd_distance_from_canal": dist_canal.std(skipna=True),
                "mean_abs_delta_canal": t["abs_delta_distance_from_canal"].mean(skipna=True) if not t.empty else np.nan,
                "max_abs_delta_canal": t["abs_delta_distance_from_canal"].max(skipna=True) if not t.empty else np.nan,
                "mean_depth_prev30": g["ActualDepthOverPrev30Days"].mean(skipna=True) if "ActualDepthOverPrev30Days" in g else np.nan,
                "mean_temp_prev30": g["MaximumTempOverPrev30Days©"].mean(skipna=True) if "MaximumTempOverPrev30Days©" in g else np.nan,
            }
            records.append(rec)

        fp = pd.DataFrame.from_records(records).sort_values("Fish").reset_index(drop=True)
        return fp


# ---------------------------------------------------------------------
# Fish Explorer tab
# ---------------------------------------------------------------------
class FishExplorerTab(QWidget):
    def __init__(self):
        super().__init__()
        self.data: Optional[FishData] = None

        layout = QHBoxLayout(self)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left.setFixedWidth(330)

        selector_box = QGroupBox("Fish Biography")
        form = QFormLayout(selector_box)

        self.fish_combo = QComboBox()
        self.color_combo = QComboBox()
        self.color_combo.addItems(["Habitat", "Before/After", "Control/Impact"])

        form.addRow("Fish", self.fish_combo)
        form.addRow("Color by", self.color_combo)

        self.summary = QTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setMinimumHeight(340)

        left_layout.addWidget(selector_box)
        left_layout.addWidget(QLabel("<b>Movement summary</b>"))
        left_layout.addWidget(self.summary)
        left_layout.addStretch(1)

        self.canvas = MplCanvas(width=8.2, height=5.8)

        layout.addWidget(left, stretch=0)
        layout.addWidget(self.canvas, stretch=1)

        self.fish_combo.currentIndexChanged.connect(self.update_view)
        self.color_combo.currentIndexChanged.connect(self.update_view)

    def set_data(self, data: FishData):
        self.data = data
        self.fish_combo.blockSignals(True)
        self.fish_combo.clear()

        for fish_id in sorted(data.clean["Fish"].unique()):
            n = len(data.clean[data.clean["Fish"] == fish_id])
            self.fish_combo.addItem(f"Fish {fish_id} ({n} relocations)", int(fish_id))

        self.fish_combo.blockSignals(False)
        self.update_view()

    def update_view(self):
        if self.data is None or self.fish_combo.count() == 0:
            self._draw_placeholder()
            return

        fish_id = self.fish_combo.currentData()
        if fish_id is None:
            return

        df = self.data.clean[self.data.clean["Fish"] == int(fish_id)].sort_values("_row_order").copy()
        fp = self.data.fingerprints[self.data.fingerprints["Fish"] == int(fish_id)].iloc[0]

        self._update_summary(fish_id, df, fp)
        self._draw_fish(fish_id, df)

    def _update_summary(self, fish_id, df, fp):
        first_hab = df["Habitat"].iloc[0]
        last_hab = df["Habitat"].iloc[-1]
        first_ba = df["Before/After"].iloc[0]
        last_ba = df["Before/After"].iloc[-1]

        lines = [
            f"Fish {fish_id}",
            "=" * 30,
            f"Observations: {int(fp['n_observations'])}",
            f"Transitions: {int(fp['n_transitions'])}",
            "",
            f"First habitat: {first_hab}",
            f"Last habitat: {last_hab}",
            f"Majority habitat: {fp['majority_habitat']}",
            f"Percent canal: {format_float(fp['percent_canal'], 1)}%",
            f"Percent marsh: {format_float(fp['percent_marsh'], 1)}%",
            f"Habitat switches: {int(fp['habitat_switches'])}",
            f"INTO/OUT observations: {format_float(fp['percent_into_out'], 1)}%",
            "",
            f"Before/After start: {first_ba}",
            f"Before/After end: {last_ba}",
            f"Percent after: {format_float(fp['percent_after'], 1)}%",
            "",
            f"Median movement rate: {format_float(fp['median_move_rate'])} m/day",
            f"Mean movement rate: {format_float(fp['mean_move_rate'])} m/day",
            f"Max movement rate: {format_float(fp['max_move_rate'])} m/day",
            f"Total distance moved: {format_float(fp['total_distance_moved'], 1)} m",
            "",
            f"Mean distance from canal: {format_float(fp['mean_distance_from_canal'], 1)} m",
            f"Median distance from canal: {format_float(fp['median_distance_from_canal'], 1)} m",
            f"Max distance from canal: {format_float(fp['max_distance_from_canal'], 1)} m",
            f"Mean |Δ canal distance|: {format_float(fp['mean_abs_delta_canal'], 1)} m",
        ]

        self.summary.setPlainText("\n".join(lines))

    def _draw_fish(self, fish_id, df):
        fig = self.canvas.figure
        fig.clear()
        ax1 = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)

        color_col = self.color_combo.currentText()
        categories = df[color_col].astype(str).tolist()
        cmap = category_colors(categories)

        x = df["relocation_index"].to_numpy()
        dist = df["distance_from_canal"].to_numpy(dtype=float)
        move = df["move_rate_m_per_day"].to_numpy(dtype=float)

        for cat in cmap:
            mask = np.array([c == cat for c in categories])
            ax1.scatter(x[mask], dist[mask], s=42, color=cmap[cat], label=cat, alpha=0.88)
            ax2.scatter(x[mask], move[mask], s=42, color=cmap[cat], label=cat, alpha=0.88)

        ax1.plot(x, dist, color="0.55", linewidth=1.0, alpha=0.65)
        ax2.plot(x, move, color="0.55", linewidth=1.0, alpha=0.65)

        ax1.set_title(f"Fish {fish_id}: distance from canal")
        ax1.set_ylabel("Distance from canal (m)")
        ax1.grid(True, alpha=0.25)
        ax1.legend(loc="best", fontsize=8)

        ax2.set_title("Movement rate between relocations")
        ax2.set_xlabel("Relocation sequence")
        ax2.set_ylabel("Movement rate (m/day)")
        ax2.grid(True, alpha=0.25)

        fig.tight_layout()
        self.canvas.draw()

    def _draw_placeholder(self):
        fig = self.canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.55, "Fish Explorer", ha="center", va="center", fontsize=16, transform=ax.transAxes)
        ax.text(0.5, 0.42, "Load the large fish dataset to inspect movement biographies.", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        self.canvas.draw()


# ---------------------------------------------------------------------
# Transition View tab
# ---------------------------------------------------------------------
class TransitionViewTab(QWidget):
    def __init__(self):
        super().__init__()
        self.data: Optional[FishData] = None

        layout = QHBoxLayout(self)

        left = QWidget()
        left.setFixedWidth(330)
        left_layout = QVBoxLayout(left)

        control_box = QGroupBox("Canal-Distance Transitions")
        form = QFormLayout(control_box)

        self.scope_combo = QComboBox()
        self.scope_combo.addItems(["All fish", "Single fish"])

        self.fish_combo = QComboBox()

        self.color_combo = QComboBox()
        self.color_combo.addItems(["from_habitat", "habitat_transition", "from_before_after", "from_control_impact"])

        self.normalize_checkbox = QCheckBox("Show Δ distance per day")
        self.normalize_checkbox.setChecked(False)

        form.addRow("Scope", self.scope_combo)
        form.addRow("Fish", self.fish_combo)
        form.addRow("Color by", self.color_combo)
        form.addRow("", self.normalize_checkbox)

        self.explainer = QTextEdit()
        self.explainer.setReadOnly(True)
        self.explainer.setMinimumHeight(260)
        self.explainer.setPlainText(
            "This tab adapts the displacement-kernel idea from the prototype.\n\n"
            "We do not have full x-y coordinates in this dataset, but we do have "
            "DistanceFromCanal. For each fish, we compare the current distance "
            "from canal to the next relocation's distance from canal.\n\n"
            "Positive Δ means movement away from the canal. Negative Δ means "
            "movement toward the canal."
        )

        left_layout.addWidget(control_box)
        left_layout.addWidget(QLabel("<b>Interpretation</b>"))
        left_layout.addWidget(self.explainer)
        left_layout.addStretch(1)

        self.canvas = MplCanvas(width=8.2, height=5.8)

        layout.addWidget(left, stretch=0)
        layout.addWidget(self.canvas, stretch=1)

        self.scope_combo.currentIndexChanged.connect(self._refresh_controls)
        self.scope_combo.currentIndexChanged.connect(self.update_view)
        self.fish_combo.currentIndexChanged.connect(self.update_view)
        self.color_combo.currentIndexChanged.connect(self.update_view)
        self.normalize_checkbox.stateChanged.connect(self.update_view)
        self._refresh_controls()

    def set_data(self, data: FishData):
        self.data = data
        self.fish_combo.blockSignals(True)
        self.fish_combo.clear()
        for fish_id in sorted(data.clean["Fish"].unique()):
            self.fish_combo.addItem(f"Fish {fish_id}", int(fish_id))
        self.fish_combo.blockSignals(False)
        self.update_view()

    def _refresh_controls(self):
        self.fish_combo.setEnabled(self.scope_combo.currentText() == "Single fish")

    def _get_transitions(self):
        if self.data is None:
            return pd.DataFrame()

        t = self.data.transitions.copy()
        if self.scope_combo.currentText() == "Single fish":
            fish_id = self.fish_combo.currentData()
            if fish_id is not None:
                t = t[t["Fish"] == int(fish_id)]

        t = t.dropna(subset=["distance_from_canal_current", "distance_from_canal_next", "delta_distance_from_canal"])
        return t

    def update_view(self):
        if self.data is None:
            self._draw_placeholder()
            return

        t = self._get_transitions()
        if t.empty:
            self._draw_empty()
            return

        fig = self.canvas.figure
        fig.clear()
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)

        color_col = self.color_combo.currentText()
        categories = t[color_col].astype(str).tolist()
        cmap = category_colors(categories)

        x = t["distance_from_canal_current"].to_numpy(dtype=float)
        y = t["distance_from_canal_next"].to_numpy(dtype=float)

        for cat in cmap:
            mask = np.array([c == cat for c in categories])
            ax1.scatter(x[mask], y[mask], s=30, color=cmap[cat], label=cat, alpha=0.72)

        max_val = np.nanmax([np.nanmax(x), np.nanmax(y)])
        ax1.plot([0, max_val], [0, max_val], linestyle="--", color="0.4", linewidth=1)
        ax1.set_xlabel("Current distance from canal (m)")
        ax1.set_ylabel("Next distance from canal (m)")
        ax1.set_title("Canal-distance transition map")
        ax1.grid(True, alpha=0.25)
        ax1.legend(loc="best", fontsize=8)

        delta = t["delta_distance_from_canal"].to_numpy(dtype=float)
        xlabel = "Δ distance from canal (m)"
        if self.normalize_checkbox.isChecked():
            days = t["time_between_days"].to_numpy(dtype=float)
            delta = delta / days
            xlabel = "Δ distance from canal per day (m/day)"

        ax2.hist(delta[np.isfinite(delta)], bins=28, alpha=0.85)
        ax2.axvline(0, color="black", linestyle="--", linewidth=1)
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel("Count")
        ax2.set_title("Toward canal ← 0 → away from canal")
        ax2.grid(True, alpha=0.25)

        n = len(t)
        mean_delta = np.nanmean(delta)
        med_delta = np.nanmedian(delta)
        ax2.text(
            0.03,
            0.97,
            f"n={n}\nmean={mean_delta:.2f}\nmedian={med_delta:.2f}",
            ha="left",
            va="top",
            transform=ax2.transAxes,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        )

        fig.tight_layout()
        self.canvas.draw()

    def _draw_placeholder(self):
        fig = self.canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.55, "Transition View", ha="center", va="center", fontsize=16, transform=ax.transAxes)
        ax.text(0.5, 0.42, "Load the large fish dataset to inspect canal-distance transitions.", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        self.canvas.draw()

    def _draw_empty(self):
        fig = self.canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "No valid transitions for this selection.", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        self.canvas.draw()


# ---------------------------------------------------------------------
# Fish Fingerprints tab
# ---------------------------------------------------------------------
class FishFingerprintsTab(QWidget):
    SCATTER_FEATURES = [
        "median_move_rate",
        "mean_move_rate",
        "max_move_rate",
        "total_distance_moved",
        "mean_distance_from_canal",
        "median_distance_from_canal",
        "max_distance_from_canal",
        "sd_distance_from_canal",
        "mean_abs_delta_canal",
        "max_abs_delta_canal",
        "percent_canal",
        "habitat_switches",
        "switch_rate",
        "percent_into_out",
        "percent_after",
    ]

    def __init__(self):
        super().__init__()
        self.data: Optional[FishData] = None

        layout = QVBoxLayout(self)

        controls = QHBoxLayout()
        self.x_combo = QComboBox()
        self.y_combo = QComboBox()
        self.color_combo = QComboBox()
        self.color_combo.addItems(["majority_habitat", "percent_after", "percent_into_out"])

        controls.addWidget(QLabel("x-axis:"))
        controls.addWidget(self.x_combo)
        controls.addWidget(QLabel("y-axis:"))
        controls.addWidget(self.y_combo)
        controls.addWidget(QLabel("color by:"))
        controls.addWidget(self.color_combo)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        self.canvas = MplCanvas(width=7.6, height=5.6)

        self.table = QTableWidget()
        self.table.setMinimumWidth(480)

        splitter.addWidget(self.canvas)
        splitter.addWidget(self.table)
        splitter.setSizes([650, 450])

        layout.addLayout(controls)
        layout.addWidget(splitter)

        self.x_combo.currentIndexChanged.connect(self.update_view)
        self.y_combo.currentIndexChanged.connect(self.update_view)
        self.color_combo.currentIndexChanged.connect(self.update_view)

    def set_data(self, data: FishData):
        self.data = data
        self.x_combo.blockSignals(True)
        self.y_combo.blockSignals(True)
        self.x_combo.clear()
        self.y_combo.clear()
        self.x_combo.addItems(self.SCATTER_FEATURES)
        self.y_combo.addItems(self.SCATTER_FEATURES)
        self.x_combo.setCurrentText("mean_distance_from_canal")
        self.y_combo.setCurrentText("median_move_rate")
        self.x_combo.blockSignals(False)
        self.y_combo.blockSignals(False)

        self._populate_table()
        self.update_view()

    def _populate_table(self):
        if self.data is None:
            return

        fp = self.data.fingerprints.copy()
        cols = [
            "Fish", "n_observations", "majority_habitat",
            "percent_canal", "median_move_rate", "max_move_rate",
            "mean_distance_from_canal", "max_distance_from_canal",
            "habitat_switches", "mean_abs_delta_canal",
        ]

        self.table.clear()
        self.table.setRowCount(len(fp))
        self.table.setColumnCount(len(cols))
        self.table.setHorizontalHeaderLabels(cols)

        for r, (_, row) in enumerate(fp[cols].iterrows()):
            for c, col in enumerate(cols):
                val = row[col]
                if isinstance(val, (float, np.floating)):
                    text = format_float(val, 2)
                else:
                    text = str(val)
                self.table.setItem(r, c, QTableWidgetItem(text))

        self.table.resizeColumnsToContents()

    def update_view(self):
        if self.data is None:
            self._draw_placeholder()
            return

        fp = self.data.fingerprints.copy()
        xcol = self.x_combo.currentText()
        ycol = self.y_combo.currentText()
        ccol = self.color_combo.currentText()

        fig = self.canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)

        x = fp[xcol].to_numpy(dtype=float)
        y = fp[ycol].to_numpy(dtype=float)

        if ccol == "majority_habitat":
            cats = fp[ccol].astype(str).tolist()
            cmap = category_colors(cats)
            for cat in cmap:
                mask = np.array([c == cat for c in cats])
                ax.scatter(x[mask], y[mask], s=80, color=cmap[cat], label=cat, alpha=0.85)
            ax.legend(loc="best")
        else:
            color_vals = fp[ccol].to_numpy(dtype=float)
            sc = ax.scatter(x, y, c=color_vals, s=80, alpha=0.85)
            cbar = fig.colorbar(sc, ax=ax)
            cbar.set_label(ccol)

        for _, row in fp.iterrows():
            if pd.notna(row[xcol]) and pd.notna(row[ycol]):
                ax.text(row[xcol], row[ycol], str(int(row["Fish"])), fontsize=8, ha="center", va="center")

        ax.set_xlabel(xcol)
        ax.set_ylabel(ycol)
        ax.set_title("Fish movement fingerprints")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        self.canvas.draw()

    def _draw_placeholder(self):
        fig = self.canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.55, "Fish Fingerprints", ha="center", va="center", fontsize=16, transform=ax.transAxes)
        ax.text(0.5, 0.42, "Load the large fish dataset to compute fish-level movement features.", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        self.canvas.draw()


# ---------------------------------------------------------------------
# Movement Types tab
# ---------------------------------------------------------------------
class MovementTypesTab(QWidget):
    CLUSTER_FEATURES = [
        "median_move_rate",
        "max_move_rate",
        "mean_distance_from_canal",
        "max_distance_from_canal",
        "sd_distance_from_canal",
        "mean_abs_delta_canal",
        "percent_canal",
        "switch_rate",
        "percent_into_out",
    ]

    def __init__(self):
        super().__init__()
        self.data: Optional[FishData] = None
        self.clustered: Optional[pd.DataFrame] = None

        layout = QHBoxLayout(self)

        left = QWidget()
        left.setFixedWidth(360)
        left_layout = QVBoxLayout(left)

        control_box = QGroupBox("Candidate Movement Types")
        form = QFormLayout(control_box)

        self.k_spin = QSpinBox()
        self.k_spin.setRange(2, 8)
        self.k_spin.setValue(4)

        self.run_button = QPushButton("Run Clustering")
        self.run_button.clicked.connect(self.run_clustering)

        form.addRow("Number of clusters", self.k_spin)
        form.addRow("", self.run_button)

        self.summary = QTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setMinimumHeight(440)

        left_layout.addWidget(control_box)
        left_layout.addWidget(QLabel("<b>Cluster interpretation</b>"))
        left_layout.addWidget(self.summary)
        left_layout.addStretch(1)

        self.canvas = MplCanvas(width=8.0, height=5.8)

        layout.addWidget(left, stretch=0)
        layout.addWidget(self.canvas, stretch=1)

    def set_data(self, data: FishData):
        self.data = data
        self.summary.setPlainText(
            "Click Run Clustering to group fish into candidate movement types.\n\n"
            "These clusters are exploratory. They suggest patterns, but they do not prove biological categories."
        )
        self.run_clustering()

    def run_clustering(self):
        if self.data is None:
            self._draw_placeholder()
            return

        if not SKLEARN_AVAILABLE:
            QMessageBox.warning(
                self,
                "Missing scikit-learn",
                "scikit-learn is required for clustering. Install with: pip install scikit-learn",
            )
            return

        fp = self.data.fingerprints.copy()
        features = [f for f in self.CLUSTER_FEATURES if f in fp.columns]

        X = fp[features].copy()

        # Log-transform highly skewed movement rates/distances.
        for col in ["median_move_rate", "max_move_rate", "mean_abs_delta_canal", "max_distance_from_canal"]:
            if col in X.columns:
                X[col] = np.log1p(X[col])

        # Fill missing values with feature medians.
        X = X.apply(pd.to_numeric, errors="coerce")
        X = X.fillna(X.median(numeric_only=True))

        n_fish = len(fp)
        k = min(self.k_spin.value(), max(2, n_fish - 1))

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        km = KMeans(n_clusters=k, random_state=17, n_init=20)
        clusters = km.fit_predict(X_scaled)

        pca = PCA(n_components=2, random_state=17)
        coords = pca.fit_transform(X_scaled)

        clustered = fp.copy()
        clustered["cluster"] = clusters
        clustered["PC1"] = coords[:, 0]
        clustered["PC2"] = coords[:, 1]
        self.clustered = clustered

        self._draw_clusters(clustered, pca)
        self._write_cluster_summary(clustered, features)

    def _draw_clusters(self, clustered: pd.DataFrame, pca: PCA):
        fig = self.canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)

        clusters = sorted(clustered["cluster"].unique())
        colors = category_colors([str(c) for c in clusters])

        for c in clusters:
            sub = clustered[clustered["cluster"] == c]
            ax.scatter(sub["PC1"], sub["PC2"], s=100, alpha=0.85, label=f"Cluster {c}", color=colors[str(c)])
            for _, row in sub.iterrows():
                ax.text(row["PC1"], row["PC2"], str(int(row["Fish"])), fontsize=8, ha="center", va="center")

        ax.axhline(0, color="0.75", linewidth=0.8)
        ax.axvline(0, color="0.75", linewidth=0.8)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var.)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var.)")
        ax.set_title("Candidate movement types from fish fingerprints")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        self.canvas.draw()

    def _write_cluster_summary(self, clustered: pd.DataFrame, features: List[str]):
        lines = []
        lines.append("Candidate movement types")
        lines.append("=" * 34)
        lines.append("These groups are created from fish-level movement fingerprints.")
        lines.append("Treat them as hypotheses to inspect, not proven biological types.")
        lines.append("")

        for c in sorted(clustered["cluster"].unique()):
            sub = clustered[clustered["cluster"] == c]
            fish_ids = ", ".join(str(int(x)) for x in sub["Fish"].tolist())
            lines.append(f"Cluster {c}: fish {fish_ids}")
            lines.append("-" * 34)
            lines.append(f"n fish: {len(sub)}")
            lines.append(f"majority habitats: {', '.join(sub['majority_habitat'].astype(str).tolist())}")
            lines.append(f"median movement rate: {sub['median_move_rate'].median():.2f} m/day")
            lines.append(f"max movement rate median: {sub['max_move_rate'].median():.2f} m/day")
            lines.append(f"mean distance from canal: {sub['mean_distance_from_canal'].mean():.1f} m")
            lines.append(f"max distance from canal: {sub['max_distance_from_canal'].mean():.1f} m")
            lines.append(f"percent canal: {sub['percent_canal'].mean():.1f}%")
            lines.append(f"habitat switches: {sub['habitat_switches'].mean():.1f}")
            lines.append(f"INTO/OUT observations: {sub['percent_into_out'].mean():.1f}%")
            lines.append("")

        lines.append("Suggested interpretation prompts:")
        lines.append("• Which cluster looks most canal-associated?")
        lines.append("• Which cluster looks most marsh-associated?")
        lines.append("• Which cluster contains the strongest roamers?")
        lines.append("• Are INTO/OUT observations concentrated in one group?")
        lines.append("• Do the clusters suggest real movement strategies or just data imbalance?")

        self.summary.setPlainText("\n".join(lines))

    def _draw_placeholder(self):
        fig = self.canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.55, "Movement Types", ha="center", va="center", fontsize=16, transform=ax.transAxes)
        ax.text(0.5, 0.42, "Load the large fish dataset to cluster movement fingerprints.", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        self.canvas.draw()


# ---------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------
class CobberEcoTrackerApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("CobberEcoTracker v1")
        self.setGeometry(100, 100, 1380, 790)
        self.setFont(QFont("Lato"))

        self.data: Optional[FishData] = None
        self.current_file: Optional[Path] = None

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)

        header = QHBoxLayout()
        self.load_button = QPushButton("Load Large Fish Dataset")
        self.load_button.clicked.connect(self.load_file_dialog)

        self.file_label = QLabel("No dataset loaded.")
        self.file_label.setWordWrap(True)

        header.addWidget(self.load_button)
        header.addWidget(self.file_label, stretch=1)

        self.tabs = QTabWidget()
        self.fish_explorer = FishExplorerTab()
        self.transition_view = TransitionViewTab()
        self.fingerprints = FishFingerprintsTab()
        self.movement_types = MovementTypesTab()

        self.tabs.addTab(self.fish_explorer, "Fish Explorer")
        self.tabs.addTab(self.transition_view, "Transition View")
        self.tabs.addTab(self.fingerprints, "Fish Fingerprints")
        self.tabs.addTab(self.movement_types, "Movement Types")

        root_layout.addLayout(header)
        root_layout.addWidget(self.tabs)

        self._try_auto_load()

    def _try_auto_load(self):
        candidates = [
            APP_ROOT / "LargeFish_MockDataset.xlsx",
            APP_ROOT / "LargeFish_MockDataset(1).xlsx",
            APP_ROOT / "LargeFish_MockDataset.xls",
        ]

        for path in candidates:
            if path.exists():
                try:
                    self.load_dataset(path)
                    return
                except Exception as exc:
                    self.file_label.setText(f"Found {path.name}, but could not load it: {exc}")
                    return

        self.file_label.setText(
            "No dataset loaded. Put LargeFish_MockDataset.xlsx beside the program, "
            "or click 'Load Large Fish Dataset'."
        )

    def load_file_dialog(self):
        path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Select Large Fish Dataset",
            str(APP_ROOT),
            "Excel files (*.xlsx *.xls);;All files (*)",
        )
        if path_str:
            self.load_dataset(Path(path_str))

    def load_dataset(self, path: Path):
        try:
            data = FishDataProcessor.load_excel(path)
        except Exception as exc:
            QMessageBox.critical(self, "Load error", f"Could not load dataset:\n{exc}")
            return

        self.data = data
        self.current_file = path

        n_rows = len(data.clean)
        n_fish = data.clean["Fish"].nunique()
        n_trans = len(data.transitions)

        self.file_label.setText(
            f"Loaded: {path.name}  |  rows={n_rows}, fish={n_fish}, transitions={n_trans}"
        )

        self.fish_explorer.set_data(data)
        self.transition_view.set_data(data)
        self.fingerprints.set_data(data)
        self.movement_types.set_data(data)


def apply_app_stylesheet(app: QApplication) -> None:
    app.setStyleSheet(
        """
        QWidget { color: #222222; background-color: #ffffff; }
        QMainWindow, QDialog { background-color: #ffffff; }
        QTabWidget::pane { border: 1px solid #cccccc; }
        QTabBar::tab { padding: 6px 12px; }
        QTabBar::tab:selected { background: #6c1d45; color: #ffffff; }
        QLabel { color: #222222; background-color: transparent; }
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
        QComboBox, QTextEdit, QTableWidget, QSpinBox {
            background-color: #ffffff;
            color: #111111;
            border: 1px solid #a0a0a0;
            border-radius: 3px;
            padding: 3px 6px;
            selection-background-color: #6c1d45;
            selection-color: #ffffff;
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
        """
    )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    apply_app_stylesheet(app)
    window = CobberEcoTrackerApp()
    window.show()
    sys.exit(app.exec())
