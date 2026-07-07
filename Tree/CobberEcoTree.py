#!/usr/bin/env python3
"""
CobberEcoTree.py

An urban-ecology adaptation of CobberTree for exploring decision trees.

This version mirrors the original CobberTree workflow:

    Manual Sorter:
        Students manually drag urban site cards into bins and watch the
        weighted variance in summer surface temperature change.

    Automated Tree:
        A DecisionTreeRegressor learns rules for predicting surface temperature
        from interpretable urban land-cover and vegetation features.

Expected CSV files in the same directory as this script:
    urban_heat_manual_subset.csv
    urban_heat_tree_dataset_for_tree.csv

Dependencies:
    pip install PyQt6 pandas numpy matplotlib scikit-learn

Run:
    python CobberEcoTree.py
"""

from __future__ import annotations

import sys
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from sklearn.tree import DecisionTreeRegressor, plot_tree

from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QFont, QColor, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QAbstractItemView,
    QComboBox,
    QFrame,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


APP_TITLE = "CobberEcoTree"

MANUAL_FILE = "urban_heat_manual_subset.csv"
TREE_FILE = "urban_heat_tree_dataset_for_tree.csv"
TARGET = "Surface_Temp_C"


def app_root() -> Path:
    """Return a sensible root directory for script or PyInstaller use."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def find_data_file(filename: str) -> Optional[Path]:
    root = app_root()
    candidates = [
        root / filename,
        Path.cwd() / filename,
        root / "data" / filename,
        root / "EcoData" / filename,
        root / "assets" / filename,
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def clean_float(value, default: float = np.nan) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


@dataclass
class UrbanSite:
    site_id: str
    site_name: str
    land_use_class: str
    surface_temp_c: float
    tree_canopy_pct: float
    impervious_pct: float
    grass_shrub_pct: float
    water_pct: float
    ndvi: float
    distance_to_water_m: float
    distance_to_park_m: float
    building_density: float
    road_density: float
    notes: str = ""

    @classmethod
    def from_row(cls, row: pd.Series) -> "UrbanSite":
        return cls(
            site_id=str(row.get("Site_ID", "")).strip(),
            site_name=str(row.get("Site_Name", "")).strip(),
            land_use_class=str(row.get("Land_Use_Class", "")).strip(),
            surface_temp_c=clean_float(row.get("Surface_Temp_C")),
            tree_canopy_pct=clean_float(row.get("Tree_Canopy_Pct")),
            impervious_pct=clean_float(row.get("Impervious_Pct")),
            grass_shrub_pct=clean_float(row.get("Grass_Shrub_Pct")),
            water_pct=clean_float(row.get("Water_Pct")),
            ndvi=clean_float(row.get("NDVI")),
            distance_to_water_m=clean_float(row.get("Distance_To_Water_m")),
            distance_to_park_m=clean_float(row.get("Distance_To_Park_m")),
            building_density=clean_float(row.get("Building_Density")),
            road_density=clean_float(row.get("Road_Density")),
            notes=str(row.get("Notes", "")).strip(),
        )

    @property
    def display_name(self) -> str:
        return self.site_id if self.site_id else self.site_name[:18]

    def card_text(self) -> str:
        return (
            f"{self.display_name:<5s} | Temp: {self.surface_temp_c:>4.1f} C | "
            f"Canopy: {self.tree_canopy_pct:>4.0f}% | Imp.: {self.impervious_pct:>4.0f}% | "
            f"{self.land_use_class}"
        )

    def detail_text(self) -> str:
        return (
            f"{self.display_name}: {self.site_name}\n\n"
            f"Land-use context: {self.land_use_class}\n"
            f"Summer surface temperature: {self.surface_temp_c:.1f} C\n\n"
            f"Tree canopy: {self.tree_canopy_pct:.1f}%\n"
            f"Impervious surface: {self.impervious_pct:.1f}%\n"
            f"Grass/shrub cover: {self.grass_shrub_pct:.1f}%\n"
            f"Water cover nearby: {self.water_pct:.1f}%\n"
            f"NDVI: {self.ndvi:.3f}\n"
            f"Distance to water: {self.distance_to_water_m:.0f} m\n"
            f"Distance to park: {self.distance_to_park_m:.0f} m\n"
            f"Building density index: {self.building_density:.1f}\n"
            f"Road density index: {self.road_density:.1f}\n\n"
            f"Notes: {self.notes}\n\n"
            "Interpretation reminder:\n"
            "The decision tree is not proving that one variable caused this site to be hot. "
            "It is learning threshold rules that separate cooler and hotter urban sites in this dataset."
        )


def calculate_weighted_variance(group1: List[UrbanSite], group2: List[UrbanSite]) -> float:
    n1, n2 = len(group1), len(group2)
    n_total = n1 + n2
    if n_total == 0:
        return 0.0
    var1 = np.var([p.surface_temp_c for p in group1]) if n1 > 1 else 0.0
    var2 = np.var([p.surface_temp_c for p in group2]) if n2 > 1 else 0.0
    return float((n1 / n_total) * var1 + (n2 / n_total) * var2)


class SiteListWidget(QListWidget):
    def __init__(self, name: str, parent=None):
        super().__init__(parent)
        self.name = name
        self.setDragDropMode(QAbstractItemView.DragDropMode.DragDrop)
        self.setAcceptDrops(True)
        self.setIconSize(QSize(80, 50))
        self.setAlternatingRowColors(True)

    def dropEvent(self, event):
        source_widget = event.source()
        if not isinstance(source_widget, SiteListWidget):
            event.ignore()
            return

        source_name = source_widget.name
        target_name = self.name

        # Same basic allowed-move structure as the original CobberTree:
        # deck -> level 1; level 1 -> deck, siblings, or level 2;
        # level 2 -> parent or sibling.
        allowed_moves = {
            "deck": ["bin1", "bin2"],
            "bin1": ["deck", "bin1_1", "bin1_2", "bin2"],
            "bin2": ["deck", "bin2_1", "bin2_2", "bin1"],
            "bin1_1": ["bin1", "bin1_2"],
            "bin1_2": ["bin1", "bin1_1"],
            "bin2_1": ["bin2", "bin2_2"],
            "bin2_2": ["bin2", "bin2_1"],
        }

        if target_name in allowed_moves.get(source_name, []):
            current = source_widget.currentItem()
            if current is None:
                event.ignore()
                return
            item = source_widget.takeItem(source_widget.row(current))
            self.addItem(item)
            event.accept()
        else:
            event.ignore()


class AutomatedTreeWidget(QWidget):
    """Interactive automated decision-tree tab.

    The underlying model is still the same DecisionTreeRegressor used in the
    first urban-canopy version. The presentation is different: the tree is drawn
    on a matplotlib canvas so students can click a node and read a larger,
    more narrative version of that node's text.
    """

    FRIENDLY_FEATURE_NAMES = {
        "Tree_Canopy_Pct": "Tree canopy (%)",
        "Impervious_Pct": "Impervious surface (%)",
        "Grass_Shrub_Pct": "Grass/shrub cover (%)",
        "Water_Pct": "Water nearby (%)",
        "NDVI": "NDVI greenness",
        "Distance_To_Water_m": "Distance to water (m)",
        "Distance_To_Park_m": "Distance to park (m)",
        "Building_Density": "Building density index",
        "Road_Density": "Road density index",
    }

    def __init__(self):
        super().__init__()
        self.root = app_root()
        self.tree_image_path = str(self.root / "urban_heat_decision_tree.png")
        self.model: Optional[DecisionTreeRegressor] = None
        self.df: Optional[pd.DataFrame] = None
        self.features: List[str] = []
        self.target = TARGET
        self.node_artists = []

        main_layout = QHBoxLayout(self)

        controls_layout = QVBoxLayout()
        display_layout = QVBoxLayout()

        form_layout = QFormLayout()
        self.depth_spinner = QSpinBox()
        self.depth_spinner.setRange(1, 10)
        self.depth_spinner.setValue(3)

        self.target_combo = QComboBox()
        self.target_combo.addItem("Surface temperature (C)")

        self.generate_button = QPushButton("Generate Decision Tree")
        self.predict_button = QPushButton("Predict S01-S11 Sites")
        self.predict_button.setEnabled(False)

        form_layout.addRow("Prediction target:", self.target_combo)
        form_layout.addRow("Max Tree Depth:", self.depth_spinner)

        self.info_text = QLabel(
            "Generate a tree, then click any node to open a larger explanation."
        )
        self.info_text.setWordWrap(True)
        self.info_text.setStyleSheet(
            "QLabel { color: #333333; background-color: #fffaf0; "
            "border: 1px solid #6C1D45; border-radius: 4px; padding: 6px; }"
        )

        controls_layout.addLayout(form_layout)
        controls_layout.addWidget(self.generate_button)
        controls_layout.addWidget(self.predict_button)
        controls_layout.addWidget(self.info_text)

        self.results_text = QTextEdit(readOnly=True)
        self.results_text.setMinimumHeight(360)
        self.results_text.setStyleSheet(
            "QTextEdit { font-size: 12px; background-color: #ffffff; "
            "border: 1px solid #999999; padding: 4px; }"
        )
        controls_layout.addWidget(QLabel("<b>Analysis & Predictions:</b>"))
        controls_layout.addWidget(self.results_text, 1)

        self.figure = Figure(figsize=(16, 8), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumSize(850, 520)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.canvas.mpl_connect("button_press_event", self.on_tree_click)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.canvas)

        display_layout.addWidget(self.toolbar)
        display_layout.addWidget(scroll, 1)

        main_layout.addLayout(controls_layout, 1)
        main_layout.addLayout(display_layout, 4)

        self.generate_button.clicked.connect(self.generate_tree)
        self.predict_button.clicked.connect(self.predict_demo_sites)

    def friendly_feature(self, technical_name: str) -> str:
        return self.FRIENDLY_FEATURE_NAMES.get(technical_name, technical_name.replace("_", " "))

    def load_tree_df(self) -> Optional[pd.DataFrame]:
        path = find_data_file(TREE_FILE)
        if path is None:
            QMessageBox.critical(
                self,
                "Data file not found",
                f"Could not find {TREE_FILE} in the script folder or current directory.",
            )
            return None

        try:
            df = pd.read_csv(path)
        except Exception as exc:
            QMessageBox.critical(self, "Could not read data", str(exc))
            return None

        if TARGET not in df.columns:
            QMessageBox.critical(self, "Missing target", f"The dataset must contain a {TARGET} column.")
            return None

        df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
        df = df.dropna(subset=[TARGET]).copy()

        if len(df) < 5:
            QMessageBox.critical(self, "Too little data", f"Fewer than 5 records have usable {TARGET} values.")
            return None

        return df

    def select_features(self, df: pd.DataFrame) -> List[str]:
        candidates = [
            "Tree_Canopy_Pct",
            "Impervious_Pct",
            "Grass_Shrub_Pct",
            "Water_Pct",
            "NDVI",
            "Distance_To_Water_m",
            "Distance_To_Park_m",
            "Building_Density",
            "Road_Density",
        ]

        usable: List[str] = []
        for col in candidates:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                if df[col].notna().sum() >= 5 and df[col].nunique(dropna=True) > 1:
                    usable.append(col)

        return usable

    def generate_tree(self):
        df = self.load_tree_df()
        if df is None:
            return

        features = self.select_features(df)
        if not features:
            QMessageBox.critical(self, "No usable features", "No numeric feature columns were found.")
            return

        model_df = df.dropna(subset=features + [TARGET]).copy()
        if len(model_df) < 5:
            QMessageBox.critical(self, "Too little complete data", "Too few records are complete for the selected features.")
            return

        X = model_df[features]
        y = model_df[TARGET]

        max_depth = self.depth_spinner.value()
        self.model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
        self.model.fit(X, y)
        self.df = model_df
        self.features = features

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        friendly_names = [self.friendly_feature(f) for f in features]

        artists = plot_tree(
            self.model,
            feature_names=friendly_names,
            filled=False,
            rounded=True,
            fontsize=10,
            impurity=False,
            ax=ax,
        )

        ax.set_title(f"Decision Tree for Urban Surface Temperature (Max Depth = {max_depth})", fontsize=17)

        self.node_artists = []
        for artist in artists:
            raw = artist.get_text()
            cleaned = self.clean_node_label(raw)
            artist.set_text(cleaned)

            if raw.strip() in {"True", "False"}:
                # Keep branch labels the same size as node labels, but make them visually clear.
                artist.set_fontsize(10)
                artist.set_fontweight("bold")
                artist.set_color("#6C1D45")
            else:
                self.node_artists.append(artist)

        self.figure.tight_layout()
        self.canvas.draw()

        # Save a copy for students who want the image in their portfolio.
        self.figure.savefig(self.tree_image_path, dpi=150, bbox_inches="tight")

        importances = self.model.feature_importances_
        feature_importance_text = "\n".join(
            [f"  - {self.friendly_feature(name)} ({name}): {imp:.2%}" for name, imp in zip(features, importances)]
        )

        results = (
            f"--- Model Training Results ---\n"
            f"Dataset size: {len(model_df)} urban site records\n"
            f"Target: summer surface temperature (C)\n"
            f"Tree Depth: {max_depth}\n\n"
            f"Features used:\n"
            + "\n".join([f"  - {self.friendly_feature(f)} ({f})" for f in features])
            + f"\n\nFeature Importances:\n{feature_importance_text}\n\n"
            "Interpretation reminder:\n"
            "A split is useful when it creates groups with more similar surface temperatures. "
            "The tree is not proving that canopy or pavement alone caused a site to be hot; "
            "it is learning threshold rules in this dataset.\n\n"
            "Click any node in the tree to open a larger explanation."
        )
        self.results_text.setText(results)
        self.predict_button.setEnabled(True)

    def clean_node_label(self, raw: str) -> str:
        """Turn sklearn's compact labels into language closer to the book narrative."""
        text = raw

        # Keep the branch labels exactly so we can identify them separately.
        if text.strip() in {"True", "False"}:
            return text.strip()

        text = text.replace("samples = ", "sites = ")

        # Regression tree values sometimes appear as "value = 34.2" and sometimes "value = [34.2]".
        text = re.sub(
            r"value = \[?([0-9]+(?:\.[0-9]+)?)\]?",
            r"predicted temp = \1 C",
            text,
        )

        # Make thresholds slightly more readable.
        text = text.replace("<=", "≤")

        return text

    def explain_node_label(self, label: str) -> str:
        """Create an enlarged, student-friendly explanation of a clicked node."""
        lines = [line.strip() for line in label.splitlines() if line.strip()]
        if not lines:
            return "This node has no text to display."

        first = lines[0]
        sites_line = next((line for line in lines if line.startswith("sites =")), "")
        pred_line = next((line for line in lines if line.startswith("predicted temp =")), "")

        parts = []

        if "≤" in first:
            feature, threshold = [piece.strip() for piece in first.split("≤", 1)]
            parts.append(f"Decision question:\nIs {feature} less than or equal to {threshold}?")
            parts.append("If the answer is True, follow the True branch. If the answer is False, follow the False branch.")
        else:
            parts.append("Leaf node:\nThis is the end of one decision path. The tree makes its prediction here.")

        if sites_line:
            parts.append(sites_line.replace("sites =", "Number of training sites in this node:"))
        if pred_line:
            parts.append(pred_line.replace("predicted temp =", "Predicted summer surface temperature:"))

        parts.append(
            "Ecology reminder:\nThis node describes a pattern in the training data. "
            "It is a threshold rule, not proof that one feature alone caused the temperature."
        )

        return "\n\n".join(parts)

    def on_tree_click(self, event):
        if event.inaxes is None or not self.node_artists:
            return

        renderer = self.canvas.get_renderer()
        for artist in self.node_artists:
            # Make the click target generous so students do not need to click perfectly.
            bbox = artist.get_window_extent(renderer=renderer).expanded(1.22, 1.35)
            if bbox.contains(event.x, event.y):
                label = artist.get_text()
                self.show_node_popup(label)
                return

    def show_node_popup(self, label: str):
        dialog = QDialog(self)
        dialog.setWindowTitle("Decision Tree Node")
        dialog.resize(560, 420)

        layout = QVBoxLayout(dialog)

        title = QLabel("<h2 style='color:#6C1D45;'>Decision Tree Node</h2>")
        title.setWordWrap(True)
        layout.addWidget(title)

        body = QTextEdit(readOnly=True)
        body.setPlainText(self.explain_node_label(label))
        body.setStyleSheet(
            "QTextEdit { font-size: 15px; line-height: 130%; background-color: #fffaf0; "
            "border: 2px solid #6C1D45; padding: 10px; }"
        )
        layout.addWidget(body, 1)

        close_button = QPushButton("Close")
        close_button.setFixedWidth(110)
        close_button.clicked.connect(dialog.accept)

        button_row = QHBoxLayout()
        button_row.addStretch()
        button_row.addWidget(close_button)
        layout.addLayout(button_row)

        dialog.exec()

    def predict_demo_sites(self):
        if self.model is None or self.df is None or not self.features:
            QMessageBox.warning(self, "No Model", "Please generate a tree before making predictions.")
            return

        df = self.df.copy()

        # Show predictions for the manual-sorter sites S01-S11. This links the
        # automated tree directly back to the same cards students handled by hand.
        preferred = [f"S{i:02d}" for i in range(1, 12)]
        rows = []
        for site_id in preferred:
            hits = df[df.get("Site_ID", pd.Series(dtype=str)).astype(str) == site_id]
            if not hits.empty:
                rows.append(hits.iloc[0])

        if not rows:
            QMessageBox.warning(
                self,
                "Manual sites not found",
                "Could not find S01-S11 in the tree dataset."
            )
            return

        prediction_text = "\n\n--- Predictions for Manual Sorter Sites (S01-S11) ---\n"
        prediction_text += (
            "These are the same sites used in the Manual Sorter tab. "
            "The tree predicts each site's surface temperature, then compares it "
            "with the actual value in the dataset.\n\n"
        )

        total_abs_error = 0.0
        for row in rows:
            X_row = pd.DataFrame([row[self.features].astype(float).values], columns=self.features)
            prediction = float(self.model.predict(X_row)[0])
            actual = float(row[TARGET])
            error = prediction - actual
            total_abs_error += abs(error)

            name = str(row.get("Site_ID", "Urban site"))
            full = str(row.get("Site_Name", ""))
            prediction_text += (
                f"{name}  {full}\n"
                f"  Predicted temperature: {prediction:.1f} C\n"
                f"  Actual temperature:    {actual:.1f} C\n"
                f"  Error:                 {error:+.1f} C\n\n"
            )

        mean_abs_error = total_abs_error / len(rows)
        prediction_text += f"Mean absolute error for S01-S11: {mean_abs_error:.2f} C\n"

        self.results_text.append(prediction_text)


class CobberEcoTreeApp(QMainWindow):
    def __init__(self, manual_dataset: List[UrbanSite]):
        super().__init__()

        self.cobber_maroon = QColor(108, 29, 69)
        self.cobber_gold = QColor(234, 170, 0)
        self.lato_font = QFont("Lato")

        self.setWindowTitle("CobberEcoTree")
        self.resize(1200, 800)
        self.setFont(self.lato_font)

        tabs = QTabWidget()
        self.setCentralWidget(tabs)

        manual_sorter_widget = self.create_manual_sorter_tab(manual_dataset)
        tabs.addTab(manual_sorter_widget, "Manual Sorter")

        automated_tree_widget = AutomatedTreeWidget()
        tabs.addTab(automated_tree_widget, "Automated Tree")

    def create_manual_sorter_tab(self, dataset: List[UrbanSite]) -> QWidget:
        container = QWidget()
        self.manual_dataset = dataset
        self.site_map = {p.display_name: p for p in self.manual_dataset}

        main_layout = QVBoxLayout(container)

        top_layout = QHBoxLayout()
        title = QLabel(
            "<b>Manual sorter:</b> Drag urban site cards into bins. "
            "Try to create groups with similar summer surface temperatures."
        )
        title.setWordWrap(True)

        self.reset_button = QPushButton("Reset")
        self.reset_button.setFixedWidth(90)
        self.reset_button.setStyleSheet("background-color: #e0e0e0; font-weight: bold;")
        self.reset_button.clicked.connect(self.reset_manual_sorting)

        top_layout.addWidget(title)
        top_layout.addStretch()
        top_layout.addWidget(self.reset_button)
        main_layout.addLayout(top_layout)

        self.deck_list_widget = SiteListWidget("deck")
        self.bin1_list_widget = SiteListWidget("bin1")
        self.bin2_list_widget = SiteListWidget("bin2")
        self.bin1_1_list_widget = SiteListWidget("bin1_1")
        self.bin1_2_list_widget = SiteListWidget("bin1_2")
        self.bin2_1_list_widget = SiteListWidget("bin2_1")
        self.bin2_2_list_widget = SiteListWidget("bin2_2")

        level0_layout = QHBoxLayout()
        deck_frame = self.create_bin_section("Unsorted Deck", self.deck_list_widget)
        level0_layout.addStretch(1)
        level0_layout.addWidget(deck_frame, 2)
        level0_layout.addStretch(1)
        main_layout.addLayout(level0_layout)

        level1_layout = QHBoxLayout()
        bin1_frame = self.create_bin_section("Bin 1", self.bin1_list_widget)
        bin2_frame = self.create_bin_section("Bin 2", self.bin2_list_widget)
        level1_layout.addWidget(bin1_frame)
        level1_layout.addWidget(bin2_frame)
        main_layout.addLayout(level1_layout)

        level2_layout = QHBoxLayout()
        bin1_1_frame = self.create_bin_section("Bin 1,1", self.bin1_1_list_widget)
        bin1_2_frame = self.create_bin_section("Bin 1,2", self.bin1_2_list_widget)
        bin2_1_frame = self.create_bin_section("Bin 2,1", self.bin2_1_list_widget)
        bin2_2_frame = self.create_bin_section("Bin 2,2", self.bin2_2_list_widget)
        level2_layout.addWidget(bin1_1_frame)
        level2_layout.addWidget(bin1_2_frame)
        level2_layout.addWidget(bin2_1_frame)
        level2_layout.addWidget(bin2_2_frame)
        main_layout.addLayout(level2_layout, stretch=1)

        bottom_layout = QHBoxLayout()

        scoreboard_layout = QHBoxLayout()
        self.score_labels: Dict[str, QLabel] = {}
        score_frame_1 = self._create_score_frame("Split 1: Deck -> Lvl 1", "split1", "Bin 1", "Bin 2")
        score_frame_2 = self._create_score_frame("Split 2: Bin 1 -> Lvl 2", "split2", "Bin 1,1", "Bin 1,2")
        score_frame_3 = self._create_score_frame("Split 3: Bin 2 -> Lvl 2", "split3", "Bin 2,1", "Bin 2,2")
        scoreboard_layout.addWidget(score_frame_1)
        scoreboard_layout.addWidget(score_frame_2)
        scoreboard_layout.addWidget(score_frame_3)

        score_container = QWidget()
        score_container.setLayout(scoreboard_layout)
        bottom_layout.addWidget(score_container, stretch=3)

        self.detail_box = QTextEdit(readOnly=True)
        self.detail_box.setMaximumHeight(165)
        self.detail_box.setText(
            "Click an urban site card to see details.\n\n"
            "Surface temperature is the target value. The decision-tree cost is the weighted variance "
            "of surface temperature in the two child bins."
        )
        bottom_layout.addWidget(self.detail_box, stretch=1)

        main_layout.addLayout(bottom_layout)

        self.populate_deck()
        self._connect_manual_signals()
        self.update_manual_calculations()
        return container

    def reset_manual_sorting(self):
        for list_widget in [
            self.deck_list_widget,
            self.bin1_list_widget,
            self.bin2_list_widget,
            self.bin1_1_list_widget,
            self.bin1_2_list_widget,
            self.bin2_1_list_widget,
            self.bin2_2_list_widget,
        ]:
            list_widget.clear()
        self.populate_deck()
        self.update_manual_calculations()
        self.detail_box.setText("Click an urban site card to see details.")

    def _connect_manual_signals(self):
        for list_widget in [
            self.deck_list_widget,
            self.bin1_list_widget,
            self.bin2_list_widget,
            self.bin1_1_list_widget,
            self.bin1_2_list_widget,
            self.bin2_1_list_widget,
            self.bin2_2_list_widget,
        ]:
            list_widget.model().rowsInserted.connect(self.update_manual_calculations)
            list_widget.model().rowsRemoved.connect(self.update_manual_calculations)
            list_widget.itemClicked.connect(self.show_site_details)

    def show_site_details(self, item: QListWidgetItem):
        key = item.data(Qt.ItemDataRole.UserRole)
        site = self.site_map.get(key)
        if site:
            self.detail_box.setText(site.detail_text())

    def _get_sites_from_list(self, list_widget: SiteListWidget) -> List[UrbanSite]:
        out = []
        for i in range(list_widget.count()):
            item = list_widget.item(i)
            key = item.data(Qt.ItemDataRole.UserRole)
            if key in self.site_map:
                out.append(self.site_map[key])
        return out

    def update_manual_calculations(self):
        def _update_bin_stats(site_list: List[UrbanSite], key_prefix: str):
            n = len(site_list)
            self.score_labels[f"{key_prefix}_count"].setText(f"Count (n): {n}")
            if n > 0:
                mean = float(np.mean([p.surface_temp_c for p in site_list]))
                var = float(np.var([p.surface_temp_c for p in site_list])) if n > 1 else 0.0
                self.score_labels[f"{key_prefix}_mean"].setText(f"Mean Temp: {mean:.1f} C")
                self.score_labels[f"{key_prefix}_var"].setText(f"Variance: {var:.2f}")
            else:
                self.score_labels[f"{key_prefix}_mean"].setText("Mean Temp: N/A")
                self.score_labels[f"{key_prefix}_var"].setText("Variance: N/A")

        sites_bin1 = self._get_sites_from_list(self.bin1_list_widget)
        sites_bin2 = self._get_sites_from_list(self.bin2_list_widget)
        sites_bin1_1 = self._get_sites_from_list(self.bin1_1_list_widget)
        sites_bin1_2 = self._get_sites_from_list(self.bin1_2_list_widget)
        sites_bin2_1 = self._get_sites_from_list(self.bin2_1_list_widget)
        sites_bin2_2 = self._get_sites_from_list(self.bin2_2_list_widget)

        _update_bin_stats(sites_bin1, "split1_binA")
        _update_bin_stats(sites_bin2, "split1_binB")
        cost1 = calculate_weighted_variance(sites_bin1, sites_bin2)
        self.score_labels["split1_cost"].setText(f"<b>TOTAL COST: {cost1:.2f}</b>")

        _update_bin_stats(sites_bin1_1, "split2_binA")
        _update_bin_stats(sites_bin1_2, "split2_binB")
        cost2 = calculate_weighted_variance(sites_bin1_1, sites_bin1_2)
        self.score_labels["split2_cost"].setText(f"<b>TOTAL COST: {cost2:.2f}</b>")

        _update_bin_stats(sites_bin2_1, "split3_binA")
        _update_bin_stats(sites_bin2_2, "split3_binB")
        cost3 = calculate_weighted_variance(sites_bin2_1, sites_bin2_2)
        self.score_labels["split3_cost"].setText(f"<b>TOTAL COST: {cost3:.2f}</b>")

    def populate_deck(self):
        for site in self.manual_dataset:
            item = QListWidgetItem(site.card_text())
            item.setData(Qt.ItemDataRole.UserRole, site.display_name)
            font = QFont("Lato")
            font.setPointSize(9)
            item.setFont(font)
            self.deck_list_widget.addItem(item)

    def create_bin_section(self, title: str, list_widget: SiteListWidget) -> QFrame:
        layout = QVBoxLayout()

        label = QLabel(f"<h3>{title}</h3>")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)
        layout.addWidget(list_widget)

        frame = QFrame()
        frame.setFrameShape(QFrame.Shape.StyledPanel)
        frame.setLayout(layout)
        frame.setStyleSheet(
            """
            QFrame {
                border: 3px solid #6C1D45;
                border-radius: 6px;
                background-color: #FAFAFA;
            }
            QLabel {
                color: #6C1D45;
                border: none;
            }
            QListWidget {
                border: 1px solid #333333;
                border-radius: 4px;
                background-color: white;
            }
            """
        )
        return frame

    def _create_score_frame(self, title: str, key_prefix: str, binA_name: str, binB_name: str) -> QFrame:
        frame = QFrame()
        frame.setFrameShape(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout(frame)

        title_label = QLabel(f"<b>{title}</b>")
        layout.addWidget(title_label)

        bins_layout = QHBoxLayout()

        binA_layout = QVBoxLayout()
        binA_label = QLabel(f"<i>{binA_name}</i>")
        self.score_labels[f"{key_prefix}_binA_count"] = QLabel("Count (n): 0")
        self.score_labels[f"{key_prefix}_binA_mean"] = QLabel("Mean Temp: N/A")
        self.score_labels[f"{key_prefix}_binA_var"] = QLabel("Variance: N/A")
        binA_layout.addWidget(binA_label)
        binA_layout.addWidget(self.score_labels[f"{key_prefix}_binA_count"])
        binA_layout.addWidget(self.score_labels[f"{key_prefix}_binA_mean"])
        binA_layout.addWidget(self.score_labels[f"{key_prefix}_binA_var"])

        binB_layout = QVBoxLayout()
        binB_label = QLabel(f"<i>{binB_name}</i>")
        self.score_labels[f"{key_prefix}_binB_count"] = QLabel("Count (n): 0")
        self.score_labels[f"{key_prefix}_binB_mean"] = QLabel("Mean Temp: N/A")
        self.score_labels[f"{key_prefix}_binB_var"] = QLabel("Variance: N/A")
        binB_layout.addWidget(binB_label)
        binB_layout.addWidget(self.score_labels[f"{key_prefix}_binB_count"])
        binB_layout.addWidget(self.score_labels[f"{key_prefix}_binB_mean"])
        binB_layout.addWidget(self.score_labels[f"{key_prefix}_binB_var"])

        bins_layout.addLayout(binA_layout)
        bins_layout.addLayout(binB_layout)
        layout.addLayout(bins_layout)

        self.score_labels[f"{key_prefix}_cost"] = QLabel("<b>TOTAL COST: N/A</b>")
        layout.addWidget(self.score_labels[f"{key_prefix}_cost"])

        return frame


def load_manual_dataset() -> List[UrbanSite]:
    manual_path = find_data_file(MANUAL_FILE)
    tree_path = find_data_file(TREE_FILE)

    if manual_path is not None:
        df = pd.read_csv(manual_path)
    elif tree_path is not None:
        df = pd.read_csv(tree_path).head(11)
    else:
        raise FileNotFoundError(
            f"Could not find {MANUAL_FILE} or {TREE_FILE} in the script folder/current directory."
        )

    if TARGET not in df.columns:
        raise ValueError(f"Manual dataset must contain a {TARGET} column.")

    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET]).copy()

    if len(df) == 0:
        raise ValueError(f"Manual dataset has no usable {TARGET} values.")

    # Sort cards by temperature only for a stable deck order. Students can still
    # use canopy, pavement, vegetation, or any custom feature for sorting.
    df = df.sort_values([TARGET, "Site_ID"], na_position="last").reset_index(drop=True)

    return [UrbanSite.from_row(row) for _, row in df.iterrows()]


def apply_app_stylesheet(app: QApplication) -> None:
    app.setStyleSheet(
        """
        QWidget { color: #222222; background-color: #ffffff; }
        QMainWindow, QDialog { background-color: #ffffff; }
        QTabWidget::pane { border: 1px solid #cccccc; }
        QTabBar::tab { padding: 6px 12px; }
        QTabBar::tab:selected { background: #6c1d45; color: #ffffff; }
        QGroupBox {
            color: #222222;
            font-weight: bold;
            border: 1px solid #d6d6d6;
            border-radius: 5px;
            margin-top: 8px;
            padding-top: 10px;
            background-color: #fafafa;
        }
        QLabel { color: #222222; background-color: transparent; }
        QComboBox, QSpinBox, QTextEdit {
            background-color: #ffffff;
            color: #111111;
            selection-background-color: #6c1d45;
            selection-color: #ffffff;
        }
        QListWidget {
            background-color: #ffffff;
            color: #111111;
            alternate-background-color: #f4f4f4;
            selection-background-color: #e8d7df;
            selection-color: #111111;
        }
        QListWidget::item {
            color: #111111;
            background-color: #ffffff;
            padding: 2px;
        }
        QListWidget::item:alternate {
            background-color: #f4f4f4;
        }
        QListWidget::item:selected {
            background-color: #e8d7df;
            color: #111111;
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


def main() -> int:
    app = QApplication(sys.argv)
    apply_app_stylesheet(app)

    try:
        manual_dataset = load_manual_dataset()
    except Exception as exc:
        QMessageBox.critical(None, "Could not load urban heat dataset", str(exc))
        return 1

    window = CobberEcoTreeApp(manual_dataset)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
