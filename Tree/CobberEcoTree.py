#!/usr/bin/env python3
"""
CobberEcoTree.py

An ecology/environmental-science adaptation of CobberTree for exploring
decision trees using PFAS structure/property data.

This version mirrors the chemistry CobberTree workflow:

    Manual Sorter:
        Students manually drag PFAS compounds into bins and watch the
        weighted variance in XLogP change.

    Automated Tree:
        A DecisionTreeRegressor learns rules for predicting XLogP from
        simple structural descriptors.

Expected CSV files in the same directory as this script:
    pfas_manual_subset.csv
    pfas_tree_dataset_for_tree.csv

These are produced by:
    build_pfas_tree_dataset.py

Dependencies:
    pip install PyQt6 pandas numpy matplotlib scikit-learn

Run:
    python CobberEcoTree.py
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor, plot_tree

from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QFont, QColor, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
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

MANUAL_FILE = "pfas_manual_subset.csv"
TREE_FILE = "pfas_tree_dataset_for_tree.csv"


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


def clean_int(value, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default
        return int(round(float(value)))
    except Exception:
        return default


@dataclass
class PFASCompound:
    abbreviation: str
    preferred_name: str
    pfas_class: str
    formula: str
    molecular_weight: float
    xlogp: float
    carbon_count: int
    fluorine_count: int
    oxygen_count: int
    sulfur_count: int
    ether_oxygen_count: int
    carboxylate_flag: int
    sulfonate_flag: int
    ether_flag: int
    pubchem_cid: str = ""

    @classmethod
    def from_row(cls, row: pd.Series) -> "PFASCompound":
        return cls(
            abbreviation=str(row.get("Abbreviation", "")).strip(),
            preferred_name=str(row.get("Preferred_Name", "")).strip(),
            pfas_class=str(row.get("PFAS_Class", "")).strip(),
            formula=str(row.get("MolecularFormula", "")).strip(),
            molecular_weight=clean_float(row.get("MolecularWeight")),
            xlogp=clean_float(row.get("XLogP")),
            carbon_count=clean_int(row.get("Carbon_Count")),
            fluorine_count=clean_int(row.get("Fluorine_Count")),
            oxygen_count=clean_int(row.get("Oxygen_Count")),
            sulfur_count=clean_int(row.get("Sulfur_Count")),
            ether_oxygen_count=clean_int(row.get("Ether_Oxygen_Count")),
            carboxylate_flag=clean_int(row.get("Carboxylate_Flag")),
            sulfonate_flag=clean_int(row.get("Sulfonate_Flag")),
            ether_flag=clean_int(row.get("Ether_Flag")),
            pubchem_cid=str(row.get("PubChem_CID", "")).strip(),
        )

    @property
    def display_name(self) -> str:
        if self.abbreviation and self.abbreviation.lower() != "nan":
            return self.abbreviation
        return self.preferred_name[:18]

    def card_text(self) -> str:
        return (
            f"{self.display_name:<12s} | XLogP: {self.xlogp:>5.2f} | "
            f"C: {self.carbon_count:>2d} | F: {self.fluorine_count:>2d} | {self.pfas_class}"
        )

    def detail_text(self) -> str:
        flags = []
        if self.carboxylate_flag:
            flags.append("carboxylate/carboxylic acid")
        if self.sulfonate_flag:
            flags.append("sulfonate/sulfonic acid")
        if self.ether_flag:
            flags.append("ether-containing")
        if not flags:
            flags.append("no simple flag detected")

        return (
            f"{self.display_name}\n"
            f"{self.preferred_name}\n\n"
            f"PFAS class: {self.pfas_class}\n"
            f"Formula: {self.formula}\n"
            f"PubChem CID: {self.pubchem_cid}\n\n"
            f"XLogP: {self.xlogp:.3f}\n"
            f"Molecular weight: {self.molecular_weight:.3f}\n"
            f"Carbon count: {self.carbon_count}\n"
            f"Fluorine count: {self.fluorine_count}\n"
            f"Oxygen count: {self.oxygen_count}\n"
            f"Sulfur count: {self.sulfur_count}\n"
            f"Estimated ether oxygen count: {self.ether_oxygen_count}\n"
            f"Flags: {', '.join(flags)}\n\n"
            "Interpretation reminder:\n"
            "XLogP is a computed descriptor related to partitioning between "
            "water-like and organic-like environments. It is useful for teaching, "
            "but it is not the entire story of PFAS environmental fate."
        )


def calculate_weighted_variance(group1: List[PFASCompound], group2: List[PFASCompound]) -> float:
    n1, n2 = len(group1), len(group2)
    n_total = n1 + n2
    if n_total == 0:
        return 0.0
    var1 = np.var([p.xlogp for p in group1]) if n1 > 1 else 0.0
    var2 = np.var([p.xlogp for p in group2]) if n2 > 1 else 0.0
    return float((n1 / n_total) * var1 + (n2 / n_total) * var2)


class PFASListWidget(QListWidget):
    def __init__(self, name: str, parent=None):
        super().__init__(parent)
        self.name = name
        self.setDragDropMode(QAbstractItemView.DragDropMode.DragDrop)
        self.setAcceptDrops(True)
        self.setIconSize(QSize(80, 50))
        self.setAlternatingRowColors(True)

    def dropEvent(self, event):
        source_widget = event.source()
        if not isinstance(source_widget, PFASListWidget):
            event.ignore()
            return

        source_name = source_widget.name
        target_name = self.name

        # Same basic allowed-move structure as the chemistry CobberTree:
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
    def __init__(self):
        super().__init__()
        self.root = app_root()
        self.tree_image_path = str(self.root / "pfas_decision_tree.png")
        self.model: Optional[DecisionTreeRegressor] = None
        self.df: Optional[pd.DataFrame] = None
        self.features: List[str] = []
        self.target = "XLogP"

        main_layout = QHBoxLayout(self)

        controls_layout = QVBoxLayout()
        display_layout = QVBoxLayout()

        form_layout = QFormLayout()
        self.depth_spinner = QSpinBox()
        self.depth_spinner.setRange(1, 10)
        self.depth_spinner.setValue(3)

        self.target_combo = QComboBox()
        self.target_combo.addItem("XLogP")

        self.generate_button = QPushButton("Generate Decision Tree")
        self.predict_button = QPushButton("Predict Demo PFAS")
        self.predict_button.setEnabled(False)

        form_layout.addRow("Prediction target:", self.target_combo)
        form_layout.addRow("Max Tree Depth:", self.depth_spinner)

        self.info_text = QTextEdit(readOnly=True)
        self.info_text.setMaximumHeight(210)
        self.info_text.setText(
            "The automated tree uses the larger PFAS dataset.\n\n"
            "The target is XLogP, a computed partitioning descriptor. "
            "The features are simple structural descriptors such as carbon count, "
            "fluorine count, molecular weight, oxygen/sulfur counts, and functional-group flags.\n\n"
            "The decision tree asks threshold questions that reduce the remaining variance in XLogP."
        )

        controls_layout.addLayout(form_layout)
        controls_layout.addWidget(self.generate_button)
        controls_layout.addWidget(self.predict_button)
        controls_layout.addWidget(self.info_text)
        controls_layout.addStretch()

        self.tree_image_label = QLabel("Click 'Generate Decision Tree' to train and display the tree.")
        self.tree_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.tree_image_label.setMinimumSize(700, 430)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.tree_image_label)

        self.results_text = QTextEdit(readOnly=True)

        display_layout.addWidget(scroll, 3)
        display_layout.addWidget(QLabel("<b>Analysis & Predictions:</b>"))
        display_layout.addWidget(self.results_text, 1)

        main_layout.addLayout(controls_layout, 1)
        main_layout.addLayout(display_layout, 4)

        self.generate_button.clicked.connect(self.generate_tree)
        self.predict_button.clicked.connect(self.predict_demo_pfas)

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

        if "XLogP" not in df.columns:
            QMessageBox.critical(self, "Missing target", "The dataset must contain an XLogP column.")
            return None

        df["XLogP"] = pd.to_numeric(df["XLogP"], errors="coerce")
        df = df.dropna(subset=["XLogP"]).copy()

        if len(df) < 5:
            QMessageBox.critical(self, "Too little data", "Fewer than 5 records have usable XLogP values.")
            return None

        return df

    def select_features(self, df: pd.DataFrame) -> List[str]:
        candidates = [
            "Carbon_Count",
            "Fluorine_Count",
            "Estimated_Fluorinated_Carbons",
            "MolecularWeight",
            "Oxygen_Count",
            "Sulfur_Count",
            "Nitrogen_Count",
            "Ether_Oxygen_Count",
            "Carboxylate_Flag",
            "Sulfonate_Flag",
            "Sulfonamide_Flag",
            "Ether_Flag",
            "TPSA",
            "HBondDonorCount",
            "HBondAcceptorCount",
            "RotatableBondCount",
            "HeavyAtomCount",
        ]

        usable: List[str] = []
        for col in candidates:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                if df[col].notna().sum() >= 5 and df[col].nunique(dropna=True) > 1:
                    usable.append(col)

        # Keep a compact, interpretable set if many are available.
        preferred_order = [
            "Carbon_Count",
            "Fluorine_Count",
            "MolecularWeight",
            "Oxygen_Count",
            "Sulfur_Count",
            "Ether_Oxygen_Count",
            "Carboxylate_Flag",
            "Sulfonate_Flag",
            "Ether_Flag",
            "TPSA",
            "HBondAcceptorCount",
            "RotatableBondCount",
        ]
        ordered = [c for c in preferred_order if c in usable]
        extras = [c for c in usable if c not in ordered]
        return (ordered + extras)[:12]

    def generate_tree(self):
        df = self.load_tree_df()
        if df is None:
            return

        features = self.select_features(df)
        if not features:
            QMessageBox.critical(self, "No usable features", "No numeric feature columns were found.")
            return

        model_df = df.dropna(subset=features + ["XLogP"]).copy()
        if len(model_df) < 5:
            QMessageBox.critical(self, "Too little complete data", "Too few records are complete for the selected features.")
            return

        X = model_df[features]
        y = model_df["XLogP"]

        max_depth = self.depth_spinner.value()
        self.model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
        self.model.fit(X, y)
        self.df = model_df
        self.features = features

        plt.figure(figsize=(22, 11))
        plot_tree(
            self.model,
            feature_names=features,
            filled=False,
            rounded=True,
            fontsize=9,
        )
        plt.title(f"Decision Tree for PFAS XLogP (Max Depth = {max_depth})", fontsize=16)
        plt.savefig(self.tree_image_path, dpi=150, bbox_inches="tight")
        plt.close()

        pixmap = QPixmap(self.tree_image_path)
        if pixmap.isNull():
            self.tree_image_label.setText("Tree image could not be loaded.")
        else:
            self.tree_image_label.setPixmap(
                pixmap.scaled(
                    self.tree_image_label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )

        importances = self.model.feature_importances_
        feature_importance_text = "\n".join(
            [f"  - {name}: {imp:.2%}" for name, imp in zip(features, importances)]
        )

        results = (
            f"--- Model Training Results ---\n"
            f"Dataset size: {len(model_df)} PFAS records\n"
            f"Target: XLogP\n"
            f"Tree Depth: {max_depth}\n\n"
            f"Features used:\n"
            + "\n".join([f"  - {f}" for f in features])
            + f"\n\nFeature Importances:\n{feature_importance_text}\n\n"
            "Interpretation reminder:\n"
            "A split is useful when it creates groups with more similar XLogP values. "
            "The tree is not proving environmental fate; it is learning structural rules "
            "related to a partitioning descriptor."
        )
        self.results_text.setText(results)
        self.predict_button.setEnabled(True)

    def predict_demo_pfas(self):
        if self.model is None or self.df is None or not self.features:
            QMessageBox.warning(self, "No Model", "Please generate a tree before making predictions.")
            return

        df = self.df.copy()

        preferred = ["PFBA", "PFOA", "PFOS", "PFHxS", "HFPO-DA", "6:2 FTS", "8:2 FTOH"]
        rows = []
        for abbr in preferred:
            hits = df[df.get("Abbreviation", pd.Series(dtype=str)).astype(str) == abbr]
            if not hits.empty:
                rows.append(hits.iloc[0])
            if len(rows) >= 4:
                break

        if len(rows) < 4:
            # Use low/mid/high examples if named examples are missing.
            sorted_df = df.sort_values("XLogP")
            idxs = sorted(set([0, len(sorted_df)//3, 2*len(sorted_df)//3, len(sorted_df)-1]))
            rows = [sorted_df.iloc[i] for i in idxs]

        prediction_text = "\n\n--- Demo Predictions for Selected PFAS Records ---\n"
        prediction_text += "(These are drawn from the dataset so the actual XLogP can be shown for comparison.)\n\n"

        for row in rows:
            X_row = pd.DataFrame([row[self.features].astype(float).values], columns=self.features)
            prediction = float(self.model.predict(X_row)[0])
            actual = float(row["XLogP"])
            name = str(row.get("Abbreviation", row.get("Preferred_Name", "PFAS")))
            full = str(row.get("Preferred_Name", ""))
            prediction_text += (
                f"{name}  {full}\n"
                f"  Predicted XLogP: {prediction:.3f}\n"
                f"  Actual XLogP:    {actual:.3f}\n"
                f"  Error:           {prediction - actual:+.3f}\n\n"
            )

        self.results_text.append(prediction_text)


class CobberEcoTreeApp(QMainWindow):
    def __init__(self, manual_dataset: List[PFASCompound]):
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

    def create_manual_sorter_tab(self, dataset: List[PFASCompound]) -> QWidget:
        container = QWidget()
        self.manual_dataset = dataset
        self.pfas_map = {p.display_name: p for p in self.manual_dataset}

        main_layout = QVBoxLayout(container)

        top_layout = QHBoxLayout()
        title = QLabel(
            "<b>Manual sorter:</b> Drag PFAS cards into bins. "
            "Try to create groups with similar XLogP values."
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

        # Lists/bins
        self.deck_list_widget = PFASListWidget("deck")
        self.bin1_list_widget = PFASListWidget("bin1")
        self.bin2_list_widget = PFASListWidget("bin2")
        self.bin1_1_list_widget = PFASListWidget("bin1_1")
        self.bin1_2_list_widget = PFASListWidget("bin1_2")
        self.bin2_1_list_widget = PFASListWidget("bin2_1")
        self.bin2_2_list_widget = PFASListWidget("bin2_2")

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
            "Click a PFAS card to see details.\n\n"
            "XLogP is the target value. The decision-tree cost is the weighted variance "
            "of XLogP in the two child bins."
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
        self.detail_box.setText("Click a PFAS card to see details.")

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
            list_widget.itemClicked.connect(self.show_pfas_details)

    def show_pfas_details(self, item: QListWidgetItem):
        key = item.data(Qt.ItemDataRole.UserRole)
        pfas = self.pfas_map.get(key)
        if pfas:
            self.detail_box.setText(pfas.detail_text())

    def _get_pfas_from_list(self, list_widget: PFASListWidget) -> List[PFASCompound]:
        out = []
        for i in range(list_widget.count()):
            item = list_widget.item(i)
            key = item.data(Qt.ItemDataRole.UserRole)
            if key in self.pfas_map:
                out.append(self.pfas_map[key])
        return out

    def update_manual_calculations(self):
        def _update_bin_stats(pfas_list: List[PFASCompound], key_prefix: str):
            n = len(pfas_list)
            self.score_labels[f"{key_prefix}_count"].setText(f"Count (n): {n}")
            if n > 0:
                mean = float(np.mean([p.xlogp for p in pfas_list]))
                var = float(np.var([p.xlogp for p in pfas_list])) if n > 1 else 0.0
                self.score_labels[f"{key_prefix}_mean"].setText(f"Mean XLogP: {mean:.2f}")
                self.score_labels[f"{key_prefix}_var"].setText(f"Variance: {var:.2f}")
            else:
                self.score_labels[f"{key_prefix}_mean"].setText("Mean XLogP: N/A")
                self.score_labels[f"{key_prefix}_var"].setText("Variance: N/A")

        pfas_bin1 = self._get_pfas_from_list(self.bin1_list_widget)
        pfas_bin2 = self._get_pfas_from_list(self.bin2_list_widget)
        pfas_bin1_1 = self._get_pfas_from_list(self.bin1_1_list_widget)
        pfas_bin1_2 = self._get_pfas_from_list(self.bin1_2_list_widget)
        pfas_bin2_1 = self._get_pfas_from_list(self.bin2_1_list_widget)
        pfas_bin2_2 = self._get_pfas_from_list(self.bin2_2_list_widget)

        _update_bin_stats(pfas_bin1, "split1_binA")
        _update_bin_stats(pfas_bin2, "split1_binB")
        cost1 = calculate_weighted_variance(pfas_bin1, pfas_bin2)
        self.score_labels["split1_cost"].setText(f"<b>TOTAL COST: {cost1:.2f}</b>")

        _update_bin_stats(pfas_bin1_1, "split2_binA")
        _update_bin_stats(pfas_bin1_2, "split2_binB")
        cost2 = calculate_weighted_variance(pfas_bin1_1, pfas_bin1_2)
        self.score_labels["split2_cost"].setText(f"<b>TOTAL COST: {cost2:.2f}</b>")

        _update_bin_stats(pfas_bin2_1, "split3_binA")
        _update_bin_stats(pfas_bin2_2, "split3_binB")
        cost3 = calculate_weighted_variance(pfas_bin2_1, pfas_bin2_2)
        self.score_labels["split3_cost"].setText(f"<b>TOTAL COST: {cost3:.2f}</b>")

    def populate_deck(self):
        for pfas in self.manual_dataset:
            item = QListWidgetItem(pfas.card_text())
            item.setData(Qt.ItemDataRole.UserRole, pfas.display_name)
            font = QFont("Lato")
            font.setPointSize(9)
            item.setFont(font)
            self.deck_list_widget.addItem(item)

    def create_bin_section(self, title: str, list_widget: PFASListWidget) -> QFrame:
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
        self.score_labels[f"{key_prefix}_binA_mean"] = QLabel("Mean XLogP: N/A")
        self.score_labels[f"{key_prefix}_binA_var"] = QLabel("Variance: N/A")
        binA_layout.addWidget(binA_label)
        binA_layout.addWidget(self.score_labels[f"{key_prefix}_binA_count"])
        binA_layout.addWidget(self.score_labels[f"{key_prefix}_binA_mean"])
        binA_layout.addWidget(self.score_labels[f"{key_prefix}_binA_var"])

        binB_layout = QVBoxLayout()
        binB_label = QLabel(f"<i>{binB_name}</i>")
        self.score_labels[f"{key_prefix}_binB_count"] = QLabel("Count (n): 0")
        self.score_labels[f"{key_prefix}_binB_mean"] = QLabel("Mean XLogP: N/A")
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


def load_manual_dataset() -> List[PFASCompound]:
    manual_path = find_data_file(MANUAL_FILE)
    tree_path = find_data_file(TREE_FILE)

    if manual_path is not None:
        df = pd.read_csv(manual_path)
    elif tree_path is not None:
        df = pd.read_csv(tree_path).head(12)
    else:
        raise FileNotFoundError(
            f"Could not find {MANUAL_FILE} or {TREE_FILE} in the script folder/current directory."
        )

    if "XLogP" not in df.columns:
        raise ValueError("Manual dataset must contain an XLogP column.")

    df["XLogP"] = pd.to_numeric(df["XLogP"], errors="coerce")
    df = df.dropna(subset=["XLogP"]).copy()

    if len(df) == 0:
        raise ValueError("Manual dataset has no usable XLogP values.")

    # Sort cards by XLogP only for a stable deck order. Students can still
    # use any structure/property logic they want for sorting.
    df = df.sort_values(["XLogP", "Abbreviation"], na_position="last").reset_index(drop=True)

    return [PFASCompound.from_row(row) for _, row in df.iterrows()]


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
        QMessageBox.critical(None, "Could not load PFAS dataset", str(exc))
        return 1

    window = CobberEcoTreeApp(manual_dataset)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
