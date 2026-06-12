# CobberEcoK.py
# A PyQt6 application for exploring K-Means clustering and K-Nearest Neighbors
# classification using a teaching-realistic macroinvertebrate bioassessment dataset.
#
# Ecology version of CobberK.
#
# Teaching goal:
#   K-Means and KNN are both distance-based methods, but they answer different
#   questions.
#
#   K-Means:  unsupervised clustering
#             "If labels are hidden, what groups appear in trait space?"
#
#   KNN:      supervised classification
#             "Given labeled examples, what label should an unknown organism get?"
#
# Expected optional image files in the same folder as this script/exe:
#   Mayfly.png
#   Stonefly.png
#   Chadisfly.png       # user-provided spelling supported
#   Caddisfly.png       # also supported
#   RiffleBeetle.png
#   Dragonfly.png
#   Scud.png
#   AquaticSowbub.png   # user-provided spelling supported
#   AquaticSowbug.png   # also supported
#   Midge.png
#   AquaticWorm.png
#   Leech.png
#
# Dependencies:
#   pip install PyQt6 numpy matplotlib
#
# Run:
#   python CobberEcoK.py

from __future__ import annotations

import sys
import math
import random
import copy
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QFrame, QTabWidget, QPushButton, QSpinBox, QFormLayout, QMessageBox,
    QTextEdit, QComboBox, QGroupBox
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QColor, QPixmap

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Circle


# ---------------------------------------------------------------------
# Application/root helpers
# ---------------------------------------------------------------------
def app_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


APP_ROOT = app_root()


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------
TRAIT_INFO = {
    "oxygen_need": {
        "label": "Oxygen need score",
        "short": "Oxygen",
        "description": "Higher values represent organisms more associated with well-oxygenated water.",
    },
    "sediment_sensitivity": {
        "label": "Sediment sensitivity score",
        "short": "Sediment sensitivity",
        "description": "Higher values represent organisms less tolerant of fine sediment and embedded habitat.",
    },
    "flow_preference": {
        "label": "Flow preference score",
        "short": "Flow preference",
        "description": "Higher values represent organisms more associated with flowing riffle habitat.",
    },
    "body_size_mm": {
        "label": "Body size (mm)",
        "short": "Body size",
        "description": "Approximate body length for a teaching dataset; real values vary by life stage.",
    },
}

TRAIT_KEYS = list(TRAIT_INFO.keys())


IMAGE_ALIASES = {
    "Mayfly": ["Mayfly.png"],
    "Stonefly": ["Stonefly.png"],
    "Caddisfly": ["Chadisfly.png", "Caddisfly.png", "Caddishfly.png"],
    "RiffleBeetle": ["RiffleBeetle.png", "Riffle_Beetle.png"],
    "Dragonfly": ["Dragonfly.png"],
    "Scud": ["Scud.png"],
    "AquaticSowbug": ["AquaticSowbub.png", "AquaticSowbug.png", "AquaticSnowbug.png"],
    "Midge": ["Midge.png"],
    "AquaticWorm": ["AquaticWorm.png"],
    "Leech": ["Leech.png"],
}


@dataclass
class Macroinvertebrate:
    name: str
    short_label: str
    group: str  # Sensitive, Moderate, Tolerant
    archetype: str
    traits: Dict[str, float]
    notes: str
    cluster: int = -1
    coords_original: Tuple[float, float] = field(default=(0.0, 0.0))
    coords_scaled: Tuple[float, float] = field(default=(0.0, 0.0))


MACRO_DATASET: List[Macroinvertebrate] = [
    # Sensitive organisms: high oxygen, high sediment sensitivity, often high flow preference.
    Macroinvertebrate(
        "Flatheaded mayfly nymph", "May1", "Sensitive", "Mayfly",
        {"oxygen_need": 9.2, "sediment_sensitivity": 8.6, "flow_preference": 8.8, "body_size_mm": 11.0},
        "A mayfly-like nymph associated with clear, rocky, well-oxygenated riffles."
    ),
    Macroinvertebrate(
        "Small swimmer mayfly nymph", "May2", "Sensitive", "Mayfly",
        {"oxygen_need": 8.4, "sediment_sensitivity": 7.8, "flow_preference": 7.1, "body_size_mm": 8.5},
        "A smaller mayfly-like nymph still associated with relatively clean water."
    ),
    Macroinvertebrate(
        "Clinger mayfly nymph", "May3", "Sensitive", "Mayfly",
        {"oxygen_need": 9.5, "sediment_sensitivity": 9.1, "flow_preference": 9.2, "body_size_mm": 12.5},
        "A strong riffle-associated mayfly-like organism."
    ),
    Macroinvertebrate(
        "Burrower mayfly nymph", "May4", "Moderate", "Mayfly",
        {"oxygen_need": 6.6, "sediment_sensitivity": 5.6, "flow_preference": 4.8, "body_size_mm": 16.0},
        "A mayfly-like organism that is less strongly tied to fast rocky riffles."
    ),

    Macroinvertebrate(
        "Stonefly nymph", "Sto1", "Sensitive", "Stonefly",
        {"oxygen_need": 9.7, "sediment_sensitivity": 9.3, "flow_preference": 9.4, "body_size_mm": 18.0},
        "Stonefly-like nymphs are classic indicators of cold, oxygen-rich stream habitat."
    ),
    Macroinvertebrate(
        "Small stonefly nymph", "Sto2", "Sensitive", "Stonefly",
        {"oxygen_need": 9.1, "sediment_sensitivity": 8.8, "flow_preference": 8.9, "body_size_mm": 10.5},
        "A smaller stonefly-like nymph in the sensitive region of trait space."
    ),
    Macroinvertebrate(
        "Winter stonefly nymph", "Sto3", "Sensitive", "Stonefly",
        {"oxygen_need": 8.8, "sediment_sensitivity": 8.1, "flow_preference": 8.5, "body_size_mm": 13.0},
        "A stonefly-like organism with high oxygen and flow scores."
    ),

    Macroinvertebrate(
        "Case-building caddisfly larva", "Cad1", "Sensitive", "Caddisfly",
        {"oxygen_need": 8.0, "sediment_sensitivity": 7.5, "flow_preference": 7.8, "body_size_mm": 14.0},
        "A caddisfly-like larva with a protective case in a rocky stream habitat."
    ),
    Macroinvertebrate(
        "Net-spinning caddisfly larva", "Cad2", "Sensitive", "Caddisfly",
        {"oxygen_need": 8.5, "sediment_sensitivity": 7.9, "flow_preference": 8.4, "body_size_mm": 13.0},
        "A caddisfly-like larva associated with flowing water."
    ),
    Macroinvertebrate(
        "Soft-case caddisfly larva", "Cad3", "Moderate", "Caddisfly",
        {"oxygen_need": 6.4, "sediment_sensitivity": 5.9, "flow_preference": 5.7, "body_size_mm": 12.0},
        "A caddisfly-like organism placed near the moderate/sensitive boundary."
    ),
    Macroinvertebrate(
        "Large caddisfly larva", "Cad4", "Sensitive", "Caddisfly",
        {"oxygen_need": 7.6, "sediment_sensitivity": 7.2, "flow_preference": 7.0, "body_size_mm": 20.0},
        "A larger caddisfly-like organism with moderately high sensitivity traits."
    ),

    Macroinvertebrate(
        "Riffle beetle adult", "Rif1", "Sensitive", "RiffleBeetle",
        {"oxygen_need": 8.2, "sediment_sensitivity": 8.0, "flow_preference": 9.0, "body_size_mm": 5.0},
        "A small beetle associated with oxygenated flowing water and clean rocky substrate."
    ),
    Macroinvertebrate(
        "Riffle beetle larva", "Rif2", "Sensitive", "RiffleBeetle",
        {"oxygen_need": 8.7, "sediment_sensitivity": 8.4, "flow_preference": 8.7, "body_size_mm": 6.5},
        "A riffle-associated beetle larva-like organism."
    ),
    Macroinvertebrate(
        "Stream beetle", "Rif3", "Moderate", "RiffleBeetle",
        {"oxygen_need": 6.9, "sediment_sensitivity": 6.4, "flow_preference": 7.2, "body_size_mm": 7.0},
        "A beetle-like organism placed near the moderate/sensitive boundary."
    ),

    # Moderate organisms: intermediate oxygen/sensitivity; may live in slower water or vegetation.
    Macroinvertebrate(
        "Dragonfly nymph", "Dra1", "Moderate", "Dragonfly",
        {"oxygen_need": 5.8, "sediment_sensitivity": 4.5, "flow_preference": 2.7, "body_size_mm": 25.0},
        "A predatory nymph often associated with slower water, vegetation, and detritus."
    ),
    Macroinvertebrate(
        "Large dragonfly nymph", "Dra2", "Moderate", "Dragonfly",
        {"oxygen_need": 5.2, "sediment_sensitivity": 4.2, "flow_preference": 2.2, "body_size_mm": 32.0},
        "A large predatory aquatic nymph in slower-water habitat."
    ),
    Macroinvertebrate(
        "Marsh dragonfly nymph", "Dra3", "Moderate", "Dragonfly",
        {"oxygen_need": 4.8, "sediment_sensitivity": 3.8, "flow_preference": 1.8, "body_size_mm": 28.0},
        "A dragonfly-like nymph that tolerates slower, muddier habitat."
    ),
    Macroinvertebrate(
        "Damselfly-like nymph", "Dra4", "Moderate", "Dragonfly",
        {"oxygen_need": 5.9, "sediment_sensitivity": 5.0, "flow_preference": 3.5, "body_size_mm": 19.0},
        "A slender odonate-like nymph with intermediate bioassessment traits."
    ),

    Macroinvertebrate(
        "Freshwater scud", "Scu1", "Moderate", "Scud",
        {"oxygen_need": 5.6, "sediment_sensitivity": 4.7, "flow_preference": 3.3, "body_size_mm": 8.0},
        "A small amphipod-like crustacean found among leaves, plants, and detritus."
    ),
    Macroinvertebrate(
        "Leaf-litter scud", "Scu2", "Moderate", "Scud",
        {"oxygen_need": 5.0, "sediment_sensitivity": 3.9, "flow_preference": 2.7, "body_size_mm": 7.0},
        "A scud-like organism associated with leaf litter and slower margins."
    ),
    Macroinvertebrate(
        "Stream-margin scud", "Scu3", "Moderate", "Scud",
        {"oxygen_need": 4.9, "sediment_sensitivity": 4.2, "flow_preference": 3.0, "body_size_mm": 9.5},
        "A moderate-tolerance amphipod-like organism."
    ),
    Macroinvertebrate(
        "Coldwater scud", "Scu4", "Moderate", "Scud",
        {"oxygen_need": 6.3, "sediment_sensitivity": 5.2, "flow_preference": 4.0, "body_size_mm": 8.5},
        "A scud-like organism pulled toward cleaner, cooler water."
    ),

    Macroinvertebrate(
        "Aquatic sowbug", "Sow1", "Tolerant", "AquaticSowbug",
        {"oxygen_need": 3.6, "sediment_sensitivity": 2.7, "flow_preference": 1.8, "body_size_mm": 10.0},
        "A freshwater isopod-like organism often found in detritus-rich, slower water."
    ),
    Macroinvertebrate(
        "Leaf-pack sowbug", "Sow2", "Tolerant", "AquaticSowbug",
        {"oxygen_need": 3.2, "sediment_sensitivity": 2.3, "flow_preference": 1.5, "body_size_mm": 11.5},
        "An isopod-like organism associated with organic matter and low-flow habitat."
    ),
    Macroinvertebrate(
        "Stream sowbug", "Sow3", "Moderate", "AquaticSowbug",
        {"oxygen_need": 4.4, "sediment_sensitivity": 3.5, "flow_preference": 2.5, "body_size_mm": 9.0},
        "A sowbug-like organism near the moderate/tolerant boundary."
    ),

    # Tolerant organisms: low oxygen need, low sediment sensitivity, often slow water/fine sediment.
    Macroinvertebrate(
        "Red midge larva", "Mid1", "Tolerant", "Midge",
        {"oxygen_need": 2.2, "sediment_sensitivity": 1.6, "flow_preference": 1.3, "body_size_mm": 7.0},
        "A chironomid-like larva, often a tolerant group in stream bioassessment."
    ),
    Macroinvertebrate(
        "Small midge larva", "Mid2", "Tolerant", "Midge",
        {"oxygen_need": 2.8, "sediment_sensitivity": 2.0, "flow_preference": 1.6, "body_size_mm": 5.5},
        "A small midge-like larva in soft sediment or detritus."
    ),
    Macroinvertebrate(
        "Bloodworm midge larva", "Mid3", "Tolerant", "Midge",
        {"oxygen_need": 1.8, "sediment_sensitivity": 1.3, "flow_preference": 1.1, "body_size_mm": 10.0},
        "A red bloodworm-like midge larva associated with low-oxygen conditions."
    ),
    Macroinvertebrate(
        "Midge larva in detritus", "Mid4", "Tolerant", "Midge",
        {"oxygen_need": 3.0, "sediment_sensitivity": 2.1, "flow_preference": 1.4, "body_size_mm": 6.5},
        "A tolerant midge-like larva in detritus-rich habitat."
    ),

    Macroinvertebrate(
        "Aquatic worm", "Wor1", "Tolerant", "AquaticWorm",
        {"oxygen_need": 1.7, "sediment_sensitivity": 1.2, "flow_preference": 1.0, "body_size_mm": 22.0},
        "An oligochaete-like worm associated with fine sediment and organic enrichment."
    ),
    Macroinvertebrate(
        "Slender aquatic worm", "Wor2", "Tolerant", "AquaticWorm",
        {"oxygen_need": 2.0, "sediment_sensitivity": 1.4, "flow_preference": 1.2, "body_size_mm": 18.0},
        "A small worm-like organism in muddy substrate."
    ),
    Macroinvertebrate(
        "Organic-mud worm", "Wor3", "Tolerant", "AquaticWorm",
        {"oxygen_need": 1.4, "sediment_sensitivity": 1.1, "flow_preference": 1.0, "body_size_mm": 30.0},
        "A larger worm-like organism associated with soft organic sediment."
    ),

    Macroinvertebrate(
        "Freshwater leech", "Lee1", "Tolerant", "Leech",
        {"oxygen_need": 2.9, "sediment_sensitivity": 2.1, "flow_preference": 1.4, "body_size_mm": 28.0},
        "A leech-like organism often found in slower water and vegetation."
    ),
    Macroinvertebrate(
        "Small leech", "Lee2", "Tolerant", "Leech",
        {"oxygen_need": 3.3, "sediment_sensitivity": 2.5, "flow_preference": 1.8, "body_size_mm": 18.0},
        "A smaller leech-like organism in a tolerant region of trait space."
    ),
    Macroinvertebrate(
        "Vegetation leech", "Lee3", "Tolerant", "Leech",
        {"oxygen_need": 3.7, "sediment_sensitivity": 2.8, "flow_preference": 1.6, "body_size_mm": 35.0},
        "A leech-like organism associated with slow water and aquatic vegetation."
    ),
]


# ---------------------------------------------------------------------
# Core methods
# ---------------------------------------------------------------------
def trait_label(key: str) -> str:
    return TRAIT_INFO[key]["label"]


def trait_short(key: str) -> str:
    return TRAIT_INFO[key]["short"]


def candidate_image_dirs() -> List[Path]:
    """
    Return directories to search for organism image files.

    This is intentionally redundant because PyInstaller can be run in several
    ways, and users may place the PNG files beside the .exe, beside the .py
    file, in the working directory, or in a Figures/ subfolder.

    For a PyInstaller one-file executable, sys.executable points to the .exe,
    while __file__ may point inside PyInstaller's temporary extraction folder.
    The directory beside the .exe is therefore the most important location.
    """
    dirs: List[Path] = []

    def add(path_like):
        try:
            p = Path(path_like).resolve()
            if p not in dirs:
                dirs.append(p)
        except Exception:
            pass

    # 1. Folder containing the executable when frozen, or script folder when not frozen.
    add(APP_ROOT)

    # 2. Current working directory. This helps if the app is launched from a terminal
    #    or shortcut with a different working directory.
    add(Path.cwd())

    # 3. Script/source folder when running as a .py file.
    try:
        add(Path(__file__).resolve().parent)
    except Exception:
        pass

    # 4. PyInstaller temporary extraction folder, useful if images are bundled later.
    if hasattr(sys, "_MEIPASS"):
        add(Path(sys._MEIPASS))  # type: ignore[attr-defined]

    # 5. Also check common image subfolders under each base directory.
    bases = list(dirs)
    for base in bases:
        add(base / "Figures")
        add(base / "figures")
        add(base / "Images")
        add(base / "images")
        add(base / "Organisms")
        add(base / "organisms")

    return dirs


def image_path_for_archetype(archetype: str) -> Optional[Path]:
    """
    Find the image file for an organism archetype.

    The lookup is case-insensitive as a fallback, which helps on Windows if a
    filename differs by capitalization.
    """
    filenames = IMAGE_ALIASES.get(archetype, [])

    for directory in candidate_image_dirs():
        for filename in filenames:
            path = directory / filename
            if path.exists():
                return path

    # Case-insensitive fallback for Windows/macOS/path typo resilience.
    lowered = {name.lower() for name in filenames}
    for directory in candidate_image_dirs():
        if not directory.exists() or not directory.is_dir():
            continue
        try:
            for child in directory.iterdir():
                if child.name.lower() in lowered:
                    return child
        except Exception:
            pass

    return None


def image_search_report(archetype: str) -> str:
    filenames = ", ".join(IMAGE_ALIASES.get(archetype, []))
    dirs = "\n".join(str(d) for d in candidate_image_dirs())
    return f"Expected filenames:\n{filenames}\n\nSearched folders:\n{dirs}"


def normalize_dataset_for_traits(dataset: List[Macroinvertebrate], x_trait: str, y_trait: str):
    x_values = [org.traits[x_trait] for org in dataset]
    y_values = [org.traits[y_trait] for org in dataset]

    min_x, max_x = min(x_values), max(x_values)
    min_y, max_y = min(y_values), max(y_values)

    for org in dataset:
        x = org.traits[x_trait]
        y = org.traits[y_trait]
        sx = (x - min_x) / (max_x - min_x) if max_x != min_x else 0.0
        sy = (y - min_y) / (max_y - min_y) if max_y != min_y else 0.0
        org.coords_original = (x, y)
        org.coords_scaled = (sx, sy)


def calculate_distance(p1, p2) -> float:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def assign_to_clusters(organisms, centroids):
    for org in organisms:
        if not centroids:
            org.cluster = -1
            continue
        org.cluster = min(
            centroids.keys(),
            key=lambda cid: calculate_distance(org.coords_scaled, centroids[cid])
        )


def update_centroids(organisms, k):
    clusters = {i: [] for i in range(k)}
    for org in organisms:
        if org.cluster != -1:
            clusters[org.cluster].append(org.coords_scaled)

    new_centroids = {}
    for i, points in clusters.items():
        if points:
            new_centroids[i] = (
                float(np.mean([p[0] for p in points])),
                float(np.mean([p[1] for p in points])),
            )
    return new_centroids


def calculate_inertia(organisms, centroids):
    return sum(
        calculate_distance(org.coords_scaled, centroids[org.cluster]) ** 2
        for org in organisms
        if org.cluster != -1 and org.cluster in centroids
    )


def find_k_neighbors(training_data, unknown, k):
    distances = sorted([
        (calculate_distance(unknown.coords_scaled, org.coords_scaled), org)
        for org in training_data
    ])
    return [org for dist, org in distances[:k]]


def predict_class(neighbors):
    return Counter(n.group for n in neighbors).most_common(1)[0][0]


# ---------------------------------------------------------------------
# Canvas and selection panel
# ---------------------------------------------------------------------
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)


class OrganismInfoPanel(QGroupBox):
    def __init__(self, title="Selected organism"):
        super().__init__(title)

        layout = QVBoxLayout(self)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumHeight(210)
        self.image_label.setStyleSheet(
            "background-color: #f5f5f5; border: 1px solid #bbbbbb; padding: 4px;"
        )

        self.name_label = QLabel("<b>Click a point to inspect an organism.</b>")
        self.name_label.setWordWrap(True)

        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMinimumHeight(230)

        layout.addWidget(self.image_label)
        layout.addWidget(self.name_label)
        layout.addWidget(self.info_text, 1)

        self.show_empty()

    def show_empty(self):
        self.image_label.setText("No organism selected")
        self.image_label.setPixmap(QPixmap())
        self.name_label.setText("<b>Click a point to inspect an organism.</b>")
        self.info_text.setText(
            "The image panel is here for context, not for the ML calculation.\n\n"
            "K-Means and KNN use only the numeric trait values shown on the axes."
        )

    def show_organism(self, org: Macroinvertebrate, x_trait: str, y_trait: str):
        self.name_label.setText(f"<b>{org.name}</b><br>Group: {org.group}")

        path = image_path_for_archetype(org.archetype)
        if path and path.exists():
            pix = QPixmap(str(path))
            if not pix.isNull():
                pix = pix.scaled(
                    230, 230,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                self.image_label.setPixmap(pix)
                self.image_label.setText("")
            else:
                self.image_label.setPixmap(QPixmap())
                self.image_label.setText(f"Could not load image:\n{path.name}")
        else:
            self.image_label.setPixmap(QPixmap())
            self.image_label.setText("Image not found.\nSee details below.")
            self.info_text.setText(
                f"Image lookup failed for: {org.archetype}\n\n"
                + image_search_report(org.archetype)
                + "\n\n"
                + f"Organism: {org.name}\n"
                + f"Group: {org.group}\n\n"
                + "The ML parts of the app still work. This message is only about locating the PNG file."
            )
            return

        traits_lines = []
        for key in TRAIT_KEYS:
            traits_lines.append(f"{trait_short(key)}: {org.traits[key]:.2f}")

        selected_text = (
            f"Plotted x-axis: {trait_label(x_trait)} = {org.traits[x_trait]:.2f}\n"
            f"Plotted y-axis: {trait_label(y_trait)} = {org.traits[y_trait]:.2f}\n"
        )

        self.info_text.setText(
            f"Short label: {org.short_label}\n"
            f"Archetype image: {org.archetype}\n\n"
            f"{selected_text}\n"
            "All numeric traits:\n"
            + "\n".join(traits_lines)
            + "\n\nEcology note:\n"
            + org.notes
            + "\n\nReminder:\n"
            "The organism image helps you connect a point to a living stream animal. "
            "The algorithm uses only the selected numeric traits."
        )


# ---------------------------------------------------------------------
# Shared plot click utility
# ---------------------------------------------------------------------
def nearest_organism_from_click(event, organisms: List[Macroinvertebrate], max_pixels: float = 18.0):
    if event.inaxes is None or event.x is None or event.y is None:
        return None

    best_org = None
    best_dist = float("inf")

    for org in organisms:
        x, y = org.coords_scaled
        px, py = event.inaxes.transData.transform((x, y))
        dist = math.sqrt((event.x - px) ** 2 + (event.y - py) ** 2)
        if dist < best_dist:
            best_dist = dist
            best_org = org

    if best_dist <= max_pixels:
        return best_org
    return None


# ---------------------------------------------------------------------
# K-Means tab
# ---------------------------------------------------------------------
class KMeansTab(QWidget):
    def __init__(self, dataset, main_window):
        super().__init__()

        self.main_window = main_window
        self.dataset = copy.deepcopy(dataset)
        self.centroids = {}
        self.selected_org = None

        self.cluster_colors = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
            "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"
        ]

        main_layout = QHBoxLayout(self)

        controls_layout = QVBoxLayout()
        form_layout = QFormLayout()

        self.x_trait_combo = QComboBox()
        self.y_trait_combo = QComboBox()
        for key in TRAIT_KEYS:
            self.x_trait_combo.addItem(trait_label(key), key)
            self.y_trait_combo.addItem(trait_label(key), key)

        self.x_trait_combo.setCurrentIndex(0)  # oxygen
        self.y_trait_combo.setCurrentIndex(1)  # sediment

        self.k_spinner = QSpinBox()
        self.k_spinner.setRange(2, 8)
        self.k_spinner.setValue(3)

        form_layout.addRow("X trait:", self.x_trait_combo)
        form_layout.addRow("Y trait:", self.y_trait_combo)
        form_layout.addRow("Number of clusters (k):", self.k_spinner)

        self.init_button = QPushButton("Initialize Centroids")
        self.run_step_button = QPushButton("Update Centroids (1 Step)")
        self.run_full_button = QPushButton("Run Full Algorithm")
        self.inertia_label = QLabel("<b>Total Inertia:</b> N/A")

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(170)

        controls_layout.addWidget(QLabel("<b>K-Means controls</b>"))
        controls_layout.addLayout(form_layout)
        controls_layout.addWidget(self.init_button)
        controls_layout.addWidget(self.run_step_button)
        controls_layout.addWidget(self.run_full_button)
        controls_layout.addWidget(self.inertia_label)
        controls_layout.addWidget(QLabel("<b>Log:</b>"))
        controls_layout.addWidget(self.log_text, 1)

        self.canvas = MplCanvas(self, width=8, height=6, dpi=100)
        self.info_panel = OrganismInfoPanel()

        main_layout.addLayout(controls_layout, 1)
        main_layout.addWidget(self.canvas, 3)
        main_layout.addWidget(self.info_panel, 1)

        self.init_button.clicked.connect(self.initialize_centroids)
        self.run_step_button.clicked.connect(self.run_one_step)
        self.run_full_button.clicked.connect(self.run_full_algorithm)
        self.x_trait_combo.currentIndexChanged.connect(self.feature_space_changed)
        self.y_trait_combo.currentIndexChanged.connect(self.feature_space_changed)
        self.k_spinner.valueChanged.connect(self.k_changed)

        self.canvas.mpl_connect("button_press_event", self.handle_plot_click)

        self.run_step_button.setEnabled(False)
        self.feature_space_changed()

    def x_trait(self):
        return self.x_trait_combo.currentData()

    def y_trait(self):
        return self.y_trait_combo.currentData()

    def feature_space_changed(self):
        if self.x_trait() == self.y_trait():
            self.log_text.append("Choose two different traits for a useful trait-space plot.")
        normalize_dataset_for_traits(self.dataset, self.x_trait(), self.y_trait())
        self.centroids = {}
        for org in self.dataset:
            org.cluster = -1
        self.selected_org = None
        self.run_step_button.setEnabled(False)
        self.inertia_label.setText("<b>Total Inertia:</b> N/A")
        self.info_panel.show_empty()
        self.redraw_plot()

    def k_changed(self):
        self.log_text.append(f"\n--- New K-Means test for k={self.k_spinner.value()} ---")
        self.centroids = {}
        for org in self.dataset:
            org.cluster = -1
        self.run_step_button.setEnabled(False)
        self.inertia_label.setText("<b>Total Inertia:</b> N/A")
        self.redraw_plot()

    def initialize_centroids(self):
        if self.x_trait() == self.y_trait():
            QMessageBox.warning(self, "Same Trait", "Please choose two different traits.")
            return

        for org in self.dataset:
            org.cluster = -1

        k = self.k_spinner.value()
        initial_points = random.sample(self.dataset, k)
        self.centroids = {i: pt.coords_scaled for i, pt in enumerate(initial_points)}
        self.run_step_button.setEnabled(True)
        self.update_clusters_and_inertia(assign_only=True)
        self.log_text.append(f"Initialized {k} centroids.")

    def run_one_step(self):
        new_centroids = update_centroids(self.dataset, self.k_spinner.value())

        if new_centroids and all(
            i in self.centroids
            and i in new_centroids
            and calculate_distance(self.centroids[i], new_centroids[i]) < 1e-6
            for i in new_centroids
        ):
            QMessageBox.information(self, "Converged!", "The centroids did not move significantly.")

        self.centroids.update(new_centroids)
        self.update_clusters_and_inertia()

    def run_full_algorithm(self):
        self.initialize_centroids()
        if not self.centroids:
            return

        for step in range(30):
            old_centroids = self.centroids.copy()
            assign_to_clusters(self.dataset, self.centroids)
            self.centroids.update(update_centroids(self.dataset, self.k_spinner.value()))

            converged = all(
                i in old_centroids
                and i in self.centroids
                and calculate_distance(old_centroids[i], self.centroids[i]) < 1e-7
                for i in self.centroids
            )
            if converged:
                break

        assign_to_clusters(self.dataset, self.centroids)
        inertia = calculate_inertia(self.dataset, self.centroids)

        self.log_text.append(
            f"Converged for k={self.k_spinner.value()} with final inertia: {inertia:.4f}"
        )
        self.log_cluster_summary()
        self.inertia_label.setText(f"<b>Total Inertia:</b> {inertia:.4f}")
        self.redraw_plot()

    def update_clusters_and_inertia(self, assign_only=False):
        if not assign_only:
            self.centroids.update(update_centroids(self.dataset, self.k_spinner.value()))

        assign_to_clusters(self.dataset, self.centroids)
        inertia = calculate_inertia(self.dataset, self.centroids)
        self.inertia_label.setText(f"<b>Total Inertia:</b> {inertia:.4f}")
        self.redraw_plot()

    def log_cluster_summary(self):
        clusters = defaultdict(list)
        for org in self.dataset:
            if org.cluster != -1:
                clusters[org.cluster].append(org)

        for cluster_id in sorted(clusters):
            organisms = clusters[cluster_id]
            vote_counts = Counter(org.group for org in organisms)
            names = ", ".join(org.short_label for org in organisms[:8])
            if len(organisms) > 8:
                names += ", ..."
            self.log_text.append(
                f"Cluster {cluster_id}: n={len(organisms)}; "
                f"groups={dict(vote_counts)}; examples={names}"
            )

    def handle_plot_click(self, event):
        org = nearest_organism_from_click(event, self.dataset)
        if org:
            self.selected_org = org
            self.info_panel.show_organism(org, self.x_trait(), self.y_trait())
            self.redraw_plot()

    def redraw_plot(self):
        if getattr(self, "_drawing", False):
            return
        self._drawing = True
        try:
            ax = self.canvas.axes
            ax.clear()

            for org in self.dataset:
                x, y = org.coords_scaled
                color = self.cluster_colors[org.cluster % len(self.cluster_colors)] if org.cluster != -1 else "grey"
                size = 70 if org is self.selected_org else 42
                edge = "black" if org is self.selected_org else "none"
                lw = 1.8 if org is self.selected_org else 0
                ax.scatter(x, y, c=color, s=size, alpha=0.85, zorder=2, edgecolors=edge, linewidths=lw)
                ax.text(x + 0.01, y + 0.006, org.short_label, fontsize=8, zorder=3)

            for i, coords in self.centroids.items():
                ax.scatter(
                    coords[0], coords[1],
                    c=self.cluster_colors[i % len(self.cluster_colors)],
                    marker="X", s=250, edgecolor="black", linewidth=1.3, zorder=5
                )

            ax.set_title("Macroinvertebrates in Trait Space (Normalized Coordinates)")
            ax.set_xlabel("Normalized " + trait_short(self.x_trait()))
            ax.set_ylabel("Normalized " + trait_short(self.y_trait()))
            ax.set_xlim(-0.1, 1.1)
            ax.set_ylim(-0.1, 1.1)
            ax.grid(True, linestyle="--", alpha=0.55)
            ax.text(
                0.02, 0.02,
                "K-Means ignores the bioassessment labels.\\nIt only sees the selected numeric traits.",
                transform=ax.transAxes,
                fontsize=8,
                va="bottom",
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none")
            )
            self.canvas.fig.tight_layout()
            self.canvas.draw_idle()
        finally:
            self._drawing = False


# ---------------------------------------------------------------------
# KNN tab
# ---------------------------------------------------------------------
class KNNTab(QWidget):
    def __init__(self, dataset, main_window):
        super().__init__()

        self.main_window = main_window
        self.dataset = copy.deepcopy(dataset)
        self.unknown_org = None
        self.selected_org = None
        self.neighbors = []

        # Windows/Qt/Matplotlib stability helpers.
        # We coalesce redraw requests through the Qt event loop instead of forcing
        # immediate redraws from combo-box and button callbacks.
        self._drawing = False
        self._redraw_pending = False

        self.group_colors = {
            "Sensitive": "#1f77b4",
            "Moderate": "#2ca02c",
            "Tolerant": "#d62728",
        }

        main_layout = QHBoxLayout(self)

        controls_layout = QVBoxLayout()
        form_layout = QFormLayout()

        self.x_trait_combo = QComboBox()
        self.y_trait_combo = QComboBox()
        for key in TRAIT_KEYS:
            self.x_trait_combo.addItem(trait_label(key), key)
            self.y_trait_combo.addItem(trait_label(key), key)

        self.x_trait_combo.setCurrentIndex(0)
        self.y_trait_combo.setCurrentIndex(1)

        self.unknown_selector = QComboBox()
        self.unknown_selector.addItem("Select an organism...")
        for org in sorted(self.dataset, key=lambda o: o.name):
            self.unknown_selector.addItem(org.name)

        self.k_spinner = QSpinBox()
        self.k_spinner.setRange(1, 15)
        self.k_spinner.setValue(5)

        form_layout.addRow("X trait:", self.x_trait_combo)
        form_layout.addRow("Y trait:", self.y_trait_combo)
        form_layout.addRow("Select unknown:", self.unknown_selector)
        form_layout.addRow("Number of neighbors (k):", self.k_spinner)

        self.prediction_text = QTextEdit()
        self.prediction_text.setReadOnly(True)
        self.prediction_text.setMinimumHeight(250)

        self.test_all_button = QPushButton("Run Full Leave-One-Out Test")

        controls_layout.addWidget(QLabel("<b>KNN controls</b>"))
        controls_layout.addLayout(form_layout)
        controls_layout.addWidget(QLabel("<b>Prediction details:</b>"))
        controls_layout.addWidget(self.prediction_text, 1)
        controls_layout.addWidget(self.test_all_button)
        controls_layout.addStretch()

        self.canvas = MplCanvas(self, width=8, height=6, dpi=100)
        self.info_panel = OrganismInfoPanel()

        main_layout.addLayout(controls_layout, 1)
        main_layout.addWidget(self.canvas, 3)
        main_layout.addWidget(self.info_panel, 1)

        self.x_trait_combo.currentIndexChanged.connect(self.feature_space_changed)
        self.y_trait_combo.currentIndexChanged.connect(self.feature_space_changed)
        self.unknown_selector.currentTextChanged.connect(self.set_unknown)
        self.k_spinner.valueChanged.connect(self.update_view)
        self.test_all_button.clicked.connect(self.run_full_test)

        self.canvas.mpl_connect("button_press_event", self.handle_plot_click)

        self.feature_space_changed()

    def x_trait(self):
        return self.x_trait_combo.currentData()

    def y_trait(self):
        return self.y_trait_combo.currentData()

    def request_redraw(self):
        """Coalesce repeated redraw requests into one Qt event-loop callback."""
        if self._redraw_pending:
            return
        self._redraw_pending = True
        QTimer.singleShot(0, self._perform_redraw)

    def _perform_redraw(self):
        self._redraw_pending = False
        self.redraw_plot()

    def feature_space_changed(self):
        normalize_dataset_for_traits(self.dataset, self.x_trait(), self.y_trait())
        self.neighbors = []
        self.update_view()

    def set_unknown(self, name):
        self.neighbors = []
        if name == "Select an organism...":
            self.unknown_org = None
            self.selected_org = None
            self.info_panel.show_empty()
        else:
            self.unknown_org = next((org for org in self.dataset if org.name == name), None)
            self.selected_org = self.unknown_org
            if self.unknown_org:
                self.info_panel.show_organism(self.unknown_org, self.x_trait(), self.y_trait())
        self.update_view()

    def update_view(self):
        if self.x_trait() == self.y_trait():
            self.prediction_text.setPlainText("Choose two different traits for a useful KNN test.")
            self.request_redraw()
            return

        if not self.unknown_org:
            self.neighbors = []
            self.prediction_text.setPlainText(
                "Select one organism to treat as unknown.\n\n"
                "KNN will hide that organism's group label, find nearby labeled organisms, "
                "and predict the unknown group by neighbor vote."
            )
            self.request_redraw()
            return

        training_data = [org for org in self.dataset if org.name != self.unknown_org.name]
        k = min(self.k_spinner.value(), len(training_data))
        self.neighbors = find_k_neighbors(training_data, self.unknown_org, k)
        prediction = predict_class(self.neighbors)
        vote_counts = Counter(n.group for n in self.neighbors)

        correct = "YES" if prediction == self.unknown_org.group else "NO"

        lines = [
            f"Prediction for {self.unknown_org.name}",
            "=" * (15 + len(self.unknown_org.name)),
            f"Predicted group: {prediction}",
            f"Actual group:    {self.unknown_org.group}",
            f"Correct?         {correct}",
            "",
            "Neighbor votes:",
        ]

        for group, count in vote_counts.most_common():
            lines.append(f"  {group}: {count} vote(s)")

        lines.append("")
        lines.append("Nearest organisms:")
        for n in self.neighbors:
            lines.append(f"  {n.short_label}: {n.name} ({n.group})")

        self.prediction_text.setPlainText("\n".join(lines))
        self.request_redraw()

    def run_full_test(self):
        if self.x_trait() == self.y_trait():
            QMessageBox.warning(self, "Same Trait", "Please choose two different traits.")
            return

        correct = 0
        total = len(self.dataset)
        k = self.k_spinner.value()
        confusion = defaultdict(Counter)
        misses = []

        for test_org in self.dataset:
            training_data = [org for org in self.dataset if org.name != test_org.name]
            neighbors = find_k_neighbors(training_data, test_org, min(k, len(training_data)))
            prediction = predict_class(neighbors)
            confusion[test_org.group][prediction] += 1
            if prediction == test_org.group:
                correct += 1
            else:
                misses.append((test_org.short_label, test_org.group, prediction))

        success = 100.0 * correct / total

        lines = [
            "Full Leave-One-Out Test Results",
            "===============================",
            f"Traits: {trait_short(self.x_trait())} vs. {trait_short(self.y_trait())}",
            f"k = {k}",
            f"Correctly classified {correct} out of {total}.",
            f"Success Rate: {success:.1f}%",
            "",
            "Confusion summary:",
        ]

        for actual in ["Sensitive", "Moderate", "Tolerant"]:
            lines.append(f"  Actual {actual}: {dict(confusion[actual])}")

        if misses:
            lines.append("")
            lines.append("Example misses:")
            for short, actual, pred in misses[:8]:
                lines.append(f"  {short}: actual {actual}, predicted {pred}")

        if self.unknown_org:
            lines.append("")
            lines.append(f"Currently selected unknown: {self.unknown_org.name}")
            lines.append("Change the selected organism to refresh the individual KNN prediction.")

        self.prediction_text.setPlainText("\n".join(lines))
        self.request_redraw()

    def handle_plot_click(self, event):
        org = nearest_organism_from_click(event, self.dataset)
        if org:
            self.selected_org = org
            self.info_panel.show_organism(org, self.x_trait(), self.y_trait())
            self.request_redraw()

    def redraw_plot(self):
        if self._drawing:
            return
        self._drawing = True
        try:
            ax = self.canvas.axes
            ax.clear()

            training_data = [
                org for org in self.dataset
                if not self.unknown_org or org.name != self.unknown_org.name
            ]

            for org in training_data:
                x, y = org.coords_scaled
                color = self.group_colors.get(org.group, "grey")

                size = 70 if org is self.selected_org else 42
                edge = "black" if org is self.selected_org else "none"
                lw = 1.8 if org is self.selected_org else 0

                ax.scatter(
                    x, y,
                    c=color,
                    s=size,
                    alpha=0.85,
                    edgecolors=edge,
                    linewidths=lw,
                    zorder=2,
                )
                ax.text(x + 0.01, y + 0.006, org.short_label, fontsize=8, zorder=3)

            if self.unknown_org:
                ux, uy = self.unknown_org.coords_scaled
                ax.scatter(ux, uy, c="black", marker="X", s=220, zorder=6)
                ax.text(
                    ux + 0.015,
                    uy + 0.015,
                    self.unknown_org.short_label,
                    fontsize=9,
                    weight="bold",
                    zorder=7,
                )

                if self.neighbors:
                    for n in self.neighbors:
                        ax.scatter(
                            n.coords_scaled[0], n.coords_scaled[1],
                            s=165,
                            facecolors="none",
                            edgecolors="gold",
                            linewidth=2.4,
                            zorder=5,
                        )
                    # Deliberately not drawing the dashed radius circle here.
                    # The nearest-neighbor outlines convey the same lesson and
                    # the simpler Matplotlib state is more stable on Windows/Qt.

            ax.set_title("KNN Bioassessment Classification in Trait Space")
            ax.set_xlabel("Normalized " + trait_short(self.x_trait()))
            ax.set_ylabel("Normalized " + trait_short(self.y_trait()))
            ax.set_xlim(-0.1, 1.1)
            ax.set_ylim(-0.1, 1.1)
            ax.grid(True, linestyle="--", alpha=0.55)

            ax.text(
                0.02, 0.02,
                "Blue: Sensitive   Green: Moderate   Red: Tolerant\n"
                "Black X: unknown   Gold outlines: nearest neighbors",
                transform=ax.transAxes,
                fontsize=8,
                va="bottom",
                bbox=dict(facecolor="white", alpha=0.78, edgecolor="none"),
            )

            self.canvas.fig.tight_layout()
            self.canvas.draw_idle()
        finally:
            self._drawing = False


# ---------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------
class CobberEcoKApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.cobber_maroon = QColor(108, 29, 69)
        self.cobber_gold = QColor(234, 170, 0)
        self.lato_font = QFont("Lato")

        self.setWindowTitle("CobberEcoK")
        self.setGeometry(100, 100, 1320, 760)
        self.setFont(self.lato_font)

        tabs = QTabWidget()
        self.setCentralWidget(tabs)

        tabs.addTab(KMeansTab(MACRO_DATASET, self), "K-Means Clustering")
        tabs.addTab(KNNTab(MACRO_DATASET, self), "K-Nearest Neighbors (KNN)")


def apply_app_stylesheet(app: QApplication):
    app.setStyleSheet(
        """
        QWidget { color: #222222; background-color: #ffffff; }
        QMainWindow, QDialog { background-color: #ffffff; }
        QLabel { color: #222222; background-color: transparent; }
        QComboBox, QSpinBox, QTextEdit {
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
        QGroupBox {
            border: 1px solid #bbbbbb;
            border-radius: 5px;
            margin-top: 8px;
            padding-top: 10px;
            font-weight: bold;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0px 5px;
            color: #6c1d45;
        }
        QTabWidget::pane { border: 1px solid #cccccc; }
        QTabBar::tab { padding: 6px 12px; }
        QTabBar::tab:selected { background: #6c1d45; color: #ffffff; }
        """
    )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    apply_app_stylesheet(app)
    window = CobberEcoKApp()
    window.show()
    sys.exit(app.exec())
