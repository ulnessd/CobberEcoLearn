# CobberEcoCNN_v1.py
# An ecology-inspired adaptation of CobberCNN for exploring convolutional
# neural-network ideas using simplified animal-track glyphs.
#
# The program intentionally keeps the same look, feel, and classroom workflow
# as the chemistry CobberCNN program:
#   1. choose a simple input image,
#   2. choose a 3x3 filter,
#   3. watch the convolution scan,
#   4. inspect the feature map,
#   5. optionally apply filters to feature maps,
#   6. toggle a matrix/numeric view.

import sys
import os
import numpy as np
from typing import Dict

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QFrame, QComboBox, QListWidget, QPushButton, QTabWidget,
    QMessageBox, QFileDialog
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QColor

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle


# ---------------------------------------------------------------------
# Stylized animal-track glyphs
# ---------------------------------------------------------------------
# These are intentionally simple 11x11 binary images. They are not intended
# to be forensic-quality animal tracks. They are track-inspired teaching
# glyphs designed to preserve sharp lines, junctions, diagonals, and repeated
# local features so that convolutional filters are easy to interpret.

IMAGES: Dict[str, np.ndarray] = {
    "Hoof Track": np.zeros((11, 11), dtype=float),
    "Bird Track": np.zeros((11, 11), dtype=float),
    "Paw Track": np.zeros((11, 11), dtype=float),
}

# Hoof Track: two elongated hoof marks with slight separation.
# Good for vertical and edge-like responses.
hoof = IMAGES["Hoof Track"]
hoof[2:9, 3] = 1
hoof[2:9, 4] = 1
hoof[2:9, 6] = 1
hoof[2:9, 7] = 1
hoof[2, 4] = 0
hoof[2, 6] = 0
hoof[8, 3] = 0
hoof[8, 7] = 0

# Bird Track: three forward toes plus a small rear toe.
# Good for diagonal and junction responses.
bird = IMAGES["Bird Track"]
bird[6, 5] = 1
bird[5, 5] = 1
bird[2:6, 5] = 1
for y, x in [(5, 4), (4, 3), (3, 2)]:
    bird[y, x] = 1
for y, x in [(5, 6), (4, 7), (3, 8)]:
    bird[y, x] = 1
bird[7:10, 5] = 1

# Paw Track: central pad plus toe marks.
# Good for local spot/cluster responses and repeated-feature thinking.
paw = IMAGES["Paw Track"]
paw[6:9, 4:7] = 1
paw[5, 5] = 1
paw[8, 4] = 0
paw[8, 6] = 0
paw[2:4, 2:4] = 1
paw[1:3, 4:6] = 1
paw[2:4, 7:9] = 1
paw[4:6, 1:3] = 1
paw[4:6, 8:10] = 1


FILTERS = {
    "Vertical Line": np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=float),
    "Horizontal Line": np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=float),
    "Junction Detector": np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=float),
    "Diagonal (45\u00b0)": np.array([[-1, -1, 2], [-1, 2, -1], [2, -1, -1]], dtype=float),
}


def apply_convolution(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    h, w = image.shape
    kh, kw = kernel.shape
    out = np.zeros((h - kh + 1, w - kw + 1))
    for y in range(out.shape[0]):
        for x in range(out.shape[1]):
            out[y, x] = np.sum(image[y:y + kh, x:x + kw] * kernel)
    return out


def relu(x):
    return np.maximum(0, x)


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=5, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.fig.tight_layout()


class CobberEcoCNNApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.cobber_maroon = QColor(108, 29, 69)
        self.cobber_gold = QColor(234, 170, 0)
        self.lato_font = QFont("Lato")

        self.setWindowTitle("CobberEcoCNN")
        self.setGeometry(100, 100, 1400, 600)
        self.setFont(self.lato_font)

        self.current_input_image = IMAGES["Hoof Track"]
        self.feature_maps: Dict[str, np.ndarray] = {}
        self.animation_timer = QTimer(self)
        self.animation_timer.timeout.connect(self.animation_step)
        self.show_matrix_view = False

        main_layout = QHBoxLayout()

        controls_panel = QFrame()
        controls_panel.setFrameShape(QFrame.Shape.StyledPanel)
        controls_layout = QVBoxLayout(controls_panel)

        process_panel = QFrame()
        process_panel.setFrameShape(QFrame.Shape.StyledPanel)
        process_layout = QVBoxLayout(process_panel)

        self.output_tabs = QTabWidget()

        controls_layout.addWidget(QLabel("<h3>1. Track Selection</h3>"))
        self.track_selector = QComboBox()
        self.track_selector.addItems(IMAGES.keys())
        controls_layout.addWidget(self.track_selector)

        self.input_image_canvas = MplCanvas(self, width=2, height=2, dpi=70)
        controls_layout.addWidget(self.input_image_canvas)

        controls_layout.addWidget(QLabel("<h3>2. Filter Library</h3>"))
        self.filter_list = QListWidget()
        self.filter_list.addItems(FILTERS.keys())
        controls_layout.addWidget(self.filter_list)

        self.apply_to_input_button = QPushButton("Apply to Input Image")
        self.apply_to_feature_map_button = QPushButton("Apply to Current Feature Map")
        self.toggle_matrix_btn = QPushButton("Toggle Matrix View")
        self.toggle_matrix_btn.setCheckable(True)
        self.toggle_matrix_btn.toggled.connect(self.toggle_matrix_view)

        controls_layout.addWidget(self.apply_to_input_button)
        controls_layout.addWidget(self.apply_to_feature_map_button)
        controls_layout.addWidget(self.toggle_matrix_btn)

        self.save_button = QPushButton("Save All Open Maps")
        self.clear_button = QPushButton("Clear/Reset All")
        controls_layout.addWidget(self.save_button)
        controls_layout.addWidget(self.clear_button)

        controls_layout.addStretch()

        process_layout.addWidget(QLabel("<h3>Convolution Process</h3>"))
        self.process_canvas = MplCanvas(self)
        process_layout.addWidget(self.process_canvas)

        main_layout.addWidget(controls_panel, 2)
        main_layout.addWidget(process_panel, 3)
        main_layout.addWidget(self.output_tabs, 3)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        self.track_selector.currentTextChanged.connect(self.update_input_image)
        self.apply_to_input_button.clicked.connect(self.start_convolution_on_input)
        self.apply_to_feature_map_button.clicked.connect(self.start_convolution_on_feature_map)
        self.save_button.clicked.connect(self.save_all_maps)
        self.clear_button.clicked.connect(self.clear_all_tabs)

        self.update_input_image("Hoof Track")

    def toggle_matrix_view(self, checked):
        self.show_matrix_view = checked
        index = self.output_tabs.currentIndex()
        if index >= 0:
            tab = self.output_tabs.widget(index)
            name = self.output_tabs.tabText(index)
            data = self.feature_maps.get(name)
            if data is not None:
                self.draw_image(tab, data, f"Output: {name}")

        self.draw_image(self.input_image_canvas, self.current_input_image, f"Input: {self.track_selector.currentText()}")
        self.draw_image(self.process_canvas, self.current_input_image, "Ready to Convolve")

    def update_input_image(self, name):
        self.current_input_image = IMAGES[name]
        self.draw_image(self.input_image_canvas, self.current_input_image, f"Input: {name}")
        self.draw_image(self.process_canvas, self.current_input_image, "Ready to Convolve")

    def start_convolution_on_input(self):
        self.start_animation(self.current_input_image, self.track_selector.currentText())

    def start_convolution_on_feature_map(self):
        idx = self.output_tabs.currentIndex()
        if idx < 0:
            QMessageBox.warning(self, "No Feature Map", "Please generate a feature map first.")
            return

        name = self.output_tabs.tabText(idx)
        fmap = self.feature_maps.get(name)
        if fmap is not None:
            self.start_animation(fmap, name)

    def start_animation(self, image, label):
        item = self.filter_list.currentItem()
        if not item:
            QMessageBox.warning(self, "No Filter Selected", "Please select a filter.")
            return

        if image.shape[0] < 3 or image.shape[1] < 3:
            QMessageBox.warning(self, "Image Too Small", "This feature map is too small for a 3x3 filter.")
            return

        self.set_buttons_enabled(False)

        self.anim_input_image = image
        self.anim_kernel = FILTERS[item.text()]
        self.anim_pos = (0, 0)

        h, w = image.shape[0] - 2, image.shape[1] - 2
        self.anim_feature_map = np.zeros((h, w))

        tab_name = f"{label} + {item.text()}"
        self.current_feature_map_canvas = MplCanvas(self)
        self.output_tabs.addTab(self.current_feature_map_canvas, tab_name)
        self.output_tabs.setCurrentWidget(self.current_feature_map_canvas)

        self.draw_image(self.process_canvas, self.anim_input_image, "Convolving...")
        self.animation_timer.start(30)

    def animation_step(self):
        y, x = self.anim_pos

        val = relu(np.sum(self.anim_input_image[y:y + 3, x:x + 3] * self.anim_kernel))
        self.anim_feature_map[y, x] = val

        self.draw_image(self.process_canvas, self.anim_input_image, "Convolving...", (x, y))
        self.draw_image(self.current_feature_map_canvas, self.anim_feature_map, "Building Feature Map...")

        x += 1
        if x >= self.anim_feature_map.shape[1]:
            x = 0
            y += 1

        if y >= self.anim_feature_map.shape[0]:
            self.animation_timer.stop()
            self.finalize_convolution()
        else:
            self.anim_pos = (y, x)

    def finalize_convolution(self):
        name = self.output_tabs.tabText(self.output_tabs.currentIndex())
        self.feature_maps[name] = self.anim_feature_map
        self.draw_image(self.current_feature_map_canvas, self.anim_feature_map, f"Output: {name}")
        self.draw_image(self.process_canvas, self.anim_input_image, "Convolution Complete")
        self.set_buttons_enabled(True)

    def draw_image(self, canvas, data, title, scan_box_pos=None):
        canvas.axes.clear()

        cmap = "viridis"
        vmin, vmax = 0, max(1.0, np.max(data))

        if title.startswith("Input") or "Convolve" in title or title == "Ready to Convolve":
            cmap, vmin, vmax = "gray_r", 0, 1.0

        canvas.axes.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
        canvas.axes.set_title(title, fontsize=10)
        canvas.axes.set_xticks([])
        canvas.axes.set_yticks([])

        if self.show_matrix_view:
            h, w = data.shape
            for y in range(h):
                for x in range(w):
                    val = data[y, x]
                    canvas.axes.add_patch(
                        Rectangle(
                            (x - 0.5, y - 0.5),
                            1,
                            1,
                            facecolor="white",
                            edgecolor="black",
                            lw=1,
                        )
                    )
                    canvas.axes.text(
                        x,
                        y,
                        f"{val:.0f}",
                        ha="center",
                        va="center",
                        fontsize=18 if max(h, w) <= 11 else 10,
                    )

        if scan_box_pos:
            canvas.axes.add_patch(
                Rectangle(
                    (scan_box_pos[0] - 0.5, scan_box_pos[1] - 0.5),
                    3,
                    3,
                    edgecolor="red",
                    facecolor="red",
                    alpha=0.3,
                )
            )

        canvas.draw()

    def set_buttons_enabled(self, state):
        self.apply_to_input_button.setEnabled(state)
        self.apply_to_feature_map_button.setEnabled(state)
        self.save_button.setEnabled(state)
        self.clear_button.setEnabled(state)

    def clear_all_tabs(self):
        if QMessageBox.question(
            self,
            "Confirm Clear",
            "Clear all generated feature maps?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        ) == QMessageBox.StandardButton.Yes:
            while self.output_tabs.count():
                self.output_tabs.removeTab(0)
            self.feature_maps.clear()

    def save_all_maps(self):
        if self.output_tabs.count() == 0:
            QMessageBox.warning(self, "No Maps to Save", "Generate at least one feature map first.")
            return

        directory = QFileDialog.getExistingDirectory(self, "Select Save Location")
        if directory:
            saved = 0
            for i in range(self.output_tabs.count()):
                widget = self.output_tabs.widget(i)
                if isinstance(widget, MplCanvas):
                    name = (
                        self.output_tabs.tabText(i)
                        .replace(" + ", "_")
                        .replace(" ", "")
                        .replace("/", "_")
                    )
                    widget.fig.savefig(os.path.join(directory, f"{name}.png"), dpi=300)
                    saved += 1
            QMessageBox.information(self, "Saved", f"{saved} map(s) saved to:\n{directory}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CobberEcoCNNApp()
    window.show()
    sys.exit(app.exec())
