# CobberEcoBloom.py
# A PyQt6 application to classify synthetic drone-like lake-surface images
# for algal bloom monitoring using a pre-trained CNN.
#
# Ecology adaptation of CobberSorter.
#
# The app loads:
#   - a trained Keras model, for example:
#       cobber_bloom_model_2000.keras
#   - a label encoder, for example:
#       cobber_bloom_label_encoder_2000.pkl
#
# Then it generates synthetic lake images on the fly and asks the model to sort them.
#
# Dependencies:
#   pip install PyQt6 numpy opencv-python tensorflow scikit-learn matplotlib
#
# Run:
#   python CobberEcoBloom.py

from __future__ import annotations

import sys
import os
import pickle
import random
import time
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QFileDialog, QVBoxLayout, QHBoxLayout, QMessageBox,
    QProgressBar, QTextEdit, QStackedWidget
)
from PyQt6.QtGui import QPixmap, QImage, QFont, QColor, QTextCursor
from PyQt6.QtCore import Qt, pyqtSignal, QObject, QRunnable, QThreadPool

import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas


# ----------------------------
# Path helpers
# ----------------------------
def app_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


APP_ROOT = app_root()


# ----------------------------
# Synthetic bloom image engine
# ----------------------------
CATEGORIES: Dict[str, Tuple[float, float]] = {
    "clear_water": (0.00, 0.04),
    "mostly_clear": (0.06, 0.15),
    "field_check": (0.25, 0.75),
    "mostly_bloom": (0.85, 0.94),
    "dense_bloom": (0.96, 1.00),
}

CATEGORY_ORDER = [
    "clear_water",
    "mostly_clear",
    "field_check",
    "mostly_bloom",
    "dense_bloom",
]


def make_water_background(size: int, rng: np.random.Generator) -> np.ndarray:
    """Create a synthetic RGB lake-water background."""
    y = np.linspace(0, 1, size)[:, None]
    x = np.linspace(0, 1, size)[None, :]

    base_r = 22 + 8 * y + 4 * np.sin(2 * np.pi * (x * 2.2 + y * 0.4))
    base_g = 92 + 20 * y + 8 * np.sin(2 * np.pi * (x * 1.4 - y * 0.7))
    base_b = 145 + 35 * (1 - y) + 8 * np.sin(2 * np.pi * (x * 1.8 + y * 1.1))

    img = np.dstack([base_r, base_g, base_b]).astype(np.float32)
    img += rng.normal(0, 5, (size, size, 3))

    # Gentle wind-like streaks.
    for _ in range(int(rng.integers(8, 18))):
        yy = int(rng.integers(0, size))
        x0 = int(rng.integers(0, size // 2))
        x1 = int(rng.integers(size // 2, size))
        color = (35, 120, 170)
        thickness = int(rng.integers(1, 3))
        cv2.line(img, (x0, yy), (x1, yy + int(rng.integers(-8, 8))), color, thickness)

    return np.clip(img, 0, 255).astype(np.uint8)


def generate_blob_mask(size: int, target_coverage: float, rng: np.random.Generator) -> np.ndarray:
    """Generate a patchy bloom mask with approximately the requested coverage."""
    if target_coverage <= 0.005:
        return np.zeros((size, size), dtype=np.uint8)

    if target_coverage >= 0.98:
        base = np.ones((size, size), dtype=np.float32)
        for _ in range(int(rng.integers(5, 14))):
            center = (int(rng.integers(0, size)), int(rng.integers(0, size)))
            axes = (int(rng.integers(8, 30)), int(rng.integers(8, 30)))
            angle = float(rng.uniform(0, 180))
            cv2.ellipse(base, center, axes, angle, 0, 360, 0.0, -1)
        base = cv2.GaussianBlur(base, (0, 0), sigmaX=5, sigmaY=5)
        threshold = np.quantile(base, 1.0 - target_coverage)
        return (base > threshold).astype(np.uint8)

    field = np.zeros((size, size), dtype=np.float32)
    n_blobs = int(np.interp(target_coverage, [0.02, 0.95], [3, 42]))
    n_blobs = max(1, n_blobs)

    for _ in range(n_blobs):
        center = (int(rng.integers(0, size)), int(rng.integers(0, size)))
        min_axis = int(np.interp(target_coverage, [0.02, 0.95], [7, 16]))
        max_axis = int(np.interp(target_coverage, [0.02, 0.95], [22, 52]))
        axes = (
            int(rng.integers(min_axis, max_axis + 1)),
            int(rng.integers(min_axis, max_axis + 1)),
        )
        angle = float(rng.uniform(0, 180))
        intensity = float(rng.uniform(0.5, 1.0))
        cv2.ellipse(field, center, axes, angle, 0, 360, intensity, -1)

    texture = rng.normal(0, 1, (size, size)).astype(np.float32)
    texture = cv2.GaussianBlur(texture, (0, 0), sigmaX=7, sigmaY=7)
    field = cv2.GaussianBlur(field, (0, 0), sigmaX=5, sigmaY=5)
    field = field + 0.45 * texture

    threshold = np.quantile(field, 1.0 - target_coverage)
    mask = (field > threshold).astype(np.uint8)

    mask_blur = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigmaX=1.1, sigmaY=1.1)
    mask = (mask_blur > 0.35).astype(np.uint8)

    return mask


def add_bloom_to_water(water: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Overlay green bloom patches on a water background."""
    img = water.astype(np.float32).copy()

    h, w = mask.shape
    bloom_color = np.zeros((h, w, 3), dtype=np.float32)
    bloom_color[:, :, 0] = rng.normal(65, 14, (h, w))
    bloom_color[:, :, 1] = rng.normal(150, 28, (h, w))
    bloom_color[:, :, 2] = rng.normal(55, 16, (h, w))

    streak = rng.normal(0, 1, (h, w)).astype(np.float32)
    streak = cv2.GaussianBlur(streak, (0, 0), sigmaX=8, sigmaY=2)
    bloom_color[:, :, 1] += 20 * streak
    bloom_color[:, :, 2] += 10 * streak

    alpha = (mask.astype(np.float32) * rng.uniform(0.62, 0.82))[:, :, None]
    img = (1 - alpha) * img + alpha * bloom_color

    # Occasional shoreline edge, matching the dataset generator.
    if rng.random() < 0.35:
        side = rng.choice(["top", "bottom", "left", "right"])
        shore_color = np.array([55, 75, 38], dtype=np.float32)
        thickness = int(rng.integers(8, 24))
        if side == "top":
            img[:thickness, :, :] = 0.65 * img[:thickness, :, :] + 0.35 * shore_color
        elif side == "bottom":
            img[-thickness:, :, :] = 0.65 * img[-thickness:, :, :] + 0.35 * shore_color
        elif side == "left":
            img[:, :thickness, :] = 0.65 * img[:, :thickness, :] + 0.35 * shore_color
        else:
            img[:, -thickness:, :] = 0.65 * img[:, -thickness:, :] + 0.35 * shore_color

    if rng.random() < 0.7:
        img = cv2.GaussianBlur(img, (3, 3), sigmaX=0.5)

    return np.clip(img, 0, 255).astype(np.uint8)


def make_bloom_image(size: int, coverage: float, seed: int | None = None):
    rng = np.random.default_rng(seed)
    water = make_water_background(size, rng)
    mask = generate_blob_mask(size, coverage, rng)
    actual_coverage = float(mask.mean())
    img = add_bloom_to_water(water, mask, rng)
    return img, actual_coverage


def generate_bloom_image_for_category(category: str, image_size: int):
    lo, hi = CATEGORIES[category]
    requested_coverage = random.uniform(lo, hi)
    seed = random.randint(0, 2_000_000_000)
    img, actual_coverage = make_bloom_image(image_size, requested_coverage, seed)
    return img, requested_coverage, actual_coverage


def choose_random_actual_category() -> str:
    """
    Choose classes so that extremes, near-extremes, and field-check cases
    all appear often in the classroom sorting test.
    """
    r = random.random()
    if r < 0.20:
        return "clear_water"
    if r < 0.40:
        return "dense_bloom"
    if r < 0.60:
        return "mostly_clear"
    if r < 0.80:
        return "mostly_bloom"
    return "field_check"


# ----------------------------
# Worker threads
# ----------------------------
class WorkerSignals(QObject):
    progress = pyqtSignal(int)
    finished = pyqtSignal(float, list, list)
    error = pyqtSignal(str)
    log = pyqtSignal(str, QColor)
    display_image = pyqtSignal(QImage)
    single_result = pyqtSignal(str, str)


class BloomSorterWorker(QRunnable):
    def __init__(self, model, le, num_iterations=200):
        super().__init__()
        self.model = model
        self.le = le
        self.signals = WorkerSignals()
        self.num_iterations = num_iterations
        self.is_running = True
        self.y_true = []
        self.y_pred = []

        self.model_image_size = self._infer_model_image_size()

    def _infer_model_image_size(self) -> int:
        try:
            shape = self.model.input_shape
            if isinstance(shape, list):
                shape = shape[0]
            if shape[1] is not None:
                return int(shape[1])
        except Exception:
            pass
        return 128

    def run(self):
        serious_errors = 0

        try:
            for i in range(self.num_iterations):
                if not self.is_running:
                    break

                actual_bin = choose_random_actual_category()
                img_np, requested_cov, actual_cov = generate_bloom_image_for_category(
                    actual_bin,
                    self.model_image_size,
                )

                prediction = self.predict_class(img_np)

                if self.num_iterations == 1:
                    self.signals.single_result.emit(actual_bin, prediction)
                else:
                    self.y_true.append(actual_bin)
                    self.y_pred.append(prediction)

                log_color = QColor("green")
                error_type = "correct"

                if actual_bin != prediction:
                    if self.is_bad_error(actual_bin, prediction):
                        serious_errors += 1
                        log_color = QColor("red")
                        error_type = "serious error"
                    else:
                        log_color = QColor("blue")
                        error_type = "minor category mix-up"

                self.signals.log.emit(
                    (
                        f"Image {i + 1}: "
                        f"Actual='{actual_bin}', Predicted='{prediction}', "
                        f"Coverage={actual_cov:.2f}, {error_type}"
                    ),
                    log_color,
                )

                q_img = self.numpy_rgb_to_qimage(img_np)
                self.signals.display_image.emit(q_img)

                if self.num_iterations > 1:
                    self.signals.progress.emit(int(((i + 1) / self.num_iterations) * 100))
                    time.sleep(0.05)

            if self.num_iterations > 1:
                error_rate = (serious_errors / max(1, self.num_iterations)) * 100.0
                self.signals.finished.emit(error_rate, self.y_true, self.y_pred)

        except Exception as exc:
            self.signals.error.emit(f"An error occurred in the bloom-sorting thread: {exc}")

    def predict_class(self, img_rgb: np.ndarray) -> str:
        img_resized = cv2.resize(img_rgb, (self.model_image_size, self.model_image_size))
        img_normalized = img_resized.astype(np.float32) / 255.0
        img_input = np.expand_dims(img_normalized, axis=0)

        preds = self.model.predict(img_input, verbose=0)
        pred_class_idx = int(np.argmax(preds, axis=1)[0])
        return self.le.inverse_transform([pred_class_idx])[0]

    def numpy_rgb_to_qimage(self, img_rgb: np.ndarray) -> QImage:
        img_rgb = np.ascontiguousarray(img_rgb)
        height, width, channels = img_rgb.shape
        bytes_per_line = channels * width
        q_img = QImage(
            img_rgb.data,
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_RGB888,
        )
        return q_img.copy()

    def is_bad_error(self, actual: str, predicted: str) -> bool:
        """
        Serious ecological sorting errors.

        Minor examples:
            clear_water <-> mostly_clear
            dense_bloom <-> mostly_bloom

        Serious examples:
            clear/mostly_clear predicted as bloom/mostly_bloom
            bloom/mostly_bloom predicted as clear/mostly_clear
            field_check not predicted as field_check
            non-field-check predicted as field_check
        """
        clear_types = {"clear_water", "mostly_clear"}
        bloom_types = {"dense_bloom", "mostly_bloom"}

        if actual == predicted:
            return False

        if actual == "field_check" or predicted == "field_check":
            return True

        if actual in clear_types and predicted in bloom_types:
            return True

        if actual in bloom_types and predicted in clear_types:
            return True

        return False

    def stop(self):
        self.is_running = False


# ----------------------------
# Main Application
# ----------------------------
class CobberEcoBloomApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.cobber_maroon = QColor(108, 29, 69)
        self.cobber_gold = QColor(234, 170, 0)
        self.lato_font = QFont("Lato")

        self.setWindowTitle("CobberEcoBloom")
        self.setGeometry(100, 100, 1100, 700)
        self.setFont(self.lato_font)

        self.model = None
        self.le = None
        self.cm_y_true = []
        self.cm_y_pred = []
        self.threadpool = QThreadPool()

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        controls_layout = QHBoxLayout()

        self.load_model_button = QPushButton("Load Trained Model")
        self.inspect_batch_button = QPushButton("Inspect Single Lake Image")
        self.run_sort_button = QPushButton("Run Full Bloom Survey (200 Images)")

        self.inspect_batch_button.setEnabled(False)
        self.run_sort_button.setEnabled(False)

        controls_layout.addWidget(self.load_model_button)
        controls_layout.addWidget(self.inspect_batch_button)
        controls_layout.addWidget(self.run_sort_button)

        display_layout = QHBoxLayout()

        image_area_layout = QVBoxLayout()

        self.display_stack = QStackedWidget()

        self.image_label = QLabel("Load a model to begin.")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setFixedSize(512, 512)
        self.image_label.setStyleSheet("border: 2px solid #ccc; background-color: #f0f0f0;")

        self.cm_canvas = FigureCanvas(Figure(figsize=(6, 6)))

        self.display_stack.addWidget(self.image_label)
        self.display_stack.addWidget(self.cm_canvas)

        image_area_layout.addWidget(self.display_stack)

        self.cm_button = QPushButton("Show Confusion Matrix")
        self.cm_button.setEnabled(False)
        image_area_layout.addWidget(self.cm_button)

        log_v_layout = QVBoxLayout()
        log_label = QLabel("<b>Classification Log:</b>")
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Courier", 9))

        log_v_layout.addWidget(log_label)
        log_v_layout.addWidget(self.log_text)

        display_layout.addLayout(image_area_layout)
        display_layout.addLayout(log_v_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)

        main_layout.addLayout(controls_layout)
        main_layout.addLayout(display_layout)
        main_layout.addWidget(self.progress_bar)

        self.load_model_button.clicked.connect(self.load_trained_model)
        self.inspect_batch_button.clicked.connect(self.inspect_batch)
        self.run_sort_button.clicked.connect(self.run_sort)
        self.cm_button.clicked.connect(self.toggle_cm_view)

    def load_trained_model(self):
        default_dir = str(APP_ROOT)
        model_file, _ = QFileDialog.getOpenFileName(
            self,
            "Select Trained Bloom Model",
            default_dir,
            "Keras Model Files (*.keras *.h5)"
        )
        if not model_file:
            return

        le_file, _ = QFileDialog.getOpenFileName(
            self,
            "Select Label Encoder",
            os.path.dirname(model_file),
            "Pickle Files (*.pkl)"
        )
        if not le_file:
            QMessageBox.warning(
                self,
                "Load Canceled",
                "A label encoder file (.pkl) is required to proceed."
            )
            return

        try:
            self.model = load_model(model_file)
            with open(le_file, "rb") as f:
                self.le = pickle.load(f)

            self.log_text.setText(
                f"Successfully loaded model:\n"
                f"{os.path.basename(model_file)}\n\n"
                f"Successfully loaded label encoder:\n"
                f"{os.path.basename(le_file)}\n\n"
                f"Classes: {', '.join(self.le.classes_)}\n\n"
            )

            self.inspect_batch_button.setEnabled(True)
            self.run_sort_button.setEnabled(True)
            self.cm_button.setEnabled(False)

            self.cm_y_true = []
            self.cm_y_pred = []

            self.display_stack.setCurrentWidget(self.image_label)
            self.cm_button.setText("Show Confusion Matrix")
            self.image_label.setText("Model loaded. Inspect a lake image or run a full bloom survey.")

            QMessageBox.information(self, "Success", "Model and label encoder loaded successfully.")

        except Exception as exc:
            QMessageBox.critical(self, "Loading Error", f"Failed to load files: {exc}")
            self.model = None
            self.le = None

    def inspect_batch(self):
        if not self.model or not self.le:
            return

        self.display_stack.setCurrentWidget(self.image_label)
        self.cm_button.setText("Show Confusion Matrix")
        self.set_buttons_enabled(False)

        worker = BloomSorterWorker(self.model, self.le, num_iterations=1)
        worker.signals.log.connect(self.log_message)
        worker.signals.display_image.connect(self.display_image)
        worker.signals.single_result.connect(self.handle_single_result)
        worker.signals.error.connect(self.sorting_error)
        self.threadpool.start(worker)

    def handle_single_result(self, actual, predicted):
        self.cm_y_true.append(actual)
        self.cm_y_pred.append(predicted)
        self.cm_button.setEnabled(True)
        self.set_buttons_enabled(True)

    def run_sort(self):
        if not self.model or not self.le:
            return

        self.log_text.clear()
        self.cm_y_true = []
        self.cm_y_pred = []

        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)

        self.set_buttons_enabled(False)

        self.display_stack.setCurrentWidget(self.image_label)
        self.cm_button.setText("Show Confusion Matrix")

        self.sorter_worker = BloomSorterWorker(self.model, self.le, num_iterations=200)
        self.sorter_worker.signals.progress.connect(self.update_progress)
        self.sorter_worker.signals.log.connect(self.log_message)
        self.sorter_worker.signals.display_image.connect(self.display_image)
        self.sorter_worker.signals.finished.connect(self.sorting_finished)
        self.sorter_worker.signals.error.connect(self.sorting_error)

        self.threadpool.start(self.sorter_worker)

    def sorting_finished(self, error_rate, y_true, y_pred):
        self.progress_bar.setVisible(False)
        self.set_buttons_enabled(True)

        self.cm_y_true = y_true
        self.cm_y_pred = y_pred
        self.cm_button.setEnabled(True)

        final_message = (
            f"<b>Bloom Survey Complete. "
            f"Final Percent Serious Identification Error: {error_rate:.2f}%</b>"
        )
        self.log_text.append(final_message)
        QMessageBox.information(self, "Bloom Survey Complete", final_message)

    def toggle_cm_view(self):
        if not self.cm_y_true:
            QMessageBox.information(
                self,
                "No Data",
                "Please inspect at least one image to generate data for the matrix."
            )
            return

        if self.display_stack.currentWidget() == self.image_label:
            self.plot_confusion_matrix()
            self.display_stack.setCurrentWidget(self.cm_canvas)
            self.cm_button.setText("Show Image")
        else:
            self.display_stack.setCurrentWidget(self.image_label)
            self.cm_button.setText("Show Confusion Matrix")

    def plot_confusion_matrix(self):
        if not self.cm_y_true or not self.le:
            return

        labels = list(self.le.classes_)
        cm = confusion_matrix(self.cm_y_true, self.cm_y_pred, labels=labels)

        fig = self.cm_canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)

        im = ax.imshow(cm, cmap="Blues")
        ax.set_title("Confusion Matrix", fontsize=14)
        ax.set_ylabel("Actual Class", fontsize=12)
        ax.set_xlabel("Predicted Class", fontsize=12)

        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        self.cm_canvas.draw()

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def log_message(self, text, color):
        self.log_text.setTextColor(color)
        self.log_text.append(text)
        self.log_text.moveCursor(QTextCursor.MoveOperation.End)

    def display_image(self, q_img):
        self.image_label.setPixmap(
            QPixmap.fromImage(q_img).scaled(
                512,
                512,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def sorting_error(self, error_message):
        self.progress_bar.setVisible(False)
        self.set_buttons_enabled(True)
        self.log_text.append(f"<font color='red'><b>CRITICAL ERROR:</b> {error_message}</font>")
        QMessageBox.critical(self, "Sorting Error", f"A critical error occurred: {error_message}")

    def set_buttons_enabled(self, enabled):
        is_model_loaded = self.model is not None
        self.load_model_button.setEnabled(enabled)
        self.inspect_batch_button.setEnabled(enabled and is_model_loaded)
        self.run_sort_button.setEnabled(enabled and is_model_loaded)

    def closeEvent(self, event):
        if hasattr(self, "sorter_worker"):
            self.sorter_worker.stop()
        self.threadpool.waitForDone()
        event.accept()


def apply_app_stylesheet(app: QApplication) -> None:
    app.setStyleSheet(
        """
        QWidget { color: #222222; background-color: #ffffff; }
        QMainWindow, QDialog { background-color: #ffffff; }
        QLabel { color: #222222; background-color: transparent; }
        QTextEdit {
            background-color: #ffffff;
            color: #111111;
            border: 1px solid #a0a0a0;
            border-radius: 3px;
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
        QProgressBar {
            border: 1px solid #9a9a9a;
            border-radius: 4px;
            text-align: center;
            background: #ffffff;
        }
        QProgressBar::chunk {
            background-color: #6c1d45;
        }
        """
    )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    apply_app_stylesheet(app)
    window = CobberEcoBloomApp()
    window.show()
    sys.exit(app.exec())
