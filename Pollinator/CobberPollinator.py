# CobberPollinator.py
# Enlarges the vector scene for better alignment and adds pollination interpretation to Results.

from __future__ import annotations

import os
import random
import sys
from functools import lru_cache
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, Rectangle
from PIL import Image
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.path.abspath('.')

ASSET_DIR = os.path.join(SCRIPT_DIR, 'imagesets')

PLANT_STYLES = {
    "Composite Daisy": {"petal": "#f7d44c", "center": "#8b5e34", "petals": 10, "elongation": (0.52, 0.22), "hotspot": (0.20, 0.10)},
    "Deep Tube": {"petal": "#b04bd6", "center": "#f2d0ff", "petals": 6, "elongation": (0.65, 0.18), "hotspot": (0.05, -0.25)},
    "Bell Flower": {"petal": "#5c7cfa", "center": "#dbe4ff", "petals": 5, "elongation": (0.58, 0.24), "hotspot": (-0.10, -0.20)},
    "Flat Cluster": {"petal": "#f29e4c", "center": "#fff2cc", "petals": 8, "elongation": (0.40, 0.18), "hotspot": (0.00, 0.05)},
    "Pea Flower": {"petal": "#ef476f", "center": "#ffd6df", "petals": 5, "elongation": (0.58, 0.26), "hotspot": (0.22, -0.08)},
    "Star Flower": {"petal": "#e8e8e8", "center": "#f1c40f", "petals": 5, "elongation": (0.64, 0.14), "hotspot": (-0.15, 0.12)},
}

POLLINATOR_STYLES = {
    "Bumblebee": {"body": "#222222", "accent": "#f1c40f", "size": 0.15},
    "Honeybee": {"body": "#6b4423", "accent": "#f7c948", "size": 0.12},
    "Solitary Bee": {"body": "#6a4c93", "accent": "#ffca3a", "size": 0.10},
    "Hoverfly": {"body": "#364fc7", "accent": "#f08c00", "size": 0.11},
    "Butterfly": {"body": "#495057", "accent": "#ff6b6b", "size": 0.16},
    "Beetle": {"body": "#1b4332", "accent": "#74c69d", "size": 0.13},
}

FLOWER_VISUAL_SCALE = 1.45
CONTACT_ZONE_RADIUS = 0.18


PLANT_FILE_CANDIDATES = {
    'Composite Daisy': ['Daisy.png', 'CompositeDaisy.png'],
    'Deep Tube': ['DeepTube.png', 'TrumpetFlower.png', 'TubeFlower.png'],
    'Bell Flower': ['BellFlower.png', 'Bluebell.png'],
    'Flat Cluster': ['FlatCluster.png', 'FloralCluster.png'],
    'Pea Flower': ['PeaFlower.png', 'PinkPea.png'],
    'Star Flower': ['StarFlower.png', 'StarShape.png'],
}

POLLINATOR_FILE_CANDIDATES = {
    'Bumblebee': ['Bumblebee.png', 'BumbleBee.png'],
    'Honeybee': ['Honeybee.png', 'HoneyBee.png'],
    'Solitary Bee': ['SolitaryBee.png', 'Solitary_Bee.png'],
    'Hoverfly': ['Hoverfly.png', 'HoverFly.png'],
    'Butterfly': ['Butterfly.png', 'ButterFly.png'],
    'Beetle': ['Beetle.png', 'Beatlle.png'],
}


def _resolve_asset_path(candidates: list[str]) -> Optional[str]:
    for name in candidates:
        path = os.path.join(ASSET_DIR, name)
        if os.path.exists(path):
            return path
    return None


class AssetManager:
    @staticmethod
    def plant_asset_path(plant_type: str) -> Optional[str]:
        return _resolve_asset_path(PLANT_FILE_CANDIDATES.get(plant_type, []))

    @staticmethod
    def pollinator_asset_path(pollinator_type: str) -> Optional[str]:
        return _resolve_asset_path(POLLINATOR_FILE_CANDIDATES.get(pollinator_type, []))

    @staticmethod
    @lru_cache(maxsize=128)
    def load_rgba_image(path: str, target_px: int) -> np.ndarray:
        """Load, downsample, and cache an RGBA image for repeated plotting.

        Images are resized only if larger than target_px in either dimension.
        This keeps the original files intact but reduces redraw cost.
        """
        img = Image.open(path).convert('RGBA')
        w, h = img.size
        scale = min(target_px / max(w, h), 1.0)
        if scale < 1.0:
            new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        return np.asarray(img)

    @staticmethod
    def get_image(path: Optional[str], target_px: int) -> Optional[np.ndarray]:
        if not path:
            return None
        try:
            return AssetManager.load_rgba_image(path, target_px)
        except Exception:
            return None


class PollinatorVisualizer:
    @staticmethod
    def _draw_flower_vector(ax, plant_type: str):
        style = PLANT_STYLES.get(plant_type, PLANT_STYLES['Composite Daisy'])
        petal_color = style['petal']
        center_color = style['center']
        n_petals = style['petals']
        w, h = style['elongation']
        scale = FLOWER_VISUAL_SCALE

        for i in range(n_petals):
            theta = 2 * np.pi * i / n_petals
            cx = 0.45 * scale * np.cos(theta)
            cy = 0.45 * scale * np.sin(theta)
            petal = Ellipse(
                (cx, cy),
                width=w * scale,
                height=h * scale,
                angle=np.degrees(theta),
                facecolor=petal_color,
                edgecolor='black',
                lw=1.0,
                alpha=0.95,
            )
            ax.add_patch(petal)

        center = Circle((0, 0), radius=0.24 * scale, facecolor=center_color, edgecolor='black', lw=1.4, zorder=4)
        ax.add_patch(center)
        ax.plot([0, 0], [-0.24 * scale, -1.15], color='#2b9348', lw=4, solid_capstyle='round', zorder=1)
        ax.plot([0, -0.25 * scale], [-0.65, -0.95], color='#55a630', lw=3, solid_capstyle='round', zorder=1)
        ax.plot([0, 0.22 * scale], [-0.78, -1.02], color='#55a630', lw=3, solid_capstyle='round', zorder=1)

    @staticmethod
    def _draw_flower(ax, plant_type: str):
        PollinatorVisualizer._draw_flower_vector(ax, plant_type)

        hotspot = PLANT_STYLES.get(plant_type, PLANT_STYLES['Composite Daisy']).get('hotspot', (0.0, 0.0))
        hotspot_patch = Circle(hotspot, radius=CONTACT_ZONE_RADIUS, facecolor='none', edgecolor='#d62828', lw=1.6,
                               linestyle='--', zorder=5)
        ax.add_patch(hotspot_patch)
        ax.text(hotspot[0], hotspot[1] + 0.18, 'contact zone', color='#d62828', fontsize=8,
                ha='center', va='bottom', zorder=6)

    @staticmethod
    def _draw_pollinator_vector(ax, pollinator_type: str, landing_x: float, landing_y: float, landing_angle_deg: float):
        style = POLLINATOR_STYLES.get(pollinator_type, POLLINATOR_STYLES['Honeybee'])
        body_color = style['body']
        accent_color = style['accent']
        size = style['size']

        body = Ellipse((landing_x, landing_y), width=size * 1.7, height=size * 0.95, angle=landing_angle_deg,
                       facecolor=body_color, edgecolor='black', lw=1.2, zorder=8)
        ax.add_patch(body)
        stripe = Ellipse((landing_x, landing_y), width=size * 1.05, height=size * 0.50, angle=landing_angle_deg,
                         facecolor=accent_color, edgecolor='none', alpha=0.9, zorder=9)
        ax.add_patch(stripe)

        theta = np.deg2rad(landing_angle_deg)
        wing_offset = size * 0.42
        wx = wing_offset * np.cos(theta + np.pi / 2)
        wy = wing_offset * np.sin(theta + np.pi / 2)
        wing1 = Ellipse((landing_x + wx, landing_y + wy), width=size * 1.15, height=size * 0.62,
                        angle=landing_angle_deg + 30, facecolor='#dbeafe', edgecolor='#94a3b8',
                        alpha=0.65, lw=0.9, zorder=7)
        wing2 = Ellipse((landing_x - wx, landing_y - wy), width=size * 1.15, height=size * 0.62,
                        angle=landing_angle_deg - 30, facecolor='#dbeafe', edgecolor='#94a3b8',
                        alpha=0.65, lw=0.9, zorder=7)
        ax.add_patch(wing1)
        ax.add_patch(wing2)
        ax.plot([landing_x], [landing_y], 'ko', markersize=2.8, zorder=10)

    @staticmethod
    def _draw_pollinator(ax, pollinator_type: str, landing_x: float, landing_y: float, landing_angle_deg: float):
        theta = np.deg2rad(landing_angle_deg)
        ux, uy = np.cos(theta), np.sin(theta)
        start_x = landing_x + 0.45 * ux
        start_y = landing_y + 0.45 * uy
        arrow = FancyArrowPatch((start_x, start_y), (landing_x, landing_y), arrowstyle='->',
                                mutation_scale=12, lw=1.6, color='#495057', zorder=6)
        ax.add_patch(arrow)

        PollinatorVisualizer._draw_pollinator_vector(ax, pollinator_type, landing_x, landing_y, landing_angle_deg)
        ax.plot([landing_x], [landing_y], 'ko', markersize=2.5, zorder=10)

    @staticmethod
    def _single_visit_summary(row: pd.Series) -> tuple[str, str]:
        hotspot = PLANT_STYLES.get(row['Plant_Type'], PLANT_STYLES['Composite Daisy']).get('hotspot', (0.0, 0.0))
        dx = float(row['Landing_X']) - hotspot[0]
        dy = float(row['Landing_Y']) - hotspot[1]
        in_zone = (dx * dx + dy * dy) ** 0.5 <= CONTACT_ZONE_RADIUS
        dwell_score = min(float(row['Dwell_Time_s']) / 8.0, 1.0)
        score = (
            0.42 * float(row['Contact_Fit'])
            + 0.28 * float(row['Return_Regularity'])
            + 0.18 * dwell_score
            + 0.12 * (1.0 if in_zone else 0.0)
        )
        if score >= 0.62:
            return "Likely effective pollination", "#2b9348"
        if score >= 0.36:
            return "Possible but limited pollination", "#c77d00"
        return "Unlikely pollination", "#c1121f"

    @staticmethod
    def _draw_attribute_bars(ax, row: pd.Series):
        ax.clear()
        ax.set_xlim(0, 1.05)
        ax.set_ylim(-0.7, 3.2)
        ax.axis('off')

        metrics = [
            ('Dwell Time', min(float(row['Dwell_Time_s']) / 15.0, 1.0), f"{float(row['Dwell_Time_s']):.2f} s", '#3a86ff'),
            ('Return Regularity', float(row['Return_Regularity']), f"{float(row['Return_Regularity']):.3f}", '#8338ec'),
            ('Contact Fit', float(row['Contact_Fit']), f"{float(row['Contact_Fit']):.3f}", '#ff006e'),
        ]

        y_positions = [2.3, 1.3, 0.3]
        for (label, value, text, color), y in zip(metrics, y_positions):
            ax.text(0.02, y + 0.36, label, fontsize=10, fontweight='bold', ha='left', va='center')
            bg = Rectangle((0.02, y), 0.86, 0.22, facecolor='#e9ecef', edgecolor='#adb5bd', lw=0.8)
            fill = Rectangle((0.02, y), 0.86 * value, 0.22, facecolor=color, edgecolor='none', alpha=0.9)
            ax.add_patch(bg)
            ax.add_patch(fill)
            ax.text(0.92, y + 0.11, text, fontsize=10, ha='right', va='center')

        summary_text, summary_color = PollinatorVisualizer._single_visit_summary(row)
        ax.text(0.02, -0.05, summary_text, fontsize=10, fontweight='bold', color=summary_color, ha='left', va='center')
        ax.text(0.02, -0.25, f"Plant: {row['Plant_Type']}", fontsize=10, ha='left', va='center')
        ax.text(0.02, -0.45, f"Pollinator: {row['Pollinator_Type']}", fontsize=10, ha='left', va='center')

    @classmethod
    def plot_interaction(cls, fig: Figure, row: pd.Series):
        fig.clear()
        gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1.0])
        ax_scene = fig.add_subplot(gs[0, 0])
        ax_bars = fig.add_subplot(gs[0, 1])

        cls._draw_flower(ax_scene, row['Plant_Type'])
        cls._draw_pollinator(ax_scene, row['Pollinator_Type'], float(row['Landing_X']), float(row['Landing_Y']),
                             float(row['Landing_Angle_deg']))

        ax_scene.set_xlim(-1.35, 1.35)
        ax_scene.set_ylim(-1.35, 1.35)
        ax_scene.set_aspect('equal', 'box')
        ax_scene.set_title(f"Single Interaction View\n{row['Plant_Type']} × {row['Pollinator_Type']}")
        ax_scene.set_xticks([])
        ax_scene.set_yticks([])
        for spine in ax_scene.spines.values():
            spine.set_visible(False)

        cls._draw_attribute_bars(ax_bars, row)
        fig.tight_layout()


class DataTab(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.dataset: Optional[pd.DataFrame] = None
        self.timer = QTimer(self)
        self.features = ['Dwell_Time_s', 'Return_Regularity', 'Contact_Fit']

        self.layout = QVBoxLayout(self)
        top_controls_layout = QHBoxLayout()
        self.load_button = QPushButton('Load Dataset')
        self.load_button.clicked.connect(self.load_data)
        top_controls_layout.addWidget(self.load_button)
        self.data_info_label = QLabel('No dataset loaded.')
        self.data_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        top_controls_layout.addWidget(self.data_info_label, 1)
        self.layout.addLayout(top_controls_layout)

        main_layout = QHBoxLayout()
        left_column_layout = QVBoxLayout()

        plotting_groupbox = QGroupBox('Data Visualization')
        plotting_layout = QVBoxLayout(plotting_groupbox)
        self.plot_combo = QComboBox()
        self.plot_combo.addItems(['Histogram', '3D Scatter Plot', '2D Scatter Plot'])
        self.plot_combo.currentTextChanged.connect(self.update_plot_controls)
        plotting_layout.addWidget(self.plot_combo)
        self.plot_button = QPushButton('Plot Histogram')
        self.plot_button.clicked.connect(self.plot_data)
        plotting_layout.addWidget(self.plot_button)

        self.scatter_controls = QWidget()
        scatter_layout = QHBoxLayout(self.scatter_controls)
        scatter_layout.setContentsMargins(0, 0, 0, 0)
        self.x_axis_combo = QComboBox()
        self.y_axis_combo = QComboBox()
        scatter_layout.addWidget(QLabel('X-Axis:'))
        scatter_layout.addWidget(self.x_axis_combo)
        scatter_layout.addWidget(QLabel('Y-Axis:'))
        scatter_layout.addWidget(self.y_axis_combo)
        plotting_layout.addWidget(self.scatter_controls)
        left_column_layout.addWidget(plotting_groupbox)

        interaction_groupbox = QGroupBox('Single Visit Explorer')
        interaction_layout = QVBoxLayout(interaction_groupbox)
        self.explore_button = QPushButton('Show Single Random Visit')
        self.explore_button.clicked.connect(self.explore_visit_interaction)
        interaction_layout.addWidget(self.explore_button)
        self.animate_button = QPushButton('Animate Visit Events')
        self.animate_button.clicked.connect(self.animate_interactions)
        interaction_layout.addWidget(self.animate_button)
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel('Slower'))
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(20, 1000)
        self.speed_slider.setValue(700)
        self.speed_slider.setInvertedAppearance(True)
        speed_layout.addWidget(self.speed_slider)
        speed_layout.addWidget(QLabel('Faster'))
        interaction_layout.addLayout(speed_layout)
        left_column_layout.addWidget(interaction_groupbox)
        left_column_layout.addStretch()
        main_layout.addLayout(left_column_layout)

        self.plot_figure = Figure(figsize=(8, 7))
        self.plot_canvas = FigureCanvas(self.plot_figure)
        main_layout.addWidget(self.plot_canvas, 1)
        self.layout.addLayout(main_layout)

        self.update_plot_controls(self.plot_combo.currentText())
        self.timer.timeout.connect(self._animation_step)
        self.speed_slider.valueChanged.connect(lambda value: self.timer.setInterval(value))

    def update_plot_controls(self, text: str):
        if text == '2D Scatter Plot':
            self.scatter_controls.setVisible(True)
            self.plot_button.setText('Plot 2D Scatter')
        else:
            self.scatter_controls.setVisible(False)
            self.plot_button.setText(f'Plot {text}')

    def load_data(self):
        file_name, _ = QFileDialog.getOpenFileName(self, 'Open Dataset', SCRIPT_DIR, 'CSV Files (*.csv)')
        if file_name:
            try:
                self.dataset = pd.read_csv(file_name)
                required = {
                    'Plant_Type', 'Pollinator_Type', 'Dwell_Time_s', 'Return_Regularity', 'Contact_Fit',
                    'Landing_X', 'Landing_Y', 'Landing_Angle_deg'
                }
                missing = required - set(self.dataset.columns)
                if missing:
                    raise ValueError(f'Dataset is missing required columns: {sorted(missing)}')
                if 'Interaction_Pair' not in self.dataset.columns:
                    self.dataset['Interaction_Pair'] = self.dataset['Plant_Type'] + ' :: ' + self.dataset['Pollinator_Type']
                self.x_axis_combo.clear(); self.x_axis_combo.addItems(self.features)
                self.y_axis_combo.clear(); self.y_axis_combo.addItems(self.features)
                self.data_info_label.setText(f'Loaded: {os.path.basename(file_name)}')
                self.main_window.on_data_loaded()
            except Exception as e:
                self.dataset = None
                QMessageBox.warning(self, 'Error', f'Failed to load dataset: {e}')

    def plot_data(self):
        if self.dataset is None:
            QMessageBox.warning(self, 'Error', 'Please load a dataset first.')
            return
        self.plot_figure.clear()
        plot_type = self.plot_combo.currentText()
        if plot_type == 'Histogram':
            self.plot_histogram()
        elif plot_type == '2D Scatter Plot':
            self.plot_scatter()
        elif plot_type == '3D Scatter Plot':
            self.plot_3d_scatter()

    def plot_histogram(self):
        axes = self.plot_figure.subplots(1, len(self.features))
        if len(self.features) == 1:
            axes = [axes]
        for ax, feature in zip(axes, self.features):
            ax.hist(self.dataset[feature], bins=35, edgecolor='black')
            ax.set_title(feature)
            ax.grid(True, alpha=0.25)
        self.plot_figure.tight_layout()
        self.plot_canvas.draw()

    def _prepare_plot_df(self):
        df_plot = self.dataset.copy()
        reply = QMessageBox.question(self, 'Normalize Features', 'Plot normalized features?',
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                     QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            scaler = MinMaxScaler()
            df_plot[self.features] = scaler.fit_transform(df_plot[self.features])
        return df_plot

    def plot_scatter(self):
        ax = self.plot_figure.add_subplot(111)
        x_feature = self.x_axis_combo.currentText()
        y_feature = self.y_axis_combo.currentText()
        df_plot = self._prepare_plot_df()
        ax.scatter(df_plot[x_feature], df_plot[y_feature], alpha=0.45, s=4)
        ax.set_xlabel(x_feature)
        ax.set_ylabel(y_feature)
        ax.set_title(f'{y_feature} vs. {x_feature}')
        ax.grid(True, alpha=0.3)
        self.plot_canvas.draw()

    def plot_3d_scatter(self):
        ax = self.plot_figure.add_subplot(111, projection='3d')
        df_plot = self._prepare_plot_df()
        ax.scatter(df_plot[self.features[0]], df_plot[self.features[1]], df_plot[self.features[2]], alpha=0.45, s=3)
        ax.set_xlabel(self.features[0])
        ax.set_ylabel(self.features[1])
        ax.set_zlabel(self.features[2])
        self.plot_canvas.draw()

    def explore_visit_interaction(self):
        if self.dataset is None:
            return
        random_row = self.dataset.sample(n=1).iloc[0]
        PollinatorVisualizer.plot_interaction(self.plot_figure, random_row)
        self.plot_canvas.draw()

    def animate_interactions(self):
        if self.dataset is None:
            QMessageBox.warning(self, 'Error', 'Please load a dataset first.')
            return
        self.molecule_indices = random.sample(range(len(self.dataset)), min(100, len(self.dataset)))
        self.current_index = 0
        self.timer.start(self.speed_slider.value())

    def _animation_step(self):
        if self.current_index < len(self.molecule_indices):
            idx = self.molecule_indices[self.current_index]
            row = self.dataset.iloc[idx]
            PollinatorVisualizer.plot_interaction(self.plot_figure, row)
            self.plot_canvas.draw()
            self.current_index += 1
        else:
            self.timer.stop()
            QMessageBox.information(self, 'Animation Complete', 'Finished showing random visit events.')


class ModelTab(QWidget):
    def __init__(self, data_tab, main_window):
        super().__init__()
        self.main_window = main_window
        self.data_tab = data_tab
        self.results_tab = None
        self.trained_model = None

        self.layout = QVBoxLayout(self)
        groupbox = QGroupBox('K-Means Clustering Setup')
        box_layout = QVBoxLayout(groupbox)
        params_layout = QHBoxLayout()
        params_layout.addWidget(QLabel('Number of Clusters (k):'))
        self.num_clusters_combo = QComboBox()
        self.num_clusters_combo.addItems([str(i) for i in range(2, 11)])
        self.num_clusters_combo.setCurrentText('7')
        params_layout.addWidget(self.num_clusters_combo)
        box_layout.addLayout(params_layout)
        self.scaling_checkbox = QCheckBox('Normalize Features for Clustering')
        self.scaling_checkbox.setChecked(True)
        self.scaling_checkbox.setToolTip('It is highly recommended to keep this checked for distance-based algorithms.')
        box_layout.addWidget(self.scaling_checkbox)
        self.run_button = QPushButton('Run K-Means Clustering')
        self.run_button.clicked.connect(self.run_clustering)
        box_layout.addWidget(self.run_button)
        self.layout.addWidget(groupbox)
        self.layout.addStretch()

    def run_clustering(self):
        if self.data_tab.dataset is None:
            QMessageBox.warning(self, 'Error', 'Please load a dataset first.')
            return
        n_clusters = int(self.num_clusters_combo.currentText())
        features = self.data_tab.dataset[self.data_tab.features]
        features_scaled = features.values
        if self.scaling_checkbox.isChecked():
            scaler = MinMaxScaler()
            features_scaled = scaler.fit_transform(features)
        try:
            model = KMeans(n_clusters=n_clusters, n_init='auto', random_state=42)
            self.data_tab.dataset['Cluster'] = model.fit_predict(features_scaled)
            self.trained_model = model
            QMessageBox.information(self, 'Success',
                                    'K-Means clustering completed successfully.\nData is now ready for analysis in the Results tab.')
            if self.results_tab:
                self.results_tab.on_clustering_complete()
        except Exception as e:
            QMessageBox.critical(self, 'Clustering Error', f'An error occurred: {e}')



class ResultsTab(QWidget):
    def __init__(self, data_tab, model_tab, main_window):
        super().__init__()
        self.main_window = main_window
        self.data_tab = data_tab
        self.model_tab = model_tab
        self.selected_cluster: Optional[int] = None

        self.layout = QVBoxLayout(self)

        controls_layout = QHBoxLayout()
        controls_layout.addWidget(QLabel('Color by:'))
        self.color_by_combo = QComboBox()
        self.color_by_combo.addItems(['Cluster', 'Interaction Pair', 'Plant Type', 'Pollinator Type'])
        controls_layout.addWidget(self.color_by_combo)

        self.centroids_checkbox = QCheckBox('Show Centroids Only')
        controls_layout.addWidget(self.centroids_checkbox)

        controls_layout.addWidget(QLabel('Selected cluster:'))
        self.cluster_combo = QComboBox()
        self.cluster_combo.currentIndexChanged.connect(self.on_cluster_changed)
        controls_layout.addWidget(self.cluster_combo)

        controls_layout.addStretch()
        self.plot_button = QPushButton('Plot Clusters')
        self.plot_button.clicked.connect(self.plot_clusters)
        controls_layout.addWidget(self.plot_button)
        self.layout.addLayout(controls_layout)

        self.color_by_combo.currentTextChanged.connect(self.plot_clusters)
        self.centroids_checkbox.toggled.connect(self.plot_clusters)

        body_layout = QHBoxLayout()

        self.histogram_figure = Figure(figsize=(7, 9))
        self.histogram_canvas = FigureCanvas(self.histogram_figure)
        body_layout.addWidget(self.histogram_canvas, 1)

        right_layout = QVBoxLayout()
        self.figure = Figure(figsize=(6, 4.0))
        self.canvas = FigureCanvas(self.figure)
        right_layout.addWidget(self.canvas, 1)

        self.cluster_summary_label = QLabel('Cluster summary will appear here.')
        self.cluster_summary_label.setWordWrap(True)
        self.cluster_summary_label.setMinimumHeight(78)
        right_layout.addWidget(self.cluster_summary_label)

        self.heatmap_figure = Figure(figsize=(6, 4.0))
        self.heatmap_canvas = FigureCanvas(self.heatmap_figure)
        right_layout.addWidget(self.heatmap_canvas, 1)

        body_layout.addLayout(right_layout, 1)
        self.layout.addLayout(body_layout, 1)

        self.on_clustering_complete(clear_only=True)

    def on_clustering_complete(self, clear_only=False):
        self.figure.clear(); self.canvas.draw()
        self.heatmap_figure.clear(); self.heatmap_canvas.draw()
        self.histogram_figure.clear(); self.histogram_canvas.draw()

        if clear_only:
            self.plot_button.setEnabled(False)
            self.centroids_checkbox.setEnabled(False)
            self.color_by_combo.setEnabled(False)
            self.cluster_combo.setEnabled(False)
            self.cluster_combo.clear()
            self.cluster_summary_label.setText('Cluster summary will appear here.')
            return

        if self.data_tab.dataset is not None and 'Cluster' in self.data_tab.dataset.columns:
            clusters = sorted(self.data_tab.dataset['Cluster'].unique())
            self.cluster_combo.blockSignals(True)
            self.cluster_combo.clear()
            for cluster in clusters:
                self.cluster_combo.addItem(f'Cluster {cluster}', int(cluster))
            self.cluster_combo.blockSignals(False)
            self.selected_cluster = int(clusters[0]) if len(clusters) else None

            self.plot_button.setEnabled(True)
            self.centroids_checkbox.setEnabled(True)
            self.color_by_combo.setEnabled(True)
            self.cluster_combo.setEnabled(True)
            self.plot_clusters()
            self.update_cluster_views()

    def on_cluster_changed(self, index):
        if index < 0:
            self.selected_cluster = None
        else:
            self.selected_cluster = self.cluster_combo.itemData(index)
        self.update_cluster_views()

    def update_cluster_views(self):
        if self.selected_cluster is None:
            return
        self.display_cluster_histogram()
        self.display_heatmap()
        self.update_cluster_summary()
        self.plot_clusters()

    def update_cluster_summary(self):
        if self.selected_cluster is None or self.data_tab.dataset is None:
            self.cluster_summary_label.setText('Cluster summary will appear here.')
            return

        cluster_data = self.data_tab.dataset[self.data_tab.dataset['Cluster'] == self.selected_cluster]
        if cluster_data.empty:
            self.cluster_summary_label.setText('No data available for the selected cluster.')
            return

        hotspot_lookup = cluster_data['Plant_Type'].map(
            lambda p: PLANT_STYLES.get(p, PLANT_STYLES['Composite Daisy']).get('hotspot', (0.0, 0.0))
        )
        hx = np.array([pt[0] for pt in hotspot_lookup], dtype=float)
        hy = np.array([pt[1] for pt in hotspot_lookup], dtype=float)
        dx = cluster_data['Landing_X'].to_numpy(dtype=float) - hx
        dy = cluster_data['Landing_Y'].to_numpy(dtype=float) - hy
        in_zone = np.sqrt(dx * dx + dy * dy) <= CONTACT_ZONE_RADIUS
        zone_pct = 100.0 * float(np.mean(in_zone))

        mean_dwell = float(cluster_data['Dwell_Time_s'].mean())
        mean_return = float(cluster_data['Return_Regularity'].mean())
        mean_contact = float(cluster_data['Contact_Fit'].mean())

        score = (
            0.38 * mean_contact
            + 0.24 * mean_return
            + 0.18 * min(mean_dwell / 8.0, 1.0)
            + 0.20 * (zone_pct / 100.0)
        )
        if score >= 0.62:
            likelihood = 'High'
            tone = 'This cluster looks like effective pollination.'
        elif score >= 0.38:
            likelihood = 'Moderate'
            tone = 'This cluster suggests partial or inconsistent pollination.'
        else:
            likelihood = 'Low'
            tone = 'This cluster looks like weak or off-target visitation.'

        top_pair = cluster_data['Interaction_Pair'].value_counts().idxmax()
        top_pair = self._format_pair_label(top_pair)

        summary = (
            f"<b>Pollination likelihood:</b> {likelihood}<br>"
            f"<b>Mean dwell time:</b> {mean_dwell:.2f} s &nbsp;&nbsp; "
            f"<b>Mean return regularity:</b> {mean_return:.3f} &nbsp;&nbsp; "
            f"<b>Mean contact fit:</b> {mean_contact:.3f}<br>"
            f"<b>Landings in contact zone:</b> {zone_pct:.1f}%<br>"
            f"<b>Most common pair:</b> {top_pair}<br>"
            f"{tone}"
        )
        self.cluster_summary_label.setText(summary)

    def plot_clusters(self):
        if self.data_tab.dataset is None or 'Cluster' not in self.data_tab.dataset.columns:
            return

        self.figure.clear()
        ax = self.figure.add_subplot(111, projection='3d')
        data_to_plot = self.data_tab.dataset.copy()

        if self.model_tab.scaling_checkbox.isChecked():
            scaler = MinMaxScaler()
            data_to_plot[self.data_tab.features] = scaler.fit_transform(data_to_plot[self.data_tab.features])

        if self.centroids_checkbox.isChecked():
            model = self.model_tab.trained_model
            if model and isinstance(model, KMeans):
                centroids = model.cluster_centers_
                ax.scatter(
                    centroids[:, 0], centroids[:, 1], centroids[:, 2],
                    marker='x', s=120, c='#adb5bd', label='Centroids'
                )
                if self.selected_cluster is not None and 0 <= int(self.selected_cluster) < len(centroids):
                    sc = int(self.selected_cluster)
                    ax.scatter(
                        [centroids[sc, 0]], [centroids[sc, 1]], [centroids[sc, 2]],
                        marker='x', s=260, c='red', label='Selected cluster'
                    )
                    ax.text(
                        centroids[sc, 0], centroids[sc, 1], centroids[sc, 2] + 0.03,
                        f'C{sc}', color='red', fontsize=10, ha='center'
                    )
                ax.set_title('K-Means Cluster Centroids')
        else:
            display_choice = self.color_by_combo.currentText()
            group_field_map = {
                'Cluster': 'Cluster',
                'Interaction Pair': 'Interaction_Pair',
                'Plant Type': 'Plant_Type',
                'Pollinator Type': 'Pollinator_Type',
            }
            group_field = group_field_map[display_choice]
            groups = data_to_plot[group_field].unique()
            cmap = plt.get_cmap('viridis')
            colors = cmap(np.linspace(0, 1, len(groups)))
            group_to_color = dict(zip(groups, colors))
            for name, group in data_to_plot.groupby(group_field):
                alpha = 0.58
                size = 3
                edgecolors = 'none'
                linewidths = 0.0
                if group_field == 'Cluster' and self.selected_cluster is not None and int(name) == int(self.selected_cluster):
                    alpha = 0.92
                    size = 12
                    edgecolors = 'black'
                    linewidths = 0.3
                ax.scatter(
                    group[self.data_tab.features[0]],
                    group[self.data_tab.features[1]],
                    group[self.data_tab.features[2]],
                    label=name,
                    color=group_to_color[name],
                    alpha=alpha,
                    s=size,
                    edgecolors=edgecolors,
                    linewidths=linewidths,
                )
            ax.set_title(f'3D Scatter Colored by {display_choice}')

        ax.set_xlabel('Dwell Time')
        ax.set_ylabel('Return Regularity')
        ax.set_zlabel('Contact Fit')
        self.figure.subplots_adjust(left=0.03, right=0.97, top=0.92, bottom=0.08)
        self.canvas.draw()

    def _format_pair_label(self, pair: str) -> str:
        if '::' in pair:
            plant, pollinator = [x.strip() for x in pair.split('::', 1)]
            return f'{plant} × {pollinator}'
        return pair

    def display_cluster_histogram(self):
        if self.selected_cluster is None:
            return
        data = self.data_tab.dataset
        cluster_data = data[data['Cluster'] == self.selected_cluster].copy()
        pair_counts = cluster_data['Interaction_Pair'].value_counts().sort_values(ascending=True)
        labels = [self._format_pair_label(pair) for pair in pair_counts.index]

        self.histogram_figure.clear()
        ax = self.histogram_figure.add_subplot(111)
        y = np.arange(len(pair_counts))
        ax.barh(y, pair_counts.values, color='#2b83ba')
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel('Count')
        ax.set_ylabel('Plant–Pollinator Pair')
        ax.set_title(f'Interaction Pair Distribution in Cluster {self.selected_cluster}')
        ax.grid(True, axis='x', alpha=0.25)
        self.histogram_figure.subplots_adjust(left=0.42, right=0.97, top=0.95, bottom=0.06)
        self.histogram_canvas.draw()

    def display_heatmap(self):
        if self.selected_cluster is None:
            return
        self.heatmap_figure.clear()
        ax = self.heatmap_figure.add_subplot(111)
        cluster_data = self.data_tab.dataset[self.data_tab.dataset['Cluster'] == self.selected_cluster]
        x_positions = cluster_data['Landing_X'].to_numpy()
        y_positions = cluster_data['Landing_Y'].to_numpy()
        hotspot_lookup = cluster_data['Plant_Type'].map(
            lambda p: PLANT_STYLES.get(p, PLANT_STYLES['Composite Daisy']).get('hotspot', (0.0, 0.0))
        )
        hotspot_x = float(np.mean([pt[0] for pt in hotspot_lookup])) if len(cluster_data) else 0.0
        hotspot_y = float(np.mean([pt[1] for pt in hotspot_lookup])) if len(cluster_data) else 0.0
        heatmap, xedges, yedges = np.histogram2d(x_positions, y_positions, bins=55, range=[[-1.35, 1.35], [-1.35, 1.35]])
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        im = ax.imshow(heatmap.T, extent=extent, origin='lower', cmap='hot', aspect='equal', alpha=0.82)
        self.heatmap_figure.colorbar(im, ax=ax, label='Frequency')

        scale = FLOWER_VISUAL_SCALE
        for i in range(8):
            theta = 2 * np.pi * i / 8
            cx = 0.45 * scale * np.cos(theta)
            cy = 0.45 * scale * np.sin(theta)
            petal = Ellipse(
                (cx, cy),
                width=0.52 * scale,
                height=0.18 * scale,
                angle=np.degrees(theta),
                facecolor='none',
                edgecolor='white',
                lw=1.2,
                alpha=0.85,
            )
            ax.add_patch(petal)
        center = Circle((0, 0), radius=0.24 * scale, facecolor='none', edgecolor='white', lw=1.6, alpha=0.9)
        ax.add_patch(center)
        ax.plot([0, 0], [-0.24 * scale, -1.15], color='white', lw=3, alpha=0.9)
        contact_zone = Circle((hotspot_x, hotspot_y), radius=CONTACT_ZONE_RADIUS, facecolor='none', edgecolor='#8ecae6', lw=1.4, linestyle='--', alpha=0.95)
        ax.add_patch(contact_zone)
        ax.text(hotspot_x, hotspot_y + CONTACT_ZONE_RADIUS + 0.05, 'mean contact zone', color='#8ecae6', fontsize=8, ha='center')

        ax.set_title(f'Landing Position Heatmap for Cluster {self.selected_cluster}')
        ax.set_xlabel('Landing X')
        ax.set_ylabel('Landing Y')
        ax.set_xlim(-1.35, 1.35)
        ax.set_ylim(-1.35, 1.35)
        self.heatmap_figure.subplots_adjust(left=0.10, right=0.92, top=0.90, bottom=0.12)
        self.heatmap_canvas.draw()


class CobberPollinatorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.cobber_maroon = QColor(108, 29, 69)
        self.cobber_gold = QColor(234, 170, 0)
        self.lato_font = QFont('Lato')

        self.setWindowTitle('CobberPollinator v4')
        self.setGeometry(50, 50, 1325, 860)
        self.setFont(self.lato_font)

        self.tabs = QTabWidget()
        self.data_tab = DataTab(self)
        self.model_tab = ModelTab(self.data_tab, self)
        self.results_tab = ResultsTab(self.data_tab, self.model_tab, self)
        self.model_tab.results_tab = self.results_tab

        self.tabs.addTab(self.data_tab, 'Data')
        self.tabs.addTab(self.model_tab, 'Model')
        self.tabs.addTab(self.results_tab, 'Results')
        self.setCentralWidget(self.tabs)

    def on_data_loaded(self):
        QMessageBox.information(self, 'Dataset Loaded', 'Dataset has loaded successfully!')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = CobberPollinatorApp()
    main_window.show()
    sys.exit(app.exec())
