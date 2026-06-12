# CobberDescender.py
# An application for exploring gradient descent.
# Refactored for the CobberLearnChem launcher.

import sys
import numpy as np
from typing import List, Tuple

# --- Matplotlib and PyQt6 Integration ---
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QFrame, QPushButton, QFormLayout, QMessageBox, QLineEdit
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QColor  # Added for branding
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch

# --- Core Engine (LOGIC IS 100% UNCHANGED) ---

Dataset = List[Tuple[float, float]]


def generate_linear_data(num_points: int = 50, m_range=(-3, 3), b_range=(-3, 3), noise_level: float = 2.0) -> Tuple[
    Dataset, float, float]:
    true_slope = round(np.random.uniform(*m_range), 2)
    true_intercept = round(np.random.uniform(*b_range), 2)
    x_data = np.linspace(0, 10, num_points)
    y_data = true_slope * x_data + true_intercept
    noise = np.random.normal(0, noise_level, num_points)
    y_data_noisy = y_data + noise
    return list(zip(x_data, y_data_noisy)), true_slope, true_intercept


def calculate_mse(data: Dataset, m: float, b: float) -> float:
    n = len(data)
    if n == 0: return 0.0
    return sum((y_i - (m * x_i + b)) ** 2 for x_i, y_i in data) / n


def calculate_gradient(data: Dataset, m: float, b: float) -> Tuple[float, float]:
    n = len(data)
    if n == 0: return 0.0, 0.0
    dm, db = 0.0, 0.0
    for x_i, y_i in data:
        error_term = y_i - (m * x_i + b)
        dm += -2 * x_i * error_term
        db += -2 * error_term
    return dm / n, db / n


# --- GUI Development ---

class MplCanvas(FigureCanvas):
    """A custom matplotlib canvas."""

    def __init__(self, parent=None, width=7, height=7, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)
        self.setParent(parent)

    def resizeEvent(self, event):
        super(MplCanvas, self).resizeEvent(event)
        try:
            self.fig.tight_layout()
        except ValueError:
            pass


# --- Main Application Class (RENAMED) ---
class CobberDescenderApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # --- BRANDING ---
        self.cobber_maroon = QColor(108, 29, 69)
        self.cobber_gold = QColor(234, 170, 0)
        self.lato_font = QFont("Lato")

        self.setWindowTitle("CobberDescender")
        self.setGeometry(100, 100, 1550, 625)
        self.setFont(self.lato_font)

        self.dataset = [];
        self.true_m, self.true_b = 0, 0
        self.path_points = [];
        self.current_pos = None
        self.mse_grid = None;
        self.best_fit_pos = None

        main_layout = QHBoxLayout()
        controls_panel = QFrame();
        controls_panel.setFrameShape(QFrame.Shape.StyledPanel)
        controls_layout = QVBoxLayout(controls_panel)

        self.generate_data_button = QPushButton("1. Generate Simulated Data")
        self.show_contours_button = QPushButton("3. Show Contours & Best Fit")

        # --- BRANDING ---
        #self.generate_data_button.setStyleSheet(
         #   f"background-color: {self.cobber_gold.name()}; color: {self.cobber_maroon.name()}; font-weight: bold; padding: 5px; border-radius: 3px;")
        #self.show_contours_button.setStyleSheet(
          #  f"background-color: {self.cobber_gold.name()}; color: {self.cobber_maroon.name()}; font-weight: bold; padding: 5px; border-radius: 3px;")

        self.step_size_input = QLineEdit("0.5")
        self.noise_input = QLineEdit("2.0")
        self.gradient_label = QLabel("Gradient: N/A")
        self.gradient_label.setStyleSheet("font-size: 18px; font-weight: bold;")

        form_layout = QFormLayout()
        form_layout.addRow(self.generate_data_button)
        form_layout.addRow("Step Size:", self.step_size_input)
        form_layout.addRow("Noise Level:", self.noise_input)
        controls_layout.addLayout(form_layout)
        controls_layout.addWidget(QLabel("<b>2. Find steepest descent and click to move.</b>"))
        controls_layout.addWidget(self.gradient_label)
        controls_layout.addStretch()
        controls_layout.addWidget(self.show_contours_button)

        contour_panel = QFrame();
        contour_panel.setFrameShape(QFrame.Shape.StyledPanel)
        contour_layout = QVBoxLayout(contour_panel)
        contour_layout.addWidget(QLabel("<b>Loss Function Landscape (MSE)</b>"))
        self.contour_canvas = MplCanvas(self)
        contour_layout.addWidget(self.contour_canvas)

        scatter_panel = QFrame();
        scatter_panel.setFrameShape(QFrame.Shape.StyledPanel)
        scatter_layout = QVBoxLayout(scatter_panel)
        scatter_layout.addWidget(QLabel("<b>Data and Model Fit</b>"))
        self.scatter_canvas = MplCanvas(self)
        scatter_layout.addWidget(self.scatter_canvas)

        main_layout.addWidget(controls_panel, 1)
        main_layout.addWidget(contour_panel, 7)
        main_layout.addWidget(scatter_panel, 7)

        central_widget = QWidget();
        central_widget.setLayout(main_layout);
        self.setCentralWidget(central_widget)

        self.generate_data_button.clicked.connect(self.run_data_generation)
        self.show_contours_button.clicked.connect(self.run_show_contours)
        self.contour_canvas.mpl_connect('button_press_event', self.on_canvas_click)
        self.contour_canvas.mpl_connect('motion_notify_event', self.on_canvas_motion)
        self.contour_canvas.mpl_connect('axes_leave_event', self.on_canvas_leave)

        self.reset_plots()

    # --- All other methods have UNCHANGED logic ---

    def run_data_generation(self):
        try:
            noise = float(self.noise_input.text())
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Noise level must be a number."); return
        self.dataset, self.true_m, self.true_b = generate_linear_data(noise_level=noise)
        self.mse_grid = None;
        self.path_points = [];
        self.current_pos = None;
        self.best_fit_pos = None
        self.gradient_label.setText("Gradient: N/A");
        self.gradient_label.setStyleSheet("font-size: 18px; font-weight: bold; color: black;")
        self.reset_plots();
        self.draw_scatter_plot()
        QMessageBox.information(self, "Data Generated",
                                "Click a point on the middle plot to choose your starting (m, b) parameters.")

    def on_canvas_click(self, event):
        if not self.dataset: QMessageBox.warning(self, "No Data", "Please generate data first."); return
        if event.inaxes != self.contour_canvas.axes: return
        click_pos = (event.xdata, event.ydata)
        if self.current_pos is None:
            self.current_pos = click_pos
        else:
            try:
                step_size = float(self.step_size_input.text())
            except ValueError:
                QMessageBox.warning(self, "Invalid Input", "Step size must be a number."); return
            dx = click_pos[0] - self.current_pos[0];
            dy = click_pos[1] - self.current_pos[1]
            norm = np.sqrt(dx ** 2 + dy ** 2)
            if norm == 0: return
            unit_vector = (dx / norm, dy / norm)
            new_m = self.current_pos[0] + step_size * unit_vector[0]
            new_b = self.current_pos[1] + step_size * unit_vector[1]
            self.current_pos = (new_m, new_b)
        self.path_points.append(self.current_pos)
        self.draw_contour_plot();
        self.draw_scatter_plot()

    def on_canvas_motion(self, event):
        if self.current_pos is None or event.inaxes != self.contour_canvas.axes: return
        mouse_pos = (event.xdata, event.ydata)
        dx = mouse_pos[0] - self.current_pos[0];
        dy = mouse_pos[1] - self.current_pos[1]
        norm = np.sqrt(dx ** 2 + dy ** 2)
        if norm == 0: return
        unit_vector = (dx / norm, dy / norm)
        grad_m, grad_b = calculate_gradient(self.dataset, self.current_pos[0], self.current_pos[1])
        directional_derivative = grad_m * unit_vector[0] + grad_b * unit_vector[1]
        self.gradient_label.setText(f"Gradient: {directional_derivative:.2f}")
        color = "green" if directional_derivative < 0 else "red"
        self.gradient_label.setStyleSheet(f"font-size: 18px; font-weight: bold; color: {color};")
        arrow_end_pos = (self.current_pos[0] + 0.5 * unit_vector[0], self.current_pos[1] + 0.5 * unit_vector[1])
        self.draw_contour_plot(arrow_end=arrow_end_pos)

    def on_canvas_leave(self, event):
        if self.current_pos: self.draw_contour_plot()

    def run_show_contours(self):
        if not self.dataset: return
        if self.mse_grid is None:
            m_vals = np.linspace(-4, 4, 50);
            b_vals = np.linspace(-4, 4, 50)
            self.mse_grid = np.array([[calculate_mse(self.dataset, m, b) for m in m_vals] for b in b_vals])
            min_b_idx, min_m_idx = np.unravel_index(np.argmin(self.mse_grid), self.mse_grid.shape)
            self.best_fit_pos = (m_vals[min_m_idx], b_vals[min_b_idx])
        self.draw_contour_plot(show_contours=True, show_minimum=True, show_best_fit=True)
        self.draw_scatter_plot(show_best_fit=True)

    def reset_plots(self):
        for canvas in [self.contour_canvas, self.scatter_canvas]: canvas.axes.clear()
        self.contour_canvas.axes.set_xlim(-4, 4);
        self.contour_canvas.axes.set_ylim(-4, 4)
        self.contour_canvas.axes.set_xlabel("Slope (m)");
        self.contour_canvas.axes.set_ylabel("Intercept (b)")
        self.contour_canvas.axes.set_title("Click to Choose Initial (m, b) Point");
        self.contour_canvas.axes.set_aspect('equal', adjustable='box');
        self.contour_canvas.axes.grid(True, linestyle='--', alpha=0.5)
        self.scatter_canvas.axes.set_xlim(-1, 11);
        y_min, y_max = (-5, 35)
        if self.dataset: y_data = [p[1] for p in self.dataset]; y_min, y_max = (min(y_data) - 5, max(y_data) + 5)
        self.scatter_canvas.axes.set_ylim(y_min, y_max)
        self.scatter_canvas.axes.set_xlabel("x");
        self.scatter_canvas.axes.set_ylabel("y");
        self.scatter_canvas.axes.set_title("Generated Data");
        self.scatter_canvas.axes.grid(True, linestyle='--', alpha=0.5)
        self.contour_canvas.draw();
        self.scatter_canvas.draw()

    def draw_scatter_plot(self, show_best_fit=False):
        ax = self.scatter_canvas.axes;
        ax.clear()
        if not self.dataset: self.reset_plots(); return
        x_data, y_data = zip(*self.dataset);
        ax.scatter(x_data, y_data, label='Simulated Data', alpha=0.7, zorder=2)
        if self.current_pos:
            m, b = self.current_pos;
            x_line = np.array([-1, 11]);
            y_line = m * x_line + b
            ax.plot(x_line, y_line, color='red', linestyle='--', label=f'Current Fit (m={m:.2f}, b={b:.2f})', zorder=3)
        if show_best_fit:
            x_line = np.array([-1, 11]);
            y_line = self.true_m * x_line + self.true_b
            ax.plot(x_line, y_line, color='green', linewidth=3,
                    label=f'True Parameters (m={self.true_m:.2f}, b={self.true_b:.2f})', zorder=4)
        ax.set_xlim(-1, 11);
        y_min = min(y_data) - 5;
        y_max = max(y_data) + 5;
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("x");
        ax.set_ylabel("y");
        ax.set_title("Data and Model Fit");
        ax.grid(True);
        ax.legend(loc='upper left');
        self.scatter_canvas.draw()

    def draw_contour_plot(self, show_contours=False, show_minimum=False, show_best_fit=False, arrow_end=None):
        ax = self.contour_canvas.axes;
        ax.clear()
        if show_contours and self.mse_grid is not None:
            m_vals = np.linspace(-4, 4, 50);
            b_vals = np.linspace(-4, 4, 50)
            levels = np.geomspace(self.mse_grid.min() + 0.01, self.mse_grid.max(), 20)
            ax.contourf(m_vals, b_vals, self.mse_grid, levels=levels, cmap='viridis_r', zorder=1)
        if self.path_points: path_m, path_b = zip(*self.path_points); ax.plot(path_m, path_b, 'r-o',
                                                                              label='Descent Path', zorder=5)
        if arrow_end and self.current_pos: ax.add_patch(
            FancyArrowPatch(self.current_pos, arrow_end, color='darkorange', mutation_scale=20, arrowstyle='->',
                            zorder=10))
        if show_minimum: ax.plot(self.true_m, self.true_b, 'g*', markersize=20, label='True Parameters', zorder=10,
                                 markeredgecolor='black')
        if show_best_fit and self.best_fit_pos: ax.plot(self.best_fit_pos[0], self.best_fit_pos[1], 'bX', markersize=15,
                                                        label='Minimum Parameters for MSE', zorder=9,
                                                        markeredgecolor='white')
        ax.set_xlim(-4, 4);
        ax.set_ylim(-4, 4);
        ax.set_aspect('equal', adjustable='box');
        ax.set_xlabel("Slope (m)");
        ax.set_ylabel("Intercept (b)")
        ax.set_title("Loss Function Landscape");
        ax.grid(True);
        ax.legend(loc='upper left');
        self.contour_canvas.draw()


# --- Standalone Execution Guard ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CobberDescenderApp()  # Use the new class name
    window.show()
    sys.exit(app.exec())