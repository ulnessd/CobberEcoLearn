# CobberEcoNeuron.py
# A PyQt6 application for exploring single neurons and full neural networks.
#
# Ecology version of CobberNeuron.
#
# Teaching story:
#   A simplified ecological water-demand curve relates air temperature to
#   reference evapotranspiration. Professional evapotranspiration estimates
#   depend on temperature, radiation, humidity, wind, crop/vegetation type,
#   and soil conditions. Here we use a one-variable teaching dataset so that
#   students can focus on how a neuron and a neural network learn a curved
#   ecological response.
#
# Dependencies:
#   pip install PyQt6 numpy matplotlib scikit-learn
#
# Run:
#   python CobberEcoNeuron.py

from __future__ import annotations

import sys
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QFrame, QTabWidget, QPushButton, QSlider, QFormLayout,
    QSpinBox, QLineEdit, QMessageBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QColor
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------
# Teaching dataset
# ---------------------------------------------------------------------
# Simplified reference evapotranspiration data.
#
# Input:
#   Air temperature, degrees Celsius
#
# Output:
#   Reference evapotranspiration, mm/day
#
# These values are generated from a simplified environmental water-demand
# model inspired by professional evapotranspiration calculations. The exact
# equation is intentionally not central to this chapter. The pedagogical
# goal is a realistic nonlinear ecological response curve.
REFERENCE_ET_DATA = {
    5: 1.08,
    8: 1.47,
    11: 1.93,
    14: 2.45,
    17: 3.03,
    20: 3.67,
    23: 4.36,
    26: 5.09,
    29: 5.86,
    32: 6.67,
    35: 7.51,
    38: 8.37,
    41: 9.25,
}


# ---------------------------------------------------------------------
# Core model helpers
# ---------------------------------------------------------------------
def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def calculate_neuron_output(x_input: np.ndarray, weight: float, bias: float) -> np.ndarray:
    return sigmoid(weight * x_input + bias)


def calculate_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def draw_mid_arrow(ax, x0, y0, x1, y1, frac=0.18, color="grey", lw=1, alpha=0.8, head=12):
    """Draw a line with a small arrow centered on the segment."""
    ax.plot([x0, x1], [y0, y1], color=color, alpha=alpha, linewidth=lw)

    dx, dy = (x1 - x0), (y1 - y0)
    length = (dx ** 2 + dy ** 2) ** 0.5
    if length == 0:
        return

    ux, uy = dx / length, dy / length
    midx, midy = (x0 + x1) / 2, (y0 + y1) / 2
    half = (frac * length) / 2
    xa, ya = midx - ux * half, midy - uy * half
    xb, yb = midx + ux * half, midy + uy * half

    arrow = FancyArrowPatch(
        (xa, ya), (xb, yb),
        arrowstyle="-|>",
        mutation_scale=head,
        linewidth=max(1, lw - 1),
        color=color,
        alpha=alpha,
    )
    ax.add_patch(arrow)


def draw_mid_arrow_on_line(ax, x0, x1, y, frac=0.18, color="grey", lw=1, alpha=0.8, head=12):
    """Draw a short centered arrow on a horizontal line."""
    ax.plot([x0, x1], [y, y], color=color, alpha=alpha, linewidth=lw)

    mid = 0.5 * (x0 + x1)
    half = 0.5 * frac * (x1 - x0)
    xa, xb = mid - half, mid + half

    arrow = FancyArrowPatch(
        (xa, y), (xb, y),
        arrowstyle="-|>",
        mutation_scale=head,
        linewidth=lw,
        color=color,
        alpha=alpha,
    )
    ax.add_patch(arrow)


def y_nudge_pixels(ax, y, pixels):
    """Return a y value nudged by a given number of screen pixels."""
    _, y_disp = ax.transData.transform((0, y))
    y2 = ax.transData.inverted().transform((0, y_disp + pixels))[1]
    return y2


# ---------------------------------------------------------------------
# Matplotlib canvas
# ---------------------------------------------------------------------
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, constrained_layout=True)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)


# ---------------------------------------------------------------------
# Tab 1: Single neuron
# ---------------------------------------------------------------------
class SingleNeuronTab(QWidget):
    def __init__(self, air_temps_c: np.ndarray, et0_mm_day: np.ndarray):
        super().__init__()

        self.air_temps_c = air_temps_c
        self.et0_mm_day = et0_mm_day

        self.scaled_temps, self.temp_min, self.temp_max = self.normalize(self.air_temps_c)
        self.scaled_et0, self.et0_min, self.et0_max = self.normalize(self.et0_mm_day)

        layout = QHBoxLayout(self)

        controls_panel = QFrame()
        controls_panel.setFrameShape(QFrame.Shape.StyledPanel)
        controls_layout = QVBoxLayout(controls_panel)

        schematic_label = QLabel("<h3>The Single Neuron Model</h3>")
        self.schematic_canvas = MplCanvas(self, width=3, height=2, dpi=70)

        schematic_text = QLabel(
            "<p>Your goal is to adjust the Weight and Bias to find the lowest MSE. "
            "The single neuron tries to learn a curved ecological response from "
            "air temperature to reference evapotranspiration.</p>"
        )
        schematic_text.setWordWrap(True)

        form_layout = QFormLayout()

        self.weight_slider = QSlider(Qt.Orientation.Horizontal)
        self.bias_slider = QSlider(Qt.Orientation.Horizontal)
        self.mse_label = QLabel("<b>MSE Score:</b> N/A")
        self.mse_label.setStyleSheet("font-size: 16px;")

        self.weight_slider.setRange(-100, 100)
        self.weight_slider.setValue(50)

        self.bias_slider.setRange(-50, 50)
        self.bias_slider.setValue(-25)

        form_layout.addRow("Weight (w):", self.weight_slider)
        form_layout.addRow("Bias (b):", self.bias_slider)

        controls_layout.addWidget(schematic_label)
        controls_layout.addWidget(self.schematic_canvas)
        controls_layout.addWidget(schematic_text)
        controls_layout.addStretch(1)
        controls_layout.addLayout(form_layout)
        controls_layout.addWidget(self.mse_label)
        controls_layout.addStretch(2)

        plot_panel = QFrame()
        plot_panel.setFrameShape(QFrame.Shape.StyledPanel)
        plot_layout = QVBoxLayout(plot_panel)

        self.canvas = MplCanvas(self, width=7, height=6, dpi=100)
        plot_layout.addWidget(self.canvas)

        layout.addWidget(controls_panel, 1)
        layout.addWidget(plot_panel, 2)

        self.weight_slider.valueChanged.connect(self.update_prediction)
        self.bias_slider.valueChanged.connect(self.update_prediction)

        self.update_prediction()

    def normalize(self, data):
        min_val, max_val = np.min(data), np.max(data)
        if max_val == min_val:
            return data, min_val, max_val
        return (data - min_val) / (max_val - min_val), min_val, max_val

    def denormalize(self, scaled_data, min_val, max_val):
        return scaled_data * (max_val - min_val) + min_val

    def update_prediction(self):
        weight = self.weight_slider.value() / 10.0
        bias = self.bias_slider.value() / 10.0

        self.draw_schematic(weight, bias)

        predicted_scaled = calculate_neuron_output(self.scaled_temps, weight, bias)
        mse = calculate_mse(self.scaled_et0, predicted_scaled)
        self.mse_label.setText(f"<b>MSE Score:</b> {mse:.4f}")

        predicted_denormalized = self.denormalize(predicted_scaled, self.et0_min, self.et0_max)

        ax = self.canvas.axes
        ax.clear()

        ax.scatter(self.air_temps_c, self.et0_mm_day, label="Ecological Response Data")
        ax.plot(
            self.air_temps_c,
            predicted_denormalized,
            "r--",
            label=f"Neuron Prediction (w={weight:.1f}, b={bias:.1f})",
        )

        ax.set_title("Reference Evapotranspiration vs. Air Temperature")
        ax.set_xlabel("Air Temperature (°C)")
        ax.set_ylabel("Reference Evapotranspiration (mm/day)")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="lower right")

        residuals = self.et0_mm_day - predicted_denormalized

        inset_ax = ax.inset_axes([0.07, 0.55, 0.38, 0.38])
        inset_ax.axhline(0.0, linestyle="--", alpha=0.7)
        inset_ax.vlines(self.air_temps_c, 0.0, residuals, alpha=0.6)
        inset_ax.scatter(self.air_temps_c, residuals, s=18)

        inset_ax.set_title("Residuals", fontsize=9)
        inset_ax.set_xlabel("Air Temp (°C)", fontsize=8)
        inset_ax.set_ylabel("mm/day", fontsize=8)
        inset_ax.tick_params(labelsize=8)
        inset_ax.grid(True, linestyle=":", alpha=0.4)

        self.canvas.draw()

    def draw_schematic(self, weight, bias):
        ax = self.schematic_canvas.axes
        ax.clear()

        input_y, hidden_y, output_y = 0.5, 0.5, 0.5

        draw_mid_arrow(ax, 0.94, input_y, 1.02, hidden_y, color="grey", lw=2, alpha=0.8, head=14)
        draw_mid_arrow(ax, -0.03, input_y, 0.07, hidden_y, color="grey", lw=2, alpha=0.8, head=14)
        draw_mid_arrow(ax, 0.1, input_y, 0.5, hidden_y, color="grey", lw=2, alpha=0.8, head=14)
        draw_mid_arrow(ax, 0.5, hidden_y, 0.9, output_y, color="grey", lw=2, alpha=0.8, head=14)

        ax.text(
            0.37,
            0.62,
            f"w={weight:.1f}",
            ha="center",
            va="center",
            fontsize=16,
            bbox=dict(facecolor="white", alpha=0.0, edgecolor="none"),
        )
        ax.text(
            0.53,
            0.25,
            f"b={bias:.1f}",
            ha="center",
            va="center",
            fontsize=16,
            bbox=dict(facecolor="white", alpha=0.0, edgecolor="none"),
        )

        ax.scatter([0.1], [input_y], s=1000, c="#d3d3d3", marker="s", zorder=5)
        ax.text(-0.05, output_y, r"$T_{\mathrm{air}}$", ha="center", va="center", fontsize=15)
        ax.text(0.1, input_y - 0.4, "Input\nNeuron", ha="center", va="center", fontsize=16)

        ax.scatter([0.5], [hidden_y], s=1000, c="#6C1D45", marker="s", zorder=5)

        ax.scatter([0.9], [output_y], s=1000, c="#3D3D3D", marker="s", zorder=5)
        ax.text(0.9, output_y - 0.4, "Output\nNeuron", ha="center", va="center", fontsize=16)
        ax.text(1.07, output_y, r"$ET_0$", ha="center", va="center", fontsize=16)

        ax.set_xlim(-0.12, 1.12)
        ax.set_ylim(0, 1)
        ax.axis("off")
        self.schematic_canvas.draw()


# ---------------------------------------------------------------------
# Tab 2: Neural network
# ---------------------------------------------------------------------
class NeuralNetworkTab(QWidget):
    def __init__(self, air_temps_c: np.ndarray, et0_mm_day: np.ndarray, main_window):
        super().__init__()

        self.main_window = main_window
        self.air_temps_c = air_temps_c
        self.et0_mm_day = et0_mm_day

        self.model = None
        self.scaler_X = None
        self.scaler_y = None

        layout = QHBoxLayout(self)

        controls_panel = QFrame()
        controls_panel.setFrameShape(QFrame.Shape.StyledPanel)
        controls_layout = QVBoxLayout(controls_panel)

        form_layout = QFormLayout()

        self.neurons_spinner = QSpinBox()
        self.neurons_spinner.setRange(2, 16)
        self.neurons_spinner.setValue(8)

        self.learning_rate_input = QLineEdit("0.005")
        self.epochs_input = QLineEdit("10000")
        self.train_button = QPushButton("Train Network")

        form_layout.addRow("Neurons in Hidden Layer:", self.neurons_spinner)
        form_layout.addRow("Learning Rate:", self.learning_rate_input)
        form_layout.addRow("Training Cycles (Epochs):", self.epochs_input)

        controls_layout.addWidget(QLabel("<h3>Network Configuration</h3>"))
        controls_layout.addLayout(form_layout)
        controls_layout.addWidget(self.train_button)
        controls_layout.addStretch()

        schematic_panel = QFrame()
        schematic_panel.setFrameShape(QFrame.Shape.StyledPanel)
        schematic_layout = QVBoxLayout(schematic_panel)

        schematic_layout.addWidget(QLabel("<b>Network Architecture</b>"))
        self.schematic_canvas = MplCanvas(self, width=7, height=7)
        schematic_layout.addWidget(self.schematic_canvas)

        results_panel = QTabWidget()

        self.loss_canvas = MplCanvas(self)
        results_panel.addTab(self.loss_canvas, "Training Progress")

        self.fit_canvas = MplCanvas(self)
        results_panel.addTab(self.fit_canvas, "Final Model Fit")

        self.resid_tab = QWidget()
        resid_layout = QVBoxLayout(self.resid_tab)
        self.resid_canvas = MplCanvas(self)

        self.mae_label = QLabel("MAE: -")
        self.mse_label = QLabel("MSE: -")

        mae_font = QFont(self.font())
        mae_font.setPointSize(16)
        self.mae_label.setFont(mae_font)

        mse_font = QFont(self.font())
        mse_font.setPointSize(16)
        self.mse_label.setFont(mse_font)

        labels_layout = QHBoxLayout()
        labels_layout.setSpacing(12)
        labels_layout.addStretch(1)
        labels_layout.addWidget(self.mae_label)
        labels_layout.addWidget(self.mse_label)
        labels_layout.addStretch(1)

        resid_layout.setSpacing(6)
        resid_layout.setContentsMargins(6, 6, 6, 6)
        resid_layout.addWidget(self.resid_canvas)
        resid_layout.addLayout(labels_layout)

        results_panel.addTab(self.resid_tab, "Residuals")

        layout.addWidget(controls_panel, 3)
        layout.addWidget(schematic_panel, 7)
        layout.addWidget(results_panel, 7)

        self.train_button.clicked.connect(self.train_network)
        self.neurons_spinner.valueChanged.connect(self.update_schematic)

        self.setup_initial_plots()
        self.update_schematic()

    def train_network(self):
        try:
            hidden_neurons = self.neurons_spinner.value()
            learning_rate = float(self.learning_rate_input.text())
            epochs = int(self.epochs_input.text())
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please ensure all parameters are valid numbers.")
            return

        X = self.air_temps_c.reshape(-1, 1)
        y = self.et0_mm_day

        self.train_button.setText("Training...")
        self.train_button.setEnabled(False)
        QApplication.processEvents()

        scaler_X = StandardScaler().fit(X)
        scaler_y = StandardScaler().fit(y.reshape(-1, 1))

        X_scaled = scaler_X.transform(X)
        y_scaled = scaler_y.transform(y.reshape(-1, 1)).ravel()

        self.model = MLPRegressor(
            hidden_layer_sizes=(hidden_neurons,),
            activation="relu",
            solver="adam",
            learning_rate_init=learning_rate,
            max_iter=epochs,
            random_state=42,
            early_stopping=True,
            n_iter_no_change=500,
        )
        self.model.fit(X_scaled, y_scaled)

        self.train_button.setText("Train Network")
        self.train_button.setEnabled(True)

        self.update_schematic()

        self.loss_canvas.axes.clear()
        self.loss_canvas.axes.plot(self.model.loss_curve_)
        self.loss_canvas.axes.set_title("Loss vs. Epoch")
        self.loss_canvas.axes.set_xlabel("Epoch")
        self.loss_canvas.axes.set_ylabel("Mean Squared Error (Loss)")
        self.loss_canvas.axes.grid(True)
        self.loss_canvas.draw()

        x_smooth = np.linspace(self.air_temps_c.min(), self.air_temps_c.max(), 240).reshape(-1, 1)
        x_smooth_scaled = scaler_X.transform(x_smooth)
        y_pred_scaled = self.model.predict(x_smooth_scaled)
        y_pred_denormalized = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

        self.fit_canvas.axes.clear()
        self.fit_canvas.axes.scatter(self.air_temps_c, self.et0_mm_day, label="Ecological Response Data")
        self.fit_canvas.axes.plot(
            x_smooth.ravel(),
            y_pred_denormalized,
            "r-",
            label="Neural Network Fit",
            linewidth=2,
        )
        self.fit_canvas.axes.set_title("Evapotranspiration Data & Neural Network Fit")
        self.fit_canvas.axes.set_xlabel("Air Temperature (°C)")
        self.fit_canvas.axes.set_ylabel("Reference Evapotranspiration (mm/day)")
        self.fit_canvas.axes.legend()
        self.fit_canvas.axes.grid(True)
        self.fit_canvas.draw()

        self.scaler_X = scaler_X
        self.scaler_y = scaler_y

        y_train_pred_scaled = self.model.predict(X_scaled)
        y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).ravel()

        residuals = self.et0_mm_day - y_train_pred
        mae = float(np.mean(np.abs(residuals)))
        mse = float(np.mean(residuals ** 2))

        axr = self.resid_canvas.axes
        axr.clear()

        axr.axhline(0.0, linestyle="--", alpha=0.7)
        axr.vlines(self.air_temps_c, 0.0, residuals, alpha=0.6)
        axr.scatter(self.air_temps_c, residuals, s=25)

        axr.set_title("Residuals (Observed − Predicted)")
        axr.set_xlabel("Air Temperature (°C)")
        axr.set_ylabel("Residual (mm/day)")
        axr.grid(True, linestyle=":", alpha=0.5)
        self.resid_canvas.draw()

        self.mae_label.setText(f"MAE: {mae:.4f} mm/day")
        self.mse_label.setText(f"MSE: {mse:.4f} (mm/day)²")

    def update_schematic(self):
        ax = self.schematic_canvas.axes
        ax.clear()

        num_neurons = self.neurons_spinner.value()
        scale_font_size = max(7, 20 - num_neurons)

        input_y = [0.5]
        hidden_y = np.linspace(0.1, 0.9, num_neurons)
        output_y = [0.5]

        if self.model and self.model.hidden_layer_sizes[0] == num_neurons:
            input_weights = self.model.coefs_[0]
            hidden_biases = self.model.intercepts_[0]
            output_weights = self.model.coefs_[1]

            for i, hy in enumerate(hidden_y):
                draw_mid_arrow(ax, 0.1, input_y[0], 0.5, hy, color="grey", lw=2, alpha=0.8, head=14)
                ax.text(
                    0.3,
                    (input_y[0] + hy) / 2,
                    f"w={input_weights[0][i]:.1f}",
                    ha="center",
                    va="center",
                    fontsize=scale_font_size,
                    bbox=dict(facecolor="white", alpha=0.0, edgecolor="none"),
                )
                ax.text(
                    0.5,
                    (input_y[0] + hy) - 0.545,
                    f"b={hidden_biases[i]:.1f}",
                    ha="center",
                    va="center",
                    fontsize=scale_font_size,
                    bbox=dict(facecolor="white", alpha=0.0, edgecolor="none"),
                )

            for i, hy in enumerate(hidden_y):
                draw_mid_arrow(ax, 0.5, hy, 0.9, output_y[0], color="grey", lw=2, alpha=0.8, head=14)
                ax.text(
                    0.7,
                    (hy + output_y[0]) / 2,
                    f"w={output_weights[i][0]:.1f}",
                    ha="center",
                    va="center",
                    fontsize=scale_font_size,
                    bbox=dict(facecolor="white", alpha=0.0, edgecolor="none"),
                )
        else:
            for hy in hidden_y:
                draw_mid_arrow(ax, 0.1, input_y[0], 0.5, hy, color="grey", lw=1.8, alpha=0.8, head=12)
                draw_mid_arrow(ax, 0.5, hy, 0.9, output_y[0], color="grey", lw=1.8, alpha=0.8, head=12)

        yin = y_nudge_pixels(ax, input_y[0], +1)
        yout = y_nudge_pixels(ax, output_y[0], +1)

        draw_mid_arrow_on_line(ax, -0.02, 0.08, yin, color="grey", lw=2, alpha=0.8, head=14)
        draw_mid_arrow_on_line(ax, 0.95, 1.02, yout, color="grey", lw=2, alpha=0.8, head=14)

        ax.scatter([0.1], input_y, s=1000, c="#d3d3d3", marker="s", zorder=5)
        ax.text(0.075, input_y[0] - 0.09, "Input\nNeuron", ha="center", va="center", fontsize=13)

        ax.scatter([0.5] * num_neurons, hidden_y, s=400, c="#6C1D45", marker="s", zorder=5)
        ax.text(0.5, input_y[0] + 0.475, "Hidden Layer", ha="center", va="center", fontsize=16)

        ax.scatter([0.9], output_y, s=1000, c="#3D3D3D", marker="s", zorder=5)
        ax.text(0.925, output_y[0] - 0.09, "Output\nNeuron", ha="center", va="center", fontsize=13)

        ax.text(-0.055, output_y[0], r"$T_{\mathrm{air}}$", ha="center", va="center", fontsize=15)
        ax.text(1.075, output_y[0], r"$ET_0$", ha="center", va="center", fontsize=16)

        ax.set_xlim(-0.12, 1.12)
        ax.set_ylim(0, 1)
        ax.axis("off")
        self.schematic_canvas.draw()

    def setup_initial_plots(self):
        self.loss_canvas.axes.set_title("Loss vs. Epoch")
        self.loss_canvas.axes.set_xlabel("Epoch")
        self.loss_canvas.axes.set_ylabel("Mean Squared Error (Loss)")
        self.loss_canvas.axes.grid(True)
        self.loss_canvas.draw()

        self.fit_canvas.axes.clear()
        self.fit_canvas.axes.scatter(self.air_temps_c, self.et0_mm_day, label="Ecological Response Data")
        self.fit_canvas.axes.set_title("Evapotranspiration Data & Model Fit")
        self.fit_canvas.axes.set_xlabel("Air Temperature (°C)")
        self.fit_canvas.axes.set_ylabel("Reference Evapotranspiration (mm/day)")
        self.fit_canvas.axes.legend()
        self.fit_canvas.axes.grid(True)
        self.fit_canvas.draw()

        self.resid_canvas.axes.clear()
        self.resid_canvas.axes.axhline(0.0, linestyle="--", alpha=0.7)
        self.resid_canvas.axes.set_title("Residuals (Observed − Predicted)")
        self.resid_canvas.axes.set_xlabel("Air Temperature (°C)")
        self.resid_canvas.axes.set_ylabel("Residual (mm/day)")
        self.resid_canvas.axes.text(
            0.5,
            0.5,
            "Train the network to see residuals.",
            ha="center",
            va="center",
            transform=self.resid_canvas.axes.transAxes,
            alpha=0.7,
        )
        self.resid_canvas.draw()


# ---------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------
class CobberEcoNeuronApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.cobber_maroon = QColor(108, 29, 69)
        self.cobber_gold = QColor(234, 170, 0)
        self.lato_font = QFont("Lato")

        self.setWindowTitle("CobberEcoNeuron")
        self.setGeometry(100, 100, 1400, 700)
        self.setFont(self.lato_font)

        self.air_temps_c = np.array(list(REFERENCE_ET_DATA.keys()), dtype=float)
        self.et0_mm_day = np.array(list(REFERENCE_ET_DATA.values()), dtype=float)

        tabs = QTabWidget()
        self.setCentralWidget(tabs)

        tabs.addTab(SingleNeuronTab(self.air_temps_c, self.et0_mm_day), "The Single Neuron")
        tabs.addTab(NeuralNetworkTab(self.air_temps_c, self.et0_mm_day, self), "The Neural Network")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CobberEcoNeuronApp()
    window.show()
    sys.exit(app.exec())
