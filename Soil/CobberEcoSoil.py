# CobberEcoSoil.py
# A PyQt6 application for exploring regression models on a synthetic
# prairie-wetland "Hidden Carbon" dataset.
#
# Ecology version of CobberLand.
#
# Teaching goal:
#   Some ecological properties are hard, slow, expensive, or destructive
#   to measure directly. Belowground carbon is one example. Easier field
#   measurements such as soil moisture, vegetation height, canopy cover,
#   greenness, elevation, and distance to water may help estimate it.
#
# Expected CSV files from generate_hidden_carbon_data.py:
#   hidden_carbon_training.csv
#   hidden_carbon_new_sites.csv       optional, for the 10 new-site predictions
#
# Dependencies:
#   pip install PyQt6 pandas numpy matplotlib scikit-learn
#
# Run:
#   python CobberEcoSoil.py

from __future__ import annotations

import sys
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QLabel, QComboBox, QSlider, QLineEdit,
    QMessageBox, QStatusBar, QFormLayout, QTableWidget, QTableWidgetItem,
    QTabWidget, QTextEdit, QHeaderView
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QColor

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def app_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


APP_ROOT = app_root()


TARGET_COLUMN = "belowground_carbon_Mg_ha"

FEATURE_SETS = {
    "Moisture only": [
        "soil_moisture_frac",
    ],
    "Moisture + greenness": [
        "soil_moisture_frac",
        "greenness_index",
    ],
    "Field quick survey": [
        "soil_moisture_frac",
        "vegetation_height_cm",
        "canopy_cover_pct",
        "greenness_index",
    ],
    "All easy measurements": [
        "soil_moisture_frac",
        "surface_elevation_cm",
        "distance_to_water_m",
        "vegetation_height_cm",
        "canopy_cover_pct",
        "greenness_index",
        "plant_wetness_score",
    ],
}

ALL_FEATURE_COLUMNS = FEATURE_SETS["All easy measurements"]

FRIENDLY_NAMES = {
    "soil_moisture_frac": "Soil moisture fraction",
    "surface_elevation_cm": "Surface elevation (cm)",
    "distance_to_water_m": "Distance to water (m)",
    "vegetation_height_cm": "Vegetation height (cm)",
    "canopy_cover_pct": "Canopy cover (%)",
    "greenness_index": "Greenness index",
    "plant_wetness_score": "Plant wetness score",
    "belowground_carbon_Mg_ha": "Belowground carbon (Mg C/ha)",
}


DEFAULT_MANUAL_VALUES = {
    "soil_moisture_frac": "0.62",
    "surface_elevation_cm": "62",
    "distance_to_water_m": "35",
    "vegetation_height_cm": "92",
    "canopy_cover_pct": "84",
    "greenness_index": "0.78",
    "plant_wetness_score": "4.2",
}


class PlotCanvas(FigureCanvas):
    """A helper class for an embedded Matplotlib canvas."""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.figure = Figure(figsize=(width, height), dpi=dpi, tight_layout=True)
        self.axes = self.figure.add_subplot(111)
        super().__init__(self.figure)
        self.setParent(parent)


class CobberEcoSoilApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.cobber_maroon = QColor(108, 29, 69)
        self.cobber_gold = QColor(234, 170, 0)
        self.lato_font = QFont("Lato")

        self.setWindowTitle("CobberEcoSoil")
        self.setGeometry(100, 100, 1420, 780)
        self.setFont(self.lato_font)

        self.data: pd.DataFrame | None = None
        self.new_sites: pd.DataFrame | None = None
        self.model = None
        self.current_features: list[str] = []
        self.latest_predictions_df: pd.DataFrame | None = None

        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.main_layout = QHBoxLayout(self.main_widget)

        self.create_controls_panel()
        self.create_results_panel()

        self.main_layout.addWidget(self.controls_widget, 1)
        self.main_layout.addWidget(self.results_widget, 3)

        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage(
            "Ready. Load hidden_carbon_training.csv to begin."
        )

        self.try_auto_load_training()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def create_controls_panel(self):
        self.controls_widget = QWidget()
        controls_layout = QVBoxLayout(self.controls_widget)
        self.controls_widget.setMaximumWidth(380)

        self.open_button = QPushButton("Load hidden_carbon_training.csv")
        self.open_button.clicked.connect(self.open_training_file)

        self.new_sites_button = QPushButton("Load hidden_carbon_new_sites.csv")
        self.new_sites_button.clicked.connect(self.open_new_sites_file)

        alg_label = QLabel("<b>1. Select ML Algorithm:</b>")
        self.alg_combo = QComboBox()
        self.algorithms = [
            "Linear Regression",
            "Decision Tree",
            "Random Forest",
            "Support Vector Machine",
            "k-Nearest Neighbors",
        ]
        self.alg_combo.addItems(self.algorithms)

        feature_label = QLabel("<b>2. Select Easy Field Measurements:</b>")
        self.feature_combo = QComboBox()
        self.feature_combo.addItems(list(FEATURE_SETS.keys()))
        self.feature_combo.setCurrentText("All easy measurements")
        self.feature_combo.currentTextChanged.connect(self.update_feature_note)

        self.feature_note = QLabel("")
        self.feature_note.setWordWrap(True)
        self.feature_note.setStyleSheet("font-size: 10pt; color: #444;")

        split_label = QLabel("<b>3. Set Training Data Percentage:</b>")
        self.split_slider = QSlider(Qt.Orientation.Horizontal)
        self.split_slider.setRange(50, 90)
        self.split_slider.setValue(80)
        self.split_slider.setTickInterval(10)
        self.split_slider.setTickPosition(QSlider.TickPosition.TicksBelow)

        self.split_value_label = QLabel(f"{self.split_slider.value()}%")
        self.split_value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.split_slider.valueChanged.connect(
            lambda value: self.split_value_label.setText(f"{value}%")
        )

        self.train_button = QPushButton("Run ML Training")
        self.train_button.clicked.connect(self.run_training)

        pred_label = QLabel("<b>4. Predict Hidden Carbon Manually:</b>")
        pred_label.setWordWrap(True)

        self.manual_form = QFormLayout()
        self.manual_entries: dict[str, QLineEdit] = {}

        for col in ALL_FEATURE_COLUMNS:
            entry = QLineEdit()
            entry.setPlaceholderText(DEFAULT_MANUAL_VALUES.get(col, ""))
            entry.setText(DEFAULT_MANUAL_VALUES.get(col, ""))
            self.manual_entries[col] = entry
            self.manual_form.addRow(FRIENDLY_NAMES[col] + ":", entry)

        self.pred_button = QPushButton("Predict Belowground Carbon")
        self.pred_button.clicked.connect(self.predict_manual)

        self.prediction_result_label = QLabel("Prediction: -")
        self.prediction_result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.prediction_result_label.setStyleSheet(
            "background-color: #f0f0f0; border: 1px solid #ccc; "
            "padding: 8px; border-radius: 5px;"
        )

        self.predict_new_sites_button = QPushButton("Predict 10 New Sites")
        self.predict_new_sites_button.clicked.connect(self.predict_new_sites)

        controls_layout.addWidget(self.open_button)
        controls_layout.addWidget(self.new_sites_button)
        controls_layout.addSpacing(12)
        controls_layout.addWidget(alg_label)
        controls_layout.addWidget(self.alg_combo)
        controls_layout.addSpacing(10)
        controls_layout.addWidget(feature_label)
        controls_layout.addWidget(self.feature_combo)
        controls_layout.addWidget(self.feature_note)
        controls_layout.addSpacing(10)
        controls_layout.addWidget(split_label)
        controls_layout.addWidget(self.split_slider)
        controls_layout.addWidget(self.split_value_label)
        controls_layout.addSpacing(12)
        controls_layout.addWidget(self.train_button)
        controls_layout.addSpacing(14)
        controls_layout.addWidget(pred_label)
        controls_layout.addLayout(self.manual_form)
        controls_layout.addWidget(self.pred_button)
        controls_layout.addWidget(self.prediction_result_label)
        controls_layout.addSpacing(10)
        controls_layout.addWidget(self.predict_new_sites_button)
        controls_layout.addStretch()

        self.update_feature_note()

    def create_results_panel(self):
        self.results_widget = QWidget()
        results_layout = QVBoxLayout(self.results_widget)

        self.tabs = QTabWidget()

        self.model_tab = QWidget()
        model_tab_layout = QVBoxLayout(self.model_tab)

        plots_layout = QHBoxLayout()
        self.pv_a_plot = PlotCanvas(self)
        self.residuals_plot = PlotCanvas(self)
        plots_layout.addWidget(self.pv_a_plot)
        plots_layout.addWidget(self.residuals_plot)

        metrics_label = QLabel("<b>Model Evaluation Metrics:</b>")
        self.metrics_display = QLabel("Train a model to see results.")
        self.metrics_display.setStyleSheet(
            "background-color: #f0f0f0; border: 1px solid #ccc; "
            "padding: 10px; border-radius: 5px;"
        )
        self.metrics_display.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.metrics_display.setWordWrap(True)

        model_tab_layout.addLayout(plots_layout)
        model_tab_layout.addWidget(metrics_label)
        model_tab_layout.addWidget(self.metrics_display)

        self.data_tab = QWidget()
        data_tab_layout = QHBoxLayout(self.data_tab)
        self.data_plot = PlotCanvas(self)
        self.data_explainer = QTextEdit()
        self.data_explainer.setReadOnly(True)
        self.data_explainer.setMaximumWidth(380)
        self.data_explainer.setText(
            "The training dataset represents a fictional prairie–wetland "
            "restoration landscape.\n\n"
            "The easy measurements are field-friendly variables such as soil "
            "moisture, vegetation height, canopy cover, greenness, elevation, "
            "and distance to water.\n\n"
            "The hard measurement is belowground carbon stock. The training "
            "data intentionally contain a gap between mesic prairie and "
            "saturated wetland regimes. Models must learn across a landscape "
            "that is not one smooth continuous trend."
        )
        data_tab_layout.addWidget(self.data_plot, 3)
        data_tab_layout.addWidget(self.data_explainer, 1)

        self.predictions_tab = QWidget()
        predictions_layout = QVBoxLayout(self.predictions_tab)
        self.predictions_table = QTableWidget()
        self.predictions_note = QLabel(
            "Load new sites, train a model, and click 'Predict 10 New Sites' "
            "to estimate hidden belowground carbon for new restoration sites."
        )
        self.predictions_note.setWordWrap(True)
        predictions_layout.addWidget(self.predictions_note)
        predictions_layout.addWidget(self.predictions_table)

        self.tabs.addTab(self.model_tab, "Model Fit")
        self.tabs.addTab(self.data_tab, "Training Data Story")
        self.tabs.addTab(self.predictions_tab, "New Site Predictions")

        results_layout.addWidget(self.tabs)

        self.clear_plots()
        self.draw_empty_data_story()

    # ------------------------------------------------------------------
    # Loading data
    # ------------------------------------------------------------------
    def try_auto_load_training(self):
        path = APP_ROOT / "hidden_carbon_training.csv"
        if path.exists():
            self.load_training_file(path)

        new_path = APP_ROOT / "hidden_carbon_new_sites.csv"
        if new_path.exists():
            self.load_new_sites_file(new_path, quiet=True)

    def open_training_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Hidden Carbon Training CSV",
            str(APP_ROOT),
            "CSV Files (*.csv);;All Files (*)",
        )
        if file_path:
            self.load_training_file(Path(file_path))

    def open_new_sites_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Hidden Carbon New Sites CSV",
            str(APP_ROOT),
            "CSV Files (*.csv);;All Files (*)",
        )
        if file_path:
            self.load_new_sites_file(Path(file_path), quiet=False)

    def load_training_file(self, path: Path):
        try:
            data = pd.read_csv(path)
            self.validate_training_data(data)
            self.data = data.copy()
            self.model = None
            self.latest_predictions_df = None

            self.statusBar().showMessage(
                f"Loaded training data: {path.name} ({len(self.data)} rows)",
                6000,
            )
            self.metrics_display.setText("Training data loaded. Select a model and click Run ML Training.")
            self.draw_data_story()
            self.clear_plots()
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to load training data:\n{exc}")
            self.data = None

    def load_new_sites_file(self, path: Path, quiet: bool = False):
        try:
            data = pd.read_csv(path)
            self.validate_new_sites_data(data)
            self.new_sites = data.copy()
            self.statusBar().showMessage(
                f"Loaded new sites: {path.name} ({len(self.new_sites)} rows)",
                6000,
            )
            if not quiet:
                QMessageBox.information(
                    self,
                    "New Sites Loaded",
                    f"Loaded {len(self.new_sites)} new sites for prediction.",
                )
        except Exception as exc:
            if not quiet:
                QMessageBox.critical(self, "Error", f"Failed to load new sites:\n{exc}")
            self.new_sites = None

    def validate_training_data(self, df: pd.DataFrame):
        required = set(ALL_FEATURE_COLUMNS + [TARGET_COLUMN])
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"CSV is missing required columns: {', '.join(sorted(missing))}")

    def validate_new_sites_data(self, df: pd.DataFrame):
        required = set(ALL_FEATURE_COLUMNS)
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"New-sites CSV is missing required columns: {', '.join(sorted(missing))}")

    # ------------------------------------------------------------------
    # Training and prediction
    # ------------------------------------------------------------------
    def selected_features(self) -> list[str]:
        return FEATURE_SETS[self.feature_combo.currentText()]

    def make_model(self, alg_name: str):
        if alg_name == "Linear Regression":
            return LinearRegression()
        if alg_name == "Decision Tree":
            return DecisionTreeRegressor(random_state=42, max_depth=8)
        if alg_name == "Random Forest":
            return RandomForestRegressor(random_state=42, n_estimators=160, max_depth=None)
        if alg_name == "Support Vector Machine":
            return make_pipeline(StandardScaler(), SVR(C=20.0, gamma="scale", epsilon=4.0))
        if alg_name == "k-Nearest Neighbors":
            return make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=7))
        raise ValueError(f"Unknown algorithm: {alg_name}")

    def run_training(self):
        if self.data is None:
            QMessageBox.warning(self, "No Data", "Please load hidden_carbon_training.csv first.")
            return

        try:
            features = self.selected_features()
            self.current_features = features

            X = self.data[features]
            y = self.data[TARGET_COLUMN]

            train_size = self.split_slider.value() / 100.0
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                train_size=train_size,
                random_state=42,
            )

            alg_name = self.alg_combo.currentText()
            self.model = self.make_model(alg_name)

            start_time = time.time()
            self.model.fit(X_train, y_train)
            training_time = time.time() - start_time

            y_pred = self.model.predict(X_test)

            self.update_plots(y_test, y_pred, alg_name)
            self.update_metrics(y_test, y_pred, training_time, alg_name, features)
            self.statusBar().showMessage(f"Trained {alg_name} model.", 5000)

        except Exception as exc:
            QMessageBox.critical(self, "Training Error", f"An error occurred during training:\n{exc}")

    def predict_manual(self):
        if self.model is None:
            QMessageBox.warning(self, "No Model", "Please train a model first.")
            return

        try:
            values = {}
            for col in self.current_features:
                text = self.manual_entries[col].text().strip()
                values[col] = float(text)

            X_new = pd.DataFrame([values], columns=self.current_features)
            prediction = float(self.model.predict(X_new)[0])
            self.prediction_result_label.setText(
                f"<b>Prediction: {prediction:.2f} Mg C/ha</b>"
            )
        except ValueError:
            QMessageBox.critical(
                self,
                "Input Error",
                "Please enter valid numbers for the selected feature set.",
            )
        except Exception as exc:
            QMessageBox.critical(self, "Prediction Error", f"Could not make prediction:\n{exc}")

    def predict_new_sites(self):
        if self.model is None:
            QMessageBox.warning(self, "No Model", "Please train a model first.")
            return

        if self.new_sites is None:
            path = APP_ROOT / "hidden_carbon_new_sites.csv"
            if path.exists():
                self.load_new_sites_file(path, quiet=True)
            else:
                QMessageBox.warning(
                    self,
                    "No New Sites",
                    "Please load hidden_carbon_new_sites.csv first.",
                )
                return

        try:
            features = self.current_features
            preds = self.model.predict(self.new_sites[features])

            out = self.new_sites.copy()
            out["predicted_belowground_carbon_Mg_ha"] = np.round(preds, 2)

            if TARGET_COLUMN in out.columns:
                out["prediction_error"] = np.round(
                    out["predicted_belowground_carbon_Mg_ha"] - out[TARGET_COLUMN],
                    2,
                )

            self.latest_predictions_df = out
            self.populate_predictions_table(out)
            self.tabs.setCurrentWidget(self.predictions_tab)
            self.statusBar().showMessage("Predicted belowground carbon for new sites.", 5000)

        except Exception as exc:
            QMessageBox.critical(self, "Prediction Error", f"Could not predict new sites:\n{exc}")

    # ------------------------------------------------------------------
    # Plotting and display
    # ------------------------------------------------------------------
    def update_plots(self, y_test, y_pred, alg_name):
        residuals = y_pred - y_test

        self.pv_a_plot.figure.clear()
        ax1 = self.pv_a_plot.figure.add_subplot(111)

        ax1.scatter(y_test, y_pred, alpha=0.7, s=12)
        lims = [
            min(float(y_test.min()), float(np.min(y_pred))),
            max(float(y_test.max()), float(np.max(y_pred))),
        ]
        ax1.plot(lims, lims, "r--", alpha=0.75, zorder=0, label="Ideal fit")
        ax1.set_xlabel("Actual belowground carbon (Mg C/ha)")
        ax1.set_ylabel("Predicted belowground carbon (Mg C/ha)")
        ax1.set_title(f"{alg_name}\nActual vs. Predicted")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        self.pv_a_plot.draw()

        self.residuals_plot.figure.clear()
        ax2 = self.residuals_plot.figure.add_subplot(111)

        ax2.scatter(y_test, residuals, alpha=0.7, s=12)
        ax2.axhline(0, color="r", linestyle="--", label="Zero error")
        ax2.set_xlabel("Actual belowground carbon (Mg C/ha)")
        ax2.set_ylabel("Residual (Predicted - Actual)")
        ax2.set_title("Residuals Plot")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        self.residuals_plot.draw()

    def update_metrics(self, y_test, y_pred, training_time, alg_name, features):
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        feature_list = "<br>".join([f"• {FRIENDLY_NAMES.get(f, f)}" for f in features])

        metrics_html = f"""
        <b>Model:</b> {alg_name}<br>
        <b>Training time:</b> {training_time:.4f} seconds<br>
        <b>Training data:</b> {self.split_slider.value()}%<br><br>

        <b>Features used:</b><br>
        {feature_list}<br><br>

        <b>Mean Absolute Error (MAE):</b> {mae:.4f} Mg C/ha<br>
        <b>Mean Squared Error (MSE):</b> {mse:.4f} (Mg C/ha)²<br>
        <b>R-squared (R²):</b> {r2:.4f}<br><br>

        <i>Interpretation reminder:</i> low error is good, but inspect the residual plot.
        The training data contain a prairie/wetland gap, so models may behave differently
        near the transition between regimes.
        """
        self.metrics_display.setText(metrics_html)

    def draw_data_story(self):
        if self.data is None:
            self.draw_empty_data_story()
            return

        df = self.data

        self.data_plot.figure.clear()
        ax = self.data_plot.figure.add_subplot(111)

        if "regime" in df.columns:
            regimes = sorted(df["regime"].dropna().unique())
            colors = {
                "prairie_mesic": "#d95f02",
                "wetland_saturated": "#1b9e77",
            }
            for regime in regimes:
                sub = df[df["regime"] == regime]
                ax.scatter(
                    sub["soil_moisture_frac"],
                    sub[TARGET_COLUMN],
                    s=14,
                    alpha=0.65,
                    label=regime,
                    color=colors.get(regime, None),
                )
        else:
            ax.scatter(df["soil_moisture_frac"], df[TARGET_COLUMN], s=14, alpha=0.65)

        ax.axvspan(0.43, 0.57, color="gray", alpha=0.15, label="training gap")
        ax.set_xlabel("Soil moisture fraction")
        ax.set_ylabel("Belowground carbon (Mg C/ha)")
        ax.set_title("Hidden carbon data: easy moisture measurement vs. hard carbon measurement")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        self.data_plot.draw()

    def draw_empty_data_story(self):
        self.data_plot.figure.clear()
        ax = self.data_plot.figure.add_subplot(111)
        ax.text(
            0.5,
            0.5,
            "Load hidden_carbon_training.csv\nto see the prairie–wetland carbon dataset.",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        self.data_plot.draw()

    def clear_plots(self):
        for canvas, title in [
            (self.pv_a_plot, "Actual vs. Predicted"),
            (self.residuals_plot, "Residuals"),
        ]:
            canvas.figure.clear()
            ax = canvas.figure.add_subplot(111)
            ax.text(
                0.5,
                0.5,
                "Train a model to see plot",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(title)
            ax.set_axis_off()
            canvas.draw()

        self.predictions_table.clear()
        self.predictions_table.setRowCount(0)
        self.predictions_table.setColumnCount(0)

    def populate_predictions_table(self, df: pd.DataFrame):
        display_cols = [
            "site_id",
            "soil_moisture_frac",
            "regime",
            "predicted_belowground_carbon_Mg_ha",
        ]
        if TARGET_COLUMN in df.columns:
            display_cols.append(TARGET_COLUMN)
        if "prediction_error" in df.columns:
            display_cols.append("prediction_error")

        # Also include the features used by the current model if not already shown.
        for col in self.current_features:
            if col not in display_cols:
                display_cols.insert(-1 if TARGET_COLUMN in display_cols else len(display_cols), col)

        display_cols = [c for c in display_cols if c in df.columns]

        self.predictions_table.clear()
        self.predictions_table.setRowCount(len(df))
        self.predictions_table.setColumnCount(len(display_cols))
        self.predictions_table.setHorizontalHeaderLabels(display_cols)

        for r, (_, row) in enumerate(df[display_cols].iterrows()):
            for c, col in enumerate(display_cols):
                value = row[col]
                if isinstance(value, (float, np.floating)):
                    text = f"{value:.3f}"
                else:
                    text = str(value)
                item = QTableWidgetItem(text)
                self.predictions_table.setItem(r, c, item)

        self.predictions_table.resizeColumnsToContents()
        self.predictions_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)

    def update_feature_note(self):
        features = FEATURE_SETS[self.feature_combo.currentText()]
        text = "Using: " + ", ".join(FRIENDLY_NAMES.get(f, f) for f in features)
        self.feature_note.setText(text)


def apply_app_stylesheet(app: QApplication) -> None:
    app.setStyleSheet(
        """
        QWidget { color: #222222; background-color: #ffffff; }
        QMainWindow, QDialog { background-color: #ffffff; }
        QLabel { color: #222222; background-color: transparent; }
        QComboBox, QLineEdit, QTextEdit, QTableWidget {
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
        QTabWidget::pane { border: 1px solid #cccccc; }
        QTabBar::tab { padding: 6px 12px; }
        QTabBar::tab:selected { background: #6c1d45; color: #ffffff; }
        """
    )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    apply_app_stylesheet(app)
    window = CobberEcoSoilApp()
    window.show()
    sys.exit(app.exec())
