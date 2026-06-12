#!/usr/bin/env python3
"""
CobberEcoLSTM_v1.py

A guided PyQt6 application for exploring ecological sequence prediction with
Mike Bush's small-fish community dataset.

Expected data location:
    SmallFish_MockDataset.xlsx
in the same folder as this script.

Dependencies:
    pip install pandas numpy matplotlib scikit-learn PyQt6 openpyxl
Optional for the LSTM tab:
    pip install tensorflow

The app still works without TensorFlow. Baselines and sequence construction run
normally, and the Train LSTM tab will explain what is missing.
"""

from __future__ import annotations

import os

# Keep native math libraries conservative on Windows.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PyQt6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PyQt6.QtGui import QAction, QBrush, QColor, QFont
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QStatusBar,
    QTableView,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as Canvas
from matplotlib.figure import Figure

APP_TITLE = "CobberEcoLSTM"
DATA_FILE = "SmallFish_MockDataset.xlsx"
BASE_TARGETS = ["Total density", "Top 5 species vector"]
INPUT_MODES = ["Previous target only", "Previous target + ecology inputs"]
POOL_MODES = ["All sites", "Selected site"]
TRAIN_TEST_MODES = ["Hold out last window from each plot"]


def app_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def find_data_file() -> Optional[Path]:
    candidate = app_dir() / DATA_FILE
    return candidate if candidate.exists() else None


class DataFrameModel(QAbstractTableModel):
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self._df = df.copy()

    def rowCount(self, parent=QModelIndex()):
        return len(self._df)

    def columnCount(self, parent=QModelIndex()):
        return self._df.shape[1]

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        val = self._df.iat[index.row(), index.column()]
        if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole):
            if pd.isna(val):
                return ""
            if isinstance(val, (float, np.floating)):
                return f"{float(val):.4g}"
            return str(val)
        if role == Qt.ItemDataRole.ForegroundRole and pd.isna(val):
            return QBrush(QColor("red"))
        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return str(self._df.columns[section])
        return str(self._df.index[section])


class PlotCanvas(Canvas):
    def __init__(self, width=6, height=4):
        self.fig = Figure(figsize=(width, height), dpi=100)
        super().__init__(self.fig)

    def clear_with_message(self, message: str) -> None:
        self.fig.clf()
        ax = self.fig.add_subplot(111)
        ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
        ax.set_axis_off()
        self.fig.tight_layout()
        self.draw()


def read_small_fish_file(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    df.columns = [str(c).strip() for c in df.columns]
    required = ["Depth", "Depth2", "DPM", "Site", "Plot"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    species_cols = [c for c in df.columns if str(c).startswith("Species")]
    if not species_cols:
        raise ValueError("No Species columns found.")
    for col in species_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    df["Depth2"] = pd.to_numeric(df["Depth2"], errors="coerce").fillna(0.0)
    for col in ["Depth", "DPM", "Site", "Plot"]:
        df[col] = df[col].astype(str).str.strip()
    df["OriginalRow"] = np.arange(len(df))
    df["SequenceID"] = df["Site"] + " | Plot " + df["Plot"].astype(str)
    df["TimeIndex"] = df.groupby(["Site", "Plot"]).cumcount()
    df["TotalDensity"] = df[species_cols].sum(axis=1)
    return df


def species_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if str(c).startswith("Species")]


def top_species_columns(df: pd.DataFrame, n: int = 5) -> List[str]:
    cols = species_columns(df)
    means = df[cols].mean().sort_values(ascending=False)
    return means.head(n).index.tolist()


class SimpleScaler:
    """Small NumPy-only scaler to avoid scikit-learn/native-library issues."""

    def fit(self, x):
        x = np.asarray(x, dtype=float)
        self.mean_ = np.nanmean(x, axis=0)
        self.scale_ = np.nanstd(x, axis=0)
        self.scale_ = np.where(self.scale_ > 1e-12, self.scale_, 1.0)
        return self

    def transform(self, x):
        x = np.asarray(x, dtype=float)
        return (x - self.mean_) / self.scale_

    def fit_transform(self, x):
        return self.fit(x).transform(x)

    def inverse_transform(self, z):
        z = np.asarray(z, dtype=float)
        return z * self.scale_ + self.mean_


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot <= 1e-12:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


@dataclass
class SequenceBundle:
    X: np.ndarray
    y: np.ndarray
    X_scaled: np.ndarray
    y_scaled: np.ndarray
    train_idx: np.ndarray
    test_idx: np.ndarray
    feature_names: List[str]
    target_names: List[str]
    meta: pd.DataFrame
    x_scaler: SimpleScaler
    y_scaler: SimpleScaler
    target_mode: str
    input_mode: str
    window_length: int
    pool_mode: str
    selected_site: str


@dataclass
class ModelResult:
    name: str
    y_true: np.ndarray
    y_pred: np.ndarray
    meta: pd.DataFrame
    metrics: Dict[str, float]
    per_target_metrics: pd.DataFrame


@dataclass
class LSTMTrainResult:
    history: Optional[object]
    result: Optional[ModelResult]
    error_message: str = ""


class SequenceBuilder:
    @staticmethod
    def build(
        df: pd.DataFrame,
        target_mode: str,
        input_mode: str,
        window_length: int,
        pool_mode: str,
        selected_site: str,
    ) -> SequenceBundle:
        if df is None or df.empty:
            raise ValueError("No data loaded.")
        work = df.copy()
        if pool_mode == "Selected site":
            work = work[work["Site"] == selected_site].copy()
        if work.empty:
            raise ValueError("No rows available after site filtering.")

        if target_mode == "Total density":
            target_cols = ["TotalDensity"]
        elif target_mode == "Top 5 species vector":
            target_cols = top_species_columns(work, 5)
        else:
            raise ValueError(f"Unknown target mode: {target_mode}")

        feature_base_cols: List[str] = list(target_cols)
        feature_names: List[str] = list(target_cols)
        encoded = pd.DataFrame(index=work.index)
        if input_mode == "Previous target + ecology inputs":
            encoded["Depth2"] = pd.to_numeric(work["Depth2"], errors="coerce").fillna(0.0)
            for col in ["Depth", "DPM", "Site"]:
                dummies = pd.get_dummies(work[col].astype(str), prefix=col)
                encoded = pd.concat([encoded, dummies], axis=1)
            for col in encoded.columns:
                work[col] = encoded[col].values
            feature_base_cols += list(encoded.columns)
            feature_names += list(encoded.columns)

        X_list: List[np.ndarray] = []
        y_list: List[np.ndarray] = []
        meta_rows: List[Dict] = []
        train_indices: List[int] = []
        test_indices: List[int] = []
        sample_index = 0

        for (site, plot), group in work.groupby(["Site", "Plot"], sort=False):
            group = group.sort_values("TimeIndex").copy()
            if len(group) <= window_length:
                continue
            group_windows: List[int] = []
            values = group[feature_base_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
            targets = group[target_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
            rows = group.reset_index(drop=True)
            for i in range(window_length, len(group)):
                X_list.append(values[i - window_length:i, :])
                y_list.append(targets[i, :])
                meta_rows.append({
                    "Site": str(site),
                    "Plot": str(plot),
                    "SequenceID": f"{site} | Plot {plot}",
                    "TimeIndex": int(rows.loc[i, "TimeIndex"]),
                    "DPM": str(rows.loc[i, "DPM"]),
                    "Depth": str(rows.loc[i, "Depth"]),
                    "Depth2": float(rows.loc[i, "Depth2"]),
                    "OriginalRow": int(rows.loc[i, "OriginalRow"]),
                })
                group_windows.append(sample_index)
                sample_index += 1
            if len(group_windows) >= 2:
                test_indices.append(group_windows[-1])
                train_indices.extend(group_windows[:-1])
            elif len(group_windows) == 1:
                train_indices.extend(group_windows)

        if not X_list:
            raise ValueError("No sequence windows could be built. Try a shorter window length.")
        X = np.stack(X_list).astype(float)
        y = np.stack(y_list).astype(float)
        train_idx = np.array(sorted(set(train_indices)), dtype=int)
        test_idx = np.array(sorted(set(test_indices)), dtype=int)
        if len(test_idx) == 0:
            if len(X) < 2:
                raise ValueError("Not enough windows to make a train/test split.")
            test_idx = np.array([len(X) - 1], dtype=int)
            train_idx = np.array(list(range(len(X) - 1)), dtype=int)
        if len(train_idx) == 0:
            raise ValueError("No training windows available.")

        x_scaler = SimpleScaler()
        y_scaler = SimpleScaler()
        n_samples, steps, n_features = X.shape
        X_2d = X.reshape(n_samples, steps * n_features)
        x_scaler.fit(X_2d[train_idx])
        X_scaled = x_scaler.transform(X_2d).reshape(n_samples, steps, n_features)
        y_scaler.fit(y[train_idx])
        y_scaled = y_scaler.transform(y)
        meta = pd.DataFrame(meta_rows)
        return SequenceBundle(
            X=X,
            y=y,
            X_scaled=X_scaled,
            y_scaled=y_scaled,
            train_idx=train_idx,
            test_idx=test_idx,
            feature_names=feature_names,
            target_names=target_cols,
            meta=meta,
            x_scaler=x_scaler,
            y_scaler=y_scaler,
            target_mode=target_mode,
            input_mode=input_mode,
            window_length=window_length,
            pool_mode=pool_mode,
            selected_site=selected_site,
        )


class ModelRunner:
    @staticmethod
    def _metrics(y_true: np.ndarray, y_pred: np.ndarray, target_names: Sequence[str]) -> Tuple[Dict[str, float], pd.DataFrame]:
        overall = {
            "MAE": mae(y_true, y_pred),
            "RMSE": rmse(y_true, y_pred),
            "Bias": float(np.mean(y_pred - y_true)),
            "R2": safe_r2(y_true.reshape(len(y_true), -1), y_pred.reshape(len(y_pred), -1)),
            "N": int(len(y_true)),
        }
        rows = []
        for i, name in enumerate(target_names):
            rows.append({
                "Target": name,
                "MAE": mae(y_true[:, i], y_pred[:, i]),
                "RMSE": rmse(y_true[:, i], y_pred[:, i]),
                "Bias": float(np.mean(y_pred[:, i] - y_true[:, i])),
                "MeanActual": float(np.mean(y_true[:, i])),
            })
        return overall, pd.DataFrame(rows)

    @staticmethod
    def run_baselines(bundle: SequenceBundle) -> Dict[str, ModelResult]:
        """Run two stable, pedagogically essential baselines.

        Version 2 deliberately avoids scikit-learn so the app is more stable
        on classroom Windows laptops.
        """
        test = bundle.test_idx
        train = bundle.train_idx
        y_true = bundle.y[test]
        meta = bundle.meta.iloc[test].reset_index(drop=True)
        results: Dict[str, ModelResult] = {}

        mean_pred = np.tile(bundle.y[train].mean(axis=0), (len(test), 1))
        metrics, per = ModelRunner._metrics(y_true, mean_pred, bundle.target_names)
        results["Mean baseline"] = ModelResult("Mean baseline", y_true, mean_pred, meta, metrics, per)

        persistence_pred = bundle.X[test, -1, :len(bundle.target_names)]
        metrics, per = ModelRunner._metrics(y_true, persistence_pred, bundle.target_names)
        results["Persistence baseline"] = ModelResult("Persistence baseline", y_true, persistence_pred, meta, metrics, per)

        return results

    @staticmethod
    def train_lstm(bundle: SequenceBundle, units: int, epochs: int, batch_size: int) -> LSTMTrainResult:
        try:
            import tensorflow as tf
            from tensorflow import keras
            try:
                tf.config.threading.set_intra_op_parallelism_threads(1)
                tf.config.threading.set_inter_op_parallelism_threads(1)
            except Exception:
                pass
        except Exception as exc:
            return LSTMTrainResult(None, None, f"TensorFlow/Keras could not be imported. Install tensorflow to train the LSTM. Details: {exc}")
        try:
            tf.random.set_seed(42)
            np.random.seed(42)
            n_targets = bundle.y_scaled.shape[1]
            model = keras.Sequential([
                keras.layers.Input(shape=(bundle.X_scaled.shape[1], bundle.X_scaled.shape[2])),
                keras.layers.LSTM(units),
                keras.layers.Dense(n_targets),
            ])
            model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss="mse", metrics=["mae"])
            validation_data = None
            if len(bundle.test_idx) > 0:
                validation_data = (bundle.X_scaled[bundle.test_idx], bundle.y_scaled[bundle.test_idx])
            history = model.fit(
                bundle.X_scaled[bundle.train_idx],
                bundle.y_scaled[bundle.train_idx],
                epochs=epochs,
                batch_size=batch_size,
                verbose=0,
                validation_data=validation_data,
            )
            pred_scaled = model.predict(bundle.X_scaled[bundle.test_idx], verbose=0)
            pred = bundle.y_scaler.inverse_transform(pred_scaled)
            pred = np.maximum(pred, 0.0)
            y_true = bundle.y[bundle.test_idx]
            meta = bundle.meta.iloc[bundle.test_idx].reset_index(drop=True)
            metrics, per = ModelRunner._metrics(y_true, pred, bundle.target_names)
            result = ModelResult("LSTM", y_true, pred, meta, metrics, per)
            return LSTMTrainResult(history, result, "")
        except Exception as exc:
            return LSTMTrainResult(None, None, "LSTM training failed:\n" + traceback.format_exc())


class InspectPage(QWidget):
    def __init__(self, main: "CobberEcoLSTMApp"):
        super().__init__(main)
        self.main = main
        layout = QHBoxLayout(self)
        left = QWidget(); left_layout = QVBoxLayout(left)
        controls = QGroupBox("Inspect a plot sequence")
        form = QFormLayout(controls)
        self.site_combo = QComboBox()
        self.plot_combo = QComboBox()
        self.response_combo = QComboBox()
        self.response_combo.addItem("TotalDensity")
        form.addRow("Site:", self.site_combo)
        form.addRow("Plot:", self.plot_combo)
        form.addRow("Response:", self.response_combo)
        left_layout.addWidget(controls)
        self.report = QTextEdit(readOnly=True)
        self.report.setMaximumHeight(190)
        left_layout.addWidget(self.report)
        self.table = QTableView()
        left_layout.addWidget(QLabel("Data preview"))
        left_layout.addWidget(self.table, stretch=1)
        right = QWidget(); right_layout = QVBoxLayout(right)
        self.seq_plot = PlotCanvas(width=7.5, height=3.4)
        self.heat_plot = PlotCanvas(width=7.5, height=3.4)
        right_layout.addWidget(self.seq_plot, stretch=1)
        right_layout.addWidget(self.heat_plot, stretch=1)
        layout.addWidget(left, stretch=0)
        layout.addWidget(right, stretch=1)
        self.site_combo.currentTextChanged.connect(self.refresh_plots)
        self.plot_combo.currentTextChanged.connect(self.refresh)
        self.response_combo.currentTextChanged.connect(self.refresh)
        self.refresh_all_controls()

    def refresh_all_controls(self) -> None:
        df = self.main.df
        self.site_combo.blockSignals(True); self.plot_combo.blockSignals(True); self.response_combo.blockSignals(True)
        self.site_combo.clear(); self.plot_combo.clear(); self.response_combo.clear()
        self.response_combo.addItem("TotalDensity")
        if df is not None:
            self.site_combo.addItems(sorted(df["Site"].astype(str).unique().tolist()))
            self.response_combo.addItems(species_columns(df))
        self.site_combo.blockSignals(False); self.plot_combo.blockSignals(False); self.response_combo.blockSignals(False)
        self.refresh_plots()

    def refresh_plots(self) -> None:
        df = self.main.df
        self.plot_combo.blockSignals(True)
        self.plot_combo.clear()
        if df is not None and self.site_combo.currentText():
            sub = df[df["Site"] == self.site_combo.currentText()]
            self.plot_combo.addItems(sorted(sub["Plot"].astype(str).unique().tolist()))
        self.plot_combo.blockSignals(False)
        self.refresh()

    def refresh(self) -> None:
        df = self.main.df
        if df is None or df.empty:
            return
        site = self.site_combo.currentText() or str(df["Site"].iloc[0])
        plot = self.plot_combo.currentText() or str(df[df["Site"] == site]["Plot"].iloc[0])
        response = self.response_combo.currentText() or "TotalDensity"
        sub = df[(df["Site"] == site) & (df["Plot"].astype(str) == str(plot))].sort_values("TimeIndex").copy()
        self._update_report(df, sub, site, plot)
        cols = ["TimeIndex", "Depth", "Depth2", "DPM", "Site", "Plot", "TotalDensity"] + species_columns(df)[:8]
        self.table.setModel(DataFrameModel(sub[cols].reset_index(drop=True)))
        self._plot_sequence(sub, response, site, plot)
        self._plot_heatmap(sub)

    def _update_report(self, df: pd.DataFrame, sub: pd.DataFrame, site: str, plot: str) -> None:
        sp = species_columns(df)
        lines = [
            f"Rows in dataset: {len(df)}",
            f"Sites: {', '.join(sorted(df['Site'].unique()))}",
            f"Sequence groups (Site + Plot): {df.groupby(['Site','Plot']).ngroups}",
            f"Selected sequence: {site} | Plot {plot} ({len(sub)} samples)",
            f"Species columns: {len(sp)}",
            f"Top species by mean density: {', '.join(top_species_columns(df, 5))}",
            "Rows are treated as ordered sampling events within each Site + Plot sequence.",
        ]
        self.report.setPlainText("\n".join(lines))

    def _plot_sequence(self, sub: pd.DataFrame, response: str, site: str, plot: str) -> None:
        self.seq_plot.fig.clf()
        ax = self.seq_plot.fig.add_subplot(111)
        if response not in sub.columns:
            response = "TotalDensity"
        ax.plot(sub["TimeIndex"], sub[response], marker="o", linewidth=1.6)
        for label, grp in sub.groupby("DPM"):
            ax.scatter(grp["TimeIndex"], grp[response], label=str(label), s=36)
        ax.set_title(f"{response} through the ordered samples: {site}, Plot {plot}")
        ax.set_xlabel("Sample index within plot")
        ax.set_ylabel(response)
        ax.legend(fontsize=8)
        self.seq_plot.fig.tight_layout()
        self.seq_plot.draw()

    def _plot_heatmap(self, sub: pd.DataFrame) -> None:
        self.heat_plot.fig.clf()
        ax = self.heat_plot.fig.add_subplot(111)
        sp = species_columns(sub)
        mat = sub[sp].to_numpy(dtype=float)
        if mat.size == 0:
            self.heat_plot.clear_with_message("No species data.")
            return
        im = ax.imshow(mat, aspect="auto", interpolation="nearest")
        ax.set_title("Community state heatmap")
        ax.set_xlabel("Species")
        ax.set_ylabel("Sample index")
        ax.set_xticks(range(len(sp)))
        ax.set_xticklabels(sp, rotation=90, fontsize=7)
        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels(sub["TimeIndex"].astype(str), fontsize=7)
        self.heat_plot.fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="Density")
        self.heat_plot.fig.tight_layout()
        self.heat_plot.draw()


class BuildSequencesPage(QWidget):
    def __init__(self, main: "CobberEcoLSTMApp"):
        super().__init__(main)
        self.main = main
        layout = QHBoxLayout(self)
        left = QWidget(); left_layout = QVBoxLayout(left)
        controls = QGroupBox("Build learning sequences")
        form = QFormLayout(controls)
        self.target_combo = QComboBox(); self.target_combo.addItems(BASE_TARGETS)
        self.input_combo = QComboBox(); self.input_combo.addItems(INPUT_MODES)
        self.window_spin = QSpinBox(); self.window_spin.setRange(1, 4); self.window_spin.setValue(2)
        self.pool_combo = QComboBox(); self.pool_combo.addItems(POOL_MODES)
        self.site_combo = QComboBox()
        self.build_btn = QPushButton("Build Sequences")
        form.addRow("Prediction target:", self.target_combo)
        form.addRow("Input features:", self.input_combo)
        form.addRow("Window length:", self.window_spin)
        form.addRow("Pooling:", self.pool_combo)
        form.addRow("Selected site:", self.site_combo)
        form.addRow(self.build_btn)
        left_layout.addWidget(controls)
        self.report = QTextEdit(readOnly=True)
        left_layout.addWidget(self.report, stretch=1)
        right = QWidget(); right_layout = QVBoxLayout(right)
        self.diagram = QTextEdit(readOnly=True)
        self.diagram.setMaximumHeight(200)
        self.example_table = QTableView()
        self.shape_table = QTableView()
        right_layout.addWidget(QLabel("Example training window"))
        right_layout.addWidget(self.diagram)
        right_layout.addWidget(self.example_table, stretch=1)
        right_layout.addWidget(QLabel("Window summary"))
        right_layout.addWidget(self.shape_table, stretch=1)
        layout.addWidget(left, stretch=0)
        layout.addWidget(right, stretch=1)
        self.build_btn.clicked.connect(self.build_sequences)
        self.pool_combo.currentTextChanged.connect(self.refresh_site_enabled)
        self.refresh_sites()
        self.refresh_site_enabled()
        self.refresh()

    def refresh_sites(self) -> None:
        self.site_combo.clear()
        if self.main.df is not None:
            self.site_combo.addItems(sorted(self.main.df["Site"].astype(str).unique().tolist()))

    def refresh_site_enabled(self) -> None:
        self.site_combo.setEnabled(self.pool_combo.currentText() == "Selected site")

    def build_sequences(self) -> None:
        try:
            bundle = SequenceBuilder.build(
                self.main.df,
                self.target_combo.currentText(),
                self.input_combo.currentText(),
                self.window_spin.value(),
                self.pool_combo.currentText(),
                self.site_combo.currentText(),
            )
            self.main.bundle = bundle
            self.main.baseline_results = {}
            self.main.lstm_train_result = None
            self.refresh()
            # Avoid broad automatic refreshes on Windows/PyQt; tabs refresh when opened or by user action.
        # self.main.refresh_dependent_pages()
            self.main.statusBar().showMessage(f"Built {len(bundle.X)} sequence windows.")
        except Exception as exc:
            QMessageBox.critical(self, "Sequence build error", str(exc))

    def refresh(self) -> None:
        bundle = self.main.bundle
        if bundle is None:
            self.report.setPlainText("No sequences built yet. Choose settings and click Build Sequences.")
            self.diagram.setPlainText("Example will appear after sequences are built.")
            self.example_table.setModel(DataFrameModel(pd.DataFrame()))
            self.shape_table.setModel(DataFrameModel(pd.DataFrame()))
            return
        self.report.setPlainText(
            f"Target: {bundle.target_mode}\n"
            f"Input mode: {bundle.input_mode}\n"
            f"Window length: {bundle.window_length}\n"
            f"Pool mode: {bundle.pool_mode}\n"
            f"Feature count per time step: {len(bundle.feature_names)}\n"
            f"Output count: {len(bundle.target_names)}\n"
            f"Training windows: {len(bundle.train_idx)}\n"
            f"Testing windows: {len(bundle.test_idx)}\n"
            "\nSequences are built within Site + Plot. The end of one plot is never used to predict the beginning of another plot."
        )
        self._show_example(bundle)
        summary = pd.DataFrame({
            "Quantity": ["X shape", "y shape", "Train windows", "Test windows", "Feature names", "Target names"],
            "Value": [str(bundle.X.shape), str(bundle.y.shape), len(bundle.train_idx), len(bundle.test_idx), ", ".join(bundle.feature_names), ", ".join(bundle.target_names)],
        })
        self.shape_table.setModel(DataFrameModel(summary))

    def _show_example(self, bundle: SequenceBundle) -> None:
        idx = int(bundle.test_idx[0]) if len(bundle.test_idx) else 0
        meta = bundle.meta.iloc[idx]
        lines = [
            f"Input window for {meta['SequenceID']}",
            f"The LSTM reads {bundle.window_length} ordered samples.",
            f"Target: predict the next {bundle.target_mode} value(s).",
            "",
            "t - window  →  ...  →  t - 1  →  target t",
        ]
        self.diagram.setPlainText("\n".join(lines))
        rows = []
        for step in range(bundle.window_length):
            row = {"Step": f"input {step + 1}", "TimeWithinWindow": step}
            for f_i, f_name in enumerate(bundle.feature_names[:10]):
                row[f_name] = bundle.X[idx, step, f_i]
            rows.append(row)
        target_row = {"Step": "target", "TimeWithinWindow": "next"}
        for t_i, t_name in enumerate(bundle.target_names):
            target_row[t_name] = bundle.y[idx, t_i]
        rows.append(target_row)
        self.example_table.setModel(DataFrameModel(pd.DataFrame(rows)))


class BaselinesPage(QWidget):
    def __init__(self, main: "CobberEcoLSTMApp"):
        super().__init__(main)
        self.main = main
        layout = QHBoxLayout(self)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        self.run_btn = QPushButton("Run Baselines")
        self.output = QTextEdit(readOnly=True)
        self.output.setMinimumWidth(370)
        left_layout.addWidget(self.run_btn)
        left_layout.addWidget(self.output, stretch=1)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        self.pred_plot = PlotCanvas(width=7.5, height=5.8)
        right_layout.addWidget(self.pred_plot, stretch=1)

        layout.addWidget(left, stretch=0)
        layout.addWidget(right, stretch=1)
        self.run_btn.clicked.connect(self.run_baselines)
        self.refresh()

    def run_baselines(self) -> None:
        if self.main.bundle is None:
            QMessageBox.warning(self, "No sequences", "Build sequences first.")
            return
        try:
            self.main.baseline_results = ModelRunner.run_baselines(self.main.bundle)
            self.refresh()
            self.main.statusBar().showMessage("Baseline models complete.")
        except Exception as exc:
            QMessageBox.critical(self, "Baseline error", "Baseline calculation failed:\n" + traceback.format_exc())

    def refresh(self) -> None:
        results = self.main.baseline_results
        if not results:
            self.output.setPlainText(
                "Run baseline models after building sequences.\n\n"
                "The baselines ask what the LSTM must beat:\n\n"
                "Mean baseline:\n"
                "  Predict the average future value from the training windows.\n\n"
                "Persistence baseline:\n"
                "  Predict that the next ecological state looks like the current ecological state.\n\n"
                "If the LSTM does not beat persistence, that is still scientifically useful. "
                "It may mean the two-month sampling interval does not preserve enough sequence memory for this target."
            )
            self.pred_plot.clear_with_message("Run baseline models to see predictions.")
            return

        lines = []
        lines.append("Baseline models complete.\n")
        lines.append("Model comparison:\n")
        for name, result in results.items():
            m = result.metrics
            lines.append(
                f"{name}\n"
                f"  N    = {m.get('N', '')}\n"
                f"  MAE  = {m.get('MAE', float('nan')):.4g}\n"
                f"  RMSE = {m.get('RMSE', float('nan')):.4g}\n"
                f"  Bias = {m.get('Bias', float('nan')):.4g}\n"
                f"  R2   = {m.get('R2', float('nan')):.4g}\n"
            )
        lines.append("\nInterpretation:\n")
        lines.append(
            "The LSTM must be compared against these simple models. The persistence baseline is especially important. "
            "It asks whether the current community state is already a good predictor of the next state. "
            "The LSTM only adds value if reading a short sequence improves on this simple assumption."
        )
        self.output.setPlainText("\n".join(lines))

        first = results.get("Persistence baseline") or next(iter(results.values()))
        self._plot_prediction(first)

    def _plot_prediction(self, result: ModelResult) -> None:
        self.pred_plot.fig.clf()
        ax = self.pred_plot.fig.add_subplot(111)
        y_true = result.y_true[:, 0]
        y_pred = result.y_pred[:, 0]
        ax.scatter(y_true, y_pred, alpha=0.8)
        if len(y_true):
            lo = min(float(y_true.min()), float(y_pred.min()))
            hi = max(float(y_true.max()), float(y_pred.max()))
            if hi <= lo:
                hi = lo + 1.0
            ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
        ax.set_title(f"Actual vs predicted: {result.name}")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        self.pred_plot.fig.tight_layout()
        self.pred_plot.draw()


class LSTMPage(QWidget):
    def __init__(self, main: "CobberEcoLSTMApp"):
        super().__init__(main)
        self.main = main
        layout = QHBoxLayout(self)
        left = QWidget(); left_layout = QVBoxLayout(left)
        controls = QGroupBox("Train a small LSTM")
        form = QFormLayout(controls)
        self.units_spin = QSpinBox(); self.units_spin.setRange(4, 128); self.units_spin.setValue(16)
        self.epochs_spin = QSpinBox(); self.epochs_spin.setRange(1, 500); self.epochs_spin.setValue(50)
        self.batch_spin = QSpinBox(); self.batch_spin.setRange(1, 128); self.batch_spin.setValue(8)
        self.train_btn = QPushButton("Train LSTM")
        form.addRow("Hidden units:", self.units_spin)
        form.addRow("Epochs:", self.epochs_spin)
        form.addRow("Batch size:", self.batch_spin)
        form.addRow(self.train_btn)
        left_layout.addWidget(controls)
        self.explain = QTextEdit(readOnly=True)
        self.explain.setPlainText(
            "The LSTM reads the input sequence one step at a time. It updates an internal memory as it reads the window. "
            "The final memory state is used to predict the next ecological state.\n\n"
            "This model should be compared to the baselines. It is not automatically better just because it is more complex."
        )
        left_layout.addWidget(self.explain, stretch=1)
        right = QWidget(); right_layout = QVBoxLayout(right)
        self.status = QLabel("Build sequences, then train the LSTM.")
        self.status.setWordWrap(True)
        self.loss_plot = PlotCanvas(width=7, height=4)
        self.pred_plot = PlotCanvas(width=7, height=4)
        right_layout.addWidget(self.status)
        right_layout.addWidget(self.loss_plot, stretch=1)
        right_layout.addWidget(self.pred_plot, stretch=1)
        layout.addWidget(left, stretch=0)
        layout.addWidget(right, stretch=1)
        self.train_btn.clicked.connect(self.train_lstm)
        self.refresh()
        if hasattr(self.main, "evaluate_page"):
            self.main.evaluate_page.refresh()

    def train_lstm(self) -> None:
        if self.main.bundle is None:
            QMessageBox.warning(self, "No sequences", "Build sequences first.")
            return
        self.status.setText("Training LSTM. The window may pause until training finishes.")
        QApplication.processEvents()
        train_result = ModelRunner.train_lstm(
            self.main.bundle,
            units=self.units_spin.value(),
            epochs=self.epochs_spin.value(),
            batch_size=self.batch_spin.value(),
        )
        self.main.lstm_train_result = train_result
        if train_result.error_message:
            QMessageBox.warning(self, "LSTM training", train_result.error_message)
        self.refresh()
        # Avoid broad automatic refreshes on Windows/PyQt; the Evaluate tab
        # can be opened/refreshed after the model run.
        # self.main.refresh_dependent_pages()

    def refresh(self) -> None:
        tr = self.main.lstm_train_result
        if tr is None:
            self.loss_plot.clear_with_message("No LSTM training run yet.")
            self.pred_plot.clear_with_message("No LSTM predictions yet.")
            return
        if tr.error_message:
            self.status.setText(tr.error_message)
            self.loss_plot.clear_with_message("TensorFlow/Keras is required for this tab.")
            self.pred_plot.clear_with_message("No LSTM predictions.")
            return
        if tr.result is None or tr.history is None:
            return
        m = tr.result.metrics
        self.status.setText(f"LSTM complete. N={m['N']}, MAE={m['MAE']:.4g}, RMSE={m['RMSE']:.4g}, Bias={m['Bias']:.4g}")
        self._plot_loss(tr.history)
        self._plot_prediction(tr.result)

    def _plot_loss(self, history) -> None:
        self.loss_plot.fig.clf()
        ax = self.loss_plot.fig.add_subplot(111)
        h = history.history
        ax.plot(h.get("loss", []), label="training loss")
        if "val_loss" in h:
            ax.plot(h["val_loss"], label="validation loss")
        ax.set_title("Training curve")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE loss")
        ax.legend(fontsize=8)
        self.loss_plot.fig.tight_layout()
        self.loss_plot.draw()

    def _plot_prediction(self, result: ModelResult) -> None:
        self.pred_plot.fig.clf()
        ax = self.pred_plot.fig.add_subplot(111)
        y_true = result.y_true[:, 0]
        y_pred = result.y_pred[:, 0]
        ax.scatter(y_true, y_pred, alpha=0.8)
        if len(y_true):
            lo = min(y_true.min(), y_pred.min())
            hi = max(y_true.max(), y_pred.max())
            ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
        ax.set_title("Actual vs predicted: LSTM")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        self.pred_plot.fig.tight_layout()
        self.pred_plot.draw()


class EvaluatePage(QWidget):
    def __init__(self, main: "CobberEcoLSTMApp"):
        super().__init__(main)
        self.main = main
        self.metrics_model = DataFrameModel(pd.DataFrame())
        self.baci_model = DataFrameModel(pd.DataFrame())
        layout = QVBoxLayout(self)
        self.summary = QTextEdit(readOnly=True)
        self.summary.setMaximumHeight(150)
        layout.addWidget(self.summary)
        body = QHBoxLayout()
        left = QWidget(); left_layout = QVBoxLayout(left)
        self.metrics_table = QTableView()
        self.baci_table = QTableView()
        left_layout.addWidget(QLabel("All-model metrics"))
        left_layout.addWidget(self.metrics_table, stretch=1)
        left_layout.addWidget(QLabel("BACI-style error breakdown"))
        left_layout.addWidget(self.baci_table, stretch=1)
        right = QWidget(); right_layout = QVBoxLayout(right)
        self.plot = PlotCanvas(width=7, height=6)
        right_layout.addWidget(self.plot, stretch=1)
        body.addWidget(left, stretch=0)
        body.addWidget(right, stretch=1)
        layout.addLayout(body, stretch=1)
        self.refresh()

    def all_results(self) -> Dict[str, ModelResult]:
        out = dict(self.main.baseline_results)
        if self.main.lstm_train_result and self.main.lstm_train_result.result:
            out["LSTM"] = self.main.lstm_train_result.result
        return out

    def refresh(self) -> None:
        results = self.all_results()
        if not results:
            self.summary.setPlainText("Run baselines and optionally train the LSTM to compare models.")
            self.metrics_model = DataFrameModel(pd.DataFrame())
            self.metrics_table.setModel(self.metrics_model)

            self.baci_model = DataFrameModel(pd.DataFrame())
            self.baci_table.setModel(self.baci_model)
            self.plot.clear_with_message("No model results yet.")
            return
        rows = []
        for name, result in results.items():
            row = {"Model": name}
            row.update(result.metrics)
            rows.append(row)
        metrics_df = pd.DataFrame(rows)
        self.metrics_model = DataFrameModel(metrics_df)
        self.metrics_table.setModel(self.metrics_model)

        self.baci_model = DataFrameModel(self._baci_breakdown(results))
        self.baci_table.setModel(self.baci_model)
        self.summary.setPlainText(
            "Compare the LSTM to simple baselines. If the LSTM does not beat persistence, that is not a failure of the lab. "
            "It may mean that recent history does not add much beyond the current state for this target and window length. "
            "Look especially at Pre/Post and Site differences to ask whether predictability changes with the BACI design."
        )
        self._plot_model_mae(metrics_df)

    def _baci_breakdown(self, results: Dict[str, ModelResult]) -> pd.DataFrame:
        rows = []
        for name, result in results.items():
            meta = result.meta.reset_index(drop=True)
            err = np.abs(result.y_pred[:, 0] - result.y_true[:, 0])
            temp = meta.copy()
            temp["AbsError"] = err
            for group_col in ["DPM", "Site"]:
                for group, g in temp.groupby(group_col):
                    rows.append({"GroupType": group_col, "Group": group, "Model": name, "MAE target 1": float(g["AbsError"].mean()), "N": int(len(g))})
        return pd.DataFrame(rows)

    def _plot_model_mae(self, metrics_df: pd.DataFrame) -> None:
        self.plot.fig.clf()
        ax = self.plot.fig.add_subplot(111)
        if metrics_df.empty:
            self.plot.clear_with_message("No metrics.")
            return
        ax.bar(metrics_df["Model"], metrics_df["MAE"])
        ax.set_title("Model comparison by MAE")
        ax.set_ylabel("MAE")
        ax.tick_params(axis="x", labelrotation=20)
        self.plot.fig.tight_layout()
        self.plot.draw()


class ExportPage(QWidget):
    def __init__(self, main: "CobberEcoLSTMApp"):
        super().__init__(main)
        self.main = main
        layout = QVBoxLayout(self)
        self.pred_btn = QPushButton("Save predictions CSV")
        self.metrics_btn = QPushButton("Save metrics CSV")
        self.sequence_btn = QPushButton("Save sequence metadata CSV")
        layout.addWidget(self.pred_btn)
        layout.addWidget(self.metrics_btn)
        layout.addWidget(self.sequence_btn)
        layout.addStretch()
        self.pred_btn.clicked.connect(self.save_predictions)
        self.metrics_btn.clicked.connect(self.save_metrics)
        self.sequence_btn.clicked.connect(self.save_sequence_metadata)

    def all_results(self) -> Dict[str, ModelResult]:
        out = dict(self.main.baseline_results)
        if self.main.lstm_train_result and self.main.lstm_train_result.result:
            out["LSTM"] = self.main.lstm_train_result.result
        return out

    def save_predictions(self) -> None:
        results = self.all_results()
        if not results:
            QMessageBox.warning(self, "No results", "Run at least one model first.")
            return
        first = next(iter(results.values()))
        out = first.meta.copy()
        for t_i, target in enumerate(self.main.bundle.target_names if self.main.bundle else ["Target"]):
            out[f"{target}_actual"] = first.y_true[:, t_i]
        for name, result in results.items():
            safe = name.replace(" ", "_")
            for t_i, target in enumerate(self.main.bundle.target_names if self.main.bundle else ["Target"]):
                out[f"{target}_{safe}_pred"] = result.y_pred[:, t_i]
                out[f"{target}_{safe}_error"] = result.y_pred[:, t_i] - result.y_true[:, t_i]
        fname, _ = QFileDialog.getSaveFileName(self, "Save predictions", "cobber_ecolstm_predictions.csv", "CSV files (*.csv)")
        if fname:
            out.to_csv(fname, index=False)
            self.main.statusBar().showMessage(f"Saved {fname}")

    def save_metrics(self) -> None:
        results = self.all_results()
        if not results:
            QMessageBox.warning(self, "No results", "Run at least one model first.")
            return
        rows = []
        for name, result in results.items():
            row = {"Model": name}
            row.update(result.metrics)
            rows.append(row)
        fname, _ = QFileDialog.getSaveFileName(self, "Save metrics", "cobber_ecolstm_metrics.csv", "CSV files (*.csv)")
        if fname:
            pd.DataFrame(rows).to_csv(fname, index=False)
            self.main.statusBar().showMessage(f"Saved {fname}")

    def save_sequence_metadata(self) -> None:
        if self.main.bundle is None:
            QMessageBox.warning(self, "No sequences", "Build sequences first.")
            return
        fname, _ = QFileDialog.getSaveFileName(self, "Save sequence metadata", "cobber_ecolstm_sequence_metadata.csv", "CSV files (*.csv)")
        if fname:
            self.main.bundle.meta.to_csv(fname, index=False)
            self.main.statusBar().showMessage(f"Saved {fname}")


class CobberEcoLSTMApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1400, 770)
        self.setFont(QFont("Lato", 10))
        self.cobber_maroon = QColor(108, 29, 69)
        self.cobber_gold = QColor(234, 170, 0)
        self.source_path: Optional[Path] = None
        self.df: Optional[pd.DataFrame] = None
        self.bundle: Optional[SequenceBundle] = None
        self.baseline_results: Dict[str, ModelResult] = {}
        self.lstm_train_result: Optional[LSTMTrainResult] = None
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        self.setStatusBar(QStatusBar())
        self._build_menu()
        self.load_default_data()
        self._build_pages()
        self.tabs.currentChanged.connect(self.refresh_current_tab)
        self.statusBar().showMessage(f"Ready. Loaded {self.source_path}" if self.source_path else "Ready. No data file loaded.")
    def _build_menu(self) -> None:
        menu = self.menuBar()
        file_menu = menu.addMenu("File")
        open_action = QAction("Open Small Fish Dataset...", self)
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)
        reload_action = QAction("Reload Data", self)
        reload_action.triggered.connect(self.reload_data)
        file_menu.addAction(reload_action)
        file_menu.addSeparator()
        quit_action = QAction("Quit", self)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

    def refresh_current_tab(self, index: int) -> None:
        page = self.tabs.widget(index)
        if page is not None and hasattr(page, "refresh"):
            page.refresh()
    def load_default_data(self) -> None:
        path = find_data_file()
        if path is not None:
            self.load_data(path)

    def load_data(self, path: Path) -> None:
        self.df = read_small_fish_file(path)
        self.source_path = path
        self.bundle = None
        self.baseline_results = {}
        self.lstm_train_result = None

    def open_file(self) -> None:
        fname, _ = QFileDialog.getOpenFileName(self, "Open Small Fish Dataset", str(app_dir()), "Excel files (*.xlsx *.xls);;CSV files (*.csv);;All files (*.*)")
        if not fname:
            return
        try:
            path = Path(fname)
            if path.suffix.lower() in [".csv", ".txt"]:
                df = pd.read_csv(path)
                temp = app_dir() / "_temp_smallfish_open.csv"
                df.to_csv(temp, index=False)
                raise ValueError("Please use the Excel .xlsx version for this first app version.")
            self.load_data(path)
            self._build_pages()
            self.statusBar().showMessage(f"Loaded {path.name}")
        except Exception as exc:
            QMessageBox.critical(self, "Load error", f"Could not load file:\n{exc}")

    def reload_data(self) -> None:
        if self.source_path is None:
            self.load_default_data()
        elif self.source_path.exists():
            self.load_data(self.source_path)
        self._build_pages()
        self.statusBar().showMessage("Data reloaded.")

    def _build_pages(self) -> None:
        self.tabs.clear()
        if self.df is None:
            msg = QWidget(); layout = QVBoxLayout(msg)
            label = QLabel(f"Could not find {DATA_FILE} in the same folder as this program. Use File > Open Small Fish Dataset.")
            label.setWordWrap(True)
            layout.addWidget(label)
            self.tabs.addTab(msg, "No Data")
            return
        self.inspect_page = InspectPage(self)
        self.build_page = BuildSequencesPage(self)
        self.baseline_page = BaselinesPage(self)
        self.lstm_page = LSTMPage(self)
        self.evaluate_page = EvaluatePage(self)
        self.export_page = ExportPage(self)
        self.tabs.addTab(self.inspect_page, "Inspect Data")
        self.tabs.addTab(self.build_page, "Build Sequences")
        self.tabs.addTab(self.baseline_page, "Baselines")
        self.tabs.addTab(self.lstm_page, "Train LSTM")
        self.tabs.addTab(self.evaluate_page, "Evaluate")
        self.tabs.addTab(self.export_page, "Save / Export")

    def refresh_dependent_pages(self) -> None:
        for attr in ["build_page", "baseline_page", "lstm_page", "evaluate_page"]:
            page = getattr(self, attr, None)
            if page is not None and hasattr(page, "refresh"):
                page.refresh()


def apply_app_stylesheet(app: QApplication) -> None:
    app.setStyleSheet("""
        QWidget { color: #222222; background-color: #ffffff; }
        QMainWindow, QDialog { background-color: #ffffff; }
        QMenuBar, QMenu { background-color: #ffffff; color: #222222; }
        QMenuBar::item:selected, QMenu::item:selected { background-color: #6c1d45; color: #ffffff; }
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
        QLabel { color: #222222; background-color: transparent; }
        QComboBox, QSpinBox {
            background-color: #ffffff;
            color: #111111;
            border: 1px solid #9a9a9a;
            border-radius: 3px;
            padding: 3px 6px;
            min-height: 24px;
            selection-background-color: #6c1d45;
            selection-color: #ffffff;
        }
        QComboBox:disabled, QSpinBox:disabled { background-color: #eeeeee; color: #555555; }
        QComboBox QAbstractItemView {
            background-color: #ffffff;
            color: #111111;
            selection-background-color: #6c1d45;
            selection-color: #ffffff;
            border: 1px solid #9a9a9a;
        }
        QTableView, QTextEdit {
            background-color: #ffffff;
            color: #111111;
            alternate-background-color: #f4f4f4;
            selection-background-color: #6c1d45;
            selection-color: #ffffff;
            border: 1px solid #cfcfcf;
        }
        QHeaderView::section {
            background-color: #eeeeee;
            color: #222222;
            padding: 4px;
            border: 1px solid #cccccc;
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
        QStatusBar { background-color: #f7f7f7; color: #222222; }
    """)


def main() -> None:
    app = QApplication(sys.argv)
    apply_app_stylesheet(app)
    win = CobberEcoLSTMApp()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
