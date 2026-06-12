#!/usr/bin/env python3
"""
CobberEcoLSTMbasic.py

A glass-box recurrent-memory teaching app for the Models Domain LSTM chapter.

This app intentionally avoids TensorFlow and scikit-learn. It implements a tiny
one-dimensional "LSTM-like" memory cell from scratch:

    c_t = f c_{t-1} + i x_t
    h_t = o tanh(c_t)
    yhat_{t+1} = b + w h_t

where x_t is a discrete ecological trend value:
    -1 = decreasing water
     0 = stable water
    +1 = increasing water

The app shows:
    * how a time series is broken into sliding windows
    * how one window is processed step by step
    * how a continuous yhat becomes a rounded discrete trend prediction
    * how output-layer parameters w,b can be trained with analytic gradients
    * how memory parameters f,i,o can be trained with finite-difference gradients

Dependencies:
    pip install PyQt6 numpy matplotlib

Run:
    python CobberEcoLSTMbasic.py
"""

from __future__ import annotations

import math
import sys
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as Canvas
from matplotlib.figure import Figure


APP_TITLE = "CobberEcoLSTMbasic"

PRESETS: Dict[str, List[int]] = {
    "Increasing then decreasing": [1, 1, 0, -1, -1, 0, 1, 1, 0, -1],
    "Alternating": [1, -1, 1, -1, 1, -1, 0, 1, -1, 0],
    "Stable then rising": [0, 0, 0, 1, 1, 1, 0, -1, 0, 1],
    "Long decrease": [1, 0, -1, -1, -1, 0, 0, 1, 0, -1],
}

DEFAULT_PARAMS = {
    "f": 0.60,  # forget / keep old memory
    "i": 0.40,  # write new input
    "o": 1.00,  # expose memory
    "w": 1.00,  # output weight
    "b": 0.00,  # output bias
}


@dataclass
class Params:
    f: float = 0.60
    i: float = 0.40
    o: float = 1.00
    w: float = 1.00
    b: float = 0.00

    def copy(self) -> "Params":
        return deepcopy(self)

    def as_dict(self) -> Dict[str, float]:
        return {"f": self.f, "i": self.i, "o": self.o, "w": self.w, "b": self.b}

    def get(self, name: str) -> float:
        return getattr(self, name)

    def set(self, name: str, value: float) -> None:
        setattr(self, name, float(value))


@dataclass
class ForwardStep:
    step: int
    x: float
    c_prev: float
    old_memory: float
    new_input: float
    c: float
    h: float


@dataclass
class ForwardResult:
    steps: List[ForwardStep]
    h_final: float
    yhat: float
    xhat: int
    y: float
    loss: float


class TinyMemoryCell:
    @staticmethod
    def classify_yhat(yhat: float) -> int:
        if yhat < -0.5:
            return -1
        if yhat > 0.5:
            return 1
        return 0

    @staticmethod
    def forward(inputs: List[int], target: int, params: Params) -> ForwardResult:
        c_prev = 0.0
        steps: List[ForwardStep] = []
        h = 0.0
        for step_index, x in enumerate(inputs, start=1):
            old_memory = params.f * c_prev
            new_input = params.i * float(x)
            c = old_memory + new_input
            h = params.o * math.tanh(c)
            steps.append(ForwardStep(step_index, float(x), c_prev, old_memory, new_input, c, h))
            c_prev = c

        yhat = params.b + params.w * h
        xhat = TinyMemoryCell.classify_yhat(yhat)
        y = float(target)
        loss = (yhat - y) ** 2
        return ForwardResult(steps, h, yhat, xhat, y, loss)

    @staticmethod
    def build_windows(sequence: List[int], window_len: int) -> List[Tuple[int, List[int], int]]:
        windows: List[Tuple[int, List[int], int]] = []
        for start in range(0, len(sequence) - window_len):
            inputs = sequence[start : start + window_len]
            target = sequence[start + window_len]
            windows.append((start, inputs, target))
        return windows

    @staticmethod
    def all_predictions(sequence: List[int], window_len: int, params: Params) -> List[Dict[str, object]]:
        rows: List[Dict[str, object]] = []
        for w_index, (start, inputs, target) in enumerate(TinyMemoryCell.build_windows(sequence, window_len), start=1):
            fr = TinyMemoryCell.forward(inputs, target, params)
            rows.append({
                "Window": w_index,
                "Start": start + 1,
                "Input": inputs,
                "Target": target,
                "yhat": fr.yhat,
                "xhat": fr.xhat,
                "Loss": fr.loss,
                "h_final": fr.h_final,
            })
        return rows

    @staticmethod
    def mean_loss(sequence: List[int], window_len: int, params: Params) -> float:
        preds = TinyMemoryCell.all_predictions(sequence, window_len, params)
        if not preds:
            return float("nan")
        return float(np.mean([float(p["Loss"]) for p in preds]))

    @staticmethod
    def accuracy(sequence: List[int], window_len: int, params: Params) -> float:
        preds = TinyMemoryCell.all_predictions(sequence, window_len, params)
        if not preds:
            return float("nan")
        return float(np.mean([1.0 if int(p["xhat"]) == int(p["Target"]) else 0.0 for p in preds]))

    @staticmethod
    def output_gradients(sequence: List[int], window_len: int, params: Params) -> Tuple[float, float, float]:
        preds = []
        for _, inputs, target in TinyMemoryCell.build_windows(sequence, window_len):
            preds.append(TinyMemoryCell.forward(inputs, target, params))
        if not preds:
            return 0.0, 0.0, float("nan")
        grad_w = 0.0
        grad_b = 0.0
        losses = []
        for fr in preds:
            error = fr.yhat - fr.y
            grad_w += 2.0 * error * fr.h_final
            grad_b += 2.0 * error
            losses.append(fr.loss)
        n = float(len(preds))
        return grad_w / n, grad_b / n, float(np.mean(losses))

    @staticmethod
    def finite_difference_gradients(sequence: List[int], window_len: int, params: Params, epsilon: float = 1.0e-4) -> Tuple[Dict[str, float], float]:
        base_loss = TinyMemoryCell.mean_loss(sequence, window_len, params)
        grads: Dict[str, float] = {}
        for name in ["f", "i", "o", "w", "b"]:
            p_plus = params.copy()
            p_minus = params.copy()
            p_plus.set(name, p_plus.get(name) + epsilon)
            p_minus.set(name, p_minus.get(name) - epsilon)
            loss_plus = TinyMemoryCell.mean_loss(sequence, window_len, p_plus)
            loss_minus = TinyMemoryCell.mean_loss(sequence, window_len, p_minus)
            grads[name] = float((loss_plus - loss_minus) / (2.0 * epsilon))
        return grads, base_loss


class PlotCanvas(Canvas):
    def __init__(self, width: float = 6.0, height: float = 3.0):
        self.fig = Figure(figsize=(width, height), dpi=100)
        super().__init__(self.fig)

    def clear_with_message(self, message: str) -> None:
        self.fig.clf()
        ax = self.fig.add_subplot(111)
        ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
        ax.set_axis_off()
        self.fig.tight_layout()
        self.draw()


def fmt(x: object, ndigits: int = 4) -> str:
    try:
        if isinstance(x, int):
            return str(x)
        xf = float(x)
        return f"{xf:.{ndigits}g}"
    except Exception:
        return str(x)


def make_item(text: object, align_right: bool = False) -> QTableWidgetItem:
    item = QTableWidgetItem(fmt(text))
    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
    if align_right:
        item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    return item


def fill_table(table: QTableWidget, rows: List[Dict[str, object]], columns: List[str]) -> None:
    table.clear()
    table.setColumnCount(len(columns))
    table.setRowCount(len(rows))
    table.setHorizontalHeaderLabels(columns)
    for r, row in enumerate(rows):
        for c, col in enumerate(columns):
            value = row.get(col, "")
            if isinstance(value, list):
                value = ", ".join(f"{int(v):+d}" if int(v) > 0 else str(int(v)) for v in value)
            item = make_item(value, align_right=True)
            table.setItem(r, c, item)
    table.resizeColumnsToContents()


class SequenceWindowsTab(QWidget):
    def __init__(self, main: "CobberEcoLSTMbasicApp"):
        super().__init__(main)
        self.main = main
        layout = QVBoxLayout(self)
        self.window_table = QTableWidget()
        layout.addWidget(self.window_table, stretch=1)
        note = QLabel("Each row in this table is one training example. The input window is the recent history, and the target is the next value in the sequence.")
        note.setWordWrap(True)
        layout.addWidget(note)
        self.window_table.itemSelectionChanged.connect(self._selection_changed)
        self.refresh()

    def _selection_changed(self) -> None:
        rows = self.window_table.selectionModel().selectedRows()
        if rows:
            self.main.selected_window_index = rows[0].row()
            self.main.refresh_left_panel()
            self.main.refresh_all_tabs(skip=self)

    def refresh(self) -> None:
        seq = self.main.sequence
        win_len = self.main.window_length
        selected = self.main.selected_window_index
        windows = TinyMemoryCell.build_windows(seq, win_len)
        rows = []
        for wnum, (start, inputs, target) in enumerate(windows, start=1):
            rows.append({
                "Window": wnum,
                "Time indices": f"t{start+1} ... t{start+win_len}",
                "Input sequence": inputs,
                "Target next trend": target,
            })
        fill_table(self.window_table, rows, ["Window", "Time indices", "Input sequence", "Target next trend"])
        if windows and selected < len(windows):
            self.window_table.selectRow(selected)


class ForwardPassTab(QWidget):
    def __init__(self, main: "CobberEcoLSTMbasicApp"):
        super().__init__(main)
        self.main = main
        layout = QVBoxLayout(self)
        self.forward_table = QTableWidget()
        layout.addWidget(self.forward_table, stretch=2)
        lower = QHBoxLayout()
        self.memory_plot = PlotCanvas(width=5.6, height=3.0)
        self.prediction_box = QTextEdit(readOnly=True)
        self.prediction_box.setMinimumWidth(350)
        lower.addWidget(self.memory_plot, stretch=1)
        lower.addWidget(self.prediction_box, stretch=0)
        layout.addLayout(lower, stretch=1)
        self.refresh()

    def refresh(self) -> None:
        window = self.main.current_window()
        if window is None:
            self.forward_table.clear()
            self.memory_plot.clear_with_message("No window available.")
            self.prediction_box.setPlainText("Choose a longer sequence or shorter window.")
            return
        start, inputs, target = window
        params = self.main.params
        fr = TinyMemoryCell.forward(inputs, target, params)
        rows = []
        for s in fr.steps:
            rows.append({
                "Step": s.step, "x_t": s.x, "c_(t-1)": s.c_prev, "f c_(t-1)": s.old_memory,
                "i x_t": s.new_input, "c_t": s.c, "h_t = o tanh(c_t)": s.h,
            })
        fill_table(self.forward_table, rows, ["Step", "x_t", "c_(t-1)", "f c_(t-1)", "i x_t", "c_t", "h_t = o tanh(c_t)"])
        self.memory_plot.fig.clf()
        ax = self.memory_plot.fig.add_subplot(111)
        xs = [s.step for s in fr.steps]
        c_vals = [s.c for s in fr.steps]
        h_vals = [s.h for s in fr.steps]
        ax.plot(xs, c_vals, marker="o", label="cell state c_t")
        ax.plot(xs, h_vals, marker="s", label="hidden state h_t")
        ax.axhline(0, linewidth=1, linestyle="--", color="0.55")
        ax.set_title("Memory and hidden state through the window")
        ax.set_xlabel("Step in selected window")
        ax.set_ylabel("Value")
        ax.legend(fontsize=8)
        self.memory_plot.fig.tight_layout()
        self.memory_plot.draw()
        input_str = ", ".join(f"{v:+d}" if v > 0 else str(v) for v in inputs)
        self.prediction_box.setHtml(f"""
            <h3>Prediction from selected window</h3>
            <p><b>Input window:</b> [{input_str}]</p>
            <p><b>Target:</b> y = x<sub>next</sub> = {target:+d}</p>
            <p><b>Continuous output:</b><br>
            y&#770; = b + w h = {fr.yhat:.5f}</p>
            <p><b>Discrete prediction:</b><br>
            x&#770; = {fr.xhat:+d}</p>
            <p><b>Loss:</b><br>
            (y&#770; - y)<sup>2</sup> = {fr.loss:.5f}</p>
            <hr>
            <p>The model learns from the continuous output y&#770;. The rounded trend x&#770; is useful for interpretation, but it is too coarse for gradient-based learning.</p>
        """)


class TrainOutputTab(QWidget):
    def __init__(self, main: "CobberEcoLSTMbasicApp"):
        super().__init__(main)
        self.main = main
        layout = QVBoxLayout(self)
        controls = QHBoxLayout()
        self.train_one_btn = QPushButton("Train output layer: 1 step")
        self.train_ten_btn = QPushButton("Train output layer: 10 steps")
        self.reset_btn = QPushButton("Reset w,b")
        controls.addWidget(self.train_one_btn)
        controls.addWidget(self.train_ten_btn)
        controls.addWidget(self.reset_btn)
        controls.addStretch()
        layout.addLayout(controls)
        top = QHBoxLayout()
        self.update_table = QTableWidget()
        self.loss_plot = PlotCanvas(width=5.4, height=2.8)
        top.addWidget(self.update_table, stretch=1)
        top.addWidget(self.loss_plot, stretch=1)
        layout.addLayout(top, stretch=1)
        self.prediction_table = QTableWidget()
        layout.addWidget(QLabel("Predictions across all windows"))
        layout.addWidget(self.prediction_table, stretch=1)
        self.train_one_btn.clicked.connect(lambda: self.main.train_output_steps(1))
        self.train_ten_btn.clicked.connect(lambda: self.main.train_output_steps(10))
        self.reset_btn.clicked.connect(self.reset_output)
        self.refresh()

    def reset_output(self) -> None:
        self.main.params.w = DEFAULT_PARAMS["w"]
        self.main.params.b = DEFAULT_PARAMS["b"]
        self.main.output_loss_history = []
        self.main.snapshot_params("Output reset")
        self.main.refresh_everything()

    def refresh(self) -> None:
        params = self.main.params
        gw, gb, loss = TinyMemoryCell.output_gradients(self.main.sequence, self.main.window_length, params)
        lr = self.main.learning_rate
        rows = [
            {"Parameter": "w", "Current": params.w, "Gradient": gw, "Update -lr*grad": -lr * gw, "If trained one step": params.w - lr * gw},
            {"Parameter": "b", "Current": params.b, "Gradient": gb, "Update -lr*grad": -lr * gb, "If trained one step": params.b - lr * gb},
            {"Parameter": "Mean loss", "Current": loss, "Gradient": "", "Update -lr*grad": "", "If trained one step": ""},
        ]
        fill_table(self.update_table, rows, ["Parameter", "Current", "Gradient", "Update -lr*grad", "If trained one step"])
        self.loss_plot.fig.clf()
        ax = self.loss_plot.fig.add_subplot(111)
        hist = self.main.output_loss_history
        if hist:
            ax.plot(range(1, len(hist) + 1), hist, marker="o")
            ax.set_title("Output-layer training loss")
            ax.set_xlabel("Training step")
            ax.set_ylabel("Mean squared error")
        else:
            ax.text(0.5, 0.5, "Train w,b to see a loss curve.", ha="center", va="center", wrap=True)
            ax.set_axis_off()
        self.loss_plot.fig.tight_layout()
        self.loss_plot.draw()
        rows = []
        for pred in TinyMemoryCell.all_predictions(self.main.sequence, self.main.window_length, params):
            rows.append({"Window": pred["Window"], "Input": pred["Input"], "Target": pred["Target"], "yhat": pred["yhat"], "xhat": pred["xhat"], "Loss": pred["Loss"]})
        fill_table(self.prediction_table, rows, ["Window", "Input", "Target", "yhat", "xhat", "Loss"])


class TrainMemoryTab(QWidget):
    def __init__(self, main: "CobberEcoLSTMbasicApp"):
        super().__init__(main)
        self.main = main
        layout = QVBoxLayout(self)
        controls = QHBoxLayout()
        self.train_one_btn = QPushButton("Train memory cell: 1 step")
        self.train_ten_btn = QPushButton("Train memory cell: 10 steps")
        self.reset_btn = QPushButton("Reset all parameters")
        self.constrain_box = QCheckBox("Constrain f,i,o to [0,1]")
        self.constrain_box.setChecked(True)
        controls.addWidget(self.train_one_btn)
        controls.addWidget(self.train_ten_btn)
        controls.addWidget(self.reset_btn)
        controls.addWidget(self.constrain_box)
        controls.addStretch()
        layout.addLayout(controls)
        top = QHBoxLayout()
        self.param_table = QTableWidget()
        self.loss_plot = PlotCanvas(width=5.4, height=2.8)
        top.addWidget(self.param_table, stretch=1)
        top.addWidget(self.loss_plot, stretch=1)
        layout.addLayout(top, stretch=1)
        self.param_plot = PlotCanvas(width=10, height=3.0)
        layout.addWidget(self.param_plot, stretch=1)
        self.train_one_btn.clicked.connect(lambda: self.main.train_memory_steps(1, self.constrain_box.isChecked()))
        self.train_ten_btn.clicked.connect(lambda: self.main.train_memory_steps(10, self.constrain_box.isChecked()))
        self.reset_btn.clicked.connect(self.main.reset_all_parameters)
        self.refresh()

    def refresh(self) -> None:
        params = self.main.params
        grads, loss = TinyMemoryCell.finite_difference_gradients(self.main.sequence, self.main.window_length, params, epsilon=self.main.epsilon)
        lr = self.main.learning_rate
        roles = {"f": "keep old memory", "i": "write new input", "o": "expose memory", "w": "output weight", "b": "output bias"}
        rows = []
        for name in ["f", "i", "o", "w", "b"]:
            current = params.get(name)
            grad = grads[name]
            rows.append({"Parameter": name, "Role": roles[name], "Current": current, "Gradient": grad, "Update -lr*grad": -lr * grad, "If trained one step": current - lr * grad})
        rows.append({"Parameter": "Mean loss", "Role": "", "Current": loss, "Gradient": "", "Update -lr*grad": "", "If trained one step": ""})
        fill_table(self.param_table, rows, ["Parameter", "Role", "Current", "Gradient", "Update -lr*grad", "If trained one step"])
        self.loss_plot.fig.clf()
        ax = self.loss_plot.fig.add_subplot(111)
        hist = self.main.memory_loss_history
        if hist:
            ax.plot(range(1, len(hist) + 1), hist, marker="o")
            ax.set_title("Memory-cell training loss")
            ax.set_xlabel("Training step")
            ax.set_ylabel("Mean squared error")
        else:
            ax.text(0.5, 0.5, "Train f,i,o,w,b to see a loss curve.", ha="center", va="center", wrap=True)
            ax.set_axis_off()
        self.loss_plot.fig.tight_layout()
        self.loss_plot.draw()
        self.param_plot.fig.clf()
        ax2 = self.param_plot.fig.add_subplot(111)
        ph = self.main.param_history
        if ph:
            xs = list(range(len(ph)))
            for name in ["f", "i", "o", "w", "b"]:
                ax2.plot(xs, [entry[name] for entry in ph], marker="o", label=name)
            ax2.set_title("Parameter history")
            ax2.set_xlabel("Training record")
            ax2.set_ylabel("Parameter value")
            ax2.legend(fontsize=8, ncol=5)
        else:
            ax2.text(0.5, 0.5, "Parameter history will appear after training.", ha="center", va="center")
            ax2.set_axis_off()
        self.param_plot.fig.tight_layout()
        self.param_plot.draw()


class CompareReflectTab(QWidget):
    def __init__(self, main: "CobberEcoLSTMbasicApp"):
        super().__init__(main)
        self.main = main
        layout = QVBoxLayout(self)
        top = QHBoxLayout()
        self.comparison_table = QTableWidget()
        self.prediction_table = QTableWidget()
        top.addWidget(self.comparison_table, stretch=1)
        top.addWidget(self.prediction_table, stretch=2)
        layout.addLayout(top, stretch=2)
        self.prompts = QTextEdit(readOnly=True)
        self.prompts.setHtml("""
            <h3>Reflection prompts</h3>
            <ol>
              <li>Did training reduce the continuous loss?</li>
              <li>Did training improve the rounded category predictions?</li>
              <li>Which parameters changed the most?</li>
              <li>What does <b>f</b> do when it gets larger?</li>
              <li>What does <b>i</b> do when it gets larger?</li>
              <li>Why do we learn from y&#770; rather than from the rounded x&#770;?</li>
              <li>How is this simplified model different from a real LSTM?</li>
            </ol>
            <p>A real LSTM computes f<sub>t</sub>, i<sub>t</sub>, and o<sub>t</sub> from learned weights and biases at each time step. This app uses one global f, i, and o so that you can first see what memory controls do.</p>
        """)
        layout.addWidget(self.prompts, stretch=1)
        self.refresh()

    def refresh(self) -> None:
        rows = []
        for label, params in self.main.snapshots.items():
            rows.append({"Model state": label, "MSE": TinyMemoryCell.mean_loss(self.main.sequence, self.main.window_length, params), "Rounded accuracy": TinyMemoryCell.accuracy(self.main.sequence, self.main.window_length, params), "f": params.f, "i": params.i, "o": params.o, "w": params.w, "b": params.b})
        fill_table(self.comparison_table, rows, ["Model state", "MSE", "Rounded accuracy", "f", "i", "o", "w", "b"])
        windows = TinyMemoryCell.build_windows(self.main.sequence, self.main.window_length)
        rows2 = []
        for idx, (_, inputs, target) in enumerate(windows, start=1):
            row = {"Window": idx, "Input": inputs, "Target": target}
            for label, params in self.main.snapshots.items():
                fr = TinyMemoryCell.forward(inputs, target, params)
                row[f"{label} yhat"] = fr.yhat
                row[f"{label} xhat"] = fr.xhat
            rows2.append(row)
        cols = ["Window", "Input", "Target"]
        for label in self.main.snapshots.keys():
            cols.extend([f"{label} yhat", f"{label} xhat"])
        fill_table(self.prediction_table, rows2, cols)


class CobberEcoLSTMbasicApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(1400, 770)
        self.setFont(QFont("Lato", 10))
        self.sequence: List[int] = PRESETS["Increasing then decreasing"].copy()
        self.window_length: int = 4
        self.selected_window_index: int = 0
        self.learning_rate: float = 0.05
        self.epsilon: float = 1.0e-4
        self.params = Params()
        self.snapshots: Dict[str, Params] = {"Initial": self.params.copy()}
        self.output_loss_history: List[float] = []
        self.memory_loss_history: List[float] = []
        self.param_history: List[Dict[str, float]] = [self.params.as_dict()]

        # During construction, several widgets populate themselves and may emit
        # signals before all tabs exist.  Keep cross-tab refreshes disabled until
        # the complete UI has been built.  This avoids hard Qt crashes on Windows.
        self._ui_ready = False
        self._build_ui()
        self.setStatusBar(QStatusBar())
        self._ui_ready = True
        self.refresh_everything()
        self.statusBar().showMessage("Ready.")

    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.addLayout(self._build_top_bar())
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.left_panel = self._build_left_panel()
        self.tabs = QTabWidget()
        self.seq_tab = SequenceWindowsTab(self)
        self.forward_tab = ForwardPassTab(self)
        self.output_tab = TrainOutputTab(self)
        self.memory_tab = TrainMemoryTab(self)
        self.compare_tab = CompareReflectTab(self)
        self.tabs.addTab(self.seq_tab, "1. Sequence Windows")
        self.tabs.addTab(self.forward_tab, "2. Forward Pass")
        self.tabs.addTab(self.output_tab, "3. Train Output Layer")
        self.tabs.addTab(self.memory_tab, "4. Train Memory Cell")
        self.tabs.addTab(self.compare_tab, "5. Compare / Reflect")
        splitter.addWidget(self.left_panel)
        splitter.addWidget(self.tabs)
        splitter.setSizes([360, 1040])
        root.addWidget(splitter, stretch=1)
        self.setCentralWidget(central)

    def _build_top_bar(self) -> QHBoxLayout:
        top = QHBoxLayout()
        top.addWidget(QLabel("Window length:"))
        self.window_spin = QSpinBox()
        self.window_spin.setRange(1, 6)
        self.window_spin.setValue(self.window_length)
        self.window_spin.valueChanged.connect(self.set_window_length)
        top.addWidget(self.window_spin)
        top.addWidget(QLabel("Learning rate:"))
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 2.0)
        self.lr_spin.setDecimals(4)
        self.lr_spin.setSingleStep(0.01)
        self.lr_spin.setValue(self.learning_rate)
        self.lr_spin.valueChanged.connect(self.set_learning_rate)
        top.addWidget(self.lr_spin)
        top.addWidget(QLabel("Finite-difference epsilon:"))
        self.eps_spin = QDoubleSpinBox()
        self.eps_spin.setRange(1.0e-6, 1.0e-1)
        self.eps_spin.setDecimals(6)
        self.eps_spin.setSingleStep(0.0001)
        self.eps_spin.setValue(self.epsilon)
        self.eps_spin.valueChanged.connect(self.set_epsilon)
        top.addWidget(self.eps_spin)
        self.reset_btn = QPushButton("Reset all")
        self.reset_btn.clicked.connect(self.reset_all)
        top.addWidget(self.reset_btn)
        top.addStretch()
        return top

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        panel.setMinimumWidth(340)
        panel.setMaximumWidth(390)
        layout = QVBoxLayout(panel)
        seq_group = QGroupBox("Toy ecological sequence")
        seq_layout = QVBoxLayout(seq_group)
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(list(PRESETS.keys()) + ["Custom"])
        self.preset_combo.currentTextChanged.connect(self.apply_preset)
        seq_layout.addWidget(QLabel("+1 = increasing water, 0 = stable, -1 = decreasing water"))
        seq_layout.addWidget(self.preset_combo)
        self.sequence_edit = QLineEdit(self.sequence_to_text())
        self.sequence_edit.editingFinished.connect(self.apply_sequence_text)
        seq_layout.addWidget(self.sequence_edit)
        apply_seq_btn = QPushButton("Apply sequence")
        apply_seq_btn.clicked.connect(self.apply_sequence_text)
        seq_layout.addWidget(apply_seq_btn)
        layout.addWidget(seq_group)
        eq_group = QGroupBox("Simplified recurrent memory cell")
        eq_layout = QVBoxLayout(eq_group)
        self.eq_label = QLabel()
        self.eq_label.setWordWrap(True)
        self.eq_label.setText(
            "<html>"
            "<p><b>Forward equations</b></p>"
            "<p>c<sub>t</sub> = <span style='color:#b91c1c;'>f</span> c<sub>t-1</sub> + "
            "<span style='color:#b91c1c;'>i</span> x<sub>t</sub></p>"
            "<p>h<sub>t</sub> = <span style='color:#b91c1c;'>o</span> tanh(c<sub>t</sub>)</p>"
            "<p>y&#770;<sub>t+1</sub> = <span style='color:#1d4ed8;'>b</span> + "
            "<span style='color:#1d4ed8;'>w</span> h<sub>t</sub></p>"
            "<p><span style='color:#b91c1c;'>red</span> = memory parameters; "
            "<span style='color:#1d4ed8;'>blue</span> = output parameters.</p>"
            "</html>"
        )
        eq_layout.addWidget(self.eq_label)
        layout.addWidget(eq_group)
        param_group = QGroupBox("Parameter values")
        param_layout = QGridLayout(param_group)
        self.param_spins: Dict[str, QDoubleSpinBox] = {}
        roles = {"f": "keep old memory", "i": "write new input", "o": "expose memory", "w": "output weight", "b": "output bias"}
        for row, name in enumerate(["f", "i", "o", "w", "b"]):
            param_layout.addWidget(QLabel(f"{name} ({roles[name]})"), row, 0)
            spin = QDoubleSpinBox()
            spin.setRange(-5.0, 5.0)
            spin.setDecimals(4)
            spin.setSingleStep(0.05)
            spin.setValue(self.params.get(name))
            spin.valueChanged.connect(self.update_params_from_spins)
            self.param_spins[name] = spin
            param_layout.addWidget(spin, row, 1)
        layout.addWidget(param_group)
        win_group = QGroupBox("Selected window")
        win_layout = QVBoxLayout(win_group)
        self.window_combo = QComboBox()
        self.window_combo.currentIndexChanged.connect(self.select_window)
        win_layout.addWidget(self.window_combo)
        self.window_summary = QLabel()
        self.window_summary.setWordWrap(True)
        win_layout.addWidget(self.window_summary)
        layout.addWidget(win_group)
        layout.addStretch()
        self.refresh_left_panel()
        return panel

    def sequence_to_text(self) -> str:
        return ", ".join(str(x) for x in self.sequence)

    def parse_sequence_text(self, text: str) -> List[int]:
        text = text.replace(",", " ")
        vals = []
        for part in text.split():
            try:
                value = int(part)
            except Exception:
                raise ValueError(f"Could not parse '{part}' as -1, 0, or 1.")
            if value not in (-1, 0, 1):
                raise ValueError("Sequence values must be only -1, 0, or 1.")
            vals.append(value)
        if len(vals) < 3:
            raise ValueError("Please provide at least 3 sequence values.")
        return vals

    def apply_preset(self, name: str) -> None:
        if name in PRESETS:
            self.sequence = PRESETS[name].copy()
            self.sequence_edit.setText(self.sequence_to_text())
            self.selected_window_index = 0
            self.reset_training_histories(keep_params=True)
            self.refresh_everything()

    def apply_sequence_text(self) -> None:
        try:
            self.sequence = self.parse_sequence_text(self.sequence_edit.text())
            self.preset_combo.blockSignals(True)
            self.preset_combo.setCurrentText("Custom")
            self.preset_combo.blockSignals(False)
            self.selected_window_index = 0
            self.reset_training_histories(keep_params=True)
            self.refresh_everything()
        except Exception as exc:
            QMessageBox.warning(self, "Sequence error", str(exc))
            self.sequence_edit.setText(self.sequence_to_text())

    def set_window_length(self, value: int) -> None:
        self.window_length = int(value)
        self.selected_window_index = 0
        self.reset_training_histories(keep_params=True)
        self.refresh_everything()

    def set_learning_rate(self, value: float) -> None:
        self.learning_rate = float(value)
        self.refresh_everything()

    def set_epsilon(self, value: float) -> None:
        self.epsilon = float(value)
        self.refresh_everything()

    def update_params_from_spins(self) -> None:
        for name, spin in self.param_spins.items():
            self.params.set(name, spin.value())
        self.snapshot_params("Current manual")
        self.refresh_all_tabs()

    def update_spins_from_params(self) -> None:
        for name, spin in self.param_spins.items():
            spin.blockSignals(True)
            spin.setValue(self.params.get(name))
            spin.blockSignals(False)

    def refresh_left_panel(self) -> None:
        windows = TinyMemoryCell.build_windows(self.sequence, self.window_length)
        self.window_combo.blockSignals(True)
        self.window_combo.clear()
        for idx, (_, inputs, target) in enumerate(windows):
            inp = ", ".join(f"{x:+d}" if x > 0 else str(x) for x in inputs)
            self.window_combo.addItem(f"Window {idx+1}: [{inp}] → {target:+d}")
        if windows:
            self.selected_window_index = max(0, min(self.selected_window_index, len(windows) - 1))
            self.window_combo.setCurrentIndex(self.selected_window_index)
            _, inputs, target = windows[self.selected_window_index]
            self.window_summary.setText(f"Input window: {inputs}\nTarget: y = x_next = {target}\nThe model produces a continuous yhat and then rounds it to xhat.")
        else:
            self.window_summary.setText("No windows available. Use a shorter window length or longer sequence.")
        self.window_combo.blockSignals(False)
        self.update_spins_from_params()

    def select_window(self, index: int) -> None:
        if index >= 0:
            self.selected_window_index = index
            self.refresh_left_panel()
            self.refresh_all_tabs()

    def current_window(self):
        windows = TinyMemoryCell.build_windows(self.sequence, self.window_length)
        if not windows:
            return None
        self.selected_window_index = max(0, min(self.selected_window_index, len(windows) - 1))
        return windows[self.selected_window_index]

    def reset_training_histories(self, keep_params: bool = True) -> None:
        self.output_loss_history = []
        self.memory_loss_history = []
        self.param_history = [self.params.as_dict()]
        self.snapshots = {"Initial": self.params.copy()} if keep_params else {"Initial": Params()}

    def reset_all_parameters(self) -> None:
        self.params = Params()
        self.reset_training_histories(keep_params=True)
        self.snapshot_params("Reset")
        self.refresh_everything()

    def reset_all(self) -> None:
        self.sequence = PRESETS["Increasing then decreasing"].copy()
        self.window_length = 4
        self.learning_rate = 0.05
        self.epsilon = 1.0e-4
        self.params = Params()
        self.selected_window_index = 0
        self.snapshots = {"Initial": self.params.copy()}
        self.output_loss_history = []
        self.memory_loss_history = []
        self.param_history = [self.params.as_dict()]
        self.sequence_edit.setText(self.sequence_to_text())
        self.window_spin.blockSignals(True); self.window_spin.setValue(self.window_length); self.window_spin.blockSignals(False)
        self.lr_spin.blockSignals(True); self.lr_spin.setValue(self.learning_rate); self.lr_spin.blockSignals(False)
        self.eps_spin.blockSignals(True); self.eps_spin.setValue(self.epsilon); self.eps_spin.blockSignals(False)
        self.preset_combo.blockSignals(True); self.preset_combo.setCurrentText("Increasing then decreasing"); self.preset_combo.blockSignals(False)
        self.refresh_everything()

    def snapshot_params(self, label: str) -> None:
        self.snapshots[label] = self.params.copy()

    def train_output_steps(self, n_steps: int) -> None:
        for _ in range(n_steps):
            gw, gb, loss = TinyMemoryCell.output_gradients(self.sequence, self.window_length, self.params)
            self.params.w -= self.learning_rate * gw
            self.params.b -= self.learning_rate * gb
            self.output_loss_history.append(TinyMemoryCell.mean_loss(self.sequence, self.window_length, self.params))
            self.param_history.append(self.params.as_dict())
        self.snapshot_params("Output-trained")
        self.statusBar().showMessage(f"Trained output layer for {n_steps} step(s).")
        self.refresh_everything()

    def train_memory_steps(self, n_steps: int, constrain_gates: bool = True) -> None:
        for _ in range(n_steps):
            grads, loss = TinyMemoryCell.finite_difference_gradients(self.sequence, self.window_length, self.params, epsilon=self.epsilon)
            for name in ["f", "i", "o", "w", "b"]:
                self.params.set(name, self.params.get(name) - self.learning_rate * grads[name])
            if constrain_gates:
                self.params.f = min(1.0, max(0.0, self.params.f))
                self.params.i = min(1.0, max(0.0, self.params.i))
                self.params.o = min(1.0, max(0.0, self.params.o))
            self.memory_loss_history.append(TinyMemoryCell.mean_loss(self.sequence, self.window_length, self.params))
            self.param_history.append(self.params.as_dict())
        self.snapshot_params("Memory-trained")
        self.statusBar().showMessage(f"Trained memory cell for {n_steps} step(s).")
        self.refresh_everything()

    def refresh_all_tabs(self, skip: QWidget | None = None) -> None:
        # Widgets can emit signals while the main window is still being built.
        # In that phase, some tab attributes do not exist yet.  Use getattr and
        # avoid cross-tab refreshes until construction is complete.
        if not getattr(self, "_ui_ready", False):
            return
        for attr in ["seq_tab", "forward_tab", "output_tab", "memory_tab", "compare_tab"]:
            tab = getattr(self, attr, None)
            if tab is not None and tab is not skip and hasattr(tab, "refresh"):
                tab.refresh()

    def refresh_everything(self) -> None:
        self.refresh_left_panel()
        self.refresh_all_tabs()


def apply_app_stylesheet(app: QApplication) -> None:
    app.setStyleSheet("""
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
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 4px 0 4px;
            color: #6c1d45;
            background-color: #fafafa;
        }
        QLabel { color: #222222; background-color: transparent; }
        QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {
            background-color: #ffffff;
            color: #111111;
            border: 1px solid #9a9a9a;
            border-radius: 3px;
            padding: 3px 6px;
            min-height: 24px;
            selection-background-color: #6c1d45;
            selection-color: #ffffff;
        }
        QComboBox QAbstractItemView {
            background-color: #ffffff;
            color: #111111;
            selection-background-color: #6c1d45;
            selection-color: #ffffff;
            border: 1px solid #9a9a9a;
        }
        QTableWidget, QTextEdit {
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
    win = CobberEcoLSTMbasicApp()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
