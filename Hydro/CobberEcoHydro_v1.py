# CobberEcoHydro_v1.py
# A PyQt6 application for training a reinforcement learning agent to manage
# water addition in a simplified wetland restoration setting.
#
# Adapted from CobberTitrator for the Ecology machine learning book.
#
# Scenario:
#   A wetland restoration system uses a controllable pump/valve to bring a
#   wetland cell to a target water depth. Too little water underfills the
#   wetland. Too much water overfills/floods the wetland. The agent learns
#   how to adjust flow rate so that it reaches the target quickly without
#   overshooting.
#
# Required images in the same directory as this script:
#   crusty_happy.png
#   crusty_angry.png
#
# Dependencies:
#   pip install PyQt6 numpy matplotlib tensorflow
#
# Run:
#   python CobberEcoHydro_v1.py

from __future__ import annotations

import os
import sys
import time
import random
from pathlib import Path
from collections import deque
from typing import Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QSpinBox, QFileDialog, QTextEdit, QMessageBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QColor, QPixmap


def app_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


APP_ROOT = app_root()


class WetlandControlEnv:
    """
    A small continuous-control environment for the wetland water-level task.

    State vector:
        [water_depth, flow_rate, warning_signal, estimated_target_depth]

    Actions:
        0 -> close valve / slow down
        1 -> hold valve setting
        2 -> open valve / speed up
    """

    def __init__(self, sensor_uncertainty: float = 0.5):
        self.max_flow_rate = 1.25
        self.min_flow_rate = 0.0
        self.dt = 0.1
        self.acceleration_values = [-0.5, 0.0, 0.5]
        self.action_size = len(self.acceleration_values)
        self.state_size = 4
        self.sensor_uncertainty = float(sensor_uncertainty)
        self.fixed_target_depth: Optional[float] = None
        self.reset()

    def reset(self):
        if self.fixed_target_depth is None:
            self.target_depth = np.random.uniform(10.0, 24.0)
        else:
            self.target_depth = float(self.fixed_target_depth)

        self.estimated_target_depth = self.target_depth + np.random.uniform(
            -self.sensor_uncertainty, self.sensor_uncertainty
        )
        self.water_depth = 0.0
        self.flow_rate = 0.0
        self.done = False
        self.total_time = 0.0
        self.warning_signal = 0.0
        return self._get_state()

    def set_target_depth(self, target_depth: float, sensor_uncertainty: Optional[float] = None):
        self.fixed_target_depth = float(target_depth)
        if sensor_uncertainty is not None:
            self.sensor_uncertainty = float(sensor_uncertainty)

    def step(self, action: int):
        acceleration = self.acceleration_values[int(action)]
        self.flow_rate = np.clip(
            self.flow_rate + acceleration * self.dt,
            self.min_flow_rate,
            self.max_flow_rate,
        )
        self.water_depth += self.flow_rate * self.dt
        self.total_time += self.dt
        self.warning_signal = self._calculate_warning_signal()

        # Progress reward minus a small time/effort cost.
        reward = self.flow_rate * self.dt - 0.1

        # Stop too early: underfilled wetland cell.
        if (
            self.flow_rate == 0.0
            and self.warning_signal == 0.0
            and self.water_depth < self.target_depth - 0.5
        ):
            self.done = True
            reward += -50.0
            return self._get_state(), reward, self.done, {}

        # Success: stop just below or at target. Reward speed and accuracy.
        if self.target_depth - 0.10 <= self.water_depth <= self.target_depth:
            self.done = True
            error = 2.0 * (self.water_depth - self.target_depth)
            reward += 100.0 / max(self.total_time, 0.1)
            reward += 225.0 * np.exp(np.clip(200.0 * error, -50, 50))

        # Failure: overshoot target, representing overfill/flood risk.
        elif self.water_depth > self.target_depth:
            self.done = True
            error = 2.0 * (self.water_depth - self.target_depth)
            reward += -1.0 * np.exp(np.clip(100.0 * error, -50, 50)) + 1.0

        return self._get_state(), reward, self.done, {}

    def _calculate_warning_signal(self) -> float:
        x = self.water_depth
        d_est = self.estimated_target_depth
        if x >= d_est - 0.40:
            signal = 1.0 - np.exp(-20.0 * (x - d_est + 0.4) ** 2)
            return float(np.clip(signal, 0.0, 1.0))
        return 0.0

    def _get_state(self):
        return np.array(
            [self.water_depth, self.flow_rate, self.warning_signal, self.estimated_target_depth],
            dtype=np.float32,
        )


class DQN(tf.keras.Model):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.dense1 = layers.Dense(64, activation="relu", input_shape=(state_size,))
        self.dense2 = layers.Dense(64, activation="relu")
        self.output_layer = layers.Dense(action_size)

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        return self.output_layer(x)


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class TrainingThread(QThread):
    update_log = pyqtSignal(str)
    update_animation = pyqtSignal(dict)
    training_finished = pyqtSignal()

    def __init__(self, num_episodes):
        super().__init__()
        self.num_episodes = num_episodes
        self.env = WetlandControlEnv(sensor_uncertainty=0.5)
        self.model = None
        self.target_model = None
        self.memory = ReplayMemory(10000)
        self.gamma = 0.99
        self.epsilon_start = 1.0
        self.epsilon_end = 0.01
        self.epsilon_decay = 1500
        self.batch_size = 64
        self.target_update = 10
        self.steps_done = 0

    def select_action(self, state, action_size):
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(
            -1.0 * self.steps_done / self.epsilon_decay
        )
        self.steps_done += 1
        if random.random() < epsilon:
            return random.randrange(action_size)
        q_values = self.model(np.array([state], dtype=np.float32))
        return int(np.argmax(q_values[0]))

    def run(self):
        try:
            state_size = self.env.state_size
            action_size = self.env.action_size
            self.model = DQN(state_size, action_size)
            self.target_model = DQN(state_size, action_size)
            dummy_input = tf.constant([[0.0] * state_size], dtype=tf.float32)
            self.model(dummy_input)
            self.target_model(dummy_input)
            self.target_model.set_weights(self.model.get_weights())
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

            for episode in range(1, self.num_episodes + 1):
                state = self.env.reset()
                total_reward = 0.0
                positions = [self.env.water_depth]

                for step_index in range(1, 1501):
                    action = self.select_action(state, action_size)
                    next_state, reward, done, _ = self.env.step(action)
                    total_reward += reward
                    self.memory.push(state, action, reward, next_state, done)
                    state = next_state
                    positions.append(self.env.water_depth)

                    if len(self.memory) >= self.batch_size:
                        transitions = self.memory.sample(self.batch_size)
                        states, actions, rewards, next_states, dones = map(np.array, zip(*transitions))
                        next_q_values = self.target_model(next_states.astype(np.float32))
                        max_next_q_values = tf.reduce_max(next_q_values, axis=1)
                        target_q_values = rewards + (self.gamma * max_next_q_values * (1 - dones))
                        with tf.GradientTape() as tape:
                            q_values = self.model(states.astype(np.float32))
                            action_masks = tf.one_hot(actions.astype(np.int32), action_size)
                            q_values = tf.reduce_sum(q_values * action_masks, axis=1)
                            loss = tf.keras.losses.MSE(target_q_values, q_values)
                        grads = tape.gradient(loss, self.model.trainable_variables)
                        optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

                    if done:
                        break

                if episode % self.target_update == 0:
                    self.target_model.set_weights(self.model.get_weights())

                if episode % 5 == 0:
                    final_depth = float(self.env.water_depth)
                    target = float(self.env.target_depth)
                    outcome = "SUCCESS" if target - 0.10 <= final_depth <= target else "MISS"
                    self.update_log.emit(
                        f"Ep {episode}: Reward={total_reward:.2f}, Steps={step_index}, "
                        f"Target={target:.2f} cm, Stop={final_depth:.2f} cm, {outcome}"
                    )
                    self.update_animation.emit({
                        "positions": positions.copy(),
                        "target_depth": target,
                        "estimated_target_depth": float(self.env.estimated_target_depth),
                        "final_depth": final_depth,
                    })
            self.update_log.emit("Training complete. Model is ready to be saved.")
        except Exception as exc:
            self.update_log.emit(f"Error during training: {exc}")
        finally:
            self.training_finished.emit()


class EvaluationThread(QThread):
    update_log = pyqtSignal(str)
    update_animation = pyqtSignal(list, float, float)
    update_results = pyqtSignal(str)
    evaluation_finished = pyqtSignal()

    def __init__(self, model, target_depth, sensor_uncertainty):
        super().__init__()
        self.model = model
        self.target_depth = float(target_depth)
        self.sensor_uncertainty = float(sensor_uncertainty)
        self.env = WetlandControlEnv(sensor_uncertainty=self.sensor_uncertainty)
        self.env.set_target_depth(self.target_depth, self.sensor_uncertainty)

    def run(self):
        try:
            self.update_log.emit(
                f"Starting Eval: Target Depth={self.target_depth:.2f} cm, "
                f"Sensor uncertainty=±{self.sensor_uncertainty:.2f} cm"
            )
            header = f"{'Run':<8}{'Stop Depth (cm)':<22}{'Error (cm)':<16}{'Outcome':<12}"
            self.update_log.emit(header)
            self.update_log.emit("-" * 58)
            final_errors = []
            final_positions = []

            for run_num in range(1, 4):
                final_pos, final_error, total_reward, steps = self.run_single_trial()
                final_errors.append(final_error)
                final_positions.append(final_pos)
                outcome = "success" if -0.10 <= final_error <= 0.0 else "miss"
                self.update_log.emit(f"{run_num:<8}{final_pos:<22.2f}{final_error:<16.2f}{outcome:<12}")
                time.sleep(2)

            if np.std(final_errors) >= 0.1:
                self.update_log.emit("Runs not consistent. Performing a fourth run.")
                final_pos, final_error, total_reward, steps = self.run_single_trial()
                final_errors.append(final_error)
                final_positions.append(final_pos)
                outcome = "success" if -0.10 <= final_error <= 0.0 else "miss"
                self.update_log.emit(f"{4:<8}{final_pos:<22.2f}{final_error:<16.2f}{outcome:<12}")

            avg_stop_depth = float(np.mean(final_positions))
            avg_error = avg_stop_depth - self.target_depth
            abs_error = abs(avg_error)
            if -0.10 <= avg_error <= 0.0:
                interpretation = "The agent filled the wetland cell close to the target without overfilling."
            elif avg_error < -0.10:
                interpretation = "The agent tended to underfill the wetland cell."
            else:
                interpretation = "The agent tended to overfill the wetland cell, creating flood-risk behavior."
            self.update_results.emit(
                f"""
                <p style="font-family: Lato; font-size: 12pt;">
                    Average Final Water Depth: <b>{avg_stop_depth:.2f} cm</b>
                </p>
                <p style="font-family: Lato; font-size: 11pt;">
                    Target Depth: <b>{self.target_depth:.2f} cm</b>
                </p>
                <p style="font-family: Lato; font-size: 11pt;">
                    Average Error: <b>{avg_error:+.2f} cm</b>
                    &nbsp;&nbsp; Absolute Error: <b>{abs_error:.2f} cm</b>
                </p>
                <p style="font-family: Lato; font-size: 11pt;">
                    Interpretation: {interpretation}
                </p>
                """
            )
        except Exception as exc:
            self.update_log.emit(f"Error during evaluation: {exc}")
        finally:
            self.evaluation_finished.emit()

    def run_single_trial(self):
        self.env.set_target_depth(self.target_depth, self.sensor_uncertainty)
        state = self.env.reset()
        positions = [self.env.water_depth]
        total_reward = 0.0
        for step_index in range(1, 1501):
            q_values = self.model(np.array([state], dtype=np.float32))
            action = int(np.argmax(q_values[0]))
            next_state, reward, done, _ = self.env.step(action)
            state = next_state
            total_reward += reward
            positions.append(self.env.water_depth)
            if done:
                break
        self.update_animation.emit(positions, float(self.env.target_depth), float(self.env.estimated_target_depth))
        return float(self.env.water_depth), float(self.env.water_depth - self.env.target_depth), total_reward, step_index


def load_pixmap(filename: str) -> QPixmap:
    path = APP_ROOT / filename
    if path.exists():
        return QPixmap(str(path))
    return QPixmap()


def scale_pixmap_for_label(pixmap: QPixmap, size: int = 250) -> QPixmap:
    if pixmap.isNull():
        return pixmap
    return pixmap.scaled(size, size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)


def apply_app_stylesheet(app: QApplication) -> None:
    app.setStyleSheet(
        """
        QWidget { color: #222222; background-color: #ffffff; }
        QMainWindow, QDialog { background-color: #ffffff; }
        QTabWidget::pane { border: 1px solid #cccccc; }
        QTabBar::tab { padding: 6px 12px; }
        QTabBar::tab:selected { background: #6c1d45; color: #ffffff; }
        QLabel { color: #222222; background-color: transparent; }
        QLineEdit, QSpinBox, QTextEdit {
            background-color: #ffffff; color: #111111;
            border: 1px solid #a0a0a0; border-radius: 3px; padding: 3px 6px;
            selection-background-color: #6c1d45; selection-color: #ffffff;
        }
        QPushButton {
            background-color: #f7f7f7; color: #111111;
            border: 1px solid #9a9a9a; border-radius: 4px; padding: 6px 10px;
        }
        QPushButton:hover { background-color: #eeeeee; }
        QPushButton:pressed { background-color: #dddddd; }
        """
    )


class TrainingTab(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.training_thread: Optional[TrainingThread] = None
        layout = QVBoxLayout(self)
        controls_layout = QHBoxLayout()
        self.episodes_input = QSpinBox()
        self.episodes_input.setRange(1, 100000)
        self.episodes_input.setValue(50)
        self.start_button = QPushButton("Start Training")
        self.start_button.clicked.connect(self.start_training)
        self.save_button = QPushButton("Save Model")
        self.save_button.clicked.connect(self.save_model)
        self.save_button.setEnabled(False)
        controls_layout.addWidget(QLabel("Number of Episodes:"))
        controls_layout.addWidget(self.episodes_input)
        controls_layout.addWidget(self.start_button)
        controls_layout.addWidget(self.save_button)

        display_layout = QHBoxLayout()
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.manager_label = QLabel()
        self.manager_label.setFixedSize(350, 350)
        self.manager_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.manager_label.setText("Crusty project manager feedback will appear here.")
        self.happy_pixmap = load_pixmap("crusty_happy.png")
        self.angry_pixmap = load_pixmap("crusty_angry.png")
        display_layout.addWidget(self.canvas)
        display_layout.addWidget(self.manager_label)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setFont(QFont("Lato", 11))
        layout.addLayout(controls_layout)
        layout.addWidget(QLabel("Training Animation (updated every 5 episodes):"))
        layout.addLayout(display_layout)
        layout.addWidget(QLabel("Training Log:"))
        layout.addWidget(self.log)

    def start_training(self):
        self.start_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.log.append(f"Starting training for {self.episodes_input.value()} episodes...")
        self.manager_label.clear()
        self.training_thread = TrainingThread(self.episodes_input.value())
        self.training_thread.update_log.connect(self.log.append)
        self.training_thread.update_animation.connect(self.update_animation)
        self.training_thread.training_finished.connect(self.on_training_finished)
        self.training_thread.start()

    def save_model(self):
        if self.training_thread and self.training_thread.model:
            filepath, _ = QFileDialog.getSaveFileName(
                self, "Save Trained Wetland Model", str(APP_ROOT), "H5 Weights (*.weights.h5)"
            )
            if filepath:
                if not filepath.endswith(".weights.h5"):
                    filepath += ".weights.h5"
                self.training_thread.model.save_weights(filepath)
                self.log.append(f"Model saved to {filepath}")
        else:
            QMessageBox.warning(self, "Save Model", "No trained model to save.")

    def update_animation(self, frame_data):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        positions = np.array(frame_data["positions"], dtype=float)
        target_depth = float(frame_data["target_depth"])
        estimated_target_depth = float(frame_data["estimated_target_depth"])
        final_depth = float(frame_data["final_depth"])
        t = np.arange(len(positions)) * 0.1
        colors = np.zeros_like(positions)
        mask = positions >= estimated_target_depth - 0.40
        colors[mask] = 1.0 - np.exp(-20.0 * (positions[mask] - estimated_target_depth + 0.4) ** 2)
        colors = np.clip(colors, 0.0, 1.0)
        sc = ax.scatter(t, positions, c=colors, cmap=plt.cm.YlGnBu, marker="s", s=100, vmin=0.0, vmax=1.0)
        ax.axhline(y=target_depth, color="black", linestyle="--", label=f"Target ({target_depth:.2f} cm)")
        ax.axhline(y=estimated_target_depth, color="gray", linestyle=":", label=f"Sensor estimate ({estimated_target_depth:.2f} cm)")
        ax.set_xlim(0, max(5, t.max() * 1.1))
        ax.set_ylim(0, max(target_depth * 1.1, positions.max() * 1.05, 1))
        ax.set_xlabel("Time (arbitrary units)")
        ax.set_ylabel("Water Depth Added (cm)")
        ax.legend(loc="best")
        cbar = self.figure.colorbar(sc, ax=ax)
        cbar.set_label("Near-target sensor signal")
        cbar.set_ticks([])
        self.canvas.draw()
        pixmap = self.happy_pixmap if target_depth - 0.10 <= final_depth <= target_depth else self.angry_pixmap
        fallback = "Good fill!" if target_depth - 0.10 <= final_depth <= target_depth else "Needs work!"
        self.manager_label.setText(fallback) if pixmap.isNull() else self.manager_label.setPixmap(scale_pixmap_for_label(pixmap, 350))

    def on_training_finished(self):
        self.start_button.setEnabled(True)
        self.save_button.setEnabled(True)


class EvaluationTab(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.model = None
        self.target_depth = None
        self.sensor_uncertainty = 0.5
        self.evaluation_thread: Optional[EvaluationThread] = None
        layout = QVBoxLayout(self)
        inputs_layout = QHBoxLayout()
        self.target_depth_input = QLineEdit("20.0")
        self.sensor_uncertainty_input = QLineEdit("0.50")
        inputs_layout.addWidget(QLabel("Target Wetland Depth (cm):"))
        inputs_layout.addWidget(self.target_depth_input)
        inputs_layout.addWidget(QLabel("Sensor Uncertainty (± cm):"))
        inputs_layout.addWidget(self.sensor_uncertainty_input)
        buttons_layout = QHBoxLayout()
        self.load_model_button = QPushButton("Load Model")
        self.set_target_button = QPushButton("Set Target Depth")
        self.perform_fill_button = QPushButton("Perform Water-Level Run")
        self.load_model_button.clicked.connect(self.load_model)
        self.set_target_button.clicked.connect(self.set_target_depth)
        self.perform_fill_button.clicked.connect(self.perform_fill)
        self.perform_fill_button.setEnabled(False)
        buttons_layout.addWidget(self.load_model_button)
        buttons_layout.addWidget(self.set_target_button)
        buttons_layout.addWidget(self.perform_fill_button)

        display_layout = QHBoxLayout()
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.manager_label = QLabel()
        self.manager_label.setFixedSize(350, 350)
        self.manager_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.manager_label.setText("Crusty project manager feedback will appear here.")
        self.happy_pixmap = load_pixmap("crusty_happy.png")
        self.angry_pixmap = load_pixmap("crusty_angry.png")
        display_layout.addWidget(self.canvas)
        display_layout.addWidget(self.manager_label)
        self.results_display = QTextEdit()
        self.results_display.setReadOnly(True)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setFont(QFont("Lato", 11))
        layout.addLayout(inputs_layout)
        layout.addLayout(buttons_layout)
        layout.addWidget(QLabel("Water-Level Control Animation:"))
        layout.addLayout(display_layout)
        layout.addWidget(QLabel("Results:"))
        layout.addWidget(self.results_display)
        layout.addWidget(QLabel("Evaluation Log:"))
        layout.addWidget(self.log)

    def _format_inputs(self):
        try:
            target_depth = float(self.target_depth_input.text())
            self.target_depth_input.setText(f"{target_depth:.2f}")
            sensor_uncertainty = float(self.sensor_uncertainty_input.text())
            self.sensor_uncertainty_input.setText(f"{sensor_uncertainty:.2f}")
        except ValueError:
            pass

    def load_model(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Load Trained Wetland Model", str(APP_ROOT), "H5 Weights (*.weights.h5)")
        if filepath:
            try:
                self.model = DQN(4, 3)
                self.model(tf.constant([[0.0] * 4], dtype=tf.float32))
                self.model.load_weights(filepath)
                self.log.append(f"Model loaded from {os.path.basename(filepath)}")
                self._update_perform_button_state()
            except Exception as exc:
                QMessageBox.critical(self, "Load Model Error", f"Error: {exc}")

    def set_target_depth(self):
        self._format_inputs()
        try:
            target_depth = float(self.target_depth_input.text())
            sensor_uncertainty = float(self.sensor_uncertainty_input.text())
            if not (target_depth > 0 and sensor_uncertainty >= 0):
                raise ValueError("Target depth must be positive and sensor uncertainty must be nonnegative.")
            self.target_depth = target_depth
            self.sensor_uncertainty = sensor_uncertainty
            self.log.append(f"Target set: {self.target_depth:.2f} cm with sensor uncertainty ±{self.sensor_uncertainty:.2f} cm")
            self._update_perform_button_state()
        except ValueError as exc:
            QMessageBox.warning(self, "Input Error", f"Invalid input: {exc}")

    def _update_perform_button_state(self):
        self.perform_fill_button.setEnabled(self.model is not None and self.target_depth is not None)

    def perform_fill(self):
        if self.model is None or self.target_depth is None:
            QMessageBox.warning(self, "Setup Error", "Load a model and set a target depth first.")
            return
        self.manager_label.clear()
        self.perform_fill_button.setEnabled(False)
        self.results_display.clear()
        self.evaluation_thread = EvaluationThread(self.model, self.target_depth, self.sensor_uncertainty)
        self.evaluation_thread.update_log.connect(self.log.append)
        self.evaluation_thread.update_animation.connect(self.update_animation)
        self.evaluation_thread.update_results.connect(self.results_display.append)
        self.evaluation_thread.evaluation_finished.connect(lambda: self.perform_fill_button.setEnabled(True))
        self.evaluation_thread.start()

    def update_animation(self, positions, target_depth, estimated_target_depth):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        positions = np.array(positions, dtype=float)
        t = np.arange(len(positions)) * 0.1
        colors = np.zeros_like(positions)
        mask = positions >= estimated_target_depth - 0.40
        colors[mask] = 1.0 - np.exp(-20.0 * (positions[mask] - estimated_target_depth + 0.4) ** 2)
        colors = np.clip(colors, 0.0, 1.0)
        sc = ax.scatter(t, positions, c=colors, cmap=plt.cm.YlGnBu, marker="s", s=100, vmin=0.0, vmax=1.0)
        ax.axhline(y=target_depth, color="black", linestyle="--", label=f"Target ({target_depth:.2f} cm)")
        ax.axhline(y=estimated_target_depth, color="gray", linestyle=":", label=f"Sensor estimate ({estimated_target_depth:.2f} cm)")
        ax.set_xlim(0, max(5, t.max() * 1.1))
        ax.set_ylim(0, max(target_depth * 1.1, positions.max() * 1.05, 1))
        ax.set_xlabel("Time (arbitrary units)")
        ax.set_ylabel("Water Depth Added (cm)")
        cbar = self.figure.colorbar(sc, ax=ax)
        cbar.set_label("Near-target sensor signal")
        cbar.set_ticks([])
        ax.legend(loc="best")
        self.canvas.draw()
        final_depth = float(positions[-1])
        pixmap = self.happy_pixmap if target_depth - 0.10 <= final_depth <= target_depth else self.angry_pixmap
        fallback = "Good fill!" if target_depth - 0.10 <= final_depth <= target_depth else "Needs work!"
        self.manager_label.setText(fallback) if pixmap.isNull() else self.manager_label.setPixmap(scale_pixmap_for_label(pixmap, 350))


class CobberEcoHydroApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.cobber_maroon = QColor(108, 29, 69)
        self.cobber_gold = QColor(234, 170, 0)
        self.lato_font = QFont("Lato")
        self.setWindowTitle("CobberEcoHydro")
        self.setGeometry(100, 100, 1300, 700)
        self.setFont(self.lato_font)
        self.layout = QVBoxLayout()
        self.tabs = QTabWidget()
        self.training_tab = TrainingTab(self)
        self.evaluation_tab = EvaluationTab(self)
        self.tabs.addTab(self.training_tab, "Model Training")
        self.tabs.addTab(self.evaluation_tab, "Model Testing")
        central_widget = QWidget()
        central_widget.setLayout(self.layout)
        self.layout.addWidget(self.tabs)
        self.setCentralWidget(central_widget)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    apply_app_stylesheet(app)
    window = CobberEcoHydroApp()
    window.show()
    sys.exit(app.exec())
