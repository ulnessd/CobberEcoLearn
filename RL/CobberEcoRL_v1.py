# CobberEcoRL_v1.py
# A PyQt6 application for exploring Q-learning in a simplified
# ecological management landscape.
#
# Adapted from CobberTarPit for the Ecology machine learning book.
#
# Core idea:
#   A conservation manager / field crew moves through a hidden management
#   landscape. Each movement costs effort. Reaching a restored native patch
#   gives a large positive reward. Entering a severe invasion source patch
#   gives a large negative reward. Through repeated episodes, the agent learns
#   a Q-table that stores the long-term value of actions from each state.
#
# Dependencies:
#   pip install PyQt6 numpy
#
# Run:
#   python CobberEcoRL_v1.py

import sys
import numpy as np
import random
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QComboBox, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
    QMessageBox, QFrame
)
from PyQt6.QtGui import QFont, QColor, QPixmap
from PyQt6.QtCore import Qt, QObject, QThread, pyqtSignal, QTimer


class AutoTrainerWorker(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(int)
    q_table_updated = pyqtSignal(np.ndarray)
    animation_step = pyqtSignal(int)
    episode_finished = pyqtSignal(bool)
    update_reward = pyqtSignal(float)

    def __init__(
        self,
        q_table,
        rewards,
        gamma,
        start_pos,
        goal_pos,
        severe_invasion_pos,
        max_steps=100,
        epsilon=0.1,
    ):
        super().__init__()
        self.q_table = q_table.copy()
        self.rewards = rewards
        self.gamma = gamma
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.severe_invasion_pos = severe_invasion_pos
        self.max_steps = max_steps
        self.epsilon = epsilon
        self.num_actions = 4
        self._is_running = True

    def run(self):
        episode_count = 0

        while self._is_running:
            episode_count += 1
            state = self.start_pos

            episode_reward = 0.0
            self.update_reward.emit(episode_reward)

            self.animation_step.emit(state)
            QThread.msleep(100)

            for _ in range(self.max_steps):
                if not self._is_running:
                    break

                # epsilon-greedy action selection:
                # sometimes explore randomly, otherwise choose the best known action.
                if random.uniform(0, 1) < self.epsilon:
                    action = random.randint(0, self.num_actions - 1)
                else:
                    action = int(np.argmax(self.q_table[state, :]))

                s_prime = next_state_from_action(state, action)

                reward = self.rewards[s_prime]
                episode_reward += reward
                self.update_reward.emit(episode_reward)

                self.animation_step.emit(s_prime)
                QThread.msleep(100)

                next_max = np.max(self.q_table[s_prime, :])

                # Simplified Q-learning update used for this introductory activity.
                new_value = reward + self.gamma * next_max
                self.q_table[state, action] = new_value

                state = s_prime

                if state == self.goal_pos or state == self.severe_invasion_pos:
                    break

            if not self._is_running:
                break

            self.progress.emit(episode_count)
            self.q_table_updated.emit(self.q_table.copy())
            self.episode_finished.emit(state == self.goal_pos)
            QThread.msleep(500)

        self.finished.emit()

    def stop(self):
        self._is_running = False


def next_state_from_action(state: int, action: int) -> int:
    """
    Grid is 4 x 4, states numbered:

        S0  S1  S2  S3
        S4  S5  S6  S7
        S8  S9  S10 S11
        S12 S13 S14 S15

    Actions:
        0 = UP
        1 = DOWN
        2 = LEFT
        3 = RIGHT
    """
    if action == 0:
        return state - 4 if state > 3 else state
    if action == 1:
        return state + 4 if state < 12 else state
    if action == 2:
        return state - 1 if state % 4 != 0 else state
    return state + 1 if state % 4 != 3 else state


class CobberEcoRLApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.cobber_maroon = QColor(108, 29, 69)
        self.cobber_gold = QColor(234, 170, 0)
        self.lato_font = QFont("Lato")

        self.setWindowTitle("CobberEcoRL")
        self.setGeometry(100, 100, 690, 745)
        self.setFont(self.lato_font)

        self.num_states = 16
        self.num_actions = 4
        self.gamma = 0.9

        self.q_table = np.zeros((self.num_states, self.num_actions))
        self.cumulative_reward = 0.0

        self.grid_is_hidden = True

        # In this simplified ecology world:
        # start_pos = field station / starting location
        # goal_pos = restored native patch / successful management outcome
        # severe_invasion_pos = severe invasion source patch / failure outcome
        self.start_pos = 12
        self.goal_pos = 3
        self.severe_invasion_pos = 7

        self.agent_pos = self.start_pos

        self.rewards = np.full(self.num_states, -1.0)
        self.rewards[self.goal_pos] = 100.0
        self.rewards[self.severe_invasion_pos] = -100.0

        self.last_state = -1
        self.last_action = -1
        self.s_prime = -1
        self.reward = 0.0

        self.grid_labels = []
        self.q_value_labels = {}

        self.worker = None
        self.thread = None

        self._setup_ui()
        self.full_reset()

    def _setup_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        main_layout = QHBoxLayout(main_widget)

        # ------------------------------------------------------------------
        # Left side: management landscape grid
        # ------------------------------------------------------------------
        grid_groupbox = QGroupBox("Management Landscape")
        grid_layout = QGridLayout()
        grid_layout.setSpacing(5)

        for i in range(16):
            label = QLabel()
            label.setFixedSize(80, 80)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setFont(QFont("Arial", 16, QFont.Weight.Bold))
            self.grid_labels.append(label)
            grid_layout.addWidget(label, i // 4, i % 4)

        grid_groupbox.setLayout(grid_layout)
        main_layout.addWidget(grid_groupbox)

        # ------------------------------------------------------------------
        # Right side: Q-table and controls
        # ------------------------------------------------------------------
        right_panel_layout = QVBoxLayout()

        q_table_groupbox = QGroupBox("The Manager's Brain (Q-Table)")
        q_table_layout = QGridLayout()
        q_table_layout.setSpacing(1)

        actions = ["UP", "DOWN", "LEFT", "RIGHT"]

        q_table_layout.addWidget(QLabel(""), 0, 0)

        for c, action_name in enumerate(actions):
            header = QLabel(action_name)
            header.setFont(QFont("Arial", 10, QFont.Weight.Bold))
            header.setAlignment(Qt.AlignmentFlag.AlignCenter)
            q_table_layout.addWidget(header, 0, c + 1)

        for r in range(16):
            state_header = QLabel(f"S{r}")
            state_header.setFont(QFont("Arial", 10, QFont.Weight.Bold))
            state_header.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            q_table_layout.addWidget(state_header, r + 1, 0)

            self.q_value_labels[r] = []
            for c in range(4):
                q_label = QLabel("0.00")
                q_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                q_label.setStyleSheet("border:1px solid #ccc;background-color:white;")
                q_label.setFixedSize(60, 19)
                self.q_value_labels[r].append(q_label)
                q_table_layout.addWidget(q_label, r + 1, c + 1)

        q_table_groupbox.setLayout(q_table_layout)
        right_panel_layout.addWidget(q_table_groupbox)

        controls_groupbox = QGroupBox("Controls")
        controls_layout = QVBoxLayout(controls_groupbox)

        self.status_label = QLabel("Select an action to begin.")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #333; font-style: italic; padding-bottom: 5px;")
        controls_layout.addWidget(self.status_label)

        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        controls_layout.addWidget(separator)

        reward_layout = QHBoxLayout()
        reward_layout.addWidget(QLabel("<b>Current Episode Reward:</b>"))
        self.reward_label = QLabel("0.0")
        self.reward_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        reward_layout.addWidget(self.reward_label)
        reward_layout.addStretch()
        controls_layout.addLayout(reward_layout)

        action_layout = QHBoxLayout()
        action_layout.addWidget(QLabel("Choose Action:"))
        self.action_combo = QComboBox()
        self.action_combo.addItems(actions)
        action_layout.addWidget(self.action_combo)
        controls_layout.addLayout(action_layout)

        self.execute_button = QPushButton("Execute Action")
        self.update_q_button = QPushButton("Update Q-value")
        self.reset_button = QPushButton("Reset Episode")
        self.full_reset_button = QPushButton("Reset All & Clear Brain")
        self.reveal_button = QPushButton("Reveal Management Landscape")
        self.run_animation_button = QPushButton("Run Animation")
        self.stop_animation_button = QPushButton("Stop Animation")
        self.stop_animation_button.setEnabled(False)

        animation_control_layout = QHBoxLayout()
        animation_control_layout.addWidget(self.run_animation_button)
        animation_control_layout.addWidget(self.stop_animation_button)

        self.execute_button.clicked.connect(self.execute_action)
        self.update_q_button.clicked.connect(self.update_q_value)
        self.reveal_button.clicked.connect(self.reveal_grid)
        self.reset_button.clicked.connect(self.reset_episode)
        self.full_reset_button.clicked.connect(self.full_reset)
        self.run_animation_button.clicked.connect(self.start_animation)
        self.stop_animation_button.clicked.connect(self.stop_animation)

        controls_layout.addWidget(self.execute_button)
        controls_layout.addWidget(self.update_q_button)

        controls_layout.addWidget(self.reset_button)
        controls_layout.addWidget(self.reveal_button)

        separator1 = QFrame()
        separator1.setFrameShape(QFrame.Shape.HLine)
        separator1.setFrameShadow(QFrame.Shadow.Sunken)
        controls_layout.addWidget(separator1)

        controls_layout.addLayout(animation_control_layout)

        separator2 = QFrame()
        separator2.setFrameShape(QFrame.Shape.HLine)
        separator2.setFrameShadow(QFrame.Shadow.Sunken)
        controls_layout.addWidget(separator2)

        controls_layout.addWidget(self.full_reset_button)

        teaching_note = QLabel(
            "Scenario: A field crew is learning how to move through a small "
            "management landscape. Each move costs effort. A restored native "
            "patch gives a large reward. A severe invasion source patch gives "
            "a large penalty."
        )
        teaching_note.setWordWrap(True)
        teaching_note.setStyleSheet("color:#333; padding-top: 6px;")
        controls_layout.addWidget(teaching_note)

        right_panel_layout.addWidget(controls_groupbox)
        main_layout.addLayout(right_panel_layout)

    def full_reset(self):
        self.stop_animation()
        self.q_table.fill(0)
        self.update_q_table_ui()
        self.reset_episode()
        self.status_label.setText("New session started. Manager's brain cleared.")

    def reset_episode(self):
        self.grid_is_hidden = True
        self.agent_pos = self.start_pos
        self.cumulative_reward = 0.0

        self.update_grid_ui()
        self.reward_label.setText(f"{self.cumulative_reward:.2f}")

        self.set_controls_enabled(True)
        self.update_q_button.setEnabled(False)
        self.reveal_button.setEnabled(False)

        self.status_label.setText(
            "New episode started. The landscape is hidden. Explore by choosing an action."
        )

    def execute_action(self):
        state = self.agent_pos

        action_map = {"UP": 0, "DOWN": 1, "LEFT": 2, "RIGHT": 3}
        action = action_map[self.action_combo.currentText()]

        s_prime = next_state_from_action(state, action)

        self.agent_pos = s_prime

        reward = self.rewards[s_prime]
        self.cumulative_reward += reward
        self.reward_label.setText(f"{self.cumulative_reward:.2f}")

        self.last_state = state
        self.last_action = action
        self.s_prime = s_prime
        self.reward = reward

        self.update_grid_ui()

        max_q_s_prime = np.max(self.q_table[s_prime])
        status_text = (
            f"Action in S{state} -> Reward={reward:.1f}, New State=S{s_prime}. "
            f"Formula: Q = {reward:.1f} + {self.gamma} * max(Q(S{s_prime})) "
            f"= {reward:.1f} + {self.gamma} * {max_q_s_prime:.2f}"
        )
        self.status_label.setText(status_text)

        self.action_combo.setEnabled(False)
        self.execute_button.setEnabled(False)
        self.update_q_button.setEnabled(True)

    def update_q_value(self):
        state = self.last_state
        action = self.last_action
        s_prime = self.s_prime
        reward = self.reward

        new_q_value = reward + self.gamma * np.max(self.q_table[s_prime, :])
        self.q_table[state, action] = new_q_value

        self.update_q_table_ui()

        self.status_label.setText(
            f"Q(S{state}, Action: {self.action_combo.currentText()}) "
            f"updated to {new_q_value:.2f}. Choose next action."
        )

        is_terminal = s_prime == self.goal_pos or s_prime == self.severe_invasion_pos

        if is_terminal:
            self.action_combo.setEnabled(False)
            self.execute_button.setEnabled(False)
            self.update_q_button.setEnabled(False)
            self.reveal_button.setEnabled(True)

            if s_prime == self.goal_pos:
                self.show_popup(
                    "Success!",
                    "The field crew reached the restored native patch.",
                    is_success=True
                )
            else:
                self.show_popup(
                    "Failure!",
                    "The field crew entered a severe invasion source patch.",
                    is_success=False
                )
        else:
            self.action_combo.setEnabled(True)
            self.execute_button.setEnabled(True)
            self.update_q_button.setEnabled(False)

    def start_animation(self):
        self.set_controls_enabled(False)
        self.status_label.setText("Starting continuous Q-learning animation...")

        self.thread = QThread()
        self.worker = AutoTrainerWorker(
            q_table=self.q_table,
            rewards=self.rewards,
            gamma=self.gamma,
            start_pos=self.start_pos,
            goal_pos=self.goal_pos,
            severe_invasion_pos=self.severe_invasion_pos,
        )

        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.worker.finished.connect(self._on_animation_finished)
        self.worker.progress.connect(self._on_auto_train_progress)
        self.worker.q_table_updated.connect(self._on_q_table_updated)
        self.worker.animation_step.connect(self._on_animation_step)
        self.worker.episode_finished.connect(self._on_episode_finished)
        self.worker.update_reward.connect(self._on_reward_updated)

        self.thread.start()

    def stop_animation(self):
        if self.worker:
            self.worker.stop()
        self.status_label.setText("Stopping animation...")
        self.set_controls_enabled(True)

    def _on_reward_updated(self, reward_value):
        self.reward_label.setText(f"{reward_value:.2f}")

    def _on_auto_train_progress(self, episode_num):
        self.status_label.setText(f"Running... Episode: {episode_num}")

    def _on_q_table_updated(self, new_q_table):
        self.q_table = new_q_table
        self.update_q_table_ui()

    def _on_animation_step(self, position):
        self.agent_pos = position
        self.update_grid_ui()

    def _on_episode_finished(self, is_success):
        if is_success:
            self.show_popup(
                "Success!",
                "The manager found a path to the restored native patch.",
                is_success=True,
                is_animated=True,
            )
        else:
            self.show_popup(
                "Failure!",
                "The manager's path entered the severe invasion source patch.",
                is_success=False,
                is_animated=True,
            )

    def _on_animation_finished(self):
        self.status_label.setText("Animation stopped.")
        self.set_controls_enabled(True)
        self.reward_label.setText("0.00")

    def set_controls_enabled(self, enabled):
        self.execute_button.setEnabled(enabled)
        self.update_q_button.setEnabled(enabled)
        self.reset_button.setEnabled(enabled)
        self.full_reset_button.setEnabled(enabled)
        self.reveal_button.setEnabled(enabled)
        self.run_animation_button.setEnabled(enabled)
        self.stop_animation_button.setEnabled(not enabled)
        self.action_combo.setEnabled(enabled)

    def reveal_grid(self):
        self.grid_is_hidden = False
        self.update_grid_ui()
        self.reveal_button.setEnabled(False)

    def get_popup_image_path(self, is_success: bool):
        base_dir = Path(__file__).resolve().parent

        if is_success:
            candidates = [base_dir / "win1.png", base_dir / "win2.png"]
        else:
            candidates = [base_dir / "loss1.png", base_dir / "loss2.png"]

        existing = [p for p in candidates if p.exists()]
        if not existing:
            return None

        return str(random.choice(existing))

    def show_popup(self, title, message, is_success, is_animated=False):
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle(title)
        msg_box.setText(f"<h3>{message}</h3>")

        image_path = self.get_popup_image_path(is_success)
        if image_path:
            pixmap = QPixmap(image_path)
            if not pixmap.isNull():
                msg_box.setIconPixmap(
                    pixmap.scaled(
                        220, 220,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation
                    )
                )

        if is_animated:
            msg_box.setInformativeText("The next episode will begin automatically.")
        else:
            msg_box.setInformativeText(
                "Click 'Reveal Management Landscape' to see the layout, "
                "or 'Reset Episode' to start again."
            )

        if is_animated:
            QTimer.singleShot(700, msg_box.close)
            msg_box.show()
        else:
            msg_box.exec()

    def update_grid_ui(self):
        for i, label in enumerate(self.grid_labels):
            label.setText(f"S{i}")
            style = "border: 2px solid #555; border-radius: 5px;"

            if self.grid_is_hidden:
                style += "background-color: #ddd; color: #888;"
            else:
                if i == self.start_pos:
                    style += "background-color: #aaffaa;"
                    label.setText("Base")
                elif i == self.goal_pos:
                    style += "background-color: #aaddff;"
                    label.setText("Native")
                elif i == self.severe_invasion_pos:
                    style += "background-color: #444; color: white;"
                    label.setText("Inv.")
                else:
                    style += "background-color: #f0f0f0;"
                    label.setText("")

            if i == self.agent_pos:
                style += "border: 4px solid #ff5555;"

            label.setStyleSheet(style)

    def update_q_table_ui(self):
        for r in range(self.num_states):
            for c in range(self.num_actions):
                q_val = self.q_table[r, c]
                label = self.q_value_labels[r][c]
                label.setText(f"{q_val:.2f}")

                if q_val > 0.01:
                    alpha = int(min(255, q_val * 3))
                    label.setStyleSheet(
                        f"background-color: rgba(100, 100, 255, {alpha}); border: 1px solid #ccc;"
                    )
                elif q_val < -0.01:
                    alpha = int(min(255, abs(q_val * 3)))
                    label.setStyleSheet(
                        f"background-color: rgba(255, 100, 100, {alpha}); border: 1px solid #ccc;"
                    )
                else:
                    label.setStyleSheet("background-color: white; border: 1px solid #ccc;")


def apply_app_stylesheet(app: QApplication) -> None:
    app.setStyleSheet(
        """
        QWidget { color: #222222; background-color: #ffffff; }
        QMainWindow, QDialog { background-color: #ffffff; }
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
        QComboBox {
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
        """
    )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    apply_app_stylesheet(app)
    window = CobberEcoRLApp()
    window.show()
    sys.exit(app.exec())
