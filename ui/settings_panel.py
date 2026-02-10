"""
NEXUS AI - Settings Panel
"""
import sys
from pathlib import Path
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QFrame, QLabel, QComboBox,
    QCheckBox, QPushButton, QFormLayout
)

sys.path.append(str(Path(__file__).parent.parent))
from ui.theme import theme, colors, fonts, icons
from ui.widgets import HeaderLabel, Section

class SettingsPanel(QFrame):
    def __init__(self, brain=None, parent=None):
        super().__init__(parent)
        self._brain = brain
        self.setStyleSheet(f"background-color: {colors.bg_dark};")
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        layout.addWidget(HeaderLabel("System Settings", icons.SETTINGS, colors.text_muted))

        # AI Configuration
        ai_section = Section("AI Configuration", "🤖", expanded=True)
        form = QFormLayout()
        
        self._combo_model = QComboBox()
        self._combo_model.addItems(["llama3:8b", "mistral", "gemma"])
        self._combo_model.setStyleSheet(f"background: {colors.bg_elevated}; color: {colors.text_primary}; padding: 5px;")
        
        self._check_voice = QCheckBox("Enable Voice Output")
        self._check_voice.setStyleSheet(f"color: {colors.text_primary};")
        
        form.addRow("LLM Model:", self._combo_model)
        form.addRow("", self._check_voice)
        
        ai_section.add_layout(form)
        layout.addWidget(ai_section)

        # Behavior
        beh_section = Section("Behavior", "🧠", expanded=True)
        form_beh = QFormLayout()
        
        self._check_auto_evolve = QCheckBox("Autonomous Evolution")
        self._check_auto_evolve.setChecked(True)
        self._check_auto_evolve.setStyleSheet(f"color: {colors.text_primary};")
        
        self._check_monitor = QCheckBox("User Monitoring")
        self._check_monitor.setChecked(True)
        self._check_monitor.setStyleSheet(f"color: {colors.text_primary};")
        
        form_beh.addRow(self._check_auto_evolve)
        form_beh.addRow(self._check_monitor)
        
        beh_section.add_layout(form_beh)
        layout.addWidget(beh_section)

        # Save Button
        save_btn = QPushButton("Save Settings")
        save_btn.setStyleSheet(f"background-color: {colors.accent_cyan}; color: {colors.bg_dark}; font-weight: bold; padding: 10px; border-radius: 5px;")
        layout.addWidget(save_btn)
        
        layout.addStretch()

    def set_brain(self, brain):
        self._brain = brain