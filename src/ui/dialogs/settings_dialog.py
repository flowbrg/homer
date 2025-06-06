# ui/dialogs/settings_dialog.py

from PySide6.QtWidgets import (
    QDialog, QFormLayout, QVBoxLayout, QPushButton, QLineEdit, QCheckBox,
    QLabel, QHBoxLayout, QWidget, QTextEdit, QPlainTextEdit
)
from pathlib import Path
from dataclasses import fields, is_dataclass
from typing import get_type_hints

from src.core.configuration import Configuration
from src.resources.utils import load_config, save_config

class SettingsDialog(QDialog):
    def __init__(self, config: Configuration, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Preferences")
        self.resize(600, 600)
        self.config = config
        self.fields = {}

        layout = QVBoxLayout(self)
        self.form_layout = QFormLayout()
        layout.addLayout(self.form_layout)

        self._build_form(config)

        # Buttons
        button_layout = QHBoxLayout()

        save_button = QPushButton("Save")
        save_button.clicked.connect(self._save_and_close)

        restore_button = QPushButton("Restore Defaults")
        restore_button.clicked.connect(self._restore_defaults)

        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)

        button_layout.addWidget(save_button)
        button_layout.addWidget(restore_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

    def _build_form(self, config_obj: Configuration, prefix: str = ""):
        """Builds the form based on a dataclass instance."""
        for field_obj in fields(config_obj):
            field_name = field_obj.name
            full_key = f"{prefix}.{field_name}" if prefix else field_name
            field_type = field_obj.type
            value = getattr(config_obj, field_name)
            description = field_obj.metadata.get("description", "")

            label = QLabel(f"{field_name}")
            if description:
                label.setToolTip(description)

            if isinstance(value, bool):
                widget = QCheckBox()
                widget.setChecked(value)
            elif isinstance(value, str) and "prompt" in field_name.lower():
                widget = QPlainTextEdit()
                widget.setPlainText(value)
                widget.setMinimumHeight(80)
                widget.setSizeAdjustPolicy(QPlainTextEdit.SizeAdjustPolicy.AdjustToContents)
            else:
                widget = QLineEdit(str(value))

            self.fields[full_key] = (widget, field_type)
            self.form_layout.addRow(label, widget)

    def _rebuild_config(self) -> Configuration:
        """Reconstructs Configuration from form widgets."""
        kwargs = {}
        for full_key, (widget, field_type) in self.fields.items():
            print(f"[INFO] Processing {widget}: {full_key} with type {field_type}")
            field_name = full_key.split(".")[-1]

            if isinstance(widget, QCheckBox):
                value = widget.isChecked()
            elif isinstance(widget, QPlainTextEdit):
                value = widget.toPlainText()
            elif isinstance(widget, QLineEdit):
                value = widget.text()
            else:
                raise ValueError(f"Unsupported widget type for field '{field_name}'")

            kwargs[field_name] = value

        return Configuration(**kwargs)

    def _restore_defaults(self):
        """Restore defaults from file."""
        defaults = load_config()

        self.fields.clear()
        while self.form_layout.count():
            item = self.form_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self._build_form(defaults)
        self.config = defaults

        updated_config = self._rebuild_config()
        save_config(updated_config)
        self.accept()

    def _save_and_close(self):
        updated_config = self._rebuild_config()
        save_config(updated_config)
        self.accept()

    def get_updated_config(self) -> Configuration:
        return self._rebuild_config()
