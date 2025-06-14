# ui/main_window.py

from PySide6.QtWidgets import QMainWindow, QHBoxLayout, QWidget, QToolBar
from PySide6.QtGui import QAction, QIcon
from src.ui.widgets.chat_widget import ChatWidget
from src.ui.dialogs.settings_dialog import SettingsDialog
from src.core.application import Application
from src.core import database

class MainWindow(QMainWindow):
    def __init__(self,backend: Application):
        super().__init__()
        self.setWindowIcon(QIcon("./src/resources/icons/reduce.png"))
        self.setWindowTitle("Homer")
        self.resize(700, 400)

        # Backend logic
        self.backend = backend

        # Define buttons
        self.delete_chat_action = QAction(QIcon("./src/resources/icons/minus.png"),"Delete",self)
        self.delete_chat_action.setStatusTip("Delete current chat")
        self.delete_chat_action.triggered.connect(self._delete_current_chat)
        self.delete_chat_action.setIconVisibleInMenu(False)  # Hide icon in menu
        #self.delete_chat_action.setEnabled(False)  # Initially disabled

        self.new_chat_action = QAction(QIcon("./src/resources/icons/plus.png"),"New", self)
        self.new_chat_action.setStatusTip("Create new chat")
        self.new_chat_action.triggered.connect(self._create_new_chat)
        self.new_chat_action.setIconVisibleInMenu(False)  # Hide icon in menu

        # Central widget setup
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.main_layout = QHBoxLayout(central_widget)

        # Main widget
        self.chat_widget = ChatWidget(backend=self.backend)
        self.main_layout.addWidget(self.chat_widget)

        # Connect signal to enable/disable delete buttons
        self.chat_widget.chat_selected.connect(self._update_delete_actions)

        self._create_tool_bar()
        self._create_menu_bar()
        
        self.chat_widget.refresh_chat_list()

    def _create_tool_bar(self):
        self.toolbar = QToolBar()
        self.addToolBar(self.toolbar)

        self.toolbar.addAction(self.new_chat_action)
        self.toolbar.addAction(self.delete_chat_action)

    def _create_menu_bar(self):
        self.menu_bar = self.menuBar()

        # File
        self.file_menu = self.menu_bar.addMenu("File")

        self.quit_action = QAction("Quit", self)
        self.quit_action.triggered.connect(self.close)
        
        self.file_menu.addAction(self.new_chat_action)
        self.file_menu.addAction(self.delete_chat_action)
        self.file_menu.addAction(self.quit_action)

        # Preferences
        pref_menu = self.menu_bar.addMenu("Preferences")

        settings_action = QAction("Settings", self)
        settings_action.triggered.connect(self._open_preferences)

        pref_menu.addAction(settings_action)

    def _open_preferences(self):
        dialog = SettingsDialog(self.backend.get_config(), parent=self)
        if dialog.exec():
            self.backend.set_config(dialog.get_updated_config())
            

    def _create_new_chat(self):
        from PySide6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(self, "New thread", "Create a new thread?")
        existing_thread_ids = database.get_all_threads()
        if ok and name:
            new_id = database.new_thread(
                thread_id=(int(existing_thread_ids[-1][0]) + 1) if existing_thread_ids else 1,
                thread_name=name
            )
            self.chat_widget.refresh_chat_list(select_id=new_id)
    
    def _delete_current_chat(self):
        from PySide6.QtWidgets import QMessageBox
        current_thread_id = self.chat_widget.get_current_thread_id()
        if current_thread_id is None:
            QMessageBox.warning(self, "Error", "Aucune discussion sélectionnée.")
            return

        chat_name = self.chat_widget.chat_listbox.currentItem().text()
        confirm = QMessageBox.question(
            self,
            "Confirmation",
            f"Supprimer la discussion « {chat_name} » ? Cette action est irréversible.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if confirm == QMessageBox.StandardButton.Yes:
            database.delete_thread(current_thread_id)
            self.chat_widget.refresh_chat_list()

    def _update_delete_actions(self, chat_selected: bool):
        self.delete_chat_action.setEnabled(chat_selected)
        self.chat_widget.set_entry_enabled(chat_selected)