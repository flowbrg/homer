# ui/widgets/chat_widget.py

from PySide6.QtWidgets import (
    QWidget, QListWidget, QTextEdit, QLineEdit,
    QPushButton, QHBoxLayout, QVBoxLayout, QSplitter
)


from PySide6.QtCore import QThread, Signal, QObject, Qt

from src.core.application import Application

class StreamWorker(QObject):
    update = Signal(str)
    finished = Signal(str)

    def __init__(self, backend: Application, query: str):
        super().__init__()
        self.backend = backend
        self.query = query
        self._in_think_block = False
        self._shown_thinking = False
        self._buffer = ""  # buffer for cases where <think> might span tokens

    def run(self):
        full_response = ""
        for token in self.backend.retrieval_stream(self.query):
            result = self._process_token(token)
            if result is not None:
                self.update.emit(result)
                if result != "Thinking...":
                    full_response += result
        self.finished.emit(full_response)

    def _process_token(self, token: str) -> str | None:
        self._buffer += token

        # If we detect <think> tag start
        if not self._in_think_block and "<think>" in self._buffer:
            self._in_think_block = True
            self._buffer = self._buffer.split("<think>", 1)[1]
            if not self._shown_thinking:
                self._shown_thinking = True
                return "Thinking..."

        # If inside a think block, wait until we see </think>
        if self._in_think_block:
            if "</think>" in self._buffer:
                # Exit the think block, keep what's after </think>
                _, after = self._buffer.split("</think>", 1)
                self._buffer = ""
                self._in_think_block = False
                return after
            return None  # Still inside <think> — skip output

        # If not inside a <think> block
        result = self._buffer
        self._buffer = ""
        return result

from src.core import database

from langchain_core.messages import AIMessage
from langchain_core.messages.human import HumanMessage

class ChatWidget(QWidget):

    chat_selected = Signal(bool)  # True if a chat is selected

    def __init__(self, backend: Application):
        super().__init__()
        self.backend= backend
        self.current_thread_id = None

        # Layouts
        splitter = QSplitter()
        splitter.setOrientation(Qt.Orientation.Horizontal)

        # Chat list
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        self.chat_listbox = QListWidget()
        self.chat_listbox.itemSelectionChanged.connect(self.on_chat_select)
        left_layout.addWidget(self.chat_listbox)

        # Display + entry area
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        right_layout.addWidget(self.chat_display)

        self.message_entry = QLineEdit()
        self.send_button = QPushButton("Send")
        entry_layout = QHBoxLayout()
        entry_layout.addWidget(self.message_entry)
        entry_layout.addWidget(self.send_button)
        right_layout.addLayout(entry_layout)

        self.send_button.clicked.connect(self.send_message)
        self.message_entry.returnPressed.connect(self.send_message)

        # Add both widgets to the splitter
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        #splitter.setSizes([75, 525])

        # Add splitter to main layout
        main_layout = QHBoxLayout(self)
        main_layout.addWidget(splitter)

        #self.refresh_chat_list()

    def on_chat_select(self):
        idx = self.chat_listbox.currentRow()
        if idx >= 0 and idx < len(self.chat_items):
            self.current_thread_id = self.chat_items[idx][0]
            self.update_chat_display()
        else:
            self.current_thread_id = None
        self._update_selection()

    def update_chat_display(self):
        self.chat_display.clear()
        if self.current_thread_id is None:
            return
        for message in self.backend.get_state(self.current_thread_id):
            if isinstance(message,HumanMessage):
                prefix = "<b>You:</b>" 
                color = "grey"
            elif isinstance(message, AIMessage): 
                prefix = "<b>Ai:</b>"
                color = "black"
            else:
                raise ValueError("Unexpected message type:", type(message))
            self.chat_display.append(f'<span style="color:{color}">{prefix} {message}</span><br>')

    def send_message(self):
        msg = self.message_entry.text().strip()
        if msg and self.current_thread_id is not None:
            # Store user message
            self.update_chat_display()
            self.message_entry.clear()

            # Start AI response streaming
            self.chat_display.append("AI: ")  # placeholder line
            self._start_streaming_response(msg)
    
    def set_entry_enabled(self, bool: bool):
        self.message_entry.setEnabled(bool)
        self.send_button.setEnabled(bool)

    def _start_streaming_response(self, query):
        self.stream_thread = QThread()
        self.stream_worker = StreamWorker(self.backend, query)
        self.stream_worker.moveToThread(self.stream_thread)

        self.stream_worker.update.connect(self._append_stream_token)
        self.stream_worker.finished.connect(self.stream_thread.quit)
        self.stream_worker.finished.connect(self.stream_worker.deleteLater)
        self.stream_thread.finished.connect(self.stream_thread.deleteLater)

        self.stream_thread.started.connect(self.stream_worker.run)
        self.stream_thread.start()

    def _append_stream_token(self, token: str):
        from PySide6.QtGui import QTextCursor
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(token)
        self.chat_display.setTextCursor(cursor)
        self.chat_display.ensureCursorVisible()

    def _update_selection(self):
        if self.current_thread_id == None:
            self.chat_selected.emit(False)
        else:
            self.chat_selected.emit(True)
    
    def refresh_chat_list(self, select_id: int | None = None):
        self.chat_listbox.clear()
        self.chat_items = []  # list of (thread_id, chat_name)
        for thread_id, name in database.get_all_threads():
            self.chat_listbox.addItem(name)
            self.chat_items.append((thread_id, name))
        if select_id:
            for i, (cid, _) in enumerate(self.chat_items):
                if cid == select_id:
                    self.chat_listbox.setCurrentRow(i)
                    break
        elif self.chat_items:
            self.chat_listbox.setCurrentRow(0)
        else:
            self.current_thread_id = None
        self._update_selection()

