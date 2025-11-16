#!/usr/bin/env python3
"""
T.O.M. GUI Launcher
Wraps CLI in dedicated window with system tray
FIXED: Output filtering and color support
"""

import os
import sys
import signal
from pathlib import Path
from typing import Optional

import httpx

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QTextEdit, QLineEdit,
    QVBoxLayout, QWidget, QSystemTrayIcon, QMenu, QLabel, QHBoxLayout
)
from PySide6.QtGui import (
    QIcon, QAction, QFont, QTextCursor, QTextCharFormat, 
    QColor, QTextOption
)
from PySide6.QtCore import Qt, Signal, Slot, QTimer, QThreadPool, QRunnable, QObject


class WorkerSignals(QObject):
    success = Signal(object)
    error = Signal(str)


class ApiWorker(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    def run(self):
        try:
            result = self.fn(*self.args, **self.kwargs)
        except Exception as exc:
            self.signals.error.emit(str(exc))
        else:
            self.signals.success.emit(result)


class TrayIcon(QSystemTrayIcon):
    """System tray icon with menu"""
    
    show_window = Signal()
    
    def __init__(self, icon_path=None):
        if icon_path and Path(icon_path).exists():
            icon = QIcon(str(icon_path))
        else:
            icon = QIcon.fromTheme("utilities-terminal")
        
        super().__init__(icon)
        self.setToolTip("T.O.M. Assistant")
        
        # Create menu
        menu = QMenu()
        
        show_action = QAction("Show Assistant", menu)
        show_action.triggered.connect(self.show_window.emit)
        
        quit_action = QAction("Quit", menu)
        quit_action.triggered.connect(QApplication.quit)
        
        menu.addAction(show_action)
        menu.addSeparator()
        menu.addAction(quit_action)
        
        self.setContextMenu(menu)
        self.activated.connect(self._on_activated)
    
    def _on_activated(self, reason):
        if reason == QSystemTrayIcon.Trigger:
            self.show_window.emit()


class StatusBar(QWidget):
    """Status bar showing connection state"""
    
    def __init__(self):
        super().__init__()
        layout = QHBoxLayout()
        layout.setContentsMargins(10, 5, 10, 5)
        
        self.status_label = QLabel("● Starting...")
        self.status_label.setStyleSheet("color: #888; font-size: 11px;")
        
        layout.addWidget(self.status_label)
        layout.addStretch()
        
        self.setLayout(layout)
        self.setStyleSheet("background-color: #1a1a1a; border-top: 1px solid #3e3e3e;")
    
    def set_status(self, text, color="#4ec9b0"):
        self.status_label.setText(f"● {text}")
        self.status_label.setStyleSheet(f"color: {color}; font-size: 11px;")


class TerminalWindow(QMainWindow):
    """Main window hosting API-driven assistant"""
    
    quit_requested = Signal()
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("T.O.M. Assistant")
        self.setGeometry(100, 100, 900, 700)
        self.setMinimumSize(600, 400)
        
        self.api_base = os.getenv("TOM_API_BASE", "http://127.0.0.1:8000")
        self.api_key = os.getenv("TOM_API_KEY")
        self.client: Optional[httpx.Client] = None
        self.session_id: Optional[str] = None
        self.thread_pool = QThreadPool.globalInstance()
        self.pending_request = False
        
        self._setup_ui()
        self._setup_colors()
        self._connect_api()
    
    def _setup_ui(self):
        """Setup UI components"""
        main_widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Output display
        self.output = QTextEdit()
        self.output.setReadOnly(True)
        
        # Monospace font
        font = QFont("Monaco", 12)
        if not font.exactMatch():
            font = QFont("Menlo", 12)
        if not font.exactMatch():
            font = QFont("Consolas", 11)
        if not font.exactMatch():
            font = QFont("Courier New", 11)
        
        self.output.setFont(font)
        self.output.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #d4d4d4;
                border: none;
                padding: 15px;
                selection-background-color: #264f78;
            }
        """)
        
        # Enable word wrap
        self.output.setLineWrapMode(QTextEdit.WidgetWidth)
        self.output.setWordWrapMode(QTextOption.WrapAtWordBoundaryOrAnywhere)
        
        layout.addWidget(self.output)
        
        # Input field
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Type your message...")
        self.input_field.returnPressed.connect(self._send_input)
        self.input_field.setFont(font)
        self.input_field.setStyleSheet("""
            QLineEdit {
                background-color: #252526;
                color: #ffffff;
                border: none;
                border-top: 2px solid #007acc;
                padding: 14px 15px;
                font-size: 13px;
            }
            QLineEdit:focus {
                background-color: #2d2d2d;
                border-top: 2px solid #00c9ff;
            }
        """)
        
        layout.addWidget(self.input_field)
        
        # Status bar
        self.status_bar = StatusBar()
        layout.addWidget(self.status_bar)
        
        main_widget.setLayout(layout)
        self.setCentralWidget(main_widget)
    
    def _setup_colors(self):
        """Setup text format colors (proper PySide6 approach)"""
        self.formats = {
            'default': self._make_format('#d4d4d4'),
            'user': self._make_format('#4ec9b0', bold=True),
            'assistant': self._make_format('#4ec9b0', bold=True),
            'thinking': self._make_format('#9cdcfe', italic=True),
            'tool': self._make_format('#ce9178'),
            'error': self._make_format('#f48771'),
            'info': self._make_format('#6a9fb5'),
            'dim': self._make_format('#888888'),
        }
    
    def _make_format(self, color, bold=False, italic=False):
        """Create QTextCharFormat with specified styling"""
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(color))
        if bold:
            fmt.setFontWeight(QFont.Bold)
        if italic:
            fmt.setFontItalic(True)
        return fmt
    
    def _connect_api(self):
        """Initialize HTTP client and session."""
        headers = {"X-TOM-API-Key": self.api_key} if self.api_key else None
        self.client = httpx.Client(base_url=self.api_base, timeout=60.0, headers=headers)
        self.status_bar.set_status("Connecting...", "#ffc107")
        self._append_text(f"Connecting to API at {self.api_base}\n", 'info')
        try:
            resp = self.client.post("/sessions", json={}, timeout=30)
            resp.raise_for_status()
            session = resp.json()
            self.session_id = session["session_id"]
            self.status_bar.set_status("Connected", "#4ec9b0")
            self._append_text("Connected to T.O.M. API\n", 'info')
        except httpx.HTTPError as exc:
            self.status_bar.set_status("API error", "#f48771")
            self._append_text(f"Failed to connect: {exc}\n", 'error')
    
    def _send_input(self):
        """Send input to API"""
        text = self.input_field.text().strip()
        if not text or self.pending_request:
            return
        
        # Handle quit
        if text.lower() in ['/quit', '/exit']:
            self._append_text("\nQuitting...\n", 'dim')
            self.quit_requested.emit()
            return
        
        self.input_field.clear()
        
        # Echo input
        self._append_text(f"\nYou> {text}\n", 'user')
        
        if not self.client or not self.session_id:
            self._append_text("[API not connected]\n", 'error')
            return
        
        self.pending_request = True
        self.input_field.setEnabled(False)
        self.status_bar.set_status("Waiting...", "#ffc107")
        
        worker = ApiWorker(self._chat_with_api, text)
        worker.signals.success.connect(self._handle_chat_success)
        worker.signals.error.connect(self._handle_chat_error)
        self.thread_pool.start(worker)
    
    def _chat_with_api(self, text: str):
        assert self.client and self.session_id
        payload = {"content": text, "run_tools": True}
        resp = self.client.post(f"/sessions/{self.session_id}/chat", json=payload, timeout=None)
        resp.raise_for_status()
        return resp.json()
    
    def _handle_chat_success(self, data: dict):
        self.pending_request = False
        self.input_field.setEnabled(True)
        self.status_bar.set_status("Connected", "#4ec9b0")
        
        session = data.get("session")
        if session:
            self._append_text("", 'default')  # ensure cursor at end
        thinking = data.get("thinking", "").strip()
        if thinking:
            self._append_text(f"\n💭 {thinking}\n", 'thinking')
        
        tool_calls = data.get("tool_calls", [])
        for tool in tool_calls:
            name = tool.get("name", "tool")
            output = tool.get("output", "")
            self._append_text(f"\n🔧 {name}\n", 'tool')
            self._append_text(f"{output}\n", 'tool')
        
        response = data.get("response", "").strip()
        if response:
            self._append_text(f"\nT.O.M.: {response}\n", 'assistant')
    
    def _handle_chat_error(self, message: str):
        self.pending_request = False
        self.input_field.setEnabled(True)
        self.status_bar.set_status("Error", "#f48771")
        self._append_text(f"\nError: {message}\n", 'error')
    
    def _append_text(self, text, format_name='default'):
        """Append text with proper color formatting"""
        if not text:
            return
        
        cursor = self.output.textCursor()
        cursor.movePosition(QTextCursor.End)
        
        # Apply format
        fmt = self.formats.get(format_name, self.formats['default'])
        cursor.setCharFormat(fmt)
        cursor.insertText(text)
        
        # Update cursor
        self.output.setTextCursor(cursor)
        
        # Auto-scroll if near bottom
        scrollbar = self.output.verticalScrollBar()
        at_bottom = scrollbar.value() >= scrollbar.maximum() - 10
        
        if at_bottom:
            scrollbar.setValue(scrollbar.maximum())
    
    def closeEvent(self, event):
        """Minimize to tray instead of closing"""
        self.hide()
        event.ignore()
    
    def cleanup(self):
        """Cleanup on quit"""
        if self.client:
            if self.session_id:
                try:
                    self.client.delete(f"/sessions/{self.session_id}", timeout=5)
                except httpx.HTTPError:
                    pass
            self.client.close()
            self.client = None


class TrayApplication:
    """Main application managing tray and window"""
    
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.app.setQuitOnLastWindowClosed(False)
        
        self.app.setApplicationName("T.O.M. Assistant")
        self.app.setOrganizationName("T.O.M.")
        
        # Create window
        self.window = TerminalWindow()
        self.window.quit_requested.connect(self.quit_app)
        
        # Create tray
        self.tray = TrayIcon()
        self.tray.show_window.connect(self.show_window)
        self.tray.show()
        
        # Show window initially
        self.window.show()
        
        # Cleanup handler
        self.app.aboutToQuit.connect(self.cleanup)
        
        # Signal handlers
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        """Setup Unix signal handlers"""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Timer for signal processing
        self.signal_timer = QTimer()
        self.signal_timer.timeout.connect(lambda: None)
        self.signal_timer.start(200)
    
    def _signal_handler(self, signum, frame):
        """Handle signals"""
        print("\nReceived interrupt, shutting down...")
        QTimer.singleShot(0, self.quit_app)
    
    @Slot()
    def quit_app(self):
        """Quit gracefully"""
        self.app.quit()
    
    @Slot()
    def show_window(self):
        """Show and raise window"""
        self.window.show()
        self.window.raise_()
        self.window.activateWindow()
    
    def cleanup(self):
        """Cleanup"""
        self.window.cleanup()
    
    def run(self):
        """Run application"""
        return self.app.exec()


def main():
    """Entry point"""
    app = TrayApplication()
    sys.exit(app.run())


if __name__ == "__main__":
    main()
