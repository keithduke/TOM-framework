#!/usr/bin/env python3
"""
T.O.M. GUI Launcher
Wraps CLI in dedicated window with system tray
FIXED: Output filtering and color support
"""

import sys
import re
import signal
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QTextEdit, QLineEdit,
    QVBoxLayout, QWidget, QSystemTrayIcon, QMenu, QLabel, QHBoxLayout
)
from PySide6.QtGui import (
    QIcon, QAction, QFont, QTextCursor, QTextCharFormat, 
    QColor, QTextOption
)
from PySide6.QtCore import Qt, QProcess, Signal, QTimer, Slot


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
    """Main window hosting CLI process"""
    
    quit_requested = Signal()
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("T.O.M. Assistant")
        self.setGeometry(100, 100, 900, 700)
        self.setMinimumSize(600, 400)
        
        self._setup_ui()
        self._setup_colors()
        self._start_cli_process()
        
        self.cli_ready = False
    
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
    
    def _start_cli_process(self):
        """Start CLI as subprocess"""
        self.process = QProcess()
        self.process.readyReadStandardOutput.connect(self._handle_stdout)
        self.process.readyReadStandardError.connect(self._handle_stderr)
        self.process.started.connect(self._on_process_started)
        self.process.finished.connect(self._on_process_finished)
        
        cli_script = Path(__file__).parent / "main.py"
        
        if not cli_script.exists():
            self._append_text(f"Error: CLI not found at {cli_script}\n", 'error')
            self.status_bar.set_status("CLI not found", "#f48771")
            return
        
        # Set environment
        env = QProcess.systemEnvironment()
        env.append("PYTHONUNBUFFERED=1")
        env.append("TERM=xterm-256color")
        self.process.setEnvironment(env)
        
        self.process.start(sys.executable, [str(cli_script)])
    
    def _on_process_started(self):
        """Called when CLI starts"""
        self.status_bar.set_status("Connected", "#4ec9b0")
        self._append_text("═" * 60 + "\n", 'info')
        self._append_text("       T.O.M. Assistant - GUI Edition\n", 'info')
        self._append_text("═" * 60 + "\n\n", 'info')
    
    def _on_process_finished(self, exit_code, exit_status):
        """Called when CLI exits"""
        self.status_bar.set_status("Disconnected", "#888")
        self._append_text(f"\n[Process exited with code {exit_code}]\n", 'dim')
        self.cli_ready = False
    
    def _send_input(self):
        """Send input to CLI"""
        text = self.input_field.text().strip()
        if not text:
            return
        
        # Handle quit
        if text.lower() in ['/quit', '/exit']:
            self._append_text("\nQuitting...\n", 'dim')
            self.quit_requested.emit()
            return
        
        self.input_field.clear()
        
        # Echo input
        self._append_text(f"\nYou> {text}\n", 'user')
        
        # Send to process
        if self.process.state() == QProcess.Running:
            self.process.write((text + "\n").encode('utf-8'))
        else:
            self._append_text("[Not connected]\n", 'error')
    
    def _handle_stdout(self):
        """Handle stdout - FIXED to not drop output"""
        data = self.process.readAllStandardOutput().data().decode('utf-8', errors='replace')
        
        # Strip ANSI codes
        clean = self._strip_ansi(data)
        
        # Skip terminal warnings
        if "Warning: Input is not a terminal" in clean or "not a terminal" in clean:
            return
        
        # Skip echo of "You>" prompts, but NOT the content after them
        if "You>" in clean:
            self.cli_ready = True
            # Only skip the prompt itself, keep everything else
            parts = clean.split("You>")
            # If there's content after "You>", keep it
            if len(parts) > 1:
                for i, part in enumerate(parts[1:], 1):
                    if part.strip():
                        clean = part
                        break
                else:
                    return  # Empty after prompt
            else:
                return  # Just the prompt
        
        if not clean.strip():
            return
        
        # Detect format based on content
        fmt = 'default'
        
        if "💭" in clean or "Thinking:" in clean:
            fmt = 'thinking'
        elif "T.O.M." in clean:
            fmt = 'assistant'
        elif "🔧" in clean or "Tool" in clean:
            fmt = 'tool'
        elif "ERROR" in clean or "Error" in clean:
            fmt = 'error'
        elif "INFO" in clean or "DEBUG" in clean:
            fmt = 'info'
        
        self._append_text(clean, fmt)
    
    def _handle_stderr(self):
        """Handle stderr"""
        data = self.process.readAllStandardError().data().decode('utf-8', errors='replace')
        clean = self._strip_ansi(data)
        
        if "Warning: Input is not a terminal" not in clean and clean.strip():
            self._append_text(clean, 'error')
    
    def _strip_ansi(self, text):
        """Remove ANSI escape codes"""
        # Remove color codes
        text = re.sub(r'\x1b\[[0-9;]*[mGKHF]', '', text)
        # Remove OSC sequences
        text = re.sub(r'\x1b\].*?\x07', '', text)
        # Remove carriage returns
        text = re.sub(r'\r', '', text)
        return text
    
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
        if self.process.state() == QProcess.Running:
            self.process.terminate()
            self.process.waitForFinished(2000)


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