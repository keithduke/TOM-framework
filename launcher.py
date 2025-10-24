#!/usr/bin/env python3
"""
T.O.M. Tray Launcher
Wraps the CLI in a dedicated app window with menu bar presence
"""

import sys
import re
import os
import signal
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QTextEdit, QLineEdit,
    QVBoxLayout, QWidget, QSystemTrayIcon, QMenu, QLabel, QHBoxLayout
)
from PySide6.QtGui import QIcon, QAction, QFont, QTextCursor, QPalette, QColor, QTextOption
from PySide6.QtCore import Qt, QProcess, Signal, QTimer, Slot


class TrayIcon(QSystemTrayIcon):
    """System tray icon with menu"""
    
    show_window = Signal()
    
    def __init__(self, icon_path=None):
        # Try to use custom icon, fallback to system icon
        if icon_path and Path(icon_path).exists():
            icon = QIcon(str(icon_path))
        else:
            # Use a built-in icon as fallback
            icon = QIcon.fromTheme("utilities-terminal")
        
        super().__init__(icon)
        
        self.setToolTip("T.O.M. Assistant")
        
        # Create menu
        menu = QMenu()
        
        show_action = QAction("Show Assistant", menu)
        show_action.triggered.connect(self.show_window.emit)
        
        separator = menu.addSeparator()
        
        quit_action = QAction("Quit", menu)
        quit_action.triggered.connect(QApplication.quit)
        
        menu.addAction(show_action)
        menu.addAction(separator)
        menu.addAction(quit_action)
        
        self.setContextMenu(menu)
        
        # Left click also shows window
        self.activated.connect(self._on_activated)
    
    def _on_activated(self, reason):
        if reason == QSystemTrayIcon.Trigger:  # Left click
            self.show_window.emit()


class StatusBar(QWidget):
    """Custom status bar showing connection state"""
    
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
    """Main window hosting the CLI process"""
    
    # Signal to notify app to quit
    quit_requested = Signal()
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("T.O.M. Assistant")
        
        # Better default size
        self.setGeometry(100, 100, 800, 650)
        self.setMinimumSize(600, 400)
        
        # Optional: Make it look more polished
        # Uncomment for frameless floating window:
        # self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        
        self._setup_ui()
        self._start_cli_process()
        
        # Track if we've seen the initial prompt
        self.cli_ready = False
    
    def _setup_ui(self):
        """Setup the UI components"""
        # Main widget and layout
        main_widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Output display area
        self.output = QTextEdit()
        self.output.setReadOnly(True)
        
        # Set monospace font
        font = QFont("Monaco", 12)
        if not font.exactMatch():
            font = QFont("Menlo", 12)
        if not font.exactMatch():
            font = QFont("Consolas", 11)
        if not font.exactMatch():
            font = QFont("Courier New", 11)
        
        self.output.setFont(font)
        
        # Style the output area
        self.output.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #d4d4d4;
                border: none;
                padding: 15px;
                selection-background-color: #264f78;
            }
        """)
        
        # ENABLE WORD WRAP - This is the key fix!
        self.output.setLineWrapMode(QTextEdit.WidgetWidth)
        
        # Also enable word wrap at word boundaries for better readability
        self.output.setWordWrapMode(QTextOption.WrapAtWordBoundaryOrAnywhere)
        
        layout.addWidget(self.output)
        
        # Input container with better styling
        input_container = QWidget()
        input_layout = QHBoxLayout()
        input_layout.setContentsMargins(0, 0, 0, 0)
        input_layout.setSpacing(0)
        
        # Input field with better visual emphasis
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Type your message...")
        self.input_field.returnPressed.connect(self._send_input)
        self.input_field.setFont(font)
        
        # Enhanced input styling
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
        
        input_layout.addWidget(self.input_field)
        input_container.setLayout(input_layout)
        layout.addWidget(input_container)
        
        # Status bar
        self.status_bar = StatusBar()
        layout.addWidget(self.status_bar)
        
        main_widget.setLayout(layout)
        self.setCentralWidget(main_widget)
    
    def _start_cli_process(self):
        """Start the CLI as a subprocess"""
        self.process = QProcess()
        self.process.readyReadStandardOutput.connect(self._handle_stdout)
        self.process.readyReadStandardError.connect(self._handle_stderr)
        self.process.started.connect(self._on_process_started)
        self.process.finished.connect(self._on_process_finished)
        
        # Start the CLI
        cli_script = Path(__file__).parent / "main.py"
        
        if not cli_script.exists():
            self._append_text(
                f"Error: Could not find {cli_script}\n",
                color="#f48771"
            )
            self.status_bar.set_status("CLI not found", "#f48771")
            return
        
        # Set environment to suppress "not a terminal" warnings
        env = QProcess.systemEnvironment()
        env.append("PYTHONUNBUFFERED=1")  # Unbuffered output
        env.append("TERM=xterm-256color")  # Pretend to be a terminal
        self.process.setEnvironment(env)
        
        # Use sys.executable to ensure same Python interpreter
        self.process.start(sys.executable, [str(cli_script)])
    
    def _on_process_started(self):
        """Called when CLI process starts"""
        self.status_bar.set_status("Connected", "#4ec9b0")
        self._append_text("═" * 60 + "\n", color="#4ec9b0")
        self._append_text("       T.O.M. Assistant - Menu Bar Edition\n", color="#4ec9b0")
        self._append_text("═" * 60 + "\n\n", color="#4ec9b0")
    
    def _on_process_finished(self, exit_code, exit_status):
        """Called when CLI process exits"""
        self.status_bar.set_status("Disconnected", "#888")
        self._append_text(f"\n[Process exited with code {exit_code}]\n", color="#888")
        self.cli_ready = False
    
    def _send_input(self):
        """Send user input to the CLI process"""
        text = self.input_field.text().strip()
        if not text:
            return
        
        # Check for quit commands
        if text.lower() in ['/quit', '/exit']:
            self._append_text("\nQuitting T.O.M. Assistant...\n", color="#ce9178")
            # Signal the app to quit
            self.quit_requested.emit()
            return
        
        # Clear input field
        self.input_field.clear()
        
        # Echo user input (styled)
        self._append_text(f"\nYou> {text}\n", color="#4ec9b0", bold=True)
        
        # Send to process
        if self.process.state() == QProcess.Running:
            self.process.write((text + "\n").encode('utf-8'))
        else:
            self._append_text("[Not connected to CLI process]\n", color="#f48771")
    
    def _handle_stdout(self):
        """Handle standard output from CLI - optimized for smooth streaming"""
        data = self.process.readAllStandardOutput().data().decode('utf-8', errors='replace')
        
        # For smooth streaming, process the data directly without line buffering
        # This makes the output appear character-by-character like in the CLI
        
        # Strip ANSI codes from the incoming data
        clean_data = self._strip_ansi(data)
        
        # Skip certain unwanted output
        if "Warning: Input is not a terminal" in clean_data:
            return
        
        if "You>" in clean_data:
            self.cli_ready = True
            # Skip the prompt echo but keep any text after it
            parts = clean_data.split("You>", 1)
            if len(parts) > 1 and parts[1].strip():
                clean_data = parts[1]
            else:
                return
        
        # Detect coloring based on content
        color = None
        bold = False
        italic = False
        
        # Simple heuristic coloring
        if "INFO" in clean_data or "DEBUG" in clean_data:
            color = "#6a9fb5"
        elif "WARNING" in clean_data or "Warning" in clean_data:
            color = "#ce9178"
        elif "ERROR" in clean_data or "Error" in clean_data:
            color = "#f48771"
        elif "T.O.M." in clean_data:
            color = "#4ec9b0"
            bold = True
        elif "💭" in clean_data or "Thinking:" in clean_data:
            color = "#9cdcfe"
            italic = True
        else:
            color = "#d4d4d4"
        
        # Append directly for smooth streaming
        self._append_text(clean_data, color=color, bold=bold, italic=italic)
    
    def _handle_stderr(self):
        """Handle standard error from CLI"""
        data = self.process.readAllStandardError().data().decode('utf-8', errors='replace')
        clean_data = self._strip_ansi(data)
        
        # Skip terminal warnings
        if "Warning: Input is not a terminal" not in clean_data:
            self._append_text(clean_data, color="#f48771")
    
    def _strip_ansi(self, text):
        """Remove ANSI escape codes from text"""
        # Remove ANSI color codes
        ansi_escape = re.compile(r'\x1b\[[0-9;]*[mGKHF]')
        text = ansi_escape.sub('', text)
        
        # Remove other control sequences
        text = re.sub(r'\x1b\].*?\x07', '', text)  # OSC sequences
        text = re.sub(r'\r', '', text)  # Carriage returns
        
        return text
    
    def _append_text(self, text, color=None, bold=False, italic=False):
        """Append text to output display with proper formatting - optimized for streaming"""
        if not text:
            return
            
        cursor = self.output.textCursor()
        cursor.movePosition(QTextCursor.End)
        
        # For plain text (most common case), use faster text insertion
        if not color and not bold and not italic:
            cursor.insertText(text)
        else:
            # Build HTML for styled text
            style_parts = []
            if color:
                style_parts.append(f"color: {color}")
            
            style = "; ".join(style_parts) if style_parts else ""
            
            tags = []
            if bold:
                tags.append("b")
            if italic:
                tags.append("i")
            
            html = text
            
            # Escape HTML special characters to prevent issues
            html = html.replace('&', '&amp;')
            html = html.replace('<', '&lt;')
            html = html.replace('>', '&gt;')
            
            # Preserve spaces and newlines BEFORE wrapping in tags
            html = html.replace(' ', '&nbsp;')  # Preserve spaces
            html = html.replace('\n', '<br>')   # Convert newlines to breaks
            
            for tag in tags:
                html = f"<{tag}>{html}</{tag}>"
            
            if style:
                html = f"<span style='{style}'>{html}</span>"
            
            cursor.insertHtml(html)
        
        # Update cursor position
        self.output.setTextCursor(cursor)
        
        # Batch scroll updates to reduce jank
        # Only scroll if we're already near the bottom
        scrollbar = self.output.verticalScrollBar()
        at_bottom = scrollbar.value() >= scrollbar.maximum() - 10
        
        if at_bottom:
            scrollbar.setValue(scrollbar.maximum())
    
    def closeEvent(self, event):
        """Handle window close"""
        # Don't actually close - minimize to tray instead
        self.hide()
        event.ignore()
    
    def cleanup(self):
        """Cleanup when actually quitting"""
        if self.process.state() == QProcess.Running:
            self.process.terminate()
            self.process.waitForFinished(2000)


class TrayApplication:
    """Main application managing tray icon and window"""
    
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.app.setQuitOnLastWindowClosed(False)  # Keep running in tray
        
        # Set app metadata
        self.app.setApplicationName("T.O.M. Assistant")
        self.app.setOrganizationName("T.O.M.")
        
        # Create window
        self.window = TerminalWindow()
        
        # Connect quit signal from window
        self.window.quit_requested.connect(self.quit_app)
        
        # Create tray icon
        self.tray = TrayIcon()
        self.tray.show_window.connect(self.show_window)
        self.tray.show()
        
        # Show window on first launch
        self.window.show()
        
        # Connect quit signal
        self.app.aboutToQuit.connect(self.cleanup)
        
        # Setup signal handlers for Ctrl+C
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        # Handle Ctrl+C (SIGINT) and kill signals (SIGTERM)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Create a timer to check for signals periodically
        # (Qt event loop doesn't always catch Unix signals immediately)
        self.signal_timer = QTimer()
        self.signal_timer.timeout.connect(lambda: None)  # Just process events
        self.signal_timer.start(200)  # Check every 200ms
    
    def _signal_handler(self, signum, frame):
        """Handle Unix signals"""
        print("\nReceived interrupt signal, shutting down...")
        # Use QTimer to defer the quit to the Qt event loop
        QTimer.singleShot(0, self.quit_app)
    
    @Slot()
    def quit_app(self):
        """Quit the application gracefully"""
        self.app.quit()
    
    @Slot()
    def show_window(self):
        """Show and raise the window"""
        self.window.show()
        self.window.raise_()
        self.window.activateWindow()
    
    def cleanup(self):
        """Cleanup on quit"""
        self.window.cleanup()
    
    def run(self):
        """Run the application"""
        return self.app.exec()


def main():
    """Entry point"""
    app = TrayApplication()
    sys.exit(app.run())


if __name__ == "__main__":
    main()