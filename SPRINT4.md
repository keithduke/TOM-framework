# Sprint 4 — Shell Access: Configuration & Audit

## Objective
Add user-friendly configuration system and comprehensive audit logging for shell access. Complete Sprint 4 with a production-ready shell tool that users can customize and monitor.

## Sprint 4 Deliverables

### 1. Configuration System
**Priority:** High | **Effort:** 1 week

#### 1.1 YAML Configuration File

Create `~/.tom/security.yaml` structure:

```yaml
# TOM Security Configuration
version: "1.0"

shell:
  # Enable or disable shell access
  enabled: false  # IMPORTANT: Must explicitly set to true

  # Security mode: "allowlist" or "blocklist"
  mode: allowlist

  # Commands allowed (when mode = allowlist)
  allowed_commands:
    # File operations (read-only)
    - ls
    - cat
    - head
    - tail
    - find
    - grep
    - wc
    - sort
    - uniq

    # Git operations
    - git

    # Development tools
    - python
    - python3
    - node
    - npm
    - pip
    - pytest

    # System information
    - ps
    - df
    - du
    - which
    - uname

  # Commands blocked (when mode = blocklist)
  blocked_commands:
    - rm
    - rmdir
    - sudo
    - su
    - kill
    - killall
    - dd
    - mkfs

  # Allowed working directories
  allowed_paths:
    - ~/Documents
    - ~/Projects
    - /tmp

  # Blocked paths (overrides allowed)
  blocked_paths:
    - ~/.ssh
    - ~/.aws
    - ~/.config/gcloud

  # Resource limits
  limits:
    # Maximum command execution time (seconds)
    timeout: 30

    # Maximum output size (characters)
    max_output: 10000

    # Maximum memory (MB) - future feature
    # max_memory: 512

    # Maximum CPU time (seconds) - future feature
    # max_cpu: 30

# File reading security (from Sprint 2)
file_reading:
  allowed_paths:
    - ~/Documents
    - ~/Projects
    - /tmp

  blocked_files:
    - .env
    - .env.local
    - credentials.json
    - id_rsa
    - id_ed25519

# Future: HTTP request security
# http:
#   allowed_domains:
#     - github.com
#     - api.example.com
#   blocked_domains:
#     - localhost
#   rate_limit: 10  # requests per minute
```

#### 1.2 Configuration Loader

Create `core/config_loader.py`:

```python
"""
Configuration loader for TOM security settings.
"""

import logging
import yaml
from pathlib import Path
from typing import Dict, Any, List, Set, Optional

logger = logging.getLogger("tom_cli")

# Default configuration
DEFAULT_CONFIG = {
    "version": "1.0",
    "shell": {
        "enabled": False,
        "mode": "allowlist",
        "allowed_commands": [
            "ls", "cat", "head", "tail", "grep", "find",
            "git", "python", "python3", "node", "npm",
        ],
        "blocked_commands": [
            "rm", "rmdir", "sudo", "su", "kill", "dd",
        ],
        "allowed_paths": [
            str(Path.home()),
            str(Path.cwd()),
            "/tmp",
        ],
        "blocked_paths": [
            str(Path.home() / ".ssh"),
            str(Path.home() / ".aws"),
        ],
        "limits": {
            "timeout": 30,
            "max_output": 10000,
        },
    },
    "file_reading": {
        "allowed_paths": [
            str(Path.home()),
            str(Path.cwd()),
            "/tmp",
        ],
        "blocked_files": [
            ".env", ".env.local", "credentials.json",
            "id_rsa", "id_ed25519", ".pem", ".key",
        ],
    },
}


class SecurityConfig:
    """
    Load and manage security configuration.
    """

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize configuration loader.

        Args:
            config_path: Path to security.yaml (None = use default)
        """
        if config_path is None:
            config_path = Path.home() / ".tom" / "security.yaml"

        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from file or create default.

        Returns:
            Configuration dictionary
        """
        if not self.config_path.exists():
            logger.info(f"No config found at {self.config_path}, using defaults")
            self._create_default_config()
            return DEFAULT_CONFIG.copy()

        try:
            with open(self.config_path) as f:
                user_config = yaml.safe_load(f)

            # Validate version
            version = user_config.get("version", "1.0")
            if version != "1.0":
                logger.warning(f"Unknown config version: {version}")

            # Merge with defaults (user config takes precedence)
            config = self._merge_configs(DEFAULT_CONFIG, user_config)

            logger.info(f"Loaded config from {self.config_path}")
            return config

        except yaml.YAMLError as e:
            logger.error(f"Failed to parse config: {e}")
            logger.info("Using default configuration")
            return DEFAULT_CONFIG.copy()

        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            logger.info("Using default configuration")
            return DEFAULT_CONFIG.copy()

    def _merge_configs(
        self, default: Dict[str, Any], user: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Recursively merge user config with defaults.

        Args:
            default: Default configuration
            user: User configuration

        Returns:
            Merged configuration
        """
        merged = default.copy()

        for key, value in user.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = value

        return merged

    def _create_default_config(self):
        """Create default configuration file."""
        config_dir = self.config_path.parent
        config_dir.mkdir(parents=True, exist_ok=True)

        try:
            with open(self.config_path, "w") as f:
                yaml.dump(DEFAULT_CONFIG, f, default_flow_style=False, sort_keys=False)

            logger.info(f"Created default config at {self.config_path}")
            logger.info("Please review and customize the configuration")

        except Exception as e:
            logger.error(f"Failed to create default config: {e}")

    # Shell configuration accessors

    @property
    def shell_enabled(self) -> bool:
        """Check if shell access is enabled."""
        return self.config.get("shell", {}).get("enabled", False)

    @property
    def shell_mode(self) -> str:
        """Get shell security mode (allowlist or blocklist)."""
        return self.config.get("shell", {}).get("mode", "allowlist")

    @property
    def shell_allowed_commands(self) -> Optional[Set[str]]:
        """
        Get allowed commands.

        Returns:
            Set of allowed commands, or None if allowlist not used
        """
        if self.shell_mode != "allowlist":
            return None

        commands = self.config.get("shell", {}).get("allowed_commands", [])
        return set(commands)

    @property
    def shell_blocked_commands(self) -> Set[str]:
        """Get blocked commands."""
        commands = self.config.get("shell", {}).get("blocked_commands", [])
        return set(commands)

    @property
    def shell_allowed_paths(self) -> List[Path]:
        """Get allowed working directories."""
        paths = self.config.get("shell", {}).get("allowed_paths", [])
        return [Path(p).expanduser() for p in paths]

    @property
    def shell_blocked_paths(self) -> Set[Path]:
        """Get blocked paths."""
        paths = self.config.get("shell", {}).get("blocked_paths", [])
        return {Path(p).expanduser() for p in paths}

    @property
    def shell_timeout(self) -> int:
        """Get default shell command timeout."""
        return self.config.get("shell", {}).get("limits", {}).get("timeout", 30)

    @property
    def shell_max_output(self) -> int:
        """Get maximum shell output size."""
        return self.config.get("shell", {}).get("limits", {}).get("max_output", 10000)

    # File reading configuration accessors

    @property
    def file_allowed_paths(self) -> List[Path]:
        """Get allowed paths for file reading."""
        paths = self.config.get("file_reading", {}).get("allowed_paths", [])
        return [Path(p).expanduser() for p in paths]

    @property
    def file_blocked_files(self) -> Set[str]:
        """Get blocked file patterns."""
        files = self.config.get("file_reading", {}).get("blocked_files", [])
        return set(files)

    # Configuration management

    def reload(self):
        """Reload configuration from disk."""
        logger.info("Reloading configuration")
        self.config = self._load_config()

    def validate(self) -> List[str]:
        """
        Validate configuration.

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Check shell configuration
        shell_config = self.config.get("shell", {})

        if shell_config.get("enabled") and shell_config.get("mode") not in ["allowlist", "blocklist"]:
            errors.append(f"Invalid shell mode: {shell_config.get('mode')}")

        if shell_config.get("mode") == "allowlist" and not shell_config.get("allowed_commands"):
            errors.append("Allowlist mode requires allowed_commands")

        # Check paths are valid
        for path_str in shell_config.get("allowed_paths", []):
            try:
                Path(path_str).expanduser()
            except Exception as e:
                errors.append(f"Invalid path {path_str}: {e}")

        return errors

    def get_summary(self) -> str:
        """
        Get human-readable configuration summary.

        Returns:
            Multi-line summary string
        """
        lines = ["TOM Security Configuration:"]
        lines.append(f"  Config file: {self.config_path}")
        lines.append(f"\nShell Access:")
        lines.append(f"  Enabled: {self.shell_enabled}")

        if self.shell_enabled:
            lines.append(f"  Mode: {self.shell_mode}")

            if self.shell_mode == "allowlist":
                lines.append(f"  Allowed commands: {len(self.shell_allowed_commands or [])}")
            else:
                lines.append(f"  Blocked commands: {len(self.shell_blocked_commands)}")

            lines.append(f"  Allowed paths: {len(self.shell_allowed_paths)}")
            lines.append(f"  Timeout: {self.shell_timeout}s")
            lines.append(f"  Max output: {self.shell_max_output:,} chars")

        return "\n".join(lines)


# Global configuration instance
_config_instance: Optional[SecurityConfig] = None


def get_security_config() -> SecurityConfig:
    """
    Get or create global security configuration.

    Returns:
        SecurityConfig instance
    """
    global _config_instance

    if _config_instance is None:
        _config_instance = SecurityConfig()

    return _config_instance


def reload_security_config():
    """Reload global security configuration from disk."""
    global _config_instance
    _config_instance = None
    return get_security_config()
```

#### 1.3 Update Shell Executor

Modify `core/shell_executor.py` to use configuration:

```python
from .config_loader import get_security_config

# In __init__, load from config
def __init__(self):
    """Initialize from security configuration."""
    config = get_security_config()

    self.allowed_commands = config.shell_allowed_commands
    self.blocked_commands = config.shell_blocked_commands
    self.allowed_paths = config.shell_allowed_paths
    self.blocked_paths = config.shell_blocked_paths
    self.max_output_chars = config.shell_max_output
    self.default_timeout = config.shell_timeout
```

**Tasks:**
- [ ] Create YAML configuration structure
- [ ] Implement `SecurityConfig` class
- [ ] Add validation and error handling
- [ ] Update shell executor to use config
- [ ] Create default config on first run
- [ ] Add config reload capability

**Success Criteria:**
- Configuration loads from YAML
- Defaults created if no config exists
- Validation catches errors
- Shell executor uses config values
- Config can be reloaded without restart

---

### 2. Audit Logging System
**Priority:** High | **Effort:** 1 week

#### 2.1 Audit Logger

Create `core/audit_logger.py`:

```python
"""
Audit logging for security-sensitive operations.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger("tom_cli")


class AuditLogger:
    """
    Log security-sensitive operations for review and compliance.
    """

    def __init__(self, log_path: Optional[Path] = None):
        """
        Initialize audit logger.

        Args:
            log_path: Path to audit log file (None = default)
        """
        if log_path is None:
            log_path = Path.home() / ".tom" / "audit.log"

        self.log_path = log_path

        # Ensure log directory exists
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Audit logging to {self.log_path}")

    def log_event(self, event_type: str, details: Dict[str, Any]):
        """
        Log an audit event.

        Args:
            event_type: Type of event (e.g., "shell_execute", "file_read")
            details: Event-specific details
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            **details,
        }

        try:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")

    def log_shell_command(
        self,
        command: str,
        working_dir: str,
        allowed: bool,
        success: Optional[bool] = None,
        exit_code: Optional[int] = None,
        output_size: Optional[int] = None,
        error: Optional[str] = None,
    ):
        """
        Log shell command execution.

        Args:
            command: Command that was executed
            working_dir: Working directory
            allowed: Whether command passed validation
            success: Whether command completed successfully
            exit_code: Command exit code
            output_size: Size of output in characters
            error: Error message if failed
        """
        details = {
            "command": command,
            "working_dir": working_dir,
            "allowed": allowed,
        }

        if success is not None:
            details["success"] = success
        if exit_code is not None:
            details["exit_code"] = exit_code
        if output_size is not None:
            details["output_size"] = output_size
        if error is not None:
            details["error"] = error

        self.log_event("shell_execute", details)

    def log_file_access(
        self,
        file_path: str,
        operation: str,  # "read" or "write"
        allowed: bool,
        success: Optional[bool] = None,
        size: Optional[int] = None,
        error: Optional[str] = None,
    ):
        """
        Log file access attempt.

        Args:
            file_path: Path to file
            operation: Operation type (read/write)
            allowed: Whether access was allowed
            success: Whether operation completed
            size: File size
            error: Error message if failed
        """
        details = {
            "file_path": file_path,
            "operation": operation,
            "allowed": allowed,
        }

        if success is not None:
            details["success"] = success
        if size is not None:
            details["size"] = size
        if error is not None:
            details["error"] = error

        self.log_event("file_access", details)

    def log_config_change(self, setting: str, old_value: Any, new_value: Any):
        """
        Log configuration change.

        Args:
            setting: Configuration setting that changed
            old_value: Previous value
            new_value: New value
        """
        details = {
            "setting": setting,
            "old_value": str(old_value),
            "new_value": str(new_value),
        }

        self.log_event("config_change", details)

    def get_recent_events(self, count: int = 50, event_type: Optional[str] = None) -> list:
        """
        Read recent audit events.

        Args:
            count: Number of events to return
            event_type: Filter by event type (None = all types)

        Returns:
            List of event dictionaries
        """
        if not self.log_path.exists():
            return []

        events = []

        try:
            with open(self.log_path) as f:
                for line in f:
                    try:
                        event = json.loads(line)

                        if event_type is None or event.get("event_type") == event_type:
                            events.append(event)

                    except json.JSONDecodeError:
                        continue

        except Exception as e:
            logger.error(f"Failed to read audit log: {e}")

        # Return most recent
        return events[-count:]

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get audit log statistics.

        Returns:
            Dictionary of statistics
        """
        if not self.log_path.exists():
            return {"total_events": 0}

        stats = {
            "total_events": 0,
            "by_type": {},
            "shell_commands": {
                "total": 0,
                "allowed": 0,
                "blocked": 0,
                "successful": 0,
                "failed": 0,
            },
            "file_access": {
                "total": 0,
                "read": 0,
                "write": 0,
                "allowed": 0,
                "blocked": 0,
            },
        }

        try:
            with open(self.log_path) as f:
                for line in f:
                    try:
                        event = json.loads(line)
                        stats["total_events"] += 1

                        event_type = event.get("event_type")
                        stats["by_type"][event_type] = stats["by_type"].get(event_type, 0) + 1

                        # Shell command stats
                        if event_type == "shell_execute":
                            stats["shell_commands"]["total"] += 1

                            if event.get("allowed"):
                                stats["shell_commands"]["allowed"] += 1
                            else:
                                stats["shell_commands"]["blocked"] += 1

                            if event.get("success"):
                                stats["shell_commands"]["successful"] += 1
                            elif event.get("success") is False:
                                stats["shell_commands"]["failed"] += 1

                        # File access stats
                        elif event_type == "file_access":
                            stats["file_access"]["total"] += 1

                            operation = event.get("operation", "unknown")
                            stats["file_access"][operation] = stats["file_access"].get(operation, 0) + 1

                            if event.get("allowed"):
                                stats["file_access"]["allowed"] += 1
                            else:
                                stats["file_access"]["blocked"] += 1

                    except json.JSONDecodeError:
                        continue

        except Exception as e:
            logger.error(f"Failed to compute statistics: {e}")

        return stats


# Global audit logger
_audit_logger: Optional[AuditLogger] = None


def get_audit_logger() -> AuditLogger:
    """
    Get or create global audit logger.

    Returns:
        AuditLogger instance
    """
    global _audit_logger

    if _audit_logger is None:
        _audit_logger = AuditLogger()

    return _audit_logger
```

#### 2.2 Integrate Audit Logging

Update `core/shell_executor.py`:

```python
from .audit_logger import get_audit_logger

def execute(self, command: str, ...) -> Dict[str, any]:
    """Execute with audit logging."""
    audit = get_audit_logger()

    # Log validation
    allowed, reason = self.is_command_allowed(command)
    audit.log_shell_command(
        command=command,
        working_dir=str(cwd),
        allowed=allowed,
    )

    if not allowed:
        return {
            "success": False,
            "error": f"Security: {reason}",
            # ...
        }

    # Execute command
    result = subprocess.run(...)

    # Log result
    audit.log_shell_command(
        command=command,
        working_dir=str(cwd),
        allowed=True,
        success=result.returncode == 0,
        exit_code=result.returncode,
        output_size=len(output),
        error=result.stderr if result.returncode != 0 else None,
    )

    return result
```

Update `core/tools.py` for file reading:

```python
from .audit_logger import get_audit_logger

def read_file(location: str) -> str:
    """Read file with audit logging."""
    audit = get_audit_logger()

    file_path = Path(location).expanduser().resolve()

    # Log access attempt
    if not is_path_allowed(file_path):
        audit.log_file_access(
            file_path=str(file_path),
            operation="read",
            allowed=False,
            error="Path not allowed",
        )
        return f"Error: Access denied..."

    # Read file
    try:
        with open(file_path) as f:
            content = f.read()

        audit.log_file_access(
            file_path=str(file_path),
            operation="read",
            allowed=True,
            success=True,
            size=len(content),
        )

        return content

    except Exception as e:
        audit.log_file_access(
            file_path=str(file_path),
            operation="read",
            allowed=True,
            success=False,
            error=str(e),
        )
        return f"Error: {e}"
```

**Tasks:**
- [ ] Create `AuditLogger` class
- [ ] Integrate with shell executor
- [ ] Integrate with file reading tool
- [ ] Add statistics computation
- [ ] Test audit log writing and reading

**Success Criteria:**
- All shell commands logged
- All file accesses logged
- Logs are JSON format (parseable)
- Statistics can be computed
- Log rotation not needed yet (future feature)

---

### 3. CLI Commands for Management
**Priority:** Medium | **Effort:** 3 days

Add new CLI commands to `ui/cli/cli.py`:

```python
def cmd_shell_history(self):
    """Display recent shell command history."""
    from core.audit_logger import get_audit_logger

    audit = get_audit_logger()
    events = audit.get_recent_events(count=50, event_type="shell_execute")

    if not events:
        console.print("[dim]No shell commands in history[/dim]")
        return

    table = Table(title="Shell Command History")
    table.add_column("Time", style="cyan", no_wrap=True)
    table.add_column("Command", style="yellow")
    table.add_column("Status", style="green", no_wrap=True)
    table.add_column("Exit", style="magenta", no_wrap=True)

    for event in events[-30:]:  # Show last 30
        time = event["timestamp"][:19].replace("T", " ")
        command = event["command"][:60]
        allowed = event.get("allowed", False)
        success = event.get("success")
        exit_code = event.get("exit_code", "")

        # Determine status
        if not allowed:
            status = "🚫 Blocked"
            style = "red"
        elif success is True:
            status = "✓ Success"
            style = "green"
        elif success is False:
            status = "✗ Failed"
            style = "red"
        else:
            status = "? Unknown"
            style = "dim"

        table.add_row(
            time,
            command,
            f"[{style}]{status}[/{style}]",
            str(exit_code) if exit_code != "" else "-",
        )

    console.print(table)


def cmd_audit_stats(self):
    """Display audit statistics."""
    from core.audit_logger import get_audit_logger

    audit = get_audit_logger()
    stats = audit.get_statistics()

    console.print("\n[bold]Audit Statistics[/bold]\n")

    # Overall
    console.print(f"Total events: {stats['total_events']:,}")

    # Shell commands
    shell = stats["shell_commands"]
    console.print(f"\n[bold cyan]Shell Commands:[/bold cyan]")
    console.print(f"  Total: {shell['total']}")
    console.print(f"  Allowed: [green]{shell['allowed']}[/green]")
    console.print(f"  Blocked: [red]{shell['blocked']}[/red]")
    console.print(f"  Successful: [green]{shell['successful']}[/green]")
    console.print(f"  Failed: [red]{shell['failed']}[/red]")

    # File access
    files = stats["file_access"]
    console.print(f"\n[bold cyan]File Access:[/bold cyan]")
    console.print(f"  Total: {files['total']}")
    console.print(f"  Reads: {files.get('read', 0)}")
    console.print(f"  Writes: {files.get('write', 0)}")
    console.print(f"  Allowed: [green]{files['allowed']}[/green]")
    console.print(f"  Blocked: [red]{files['blocked']}[/red]")


def cmd_config(self):
    """Display current security configuration."""
    from core.config_loader import get_security_config

    config = get_security_config()
    console.print("\n" + config.get_summary() + "\n")


def cmd_config_reload(self):
    """Reload configuration from disk."""
    from core.config_loader import reload_security_config

    console.print("[yellow]Reloading configuration...[/yellow]")

    config = reload_security_config()
    errors = config.validate()

    if errors:
        console.print("[red]Configuration errors:[/red]")
        for error in errors:
            console.print(f"  • {error}")
        console.print("\n[yellow]Using previous configuration[/yellow]")
    else:
        console.print("[green]✓ Configuration reloaded[/green]")
        console.print("\n" + config.get_summary())


def cmd_config_edit(self):
    """Open configuration file in editor."""
    from core.config_loader import get_security_config
    import os
    import subprocess

    config = get_security_config()
    config_path = config.config_path

    # Get editor from environment or use default
    editor = os.environ.get("EDITOR", "nano")

    console.print(f"[yellow]Opening {config_path} in {editor}...[/yellow]")

    try:
        subprocess.run([editor, str(config_path)])
        console.print("\n[yellow]After editing, use /config-reload to apply changes[/yellow]")
    except Exception as e:
        console.print(f"[red]Failed to open editor: {e}[/red]")
        console.print(f"[dim]Manually edit: {config_path}[/dim]")


# Register new commands
COMMANDS = {
    # ... existing commands ...
    "/shell-history": cmd_shell_history,
    "/audit-stats": cmd_audit_stats,
    "/audit": cmd_audit_stats,  # Alias
    "/config": cmd_config,
    "/config-reload": cmd_config_reload,
    "/config-edit": cmd_config_edit,
}
```

**Tasks:**
- [ ] Add `/shell-history` command
- [ ] Add `/audit-stats` command
- [ ] Add `/config` command to view current settings
- [ ] Add `/config-reload` command
- [ ] Add `/config-edit` command
- [ ] Update `/help` with new commands

**Success Criteria:**
- Users can view shell history
- Users can view audit statistics
- Users can view and reload config
- Commands are well-documented

---

### 4. Web UI Integration
**Priority:** Medium | **Effort:** 3 days

Add configuration and audit UI to web client.

Create `ui/web/static/config.html`:

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>TOM Configuration</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body>
    <div class="container">
        <h1>Security Configuration</h1>

        <div class="config-section">
            <h2>Shell Access</h2>
            <div class="config-item">
                <label>
                    <input type="checkbox" id="shell-enabled">
                    Enable Shell Access
                </label>
            </div>

            <div class="config-item">
                <label>Security Mode:</label>
                <select id="shell-mode">
                    <option value="allowlist">Allowlist (Recommended)</option>
                    <option value="blocklist">Blocklist</option>
                </select>
            </div>

            <!-- More configuration options -->
        </div>

        <div class="audit-section">
            <h2>Audit Log</h2>
            <div id="audit-stats"></div>
            <div id="audit-events"></div>
        </div>

        <button id="save-config">Save Configuration</button>
        <button id="reload-config">Reload from Disk</button>
    </div>

    <script src="config.js"></script>
</body>
</html>
```

Add API endpoints to `services/api/routes.py`:

```python
@router.get("/config")
def get_config():
    """Get current security configuration."""
    from core.config_loader import get_security_config

    config = get_security_config()
    return {
        "shell": {
            "enabled": config.shell_enabled,
            "mode": config.shell_mode,
            "allowed_commands": list(config.shell_allowed_commands or []),
            "blocked_commands": list(config.shell_blocked_commands),
            "timeout": config.shell_timeout,
            "max_output": config.shell_max_output,
        }
    }


@router.get("/audit/stats")
def get_audit_stats():
    """Get audit log statistics."""
    from core.audit_logger import get_audit_logger

    audit = get_audit_logger()
    return audit.get_statistics()


@router.get("/audit/events")
def get_audit_events(count: int = 50, event_type: Optional[str] = None):
    """Get recent audit events."""
    from core.audit_logger import get_audit_logger

    audit = get_audit_logger()
    events = audit.get_recent_events(count=count, event_type=event_type)
    return {"events": events}
```

**Tasks:**
- [ ] Create config viewing UI
- [ ] Create audit log viewer UI
- [ ] Add API endpoints for config and audit
- [ ] Implement config editing (read-only for now)
- [ ] Add shell history table

**Success Criteria:**
- Web UI shows current configuration
- Web UI shows audit statistics
- Web UI shows recent events
- Real-time updates for audit log

---

## Sprint 4 Timeline

**Total Effort:** 2 weeks

- **Week 1:** Configuration system (Days 1-5)
- **Week 2:** Audit logging and UI (Days 6-10)

### Detailed Schedule

**Days 1-2:** YAML configuration and loader
**Day 3:** Integrate config with shell executor
**Day 4:** Configuration validation and testing
**Day 5:** CLI configuration commands
**Days 6-7:** Audit logger implementation
**Day 8:** Integrate audit logging with tools
**Day 9:** CLI audit commands
**Day 10:** Web UI for config/audit (basic)
**Days 11-12:** Testing and refinement
**Days 13-14:** Documentation and polish

## Testing Checklist

### Configuration System
- [ ] Default config created correctly
- [ ] User config loaded and merged
- [ ] Validation catches errors
- [ ] Shell executor uses config
- [ ] Config reload works without restart
- [ ] Invalid YAML handled gracefully

### Audit Logging
- [ ] Shell commands logged
- [ ] File access logged
- [ ] Log format is valid JSON
- [ ] Statistics computed correctly
- [ ] Recent events retrieved correctly
- [ ] Concurrent logging works

### CLI Commands
- [ ] `/shell-history` shows events
- [ ] `/audit-stats` shows statistics
- [ ] `/config` shows current settings
- [ ] `/config-reload` reloads config
- [ ] `/config-edit` opens editor
- [ ] All commands handle errors

### Web UI
- [ ] Configuration displayed correctly
- [ ] Audit stats displayed
- [ ] Event history displayed
- [ ] UI responsive and functional

## Success Metrics

- **Configuration**: Users can customize security
- **Audit Logging**: All sensitive operations logged
- **Transparency**: Users can review what happened
- **Usability**: Clear, easy-to-use commands
- **Documentation**: Complete guides and examples

## Documentation Tasks

- [ ] Update README with configuration section
- [ ] Document all new CLI commands
- [ ] Add audit log format specification
- [ ] Create configuration guide
- [ ] Add troubleshooting section
- [ ] Update SECURITY.md with audit info

## Success Criteria

- [ ] YAML configuration system working
- [ ] All tools use configuration
- [ ] Audit logging complete
- [ ] CLI commands functional
- [ ] Web UI shows config/audit
- [ ] Documentation complete
- [ ] All tests passing

---

**Status:** 🚧 Ready to Start
**Dependencies:** Sprint 3 (shell implementation)
**Next Sprint:** [SPRINT5.md](SPRINT5.md) - Advanced Tools & Improvements
