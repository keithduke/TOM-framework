from pathlib import Path
from typing import List, Set, Optional

# Configurable allowed paths
DEFAULT_ALLOWED_PATHS = [
    Path.home(),
    Path.cwd(),
    Path("/tmp"),
]

# Sensitive files blocklist
SENSITIVE_FILES = {
    ".env",
    ".env.local",
    ".env.production",
    "credentials.json",
    "secrets.yaml",
    "id_rsa",
    "id_ed25519",
    "id_ecdsa",
    "id_dsa",
    ".pem",
    ".key",
}

# Sensitive directories
SENSITIVE_DIRS = {
    ".ssh",
    ".aws",
    ".azure",
    ".config/gcloud",
}

def is_path_allowed(file_path: Path, allowed_paths: Optional[List[Path]] = None) -> bool:
    """
    Validate that file_path is within allowed directories.

    Args:
        file_path: Path to validate
        allowed_paths: List of allowed base paths (uses defaults if None)

    Returns:
        True if path is allowed, False otherwise
    """
    if allowed_paths is None:
        allowed_paths = DEFAULT_ALLOWED_PATHS

    resolved = file_path.resolve()

    # Check against allowed paths
    for allowed in allowed_paths:
        try:
            resolved.relative_to(allowed.resolve())
            return True
        except ValueError:
            continue

    return False

def is_sensitive_file(file_path: Path) -> bool:
    """
    Check if file is in sensitive files list.

    Args:
        file_path: Path to check

    Returns:
        True if file is sensitive, False otherwise
    """
    # Check filename
    if file_path.name in SENSITIVE_FILES:
        return True

    # Check suffix
    if file_path.suffix in {".pem", ".key"}:
        return True

    # Check if in sensitive directory
    for part in file_path.parts:
        if part in SENSITIVE_DIRS:
            return True

    return False
