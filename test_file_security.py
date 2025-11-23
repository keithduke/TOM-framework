import pytest
from pathlib import Path
from core.security import is_path_allowed, is_sensitive_file

def test_path_allowed_home():
    """Test that files in home directory are allowed."""
    test_file = Path.home() / "test.txt"
    assert is_path_allowed(test_file) is True

def test_path_denied_outside_allowed():
    """Test that files outside allowed paths are denied."""
    test_file = Path("/etc/passwd")
    assert is_path_allowed(test_file) is False

def test_sensitive_file_ssh_key():
    """Test that SSH keys are detected as sensitive."""
    test_file = Path.home() / ".ssh" / "id_rsa"
    assert is_sensitive_file(test_file) is True

def test_sensitive_file_env():
    """Test that .env files are detected as sensitive."""
    test_file = Path.cwd() / ".env"
    assert is_sensitive_file(test_file) is True

def test_non_sensitive_file():
    """Test that normal files are not sensitive."""
    test_file = Path.cwd() / "README.md"
    assert is_sensitive_file(test_file) is False
