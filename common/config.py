"""Configuration and environment management."""

import os
from pathlib import Path
from typing import Optional


def load_env(env_file: Optional[str] = ".env") -> None:
    """
    Load environment variables from a file.

    Args:
        env_file: Path to environment file relative to workspace root
    """
    try:
        from dotenv import load_dotenv
        from .paths import ROOT

        env_path = ROOT / env_file
        if env_path.exists():
            load_dotenv(env_path)
        else:
            print(f"Warning: {env_file} not found at {env_path}")
    except ImportError:
        print("Warning: python-dotenv not installed. Install with: uv add python-dotenv")


def get_env(key: str, default: Optional[str] = None) -> Optional[str]:
    """
    Get environment variable with optional default.

    Args:
        key: Environment variable name
        default: Default value if not found

    Returns:
        Environment variable value or default
    """
    return os.getenv(key, default)


def get_bool_env(key: str, default: bool = False) -> bool:
    """
    Get environment variable as boolean with optional default.

    Args:
        key: Environment variable name
        default: Default value if not found

    Returns:
        Boolean value or default
    """
    value = get_env(key, str(default)).lower()
    return value in ("true", "1", "yes", "on")


def get_int_env(key: str, default: int = 0) -> int:
    """
    Get environment variable as integer with optional default.

    Args:
        key: Environment variable name
        default: Default value if not found

    Returns:
        Integer value or default
    """
    try:
        return int(get_env(key, str(default)))
    except (ValueError, TypeError):
        return default


# Example usage:
# load_env()
# API_KEY = get_env("API_KEY")
# DEBUG_MODE = get_env("DEBUG", "false").lower() == "true"
