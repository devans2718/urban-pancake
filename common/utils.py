"""General utility functions."""

from datetime import datetime
from typing import Any
import json


def now_str(format: str = "%Y%m%d_%H%M%S") -> str:
    """
    Get current timestamp as string.

    Args:
        format: strftime format string

    Returns:
        Formatted timestamp string
    """
    return datetime.now().strftime(format)


def pretty_print(obj: Any, indent: int = 2) -> None:
    """
    Pretty print a JSON-serializable object.

    Args:
        obj: Object to print
        indent: Indentation level
    """
    print(json.dumps(obj, indent=indent, default=str))


def ensure_dir(path) -> None:
    """
    Ensure a directory exists, create if it doesn't.

    Args:
        path: Path to directory (string or Path object)
    """
    from pathlib import Path
    Path(path).mkdir(parents=True, exist_ok=True)


# Usage examples:
# ts = timestamp()  # '20260110_143022'
# pretty_print({'key': 'value', 'number': 42})
# ensure_dir('new/nested/directory')
