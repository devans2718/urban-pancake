"""Path constants for the workspace."""

from pathlib import Path

# Root directory of the workspace
ROOT = Path(__file__).parent.parent

# Common directories
LOGS = ROOT / "logs"

# Create directories if they don't exist
LOGS.mkdir(exist_ok=True)