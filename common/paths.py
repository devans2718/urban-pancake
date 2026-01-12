"""Path constants for the workspace."""

from pathlib import Path

# Root directory of the workspace
ROOT = Path(__file__).parent.parent

# Common directories
LOGS = ROOT / "logs"
DATA = ROOT / "data"
OUTPUT = ROOT / "output"

def ensure_directory(directory: Path) -> None:
    """
    Ensure that a directory exists.
    """

    directory.mkdir(parents=True, exist_ok=True)

# Create directories if they don't exist
ensure_directory(LOGS)
ensure_directory(DATA)
ensure_directory(OUTPUT)