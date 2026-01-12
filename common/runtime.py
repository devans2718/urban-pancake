"""Runtime / environment detection utilities.

Designed for a uv-managed mono-repo where scripts may run from many locations.
Keeps logic small, dependency-light, and safe to import.
"""

from dataclasses import asdict, dataclass
import os
import platform
from typing import Any


def is_wsl() -> bool:
    """Return True if running inside WSL (Windows Subsystem for Linux)."""
    # WSL sets one or both of these in most environments.
    if os.getenv("WSL_DISTRO_NAME") or os.getenv("WSL_INTEROP"):
        return True

    # Fallback: look at kernel release for "microsoft".
    try:
        return "microsoft" in platform.release().lower()
    except Exception:
        return False


def has_cuda() -> bool:
    """Return True if PyTorch is installed and CUDA is available."""
    try:
        import torch  # type: ignore
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def default_device() -> str:
    """Return the preferred device string: 'cuda' if available else 'cpu'."""
    return "cuda" if has_cuda() else "cpu"


@dataclass(frozen=True)
class RuntimeInfo:
    python: str
    platform: str
    is_wsl: bool
    torch: str | None
    torch_cuda: str | None
    cuda_available: bool
    cuda_device_count: int | None
    cuda_device_name0: str | None


def runtime_info() -> dict[str, Any]:
    """Return a JSON-serializable dict describing the runtime."""
    py = platform.python_version()
    plat = f"{platform.system()} {platform.release()}"
    wsl = is_wsl()

    torch_version = None
    torch_cuda = None
    cuda_available = False
    device_count = None
    device0 = None

    try:
        import torch  # type: ignore

        torch_version = getattr(torch, "__version__", None)
        torch_cuda = getattr(getattr(torch, "version", None), "cuda", None)
        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            device_count = int(torch.cuda.device_count())
            if device_count > 0:
                device0 = torch.cuda.get_device_name(0)
    except Exception:
        pass

    info = RuntimeInfo(
        python=py,
        platform=plat,
        is_wsl=wsl,
        torch=torch_version,
        torch_cuda=torch_cuda,
        cuda_available=cuda_available,
        cuda_device_count=device_count,
        cuda_device_name0=device0,
    )
    return asdict(info)

# Example usage:
# info = runtime_info()
# print(info)
# logger = get_logger()
# logger.info(f"Runtime info: {info}")
