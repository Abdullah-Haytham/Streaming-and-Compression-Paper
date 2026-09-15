"""Host detection, disk-space guard, stage markers, GPU telemetry."""

import os
import sys
from pathlib import Path

IN_COLAB     = "google.colab" in sys.modules
IN_KAGGLE    = os.path.exists("/kaggle")
IN_LIGHTNING = os.environ.get("LIGHTNING_CLOUD_PROJECT_ID") is not None

if IN_COLAB:
    ROOT = Path("/content/drive/MyDrive/mualem_pipeline")
    print("Host: Colab")
elif IN_KAGGLE:
    ROOT = Path("/kaggle/working/mualem_pipeline")
    print("Host: Kaggle")
elif IN_LIGHTNING:
    ROOT = Path("/teamspace/studios/this_studio/mualem_pipeline")
    print("Host: Lightning AI")
else:
    ROOT = Path("./mualem_pipeline")
    print("Host: local")


def disk_pct(path=ROOT) -> float:
    try:
        st = os.statvfs(str(path))
        used  = (st.f_blocks - st.f_bfree) * st.f_frsize
        total = st.f_blocks * st.f_frsize
        return used / total * 100
    except Exception:
        return 0.0


def disk_check(label: str = "", warn_at: float = 80.0, abort_at: float = 95.0):
    pct = disk_pct()
    tag = f" [{label}]" if label else ""
    if pct > abort_at:
        raise RuntimeError(f"disk {pct:.1f}% full{tag} -- aborting to prevent corruption")
    elif pct > warn_at:
        print(f"  disk {pct:.1f}% full{tag}")
    else:
        print(f"  disk {pct:.1f}%{tag}")


def mark_done(exp_dir: Path, stage: str):
    (exp_dir / f".{stage}_done").touch()


def is_done(exp_dir: Path, stage: str) -> bool:
    return (exp_dir / f".{stage}_done").exists()


def gpu_mem_mb() -> float:
    """GPU memory allocated in MB, or 0.0 if CUDA is unavailable."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1e6
    except Exception:
        pass
    return 0.0


def gpu_util_pct():
    """GPU utilisation %, or ``None`` if pynvml is unavailable."""
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        u = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
        pynvml.nvmlShutdown()
        return float(u)
    except Exception:
        return None
