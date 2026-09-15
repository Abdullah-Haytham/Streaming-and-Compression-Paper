"""Checkpoint save/load shared by fine-tuning and QAT.

On-disk layout: one safetensors file for the weights plus three ``.pt`` files
for the optimizer / AMP scaler / training state, all in ``ckpt_dir``.
``load_checkpoint`` pulls a missing checkpoint from W&B when a logger is
supplied, so a fresh machine can resume from the artifact store alone.
"""

import json
from pathlib import Path

from .runtime import disk_check


def save_checkpoint(model, optimizer, epoch, log, ckpt_dir: Path, best=False,
                    scaler=None):
    import torch
    from safetensors.torch import save_file

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    sd = {n: p.data for n, p in model.named_parameters()}
    sd.update({n: b for n, b in model.named_buffers()})
    save_file(sd, str(ckpt_dir / "model_latest.safetensors"))
    if best:
        save_file(sd, str(ckpt_dir / "model_best.safetensors"))
    torch.save(optimizer.state_dict(), str(ckpt_dir / "optimizer_latest.pt"))

    # Persist the AMP loss-scale alongside the optimizer so it survives a
    # resume. Only present on CUDA + mixed-precision runs.
    scaler_path = ckpt_dir / "scaler_latest.pt"
    if scaler is not None:
        torch.save(scaler.state_dict(), str(scaler_path))
    else:
        # Drop any stale scaler file so a CPU-only resume doesn't pick it up.
        if scaler_path.exists():
            scaler_path.unlink()

    (ckpt_dir / "training_state.json").write_text(
        json.dumps({"completed_epoch": epoch, "training_log": log}, indent=2))
    print(f"  checkpoint saved (epoch {epoch})")
    disk_check("after checkpoint")


def load_checkpoint(ckpt_dir: Path, model, optimizer, device, scaler=None,
                    wandb_logger=None):
    """Return ``(completed_epoch, training_log)``.

    When no local checkpoint exists and a ``wandb_logger`` is supplied, the
    latest stage artifact is downloaded into ``ckpt_dir`` first. Local
    checkpoints always win: an interrupted run on the same machine resumes
    from disk and never touches W&B.
    """
    import torch

    state_path = ckpt_dir / "training_state.json"

    if not state_path.exists() and wandb_logger is not None:
        try:
            wandb_logger.restore_checkpoint(ckpt_dir)
        except Exception as e:
            print(f"  W&B checkpoint restore failed ({e})")

    if not state_path.exists():
        print("  no checkpoint -- starting from scratch")
        return 0, []

    state = json.loads(state_path.read_text())
    ep, log = state["completed_epoch"], state.get("training_log", [])

    wp = ckpt_dir / "model_latest.safetensors"
    if wp.exists():
        from safetensors.torch import load_file
        import torch.nn as nn
        sd = load_file(str(wp), device=str(device))
        for name, tensor in sd.items():
            parts = name.split(".")
            obj = model
            try:
                for p in parts[:-1]:
                    obj = getattr(obj, p)
            except AttributeError:
                continue
            attr = parts[-1]
            if attr in obj._parameters:
                obj._parameters[attr] = nn.Parameter(tensor)
            elif attr in obj._buffers:
                obj._buffers[attr] = tensor

    op = ckpt_dir / "optimizer_latest.pt"
    if op.exists():
        optimizer.load_state_dict(torch.load(str(op), map_location=device, weights_only=True))

    sp = ckpt_dir / "scaler_latest.pt"
    if scaler is not None and sp.exists():
        try:
            scaler.load_state_dict(
                torch.load(str(sp), map_location=device, weights_only=True))
            print("  restored AMP GradScaler state")
        except Exception as e:
            print(f"  could not restore GradScaler ({e}); using fresh scaler")
    elif scaler is not None and not sp.exists():
        print("  no saved GradScaler -- using fresh scaler")

    print(f"  resuming from epoch {ep + 1}")
    return ep, log
