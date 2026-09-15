"""Per-epoch LR scheduler shared by the FT and QAT stages.

Stepped once per epoch because the pipeline streams a fixed sample budget per
epoch (the per-epoch step count isn't known up front). The scheduler is a
``LambdaLR`` whose lambda closes over the schedule params -- the resume path
replays the lambda for every epoch up to ``last_epoch`` and reaches the same
lr the run had, with no scheduler state to persist.
"""

import math


def build_lr_scheduler(optimizer, cfg: dict, total_epochs: int,
                       last_completed_epoch: int = 0):
    """Return a ``LambdaLR`` or ``None``.

    ``None`` means use the optimizer's fixed lr (constant schedule with no
    warmup). Multiplier semantics:

    - epoch in [0, warmup): linear warmup 0 -> 1
    - epoch in [warmup, total_epochs): 1 -> floor, following the schedule

    ``floor`` is ``lr_min / base_lr`` so the curve lands on ``--lr_min``.
    """
    from torch.optim.lr_scheduler import LambdaLR

    schedule = cfg.get("lr_schedule", "cosine")
    if schedule == "constant" and int(cfg.get("lr_warmup_epochs", 0)) <= 0:
        return None

    warmup    = max(0, int(cfg.get("lr_warmup_epochs", 0)))
    base_lrs  = [g["lr"] for g in optimizer.param_groups]
    base_lr   = base_lrs[0] if base_lrs else 1.0
    lr_min    = float(cfg.get("lr_min", 0.0))
    floor_mult = (lr_min / base_lr) if base_lr > 0 else 0.0
    floor_mult = min(max(floor_mult, 0.0), 1.0)

    # warmup >= total_epochs: hold at base lr after warmup rather than blow up.
    decay_epochs = max(1, total_epochs - warmup)

    def lr_lambda(epoch: int) -> float:
        if warmup > 0 and epoch < warmup:
            # +1 so the first epoch isn't a wasted zero-lr step.
            return float(epoch + 1) / float(warmup)

        if schedule == "constant":
            return 1.0

        prog = (epoch - warmup) / float(decay_epochs)
        prog = min(max(prog, 0.0), 1.0)

        if schedule == "cosine":
            cosine = 0.5 * (1.0 + math.cos(math.pi * prog))
            return floor_mult + (1.0 - floor_mult) * cosine
        elif schedule == "linear":
            return floor_mult + (1.0 - floor_mult) * (1.0 - prog)
        else:
            return 1.0

    # last_epoch == -1 -> fresh run, first step() is epoch 0.
    last_epoch = last_completed_epoch - 1
    # On resume LambdaLR needs each param group to carry 'initial_lr' (it's
    # only stamped automatically when last_epoch == -1). The optimizer is
    # rebuilt fresh each resume with its base lr intact, so seed from the
    # live lr.
    if last_epoch >= 0:
        for g in optimizer.param_groups:
            g.setdefault("initial_lr", g["lr"])
    return LambdaLR(optimizer, lr_lambda=lr_lambda, last_epoch=last_epoch)
