"""Command-line entry point.

Stage entry points are imported lazily so ``--help`` doesn't touch torch.
"""

import json
from pathlib import Path

from .config import make_config
from .runtime import disk_check, is_done


def main(args=None):
    cfg = make_config(args)
    cfg["exp_dir"].mkdir(parents=True, exist_ok=True)

    print(f"\nMualem pipeline -- experiment: {cfg['name']}")
    print(f"  stages:      {cfg['stages']}")
    print(f"  working dir: {cfg['exp_dir']}")
    if cfg["kd_alpha"] > 0 or cfg["kd_pruning"]:
        print(f"  KD: alpha={cfg['kd_alpha']} "
              f"kd_pruning={cfg['kd_pruning']} "
              f"calib_batches={cfg['kd_calib_batches']}")
    if cfg["layer_target"] < 24 or cfg["hidden_target"] < 1024:
        print(f"  pruning: layers={cfg['layer_target']} "
              f"hidden={cfg['hidden_target']} "
              f"layer_score={cfg['layer_score']}")
    print()

    summary_path = cfg["exp_dir"] / "experiment_config.json"
    serialisable = {k: str(v) if isinstance(v, Path) else v
                    for k, v in cfg.items() if k not in ("loss_weights", "levels")}
    summary_path.write_text(json.dumps(serialisable, indent=2))

    for stage in ["prune", "finetune", "qat"]:
        if stage not in cfg["stages"]:
            print(f"  skip {stage} (not requested)")
            continue
        if is_done(cfg["exp_dir"], stage):
            print(f"  skip {stage} (already done)")
            continue
        _run_stage(stage, cfg)

    print(f"\nPipeline complete: {cfg['name']}")
    print(f"  INT8 model: {cfg['export_dir'] / 'model_quantized.safetensors'}")
    disk_check("pipeline complete")


def _run_stage(stage: str, cfg: dict) -> None:
    if stage == "prune":
        from .pruning import run_pruning
        run_pruning(cfg)
    elif stage == "finetune":
        from .finetune import run_finetune
        run_finetune(cfg)
    elif stage == "qat":
        from .qat import run_qat
        run_qat(cfg)
    else:
        raise ValueError(f"unknown stage: {stage}")
