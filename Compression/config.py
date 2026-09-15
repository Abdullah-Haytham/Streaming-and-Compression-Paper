"""Argparse-based config builder.

``make_config`` returns a plain ``dict`` so callers can mutate it freely for
derived paths and the static loss-weight table. Validation lives here so stage
code stays free of defensive checks.
"""

import argparse
import sys

from .runtime import ROOT


# Default subsets of the muaalem-annotated-v3 dataset. Module-level so it shows
# up in --help and is easy to override.
ALL_MOSHAFS = [
    "moshaf_0.0",  "moshaf_0.1",  "moshaf_0.2",  "moshaf_0.3",
    "moshaf_1.0",  "moshaf_2.0",  "moshaf_2.1",  "moshaf_3.0",
    "moshaf_4.0",  "moshaf_5.0",  "moshaf_6.0",  "moshaf_7.0",
    "moshaf_8.0",  "moshaf_9.0",  "moshaf_11.0", "moshaf_12.0",
    "moshaf_13.0", "moshaf_19.0", "moshaf_22.0", "moshaf_24.0",
    "moshaf_25.0", "moshaf_26.0", "moshaf_26.1", "moshaf_27.0",
    "moshaf_28.0", "moshaf_29.0", "moshaf_30.0",
]


# Per-level loss weights (thesis Eq 5.1). The dict order also defines
# ``cfg["levels"]``.
LOSS_WEIGHTS = {
    "phonemes":            0.4,
    "ghonna":              0.059875,
    "hams_or_jahr":        0.059875,
    "istitala":            0.059875,
    "itbaq":               0.059875,
    "qalqla":              0.059875,
    "safeer":              0.059875,
    "shidda_or_rakhawa":   0.0605,
    "tafashie":            0.059875,
    "tafkheem_or_taqeeq":  0.0605,
    "tikraar":             0.059875,
}


def make_config(args=None) -> dict:
    p = argparse.ArgumentParser(prog="mualem_pipeline",
                                description="Mualem pipeline")
    p.add_argument("--name",        default="default",  help="Experiment name (sub-directory)")
    p.add_argument("--stages",      nargs="+",
                   default=["prune", "finetune", "qat"],
                   choices=["prune", "finetune", "qat"],
                   help="Stages to run (skipped if already done)")

    # Pruning
    p.add_argument("--hf_repo",     default="obadx/muaalem-model-v3_2")
    p.add_argument("--head_target", type=int,   default=12,   help="Attention heads after pruning")
    p.add_argument("--ffn_target",  type=int,   default=3072, help="FFN width after pruning")
    p.add_argument(
        "--layer_target", type=int, default=24,
        help="Encoder layers after pruning. Default 24 = no change. "
             "Wav2Vec2-BERT has 24 layers; recommended minimum is 18.")
    p.add_argument(
        "--hidden_target", type=int, default=1024,
        help="Hidden size after pruning. Default 1024 = no change. "
             "Must be divisible by --head_target and a multiple of 64. "
             "E.g. 768 works with 12 heads (head_dim=64).")
    p.add_argument(
        "--layer_score", default="cosine",
        choices=["cosine", "loss_delta"],
        help="Layer importance scoring. 'cosine' = single forward pass (default). "
             "'loss_delta' = more accurate but N x slower (not implemented).")

    # Fine-tuning
    p.add_argument("--ft_epochs",   type=int,   default=10)
    p.add_argument("--ft_lr",       type=float, default=1e-5)
    p.add_argument("--ft_batch",    type=int,   default=1)
    p.add_argument("--ft_accum",    type=int,   default=16)
    p.add_argument("--ft_samples",  type=int,   default=5000)
    p.add_argument("--ft_max_dur",  type=float, default=15.0)

    # LR schedule (shared by FT and QAT).
    # Stepped once per epoch: the pipeline streams a fixed sample budget per
    # epoch, so the per-epoch step count isn't known up front. Epoch-granularity
    # keeps the schedule resumable.
    p.add_argument(
        "--lr_schedule", default="cosine",
        choices=["constant", "cosine", "linear"],
        help="LR schedule over epochs. 'cosine' (default) decays from the base "
             "lr to --lr_min following a half-cosine curve; 'linear' decays "
             "linearly; 'constant' holds the base lr. All honour --lr_warmup_epochs.")
    p.add_argument(
        "--lr_warmup_epochs", type=int, default=1,
        help="Epochs to linearly warm the lr from 0 to the base lr before the "
             "main schedule begins. 0 = no warmup. Helps freshly-pruned models "
             "whose loss surface is rough at the start of recovery.")
    p.add_argument(
        "--lr_min", type=float, default=1e-7,
        help="Floor lr the cosine/linear schedule decays to at the final epoch. "
             "Ignored for --lr_schedule constant.")

    # QAT
    p.add_argument("--qat_epochs",  type=int,   default=5)
    p.add_argument("--qat_lr",      type=float, default=5e-6)
    p.add_argument("--qat_bits",    type=int,   default=8)
    p.add_argument("--qat_samples", type=int,   default=5000)
    p.add_argument("--qat_max_dur", type=float, default=10.0)

    p.add_argument(
        "--moshaf", nargs="+", default=ALL_MOSHAFS,
        help="Moshaf subsets to train on. Defaults to all 27. "
             "Example: --moshaf moshaf_0.0 moshaf_1.0")

    # KD
    p.add_argument(
        "--kd_alpha", type=float, default=0.0,
        help="Weight of KD distillation loss vs. task CTC loss. "
             "0 = disabled (default). Recommended: 0.3-0.5 when enabled.")
    p.add_argument(
        "--kd_pruning", action="store_true", default=False,
        help="Use KD-guided Taylor importance scoring when picking which "
             "attention heads and FFN neurons to prune. Adds one GPU "
             "forward+backward pass through the full model.")
    p.add_argument(
        "--kd_calib_batches", type=int, default=30,
        help="Single-sample calibration batches used to estimate KD-guided "
             "pruning importance. Only used when --kd_pruning.")
    p.add_argument(
        "--no_grad_ckpt", action="store_true", default=False,
        help="Disable gradient checkpointing on the encoder. Faster training "
             "but uses more VRAM. Safe on GPUs with >= 40 GB VRAM.")

    # W&B (off by default; works on Kaggle / Lightning.ai / local)
    p.add_argument(
        "--wandb", action="store_true", default=False,
        help="Enable W&B tracking. Requires WANDB_API_KEY (or cached creds).")
    p.add_argument("--wandb_project",  default="mualem-pipeline",
                   help="W&B project name.")
    p.add_argument("--wandb_entity",   default="",
                   help="W&B entity (team/user). Empty = personal default.")
    p.add_argument("--wandb_run_name", default="",
                   help="Run name override. Default: '{name}-{stage}'.")
    p.add_argument("--wandb_group",    default="",
                   help="W&B group name. Default: experiment --name.")
    p.add_argument("--wandb_ckpt", default="best",
                   choices=["none", "best", "all"],
                   help="Which checkpoints to upload to / restore from W&B. "
                        "Uploads the FULL resumable checkpoint (model + optimizer "
                        "+ AMP scaler + training_state + config/vocab) so training "
                        "can continue from W&B alone on a fresh machine. "
                        "'none' = no upload/restore; "
                        "'best' = upload on epoch improvement (+ best FT weights + INT8 export); "
                        "'all' = upload every epoch. "
                        "On resume, if no local checkpoint exists the latest W&B "
                        "checkpoint for the stage is downloaded automatically.")
    p.add_argument("--wandb_log_every", type=int, default=1,
                   help="Log per-step metrics every N optimizer steps.")

    cfg = vars(p.parse_args(args if args is not None else sys.argv[1:]))

    _validate(cfg)

    cfg["exp_dir"]      = ROOT / cfg["name"]
    cfg["model_dir"]    = cfg["exp_dir"] / "pruned_model"
    cfg["ft_ckpt"]      = cfg["exp_dir"] / "finetune_checkpoints"
    cfg["qat_ckpt"]     = cfg["exp_dir"] / "qat_checkpoints"
    cfg["export_dir"]   = cfg["exp_dir"] / "quantized_model"
    # KD teacher weights cache (persists across stages).
    cfg["teacher_dir"]  = cfg["exp_dir"] / "teacher_model"

    cfg["loss_weights"] = dict(LOSS_WEIGHTS)
    cfg["levels"]       = list(LOSS_WEIGHTS.keys())
    return cfg


def _validate(cfg: dict) -> None:
    # Hidden-size constraints only apply when the user actually shrinks the
    # hidden size -- the default `--head_target 12 --hidden_target 1024` combo
    # would otherwise fail the divisibility rule despite no slicing.
    if cfg["hidden_target"] < 1024:
        if cfg["hidden_target"] % cfg["head_target"] != 0:
            nearest = cfg["head_target"] * (cfg["hidden_target"] // cfg["head_target"])
            raise ValueError(
                f"--hidden_target {cfg['hidden_target']} must be divisible by "
                f"--head_target {cfg['head_target']}. Try {nearest}.")
        # head_dim is fixed at 64 for Wav2Vec2-BERT.
        if cfg["hidden_target"] % 64 != 0:
            raise ValueError(
                f"--hidden_target {cfg['hidden_target']} must be a multiple of 64.")
        # Smaller sizes break Conformer convolution kernels.
        if cfg["hidden_target"] < 256:
            raise ValueError(
                f"--hidden_target {cfg['hidden_target']} below hard floor of 256.")
    if cfg["layer_target"] < 6 or cfg["layer_target"] > 24:
        raise ValueError(
            f"--layer_target {cfg['layer_target']} must be between 6 and 24.")
