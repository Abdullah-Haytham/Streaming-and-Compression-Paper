"""
mualem_pipeline.py  (v3 — KD Edition)
======================================
Unified Pruning → Fine-Tuning → QAT pipeline for Wav2Vec2-BERT,
now with Knowledge-Distillation guided pruning and recovery,
based on 2510.04213 (Li et al., "Enhancing Speaker Verification with
w2v-BERT 2.0 and Knowledge Distillation guided Structured Pruning").

What's new vs v2
----------------
• KD-guided pruning scoring  (--kd_pruning flag)
    Instead of selecting which attention heads / FFN neurons to remove
    purely by weight-magnitude, we run a small calibration pass through
    the *frozen* original teacher and compute a first-order Taylor
    importance for every group of parameters:
        importance(w) ≈ |w · ∂L_KD/∂w|
    This gives each head / neuron a score that reflects how much
    removing it would disturb the distillation target, not just how
    small its weights happen to be.

• KD distillation loss in fine-tuning and QAT  (--kd_alpha > 0)
    During recovery fine-tuning (and QAT), a frozen copy of the original
    full model acts as a teacher.  At every training step the student's
    per-layer hidden states are aligned to the teacher's via the loss
    from eq. (5) of the paper:

        L_distill = Σ_l Σ_t [ L1(h_t^l, ĥ_t^l) − cosine(h_t^l, ĥ_t^l) ]

    This loss is mixed with the task CTC loss:
        L_total = L_CTC + kd_alpha · L_distill

Backward-compatible
-------------------
All new flags default to off, so existing scripts / notebooks that do
    python mualem_pipeline.py --head_target 10 --ffn_target 2048
work identically to before.

New flags
---------
--kd_alpha          float  (default 0.0)   Weight of KD loss in total loss.
                                            Recommended: 0.3 – 0.5.
                                            Set > 0 to enable teacher loading
                                            in fine-tuning and QAT stages.
--kd_pruning                (default off)  Use KD-guided Taylor importance
                                            for head/FFN selection in pruning.
                                            Requires extra GPU forward+backward
                                            pass through the full model.
--kd_calib_batches  int    (default 30)    Calibration batches for KD pruning
                                            importance estimation.

Usage examples
--------------
# Full KD pipeline (KD in pruning + fine-tune + QAT):
    python mualem_pipeline.py --kd_pruning --kd_alpha 0.4

# KD only during recovery (cheaper, skip KD-guided pruning):
    python mualem_pipeline.py --kd_alpha 0.4

# Legacy (no KD at all):
    python mualem_pipeline.py
"""

# ═══════════════════════════════════════════════════════════════════
# §0  Imports & helpers
# ═══════════════════════════════════════════════════════════════════
import argparse
import gc
import io
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

# ── Runtime detection ──────────────────────────────────────────────
IN_COLAB     = "google.colab" in sys.modules
IN_KAGGLE    = os.path.exists("/kaggle")
IN_LIGHTNING = os.environ.get("LIGHTNING_CLOUD_PROJECT_ID") is not None

if IN_COLAB:
    ROOT = Path("/content/drive/MyDrive/mualem_pipeline")
    print("🟢 Colab detected")
elif IN_KAGGLE:
    ROOT = Path("/kaggle/working/mualem_pipeline")
    print("🟢 Kaggle detected")
elif IN_LIGHTNING:
    ROOT = Path("/teamspace/studios/this_studio/mualem_pipeline")
    print("🟢 Lightning AI detected")
else:
    ROOT = Path("./mualem_pipeline")
    print("🟡 Local environment")


# ── Disk-space guard ───────────────────────────────────────────────
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
        raise RuntimeError(f"💥 Disk {pct:.1f}% full{tag} — aborting to prevent corruption")
    elif pct > warn_at:
        print(f"  ⚠️  Disk {pct:.1f}% full{tag}")
    else:
        print(f"  💾 Disk {pct:.1f}%{tag}")


# ── Stage-done markers ─────────────────────────────────────────────
def mark_done(exp_dir: Path, stage: str):
    (exp_dir / f".{stage}_done").touch()

def is_done(exp_dir: Path, stage: str) -> bool:
    return (exp_dir / f".{stage}_done").exists()


# ═══════════════════════════════════════════════════════════════════
# §1  Configuration
# ═══════════════════════════════════════════════════════════════════
def make_config(args=None) -> dict:
    p = argparse.ArgumentParser(description="Mualem unified pipeline")
    p.add_argument("--name",        default="default",  help="Experiment name (used as sub-directory)")
    p.add_argument("--stages",      nargs="+",
                   default=["prune", "finetune", "qat"],
                   choices=["prune", "finetune", "qat"],
                   help="Stages to run (skipped if already done)")

    # ── Pruning ────────────────────────────────────────────────────
    p.add_argument("--hf_repo",     default="obadx/muaalem-model-v3_2")
    p.add_argument("--head_target", type=int,   default=12,   help="Attention heads after pruning")
    p.add_argument("--ffn_target",  type=int,   default=3072, help="FFN width after pruning")
    p.add_argument(
        "--layer_target", type=int, default=24,
        help="Number of encoder layers after pruning. Default 24 = no change. "
             "Wav2Vec2-BERT has 24 layers; recommended minimum is 18.")
    p.add_argument(
        "--hidden_target", type=int, default=1024,
        help="Hidden size after pruning. Default 1024 = no change. "
             "Must be divisible by --head_target and a multiple of 64. "
             "E.g. 768 works with 12 heads (head_dim=64).")
    p.add_argument(
        "--layer_score", default="cosine",
        choices=["cosine", "loss_delta"],
        help="Scoring method for layer importance. "
             "'cosine' = fast, single forward pass (default). "
             "'loss_delta' = accurate but N× slower (not implemented in v4).")

    # ── Fine-tuning ────────────────────────────────────────────────
    p.add_argument("--ft_epochs",   type=int,   default=10)
    p.add_argument("--ft_lr",       type=float, default=1e-5)
    p.add_argument("--ft_batch",    type=int,   default=1)
    p.add_argument("--ft_accum",    type=int,   default=16)
    p.add_argument("--ft_samples",  type=int,   default=5000)
    p.add_argument("--ft_max_dur",  type=float, default=15.0)

    # ── Learning-rate schedule (shared by FT and QAT) ──────────────
    # The schedule is stepped once per epoch (this pipeline streams a
    # fixed sample budget per epoch, so the per-epoch step count isn't
    # known up front — epoch-granularity keeps it robust and resumable).
    p.add_argument(
        "--lr_schedule", default="cosine",
        choices=["constant", "cosine", "linear"],
        help="Learning-rate schedule over epochs. "
             "'cosine' (default) decays lr from the base value to "
             "--lr_min following a half-cosine curve; 'linear' decays "
             "linearly; 'constant' reproduces the old fixed-lr behaviour. "
             "All schedules honour --lr_warmup_epochs.")
    p.add_argument(
        "--lr_warmup_epochs", type=int, default=1,
        help="Number of epochs to linearly warm the lr up from 0 to the "
             "base lr before the main schedule begins. 0 = no warmup. "
             "Helpful for freshly-pruned models whose loss surface is "
             "rough at the start of recovery.")
    p.add_argument(
        "--lr_min", type=float, default=1e-7,
        help="Floor learning rate the cosine/linear schedule decays to "
             "at the final epoch. Ignored for --lr_schedule constant.")

    # ── QAT ────────────────────────────────────────────────────────
    p.add_argument("--qat_epochs",  type=int,   default=5)
    p.add_argument("--qat_lr",      type=float, default=5e-6)
    p.add_argument("--qat_bits",    type=int,   default=8)
    p.add_argument("--qat_samples", type=int,   default=5000)
    p.add_argument("--qat_max_dur", type=float, default=10.0)
    ALL_MOSHAFS = [
        "moshaf_0.0",  "moshaf_0.1",  "moshaf_0.2",  "moshaf_0.3",
        "moshaf_1.0",  "moshaf_2.0",  "moshaf_2.1",  "moshaf_3.0",
        "moshaf_4.0",  "moshaf_5.0",  "moshaf_6.0",  "moshaf_7.0",
        "moshaf_8.0",  "moshaf_9.0",  "moshaf_11.0", "moshaf_12.0",
        "moshaf_13.0", "moshaf_19.0", "moshaf_22.0", "moshaf_24.0",
        "moshaf_25.0", "moshaf_26.0", "moshaf_26.1", "moshaf_27.0",
        "moshaf_28.0", "moshaf_29.0", "moshaf_30.0",
    ]
    p.add_argument(
        "--moshaf", nargs="+", default=ALL_MOSHAFS,
        help="One or more moshaf subset names to train on. "
             "Defaults to all 27 moshafs. "
             "Example: --moshaf moshaf_0.0 moshaf_1.0")

    # ── KD: new flags ──────────────────────────────────────────────
    p.add_argument(
        "--kd_alpha", type=float, default=0.0,
        help="Weight of the KD distillation loss vs. task CTC loss. "
             "0 = disabled (default).  Recommended: 0.3–0.5 when enabled.")
    p.add_argument(
        "--kd_pruning", action="store_true", default=False,
        help="Use KD-guided Taylor importance scoring when deciding which "
             "attention heads and FFN neurons to prune.  Requires one extra "
             "GPU forward+backward pass through the full model.")
    p.add_argument(
        "--kd_calib_batches", type=int, default=30,
        help="Number of single-sample calibration batches used to estimate "
             "KD-guided pruning importance scores.  Only used when --kd_pruning.")
    p.add_argument(
        "--no_grad_ckpt", action="store_true", default=False,
        help="Disable gradient checkpointing on the encoder. "
             "Faster training but uses more VRAM. "
             "Safe to enable on GPUs with >= 40 GB VRAM.")

    # ── Weights & Biases (off by default; works on Kaggle / Lightning.ai / local) ─
    p.add_argument(
        "--wandb", action="store_true", default=False,
        help="Enable Weights & Biases experiment tracking. Requires "
             "WANDB_API_KEY env var (or wandb credentials cached on disk).")
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
                   help="Which checkpoints to upload to / restore from W&B as "
                        "artifacts. Uploads the FULL resumable checkpoint "
                        "(model + optimizer + AMP scaler + training_state + "
                        "config/vocab) so training can continue from W&B alone "
                        "on a fresh machine. 'none' = no upload/restore; "
                        "'best' = upload the complete checkpoint whenever the "
                        "epoch improves (+ best FT weights + INT8 export); "
                        "'all' = upload the complete checkpoint every epoch. "
                        "On resume, if no local checkpoint exists the latest "
                        "W&B checkpoint for the stage is downloaded "
                        "automatically.")
    p.add_argument("--wandb_log_every", type=int, default=1,
                   help="Log per-step metrics every N optimizer steps.")

    cfg = vars(p.parse_args(args if args is not None else sys.argv[1:]))

    # ── Validation for new layer / hidden-size flags ─────────────────
    # Only enforce hidden-size constraints when the user is actually pruning
    # hidden_size (i.e. hidden_target < 1024).  Otherwise the default combo
    # `--head_target 12 --hidden_target 1024` would fail the divisibility
    # rule even though no hidden slicing happens.
    if cfg["hidden_target"] < 1024:
        if cfg["hidden_target"] % cfg["head_target"] != 0:
            nearest = cfg["head_target"] * (cfg["hidden_target"] // cfg["head_target"])
            raise ValueError(
                f"--hidden_target {cfg['hidden_target']} must be divisible by "
                f"--head_target {cfg['head_target']} so that head_dim remains an integer. "
                f"Try {nearest}.")
        if cfg["hidden_target"] % 64 != 0:
            raise ValueError(
                f"--hidden_target {cfg['hidden_target']} must be a multiple of 64 "
                f"(HEAD_DIM is fixed at 64 for Wav2Vec2-BERT).")
        if cfg["hidden_target"] < 256:
            raise ValueError(
                f"--hidden_target {cfg['hidden_target']} is below the hard floor of 256 "
                f"(smaller sizes break Conformer convolution kernels).")
    if cfg["layer_target"] < 6 or cfg["layer_target"] > 24:
        raise ValueError(
            f"--layer_target {cfg['layer_target']} must be between 6 and 24.")

    cfg["exp_dir"]      = ROOT / cfg["name"]
    cfg["model_dir"]    = cfg["exp_dir"] / "pruned_model"
    cfg["ft_ckpt"]      = cfg["exp_dir"] / "finetune_checkpoints"
    cfg["qat_ckpt"]     = cfg["exp_dir"] / "qat_checkpoints"
    cfg["export_dir"]   = cfg["exp_dir"] / "quantized_model"
    # ── KD: teacher weights cache (persists across stages) ─────────
    cfg["teacher_dir"]  = cfg["exp_dir"] / "teacher_model"

    # Shared loss weights (thesis Eq 5.1)
    cfg["loss_weights"] = {
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
    cfg["levels"] = list(cfg["loss_weights"].keys())
    return cfg


# ═══════════════════════════════════════════════════════════════════
# §1.5  Weights & Biases logger (self-contained, no-op when disabled)
# ═══════════════════════════════════════════════════════════════════
def _wandb_detect_env() -> str:
    import os as _os
    if "KAGGLE_KERNEL_RUN_TYPE" in _os.environ or _os.path.exists("/kaggle"):
        return "kaggle"
    if (_os.environ.get("LIGHTNING_CLOUD_PROJECT_ID")
            or "LIGHTNING_CLOUD_URL" in _os.environ
            or "LIGHTNING_APP_STATE" in _os.environ):
        return "lightning"
    if "COLAB_GPU" in _os.environ or "google.colab" in sys.modules:
        return "colab"
    return "local"


def _wandb_derive_tags(cfg: dict, stage: str, env: str) -> list:
    tags = [stage, env]
    if cfg.get("kd_alpha", 0) > 0 or cfg.get("kd_pruning"):
        tags.append("kd")
        tags.append(f"alpha={cfg['kd_alpha']:.2f}")
    else:
        tags.append("no-kd")
    tags.append(
        f"h{cfg['head_target']}-ffn{cfg['ffn_target']}-"
        f"l{cfg['layer_target']}-d{cfg['hidden_target']}"
    )
    return tags


class WandbLogger:
    """Thin wrapper around wandb. No-ops when --wandb is off or init fails.

    One instance per pipeline stage ('prune' | 'ft' | 'qat').  The wandb import
    is lazy and only happens on .start() so the import cost is paid once per
    stage and only when wandb is actually enabled.
    """

    def __init__(self, cfg: dict, stage: str):
        self.cfg = cfg
        self.stage = stage
        self.run = None
        self.enabled = bool(cfg.get("wandb"))
        self._step = 0
        self._log_every = max(1, int(cfg.get("wandb_log_every", 1)))

    def start(self, extra_config=None):
        if not self.enabled:
            return
        try:
            import os as _os
            import wandb
        except ImportError:
            print("[wandb] wandb not installed; disabling. "
                  "Install with: pip install wandb")
            self.enabled = False
            return
        try:
            key = _os.environ.get("WANDB_API_KEY")
            if key:
                wandb.login(key=key, relogin=False)
            else:
                print("[wandb] WANDB_API_KEY not set; falling back to "
                      "cached credentials if available.")
            env = _wandb_detect_env()
            tags = _wandb_derive_tags(self.cfg, self.stage, env)
            run_name = (self.cfg.get("wandb_run_name")
                        or f"{self.cfg['name']}-{self.stage}")
            group = self.cfg.get("wandb_group") or self.cfg["name"]
            # Build a JSON-safe config snapshot for wandb
            safe_cfg = {
                k: (str(v) if isinstance(v, Path) else v)
                for k, v in self.cfg.items()
                if k not in ("loss_weights", "levels")
            }
            safe_cfg["env"] = env
            if extra_config:
                safe_cfg.update(extra_config)
            self.run = wandb.init(
                project=self.cfg.get("wandb_project") or "mualem-pipeline",
                entity=self.cfg.get("wandb_entity") or None,
                name=run_name,
                group=group,
                job_type=self.stage,
                tags=tags,
                config=safe_cfg,
                reinit=True,
            )
            print(f"[wandb] tracking enabled: project="
                  f"{self.cfg.get('wandb_project')}  run={run_name}  "
                  f"group={group}  env={env}")
        except Exception as e:
            print(f"[wandb] init failed ({e}); disabling.")
            self.run = None
            self.enabled = False

    @property
    def active(self) -> bool:
        return self.run is not None

    def log_step(self, metrics: dict):
        if not self.active:
            return
        if self._step % self._log_every != 0:
            self._step += 1
            return
        try:
            import wandb
            wandb.log(metrics, step=self._step)
        except Exception as e:
            print(f"[wandb] log_step failed: {e}")
        self._step += 1

    def log_epoch(self, metrics: dict):
        if not self.active:
            return
        try:
            import wandb
            wandb.log({f"epoch/{k}": v for k, v in metrics.items()})
        except Exception as e:
            print(f"[wandb] log_epoch failed: {e}")

    def log_histogram(self, name: str, values):
        if not self.active:
            return
        try:
            import wandb
            import numpy as np
            arr = np.asarray(values).ravel()
            if arr.size == 0:
                return
            wandb.log({name: wandb.Histogram(arr.astype(float))})
        except Exception as e:
            print(f"[wandb] log_histogram({name}) failed: {e}")

    def log_summary(self, d: dict):
        if not self.active:
            return
        try:
            for k, v in d.items():
                self.run.summary[k] = v
        except Exception as e:
            print(f"[wandb] log_summary failed: {e}")

    def log_artifact(self, path, name: str, art_type: str,
                     metadata: dict = None, gate: bool = True):
        if not self.active:
            return
        if gate and self.cfg.get("wandb_ckpt", "best") == "none":
            return
        try:
            import wandb
            art = wandb.Artifact(name, type=art_type,
                                 metadata=metadata or {})
            art.add_file(str(path))
            self.run.log_artifact(art)
        except Exception as e:
            print(f"[wandb] log_artifact({name}) failed: {e}")

    # ── Full resumable-checkpoint artifacts ────────────────────────────
    # The methods below upload / download an *entire* checkpoint directory
    # (model + optimizer + AMP scaler + training_state.json + config/vocab)
    # so a run can be resumed from W&B alone on a fresh machine.  The set
    # of files mirrors exactly what save_checkpoint() writes and what
    # load_checkpoint() reads.
    CHECKPOINT_FILES = (
        "model_latest.safetensors",
        "model_best.safetensors",
        "optimizer_latest.pt",
        "scaler_latest.pt",
        "training_state.json",
        # Static side-cars handy for a clean-machine resume / hand-off.
        "config.json",
        "vocab.json",
        "preprocessor_config.json",
    )

    def _ckpt_artifact_name(self) -> str:
        """Stable per-stage artifact name; new versions accumulate under it."""
        return f"{self.cfg['name']}-{self.stage}-ckpt"

    def log_checkpoint(self, ckpt_dir, epoch: int, metadata: dict = None):
        """Upload the *complete* checkpoint dir as one resumable artifact.

        Gated by --wandb_ckpt: uploaded when 'all' (every epoch) or 'best'
        (only logged when called for the best epoch — callers pass the gate
        decision via the ``--wandb_ckpt`` value, see run_finetune/run_qat).
        Each call creates a new *version* of the same artifact name, and the
        latest version is the one restored on resume.
        """
        if not self.active:
            return
        if self.cfg.get("wandb_ckpt", "best") == "none":
            return
        from pathlib import Path as _Path
        ckpt_dir = _Path(ckpt_dir)
        try:
            import wandb
            art = wandb.Artifact(
                self._ckpt_artifact_name(),
                type="training-checkpoint",
                metadata={**(metadata or {}), "epoch": int(epoch),
                          "stage": self.stage},
            )
            added = []
            for fname in self.CHECKPOINT_FILES:
                fpath = ckpt_dir / fname
                if fpath.exists():
                    art.add_file(str(fpath), name=fname)
                    added.append(fname)
            if not added:
                print("[wandb] log_checkpoint: nothing to upload (no files)")
                return
            # Alias the freshest checkpoint as 'latest' so resume can grab it.
            self.run.log_artifact(art, aliases=["latest", f"epoch-{epoch}"])
            print(f"  ☁️  Checkpoint uploaded to W&B "
                  f"({len(added)} files, epoch {epoch})")
        except Exception as e:
            print(f"[wandb] log_checkpoint failed: {e}")

    def restore_checkpoint(self, ckpt_dir) -> bool:
        """Download the latest checkpoint artifact for this stage into
        ``ckpt_dir`` *iff* there is no usable local checkpoint already.

        Returns True if files were pulled from W&B, False otherwise.  Safe to
        call even when W&B is disabled (returns False).  Never overwrites an
        existing local training_state.json — local checkpoints win, so an
        interrupted-and-restarted run on the same machine resumes from disk.
        """
        if not self.active:
            return False
        if self.cfg.get("wandb_ckpt", "best") == "none":
            return False
        from pathlib import Path as _Path
        ckpt_dir = _Path(ckpt_dir)
        # Local checkpoint present → prefer it, don't touch W&B.
        if (ckpt_dir / "training_state.json").exists():
            return False
        try:
            import wandb
            ref = (f"{self._ckpt_artifact_name()}:latest")
            entity = self.cfg.get("wandb_entity") or self.run.entity
            project = self.cfg.get("wandb_project") or "mualem-pipeline"
            qualified = f"{entity}/{project}/{ref}" if entity else f"{project}/{ref}"
            print(f"  ☁️  No local checkpoint — trying W&B artifact {qualified}")
            try:
                art = self.run.use_artifact(qualified, type="training-checkpoint")
            except Exception:
                # Fall back to unqualified name (same project/entity context).
                art = self.run.use_artifact(ref, type="training-checkpoint")
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            art.download(root=str(ckpt_dir))
            restored = (ckpt_dir / "training_state.json").exists()
            if restored:
                meta = art.metadata or {}
                print(f"  ♻️  Restored checkpoint from W&B "
                      f"(epoch {meta.get('epoch', '?')})")
            return restored
        except Exception as e:
            print(f"[wandb] restore_checkpoint: nothing to restore ({e})")
            return False

    def finish(self, status: str = "success"):
        if not self.active:
            return
        try:
            import wandb
            self.run.summary["exit_status"] = status
            wandb.finish()
        except Exception as e:
            print(f"[wandb] finish failed: {e}")
        finally:
            self.run = None


def _gpu_mem_mb():
    """Return current GPU memory allocated in MB, or 0.0 if unavailable."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1e6
    except Exception:
        pass
    return 0.0


def _gpu_util_pct():
    """Return current GPU utilisation %, or None if pynvml unavailable."""
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        u = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
        pynvml.nvmlShutdown()
        return float(u)
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════
# §2  STAGE 1 — Structured Pruning  (optionally KD-guided)
# ═══════════════════════════════════════════════════════════════════
def run_pruning(cfg: dict):
    print("\n" + "═"*60)
    print("  ✂️  STAGE 1: Structured Pruning")
    if cfg["kd_pruning"]:
        print("  🎓  KD-guided importance scoring ENABLED")
    print("═"*60)

    import numpy as np
    from safetensors.numpy import load_file, save_file
    from huggingface_hub import hf_hub_download

    wandb_logger = WandbLogger(cfg, "prune")
    wandb_logger.start()

    exp_dir   = cfg["exp_dir"]
    model_dir = cfg["model_dir"]
    tmp_dir   = exp_dir / "_original_download"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    HEAD_TARGET = cfg["head_target"]
    FFN_TARGET  = cfg["ffn_target"]
    HF_REPO     = cfg["hf_repo"]

    FILES = [
        "config.json", "model.safetensors", "vocab.json",
        "preprocessor_config.json", "added_tokens.json",
        "special_tokens_map.json",  "tokenizer_config.json",
    ]

    # ── Download ──────────────────────────────────────────────────
    print(f"\n📥 Downloading from {HF_REPO}...")
    for fname in FILES:
        dest = tmp_dir / fname
        if dest.exists():
            print(f"  ✅ {fname} (cached)")
            continue
        disk_check(f"before downloading {fname}")
        hf_hub_download(repo_id=HF_REPO, filename=fname, local_dir=str(tmp_dir))
        size = (tmp_dir / fname).stat().st_size
        print(f"  ⬇️  {fname} → {size/1e6:.1f} MB")

    # ── KD: cache teacher weights now (same download, no extra cost) ─
    if cfg["kd_alpha"] > 0 or cfg["kd_pruning"]:
        teacher_dir = cfg["teacher_dir"]
        if not (teacher_dir / "model.safetensors").exists():
            print("\n📋 Caching teacher weights for later KD stages...")
            teacher_dir.mkdir(parents=True, exist_ok=True)
            for fname in FILES:
                src = tmp_dir / fname
                if src.exists():
                    shutil.copy2(str(src), str(teacher_dir / fname))
            print(f"  ✅ Teacher cached at {teacher_dir}")
        else:
            print("  ✅ Teacher already cached — skipping copy")

    # ── Load weights ──────────────────────────────────────────────
    print("\n📦 Loading weights...")
    disk_check("before loading")
    t0 = time.time()
    state = load_file(str(tmp_dir / "model.safetensors"))
    orig_params = sum(v.size for v in state.values())
    print(f"  Loaded {len(state)} tensors / {orig_params/1e6:.1f}M params in {time.time()-t0:.1f}s")

    # ── KD: compute gradient-based importance before pruning ───────
    kd_head_scores = None
    kd_ffn_scores  = None
    if cfg["kd_pruning"]:
        kd_head_scores, kd_ffn_scores = _compute_kd_pruning_scores(
            cfg, tmp_dir)
        if kd_head_scores is None:
            print("  ⚠️  KD pruning score computation failed — using magnitude scoring")
        else:
            # Log KD-guided Taylor importance histograms (per-head, per-FFN)
            try:
                all_head = np.concatenate(
                    [np.asarray(v).ravel() for v in kd_head_scores.values()])
                wandb_logger.log_histogram("prune/kd_head_scores", all_head)
            except Exception:
                pass
            try:
                all_ffn = np.concatenate(
                    [np.asarray(v).ravel() for v in kd_ffn_scores.values()])
                wandb_logger.log_histogram("prune/kd_ffn_scores", all_ffn)
            except Exception:
                pass
            wandb_logger.log_summary({
                "prune/kd_head_score_layers": len(kd_head_scores),
                "prune/kd_ffn_score_layers":  len(kd_ffn_scores),
            })

    # ── Pruning helpers ───────────────────────────────────────────
    def _layer_prefixes(keys):
        seen, out = set(), []
        for k in keys:
            if k.startswith("wav2vec2_bert.encoder.layers."):
                p = ".".join(k.split(".")[:4])
                if p not in seen:
                    seen.add(p); out.append(p)
        return sorted(out, key=lambda x: int(x.rsplit(".", 1)[-1]))

    def prune_heads(state, target):
        prefixes = _layer_prefixes(state.keys())
        q0 = state[f"{prefixes[0]}.self_attn.linear_q.weight"]
        old_heads, head_dim = 16, q0.shape[0] // 16
        if target >= old_heads:
            return state
        print(f"  Pruning heads: {old_heads} → {target}  (head_dim={head_dim})")

        using_kd = (kd_head_scores is not None)
        print(f"  Importance scorer: {'KD-guided Taylor' if using_kd else 'weight magnitude'}")

        for layer in prefixes:
            if using_kd and layer in kd_head_scores:
                # ── KD: use Taylor importance per head ────────────
                kd_s = kd_head_scores[layer]          # shape (old_heads,)
                scores = sorted(
                    [(h, float(kd_s[h])) for h in range(old_heads)],
                    key=lambda x: x[1]
                )
            else:
                # ── fallback: weight magnitude ────────────────────
                scores = sorted(
                    [(h, float(
                        np.linalg.norm(state[f"{layer}.self_attn.linear_q.weight"][h*head_dim:(h+1)*head_dim]) +
                        np.linalg.norm(state[f"{layer}.self_attn.linear_k.weight"][h*head_dim:(h+1)*head_dim]) +
                        np.linalg.norm(state[f"{layer}.self_attn.linear_v.weight"][h*head_dim:(h+1)*head_dim])
                    )) for h in range(old_heads)],
                    key=lambda x: x[1]
                )

            keep = sorted(h for h, _ in scores[(old_heads - target):])
            rows = [i for h in keep for i in range(h*head_dim, (h+1)*head_dim)]
            for base in ("linear_q", "linear_k", "linear_v"):
                w = f"{layer}.self_attn.{base}.weight"
                b = f"{layer}.self_attn.{base}.bias"
                state[w] = state[w][rows, :]
                if b in state: state[b] = state[b][rows]
            state[f"{layer}.self_attn.linear_out.weight"] = state[f"{layer}.self_attn.linear_out.weight"][:, rows]
        return state

    def _find_ffn_keys(state, layer_prefix):
        """
        Auto-discover the FFN intermediate+output weight key names for a given
        layer prefix.  Different HuggingFace checkpoints use different naming
        conventions, e.g.:
          - wav2vec2_bert: feed_forward.intermediate_dense / feed_forward.output_dense
          - some variants:         ffn.intermediate_dense /         ffn.output_dense
          - others:       feed_forward.intermediate /       feed_forward.output
        We search for a weight whose key starts with the layer prefix and whose
        shape looks like an 'up-projection' (rows > hidden_size, i.e. the
        intermediate dimension).
        """
        candidates_w1, candidates_w2 = [], []
        for k in state.keys():
            if not k.startswith(layer_prefix + "."):
                continue
            # Intermediate (up-projection): large first dimension
            if any(tag in k for tag in ("intermediate_dense", "intermediate.weight",
                                        "fc1", "dense_act")):
                if k.endswith(".weight"):
                    candidates_w1.append(k)
            # Output (down-projection): large second dimension
            if any(tag in k for tag in ("output_dense", "output.weight",
                                        "fc2", "dense.")):
                if k.endswith(".weight"):
                    candidates_w2.append(k)

        # Fallback: find by shape heuristic — intermediate has more rows than cols
        if not candidates_w1 or not candidates_w2:
            layer_keys = [k for k in state.keys()
                          if k.startswith(layer_prefix + ".") and k.endswith(".weight")
                          and "self_attn" not in k and "layer_norm" not in k
                          and "lm_head" not in k and "ctc" not in k]
            up, down = [], []
            for k in layer_keys:
                w = state[k]
                if w.ndim == 2:
                    if w.shape[0] > w.shape[1]:   # rows > cols → up-projection
                        up.append(k)
                    elif w.shape[1] > w.shape[0]:  # cols > rows → down-projection
                        down.append(k)
            if up and down:
                candidates_w1 = candidates_w1 or up
                candidates_w2 = candidates_w2 or down

        if not candidates_w1 or not candidates_w2:
            raise KeyError(
                f"Could not find FFN weight keys under '{layer_prefix}'.\n"
                f"Keys present: {[k for k in state if k.startswith(layer_prefix+'.')]}"
            )

        # If multiple candidates, prefer the one with the largest intermediate dim
        w1_key = max(candidates_w1, key=lambda k: state[k].shape[0])
        w2_key = max(candidates_w2, key=lambda k: state[k].shape[1])
        return w1_key, w2_key

    def prune_ffn(state, target):
        prefixes = _layer_prefixes(state.keys())

        # Discover key names from the first layer
        w1_key_0, w2_key_0 = _find_ffn_keys(state, prefixes[0])
        cur = state[w1_key_0].shape[0]

        if target >= cur:
            return state

        # Derive the sub-paths
        p0 = prefixes[0] + "."
        w1_subpath = w1_key_0[len(p0):]
        w2_subpath = w2_key_0[len(p0):]
        b1_subpath = w1_subpath.replace(".weight", ".bias")

        print(f"  Pruning FFN: {cur} → {target}")
        print(f"    up-key:   {w1_subpath}")
        print(f"    down-key: {w2_subpath}")

        using_kd = (kd_ffn_scores is not None)
        print(f"  Importance scorer: {'KD-guided Taylor' if using_kd else 'weight magnitude'}")

        for layer in prefixes:
            w1 = state[f"{layer}.{w1_subpath}"]
            w2 = state[f"{layer}.{w2_subpath}"]

            if using_kd and layer in kd_ffn_scores:
                # ── KD: use Taylor importance per neuron ──────────
                scores = kd_ffn_scores[layer]          # shape (cur_ffn_dim,)
                # Pad/trim to match actual w1 shape if needed
                if len(scores) != w1.shape[0]:
                    scores = np.linalg.norm(w1, axis=1) * np.linalg.norm(w2, axis=0)
            else:
                # ── fallback: weight magnitude ────────────────────
                scores = np.linalg.norm(w1, axis=1) * np.linalg.norm(w2, axis=0)

            keep = np.sort(np.argsort(scores)[-target:])
            state[f"{layer}.{w1_subpath}"] = w1[keep, :]
            b1_key = f"{layer}.{b1_subpath}"
            if b1_key in state:
                state[b1_key] = state[b1_key][keep]
            state[f"{layer}.{w2_subpath}"] = w2[:, keep]
        return state

    # ── Magnitude head/FFN score histograms (skipped when KD-guided already logged) ─
    if kd_head_scores is None:
        try:
            prefixes_pre = sorted(
                {".".join(k.split(".")[:4]) for k in state.keys()
                 if k.startswith("wav2vec2_bert.encoder.layers.")},
                key=lambda x: int(x.rsplit(".", 1)[-1]),
            )
            if prefixes_pre:
                q0 = state[f"{prefixes_pre[0]}.self_attn.linear_q.weight"]
                head_dim = q0.shape[0] // 16
                head_scores_mag = []
                for layer in prefixes_pre:
                    for h in range(16):
                        s = (
                            np.linalg.norm(state[f"{layer}.self_attn.linear_q.weight"][h*head_dim:(h+1)*head_dim])
                            + np.linalg.norm(state[f"{layer}.self_attn.linear_k.weight"][h*head_dim:(h+1)*head_dim])
                            + np.linalg.norm(state[f"{layer}.self_attn.linear_v.weight"][h*head_dim:(h+1)*head_dim])
                        )
                        head_scores_mag.append(float(s))
                wandb_logger.log_histogram("prune/head_scores_magnitude", head_scores_mag)
        except Exception:
            pass

    # ── Execute pruning ───────────────────────────────────────────
    t0 = time.time()
    state = prune_heads(state, HEAD_TARGET)
    state = prune_ffn(state, FFN_TARGET)

    # ── Layer (depth) pruning ─────────────────────────────────────
    LAYER_TARGET  = cfg["layer_target"]
    HIDDEN_TARGET = cfg["hidden_target"]
    if LAYER_TARGET < 24:
        layer_scores = _score_layers(cfg, tmp_dir, state, cfg["layer_score"])
        try:
            wandb_logger.log_histogram(
                "prune/layer_cosine_scores",
                np.asarray(list(layer_scores.values()), dtype=float),
            )
            wandb_logger.log_summary({
                f"prune/layer_cosine_{i:02d}": float(s)
                for i, s in sorted(layer_scores.items())
            })
        except Exception:
            pass
        state = prune_layers(state, LAYER_TARGET, layer_scores)

    # ── Hidden-size (width) pruning ───────────────────────────────
    if HIDDEN_TARGET < 1024:
        hidden_scores = _score_hidden_dims(cfg, tmp_dir, state)
        try:
            wandb_logger.log_histogram(
                "prune/hidden_dim_scores",
                np.asarray(hidden_scores, dtype=float),
            )
        except Exception:
            pass
        state = prune_hidden_size(state, HIDDEN_TARGET, hidden_scores)
        _validate_hidden_size(state, HIDDEN_TARGET)

    pruned_params = sum(v.size for v in state.values())
    reduction = (1 - pruned_params / orig_params) * 100
    print(f"\n📊 {orig_params/1e6:.1f}M → {pruned_params/1e6:.1f}M params  ({reduction:.1f}% reduction)  [{time.time()-t0:.1f}s]")

    # ── Update config & save ──────────────────────────────────────
    with open(tmp_dir / "config.json") as f:
        cfg_json = json.load(f)
    orig_hidden = int(cfg_json.get("hidden_size", 1024))
    cfg_json["num_attention_heads"] = HEAD_TARGET
    cfg_json["intermediate_size"]   = FFN_TARGET
    cfg_json["num_hidden_layers"]   = LAYER_TARGET
    cfg_json["hidden_size"]         = HIDDEN_TARGET
    # The Wav2Vec2-BERT adapter constructs its modules from `output_hidden_size`,
    # not `hidden_size`.  When the two were tied originally (the HF default),
    # the adapter's internal dim must follow hidden_size when we shrink it,
    # otherwise the freshly-built adapter LayerNorms / Linears expect 1024
    # but receive 256 — producing "Expected weight to be of same shape as
    # normalized_shape" errors at finetune time.
    if cfg_json.get("output_hidden_size", orig_hidden) == orig_hidden:
        cfg_json["output_hidden_size"] = HIDDEN_TARGET
    (model_dir / "config.json").write_text(json.dumps(cfg_json, indent=2, ensure_ascii=False))

    disk_check("before saving pruned weights")
    print("\n💾 Saving pruned model...")
    save_file(state, str(model_dir / "model.safetensors"), metadata={"pruned": "true"})

    for fname in ["vocab.json", "preprocessor_config.json", "added_tokens.json",
                  "special_tokens_map.json", "tokenizer_config.json"]:
        src = tmp_dir / fname
        if src.exists():
            shutil.copy2(str(src), str(model_dir / fname))

    # Save pruning summary
    (model_dir / "pruning_summary.json").write_text(json.dumps({
        "original_params":  int(orig_params),
        "pruned_params":    int(pruned_params),
        "reduction_pct":    round(reduction, 2),
        "head_target":      HEAD_TARGET,
        "ffn_target":       FFN_TARGET,
        "layer_target":     LAYER_TARGET,
        "hidden_target":    HIDDEN_TARGET,
        "kd_guided":        cfg["kd_pruning"],
    }, indent=2))

    size_gb = (model_dir / "model.safetensors").stat().st_size / 1e9
    print(f"  ✅ model.safetensors saved ({size_gb:.2f} GB)")

    # ── W&B summary + optional artifact upload ────────────────────
    wandb_logger.log_summary({
        "prune/params_before":   int(orig_params),
        "prune/params_after":    int(pruned_params),
        "prune/reduction_pct":   round(reduction, 2),
        "prune/compression_ratio": (orig_params / max(pruned_params, 1)),
        "prune/size_gb_after":   round(size_gb, 4),
        "prune/head_target":     HEAD_TARGET,
        "prune/ffn_target":      FFN_TARGET,
        "prune/layer_target":    LAYER_TARGET,
        "prune/hidden_target":   HIDDEN_TARGET,
        "prune/kd_guided":       bool(cfg["kd_pruning"]),
    })
    if cfg.get("wandb_ckpt") == "all":
        wandb_logger.log_artifact(
            model_dir / "model.safetensors",
            name=f"{cfg['name']}-pruned",
            art_type="model",
            metadata={
                "stage": "prune",
                "params": int(pruned_params),
                "size_gb": round(size_gb, 4),
            },
        )

    # ── Cleanup original download to free space ───────────────────
    print("\n🧹 Removing original download to free disk...")
    shutil.rmtree(str(tmp_dir))
    del state; gc.collect()
    disk_check("after cleanup")

    mark_done(exp_dir, "prune")
    wandb_logger.finish("success")
    print("\n✅ Pruning complete")


# ═══════════════════════════════════════════════════════════════════
# §2.4  Layer-pruning helpers  (Gromov 2403.17887 — cosine similarity)
# ═══════════════════════════════════════════════════════════════════

_LAYER_KEY_PREFIX  = "wav2vec2_bert.encoder.layers."
_LAYER_KEY_IDX_POS = 3  # index in key.split(".") where the layer number lives


def _layer_prefixes_from_state(keys):
    """Discover sorted encoder-layer prefixes from a state-dict's keys."""
    seen, out = set(), []
    for k in keys:
        if k.startswith(_LAYER_KEY_PREFIX):
            p = ".".join(k.split(".")[:4])
            if p not in seen:
                seen.add(p); out.append(p)
    return sorted(out, key=lambda x: int(x.rsplit(".", 1)[-1]))


def _write_calib_artifacts(state, tmp_dir: Path, cfg: dict):
    """
    Write the *current* (partially-pruned) state to a temporary safetensors
    file plus a synthesized config.json that matches the actual tensor
    shapes.  Returns (cfg_path, weights_path).
    """
    import numpy as np
    from safetensors.numpy import save_file

    weights_path = tmp_dir / "_calib_model.safetensors"
    cfg_path     = tmp_dir / "_calib_config.json"

    # Derive current architecture from state shapes (no assumptions)
    prefixes = _layer_prefixes_from_state(state.keys())
    n_layers = len(prefixes)
    # First layer's linear_q to get hidden size and head count
    q0 = state[f"{prefixes[0]}.self_attn.linear_q.weight"]
    cur_qkv_rows = q0.shape[0]                    # heads * head_dim
    cur_hidden   = q0.shape[1]                    # cols of Q = hidden
    HEAD_DIM     = 64
    cur_heads    = cur_qkv_rows // HEAD_DIM
    # FFN width — find a feed-forward intermediate weight
    ffn_w_keys = [k for k in state
                  if k.startswith(prefixes[0] + ".")
                  and k.endswith(".weight")
                  and any(t in k for t in ("intermediate_dense",
                                            "feed_forward.intermediate",
                                            "fc1"))]
    cur_ffn = state[ffn_w_keys[0]].shape[0] if ffn_w_keys else None

    with open(tmp_dir / "config.json", encoding="utf-8") as f:
        cfg_json = json.load(f)
    cfg_json["num_attention_heads"] = int(cur_heads)
    if cur_ffn is not None:
        cfg_json["intermediate_size"] = int(cur_ffn)
    cfg_json["num_hidden_layers"]   = int(n_layers)
    cfg_json["hidden_size"]         = int(cur_hidden)
    cfg_path.write_text(json.dumps(cfg_json, indent=2, ensure_ascii=False))

    save_file(state, str(weights_path), metadata={"calib": "true"})
    return cfg_path, weights_path


def _score_layers(cfg: dict, tmp_dir: Path, state, method: str):
    """
    Compute per-layer importance scores.  Lower score = more prunable.

    For ``method == "cosine"``: returns ``{layer_idx: cosine}`` where higher
    cosine means the layer barely changes its hidden states (Gromov 2024).
    Layers with the highest cosine are removed first.

    ``method == "loss_delta"`` is reserved for a future PR and raises
    ``NotImplementedError``.
    """
    if method == "loss_delta":
        raise NotImplementedError(
            "--layer_score loss_delta is not implemented yet; use 'cosine'.")
    if method != "cosine":
        raise ValueError(f"Unknown layer_score method: {method!r}")

    print("\n📐 Scoring encoder layers by mean cosine(input, output)...")
    try:
        import torch
        import torch.nn.functional as F
        import numpy as np
        from transformers import SeamlessM4TFeatureExtractor
    except ImportError as e:
        print(f"  ⚠️  Missing dependency: {e} — falling back to uniform scores")
        prefixes = _layer_prefixes_from_state(state.keys())
        return {int(p.rsplit('.', 1)[-1]): 0.0 for p in prefixes}

    cfg_path, weights_path = _write_calib_artifacts(state, tmp_dir, cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    fe = SeamlessM4TFeatureExtractor.from_pretrained(str(tmp_dir))

    # Materialise calibration samples once
    n_calib = max(1, cfg["kd_calib_batches"])
    calib_stream = stream_samples(cfg["moshaf"], n_calib * 3)
    raw_inputs, skipped = [], 0
    print(f"  📡 Collecting up to {n_calib} calibration samples...")
    for sample in calib_stream:
        if len(raw_inputs) >= n_calib:
            break
        try:
            audio, _sr = decode_audio(sample["audio"])
        except Exception:
            skipped += 1
            continue
        feats = fe([audio], sampling_rate=16000, return_tensors="pt", padding=True)
        raw_inputs.append(feats["input_features"].cpu())
        del feats
    print(f"  ✅ Collected {len(raw_inputs)} samples ({skipped} skipped)")
    if not raw_inputs:
        # Cleanup temp files
        try: cfg_path.unlink(missing_ok=True); weights_path.unlink(missing_ok=True)
        except Exception: pass
        prefixes = _layer_prefixes_from_state(state.keys())
        return {int(p.rsplit('.', 1)[-1]): 0.0 for p in prefixes}

    model, _ = build_model(cfg_path, weights_path, device)
    if device.type == "cuda":
        model = model.half()
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    n_layers = len(model.wav2vec2_bert.encoder.layers)
    sums = [0.0] * n_layers
    counts = [0]   * n_layers
    handles = []

    def _make_hook(idx):
        def hook(_mod, inp, out):
            h_in  = inp[0]
            h_out = out[0] if isinstance(out, (tuple, list)) else out
            # Both shapes (B, T, H)
            B, T, H = h_in.shape
            cos = F.cosine_similarity(
                h_in.reshape(B * T, H).float(),
                h_out.reshape(B * T, H).float(),
                dim=-1,
            ).mean().item()
            sums[idx]   += cos * (B * T)
            counts[idx] += (B * T)
        return hook

    for i, layer in enumerate(model.wav2vec2_bert.encoder.layers):
        handles.append(layer.register_forward_hook(_make_hook(i)))

    with torch.no_grad():
        for i, inp_cpu in enumerate(raw_inputs):
            inp = inp_cpu.to(device)
            if device.type == "cuda":
                inp = inp.half()
            _ = model.wav2vec2_bert(inp)
            del inp
            if (i + 1) % 10 == 0:
                print(f"    calib: {i+1}/{len(raw_inputs)}", flush=True)

    for h in handles:
        h.remove()

    scores = {i: (sums[i] / counts[i] if counts[i] else 0.0)
              for i in range(n_layers)}

    # Pretty print sorted (most-removable first)
    print("  Layer cosine scores (higher = more prunable):")
    for i, s in sorted(scores.items(), key=lambda x: -x[1]):
        print(f"    layer {i:2d}: {s:+.4f}")

    del model; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try: cfg_path.unlink(missing_ok=True); weights_path.unlink(missing_ok=True)
    except Exception: pass
    return scores


def prune_layers(state, target: int, scores: dict):
    """
    Drop the ``n_layers - target`` layers with the highest cosine scores
    (i.e. the ones whose input ≈ output) and re-index the remaining layers
    to a contiguous 0..target-1 sequence.
    """
    prefixes = _layer_prefixes_from_state(state.keys())
    layer_indices = [int(p.rsplit(".", 1)[-1]) for p in prefixes]
    n_layers = len(layer_indices)

    if target >= n_layers:
        print(f"  Layer count {n_layers} ≤ target {target} — skipping layer pruning")
        return state

    # Highest cosine = first to remove
    ranked = sorted(layer_indices, key=lambda i: -scores.get(i, 0.0))
    remove = set(ranked[:n_layers - target])
    keep   = sorted(i for i in layer_indices if i not in remove)
    remap  = {old: new for new, old in enumerate(keep)}

    print(f"  Pruning layers: {n_layers} → {target}")
    print(f"    removing: {sorted(remove)}")
    print(f"    keeping (old→new): {[(o, remap[o]) for o in keep]}")

    new_state = {}
    for k, v in state.items():
        if not k.startswith(_LAYER_KEY_PREFIX):
            new_state[k] = v
            continue
        parts = k.split(".")
        old_idx = int(parts[_LAYER_KEY_IDX_POS])
        if old_idx in remove:
            continue
        parts[_LAYER_KEY_IDX_POS] = str(remap[old_idx])
        new_state[".".join(parts)] = v
    return new_state


# ═══════════════════════════════════════════════════════════════════
# §2.45  Hidden-size pruning helpers  (SP³, LLM-Pruner — activation L2)
# ═══════════════════════════════════════════════════════════════════

# Tag tables (spec §5.2, extended to cover the Wav2Vec2-BERT adapter).
# Tags are matched as substrings of the parameter key, so leaf module names
# are used (no `.weight` suffix) — that way both `.weight` and `.bias` of
# the same module are caught by a single tag.
_HIDDEN_ROW_TAGS = (
    "feature_projection.projection",   # encoder input projection
    "linear_out",                       # any attention output projection (encoder + adapter)
    "output_dense",                     # any FFN output dense (encoder feed_forward + adapter ffn)
    "layer_norm",                       # all LayerNorms (encoder + adapter, also self_attn_layer_norm / ffn_layer_norm)
    "depthwise_conv",                   # Conformer depthwise conv: weight (hidden, 1, k), bias (hidden,)
    "masked_spec_embed",
)
_HIDDEN_COL_TAGS = (
    "linear_q", "linear_k", "linear_v", # all attention QKV projections
    "intermediate_dense",               # any FFN up-projection (encoder + adapter)
    "ctc_heads",
    "level_to_lm_head",                 # state-dict name; remapped to ctc_heads at load time (build_model L1413)
)
# Doubled-output convs: weight is (2*hidden, hidden, k) → axis-0 needs
# concatenated keep, axis-1 needs plain keep; bias is (2*hidden,).
# Pattern is shared by Conformer pointwise_conv1 and the adapter's
# residual_conv / self_attn_conv (both feed a GLU that halves them back).
_HIDDEN_DOUBLED_TAGS = (
    "pointwise_conv1",
    "residual_conv",
    "self_attn_conv",
)
# Both-axis convs: weight is (hidden, hidden, k); bias is (hidden,)
_HIDDEN_BOTH_TAGS = (
    "pointwise_conv2",
)
_HIDDEN_SKIP_TAGS = (
    "feature_extractor",
    "pos_bias_u", "pos_bias_v",
    "logit_scale",
    # NB: do NOT add "lm_head" — it is a substring of "level_to_lm_head"
    # (the multi-level CTC heads), which must be col-sliced, not skipped.
)


def _discover_hidden_size(state) -> int:
    """Robustly discover the encoder hidden size from a (possibly partially
    pruned) state dict.  Tries several reliable reference keys because FFN
    naming varies across HF Wav2Vec2-BERT checkpoints
    (`feed_forward.*` vs `ffn.*` vs `fc1/fc2`)."""
    # 1. feature_projection.projection.weight: (hidden, input_features)
    for k, v in state.items():
        if "feature_projection.projection.weight" in k:
            return int(v.shape[0])
    # 2. Any encoder layer's linear_q.weight: (n_heads*head_dim, hidden) — cols
    for k, v in state.items():
        if k.startswith("wav2vec2_bert.encoder.layers.") \
                and "self_attn.linear_q.weight" in k:
            return int(v.shape[1])
    # 3. Any *.output_dense.weight regardless of prefix: shape[0] = hidden
    for k, v in state.items():
        if k.endswith("output_dense.weight"):
            return int(v.shape[0])
    raise RuntimeError(
        "Could not infer hidden_size from state dict — none of "
        "feature_projection.projection.weight / encoder.layers.*.linear_q.weight / "
        "*.output_dense.weight were found. Keys present: "
        f"{list(state.keys())[:20]}...")


def _hidden_axis(key: str, v_shape: tuple, hidden_size: int) -> str:
    """Classify a parameter for hidden-size pruning.

    Returns one of:
      - 'row'      : slice axis-0 with `keep`        (1-D / 2-D / 3-D)
      - 'col'      : slice last axis with `keep`     (2-D / 3-D)
      - 'doubled'  : axis-0 = 2*hidden (pw1 / adapter residual / self_attn_conv).
                     Weight: slice axis-0 with `keep_doubled`, axis-1 with `keep`.
                     Bias  : slice axis-0 with `keep_doubled`.
      - 'both'     : axis-0 = axis-1 = hidden (pointwise_conv2 weight).
                     Bias falls back to 'row'.
      - 'none'     : leave untouched.
    """
    if any(t in key for t in _HIDDEN_SKIP_TAGS):
        return "none"

    # Doubled-output convs first — their weight rows are 2*hidden, not hidden,
    # so they must NOT be confused with row tensors.
    if any(t in key for t in _HIDDEN_DOUBLED_TAGS):
        if len(v_shape) >= 2 and v_shape[0] == 2 * hidden_size:
            return "doubled"
        if len(v_shape) == 1 and v_shape[0] == 2 * hidden_size:
            return "doubled"
        # Shape didn't match the (2H, H, ...) pattern — fall through, may be
        # picked up as a normal row/col below or left unchanged.

    # Both-axis (square) convs
    if any(t in key for t in _HIDDEN_BOTH_TAGS):
        if (len(v_shape) >= 2
                and v_shape[0] == hidden_size
                and v_shape[1] == hidden_size):
            return "both"
        if len(v_shape) == 1 and v_shape[0] == hidden_size:
            return "row"   # pointwise_conv2.bias is just a hidden-vector

    is_row = any(t in key for t in _HIDDEN_ROW_TAGS)
    is_col = any(t in key for t in _HIDDEN_COL_TAGS)

    # Square hidden×hidden weights: adapter's self_attn.linear_q/k/v/out are
    # Linear(output_hidden_size, output_hidden_size).  Both axes must be
    # sliced.  Restricted to attention linear_* leaves so it doesn't catch
    # FFN intermediate_dense/output_dense when ffn_target coincidentally
    # equals hidden_size.  Encoder linear_q is rectangular post-head-pruning
    # so it never trips this branch.
    is_attn_linear = any(t in key for t in (
        "self_attn.linear_q", "self_attn.linear_k",
        "self_attn.linear_v", "self_attn.linear_out",
    ))
    if is_attn_linear and len(v_shape) == 2 \
            and v_shape[0] == hidden_size and v_shape[1] == hidden_size:
        return "both"

    # Adapter linear_*.bias are sized to output_hidden_size (they ARE hidden).
    # Encoder linear_q/k/v.bias are sized to num_heads*head_dim and only
    # equal hidden_size by coincidence — those keep the prior 'none' path.
    if "adapter" in key and is_col and len(v_shape) == 1 \
            and v_shape[0] == hidden_size:
        return "row"

    if is_row and v_shape and v_shape[0] == hidden_size:
        return "row"
    # 'col' only applies to multi-D weight matrices.  1-D tensors matching a
    # col-tag substring (linear_q/k/v.bias, intermediate_dense.bias, ctc_heads.bias)
    # are biases sized to QKV-rows / FFN-dim / vocab — they may equal hidden_size
    # by coincidence (e.g. adapter QKV with 16*64=1024) but must be left alone.
    if is_col and v_shape and len(v_shape) >= 2 and v_shape[-1] == hidden_size:
        return "col"
    return "none"


def _score_hidden_dims(cfg: dict, tmp_dir: Path, state):
    """
    Per-dimension activation L2-norm scores; higher = more important.
    Falls back to weight-norm scoring if the live model can't be loaded.
    """
    import numpy as np

    # Discover current hidden size from state
    current_hidden = _discover_hidden_size(state)
    print(f"\n📐 Scoring hidden dims (current size = {current_hidden})...")

    try:
        import torch
        from transformers import SeamlessM4TFeatureExtractor
    except ImportError as e:
        print(f"  ⚠️  {e} — using weight-norm fallback")
        return _score_hidden_dims_weight_only(state, current_hidden)

    cfg_path, weights_path = _write_calib_artifacts(state, tmp_dir, cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        model, _ = build_model(cfg_path, weights_path, device)
    except Exception as e:
        print(f"  ⚠️  Live model load failed ({e}) — using weight-norm fallback")
        try: cfg_path.unlink(missing_ok=True); weights_path.unlink(missing_ok=True)
        except Exception: pass
        return _score_hidden_dims_weight_only(state, current_hidden)

    if device.type == "cuda":
        model = model.half()
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    fe = SeamlessM4TFeatureExtractor.from_pretrained(str(tmp_dir))
    n_calib = max(1, cfg["kd_calib_batches"])

    # Gather calibration audio
    raw_inputs, skipped = [], 0
    for sample in stream_samples(cfg["moshaf"], n_calib * 3):
        if len(raw_inputs) >= n_calib:
            break
        try:
            audio, _sr = decode_audio(sample["audio"])
        except Exception:
            skipped += 1
            continue
        feats = fe([audio], sampling_rate=16000, return_tensors="pt", padding=True)
        raw_inputs.append(feats["input_features"].cpu())
        del feats

    if not raw_inputs:
        del model; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        try: cfg_path.unlink(missing_ok=True); weights_path.unlink(missing_ok=True)
        except Exception: pass
        return _score_hidden_dims_weight_only(state, current_hidden)

    sumsq = np.zeros(current_hidden, dtype=np.float64)

    def hook(_mod, _inp, out):
        h = out[0] if isinstance(out, (tuple, list)) else out
        # h shape (B, T, H)
        flat = h.detach().reshape(-1, h.shape[-1]).float().cpu().numpy()
        sumsq[:] += (flat ** 2).sum(axis=0)

    handle = model.wav2vec2_bert.feature_projection.register_forward_hook(hook)
    with torch.no_grad():
        for i, inp_cpu in enumerate(raw_inputs):
            inp = inp_cpu.to(device)
            if device.type == "cuda": inp = inp.half()
            _ = model.wav2vec2_bert(inp)
            del inp
            if (i + 1) % 10 == 0:
                print(f"    calib: {i+1}/{len(raw_inputs)}", flush=True)
    handle.remove()

    scores = np.sqrt(sumsq).astype(np.float32)
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    try: cfg_path.unlink(missing_ok=True); weights_path.unlink(missing_ok=True)
    except Exception: pass
    print(f"  ✅ Activation-norm scores computed (min={scores.min():.3f}, "
          f"max={scores.max():.3f})")
    return scores


def _score_hidden_dims_weight_only(state, hidden_size: int):
    """Fallback scorer when no live model / calibration data is available."""
    import numpy as np
    scores = np.zeros(hidden_size, dtype=np.float64)
    for k, v in state.items():
        axis = _hidden_axis(k, v.shape, hidden_size)
        if axis == "row" and v.ndim >= 1 and v.shape[0] == hidden_size:
            scores += np.linalg.norm(v.reshape(hidden_size, -1), axis=1)
        elif axis == "col" and v.ndim >= 2 and v.shape[-1] == hidden_size:
            scores += np.linalg.norm(v.reshape(-1, hidden_size), axis=0)
        elif axis == "both":
            scores += np.linalg.norm(v.reshape(hidden_size, -1), axis=1)
    return scores.astype(np.float32)


def prune_hidden_size(state, target: int, scores):
    """Slice every hidden-size axis to ``target`` keeping the top-scoring dims."""
    import numpy as np

    current = _discover_hidden_size(state)

    if target >= current:
        print(f"  Hidden size {current} ≤ target {target} — skipping")
        return state

    keep = np.sort(np.argsort(scores)[-target:])
    keep_doubled = np.concatenate([keep, keep + current])  # for pointwise_conv1 / adapter convs
    print(f"  Pruning hidden size: {current} → {target}  "
          f"(keeping dims {keep[:4].tolist()}...{keep[-4:].tolist()})")

    # One-off shape audit so silent miscompiles surface loudly.
    # Cover both encoder (Conformer) and adapter shapes.
    print("    [audit] critical tensor shapes (pre-slice):")
    seen_audit = set()
    audit_tags = (
        "conv_module.pointwise_conv1.weight",
        "conv_module.pointwise_conv2.weight",
        "self_attn.linear_q.weight",
        "feed_forward.intermediate_dense.weight",
        "feed_forward.output_dense.weight",
        "adapter.layers.0.residual_conv.weight",
        "adapter.layers.0.self_attn_conv.weight",
        "adapter.layers.0.self_attn.linear_q.weight",
        "adapter.layers.0.self_attn.linear_out.weight",
    )
    for k, v in state.items():
        for tag in audit_tags:
            if tag in k and tag not in seen_audit:
                print(f"      {tag}: {tuple(v.shape)}")
                seen_audit.add(tag)
                break

    new_state = {}
    n_doubled = n_both = n_row = n_col = 0
    for k, v in state.items():
        axis = _hidden_axis(k, tuple(v.shape), current)
        if axis == "row":
            n_row += 1
            if v.ndim == 1:
                new_state[k] = v[keep]
            elif v.ndim == 2:
                new_state[k] = v[keep, :]
            elif v.ndim == 3:
                new_state[k] = v[keep, :, :]
            else:
                print(f"    ⚠️  Unexpected row shape: {k} {v.shape} — left unchanged")
                new_state[k] = v
        elif axis == "col":
            n_col += 1
            if v.ndim == 2:
                new_state[k] = v[:, keep]
            elif v.ndim == 3:
                new_state[k] = v[:, keep, :]
            else:
                print(f"    ⚠️  Unexpected col shape: {k} {v.shape} — left unchanged")
                new_state[k] = v
        elif axis == "both":
            # (hidden, hidden, k...) — slice both axes
            n_both += 1
            new_state[k] = v[keep][:, keep] if v.ndim == 2 else v[keep][:, keep, :]
        elif axis == "doubled":
            # (2*hidden, hidden, k...) — pointwise_conv1 / adapter residual_conv / self_attn_conv
            # Bias is 1-D shape (2*hidden,)
            n_doubled += 1
            if v.ndim == 1 and v.shape[0] == 2 * current:
                new_state[k] = v[keep_doubled]
            elif v.ndim == 2 and v.shape[0] == 2 * current:
                new_state[k] = v[keep_doubled][:, keep]
            elif v.ndim == 3 and v.shape[0] == 2 * current:
                new_state[k] = v[keep_doubled][:, keep, :]
            else:
                print(f"    ⚠️  doubled-conv unexpected shape: {k} {v.shape} "
                      f"(expected leading dim {2 * current}) — aborting")
                raise RuntimeError(
                    f"doubled-conv shape mismatch for {k}: {v.shape}")
        else:
            new_state[k] = v
    print(f"    sliced: {n_row} row, {n_col} col, {n_both} both, {n_doubled} doubled")

    # Catch-all: surface any non-skip tensor that STILL has a `current`-sized
    # axis after slicing.  These are likely hidden-tied tensors we missed.
    suspicious = []
    for k, v in new_state.items():
        if any(t in k for t in _HIDDEN_SKIP_TAGS):
            continue
        if any(d == current for d in v.shape):
            suspicious.append(f"      {k}: {tuple(v.shape)}")
    if suspicious:
        print(f"    ⚠️  {len(suspicious)} tensors still contain dim {current} "
              f"after slicing (likely missed tag):")
        for s in suspicious[:20]:
            print(s)
        if len(suspicious) > 20:
            print(f"      ... and {len(suspicious) - 20} more")
    return new_state


def _validate_hidden_size(state, target: int):
    """Sanity-check that every hidden-tied tensor (encoder + adapter) has been
    sliced to ``target``.  Raises RuntimeError listing every mismatch found."""
    errors = []
    for k, v in state.items():
        if "feature_extractor" in k:
            continue  # CNN front-end uses its own channel dims
        if "feature_projection.layer_norm" in k:
            continue  # input-side LayerNorm sized to input_features (e.g. 160), NOT hidden
        # LayerNorm weight/bias are 1-D, must match target exactly
        if "layer_norm" in k and v.ndim == 1 and v.shape[0] != target:
            errors.append(f"  ✗ {k}: shape {tuple(v.shape)}, expected ({target},)")
        # Attention output projection: (hidden, qkv) — encoder & adapter
        if "linear_out.weight" in k and v.shape[0] != target:
            errors.append(f"  ✗ {k}: shape {tuple(v.shape)}, expected ({target}, ...)")
        # FFN output dense: (hidden, ffn) — encoder feed_forward / adapter ffn
        if "output_dense.weight" in k and v.shape[0] != target:
            errors.append(f"  ✗ {k}: shape {tuple(v.shape)}, expected ({target}, ...)")
        # Multi-level CTC heads: (vocab, hidden) — keys are "level_to_lm_head.*" pre-load, "ctc_heads.*" post-load
        if (("ctc_heads" in k) or ("level_to_lm_head" in k)) \
                and k.endswith(".weight") and v.shape[-1] != target:
            errors.append(f"  ✗ {k}: shape {tuple(v.shape)}, expected (..., {target})")
        # Adapter doubled convs: weight (2*hidden, hidden, k); bias (2*hidden,)
        if any(t in k for t in ("residual_conv", "self_attn_conv",
                                 "pointwise_conv1")):
            if v.ndim >= 2 and v.shape[0] != 2 * target:
                errors.append(f"  ✗ {k}: shape {tuple(v.shape)}, "
                              f"expected ({2*target}, {target}, ...)")
            if v.ndim >= 2 and v.shape[1] != target:
                errors.append(f"  ✗ {k}: shape {tuple(v.shape)}, "
                              f"expected ({2*target}, {target}, ...)")
            if v.ndim == 1 and v.shape[0] != 2 * target:
                errors.append(f"  ✗ {k}: shape {tuple(v.shape)}, "
                              f"expected ({2*target},)")
    if errors:
        raise RuntimeError(
            "Hidden-size pruning left inconsistent shapes:\n" + "\n".join(errors))
    print(f"  ✅ Hidden-size validation passed — checked tensors all have dim {target} "
          f"(adapter included)")


# ═══════════════════════════════════════════════════════════════════
# §2.5  KD pruning helpers  (called from run_pruning)
# ═══════════════════════════════════════════════════════════════════

def _compute_kd_pruning_scores(cfg: dict, tmp_dir: Path):
    """
    Compute KD-guided Taylor importance scores for attention heads and FFN
    neurons, using a strict two-phase approach to avoid OOM on 15 GB GPUs.

    The root cause of OOM: loading teacher (fp16, ~5.5 GB) AND student
    (fp32, ~11 GB) simultaneously leaves no headroom for activations.
    The fix: the two models never coexist on the GPU.

    Phase 1 — Teacher  (no grad, fp16, gradient checkpointing)
        Load teacher, run all calibration samples, store every sample's
        hidden states as a list of CPU float16 tensors, then DELETE the
        teacher and empty the CUDA cache before Phase 2.

    Phase 2 — Student  (grad enabled, fp16, gradient checkpointing)
        Load student in fp16 (halves VRAM vs fp32; gradient direction is
        what matters, not numeric precision), replay the stored CPU hidden
        states sample-by-sample, compute L_distill, call .backward() each
        time to accumulate gradients, then extract Taylor scores.

    Returns
    -------
    head_scores : dict { layer_prefix -> np.ndarray(shape=(16,)) } or None
    ffn_scores  : dict { layer_prefix -> np.ndarray(shape=(orig_ffn,)) } or None
    """
    print("\n🎓 Computing KD-guided pruning importance scores (memory-safe two-phase)...")
    try:
        import torch
        import numpy as np
        from transformers import SeamlessM4TFeatureExtractor
    except ImportError as e:
        print(f"  ⚠️  Missing dependency: {e} — skipping KD pruning scoring")
        return None, None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    HEAD_DIM = 64

    fe = SeamlessM4TFeatureExtractor.from_pretrained(str(tmp_dir))

    # ── Collect audio inputs from the calibration stream ───────────
    # We materialise them into CPU RAM once so both phases see the same
    # samples without re-streaming (streaming is not rewindable).
    calib_stream = stream_samples(cfg["moshaf"], cfg["kd_calib_batches"] * 3)
    raw_inputs = []    # list of CPU float32 tensors, shape (1, 80, T)
    skipped = 0
    print(f"  📡 Collecting up to {cfg['kd_calib_batches']} calibration samples...")
    for sample in calib_stream:
        if len(raw_inputs) >= cfg["kd_calib_batches"]:
            break
        try:
            audio, sr = decode_audio(sample["audio"])
        except Exception:
            skipped += 1
            continue
        feats = fe([audio], sampling_rate=16000, return_tensors="pt", padding=True)
        raw_inputs.append(feats["input_features"].cpu())   # keep on CPU
        del feats

    n_samples = len(raw_inputs)
    print(f"  ✅ Collected {n_samples} samples ({skipped} skipped)")
    if n_samples == 0:
        return None, None

    # ════════════════════════════════════════════════════════════════
    # PHASE 1: Teacher — collect hidden states to CPU, then unload
    # ════════════════════════════════════════════════════════════════
    print(f"\n  ── Phase 1/2: teacher forward pass ──")
    disk_check("before loading teacher (KD scoring)")
    teacher, _ = build_model(tmp_dir / "config.json",
                              tmp_dir / "model.safetensors", device)
    if device.type == "cuda":
        teacher = teacher.half()
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()
    # Gradient checkpointing on encoder to further reduce activation memory
    teacher.wav2vec2_bert.encoder.gradient_checkpointing = True
    t_size = sum(p.numel() for p in teacher.parameters()) / 1e6
    print(f"  ✅ Teacher loaded  ({t_size:.0f}M params, fp16)")

    # teacher_hiddens[i] = tuple of CPU fp16 tensors, one per conformer layer
    teacher_hiddens = []
    with torch.no_grad():
        for i, inp_cpu in enumerate(raw_inputs):
            inp = inp_cpu.to(device)
            if device.type == "cuda":
                inp = inp.half()
            t_out = teacher.wav2vec2_bert(inp, output_hidden_states=True)
            # Move every layer's hidden state to CPU immediately to free GPU VRAM
            cpu_hidden = tuple(h.detach().cpu().half() for h in t_out.hidden_states)
            teacher_hiddens.append(cpu_hidden)
            del inp, t_out
            if (i + 1) % 10 == 0:
                print(f"    teacher: {i+1}/{n_samples}", flush=True)

    # Unload teacher completely before loading student
    del teacher; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        free_gb = (torch.cuda.get_device_properties(0).total_memory
                   - torch.cuda.memory_allocated()) / 1e9
        print(f"  🗑️  Teacher unloaded — GPU free: {free_gb:.1f} GB")

    # ════════════════════════════════════════════════════════════════
    # PHASE 2: Student — backward pass using stored teacher hiddens
    # ════════════════════════════════════════════════════════════════
    print(f"\n  ── Phase 2/2: student gradient accumulation ──")
    student, _ = build_model(tmp_dir / "config.json",
                              tmp_dir / "model.safetensors", device)
    if device.type == "cuda":
        # fp16 student: halves VRAM; we only need gradient direction
        student = student.half()
    student.wav2vec2_bert.encoder.gradient_checkpointing = True
    student.train()
    student.zero_grad()
    s_size = sum(p.numel() for p in student.parameters()) / 1e6
    print(f"  ✅ Student loaded  ({s_size:.0f}M params, fp16)")

    for i, (inp_cpu, t_hidden_cpu) in enumerate(zip(raw_inputs, teacher_hiddens)):
        inp = inp_cpu.to(device)
        if device.type == "cuda":
            inp = inp.half()

        # Move this sample's teacher hiddens to GPU just for the KD loss,
        # then discard them immediately after backward
        t_hidden_gpu = tuple(h.to(device) for h in t_hidden_cpu)

        s_out = student.wav2vec2_bert(inp, output_hidden_states=True)
        s_hidden = s_out.hidden_states

        kd_loss = compute_kd_loss(t_hidden_gpu, s_hidden, device)
        kd_loss.backward()

        del inp, t_hidden_gpu, s_out, s_hidden, kd_loss
        if (i + 1) % 10 == 0:
            print(f"    student: {i+1}/{n_samples}", flush=True)

    print(f"  ✅ Gradient accumulation complete ({n_samples} samples)")

    # ── Extract Taylor importance scores ───────────────────────────
    head_scores = {}
    ffn_scores  = {}

    for name, param in student.named_parameters():
        if param.grad is None:
            continue
        # Taylor importance: |w · ∂L/∂w|  — works identically in fp16
        importance = (param.data.abs() * param.grad.abs()).detach().cpu().float().numpy()

        # Attention heads (Q / K / V weight rows)
        for qkv in ("linear_q.weight", "linear_k.weight", "linear_v.weight"):
            if f"self_attn.{qkv}" in name:
                layer_prefix = name.split(".self_attn.")[0]
                n_heads = param.shape[0] // HEAD_DIM
                per_head = importance.reshape(n_heads, HEAD_DIM, -1).mean(axis=(1, 2))
                head_scores[layer_prefix] = (
                    head_scores.get(layer_prefix, 0.0) + per_head)
                break

        # FFN intermediate neurons (up-projection weight rows)
        if (any(tag in name for tag in ("intermediate_dense.weight",
                                         "feed_forward.intermediate",
                                         "fc1.weight"))
                and name.endswith(".weight")):
            for sep in (".feed_forward.", ".ffn."):
                if sep in name:
                    layer_prefix = name.split(sep)[0]
                    per_neuron = importance.mean(axis=1)
                    ffn_scores[layer_prefix] = (
                        ffn_scores.get(layer_prefix, 0.0) + per_neuron)
                    break

    print(f"  📐 Head score layers: {len(head_scores)},  FFN score layers: {len(ffn_scores)}")

    del student, raw_inputs, teacher_hiddens; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    return head_scores, ffn_scores


# ═══════════════════════════════════════════════════════════════════
# §3  Shared model / data utilities
# ═══════════════════════════════════════════════════════════════════
def _import_torch():
    import torch
    import torch.nn as nn
    return torch, nn


def build_model(config_path: Path, weights_path: Path, device):
    """
    Load Wav2Vec2BertForMultilevelCTC from a config + safetensors file.

    The forward method now accepts an optional `return_hidden_states` flag
    so that the KD loss can access per-layer backbone representations.
    """
    import torch.nn as nn
    from safetensors.torch import load_file
    from transformers import Wav2Vec2BertConfig, Wav2Vec2BertModel

    HEAD_DIM = 64  # physical constant of the Wav2Vec2-BERT family

    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)

    level_to_vocab_size = cfg.pop("level_to_vocab_size")
    for k in ("level_to_loss_weight", "architectures", "model_type", "transformers_version"):
        cfg.pop(k, None)
    # Trust config.json for num_attention_heads / intermediate_size /
    # num_hidden_layers / hidden_size — they reflect the actual pruned shapes.
    bert_cfg = Wav2Vec2BertConfig(**cfg)

    class Wav2Vec2BertForMultilevelCTC(nn.Module):
        def __init__(self, config, vocab_sizes):
            super().__init__()
            self.wav2vec2_bert = Wav2Vec2BertModel(config)
            self.dropout = nn.Dropout(config.final_dropout)
            self.ctc_heads = nn.ModuleDict({
                name: nn.Linear(config.hidden_size, vs, bias=True)
                for name, vs in vocab_sizes.items()
            })

        def forward(self, input_features, attention_mask=None,
                    return_hidden_states=False):
            """
            Parameters
            ----------
            return_hidden_states : bool
                If True, also return the tuple of per-layer hidden states
                from the Wav2Vec2BERT backbone (needed for KD loss).

            Returns
            -------
            logits : dict {level_name: Tensor}
            hidden_states : tuple[Tensor, ...], only when return_hidden_states=True
                Each element has shape (B, T, hidden_size).
            """
            out = self.wav2vec2_bert(
                input_features=input_features,
                attention_mask=attention_mask,
                output_hidden_states=return_hidden_states,
            )
            h = self.dropout(out.last_hidden_state)
            logits = {name: head(h) for name, head in self.ctc_heads.items()}
            if return_hidden_states:
                return logits, out.hidden_states
            return logits

    model = Wav2Vec2BertForMultilevelCTC(bert_cfg, level_to_vocab_size)

    # Manual param assignment (handles shape mismatch from pruning)
    sd = load_file(str(weights_path), device=str(device))
    remapped = {
        (k.replace("level_to_lm_head.", "ctc_heads.") if k.startswith("level_to_lm_head.") else k): v
        for k, v in sd.items()
    }
    for name, tensor in remapped.items():
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

    # Patch attention head counts for pruned shapes
    for layer in model.wav2vec2_bert.encoder.layers:
        a = layer.self_attn
        a.num_heads = a.linear_q.weight.shape[0] // HEAD_DIM
        a.head_dim  = HEAD_DIM
        if hasattr(a, "head_size"): a.head_size = HEAD_DIM
    if hasattr(model.wav2vec2_bert, "adapter") and model.wav2vec2_bert.adapter:
        for al in model.wav2vec2_bert.adapter.layers:
            if hasattr(al, "self_attn"):
                al.self_attn.num_heads = al.self_attn.linear_q.weight.shape[0] // HEAD_DIM
                al.self_attn.head_dim  = HEAD_DIM
                if hasattr(al.self_attn, "head_size"):
                    al.self_attn.head_size = HEAD_DIM

    model.to(device)
    return model, level_to_vocab_size


# ── KD: teacher loading ────────────────────────────────────────────

def load_teacher(cfg: dict, device):
    """
    Load and return the frozen full-size teacher model.

    The teacher weights are looked up in `cfg["teacher_dir"]`.  If that
    directory doesn't exist yet (e.g. the pruning stage was skipped),
    they are downloaded fresh from `cfg["hf_repo"]`.

    The teacher is always:
      • cast to float16 on CUDA (to save VRAM)
      • all parameters frozen (requires_grad = False)
      • in eval() mode
    """
    import torch
    from huggingface_hub import hf_hub_download

    teacher_dir = cfg["teacher_dir"]

    if not (teacher_dir / "model.safetensors").exists():
        print(f"  📥 Teacher not cached — downloading from {cfg['hf_repo']}...")
        teacher_dir.mkdir(parents=True, exist_ok=True)
        disk_check("before teacher download")
        for fname in ["config.json", "model.safetensors", "vocab.json",
                      "preprocessor_config.json", "added_tokens.json",
                      "special_tokens_map.json", "tokenizer_config.json"]:
            hf_hub_download(repo_id=cfg["hf_repo"], filename=fname,
                            local_dir=str(teacher_dir))

    print("  📦 Loading teacher model (frozen)...")
    disk_check("before loading teacher")
    teacher, _ = build_model(teacher_dir / "config.json",
                              teacher_dir / "model.safetensors", device)

    # Use float16 on GPU to halve VRAM cost
    if device.type == "cuda":
        teacher = teacher.half()

    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()

    size_m = sum(p.numel() for p in teacher.parameters()) / 1e6
    vram_gb = size_m * (2 if device.type == "cuda" else 4) / 1024
    print(f"  ✅ Teacher loaded  ({size_m:.0f}M params, ~{vram_gb:.1f} GB VRAM, frozen)")
    return teacher


# ── KD: distillation loss ──────────────────────────────────────────

def compute_kd_loss(teacher_hidden, student_hidden, device,
                    return_per_layer=False):
    """
    Distillation loss from Li et al. 2510.04213, equation (5):

        L_distill = Σ_l [L1(h_teacher^l, h_student^l) − cosine(h_teacher^l, h_student^l)] / n_layers

    Both L1 (lower = better) and −cosine (lower = better, i.e. we
    maximise cosine similarity) pull the student toward the teacher.

    Memory design: teacher_hidden tensors may be stored on CPU to avoid
    holding two full models' activations on the GPU simultaneously.
    Each teacher layer tensor is moved to `device`, used for the loss,
    then immediately freed before the next layer is processed.

    Parameters
    ----------
    teacher_hidden : tuple[Tensor, ...]
        Per-layer hidden states; may be on CPU (will be moved per-layer).
    student_hidden : tuple[Tensor, ...]
        Per-layer hidden states from the student (must have grad).
    device : torch.device
    return_per_layer : bool, optional
        When True, also return a dict with per-layer cosine similarity
        (float, detached, on CPU) for logging.  Off by default to keep
        the fast path allocation-free.

    Returns
    -------
    loss : scalar Tensor (float32)
    per_layer : dict, only when return_per_layer=True
        {"cos_per_layer": [float, ...]}  — one cosine per aligned layer.
    """
    import torch
    import torch.nn.functional as F

    n_layers = min(len(teacher_hidden), len(student_hidden))
    loss = torch.zeros(1, device=device, dtype=torch.float32).squeeze()
    n_used = 0
    per_layer_cos = [] if return_per_layer else None

    for h_t_raw, h_s in zip(teacher_hidden[:n_layers], student_hidden[:n_layers]):
        # Move teacher tensor to GPU just for this layer, then drop it
        h_t = h_t_raw.to(device=device, dtype=torch.float32)
        h_s = h_s.float()

        # Skip layers where student/teacher hidden_size differs (occurs when
        # --hidden_target < teacher hidden_size).  Either an exact projection
        # would be needed (SliceGPT) or a learned map; for v4 we just drop
        # those layers from the KD sum.
        if h_s.shape[-1] != h_t.shape[-1]:
            del h_t
            if return_per_layer:
                per_layer_cos.append(float("nan"))
            continue

        l1 = F.l1_loss(h_s, h_t)

        B, T, D = h_s.shape
        cos_sim = F.cosine_similarity(
            h_s.reshape(B * T, D),
            h_t.reshape(B * T, D),
            dim=-1,
        ).mean()

        loss = loss + l1 - cos_sim
        n_used += 1
        if return_per_layer:
            per_layer_cos.append(float(cos_sim.detach().cpu()))

        # Free the GPU copy of this teacher layer immediately
        del h_t

    if n_used == 0:
        if return_per_layer:
            return loss, {"cos_per_layer": per_layer_cos}
        return loss  # zero — nothing aligned, caller should disable KD
    out = loss / n_used
    if return_per_layer:
        return out, {"cos_per_layer": per_layer_cos}
    return out


def decode_audio(audio_field, target_sr=16000):
    import io as _io
    import soundfile as sf
    buf = _io.BytesIO(audio_field["bytes"])
    waveform, sr = sf.read(buf, dtype="float32")
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    return waveform, sr


def encode_labels(sample, vocab, levels):
    encoded = {}
    phoneme_str = sample.get("phonemes", "") or ""
    pm = vocab.get("phonemes", {})
    encoded["phonemes"] = [pm[c] for c in phoneme_str if c in pm]
    sifat = sample.get("sifat", []) or []
    for level in levels:
        if level == "phonemes":
            continue
        lm = vocab.get(level, {})
        encoded[level] = [lm[v] for entry in sifat for v in [entry.get(level, "")] if v and v in lm]
    return encoded


def stream_samples(moshaf, num_samples):
    """
    moshaf : str or list[str]
        A single moshaf name or a list of moshaf names.
        When a list is given, datasets are interleaved with equal probability
        so every moshaf contributes roughly the same number of samples.
    num_samples : int
        Total number of samples to take from the combined stream.
    """
    from datasets import Audio, load_dataset, interleave_datasets

    if isinstance(moshaf, str):
        moshaf = [moshaf]

    print(f"  📡 Creating lazy stream for {num_samples} samples "
          f"across {len(moshaf)} moshaf(s): {moshaf}...")

    streams = []
    for name in moshaf:
        ds = load_dataset(
            "obadx/muaalem-annotated-v3", name,
            split="train", streaming=True,
        )
        ds = ds.cast_column("audio", Audio(decode=False))
        streams.append(ds)

    if len(streams) == 1:
        combined = streams[0]
    else:
        combined = interleave_datasets(
            streams,
            probabilities=[1 / len(streams)] * len(streams),
            seed=42,
            stopping_strategy="all_exhausted",
        )

    return combined.take(num_samples)


def build_augmentation():
    try:
        from audiomentations import Compose, TimeStretch, GainTransition, AddGaussianNoise
        return Compose([
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.4),
            TimeStretch(min_rate=0.8, max_rate=1.5, p=0.4),
            GainTransition(min_gain_db=-6, max_gain_db=6, p=0.4),
        ])
    except ImportError:
        print("  ⚠️  audiomentations not installed — skipping augmentation")
        return None


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
    # ── AMP GradScaler state (so loss-scale survives a resume) ──────
    # Only present when training on CUDA with mixed precision.  Save it
    # next to the optimizer so load_checkpoint can restore it.
    scaler_path = ckpt_dir / "scaler_latest.pt"
    if scaler is not None:
        torch.save(scaler.state_dict(), str(scaler_path))
    else:
        # Avoid a stale scaler file lingering from a previous CUDA run
        # if this checkpoint was written without a scaler (e.g. CPU).
        if scaler_path.exists():
            scaler_path.unlink()
    (ckpt_dir / "training_state.json").write_text(
        json.dumps({"completed_epoch": epoch, "training_log": log}, indent=2))
    print(f"  💾 Checkpoint saved (epoch {epoch})")
    disk_check("after checkpoint")


def load_checkpoint(ckpt_dir: Path, model, optimizer, device, scaler=None,
                    wandb_logger=None):
    import torch
    state_path = ckpt_dir / "training_state.json"
    # ── W&B resume ─────────────────────────────────────────────────
    # If there is no local checkpoint but W&B is enabled, try to pull the
    # latest complete checkpoint artifact for this stage.  This is what
    # lets a brand-new machine / fresh Kaggle session continue training.
    if not state_path.exists() and wandb_logger is not None:
        try:
            wandb_logger.restore_checkpoint(ckpt_dir)
        except Exception as e:
            print(f"  ⚠️  W&B checkpoint restore failed ({e})")
    if not state_path.exists():
        print("  ℹ️  No checkpoint — starting from scratch")
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
                for p in parts[:-1]: obj = getattr(obj, p)
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
    # ── Restore AMP GradScaler state if both a scaler and a saved
    #    scaler file are present.  Older checkpoints won't have the
    #    file; in that case the scaler keeps its fresh default state.
    sp = ckpt_dir / "scaler_latest.pt"
    if scaler is not None and sp.exists():
        try:
            scaler.load_state_dict(
                torch.load(str(sp), map_location=device, weights_only=True))
            print("  ♻️  Restored AMP GradScaler state")
        except Exception as e:
            print(f"  ⚠️  Could not restore GradScaler state ({e}); "
                  "continuing with a fresh scaler")
    elif scaler is not None and not sp.exists():
        print("  ℹ️  No saved GradScaler state — using fresh scaler")
    print(f"  ♻️  Resuming from epoch {ep + 1}")
    return ep, log


def build_lr_scheduler(optimizer, cfg: dict, total_epochs: int,
                       last_completed_epoch: int = 0):
    """
    Build a per-epoch learning-rate scheduler shared by the FT and QAT
    stages.  The scheduler is a ``torch.optim.lr_scheduler.LambdaLR`` whose
    multiplier is a function of the (0-based) epoch index:

        epoch e in [0, warmup)            : linear warmup 0 → 1
        epoch e in [warmup, total_epochs) : decay 1 → floor following the
                                            chosen schedule (cosine / linear)

    The multiplier is applied to each param-group's *base* lr (the lr the
    optimizer was constructed with — ``ft_lr`` / ``qat_lr``).  The floor
    multiplier is ``lr_min / base_lr`` so the curve lands on ``--lr_min``.

    Resuming
    --------
    ``LambdaLR`` replays its lambda for every epoch up to ``last_epoch``,
    so passing ``last_epoch = last_completed_epoch - 1`` reconstructs the
    exact lr the run had reached — no scheduler state needs to live in the
    checkpoint.  (We rebuild from scratch each resume and fast-forward.)

    Parameters
    ----------
    total_epochs : int
        Total planned epochs for this stage (``ft_epochs`` / ``qat_epochs``).
    last_completed_epoch : int
        Number of epochs already finished (0 on a fresh run).  Used to
        fast-forward the schedule on resume.

    Returns
    -------
    scheduler : torch.optim.lr_scheduler.LambdaLR or None
        ``None`` when ``--lr_schedule constant`` (caller then leaves the lr
        untouched, reproducing the legacy fixed-lr behaviour).
    """
    import math as _math
    from torch.optim.lr_scheduler import LambdaLR

    schedule = cfg.get("lr_schedule", "cosine")
    if schedule == "constant" and int(cfg.get("lr_warmup_epochs", 0)) <= 0:
        # Nothing to schedule — keep the old constant-lr path entirely.
        return None

    warmup    = max(0, int(cfg.get("lr_warmup_epochs", 0)))
    base_lrs  = [g["lr"] for g in optimizer.param_groups]
    base_lr   = base_lrs[0] if base_lrs else 1.0
    lr_min    = float(cfg.get("lr_min", 0.0))
    # Floor as a multiplier of the base lr; guard against base_lr == 0.
    floor_mult = (lr_min / base_lr) if base_lr > 0 else 0.0
    floor_mult = min(max(floor_mult, 0.0), 1.0)

    # Number of "decay" epochs after warmup.  Guard against div-by-zero
    # when total_epochs <= warmup (degenerate config) — then we just hold
    # at the base lr after warmup.
    decay_epochs = max(1, total_epochs - warmup)

    def lr_lambda(epoch: int) -> float:
        # `epoch` is 0-based (LambdaLR convention).
        if warmup > 0 and epoch < warmup:
            # Linear warmup 0 → 1 over `warmup` epochs.  +1 so the first
            # epoch isn't exactly 0 (which would waste a whole epoch).
            return float(epoch + 1) / float(warmup)

        if schedule == "constant":
            return 1.0

        # Progress through the decay phase in [0, 1].
        prog = (epoch - warmup) / float(decay_epochs)
        prog = min(max(prog, 0.0), 1.0)

        if schedule == "cosine":
            cosine = 0.5 * (1.0 + _math.cos(_math.pi * prog))
            return floor_mult + (1.0 - floor_mult) * cosine
        elif schedule == "linear":
            return floor_mult + (1.0 - floor_mult) * (1.0 - prog)
        else:
            return 1.0

    # last_epoch is 0-based and means "the last epoch already stepped".
    # On a fresh run (last_completed_epoch == 0) we want last_epoch == -1
    # so the very first .get_last_lr() reflects epoch 0.
    last_epoch = last_completed_epoch - 1
    # When resuming (last_epoch >= 0) LambdaLR requires each param group to
    # already carry its 'initial_lr' (it normally stamps this only on a
    # fresh last_epoch == -1 construction).  The optimizer is rebuilt fresh
    # on every resume with its base lr intact, so seed initial_lr = current
    # lr here; this makes the lambda multiply against the correct base.
    if last_epoch >= 0:
        for g in optimizer.param_groups:
            g.setdefault("initial_lr", g["lr"])
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda, last_epoch=last_epoch)
    return scheduler


# ═══════════════════════════════════════════════════════════════════
# §4  Generic training loop  (shared by FT and QAT)
# ═══════════════════════════════════════════════════════════════════
def train_one_epoch(
    model, feature_extractor, dataset_stream, vocab, levels, loss_weights,
    optimizer, device, batch_size, grad_accum_steps,
    scaler=None, augment_fn=None, max_duration=15.0, epoch_num=1,
    # ── KD: new optional parameters ────────────────────────────────
    teacher_model=None,
    kd_alpha=0.0,
    # ── W&B: optional logger for per-step metrics ──────────────────
    wandb_logger=None,
):
    """
    One training epoch combining CTC task loss with optional KD distillation.

    KD is activated when `teacher_model` is not None and `kd_alpha > 0`.
    At each batch step the frozen teacher is forwarded (no_grad) to obtain
    its per-layer hidden states; then the student's hidden states are
    aligned to the teacher's via `compute_kd_loss` (eq. 5, 2510.04213).

        L_total = L_CTC + kd_alpha × L_distill

    The teacher runs in float16 on GPU and its activations are freed
    immediately after the KD loss is computed to minimise VRAM pressure.
    """
    import torch
    import torch.nn as nn
    from torch.cuda.amp import autocast

    use_kd = (teacher_model is not None) and (kd_alpha > 0.0)
    log_wb = (wandb_logger is not None) and getattr(wandb_logger, "active", False)

    model.train()
    ctc = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    total_loss, total_kd_loss, batches = 0.0, 0.0, 0
    optimizer.zero_grad()
    use_amp = scaler is not None

    # Per-optimizer-step accumulators (reset after each .step())
    step_loss_sum  = 0.0
    step_kd_sum    = 0.0
    step_level_sum = {lv: 0.0 for lv in levels}
    step_level_cnt = {lv: 0   for lv in levels}
    step_samples   = 0
    step_kd_cos_sum = None   # list of running sums per layer
    step_kd_cos_cnt = 0
    step_kd_t_time  = 0.0
    step_kd_t_mem   = 0.0
    step_start_t    = time.time()

    batch = []
    for s in dataset_stream:
        batch.append(s)
        if len(batch) < batch_size:
            continue

        audios, labels = [], {}
        for bs in batch:
            try:
                audio, sr = decode_audio(bs["audio"])
            except Exception:
                continue
            if len(audio) / sr > max_duration:
                continue
            if augment_fn:
                audio = augment_fn(samples=audio, sample_rate=sr)
            audios.append(audio)
            enc = encode_labels(bs, vocab, levels)
            for lv in levels:
                labels.setdefault(lv, []).append(enc.get(lv, []))

        batch = []

        if not audios:
            continue

        feats = feature_extractor(audios, sampling_rate=16000, return_tensors="pt", padding=True)
        inp   = feats["input_features"].to(device)
        mask  = feats.get("attention_mask")
        if mask is not None:
            mask = mask.to(device)

        # ── KD: teacher forward (no grad, fp16) ────────────────────
        # Hidden states are moved to CPU immediately after generation so
        # they don't compete for VRAM with the student's activations.
        # compute_kd_loss moves them back one layer at a time.
        t_hidden = None
        if use_kd:
            t_fwd_t0 = time.time() if log_wb else 0.0
            t_mem_before = _gpu_mem_mb() if log_wb else 0.0
            with torch.no_grad():
                _, t_hidden_gpu = teacher_model(
                    inp.half() if device.type == "cuda" else inp,
                    attention_mask=mask,
                    return_hidden_states=True,
                )
                if log_wb:
                    t_mem_peak = _gpu_mem_mb()
                    step_kd_t_mem = max(step_kd_t_mem,
                                        t_mem_peak - t_mem_before)
                # Move to CPU immediately; freed after KD loss below
                t_hidden = tuple(h.detach().cpu() for h in t_hidden_gpu)
                del t_hidden_gpu
            if log_wb:
                step_kd_t_time += (time.time() - t_fwd_t0)

        # ── Student forward ────────────────────────────────────────
        with autocast(enabled=use_amp, dtype=torch.float16):
            if use_kd:
                logits, s_hidden = model(inp, attention_mask=mask,
                                         return_hidden_states=True)
            else:
                logits = model(inp, attention_mask=mask)

            # ── CTC task loss ──────────────────────────────────────
            loss = None
            for lv in levels:
                tgts = labels.get(lv, [])
                if not tgts or all(len(t) == 0 for t in tgts):
                    continue
                lp = logits[lv].float().log_softmax(-1).permute(1, 0, 2)
                T  = lp.size(0)
                il = torch.full((len(audios),), T, dtype=torch.long, device=device)
                tl = torch.tensor([len(t) for t in tgts], dtype=torch.long, device=device)
                ts = torch.tensor([x for t in tgts for x in t], dtype=torch.long, device=device)
                if (tl > T).any():
                    continue
                lv_loss = ctc(lp, ts, il, tl)
                if torch.isnan(lv_loss) or torch.isinf(lv_loss):
                    continue
                weighted = loss_weights.get(lv, 1.0) * lv_loss
                loss = weighted if loss is None else loss + weighted
                if log_wb:
                    step_level_sum[lv] += float(lv_loss.detach())
                    step_level_cnt[lv] += 1

            # ── KD distillation loss  (eq. 5 of 2510.04213) ────────
            if use_kd and t_hidden is not None:
                if log_wb:
                    kd_loss, kd_extra = compute_kd_loss(
                        t_hidden, s_hidden, device, return_per_layer=True)
                    cos_per_layer = kd_extra["cos_per_layer"]
                    if step_kd_cos_sum is None:
                        step_kd_cos_sum = [0.0] * len(cos_per_layer)
                    elif len(cos_per_layer) > len(step_kd_cos_sum):
                        step_kd_cos_sum.extend(
                            [0.0] * (len(cos_per_layer) - len(step_kd_cos_sum)))
                    for i, c in enumerate(cos_per_layer):
                        if c == c:   # not NaN
                            step_kd_cos_sum[i] += c
                    step_kd_cos_cnt += 1
                else:
                    kd_loss = compute_kd_loss(t_hidden, s_hidden, device)
                if loss is not None:
                    loss = loss + kd_alpha * kd_loss
                else:
                    loss = kd_alpha * kd_loss
                kd_val = float(kd_loss.detach())
                total_kd_loss += kd_val
                if log_wb:
                    step_kd_sum += kd_val

        if loss is None:
            continue

        loss = loss / grad_accum_steps

        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        batches += 1
        step_samples += len(audios)
        loss_full = loss.item() * grad_accum_steps
        total_loss += loss_full
        if log_wb:
            step_loss_sum += loss_full

        # Free large tensors eagerly
        del feats, inp, logits
        if use_kd:
            del t_hidden, s_hidden, kd_loss

        if batches % grad_accum_steps == 0:
            grad_norm = None
            if use_amp:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            optimizer.zero_grad()

            if log_wb:
                step_dt = max(time.time() - step_start_t, 1e-9)
                metrics = {
                    "train/loss_total":   step_loss_sum / grad_accum_steps,
                    "train/lr":           optimizer.param_groups[0]["lr"],
                    "train/grad_norm":    float(grad_norm) if grad_norm is not None else 0.0,
                    "train/amp_scale":    float(scaler.get_scale()) if use_amp else 1.0,
                    "perf/batch_time_s":  step_dt,
                    "perf/samples_per_sec": step_samples / step_dt,
                    "perf/gpu_mem_mb":    _gpu_mem_mb(),
                    "train/epoch":        epoch_num,
                    "train/batches_seen": batches,
                }
                util = _gpu_util_pct()
                if util is not None:
                    metrics["perf/gpu_util_pct"] = util
                for lv in levels:
                    if step_level_cnt[lv]:
                        metrics[f"train/loss_{lv}"] = (
                            step_level_sum[lv] / step_level_cnt[lv])
                if use_kd:
                    metrics["train/loss_kd"]      = step_kd_sum / grad_accum_steps
                    metrics["train/loss_blended"] = (
                        step_loss_sum / grad_accum_steps)
                    metrics["kd/teacher_fwd_time_s"] = step_kd_t_time
                    metrics["kd/teacher_fwd_mem_mb"] = step_kd_t_mem
                    if step_kd_cos_sum is not None and step_kd_cos_cnt:
                        for i, s_sum in enumerate(step_kd_cos_sum):
                            metrics[f"kd/cos_sim_layer_{i:02d}"] = (
                                s_sum / step_kd_cos_cnt)
                wandb_logger.log_step(metrics)

                # Reset per-step accumulators
                step_loss_sum  = 0.0
                step_kd_sum    = 0.0
                for lv in levels:
                    step_level_sum[lv] = 0.0
                    step_level_cnt[lv] = 0
                step_samples       = 0
                step_kd_cos_sum    = None
                step_kd_cos_cnt    = 0
                step_kd_t_time     = 0.0
                step_kd_t_mem      = 0.0
                step_start_t       = time.time()

        if batches % 200 == 0:
            kd_str = f"  kd={total_kd_loss/batches:.4f}" if use_kd else ""
            print(f"    [{epoch_num}] step {batches}  "
                  f"loss={total_loss/batches:.4f}{kd_str}", flush=True)

    avg_loss = total_loss / max(batches, 1)
    avg_kd   = total_kd_loss / max(batches, 1)
    return avg_loss, avg_kd


# ═══════════════════════════════════════════════════════════════════
# §5  STAGE 2 — Fine-Tuning  (with optional KD)
# ═══════════════════════════════════════════════════════════════════
def run_finetune(cfg: dict):
    print("\n" + "═"*60)
    print("  🏋️  STAGE 2: Fine-Tuning")
    if cfg["kd_alpha"] > 0:
        print(f"  🎓  KD distillation ENABLED  (α = {cfg['kd_alpha']})")
        if cfg.get("hidden_target", 1024) < 1024:
            print("  ⚠️  KD distillation with hidden_size mismatch is approximate; "
                  "layers with mismatched hidden_size are skipped from the KD sum. "
                  "Consider --kd_alpha 0 when using --hidden_target.")
    print("═"*60)

    import torch
    from torch.cuda.amp import GradScaler
    from transformers import SeamlessM4TFeatureExtractor

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir  = cfg["model_dir"]
    ckpt_dir   = cfg["ft_ckpt"]
    exp_dir    = cfg["exp_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    wandb_logger = WandbLogger(cfg, "ft")
    wandb_logger.start()

    print(f"  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Load vocab & feature extractor
    with open(model_dir / "vocab.json", encoding="utf-8") as f:
        vocab = json.load(f)
    fe = SeamlessM4TFeatureExtractor.from_pretrained(str(model_dir))

    # Load pruned student model
    disk_check("before loading FT model")
    print("\n📦 Loading pruned model...")
    model, _ = build_model(model_dir / "config.json", model_dir / "model.safetensors", device)

    # Freeze feature extractor
    for name, p in model.named_parameters():
        if "feature_extractor" in name or "feature_projection" in name:
            p.requires_grad_(False)

    if not cfg["no_grad_ckpt"]:
        model.wav2vec2_bert.encoder.gradient_checkpointing = True
        print("  ⚡ Gradient checkpointing ON (use --no_grad_ckpt to disable)")
    else:
        print("  ⚡ Gradient checkpointing OFF (--no_grad_ckpt set)")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,}  ({trainable/total*100:.1f}%)")
    wandb_logger.log_summary({
        "ft/params_trainable": int(trainable),
        "ft/params_total":     int(total),
        "ft/trainable_pct":    trainable / max(total, 1) * 100,
    })

    # ── KD: load frozen teacher ────────────────────────────────────
    teacher = None
    if cfg["kd_alpha"] > 0:
        teacher = load_teacher(cfg, device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["ft_lr"], weight_decay=0.01,
    )
    scaler = GradScaler() if device.type == "cuda" else None

    start_ep, log = load_checkpoint(ckpt_dir, model, optimizer, device,
                                    scaler=scaler, wandb_logger=wandb_logger)

    # ── Learning-rate scheduler (per-epoch; None when constant lr) ──
    # Built AFTER load_checkpoint so it can fast-forward to the resumed
    # epoch.  No scheduler state is stored in the checkpoint: LambdaLR
    # deterministically replays its lambda from start_ep.
    scheduler = build_lr_scheduler(
        optimizer, cfg, total_epochs=cfg["ft_epochs"],
        last_completed_epoch=start_ep)
    if scheduler is not None:
        print(f"  📉 LR schedule: {cfg['lr_schedule']}  "
              f"(base={cfg['ft_lr']:.2e}  min={cfg['lr_min']:.2e}  "
              f"warmup={cfg['lr_warmup_epochs']} ep)  "
              f"start lr={optimizer.param_groups[0]['lr']:.2e}")
    else:
        print(f"  📉 LR schedule: constant ({cfg['ft_lr']:.2e})")

    if start_ep >= cfg["ft_epochs"]:
        print(f"  ✅ Fine-tuning already complete ({start_ep} epochs)")
        mark_done(exp_dir, "finetune")
        wandb_logger.log_summary({"ft/status": "already_complete"})
        wandb_logger.finish("success")
        return

    augment = build_augmentation()
    samples = stream_samples(cfg["moshaf"], cfg["ft_samples"])

    # Stage the static side-cars into the checkpoint dir up front so every
    # per-epoch W&B checkpoint artifact is self-contained (model loads need
    # config.json/vocab.json on a clean-machine resume).  Re-copied after
    # the loop too (harmless) to cover the QAT-stage hand-off.
    for fname in ["config.json", "vocab.json", "preprocessor_config.json"]:
        src = model_dir / fname
        if src.exists() and not (ckpt_dir / fname).exists():
            shutil.copy2(str(src), str(ckpt_dir / fname))

    best_loss = float("inf")

    for epoch in range(start_ep + 1, cfg["ft_epochs"] + 1):
        print(f"\n  ── Epoch {epoch}/{cfg['ft_epochs']} ──")
        disk_check(f"epoch {epoch} start")
        t0 = time.time()
        avg_loss, avg_kd = train_one_epoch(
            model, fe, samples, vocab, cfg["levels"], cfg["loss_weights"],
            optimizer, device,
            batch_size=cfg["ft_batch"], grad_accum_steps=cfg["ft_accum"],
            scaler=scaler, augment_fn=augment, max_duration=cfg["ft_max_dur"],
            epoch_num=epoch,
            # ── KD ──────────────────────────────────────────────────
            teacher_model=teacher,
            kd_alpha=cfg["kd_alpha"],
            # ── W&B ────────────────────────────────────────────────
            wandb_logger=wandb_logger,
        )
        elapsed = time.time() - t0
        kd_str  = f"  kd_loss={avg_kd:.4f}" if cfg["kd_alpha"] > 0 else ""
        print(f"  ✅ Epoch {epoch}  loss={avg_loss:.4f}{kd_str}  ({elapsed:.0f}s)")
        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss
        log.append({"epoch": epoch, "loss": avg_loss, "kd_loss": avg_kd,
                    "time_s": round(elapsed, 1)})
        save_checkpoint(model, optimizer, epoch, log, ckpt_dir, best=is_best,
                        scaler=scaler)

        # Advance the LR schedule one epoch and report the lr for the
        # epoch about to start (no-op when schedule is constant/None).
        if scheduler is not None:
            scheduler.step()
            print(f"  📉 next-epoch lr = {optimizer.param_groups[0]['lr']:.3e}")

        epoch_metrics = {
            "loss": avg_loss, "time_s": elapsed, "epoch_num": epoch,
            "best_loss": best_loss, "is_best": int(is_best),
        }
        if cfg["kd_alpha"] > 0:
            epoch_metrics["kd_loss"] = avg_kd
        wandb_logger.log_epoch(epoch_metrics)

        # ── Full resumable checkpoint → W&B ────────────────────────
        # Uploads the entire checkpoint dir (weights + optimizer + AMP
        # scaler + training_state + config/vocab) so training can be
        # resumed from W&B alone.  Under --wandb_ckpt all we push every
        # epoch; under 'best' we push only when this epoch improved, so
        # the 'latest'-aliased restore point is always a good model.
        if cfg.get("wandb_ckpt") == "all" or (
                cfg.get("wandb_ckpt") == "best" and is_best):
            wandb_logger.log_checkpoint(
                ckpt_dir, epoch,
                metadata={"loss": float(avg_loss),
                          "best_loss": float(best_loss),
                          "is_best": bool(is_best)})

        if is_best and cfg.get("wandb_ckpt") in ("best", "all"):
            best_path = ckpt_dir / "model_best.safetensors"
            if best_path.exists():
                wandb_logger.log_artifact(
                    best_path,
                    name=f"{cfg['name']}-ft-best",
                    art_type="model",
                    metadata={"epoch": epoch, "best_loss": float(avg_loss)},
                    gate=False,   # already gated by wandb_ckpt check above
                )

    # Copy config/vocab into checkpoint dir for QAT stage
    for fname in ["config.json", "vocab.json", "preprocessor_config.json"]:
        src = model_dir / fname
        if src.exists():
            shutil.copy2(str(src), str(ckpt_dir / fname))

    wandb_logger.log_summary({
        "ft/best_loss": float(best_loss),
        "ft/epochs_run": cfg["ft_epochs"] - start_ep,
    })

    del model
    if teacher is not None:
        del teacher
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    mark_done(exp_dir, "finetune")
    wandb_logger.finish("success")
    print("\n✅ Fine-tuning complete")


# ═══════════════════════════════════════════════════════════════════
# §6  QAT helpers — FakeQuantize + QATLinear
# ═══════════════════════════════════════════════════════════════════
def _build_qat_modules(bits=8, symmetric=True, ema_decay=0.999):
    import torch
    import torch.nn as nn

    class FakeQuantizeSTE(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, scale, zp, qmin, qmax):
            xq  = torch.clamp(torch.round(x / scale + zp), qmin, qmax)
            xdq = (xq - zp) * scale
            ctx.save_for_backward((xq >= qmin) & (xq <= qmax))
            return xdq
        @staticmethod
        def backward(ctx, g):
            mask, = ctx.saved_tensors
            return g * mask.float(), None, None, None, None

    class MinMaxObserver(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("ema_min",      torch.tensor(float("inf")))
            self.register_buffer("ema_max",      torch.tensor(float("-inf")))
            self.register_buffer("num_observed", torch.tensor(0, dtype=torch.long))
        def forward(self, x):
            if self.training:
                mn, mx = x.detach().min(), x.detach().max()
                if self.num_observed == 0:
                    self.ema_min.copy_(mn); self.ema_max.copy_(mx)
                else:
                    self.ema_min.copy_(ema_decay * self.ema_min + (1 - ema_decay) * mn)
                    self.ema_max.copy_(ema_decay * self.ema_max + (1 - ema_decay) * mx)
                self.num_observed += 1
            return x
        def scale_zp(self):
            if symmetric:
                qmax_ = 2**(bits-1) - 1; qmin_ = -qmax_ - 1
                amax  = torch.clamp(torch.max(self.ema_min.abs(), self.ema_max.abs()), min=1e-8)
                return amax / qmax_, torch.tensor(0.0, device=amax.device), qmin_, qmax_
            else:
                qmin_ = 0; qmax_ = 2**bits - 1
                r = torch.clamp(self.ema_max - self.ema_min, min=1e-8)
                sc = r / (qmax_ - qmin_)
                zp = torch.clamp(torch.round(qmin_ - self.ema_min / sc), qmin_, qmax_)
                return sc, zp, qmin_, qmax_

    class QATLinear(nn.Module):
        def __init__(self, linear: nn.Linear):
            super().__init__()
            self.in_features  = linear.in_features
            self.out_features = linear.out_features
            self.weight = nn.Parameter(linear.weight.data.clone())
            self.bias   = nn.Parameter(linear.bias.data.clone()) if linear.bias is not None else None
            self.weight_obs = MinMaxObserver()
            self.act_obs    = MinMaxObserver()
        def forward(self, x):
            self.act_obs(x)
            a_sc, a_zp, a_qmin, a_qmax = self.act_obs.scale_zp()
            w_sc, w_zp, w_qmin, w_qmax = self.weight_obs.scale_zp()
            with torch.cuda.amp.autocast(enabled=False):
                xf = FakeQuantizeSTE.apply(x.float(),      a_sc, a_zp, a_qmin, a_qmax)
                wf = FakeQuantizeSTE.apply(self.weight.float(), w_sc, w_zp, w_qmin, w_qmax)
                self.weight_obs(self.weight.detach())
            b = self.bias.float() if self.bias is not None else None
            return nn.functional.linear(xf, wf, b).to(x.dtype)

    return QATLinear

def replace_with_qat(model, skip_prefixes, bits, symmetric, ema_decay):
    import torch.nn as nn
    QATLinear = _build_qat_modules(bits, symmetric, ema_decay)
    replaced = 0
    def _recurse(module, prefix=""):
        nonlocal replaced
        for name, child in list(module.named_children()):
            full = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear) and not any(full.startswith(s) for s in skip_prefixes):
                setattr(module, name, QATLinear(child))
                replaced += 1
            else:
                _recurse(child, full)
    _recurse(model)
    return replaced, QATLinear


# ═══════════════════════════════════════════════════════════════════
# §7  STAGE 3 — Quantization-Aware Training  (with optional KD)
# ═══════════════════════════════════════════════════════════════════
def run_qat(cfg: dict):
    print("\n" + "═"*60)
    print("  ⚡ STAGE 3: Quantization-Aware Training (QAT)")
    if cfg["kd_alpha"] > 0:
        print(f"  🎓  KD distillation ENABLED  (α = {cfg['kd_alpha']})")
        if cfg.get("hidden_target", 1024) < 1024:
            print("  ⚠️  KD distillation with hidden_size mismatch is approximate; "
                  "layers with mismatched hidden_size are skipped from the KD sum. "
                  "Consider --kd_alpha 0 when using --hidden_target.")
    print("═"*60)

    import torch
    from torch.cuda.amp import GradScaler
    from transformers import SeamlessM4TFeatureExtractor

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ft_ckpt  = cfg["ft_ckpt"]
    ckpt_dir = cfg["qat_ckpt"]
    exp_dir  = cfg["exp_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    wandb_logger = WandbLogger(cfg, "qat")
    wandb_logger.start()

    print(f"  Device: {device}")

    with open(ft_ckpt / "vocab.json", encoding="utf-8") as f:
        vocab = json.load(f)
    fe = SeamlessM4TFeatureExtractor.from_pretrained(str(ft_ckpt))

    ft_weights = ft_ckpt / "model_best.safetensors"
    if not ft_weights.exists():
        ft_weights = ft_ckpt / "model_latest.safetensors"

    disk_check("before loading QAT model")
    print(f"\n📦 Loading fine-tuned model from {ft_weights.name}...")
    model, _ = build_model(ft_ckpt / "config.json", ft_weights, device)

    # Freeze feature extractor
    for name, p in model.named_parameters():
        if "feature_extractor" in name or "feature_projection" in name:
            p.requires_grad_(False)

    # Apply QAT
    skip = (
        "wav2vec2_bert.feature_extractor",
        "wav2vec2_bert.feature_projection",
        "wav2vec2_bert.adapter",
    )
    n_replaced, QATLinear = replace_with_qat(
        model, skip, cfg["qat_bits"], symmetric=True, ema_decay=0.999)
    print(f"  Replaced {n_replaced} Linear → QATLinear")
    wandb_logger.log_summary({
        "qat/num_replaced": int(n_replaced),
        "qat/bits":         int(cfg["qat_bits"]),
        "qat/symmetric":    True,
    })

    if not cfg["no_grad_ckpt"]:
        model.wav2vec2_bert.encoder.gradient_checkpointing = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,}")

    # ── KD: load frozen teacher ────────────────────────────────────
    teacher = None
    if cfg["kd_alpha"] > 0:
        teacher = load_teacher(cfg, device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["qat_lr"], weight_decay=0.01,
    )
    scaler = GradScaler() if device.type == "cuda" else None

    start_ep, log = load_checkpoint(ckpt_dir, model, optimizer, device,
                                    scaler=scaler, wandb_logger=wandb_logger)

    # ── Learning-rate scheduler (per-epoch; None when constant lr) ──
    scheduler = build_lr_scheduler(
        optimizer, cfg, total_epochs=cfg["qat_epochs"],
        last_completed_epoch=start_ep)
    if scheduler is not None:
        print(f"  📉 LR schedule: {cfg['lr_schedule']}  "
              f"(base={cfg['qat_lr']:.2e}  min={cfg['lr_min']:.2e}  "
              f"warmup={cfg['lr_warmup_epochs']} ep)  "
              f"start lr={optimizer.param_groups[0]['lr']:.2e}")
    else:
        print(f"  📉 LR schedule: constant ({cfg['qat_lr']:.2e})")

    if start_ep >= cfg["qat_epochs"]:
        print(f"  ✅ QAT already complete ({start_ep} epochs)")
    else:
        augment = build_augmentation()
        samples = stream_samples(cfg["moshaf"], cfg["qat_samples"])

        # Stage static side-cars (from the FT checkpoint dir) into the QAT
        # checkpoint dir so each per-epoch W&B checkpoint is self-contained.
        for fname in ["config.json", "vocab.json", "preprocessor_config.json"]:
            src = ft_ckpt / fname
            if src.exists() and not (ckpt_dir / fname).exists():
                shutil.copy2(str(src), str(ckpt_dir / fname))

        qat_best_loss = float("inf")
        for epoch in range(start_ep + 1, cfg["qat_epochs"] + 1):
            print(f"\n  ── QAT Epoch {epoch}/{cfg['qat_epochs']} ──")
            disk_check(f"QAT epoch {epoch}")
            t0 = time.time()
            avg_loss, avg_kd = train_one_epoch(
                model, fe, samples, vocab, cfg["levels"], cfg["loss_weights"],
                optimizer, device,
                batch_size=cfg["ft_batch"], grad_accum_steps=cfg["ft_accum"],
                scaler=scaler, augment_fn=augment, max_duration=cfg["qat_max_dur"],
                epoch_num=epoch,
                # ── KD ──────────────────────────────────────────────
                teacher_model=teacher,
                kd_alpha=cfg["kd_alpha"],
                # ── W&B ────────────────────────────────────────────
                wandb_logger=wandb_logger,
            )
            elapsed = time.time() - t0
            kd_str  = f"  kd_loss={avg_kd:.4f}" if cfg["kd_alpha"] > 0 else ""
            print(f"  ✅ QAT Epoch {epoch}  loss={avg_loss:.4f}{kd_str}  ({elapsed:.0f}s)")
            log.append({"epoch": epoch, "loss": avg_loss, "kd_loss": avg_kd,
                        "time_s": round(elapsed, 1)})
            is_best_qat = avg_loss < qat_best_loss
            if is_best_qat:
                qat_best_loss = avg_loss
            save_checkpoint(model, optimizer, epoch, log, ckpt_dir,
                            best=is_best_qat, scaler=scaler)

            # Advance the LR schedule one epoch (no-op when constant/None).
            if scheduler is not None:
                scheduler.step()
                print(f"  📉 next-epoch lr = {optimizer.param_groups[0]['lr']:.3e}")

            epoch_metrics = {
                "loss": avg_loss, "time_s": elapsed, "epoch_num": epoch,
                "best_loss": qat_best_loss, "is_best": int(is_best_qat),
            }
            if cfg["kd_alpha"] > 0:
                epoch_metrics["kd_loss"] = avg_kd
            wandb_logger.log_epoch(epoch_metrics)

            # ── Full resumable checkpoint → W&B ────────────────────
            # Same policy as fine-tuning: under --wandb_ckpt all push every
            # epoch, under 'best' push only on improvement.  Includes the
            # QAT observer state via the model buffers in save_checkpoint.
            if cfg.get("wandb_ckpt") == "all" or (
                    cfg.get("wandb_ckpt") == "best" and is_best_qat):
                wandb_logger.log_checkpoint(
                    ckpt_dir, epoch,
                    metadata={"loss": float(avg_loss),
                              "best_loss": float(qat_best_loss),
                              "is_best": bool(is_best_qat)})

            # ── Observer min/max stats (QATLinear weight/activation ranges) ─
            try:
                obs_summary = {}
                w_mins, w_maxs, a_mins, a_maxs = [], [], [], []
                for n, m in model.named_modules():
                    if type(m).__name__ == "QATLinear":
                        w_mins.append(float(m.weight_obs.ema_min))
                        w_maxs.append(float(m.weight_obs.ema_max))
                        a_mins.append(float(m.act_obs.ema_min))
                        a_maxs.append(float(m.act_obs.ema_max))
                if w_mins:
                    obs_summary[f"qat/epoch_{epoch:02d}/w_min_mean"] = sum(w_mins)/len(w_mins)
                    obs_summary[f"qat/epoch_{epoch:02d}/w_max_mean"] = sum(w_maxs)/len(w_maxs)
                    obs_summary[f"qat/epoch_{epoch:02d}/a_min_mean"] = sum(a_mins)/len(a_mins)
                    obs_summary[f"qat/epoch_{epoch:02d}/a_max_mean"] = sum(a_maxs)/len(a_maxs)
                    wandb_logger.log_summary(obs_summary)
                    wandb_logger.log_histogram(f"qat/epoch_{epoch:02d}/w_range",
                                               [b - a for a, b in zip(w_mins, w_maxs)])
                    wandb_logger.log_histogram(f"qat/epoch_{epoch:02d}/a_range",
                                               [b - a for a, b in zip(a_mins, a_maxs)])
            except Exception as _e:
                print(f"[wandb] observer stats logging skipped: {_e}")

    # ── Free teacher before export (no longer needed) ──────────────
    if teacher is not None:
        del teacher; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # ── Export INT8 quantized model ───────────────────────────────
    print("\n📤 Exporting INT8 quantized model...")
    import torch.nn as nn

    class QuantizedLinear(nn.Module):
        """Stores actual INT8 weights for on-device inference."""
        def __init__(self, ql):
            super().__init__()
            w_sc, w_zp, w_qmin, w_qmax = ql.weight_obs.scale_zp()
            a_sc, a_zp, a_qmin, a_qmax = ql.act_obs.scale_zp()
            w_int = torch.clamp(torch.round(ql.weight.data / w_sc + w_zp), w_qmin, w_qmax).to(torch.int8)
            self.register_buffer("weight_int8",     w_int)
            self.register_buffer("weight_scale",    w_sc)
            self.register_buffer("weight_zp",       w_zp)
            self.register_buffer("act_scale",       a_sc)
            self.register_buffer("act_zp",          a_zp)
            self.register_buffer("act_qmin",        torch.tensor(float(a_qmin)))
            self.register_buffer("act_qmax",        torch.tensor(float(a_qmax)))
            self.bias = ql.bias
            self.in_features, self.out_features = ql.in_features, ql.out_features
        def forward(self, x):
            xq  = torch.clamp(torch.round(x / self.act_scale + self.act_zp),
                               self.act_qmin.item(), self.act_qmax.item())
            xdq = (xq - self.act_zp) * self.act_scale
            wdq = (self.weight_int8.float() - self.weight_zp) * self.weight_scale
            return nn.functional.linear(xdq, wdq, self.bias)

    def convert_to_int8(module):
        for name, child in list(module.named_children()):
            if type(child).__name__ == "QATLinear":
                setattr(module, name, QuantizedLinear(child))
            else:
                convert_to_int8(child)

    model.eval()
    convert_to_int8(model)

    export_dir = cfg["export_dir"]
    export_dir.mkdir(parents=True, exist_ok=True)

    from safetensors.torch import save_file
    q_sd = {n: p.data for n, p in model.named_parameters()}
    q_sd.update({n: b for n, b in model.named_buffers()})
    disk_check("before saving INT8 export")
    save_file(q_sd, str(export_dir / "model_quantized.safetensors"))

    for fname in ["config.json", "vocab.json", "preprocessor_config.json"]:
        src = ft_ckpt / fname
        if src.exists():
            shutil.copy2(str(src), str(export_dir / fname))

    size_mb = (export_dir / "model_quantized.safetensors").stat().st_size / 1e6
    print(f"  ✅ model_quantized.safetensors  ({size_mb:.0f} MB)")

    wandb_logger.log_summary({
        "export/size_mb":      round(size_mb, 2),
        "export/path":         str(export_dir / "model_quantized.safetensors"),
        "export/qat_bits":     int(cfg["qat_bits"]),
    })
    if cfg.get("wandb_ckpt") in ("best", "all"):
        wandb_logger.log_artifact(
            export_dir / "model_quantized.safetensors",
            name=f"{cfg['name']}-int8",
            art_type="model",
            metadata={"size_mb": round(size_mb, 2),
                      "qat_bits": int(cfg["qat_bits"])},
            gate=False,
        )

    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    mark_done(exp_dir, "qat")
    wandb_logger.finish("success")
    print("\n✅ QAT + INT8 export complete")


# ═══════════════════════════════════════════════════════════════════
# §8  Entry point
# ═══════════════════════════════════════════════════════════════════
def main(args=None):
    cfg = make_config(args)
    cfg["exp_dir"].mkdir(parents=True, exist_ok=True)

    print("\n" + "╔" + "═"*58 + "╗")
    print(f"  🚀  Mualem Pipeline — experiment: {cfg['name']}")
    print(f"  Stages requested: {cfg['stages']}")
    print(f"  Working dir:      {cfg['exp_dir']}")
    if cfg["kd_alpha"] > 0 or cfg["kd_pruning"]:
        print(f"  🎓  KD alpha={cfg['kd_alpha']}  "
              f"kd_pruning={cfg['kd_pruning']}  "
              f"calib_batches={cfg['kd_calib_batches']}")
    if cfg["layer_target"] < 24 or cfg["hidden_target"] < 1024:
        print(f"  ✂️  Extra pruning: layers={cfg['layer_target']}  "
              f"hidden={cfg['hidden_target']}  "
              f"layer_score={cfg['layer_score']}")
    print("╚" + "═"*58 + "╝\n")

    # Save experiment config
    summary_path = cfg["exp_dir"] / "experiment_config.json"
    serialisable = {k: str(v) if isinstance(v, Path) else v
                    for k, v in cfg.items() if k not in ("loss_weights", "levels")}
    summary_path.write_text(json.dumps(serialisable, indent=2))

    STAGE_FNS = {
        "prune":    run_pruning,
        "finetune": run_finetune,
        "qat":      run_qat,
    }

    for stage in ["prune", "finetune", "qat"]:
        if stage not in cfg["stages"]:
            print(f"  ⏭️  Skipping {stage} (not requested)")
            continue
        if is_done(cfg["exp_dir"], stage):
            print(f"  ✅ {stage.upper()} already done — skipping")
            continue
        STAGE_FNS[stage](cfg)

    print("\n" + "╔" + "═"*58 + "╗")
    print(f"  🎉  Pipeline complete!  [{cfg['name']}]")
    print(f"  INT8 model: {cfg['export_dir'] / 'model_quantized.safetensors'}")
    print("╚" + "═"*58 + "╝\n")
    disk_check("pipeline complete")


if __name__ == "__main__":
    main()