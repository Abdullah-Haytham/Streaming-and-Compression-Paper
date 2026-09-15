"""W&B logging wrapper.

``WandbLogger`` is a thin shim around ``wandb``: every method is a no-op when
tracking is disabled or when ``wandb.init`` fails. Stages talk to one logger
for their lifetime. ``wandb`` is imported lazily in ``start()``.
"""

import os
import sys
from pathlib import Path


def _detect_env() -> str:
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ or os.path.exists("/kaggle"):
        return "kaggle"
    if (os.environ.get("LIGHTNING_CLOUD_PROJECT_ID")
            or "LIGHTNING_CLOUD_URL" in os.environ
            or "LIGHTNING_APP_STATE" in os.environ):
        return "lightning"
    if "COLAB_GPU" in os.environ or "google.colab" in sys.modules:
        return "colab"
    return "local"


def _derive_tags(cfg: dict, stage: str, env: str) -> list:
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
    """One instance per stage ('prune' | 'ft' | 'qat')."""

    # Files that make a checkpoint dir fully resumable on a clean machine.
    # Mirrors what save_checkpoint() writes and load_checkpoint() reads.
    CHECKPOINT_FILES = (
        "model_latest.safetensors",
        "model_best.safetensors",
        "optimizer_latest.pt",
        "scaler_latest.pt",
        "training_state.json",
        "config.json",
        "vocab.json",
        "preprocessor_config.json",
    )

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
            import wandb
        except ImportError:
            print("[wandb] not installed; disabling. Install: pip install wandb")
            self.enabled = False
            return
        try:
            key = os.environ.get("WANDB_API_KEY")
            if key:
                wandb.login(key=key, relogin=False)
            else:
                print("[wandb] WANDB_API_KEY not set; using cached creds if any.")
            env = _detect_env()
            tags = _derive_tags(self.cfg, self.stage, env)
            run_name = (self.cfg.get("wandb_run_name")
                        or f"{self.cfg['name']}-{self.stage}")
            group = self.cfg.get("wandb_group") or self.cfg["name"]
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
            print(f"[wandb] enabled: project={self.cfg.get('wandb_project')} "
                  f"run={run_name} group={group} env={env}")
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

    def _ckpt_artifact_name(self) -> str:
        return f"{self.cfg['name']}-{self.stage}-ckpt"

    def log_checkpoint(self, ckpt_dir, epoch: int, metadata: dict = None):
        """Upload the whole checkpoint dir as one resumable artifact.

        Gated by --wandb_ckpt: uploaded under 'all' (every epoch) or when the
        caller decided this is a 'best' epoch. Each call creates a new
        version; the freshest is aliased 'latest' so resume can grab it.
        """
        if not self.active:
            return
        if self.cfg.get("wandb_ckpt", "best") == "none":
            return
        ckpt_dir = Path(ckpt_dir)
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
                print("[wandb] log_checkpoint: nothing to upload")
                return
            self.run.log_artifact(art, aliases=["latest", f"epoch-{epoch}"])
            print(f"  checkpoint uploaded to W&B ({len(added)} files, epoch {epoch})")
        except Exception as e:
            print(f"[wandb] log_checkpoint failed: {e}")

    def restore_checkpoint(self, ckpt_dir) -> bool:
        """Download the latest stage checkpoint into ``ckpt_dir`` only if no
        local checkpoint already exists. Local wins.
        """
        if not self.active:
            return False
        if self.cfg.get("wandb_ckpt", "best") == "none":
            return False
        ckpt_dir = Path(ckpt_dir)
        if (ckpt_dir / "training_state.json").exists():
            return False
        try:
            import wandb  # noqa: F401 -- imported for side-effects on self.run
            ref = f"{self._ckpt_artifact_name()}:latest"
            entity = self.cfg.get("wandb_entity") or self.run.entity
            project = self.cfg.get("wandb_project") or "mualem-pipeline"
            qualified = f"{entity}/{project}/{ref}" if entity else f"{project}/{ref}"
            print(f"  no local checkpoint -- trying W&B artifact {qualified}")
            try:
                art = self.run.use_artifact(qualified, type="training-checkpoint")
            except Exception:
                art = self.run.use_artifact(ref, type="training-checkpoint")
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            art.download(root=str(ckpt_dir))
            restored = (ckpt_dir / "training_state.json").exists()
            if restored:
                meta = art.metadata or {}
                print(f"  restored checkpoint from W&B (epoch {meta.get('epoch', '?')})")
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
