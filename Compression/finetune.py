"""Fine-tuning stage: recover task accuracy after structural pruning.

Encoder is gradient-checkpointed by default. AdamW with weight-decay 0.01
trains everything except the (frozen) feature extractor + feature projection.
The per-epoch LR scheduler decays from ``--ft_lr`` toward ``--lr_min``. When
``--kd_alpha > 0`` a frozen teacher joins the loop and ``train_one_epoch``
blends L_distill into the total loss.
"""

import gc
import json
import shutil
import time

from .checkpoint import load_checkpoint, save_checkpoint
from .data import build_augmentation, stream_samples
from .kd import load_teacher
from .lr_schedule import build_lr_scheduler
from .model import build_model
from .runtime import disk_check, mark_done
from .training import train_one_epoch
from .wandb_logger import WandbLogger


def run_finetune(cfg: dict):
    print("\nStage: fine-tune")
    if cfg["kd_alpha"] > 0:
        print(f"  KD enabled (alpha={cfg['kd_alpha']})")
        if cfg.get("hidden_target", 1024) < 1024:
            # Layers with mismatched hidden_size are skipped from the KD sum.
            print("  KD with hidden_size mismatch is approximate; "
                  "consider --kd_alpha 0 with --hidden_target.")

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

    print(f"  device: {device}")
    if device.type == "cuda":
        print(f"  gpu: {torch.cuda.get_device_name(0)}")

    with open(model_dir / "vocab.json", encoding="utf-8") as f:
        vocab = json.load(f)
    fe = SeamlessM4TFeatureExtractor.from_pretrained(str(model_dir))

    disk_check("before loading FT model")
    print("\nloading pruned model")
    model, _ = build_model(model_dir / "config.json", model_dir / "model.safetensors", device)

    # Freeze the audio front-end. It's already well-trained and changes here
    # would dwarf the recovery signal.
    for name, p in model.named_parameters():
        if "feature_extractor" in name or "feature_projection" in name:
            p.requires_grad_(False)

    if not cfg["no_grad_ckpt"]:
        model.wav2vec2_bert.encoder.gradient_checkpointing = True
        print("  gradient checkpointing: on (--no_grad_ckpt to disable)")
    else:
        print("  gradient checkpointing: off")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  trainable: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")
    wandb_logger.log_summary({
        "ft/params_trainable": int(trainable),
        "ft/params_total":     int(total),
        "ft/trainable_pct":    trainable / max(total, 1) * 100,
    })

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

    # Built AFTER load_checkpoint so it fast-forwards to the resumed epoch.
    # No scheduler state is persisted -- LambdaLR replays its lambda from
    # start_ep deterministically.
    scheduler = build_lr_scheduler(
        optimizer, cfg, total_epochs=cfg["ft_epochs"],
        last_completed_epoch=start_ep)
    if scheduler is not None:
        print(f"  lr schedule: {cfg['lr_schedule']} "
              f"(base={cfg['ft_lr']:.2e} min={cfg['lr_min']:.2e} "
              f"warmup={cfg['lr_warmup_epochs']} ep) "
              f"start lr={optimizer.param_groups[0]['lr']:.2e}")
    else:
        print(f"  lr schedule: constant ({cfg['ft_lr']:.2e})")

    if start_ep >= cfg["ft_epochs"]:
        print(f"  fine-tuning already complete ({start_ep} epochs)")
        mark_done(exp_dir, "finetune")
        wandb_logger.log_summary({"ft/status": "already_complete"})
        wandb_logger.finish("success")
        return

    augment = build_augmentation()
    samples = stream_samples(cfg["moshaf"], cfg["ft_samples"])

    # Stage static side-cars into the checkpoint dir up front so every
    # per-epoch W&B checkpoint artifact is self-contained: model loads need
    # config.json / vocab.json on a clean-machine resume. Re-copied after the
    # loop too (harmless) to cover the QAT-stage hand-off.
    for fname in ["config.json", "vocab.json", "preprocessor_config.json"]:
        src = model_dir / fname
        if src.exists() and not (ckpt_dir / fname).exists():
            shutil.copy2(str(src), str(ckpt_dir / fname))

    best_loss = float("inf")

    for epoch in range(start_ep + 1, cfg["ft_epochs"] + 1):
        print(f"\n  epoch {epoch}/{cfg['ft_epochs']}")
        disk_check(f"epoch {epoch} start")
        t0 = time.time()
        avg_loss, avg_kd = train_one_epoch(
            model, fe, samples, vocab, cfg["levels"], cfg["loss_weights"],
            optimizer, device,
            batch_size=cfg["ft_batch"], grad_accum_steps=cfg["ft_accum"],
            scaler=scaler, augment_fn=augment, max_duration=cfg["ft_max_dur"],
            epoch_num=epoch,
            teacher_model=teacher,
            kd_alpha=cfg["kd_alpha"],
            wandb_logger=wandb_logger,
        )
        elapsed = time.time() - t0
        kd_str  = f"  kd_loss={avg_kd:.4f}" if cfg["kd_alpha"] > 0 else ""
        print(f"  epoch {epoch}  loss={avg_loss:.4f}{kd_str}  ({elapsed:.0f}s)")
        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss
        log.append({"epoch": epoch, "loss": avg_loss, "kd_loss": avg_kd,
                    "time_s": round(elapsed, 1)})
        save_checkpoint(model, optimizer, epoch, log, ckpt_dir, best=is_best,
                        scaler=scaler)

        if scheduler is not None:
            scheduler.step()
            print(f"  next-epoch lr = {optimizer.param_groups[0]['lr']:.3e}")

        epoch_metrics = {
            "loss": avg_loss, "time_s": elapsed, "epoch_num": epoch,
            "best_loss": best_loss, "is_best": int(is_best),
        }
        if cfg["kd_alpha"] > 0:
            epoch_metrics["kd_loss"] = avg_kd
        wandb_logger.log_epoch(epoch_metrics)

        # Push the full resumable checkpoint to W&B. Under 'all' we push every
        # epoch; under 'best' we push only when this epoch improved so the
        # 'latest'-aliased restore point is always a good model.
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
                    gate=False,
                )

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
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    mark_done(exp_dir, "finetune")
    wandb_logger.finish("success")
    print("\nfine-tuning complete")
