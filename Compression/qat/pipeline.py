"""QAT stage: recover task accuracy while learning int8 quantisation params,
then export the model with real int8 weights to ``cfg["export_dir"]``.

Steps:
  1. load the best (or latest) FT weights
  2. swap every non-skip ``nn.Linear`` for a ``QATLinear``
  3. optionally load a frozen teacher for KD
  4. train ``cfg["qat_epochs"]`` epochs with the shared training loop
  5. convert every ``QATLinear`` to ``QuantizedLinear`` and write the int8 export
"""

import gc
import json
import shutil
import time

from ..checkpoint import load_checkpoint, save_checkpoint
from ..data import build_augmentation, stream_samples
from ..kd import load_teacher
from ..lr_schedule import build_lr_scheduler
from ..model import build_model
from ..runtime import disk_check, mark_done
from ..training import train_one_epoch
from ..wandb_logger import WandbLogger
from .modules import QATLinear, convert_to_int8, replace_with_qat


def run_qat(cfg: dict):
    print("\nStage: QAT")
    if cfg["kd_alpha"] > 0:
        print(f"  KD enabled (alpha={cfg['kd_alpha']})")
        if cfg.get("hidden_target", 1024) < 1024:
            # Layers with mismatched hidden_size are skipped from the KD sum.
            print("  KD with hidden_size mismatch is approximate; "
                  "consider --kd_alpha 0 with --hidden_target.")

    import torch
    from safetensors.torch import save_file
    from torch.cuda.amp import GradScaler
    from transformers import SeamlessM4TFeatureExtractor

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ft_ckpt  = cfg["ft_ckpt"]
    ckpt_dir = cfg["qat_ckpt"]
    exp_dir  = cfg["exp_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    wandb_logger = WandbLogger(cfg, "qat")
    wandb_logger.start()

    print(f"  device: {device}")

    with open(ft_ckpt / "vocab.json", encoding="utf-8") as f:
        vocab = json.load(f)
    fe = SeamlessM4TFeatureExtractor.from_pretrained(str(ft_ckpt))

    ft_weights = ft_ckpt / "model_best.safetensors"
    if not ft_weights.exists():
        ft_weights = ft_ckpt / "model_latest.safetensors"

    disk_check("before loading QAT model")
    print(f"\nloading fine-tuned model from {ft_weights.name}")
    model, _ = build_model(ft_ckpt / "config.json", ft_weights, device)

    for name, p in model.named_parameters():
        if "feature_extractor" in name or "feature_projection" in name:
            p.requires_grad_(False)

    # Quantise everything except the audio front-end and the adapter -- those
    # are sensitive enough that int8 hurts accuracy more than it helps.
    skip = (
        "wav2vec2_bert.feature_extractor",
        "wav2vec2_bert.feature_projection",
        "wav2vec2_bert.adapter",
    )
    n_replaced = replace_with_qat(model, skip, cfg["qat_bits"],
                                   symmetric=True, ema_decay=0.999)
    print(f"  replaced {n_replaced} Linear -> QATLinear")
    wandb_logger.log_summary({
        "qat/num_replaced": int(n_replaced),
        "qat/bits":         int(cfg["qat_bits"]),
        "qat/symmetric":    True,
    })

    if not cfg["no_grad_ckpt"]:
        model.wav2vec2_bert.encoder.gradient_checkpointing = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  trainable: {trainable:,} / {total:,}")

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

    scheduler = build_lr_scheduler(
        optimizer, cfg, total_epochs=cfg["qat_epochs"],
        last_completed_epoch=start_ep)
    if scheduler is not None:
        print(f"  lr schedule: {cfg['lr_schedule']} "
              f"(base={cfg['qat_lr']:.2e} min={cfg['lr_min']:.2e} "
              f"warmup={cfg['lr_warmup_epochs']} ep) "
              f"start lr={optimizer.param_groups[0]['lr']:.2e}")
    else:
        print(f"  lr schedule: constant ({cfg['qat_lr']:.2e})")

    if start_ep >= cfg["qat_epochs"]:
        print(f"  QAT already complete ({start_ep} epochs)")
    else:
        augment = build_augmentation()
        samples = stream_samples(cfg["moshaf"], cfg["qat_samples"])

        for fname in ["config.json", "vocab.json", "preprocessor_config.json"]:
            src = ft_ckpt / fname
            if src.exists() and not (ckpt_dir / fname).exists():
                shutil.copy2(str(src), str(ckpt_dir / fname))

        qat_best_loss = float("inf")
        for epoch in range(start_ep + 1, cfg["qat_epochs"] + 1):
            print(f"\n  QAT epoch {epoch}/{cfg['qat_epochs']}")
            disk_check(f"QAT epoch {epoch}")
            t0 = time.time()
            avg_loss, avg_kd = train_one_epoch(
                model, fe, samples, vocab, cfg["levels"], cfg["loss_weights"],
                optimizer, device,
                batch_size=cfg["ft_batch"], grad_accum_steps=cfg["ft_accum"],
                scaler=scaler, augment_fn=augment, max_duration=cfg["qat_max_dur"],
                epoch_num=epoch,
                teacher_model=teacher,
                kd_alpha=cfg["kd_alpha"],
                wandb_logger=wandb_logger,
            )
            elapsed = time.time() - t0
            kd_str  = f"  kd_loss={avg_kd:.4f}" if cfg["kd_alpha"] > 0 else ""
            print(f"  QAT epoch {epoch}  loss={avg_loss:.4f}{kd_str}  ({elapsed:.0f}s)")
            log.append({"epoch": epoch, "loss": avg_loss, "kd_loss": avg_kd,
                        "time_s": round(elapsed, 1)})
            is_best_qat = avg_loss < qat_best_loss
            if is_best_qat:
                qat_best_loss = avg_loss
            save_checkpoint(model, optimizer, epoch, log, ckpt_dir,
                            best=is_best_qat, scaler=scaler)

            if scheduler is not None:
                scheduler.step()
                print(f"  next-epoch lr = {optimizer.param_groups[0]['lr']:.3e}")

            epoch_metrics = {
                "loss": avg_loss, "time_s": elapsed, "epoch_num": epoch,
                "best_loss": qat_best_loss, "is_best": int(is_best_qat),
            }
            if cfg["kd_alpha"] > 0:
                epoch_metrics["kd_loss"] = avg_kd
            wandb_logger.log_epoch(epoch_metrics)

            if cfg.get("wandb_ckpt") == "all" or (
                    cfg.get("wandb_ckpt") == "best" and is_best_qat):
                wandb_logger.log_checkpoint(
                    ckpt_dir, epoch,
                    metadata={"loss": float(avg_loss),
                              "best_loss": float(qat_best_loss),
                              "is_best": bool(is_best_qat)})

            # Observer min/max stats (QATLinear weight/activation ranges).
            try:
                obs_summary = {}
                w_mins, w_maxs, a_mins, a_maxs = [], [], [], []
                for n, m in model.named_modules():
                    if isinstance(m, QATLinear):
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

    if teacher is not None:
        del teacher
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nexporting INT8 model")
    model.eval()
    convert_to_int8(model)

    export_dir = cfg["export_dir"]
    export_dir.mkdir(parents=True, exist_ok=True)

    q_sd = {n: p.data for n, p in model.named_parameters()}
    q_sd.update({n: b for n, b in model.named_buffers()})
    disk_check("before saving INT8 export")
    save_file(q_sd, str(export_dir / "model_quantized.safetensors"))

    for fname in ["config.json", "vocab.json", "preprocessor_config.json"]:
        src = ft_ckpt / fname
        if src.exists():
            shutil.copy2(str(src), str(export_dir / fname))

    size_mb = (export_dir / "model_quantized.safetensors").stat().st_size / 1e6
    print(f"  model_quantized.safetensors ({size_mb:.0f} MB)")

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

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    mark_done(exp_dir, "qat")
    wandb_logger.finish("success")
    print("\nQAT + INT8 export complete")
