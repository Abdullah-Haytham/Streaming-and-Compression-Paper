"""Training loop shared by fine-tuning and QAT.

Runs CTC against the multi-level vocab, optionally blending in KD distillation
when a teacher is supplied with ``kd_alpha > 0``. W&B logging is per-optimizer
step (after grad-accum) and is opt-in via ``wandb_logger`` -- a disabled logger
short-circuits to a no-op.
"""

import time

from .data import decode_audio, encode_labels
from .kd import compute_kd_loss
from .runtime import gpu_mem_mb, gpu_util_pct


def train_one_epoch(
    model, feature_extractor, dataset_stream, vocab, levels, loss_weights,
    optimizer, device, batch_size, grad_accum_steps,
    scaler=None, augment_fn=None, max_duration=15.0, epoch_num=1,
    teacher_model=None,
    kd_alpha=0.0,
    wandb_logger=None,
):
    """One epoch of CTC training, optionally blended with KD.

        L_total = L_CTC + kd_alpha * L_distill

    The teacher runs in fp16 with no grad and its activations are moved to
    CPU immediately after the forward; ``compute_kd_loss`` brings them back
    one layer at a time so VRAM never holds two full activation graphs.
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

    # Per-optimizer-step accumulators, reset after each .step().
    step_loss_sum  = 0.0
    step_kd_sum    = 0.0
    step_level_sum = {lv: 0.0 for lv in levels}
    step_level_cnt = {lv: 0   for lv in levels}
    step_samples   = 0
    step_kd_cos_sum = None
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

        # Teacher forward (no-grad, fp16). Hidden states are moved to CPU
        # immediately so they don't compete with the student's activations.
        t_hidden = None
        if use_kd:
            t_fwd_t0 = time.time() if log_wb else 0.0
            t_mem_before = gpu_mem_mb() if log_wb else 0.0
            with torch.no_grad():
                _, t_hidden_gpu = teacher_model(
                    inp.half() if device.type == "cuda" else inp,
                    attention_mask=mask,
                    return_hidden_states=True,
                )
                if log_wb:
                    t_mem_peak = gpu_mem_mb()
                    step_kd_t_mem = max(step_kd_t_mem,
                                        t_mem_peak - t_mem_before)
                t_hidden = tuple(h.detach().cpu() for h in t_hidden_gpu)
                del t_hidden_gpu
            if log_wb:
                step_kd_t_time += (time.time() - t_fwd_t0)

        with autocast(enabled=use_amp, dtype=torch.float16):
            if use_kd:
                logits, s_hidden = model(inp, attention_mask=mask,
                                         return_hidden_states=True)
            else:
                logits = model(inp, attention_mask=mask)

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
                        if c == c:   # NaN guard
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
                    "perf/gpu_mem_mb":    gpu_mem_mb(),
                    "train/epoch":        epoch_num,
                    "train/batches_seen": batches,
                }
                util = gpu_util_pct()
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
