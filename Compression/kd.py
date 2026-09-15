"""Knowledge-distillation helpers: teacher loading and L_distill.

Both helpers support the KD pipeline from Li et al. 2510.04213.

- ``load_teacher`` materialises the frozen full-size teacher (fp16 on GPU)
  from the cached weights written during pruning, falling back to a fresh
  Hugging Face download when the cache is empty.

- ``compute_kd_loss`` is equation (5) of the paper, averaged over aligned
  encoder layers. It tolerates teacher activations stored on CPU so the
  caller can avoid co-resident full-model activations on the GPU.
"""

from .model import build_model
from .runtime import disk_check


def load_teacher(cfg: dict, device):
    """Return the frozen teacher, downloading it if not cached.

    Cast to fp16 on CUDA (halves VRAM), parameters frozen, ``eval()``.
    """
    from huggingface_hub import hf_hub_download

    teacher_dir = cfg["teacher_dir"]

    if not (teacher_dir / "model.safetensors").exists():
        print(f"  teacher not cached -- downloading from {cfg['hf_repo']}")
        teacher_dir.mkdir(parents=True, exist_ok=True)
        disk_check("before teacher download")
        for fname in ["config.json", "model.safetensors", "vocab.json",
                      "preprocessor_config.json", "added_tokens.json",
                      "special_tokens_map.json", "tokenizer_config.json"]:
            hf_hub_download(repo_id=cfg["hf_repo"], filename=fname,
                            local_dir=str(teacher_dir))

    print("  loading teacher (frozen)")
    disk_check("before loading teacher")
    teacher, _ = build_model(teacher_dir / "config.json",
                              teacher_dir / "model.safetensors", device)

    if device.type == "cuda":
        teacher = teacher.half()

    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()

    size_m = sum(p.numel() for p in teacher.parameters()) / 1e6
    vram_gb = size_m * (2 if device.type == "cuda" else 4) / 1024
    print(f"  teacher loaded: {size_m:.0f}M params, ~{vram_gb:.1f} GB VRAM, frozen")
    return teacher


def compute_kd_loss(teacher_hidden, student_hidden, device,
                    return_per_layer=False):
    """L_distill from eq. (5) of 2510.04213, averaged over aligned layers.

        L_distill = mean_l [ L1(h_t^l, h_s^l) - cosine(h_t^l, h_s^l) ]

    Teacher activations may live on CPU; each layer is moved to ``device``
    for its term and dropped before the next, so the GPU only holds one
    teacher layer's worth of activations at a time.

    Layers where teacher and student hidden sizes differ (i.e.
    ``--hidden_target`` was applied) are skipped from the sum. When every
    layer is mismatched the loss is zero and the caller should consider
    disabling KD for that run.
    """
    import torch
    import torch.nn.functional as F

    n_layers = min(len(teacher_hidden), len(student_hidden))
    loss = torch.zeros(1, device=device, dtype=torch.float32).squeeze()
    n_used = 0
    per_layer_cos = [] if return_per_layer else None

    for h_t_raw, h_s in zip(teacher_hidden[:n_layers], student_hidden[:n_layers]):
        h_t = h_t_raw.to(device=device, dtype=torch.float32)
        h_s = h_s.float()

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

        del h_t

    if n_used == 0:
        if return_per_layer:
            return loss, {"cos_per_layer": per_layer_cos}
        return loss
    out = loss / n_used
    if return_per_layer:
        return out, {"cos_per_layer": per_layer_cos}
    return out
