"""Pruning stage orchestrator.

Order:
  1. download the upstream checkpoint
  2. cache teacher weights for later KD stages (no extra download)
  3. load weights into a numpy state-dict
  4. compute KD-guided Taylor scores (when --kd_pruning)
  5. log magnitude-baseline head-score histogram (when KD scoring is off)
  6. prune attention heads
  7. prune FFN width
  8. prune encoder depth (when --layer_target < 24)
  9. prune hidden size + validate shapes (when --hidden_target < 1024)
 10. write the pruned model + summary
 11. clean up the original download
 12. mark stage done + finish W&B
"""

import gc
import json
import shutil
import time

from ..runtime import disk_check, mark_done
from ..wandb_logger import WandbLogger
from ._state import layer_prefixes_from_state
from .heads_ffn import _head_magnitude_scores, prune_ffn, prune_heads
from .hidden import prune_hidden_size, score_hidden_dims, validate_hidden_size
from .kd_scores import compute_kd_pruning_scores
from .layers import prune_layers, score_layers


# Wav2Vec2-BERT is hard-wired to 16 attention heads in the upstream config.
_DEFAULT_NUM_HEADS = 16


def _magnitude_head_score_histogram(state):
    """Flat list of per-(layer, head) magnitude scores across the whole encoder.

    Mirrors what ``prune_heads`` would compute for the magnitude scorer, but
    flattened so we can feed it straight into W&B as a histogram.
    """
    layer_prefixes = layer_prefixes_from_state(state.keys())
    if not layer_prefixes:
        return []

    q0 = state[f"{layer_prefixes[0]}.self_attn.linear_q.weight"]
    head_dim = q0.shape[0] // _DEFAULT_NUM_HEADS

    flat_scores = []
    for layer in layer_prefixes:
        per_head = _head_magnitude_scores(state, layer, head_dim,
                                          _DEFAULT_NUM_HEADS)
        # _head_magnitude_scores returns ascending by score; the original
        # histogram code iterated heads in index order, so re-sort by head idx
        # to match that ordering exactly.
        per_head.sort(key=lambda hs: hs[0])
        flat_scores.extend(score for _, score in per_head)
    return flat_scores


def run_pruning(cfg: dict):
    print("\nStage: structured pruning")
    if cfg["kd_pruning"]:
        print("  KD-guided importance scoring enabled")

    import numpy as np
    from huggingface_hub import hf_hub_download
    from safetensors.numpy import load_file, save_file

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

    print(f"\ndownloading from {HF_REPO}")
    for fname in FILES:
        dest = tmp_dir / fname
        if dest.exists():
            print(f"  {fname} (cached)")
            continue
        disk_check(f"before downloading {fname}")
        hf_hub_download(repo_id=HF_REPO, filename=fname, local_dir=str(tmp_dir))
        size = (tmp_dir / fname).stat().st_size
        print(f"  {fname} -> {size/1e6:.1f} MB")

    # Cache teacher weights now -- same download, no extra cost.
    if cfg["kd_alpha"] > 0 or cfg["kd_pruning"]:
        teacher_dir = cfg["teacher_dir"]
        if not (teacher_dir / "model.safetensors").exists():
            print("\ncaching teacher weights for later KD stages")
            teacher_dir.mkdir(parents=True, exist_ok=True)
            for fname in FILES:
                src = tmp_dir / fname
                if src.exists():
                    shutil.copy2(str(src), str(teacher_dir / fname))
            print(f"  teacher cached at {teacher_dir}")
        else:
            print("  teacher already cached")

    print("\nloading weights")
    disk_check("before loading")
    t0 = time.time()
    state = load_file(str(tmp_dir / "model.safetensors"))
    orig_params = sum(v.size for v in state.values())
    print(f"  loaded {len(state)} tensors / {orig_params/1e6:.1f}M params in {time.time()-t0:.1f}s")

    kd_head_scores = None
    kd_ffn_scores  = None
    if cfg["kd_pruning"]:
        kd_head_scores, kd_ffn_scores = compute_kd_pruning_scores(cfg, tmp_dir)
        if kd_head_scores is None:
            print("  KD pruning scoring failed -- falling back to magnitude")
        else:
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

    # Magnitude head-score histogram -- only when KD didn't already log scores.
    if kd_head_scores is None:
        try:
            head_scores_mag = _magnitude_head_score_histogram(state)
            if head_scores_mag:
                wandb_logger.log_histogram(
                    "prune/head_scores_magnitude", head_scores_mag)
        except Exception:
            pass

    t0 = time.time()
    state = prune_heads(state, HEAD_TARGET, kd_head_scores=kd_head_scores)
    state = prune_ffn(state,   FFN_TARGET,  kd_ffn_scores=kd_ffn_scores)

    LAYER_TARGET  = cfg["layer_target"]
    HIDDEN_TARGET = cfg["hidden_target"]
    if LAYER_TARGET < 24:
        layer_scores = score_layers(cfg, tmp_dir, state, cfg["layer_score"])
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

    if HIDDEN_TARGET < 1024:
        hidden_scores = score_hidden_dims(cfg, tmp_dir, state)
        try:
            wandb_logger.log_histogram(
                "prune/hidden_dim_scores",
                np.asarray(hidden_scores, dtype=float),
            )
        except Exception:
            pass
        state = prune_hidden_size(state, HIDDEN_TARGET, hidden_scores)
        validate_hidden_size(state, HIDDEN_TARGET)

    pruned_params = sum(v.size for v in state.values())
    reduction = (1 - pruned_params / orig_params) * 100
    print(f"\n{orig_params/1e6:.1f}M -> {pruned_params/1e6:.1f}M params "
          f"({reduction:.1f}% reduction) [{time.time()-t0:.1f}s]")

    with open(tmp_dir / "config.json") as f:
        cfg_json = json.load(f)
    orig_hidden = int(cfg_json.get("hidden_size", 1024))
    cfg_json["num_attention_heads"] = HEAD_TARGET
    cfg_json["intermediate_size"]   = FFN_TARGET
    cfg_json["num_hidden_layers"]   = LAYER_TARGET
    cfg_json["hidden_size"]         = HIDDEN_TARGET
    # The Wav2Vec2-BERT adapter builds its modules from `output_hidden_size`,
    # not `hidden_size`. When they were tied originally (the HF default), the
    # adapter's internal dim must follow hidden_size when we shrink it,
    # otherwise freshly-built adapter LayerNorms / Linears expect 1024 but
    # receive the smaller size -- producing "normalized_shape" errors at FT.
    if cfg_json.get("output_hidden_size", orig_hidden) == orig_hidden:
        cfg_json["output_hidden_size"] = HIDDEN_TARGET
    (model_dir / "config.json").write_text(json.dumps(cfg_json, indent=2, ensure_ascii=False))

    disk_check("before saving pruned weights")
    print("\nsaving pruned model")
    save_file(state, str(model_dir / "model.safetensors"), metadata={"pruned": "true"})

    for fname in ["vocab.json", "preprocessor_config.json", "added_tokens.json",
                  "special_tokens_map.json", "tokenizer_config.json"]:
        src = tmp_dir / fname
        if src.exists():
            shutil.copy2(str(src), str(model_dir / fname))

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
    print(f"  model.safetensors saved ({size_gb:.2f} GB)")

    wandb_logger.log_summary({
        "prune/params_before":     int(orig_params),
        "prune/params_after":      int(pruned_params),
        "prune/reduction_pct":     round(reduction, 2),
        "prune/compression_ratio": (orig_params / max(pruned_params, 1)),
        "prune/size_gb_after":     round(size_gb, 4),
        "prune/head_target":       HEAD_TARGET,
        "prune/ffn_target":        FFN_TARGET,
        "prune/layer_target":      LAYER_TARGET,
        "prune/hidden_target":     HIDDEN_TARGET,
        "prune/kd_guided":         bool(cfg["kd_pruning"]),
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

    print("\nremoving original download to free disk")
    shutil.rmtree(str(tmp_dir))
    del state
    gc.collect()
    disk_check("after cleanup")

    mark_done(exp_dir, "prune")
    wandb_logger.finish("success")
    print("\npruning complete")
