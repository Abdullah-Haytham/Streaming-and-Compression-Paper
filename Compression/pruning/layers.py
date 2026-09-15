"""Layer-depth pruning (Gromov 2403.17887, cosine-similarity scoring).

Each layer is scored by ``mean cosine(input, output)`` over a small audio
calibration set: a high score means the layer barely changes its input, so
removing it is cheap. Layers ranked highest are dropped and the remaining
ones are re-indexed to a contiguous 0..target-1 sequence.
"""

import gc

from ._state import _LAYER_KEY_IDX_POS, _LAYER_KEY_PREFIX, \
    layer_prefixes_from_state, write_calib_artifacts


def score_layers(cfg: dict, tmp_dir, state, method: str):
    """Return per-layer importance scores. Lower = more prunable.

    ``method == "cosine"`` returns ``{layer_idx: cosine}``; higher cosine means
    the layer is nearly the identity. ``method == "loss_delta"`` is not
    implemented yet.
    """
    if method == "loss_delta":
        raise NotImplementedError(
            "--layer_score loss_delta not implemented yet; use 'cosine'.")
    if method != "cosine":
        raise ValueError(f"unknown layer_score method: {method!r}")

    print("\nscoring encoder layers by mean cosine(input, output)")
    try:
        import torch
        import torch.nn.functional as F
        from transformers import SeamlessM4TFeatureExtractor
    except ImportError as e:
        print(f"  missing dependency: {e} -- falling back to uniform scores")
        prefixes = layer_prefixes_from_state(state.keys())
        return {int(p.rsplit('.', 1)[-1]): 0.0 for p in prefixes}

    from ..data import decode_audio, stream_samples
    from ..model import build_model

    cfg_path, weights_path = write_calib_artifacts(state, tmp_dir, cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}")

    fe = SeamlessM4TFeatureExtractor.from_pretrained(str(tmp_dir))

    n_calib = max(1, cfg["kd_calib_batches"])
    calib_stream = stream_samples(cfg["moshaf"], n_calib * 3)
    raw_inputs, skipped = [], 0
    print(f"  collecting up to {n_calib} calibration samples")
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
    print(f"  collected {len(raw_inputs)} samples ({skipped} skipped)")
    if not raw_inputs:
        try:
            cfg_path.unlink(missing_ok=True)
            weights_path.unlink(missing_ok=True)
        except Exception:
            pass
        prefixes = layer_prefixes_from_state(state.keys())
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

    print("  layer cosine scores (higher = more prunable):")
    for i, s in sorted(scores.items(), key=lambda x: -x[1]):
        print(f"    layer {i:2d}: {s:+.4f}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        cfg_path.unlink(missing_ok=True)
        weights_path.unlink(missing_ok=True)
    except Exception:
        pass
    return scores


def prune_layers(state, target: int, scores: dict):
    """Drop the ``n_layers - target`` layers with the highest cosine scores
    and re-index the rest to a contiguous 0..target-1 sequence.
    """
    prefixes = layer_prefixes_from_state(state.keys())
    layer_indices = [int(p.rsplit(".", 1)[-1]) for p in prefixes]
    n_layers = len(layer_indices)

    if target >= n_layers:
        print(f"  layer count {n_layers} <= target {target}; skipping")
        return state

    ranked = sorted(layer_indices, key=lambda i: -scores.get(i, 0.0))
    remove = set(ranked[:n_layers - target])
    keep   = sorted(i for i in layer_indices if i not in remove)
    remap  = {old: new for new, old in enumerate(keep)}

    print(f"  pruning layers: {n_layers} -> {target}")
    print(f"    removing: {sorted(remove)}")
    print(f"    keeping (old->new): {[(o, remap[o]) for o in keep]}")

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
