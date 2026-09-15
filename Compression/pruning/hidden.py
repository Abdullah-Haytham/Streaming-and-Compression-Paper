"""Hidden-size pruning (SP^3, LLM-Pruner, activation-L2 scoring).

The slicer touches many tensor families because Wav2Vec2-BERT's adapter sits
on top of the Conformer encoder, and both expose tensors sized to the hidden
dimension in different ways:

  - plain row/col linears (LayerNorm, Q/K/V, FFN intermediate/output, CTC heads)
  - doubled-row convs (Conformer pointwise_conv1, adapter residual_conv /
    self_attn_conv): first axis is 2*hidden because a GLU halves them back
  - square hidden*hidden tensors (Conformer pointwise_conv2; adapter
    attention Q/K/V/out, which are square because the adapter's
    heads*head_dim == hidden_size)

The tag tables below capture which substrings indicate which family, and
``classify_hidden_axis`` turns each (key, shape) into one of
{row, col, both, doubled, none}.
"""

import gc

from ._state import layer_prefixes_from_state


# Tag tables (spec sec 5.2, extended to cover the Wav2Vec2-BERT adapter).
# Tags are matched as substrings of the parameter key, so leaf module names
# are used (no '.weight' suffix) -- that way both '.weight' and '.bias' of
# the same module are caught by a single tag.
_HIDDEN_ROW_TAGS = (
    "feature_projection.projection",
    "linear_out",
    "output_dense",
    "layer_norm",
    "depthwise_conv",
    "masked_spec_embed",
)
_HIDDEN_COL_TAGS = (
    "linear_q", "linear_k", "linear_v",
    "intermediate_dense",
    "ctc_heads",
    "level_to_lm_head",                 # remapped to ctc_heads in build_model
)
# Doubled-output convs feed a GLU: weight is (2*hidden, hidden, k), bias (2*hidden,).
_HIDDEN_DOUBLED_TAGS = (
    "pointwise_conv1",
    "residual_conv",
    "self_attn_conv",
)
# Both-axis (square) convs: weight is (hidden, hidden, k); bias is (hidden,).
_HIDDEN_BOTH_TAGS = (
    "pointwise_conv2",
)
_HIDDEN_SKIP_TAGS = (
    "feature_extractor",
    "pos_bias_u", "pos_bias_v",
    "logit_scale",
    # NB: do NOT add "lm_head" -- it is a substring of "level_to_lm_head" (the
    # multi-level CTC heads), which must be col-sliced, not skipped.
)


def discover_hidden_size(state) -> int:
    """Read the encoder hidden size from a (possibly partially pruned) state
    dict. Tries several reliable reference keys because FFN naming varies
    across HF Wav2Vec2-BERT checkpoints.
    """
    for k, v in state.items():
        if "feature_projection.projection.weight" in k:
            return int(v.shape[0])
    for k, v in state.items():
        if k.startswith("wav2vec2_bert.encoder.layers.") \
                and "self_attn.linear_q.weight" in k:
            return int(v.shape[1])
    for k, v in state.items():
        if k.endswith("output_dense.weight"):
            return int(v.shape[0])
    raise RuntimeError(
        "could not infer hidden_size from state dict -- none of "
        "feature_projection.projection.weight / encoder.layers.*.linear_q.weight / "
        "*.output_dense.weight were found. keys present: "
        f"{list(state.keys())[:20]}...")


def classify_hidden_axis(key: str, v_shape: tuple, hidden_size: int) -> str:
    """Return one of {'row', 'col', 'doubled', 'both', 'none'} for a tensor.

    Branch order matters: doubled-conv weights have a 2*hidden first axis so
    they must NOT be confused with row tensors; square attention linears must
    be detected before the generic row/col branches; adapter biases of size
    hidden need their own short-circuit because they share a tag with column
    tensors but are 1-D.
    """
    if any(t in key for t in _HIDDEN_SKIP_TAGS):
        return "none"

    if any(t in key for t in _HIDDEN_DOUBLED_TAGS):
        if len(v_shape) >= 2 and v_shape[0] == 2 * hidden_size:
            return "doubled"
        if len(v_shape) == 1 and v_shape[0] == 2 * hidden_size:
            return "doubled"
        # Shape didn't match the (2H, H, ...) pattern -- fall through.

    if any(t in key for t in _HIDDEN_BOTH_TAGS):
        if (len(v_shape) >= 2
                and v_shape[0] == hidden_size
                and v_shape[1] == hidden_size):
            return "both"
        if len(v_shape) == 1 and v_shape[0] == hidden_size:
            return "row"   # pointwise_conv2.bias is just a hidden-vector

    is_row = any(t in key for t in _HIDDEN_ROW_TAGS)
    is_col = any(t in key for t in _HIDDEN_COL_TAGS)

    # Square hidden*hidden tensors arise on the adapter side: its self_attn
    # linear_{q,k,v,out} are Linear(output_hidden_size, output_hidden_size).
    # Restricted to attention linears so we don't trip on FFN intermediate /
    # output_dense when ffn_target coincidentally equals hidden_size.
    is_attn_linear = any(t in key for t in (
        "self_attn.linear_q", "self_attn.linear_k",
        "self_attn.linear_v", "self_attn.linear_out",
    ))
    if is_attn_linear and len(v_shape) == 2 \
            and v_shape[0] == hidden_size and v_shape[1] == hidden_size:
        return "both"

    # Adapter linear_*.bias are sized to output_hidden_size (they ARE hidden).
    # Encoder linear_q/k/v.bias are sized to num_heads*head_dim and only equal
    # hidden_size by coincidence -- those keep the prior 'none' path.
    if "adapter" in key and is_col and len(v_shape) == 1 \
            and v_shape[0] == hidden_size:
        return "row"

    if is_row and v_shape and v_shape[0] == hidden_size:
        return "row"
    # 'col' only applies to multi-D weight matrices. 1-D tensors matching a
    # col-tag substring (linear_q/k/v.bias, intermediate_dense.bias,
    # ctc_heads.bias) are biases sized to QKV-rows / FFN-dim / vocab -- they
    # may equal hidden_size by coincidence but must be left alone.
    if is_col and v_shape and len(v_shape) >= 2 and v_shape[-1] == hidden_size:
        return "col"
    return "none"


def score_hidden_dims(cfg: dict, tmp_dir, state):
    """Per-dimension activation L2-norm; higher = more important.

    Falls back to weight-norm scoring when the live model can't be loaded
    (missing transformers, OOM, etc.).
    """
    import numpy as np

    current_hidden = discover_hidden_size(state)
    print(f"\nscoring hidden dims (current size = {current_hidden})")

    try:
        import torch
        from transformers import SeamlessM4TFeatureExtractor
    except ImportError as e:
        print(f"  {e} -- using weight-norm fallback")
        return score_hidden_dims_weight_only(state, current_hidden)

    from ._state import write_calib_artifacts
    from ..data import decode_audio, stream_samples
    from ..model import build_model

    cfg_path, weights_path = write_calib_artifacts(state, tmp_dir, cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        model, _ = build_model(cfg_path, weights_path, device)
    except Exception as e:
        print(f"  live model load failed ({e}) -- using weight-norm fallback")
        try:
            cfg_path.unlink(missing_ok=True)
            weights_path.unlink(missing_ok=True)
        except Exception:
            pass
        return score_hidden_dims_weight_only(state, current_hidden)

    if device.type == "cuda":
        model = model.half()
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    fe = SeamlessM4TFeatureExtractor.from_pretrained(str(tmp_dir))
    n_calib = max(1, cfg["kd_calib_batches"])

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
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            cfg_path.unlink(missing_ok=True)
            weights_path.unlink(missing_ok=True)
        except Exception:
            pass
        return score_hidden_dims_weight_only(state, current_hidden)

    sumsq = np.zeros(current_hidden, dtype=np.float64)

    def hook(_mod, _inp, out):
        h = out[0] if isinstance(out, (tuple, list)) else out
        flat = h.detach().reshape(-1, h.shape[-1]).float().cpu().numpy()
        sumsq[:] += (flat ** 2).sum(axis=0)

    handle = model.wav2vec2_bert.feature_projection.register_forward_hook(hook)
    with torch.no_grad():
        for i, inp_cpu in enumerate(raw_inputs):
            inp = inp_cpu.to(device)
            if device.type == "cuda":
                inp = inp.half()
            _ = model.wav2vec2_bert(inp)
            del inp
            if (i + 1) % 10 == 0:
                print(f"    calib: {i+1}/{len(raw_inputs)}", flush=True)
    handle.remove()

    scores = np.sqrt(sumsq).astype(np.float32)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        cfg_path.unlink(missing_ok=True)
        weights_path.unlink(missing_ok=True)
    except Exception:
        pass
    print(f"  activation-norm scores: min={scores.min():.3f} max={scores.max():.3f}")
    return scores


def score_hidden_dims_weight_only(state, hidden_size: int):
    """Fallback scorer when no live model / calibration data is available."""
    import numpy as np

    scores = np.zeros(hidden_size, dtype=np.float64)
    for k, v in state.items():
        axis = classify_hidden_axis(k, v.shape, hidden_size)
        if axis == "row" and v.ndim >= 1 and v.shape[0] == hidden_size:
            scores += np.linalg.norm(v.reshape(hidden_size, -1), axis=1)
        elif axis == "col" and v.ndim >= 2 and v.shape[-1] == hidden_size:
            scores += np.linalg.norm(v.reshape(-1, hidden_size), axis=0)
        elif axis == "both":
            scores += np.linalg.norm(v.reshape(hidden_size, -1), axis=1)
    return scores.astype(np.float32)


def prune_hidden_size(state, target: int, scores):
    """Slice every hidden-tied axis to ``target`` keeping the top-scoring dims."""
    import numpy as np

    current = discover_hidden_size(state)

    if target >= current:
        print(f"  hidden size {current} <= target {target} -- skipping")
        return state

    keep = np.sort(np.argsort(scores)[-target:])
    keep_doubled = np.concatenate([keep, keep + current])  # for pointwise_conv1 / adapter convs
    print(f"  pruning hidden size: {current} -> {target} "
          f"(keeping dims {keep[:4].tolist()}...{keep[-4:].tolist()})")

    # One-off shape audit so silent miscompiles surface loudly. Covers both
    # Conformer encoder and adapter shapes.
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
        axis = classify_hidden_axis(k, tuple(v.shape), current)
        if axis == "row":
            n_row += 1
            if v.ndim == 1:
                new_state[k] = v[keep]
            elif v.ndim == 2:
                new_state[k] = v[keep, :]
            elif v.ndim == 3:
                new_state[k] = v[keep, :, :]
            else:
                print(f"    unexpected row shape: {k} {v.shape} -- left unchanged")
                new_state[k] = v
        elif axis == "col":
            n_col += 1
            if v.ndim == 2:
                new_state[k] = v[:, keep]
            elif v.ndim == 3:
                new_state[k] = v[:, keep, :]
            else:
                print(f"    unexpected col shape: {k} {v.shape} -- left unchanged")
                new_state[k] = v
        elif axis == "both":
            n_both += 1
            new_state[k] = v[keep][:, keep] if v.ndim == 2 else v[keep][:, keep, :]
        elif axis == "doubled":
            n_doubled += 1
            if v.ndim == 1 and v.shape[0] == 2 * current:
                new_state[k] = v[keep_doubled]
            elif v.ndim == 2 and v.shape[0] == 2 * current:
                new_state[k] = v[keep_doubled][:, keep]
            elif v.ndim == 3 and v.shape[0] == 2 * current:
                new_state[k] = v[keep_doubled][:, keep, :]
            else:
                print(f"    doubled-conv unexpected shape: {k} {v.shape} "
                      f"(expected leading dim {2 * current}) -- aborting")
                raise RuntimeError(
                    f"doubled-conv shape mismatch for {k}: {v.shape}")
        else:
            new_state[k] = v
    print(f"    sliced: {n_row} row, {n_col} col, {n_both} both, {n_doubled} doubled")

    # Catch-all: surface any non-skip tensor that still has a `current`-sized
    # axis after slicing -- likely hidden-tied tensors we missed.
    suspicious = []
    for k, v in new_state.items():
        if any(t in k for t in _HIDDEN_SKIP_TAGS):
            continue
        if any(d == current for d in v.shape):
            suspicious.append(f"      {k}: {tuple(v.shape)}")
    if suspicious:
        print(f"    {len(suspicious)} tensors still contain dim {current} "
              f"after slicing (likely missed tag):")
        for s in suspicious[:20]:
            print(s)
        if len(suspicious) > 20:
            print(f"      ... and {len(suspicious) - 20} more")
    return new_state


def validate_hidden_size(state, target: int):
    """Check every hidden-tied tensor was sliced to ``target``.

    Raises ``RuntimeError`` listing every mismatch found.
    """
    errors = []
    for k, v in state.items():
        if "feature_extractor" in k:
            continue  # CNN front-end uses its own channel dims
        if "feature_projection.layer_norm" in k:
            continue  # input-side LayerNorm sized to input_features, NOT hidden
        if "layer_norm" in k and v.ndim == 1 and v.shape[0] != target:
            errors.append(f"  {k}: shape {tuple(v.shape)}, expected ({target},)")
        if "linear_out.weight" in k and v.shape[0] != target:
            errors.append(f"  {k}: shape {tuple(v.shape)}, expected ({target}, ...)")
        if "output_dense.weight" in k and v.shape[0] != target:
            errors.append(f"  {k}: shape {tuple(v.shape)}, expected ({target}, ...)")
        if (("ctc_heads" in k) or ("level_to_lm_head" in k)) \
                and k.endswith(".weight") and v.shape[-1] != target:
            errors.append(f"  {k}: shape {tuple(v.shape)}, expected (..., {target})")
        if any(t in k for t in ("residual_conv", "self_attn_conv",
                                 "pointwise_conv1")):
            if v.ndim >= 2 and v.shape[0] != 2 * target:
                errors.append(f"  {k}: shape {tuple(v.shape)}, "
                              f"expected ({2*target}, {target}, ...)")
            if v.ndim >= 2 and v.shape[1] != target:
                errors.append(f"  {k}: shape {tuple(v.shape)}, "
                              f"expected ({2*target}, {target}, ...)")
            if v.ndim == 1 and v.shape[0] != 2 * target:
                errors.append(f"  {k}: shape {tuple(v.shape)}, "
                              f"expected ({2*target},)")
    if errors:
        raise RuntimeError(
            "hidden-size pruning left inconsistent shapes:\n" + "\n".join(errors))
    print(f"  hidden-size validation passed (target={target}, adapter included)")
