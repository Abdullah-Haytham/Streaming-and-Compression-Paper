"""Attention-head and FFN-neuron pruning.

Both routines accept an optional dict of KD-guided Taylor scores keyed by
layer prefix. When supplied (and shape-compatible), KD scores take precedence
over weight-magnitude scoring; otherwise we fall back to magnitude.
"""

from ._state import layer_prefixes_from_state


def _head_row_slice(state, layer, qkv, head_idx, head_dim):
    """The rows of one head inside Q/K/V's row-stacked weight matrix."""
    weight = state[f"{layer}.self_attn.linear_{qkv}.weight"]
    return weight[head_idx * head_dim : (head_idx + 1) * head_dim]


def _head_magnitude_scores(state, layer, head_dim, n_heads):
    """``[(head_idx, score), ...]`` sorted ascending; score = ||Q|| + ||K|| + ||V||."""
    import numpy as np

    scored = []
    for h in range(n_heads):
        score = sum(
            np.linalg.norm(_head_row_slice(state, layer, qkv, h, head_dim))
            for qkv in ("q", "k", "v")
        )
        scored.append((h, float(score)))
    scored.sort(key=lambda hs: hs[1])
    return scored


def prune_heads(state, target: int, kd_head_scores=None):
    """Slice each layer's Q/K/V rows down to ``target`` heads (16 -> target)."""
    prefixes = layer_prefixes_from_state(state.keys())
    q0 = state[f"{prefixes[0]}.self_attn.linear_q.weight"]
    old_heads, head_dim = 16, q0.shape[0] // 16
    if target >= old_heads:
        return state
    print(f"  pruning heads: {old_heads} -> {target} (head_dim={head_dim})")

    using_kd = (kd_head_scores is not None)
    print(f"  scorer: {'KD-guided Taylor' if using_kd else 'weight magnitude'}")

    for layer in prefixes:
        if using_kd and layer in kd_head_scores:
            kd_s = kd_head_scores[layer]   # shape (old_heads,)
            scores = sorted(
                [(h, float(kd_s[h])) for h in range(old_heads)],
                key=lambda x: x[1],
            )
        else:
            scores = _head_magnitude_scores(state, layer, head_dim, old_heads)

        keep = sorted(h for h, _ in scores[(old_heads - target):])
        rows = [i for h in keep for i in range(h*head_dim, (h+1)*head_dim)]
        for base in ("linear_q", "linear_k", "linear_v"):
            w = f"{layer}.self_attn.{base}.weight"
            b = f"{layer}.self_attn.{base}.bias"
            state[w] = state[w][rows, :]
            if b in state:
                state[b] = state[b][rows]
        state[f"{layer}.self_attn.linear_out.weight"] = state[f"{layer}.self_attn.linear_out.weight"][:, rows]
    return state


def find_ffn_keys(state, layer_prefix):
    """Locate the FFN intermediate (up) and output (down) weight keys.

    Checkpoints use different naming conventions
    (``feed_forward.intermediate_dense`` / ``ffn.intermediate_dense`` / ``fc1``);
    accept any. Final fallback locates the up/down pair from shape
    (intermediate has more rows than cols).
    """
    candidates_w1, candidates_w2 = [], []
    for k in state.keys():
        if not k.startswith(layer_prefix + "."):
            continue
        if any(tag in k for tag in ("intermediate_dense", "intermediate.weight",
                                     "fc1", "dense_act")):
            if k.endswith(".weight"):
                candidates_w1.append(k)
        if any(tag in k for tag in ("output_dense", "output.weight",
                                     "fc2", "dense.")):
            if k.endswith(".weight"):
                candidates_w2.append(k)

    if not candidates_w1 or not candidates_w2:
        layer_keys = [k for k in state.keys()
                      if k.startswith(layer_prefix + ".") and k.endswith(".weight")
                      and "self_attn" not in k and "layer_norm" not in k
                      and "lm_head" not in k and "ctc" not in k]
        up, down = [], []
        for k in layer_keys:
            w = state[k]
            if w.ndim == 2:
                if w.shape[0] > w.shape[1]:
                    up.append(k)
                elif w.shape[1] > w.shape[0]:
                    down.append(k)
        if up and down:
            candidates_w1 = candidates_w1 or up
            candidates_w2 = candidates_w2 or down

    if not candidates_w1 or not candidates_w2:
        raise KeyError(
            f"could not find FFN weight keys under '{layer_prefix}'.\n"
            f"keys present: {[k for k in state if k.startswith(layer_prefix+'.')]}"
        )

    w1_key = max(candidates_w1, key=lambda k: state[k].shape[0])
    w2_key = max(candidates_w2, key=lambda k: state[k].shape[1])
    return w1_key, w2_key


def prune_ffn(state, target: int, kd_ffn_scores=None):
    """Slice each FFN's intermediate dimension to ``target`` neurons."""
    import numpy as np

    prefixes = layer_prefixes_from_state(state.keys())

    w1_key_0, w2_key_0 = find_ffn_keys(state, prefixes[0])
    cur = state[w1_key_0].shape[0]
    if target >= cur:
        return state

    p0 = prefixes[0] + "."
    w1_subpath = w1_key_0[len(p0):]
    w2_subpath = w2_key_0[len(p0):]
    b1_subpath = w1_subpath.replace(".weight", ".bias")

    print(f"  pruning FFN: {cur} -> {target}")
    print(f"    up-key:   {w1_subpath}")
    print(f"    down-key: {w2_subpath}")

    using_kd = (kd_ffn_scores is not None)
    print(f"  scorer: {'KD-guided Taylor' if using_kd else 'weight magnitude'}")

    for layer in prefixes:
        w1 = state[f"{layer}.{w1_subpath}"]
        w2 = state[f"{layer}.{w2_subpath}"]

        if using_kd and layer in kd_ffn_scores:
            scores = kd_ffn_scores[layer]   # shape (cur_ffn_dim,)
            # Shape mismatch can happen when recorded scores came from a
            # different (e.g. partially-pruned) checkpoint.
            if len(scores) != w1.shape[0]:
                scores = np.linalg.norm(w1, axis=1) * np.linalg.norm(w2, axis=0)
        else:
            scores = np.linalg.norm(w1, axis=1) * np.linalg.norm(w2, axis=0)

        keep = np.sort(np.argsort(scores)[-target:])
        state[f"{layer}.{w1_subpath}"] = w1[keep, :]
        b1_key = f"{layer}.{b1_subpath}"
        if b1_key in state:
            state[b1_key] = state[b1_key][keep]
        state[f"{layer}.{w2_subpath}"] = w2[:, keep]
    return state
