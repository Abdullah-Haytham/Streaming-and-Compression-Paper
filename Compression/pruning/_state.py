"""Shared helpers across the pruning sub-modules.

Anything that knows about the encoder's state-dict layout lives here so the
individual techniques (heads_ffn / layers / hidden / kd_scores) don't each
re-derive it.
"""

import json
from pathlib import Path


_LAYER_KEY_PREFIX  = "wav2vec2_bert.encoder.layers."
_LAYER_KEY_IDX_POS = 3  # index in key.split(".") where the layer number lives

# Substrings that identify the FFN "up" projection (intermediate dim grows
# from hidden). Different Wav2Vec2-BERT checkpoints use different names.
_FFN_INTERMEDIATE_TAGS = (
    "intermediate_dense",
    "feed_forward.intermediate",
    "fc1",
)


def _layer_index(layer_prefix: str) -> int:
    """Extract N from a 'wav2vec2_bert.encoder.layers.N' prefix."""
    return int(layer_prefix.split(".")[_LAYER_KEY_IDX_POS])


def layer_prefixes_from_state(keys):
    """Encoder layer prefixes ('...encoder.layers.N') sorted by N."""
    unique_prefixes = set()
    for k in keys:
        if not k.startswith(_LAYER_KEY_PREFIX):
            continue
        # First four dotted parts are 'wav2vec2_bert.encoder.layers.N'.
        prefix = ".".join(k.split(".")[:_LAYER_KEY_IDX_POS + 1])
        unique_prefixes.add(prefix)
    return sorted(unique_prefixes, key=_layer_index)


def _find_ffn_intermediate_w_keys(state, layer_prefix: str):
    """All FFN-intermediate weight keys under ``layer_prefix`` (any naming)."""
    layer_scope = layer_prefix + "."
    matches = []
    for k in state:
        if not k.startswith(layer_scope):
            continue
        if not k.endswith(".weight"):
            continue
        if any(tag in k for tag in _FFN_INTERMEDIATE_TAGS):
            matches.append(k)
    return matches


def write_calib_artifacts(state, tmp_dir: Path, cfg: dict):
    """Materialise the current (possibly partially-pruned) state plus a
    config.json synthesised to match the live tensor shapes.

    Returns ``(cfg_path, weights_path)``, both inside ``tmp_dir``. Callers
    feed these to ``model.build_model`` and run calibration forward(s) against
    the resulting model.
    """
    from safetensors.numpy import save_file

    weights_path = tmp_dir / "_calib_model.safetensors"
    cfg_path     = tmp_dir / "_calib_config.json"

    # Derive architecture from tensor shapes -- don't trust the config file.
    prefixes = layer_prefixes_from_state(state.keys())
    n_layers = len(prefixes)
    q0 = state[f"{prefixes[0]}.self_attn.linear_q.weight"]
    cur_qkv_rows = q0.shape[0]    # heads * head_dim
    cur_hidden   = q0.shape[1]    # cols of Q == hidden
    HEAD_DIM     = 64
    cur_heads    = cur_qkv_rows // HEAD_DIM
    ffn_w_keys = _find_ffn_intermediate_w_keys(state, prefixes[0])
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
