"""Wav2Vec2-BERT multi-level CTC model loader.

``build_model`` reads a config + safetensors pair and rebinds ``_parameters``
and ``_buffers`` directly so weights pruned to non-default shapes load
cleanly (``load_state_dict`` would refuse the shape changes).
"""

import json


# Multi-level CTC heads ship in the upstream state dict as 'level_to_lm_head.*'
# but our nn.Module exposes them as 'ctc_heads.*'. Rename on load.
_OLD_CTC_PREFIX = "level_to_lm_head."
_NEW_CTC_PREFIX = "ctc_heads."


def _remap_state_dict_keys(state_dict):
    """Rename ``level_to_lm_head.*`` keys to ``ctc_heads.*`` (start-of-key only)."""
    remapped = {}
    for key, value in state_dict.items():
        if key.startswith(_OLD_CTC_PREFIX):
            new_key = _NEW_CTC_PREFIX + key[len(_OLD_CTC_PREFIX):]
        else:
            new_key = key
        remapped[new_key] = value
    return remapped


def build_model(config_path, weights_path, device):
    """Return ``(model, level_to_vocab_size)``.

    ``forward`` accepts ``return_hidden_states`` so KD code can grab per-layer
    encoder activations.
    """
    import torch.nn as nn
    from safetensors.torch import load_file
    from transformers import Wav2Vec2BertConfig, Wav2Vec2BertModel

    HEAD_DIM = 64  # fixed for the Wav2Vec2-BERT family

    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)

    level_to_vocab_size = cfg.pop("level_to_vocab_size")
    for k in ("level_to_loss_weight", "architectures", "model_type", "transformers_version"):
        cfg.pop(k, None)
    # Trust config.json for num_attention_heads / intermediate_size /
    # num_hidden_layers / hidden_size -- they reflect the actual pruned shapes.
    bert_cfg = Wav2Vec2BertConfig(**cfg)

    class Wav2Vec2BertForMultilevelCTC(nn.Module):
        def __init__(self, config, vocab_sizes):
            super().__init__()
            self.wav2vec2_bert = Wav2Vec2BertModel(config)
            self.dropout = nn.Dropout(config.final_dropout)
            self.ctc_heads = nn.ModuleDict({
                name: nn.Linear(config.hidden_size, vs, bias=True)
                for name, vs in vocab_sizes.items()
            })

        def forward(self, input_features, attention_mask=None,
                    return_hidden_states=False):
            out = self.wav2vec2_bert(
                input_features=input_features,
                attention_mask=attention_mask,
                output_hidden_states=return_hidden_states,
            )
            h = self.dropout(out.last_hidden_state)
            logits = {name: head(h) for name, head in self.ctc_heads.items()}
            if return_hidden_states:
                return logits, out.hidden_states
            return logits

    model = Wav2Vec2BertForMultilevelCTC(bert_cfg, level_to_vocab_size)

    # Manual param assignment tolerates pruning-induced shape mismatch. The
    # state-dict's ``level_to_lm_head.*`` keys are the multi-level CTC heads,
    # which we expose under ``ctc_heads.*``.
    sd = load_file(str(weights_path), device=str(device))
    remapped = _remap_state_dict_keys(sd)
    for name, tensor in remapped.items():
        parts = name.split(".")
        obj = model
        try:
            for p in parts[:-1]:
                obj = getattr(obj, p)
        except AttributeError:
            continue
        attr = parts[-1]
        if attr in obj._parameters:
            obj._parameters[attr] = nn.Parameter(tensor)
        elif attr in obj._buffers:
            obj._buffers[attr] = tensor

    # Patch attention head counts on the in-memory modules. Wav2Vec2BertConfig
    # is built from the (already-correct) config.json, but the modules cache
    # the original head counts as instance attributes that several forward
    # paths read directly.
    for layer in model.wav2vec2_bert.encoder.layers:
        a = layer.self_attn
        a.num_heads = a.linear_q.weight.shape[0] // HEAD_DIM
        a.head_dim  = HEAD_DIM
        if hasattr(a, "head_size"):
            a.head_size = HEAD_DIM
    if hasattr(model.wav2vec2_bert, "adapter") and model.wav2vec2_bert.adapter:
        for al in model.wav2vec2_bert.adapter.layers:
            if hasattr(al, "self_attn"):
                al.self_attn.num_heads = al.self_attn.linear_q.weight.shape[0] // HEAD_DIM
                al.self_attn.head_dim  = HEAD_DIM
                if hasattr(al.self_attn, "head_size"):
                    al.self_attn.head_size = HEAD_DIM

    model.to(device)
    return model, level_to_vocab_size
