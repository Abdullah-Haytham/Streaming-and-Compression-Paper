from transformers import AutoConfig, Wav2Vec2BertModel

import torch.nn as nn
import torch.nn.functional as F
import torch

# ── Notebook Model Definitions ───────────────────────────────────────────────

class CTCHead(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.proj = nn.Linear(hidden_size, vocab_size)
        nn.init.normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)

    def forward(self, hidden):
        return F.log_softmax(self.proj(self.dropout(hidden)), dim=-1).transpose(0, 1)

class MultiLevelCTCHeads(nn.Module):
    def __init__(self, hidden_size: int, vocabs: dict):
        super().__init__()
        self.heads = nn.ModuleDict({
            level: CTCHead(hidden_size, len(vocab)) for level, vocab in vocabs.items()
        })

    def forward(self, hidden):
        return {level: head(hidden) for level, head in self.heads.items()}

"""
super().__init__()
        enc_config = AutoConfig.from_pretrained(base_model_id)
        enc_config.add_adapter                         = True
        enc_config.adapter_kernel_size                 = 3
        enc_config.adapter_stride                      = 2
        enc_config.num_adapter_layers                  = 1
        enc_config.use_intermediate_ffn_before_adapter = False
        enc_config.activation_dropout                  = 0.0
        enc_config.attention_dropout                   = 0.0
        enc_config.feat_proj_dropout                   = 0.0
        enc_config.final_dropout                       = 0.1
        enc_config.hidden_dropout                      = 0.0
        enc_config.conformer_conv_dropout              = 0.1
        enc_config.layerdrop                           = 0.0
        enc_config.apply_spec_augment                  = True
        enc_config.mask_feature_length                 = 10
        enc_config.mask_feature_min_masks              = 0
        enc_config.mask_feature_prob                   = 0.0
        enc_config.mask_time_length                    = 10
        enc_config.mask_time_min_masks                 = 2
        enc_config.mask_time_prob                      = 0.0
"""

class QuranMultiCTC(nn.Module):
    def __init__(self, base_model_id: str, vocabs: dict, load_pretrained_encoder: bool = True):
        super().__init__()
        enc_config = AutoConfig.from_pretrained(base_model_id)
        enc_config.add_adapter                         = True
        enc_config.num_adapter_layers                  = 1
        enc_config.use_intermediate_ffn_before_adapter = False
        enc_config.attention_dropout                   = 0.0
        enc_config.feat_proj_dropout                   = 0.0
        enc_config.hidden_dropout                      = 0.0
        enc_config.conformer_conv_dropout              = 0.1
        enc_config.layerdrop                           = 0.0
        enc_config.apply_spec_augment                  = False

        if load_pretrained_encoder:
            self.encoder = Wav2Vec2BertModel.from_pretrained(base_model_id, config=enc_config)
        else:
            self.encoder = Wav2Vec2BertModel(enc_config)
            
        self.encoder.gradient_checkpointing_enable()
        self.heads = MultiLevelCTCHeads(self.encoder.config.hidden_size, vocabs)

    def forward(self, input_features, attention_mask):
        enc_out = self.encoder(input_features=input_features, attention_mask=attention_mask)
        hidden = enc_out.last_hidden_state
        input_lengths = self.encoder._get_feat_extract_output_lengths(attention_mask.sum(dim=-1).long())
        log_probs = self.heads(hidden)
        return (log_probs, input_lengths)

# ─────────────────────────────────────────────────────────────────────────────

_ARABIC_TO_EN: dict[str, str] = {
    "[همس]": "hams",
    "[جهر]": "jahr",
    "[شديد]": "shadeed",
    "[بين الشدة والرخاوة]": "between",
    "[رخو]": "rikhw",
    "[مفخم]": "mofakham",
    "[مرقق]": "moraqaq",
    "[أدنى المفخم]": "low_mofakham",
    "[منفتح]": "monfateh",
    "[مطبق]": "motbaq",
    "[صفير]": "safeer",
    "[لا صفير]": "no_safeer",
    "[مقلقل]": "moqalqal",
    "[لا قلقلة]": "not_moqalqal",
    "[مكرر]": "mokarar",
    "[لا تكرار]": "not_mokarar",
    "[متفشي]": "motafashie",
    "[لا تفشي]": "not_motafashie",
    "[مستطيل]": "mostateel",
    "[لا إستطالة]": "not_mostateel",
    "[مغن]": "maghnoon",
    "[لا غنة]": "not_maghnoon",
    "[PAD]": "[PAD]",
}

_SIFAT_NEGATIVE_LABEL = {
    "tikraar":  "not_mokarar",
    "tafashie": "not_motafashie",
    "qalqla":   "not_moqalqal",
    "istitala": "not_mostateel",
    "safeer":   "no_safeer",
    "itbaq":    "monfateh",
}


class VocabTokenizer:
    """
    Lightweight replacement for MultiLevelTokenizer when loading from a .pt bundle.

    Requires only the checkpoint's ``idx2sym`` dict  (id -> Arabic token)
    and the ``vocabs`` dict (level -> ordered list of tokens).

    Provides the same two attributes consumed by the rest of nutq_core:
        id_to_vocab       : {level: {id: arabic_token}}
        sifat_to_en_vocab : {level: {id: english_label}}  (PAD excluded)
    """

    def __init__(self, idx2sym: dict, vocabs: dict):
        # id → Arabic token, exactly as stored in the checkpoint
        self.level_to_id_vocab: dict[str, dict[int, str]] = {
            level: {int(k): v for k, v in sym_map.items()}
            for level, sym_map in idx2sym.items()
        }
        self.vocab_to_id_level = {
            level: {v: int(k) for k, v in sym_map.items()}
            for level, sym_map in idx2sym.items()
        }

        # id → English label (for sifat levels; phonemes level not needed here)
        self.sifat_level_to_id_to_en_vocab: dict[str, dict[int, str]] = {}
        for level, sym_map in self.level_to_id_vocab.items():
            if level == "phonemes":
                continue
            self.sifat_level_to_id_to_en_vocab[level] = {
                id_: _ARABIC_TO_EN.get(tok, tok)
                for id_, tok in sym_map.items()
            }

    @property
    def id_to_vocab(self) -> dict[str, dict[int, str]]:
        return self.level_to_id_vocab

    @property
    def sifat_to_en_vocab(self) -> dict[str, dict[int, str]]:
        return self.sifat_level_to_id_to_en_vocab

    def tokenize_refs(self, ref_quran_phonetic_script_list) -> dict[str, list[list[int]]]:
        """Manually tokenize references for DP alignment without HF Tokenizer."""
        level_to_ref_ids = {level: [] for level in self.level_to_id_vocab}
        for ref in ref_quran_phonetic_script_list:
            # Phonemes
            ph_ids = [self.vocab_to_id_level["phonemes"].get(char, 0) for char in ref.phonemes]
            level_to_ref_ids["phonemes"].append(ph_ids)
            # Sifat
            for level in self.level_to_id_vocab:
                if level == "phonemes": continue
                
                # We need to map the English attr (e.g. "hams") back to the Arabic token (e.g. "[همس]")
                # We can do this by using the `sifat_to_en_vocab` to find the id directly
                en_to_id = {v: k for k, v in self.sifat_level_to_id_to_en_vocab[level].items()}
                sifat_ids = []
                for s in ref.sifat:
                    # s is a Sifa object. The attr is getattr(s, level)
                    # For qdat_bench dataset, we will pass dicts or objects with english labels
                    val = getattr(s, level)
                    if val is None:
                        val = _SIFAT_NEGATIVE_LABEL.get(level)
                    if hasattr(val, "text"):
                        val = val.text # If it's SingleUnit
                    sifat_ids.append(en_to_id.get(val, 0))
                level_to_ref_ids[level].append(sifat_ids)
                
        # Convert to tensors
        for level in level_to_ref_ids:
            # pad to longest
            max_len = max(len(seq) for seq in level_to_ref_ids[level])
            padded = []
            for seq in level_to_ref_ids[level]:
                padded.append(seq + [0] * (max_len - len(seq)))
            level_to_ref_ids[level] = torch.tensor(padded, dtype=torch.long)
        return level_to_ref_ids
