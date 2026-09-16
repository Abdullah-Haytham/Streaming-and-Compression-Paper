
# This is a customized version of the app 
# https://github.com/obadx/quran-muaalem/blob/main/src/quran_muaalem/inference.py

from ctc_decoder import CTCDecoder, GreedyCTCDecoder

from imports.utils import chunck_phonemes, format_sifat
from imports.muaalem_typing import Unit, Sifa, MuaalemOutput

import torch
from transformers import AutoFeatureExtractor
from numpy.typing import NDArray

from imports.modeling.multi_level_tokenizer import MultiLevelTokenizer
from imports.modeling.modeling_multi_level_ctc import Wav2Vec2BertForMultilevelCTC

from nutq_model import *
from imports.decode import multilevel_greedy_decode
from imports.modeling.vocab import PAD_TOKEN_IDX


# Defined to improve the accuracy of specific chars with unique sifa
# Particularly, tafashie, itbaq, and qalqala
SIFAT_POSITIVE_PHONEMES = {
    "tikraar":  frozenset("ر"),
    "tafashie": frozenset("ش"),
    "qalqla":   frozenset("قطبجد"),
    "istitala": frozenset("ض"),
    "safeer":   frozenset("صزس"),
    "itbaq":    frozenset("صضطظ"),
}

_SIFAT_NEGATIVE_LABEL = {
    "tikraar":  "not_mokarar",
    "tafashie": "not_motafashie",
    "qalqla":   "not_moqalqal",
    "istitala": "not_mostateel",
    "safeer":   "no_safeer",
    "itbaq":    "monfateh",
}

class Nutq:
    def __init__(
        self,
        model_name_or_path:    str = "obadx/muaalem-model-v3_2",
        decoder: CTCDecoder | None = None,
        device:                str = "cpu",
        dtype                      = torch.bfloat16,
    ):
        """
            model_name_or_path: HuggingFace model name *or* path to a local .pt bundle
                                (e.g. "inference_ready.pt")
            decoder: The type of CTCDecoder for the phonemes (Beam or Greedy)
            device: the device to run model on
            dtype: the torch dtype. Default is `torch.bfloat16` as the model was trained on
        """

        self.device     = device
        self.dtype      = dtype
        self.decoder    = decoder if decoder is not None else GreedyCTCDecoder(blank_id=0)

        # ── Local .pt bundle (inference_ready.pt from the training notebook) ──
        if str(model_name_or_path).endswith(".pt"):
            ckpt = torch.load(model_name_or_path, map_location="cpu", weights_only=False)

            self.model = QuranMultiCTC(
                ckpt["base_model_id"], ckpt["vocabs"], load_pretrained_encoder=False
            )
            self.model.load_state_dict(ckpt["model_state"])

            # Build tokenizer directly from the checkpoint — no HF hub needed
            self.multi_level_tokenizer = VocabTokenizer(ckpt["idx2sym"], ckpt["vocabs"])

            self._sample_rate = ckpt.get("sample_rate", 16_000)
            self.processor = AutoFeatureExtractor.from_pretrained(ckpt["base_model_id"])
        # ── HuggingFace-hosted model (e.g. obadx/muaalem-model-v3_2) ─────────
        else:
            self.model = Wav2Vec2BertForMultilevelCTC.from_pretrained(model_name_or_path)
            # Lazy import: MultiLevelTokenizer requires quran_transcript
            from imports.modeling.multi_level_tokenizer import MultiLevelTokenizer
            self.multi_level_tokenizer = MultiLevelTokenizer(model_name_or_path)
            self.processor = AutoFeatureExtractor.from_pretrained(model_name_or_path)
            self._sample_rate = 16_000

        self.model.to(device, dtype=dtype)

    @torch.no_grad()
    def __call__(
        self,
        waves: list[list[float] | torch.FloatTensor | NDArray],
        ref_quran_phonetic_script_list: list | None = None,
    ) -> list[MuaalemOutput]:
        """Infrence Funcion for the Quran Muaalem Project

                waves (list[list[float] | torch.FloatTensor | NDArray]): A batch of input audio waveforms. 
                    Each element in the list represents a single audio sequence (1D array of length `seq_len`).

                sampleing_rate (int): has to be 16000 (as the processor expects this)

        Returns:
            list[MuaalemOutput]:
                A list of output objects, each containing phoneme predictions and their
                phonetic features (sifat) for a processed input.

            Each MuaalemOutput contains:
                phonemes (Unit):
                    A dataclass representing the predicted phoneme sequence with:
                        text (str): Concatenated string of all phonemes.
                        probs (Union[torch.FloatTensor, list[float]]):
                            Confidence probabilities for each predicted phoneme.
                        ids (Union[torch.LongTensor, list[int]]):
                            Token IDs corresponding to each phoneme.

                sifat (list[Sifa]):
                    A list of phonetic feature dataclasses (one per phoneme) with the
                    following optional properties (each is a SingleUnit or None):
                        - phonemes_group (str): the phonemes associated with the `sifa`
                        - hams_or_jahr (SingleUnit): either `hams` or `jahr`
                        - shidda_or_rakhawa (SingleUnit): either `shadeed`, `between`, or `rikhw`
                        - tafkheem_or_taqeeq (SingleUnit): either `mofakham`, `moraqaq`, or `low_mofakham`
                        - itbaq (SingleUnit): either `monfateh`, or `motbaq`
                        - safeer (SingleUnit): either `safeer`, or `no_safeer`
                        - qalqla (SingleUnit): eithr `moqalqal`, or `not_moqalqal`
                        - tikraar (SingleUnit): either `mokarar` or `not_mokarar`
                        - tafashie (SingleUnit): either `motafashie`, or `not_motafashie`
                        - istitala (SingleUnit): either `mostateel`, or `not_mostateel`
                        - ghonna (SingleUnit): either `maghnoon`, or `not_maghnoon`

            Each SingleUnit in Sifa properties contains:
                text (str): The feature's categorical label (e.g., "hams", "shidda").
                prob (float): Confidence probability for this feature.
                idx (int): Identifier for the feature class.
        """

        sampling_rate = self._sample_rate

        # Extract the features from the audio (Log-Mel Spectrogram)
        # Returns a dict with 'input_features' of shape (B, n_frames, n_mels) and 'attention_mask'
        features = self.processor(
            waves, sampling_rate=sampling_rate, return_tensors="pt"
        )
        
        # Move to the specified device (GPU, CPU, etc.)
        features = {
            k: (v.to(self.device, dtype=self.dtype) if k != "attention_mask" else v.to(self.device)) for k, v in features.items()
        }

        # ── Model forward ────────────────────────────────────────────────────
        if isinstance(self.model, QuranMultiCTC):
            # .pt bundle model: returns (log_probs_dict, input_lengths_tensor)
            model_out = self.model(features["input_features"], features["attention_mask"])
            # The .pt model's CTCHead transposes logits to (T, B, V) for CTCLoss.
            # We transpose back to (B, T, V) for decoding.
            levels_logits = {k: v.transpose(0, 1) for k, v in model_out[0].items()}
            lengths_tensor = model_out[1]
            input_lengths = lengths_tensor.tolist()
        else:
            # HF model: returns tuple, logits dict is first element
            levels_logits = self.model(**features, return_dict=False)[0]

            attention_mask = features.get("attention_mask")
            if attention_mask is not None:
                input_lengths = self.model._get_feat_extract_output_lengths(attention_mask.sum([-1])).tolist()
            else:
                input_lengths = [levels_logits["phonemes"].shape[1]] * len(waves)

        # Decoding only Phonemes Level using our decoder
        phonemes_units = self._decode_phonemes(levels_logits["phonemes"], input_lengths)

        chunked_phonemes_batch: list[list[str]] = [
            chunck_phonemes(pu.text) for pu in phonemes_units
        ]

        # ── Fast Narrow-Window Pooling Alignment (Default for all levels) ────
        level_to_units = self._decode_sifat(levels_logits, phonemes_units, chunked_phonemes_batch)

        if ref_quran_phonetic_script_list is not None:
            # ── Dynamic Programming (DP) Alignment for Specific Sifats ───────
            # We only run DP for fluid Sifats that significantly benefit from it
            dp_levels = ["phonemes", "ghonna", "shidda_or_rakhawa"]
            
            # Convert ref strings to token IDs
            if hasattr(self.multi_level_tokenizer, "tokenize_refs"):
                level_to_ref_ids = self.multi_level_tokenizer.tokenize_refs(ref_quran_phonetic_script_list)
            else:
                level_to_ref_ids = self.multi_level_tokenizer.tokenize(
                    [r.phonemes for r in ref_quran_phonetic_script_list],
                    [r.sifat for r in ref_quran_phonetic_script_list],
                    to_dict=True,
                    return_tensors="pt",
                    padding="longest",
                )["input_ids"]

            # Calculate raw Softmax probabilities per frame only for DP levels
            probs = {}
            for level in dp_levels:
                if level in levels_logits:
                    probs[level] = torch.nn.functional.softmax(levels_logits[level], dim=-1).cpu()

            ref_chuncked_phonemes_batch = [
                [s.phonemes for s in r.sifat] for r in ref_quran_phonetic_script_list
            ]

            dp_level_to_units = multilevel_greedy_decode(
                level_to_probs=probs,
                level_to_id_to_vocab=self.multi_level_tokenizer.id_to_vocab,
                level_to_ref_ids=level_to_ref_ids,
                chunked_phonemes_batch=chunked_phonemes_batch,
                ref_chuncked_phonemes_batch=ref_chuncked_phonemes_batch,
                phonemes_units=phonemes_units,
                pad_idx=PAD_TOKEN_IDX,
            )
            
            # Overwrite the fast-pooling results with highly accurate DP results
            for level in dp_levels:
                if level != "phonemes":
                    level_to_units[level] = dp_level_to_units[level]

        sifat_batch: list[list[Sifa]] = format_sifat(
            level_to_units,
            chunked_phonemes_batch,
            self.multi_level_tokenizer,
        )

        output = []
        # looping over the batch
        for idx in range(len(level_to_units["phonemes"])):
            output.append(
                MuaalemOutput(
                    phonemes=level_to_units["phonemes"][idx],
                    sifat=sifat_batch[idx],
                )
            )
        return output
    
    def _level_id_to_text(self, level: str, level_ids: list[int]) -> str:
        return "".join(
            [self.multi_level_tokenizer.id_to_vocab[level][int(idx)] for idx in level_ids]
        )

    def _decode_phonemes(self, phoneme_logits, input_lengths: list[int]) -> list[Unit]:
        phonemes_units = []
        for seq_idx, logits in enumerate(phoneme_logits):
            actual_len = input_lengths[seq_idx]
            
            decoded_ids, decoded_probs, decoded_frames = self.decoder(logits, 0, actual_len)
            
            text = self._level_id_to_text("phonemes", decoded_ids)

            phonemes_units.append(
                Unit(text=text, probs=decoded_probs, ids=decoded_ids, frames=decoded_frames)
            )
        return phonemes_units
    
    def _decode_sifat(
        self,
        levels_logits,
        phonemes_units: list[Unit],
        chunked_phonemes_batch: list[list[str]],
    ) -> dict:
        """
        Decode sifat by sampling frames at the first token of each phoneme chunk.

        we sample only at the first token of each phoneme chunk. This ensures:
          - len(chunks) entries per sequence
          - the right access in format_sifat's ids[chunk_idx]

        Contrast with the original quran_muaalem approach which:
          - CTC-decodes each sifat head independently
          - Runs DP alignment
        This approach avoids both steps and is cheaper for streaming.
        """

        level_to_units = {"phonemes": phonemes_units}

        for level in levels_logits:
            if level == "phonemes":
                continue
            
            level_to_units[level] = []

            for (logits, phoneme_unit, chunks) in zip(levels_logits[level], phonemes_units, chunked_phonemes_batch):
                # Find the character token index of the first character of each chunk.
                # phoneme_unit.frames[char_idx] gives the CTC frame for that token.
                # We use only the base consonant (first char of each chunk).
                chunk_token_idxs = self._chunk_first_token_indices(phoneme_unit.text, chunks)

                aligned_ids   = []
                aligned_probs = []
                chunk_frames  = []

                for token_idx in chunk_token_idxs:
                    if token_idx < len(phoneme_unit.frames):
                        start_frame = phoneme_unit.frames[token_idx]
                        
                        # Center exactly on the phoneme spike with a very tight window
                        curr_frame = phoneme_unit.frames[token_idx]
                        buffered_start = max(0, curr_frame - 1)
                        buffered_end = min(logits.shape[0], curr_frame + 2)
                        buffered_end = max(buffered_end, buffered_start + 1)
                        
                        if buffered_start < logits.shape[0]:
                            # Windowed max pooling
                            window_logits = logits[buffered_start:buffered_end].float().cpu()
                            pooled_logits = window_logits.max(dim=0).values
                            
                            probs_at = torch.nn.functional.softmax(pooled_logits, dim=-1)
                            
                            # Ignore PAD (index 0) since a phoneme must have a valid Sifat
                            valid_probs = probs_at[1:]
                            prob, token_id_offset = valid_probs.max(dim=-1)
                            token_id = token_id_offset + 1
                            
                            aligned_ids.append(token_id.item())
                            aligned_probs.append(prob.item())
                            chunk_frames.append(start_frame)
                            continue
                    
                    # Fallback: pad if out-of-range
                    aligned_ids.append(0)
                    aligned_probs.append(0.0)
                    chunk_frames.append(-1)

                # Added to fix the wrong sifat prefiction in case of a char in mot in the defined set
                if level in SIFAT_POSITIVE_PHONEMES and _SIFAT_NEGATIVE_LABEL[level] is not None:
                    aligned_ids = self._apply_sifat_constraint(
                        level, chunks, aligned_ids,
                        self.multi_level_tokenizer,
                    )

                text = self._level_id_to_text(level, aligned_ids)
                level_to_units[level].append(Unit(
                    text=text,
                    probs=aligned_probs,
                    ids=aligned_ids,
                    frames=chunk_frames,
                ))

        return level_to_units

    @staticmethod
    def _chunk_first_token_indices(phoneme_text: str, chunks: list[str]) -> list[int]:
        """
        We reutrn the first token of the chunk representing the character not the haraka.
        """
        indices = []
        pointer = 0
        for chunk in chunks:
            idx = phoneme_text.find(chunk, pointer)
            if idx == -1:
                idx = pointer
            indices.append(idx)
            pointer = idx + len(chunk)
        return indices

    @staticmethod
    def _apply_sifat_constraint(level, chunks, aligned_ids, mlt):
        positive_set   = SIFAT_POSITIVE_PHONEMES[level]
        neg_label      = _SIFAT_NEGATIVE_LABEL[level]

        en_to_id       = {v: k for k, v in mlt.sifat_to_en_vocab[level].items()}
        neg_id         = en_to_id.get(neg_label, 0)

        corrected = list(aligned_ids)
        for i, chunk in enumerate(chunks):
            if i >= len(corrected):
                break

            base = chunk[0]
            if base not in positive_set:
                # This chunk CANNOT have the positive sifat , we force negative
                corrected[i] = neg_id
        return corrected