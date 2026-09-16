
# This is a customized version of the app 
# https://github.com/obadx/quran-muaalem/blob/main/src/quran_muaalem/inference.py
# We follow the sa

from ctc_decoder import CTCDecoder, GreedyCTCDecoder
from nutq import QuranMultiCTC, VocabTokenizer

from imports.utils import chunck_phonemes, format_sifat
from imports.muaalem_typing import Unit, MuaalemOutput

import torch
from transformers import AutoFeatureExtractor

class Nutq:
    def __init__(self, model_name_or_path = "obadx/muaalem-model-v3_2", decoder: CTCDecoder | None = None, device = "cpu"):

        self.device  = device
        self.decoder = decoder if decoder is not None else GreedyCTCDecoder(blank_id=0)

        if str(model_name_or_path).endswith(".pt"):
            # Loading the model
            ckpt = torch.load(model_name_or_path, map_location="cpu", weights_only=False)
            self._sample_rate = ckpt.get("sample_rate", 16_000)

            self.model = QuranMultiCTC(ckpt["base_model_id"], ckpt["vocabs"], load_pretrained_encoder=False)
            self.model.load_state_dict(ckpt["model_state"])
            self.model.to(device, dtype=torch.bfloat16)

            # Initiating the tokenizer alongside the feature extractor
            self.tokenizer = VocabTokenizer(ckpt["idx2sym"], ckpt["vocabs"])
            self.ft_extractor = AutoFeatureExtractor.from_pretrained(ckpt["base_model_id"])
        else:
            from imports.modeling.modeling_multi_level_ctc import Wav2Vec2BertForMultilevelCTC
            from imports.modeling.multi_level_tokenizer import MultiLevelTokenizer
            
            self.model = Wav2Vec2BertForMultilevelCTC.from_pretrained(model_name_or_path)
            self.model.to(device, dtype=torch.bfloat16)
            self.tokenizer = MultiLevelTokenizer(model_name_or_path)
            self.ft_extractor = AutoFeatureExtractor.from_pretrained(model_name_or_path)
            self._sample_rate = 16_000

        self.last_inference_logits = None

    @torch.inference_mode()
    def __call__(self, waves, extract_sifat: bool = True) -> list[MuaalemOutput]:

        # Extract the features from the audio (Log-Mel Spectrogram)
        # Returns a dict with 'input_features' of shape (batch, n_frames, n_mels) and 'attention_mask'
        features = self.ft_extractor(waves, sampling_rate=self._sample_rate, return_tensors="pt")

        input_features = features['input_features'].to(self.device, dtype=torch.bfloat16)
        attention_mask = features['attention_mask'].to(self.device)
        # model returns (log_probs_dict, input_lengths_tensor)
        if isinstance(self.model, QuranMultiCTC) or str(type(self.model).__name__) == "ONNXModelWrapper":
            model_out = self.model(input_features, attention_mask)
            
            # Fix formating from (n_frames, batch, vocab_probs) to (batch, n_frames, vocab_probs)
            levels_logits = model_out[0]
            for level in levels_logits:
                levels_logits[level] = levels_logits[level].transpose(0, 1)
            
            input_lengths = model_out[1].tolist()
        else:
            # HF model
            model_out = self.model(input_features=input_features, attention_mask=attention_mask, return_dict=False)
            levels_logits = model_out[0]
            
            if attention_mask is not None:
                input_lengths = self.model._get_feat_extract_output_lengths(attention_mask.sum([-1])).tolist()
            else:
                input_lengths = [levels_logits["phonemes"].shape[1]] * len(waves)
            
        self.last_inference_logits = levels_logits
        
        phonemes_units = self._decode_phonemes(levels_logits["phonemes"], input_lengths)

        if extract_sifat:
            sifat = self.get_sifat_for_units(levels_logits, phonemes_units)
        else:
            sifat = [[] for _ in phonemes_units]

        output = []
        for idx in range(len(phonemes_units)):
            output.append(
                MuaalemOutput(
                    phonemes = phonemes_units[idx],
                    sifat    = sifat[idx],
                )
            )
        return output
    
    def level_id2txt(self, level, decoded_ids):
        result = []
        for id in decoded_ids:
            result.append(self.tokenizer.id_to_vocab[level].get(id, ""))
        return "".join(result)

    def _decode_phonemes(self, phoneme_logits, input_lengths) -> list[Unit]:
        phonemes_units = []
        
        for seq_idx, logits in enumerate(phoneme_logits):
            decoded_ids, decoded_probs, decoded_frames = self.decoder(logits, 0, input_lengths[seq_idx])
            phonemes_units.append(
                Unit(
                    text   = self.level_id2txt("phonemes", decoded_ids), 
                    probs  = decoded_probs, 
                    ids    = decoded_ids, 
                    frames = decoded_frames
                )
            )

        return phonemes_units
    
    def _decode_sifat(self, levels_logits, phonemes_units: list[Unit], chunked_phonemes_batch) -> dict:

        level_to_units = {"phonemes": phonemes_units}

        for level in levels_logits:
            if level == "phonemes":
                continue
            
            level_to_units[level] = []

            for (logits, phoneme_unit, chunks) in zip(levels_logits[level], phonemes_units, chunked_phonemes_batch):
                
                # We use only the base consonant (first char of each chunk).
                token_idxs = self._chunk_first_token_indices(chunks)
                
                aligned_ids   = []
                aligned_probs = []
                chunk_frames  = []
                base_chars    = []

                for token_idx in token_idxs:
                    
                    if token_idx < len(phoneme_unit.frames):
                        base_chars.append(phoneme_unit.text[token_idx]) # constructed but used later to apply sifat constraint
                        
                        # Center exactly on the phoneme spike with a very tight window
                        curr_frame = phoneme_unit.frames[token_idx]

                        window_start = max(0, curr_frame - 1)
                        window_end = min(logits.shape[0], curr_frame + 2)
                        window_end = max(window_end, window_start + 1)
                        
                        if window_start < logits.shape[0]:
                            # Windowed max pooling
                            window_logits = logits[window_start:window_end].float().cpu()
                            pooled_logits = window_logits.max(dim=0).values
                            
                            probs_at = torch.nn.functional.softmax(pooled_logits, dim=-1)
                            
                            # Ignore PAD index 0
                            prob, token_id_offset = probs_at[1:].max(dim=-1)
                            token_id = token_id_offset + 1
                            
                            aligned_ids.append(token_id.item())
                            aligned_probs.append(prob.item())
                            chunk_frames.append(curr_frame)
                            continue
                    
                    # Fallback: pad if out-of-range
                    curr_frame_debug = phoneme_unit.frames[token_idx] if token_idx < len(phoneme_unit.frames) else -1
                    print(f"[DEBUG] Fallback hit! token_idx: {token_idx}, len(frames): {len(phoneme_unit.frames)}, curr_frame: {curr_frame_debug}, logits_len: {logits.shape[0]}")
                    aligned_ids.append(0)
                    aligned_probs.append(0.0)
                    chunk_frames.append(-1)
                    base_chars.append("")

                # Added to fix the wrong sifat prefiction in case of a char in not in the defined set
                if hasattr(self.tokenizer, "apply_sifat_constraint"):
                    aligned_ids = self.tokenizer.apply_sifat_constraint(level, base_chars, aligned_ids)

                level_to_units[level].append(
                    Unit(
                        text   = self.level_id2txt(level, aligned_ids),
                        probs  = aligned_probs,
                        ids    = aligned_ids,
                        frames = chunk_frames
                    )
                )

        return level_to_units

    def get_sifat_for_units(self, levels_logits: dict, phonemes_units: list[Unit]):
        # For each audio, we process its phonetic string into groups of chunks of phonemes
        # You can check "chunk_phonemes" description for examples
        chunked_phonemes_batch = []
        for unit in phonemes_units:
            chunked_phonemes_batch.append(chunck_phonemes(unit.text))
            
        # Apply narrow window pooling to look for sifat around the phoneme frame
        level_to_units = self._decode_sifat(levels_logits, phonemes_units, chunked_phonemes_batch)

        # Unified formatting for o/p
        sifat = format_sifat(level_to_units, chunked_phonemes_batch, self.tokenizer)
        return sifat

    #### Helper functionss ######
    @staticmethod
    def _chunk_first_token_indices(chunks):
        """
        We return the first token of the chunk representing the character not the haraka.
        """
        indices = []
        pointer = 0
        for chunk in chunks:
            indices.append(pointer)
            pointer += len(chunk)
        return indices
    ############################