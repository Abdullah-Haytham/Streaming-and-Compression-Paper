from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
from nutq_core import Nutq, MuaalemOutput
from imports.muaalem_typing import Sifa, Unit
import torch
import copy
import numpy as np


# Default values where used as they are the same for our model
# Can be found 
# here https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/seamless_m4t/feature_extraction_seamless_m4t.py#L66
# and here https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/seamless_m4t/feature_extraction_seamless_m4t.py#L125
FEAT_EXTRACTOR_FRAME_LEN = 400
FEAT_EXTRACTOR_HOP_LEN   = 160
FEAT_EXTRACTOR_STRIDE    = 2

# Found here https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/wav2vec2_bert/configuration_wav2vec2_bert.py#L173
ADAPTER_KERNEL = 3
ADAPTER_STRIDE = 2
ADAPTER_PAD    = ADAPTER_KERNEL // 2  # https://github.com/huggingface/transformers/blob/da6c53e431f7c9ef0691239d4ce89b0f711ecad7/src/transformers/models/wav2vec2_bert/modeling_wav2vec2_bert.py#L775
ADAPTER_LAYERS = 1


## Start of data holders ##

@dataclass
class Window:
    audio:          np.ndarray
    frame_start:    int  # CTC-level, inclusive
    frame_end:      int  # CTC-level, exclusive
    audio_s:        float
    
    out:            Optional[MuaalemOutput] = None

@dataclass
class AdaptiveConfig:
    
    left_context_s:    float = 1.0
    base_chunk_s:      float = 1.5
    right_lookahead_s: float = 0.5
    right_pad_s:       float = 0.05

    expansion_s:     float = 0.5
    max_expansions:  int   = 3
    edge_frames:     int   = 3
    min_trailing_run_to_expand: int = 2
    
    seam_overlap_frames: int = 10
    seam_match_window:   int = 6 # number of tokens to match

    sampling_rate:       int = 16_000

    def __post_init__(self):
        if self.base_chunk_s <= 0:
            raise ValueError(f"base_chunk_s must be > 0, got {self.base_chunk_s}")
        if self.expansion_s <= 0:
            raise ValueError(f"expansion_s must be > 0, got {self.expansion_s}")
        if self.max_expansions < 0:
            raise ValueError(f"max_expansions must be >= 0, got {self.max_expansions}")
        if self.right_lookahead_s < 0:
            raise ValueError(f"right_lookahead_s must be >= 0, got {self.right_lookahead_s}")
        if self.sampling_rate <= 0:
            raise ValueError(f"sampling_rate must be > 0, got {self.sampling_rate}")
    
    @property
    def left_samples(self) -> int:
        return int(self.left_context_s * self.sampling_rate)
    
    @property
    def base_chunk_samples(self) -> int:
        return int(self.base_chunk_s * self.sampling_rate)

    @property
    def right_lookahead_samples(self) -> int:
        return int(self.right_lookahead_s * self.sampling_rate)

    @property
    def right_pad_samples(self) -> int:
        return int(self.right_pad_s * self.sampling_rate)

    @property
    def expansion_samples(self) -> int:
        return int(self.expansion_s * self.sampling_rate)

    # For debugging
    @property
    def worst_case_latency_s(self) -> float:
        return (self.base_chunk_s + self.max_expansions * self.expansion_s + self.right_lookahead_s)

## End of data holders ##

## Start of Wrappers ##
class _StreamingDecoderWrapper:
    def __init__(self, base_decoder, keep_logits = True):
        self.win = None
        self.base_decoder = base_decoder

        self.keep_logits = keep_logits
        self.last_logits = None

    def __call__(self, logits, start, end):

        # Used for seam recovery
        if self.keep_logits:
            self.last_logits = logits

        # Fallback to the default behaviour (decoding the whole record) if no window specified
        if self.win is None:
            return self.base_decoder(logits, start, end)

        return self.base_decoder(logits, self.win.frame_start, self.win.frame_end)

## End of Wrappers ##

class AdaptiveChunkBuffer:
    def __init__(self, cfg: AdaptiveConfig):
        self.cfg       = cfg
        self.left_ctx  = []
        self.chunk     = []

    def push(self, audio):
        self.chunk.extend(audio.tolist())

    def curr_chunk_size(self):
        return len(self.chunk)

    @property
    def left_ctx_frames(self):
        return self.samples_to_ctc_frames(len(self.left_ctx))

    def build_window(self, base_chunk_samples, no_pad = False):
        if base_chunk_samples <= 0 or not self.chunk:
            return None

        chunk_len = min(base_chunk_samples, len(self.chunk))

        if no_pad:
            right_pad = []
        else:
            right_pad = [0.0] * self.cfg.right_pad_samples

        window_audio = self.left_ctx + self.chunk[:chunk_len + self.cfg.right_lookahead_samples] + right_pad
        frame_start  = self.left_ctx_frames
        frame_end    = self.samples_to_ctc_frames(len(self.left_ctx) + chunk_len)

        frame_end = max(frame_end, frame_start + 1)

        audio_arr = np.array(window_audio, dtype=np.float32)
        return Window(
            audio=audio_arr,
            frame_start=frame_start,
            frame_end=frame_end,
            audio_s=len(audio_arr) / self.cfg.sampling_rate
        )

    # Slides the window and updates the left_ctx based on committed CTC frames
    def slide_by_frames(self, num_frames: int):
        n = self.ctc_frames_to_samples(num_frames)
        
        if n <= 0 or not self.chunk:
            return

        committed = self.left_ctx + self.chunk[:n]

        if self.cfg.left_samples > 0:
            self.left_ctx = committed[-self.cfg.left_samples:]
        else:
            self.left_ctx = []

        # Drop the committed samples from the chunk
        self.chunk = self.chunk[n:]

    def reset(self):
        self.left_ctx = []
        self.chunk = []
    
    @staticmethod
    def ctc_frames_to_samples(num_frames: int) -> int:
        """Inverse mapping: CTC frames -> original wav samples."""
        frame_stride = FEAT_EXTRACTOR_HOP_LEN * FEAT_EXTRACTOR_STRIDE * (ADAPTER_STRIDE ** ADAPTER_LAYERS)
        return num_frames * frame_stride

    @staticmethod
    def samples_to_ctc_frames(n_samples):
        """
        i/p (wav) -> feature extractor -> stride reduction -> adapter convolution -> o/p (CTC frames)
        """

        # The feature extractor converts from raw audio to feature frames (Log-Mel Spectrogram)
        # So we calculate the reduction in the input size with the values of 
        # window size (frame_length), the slide length (hop_length), and the stride
        # Reference for mel-spectogram calculations: https://www.youtube.com/watch?v=hF72sY70_IQ&list=PLoEMreTa9CNlQpbAJHTQSSym1Gw9mWNV3
        if n_samples < FEAT_EXTRACTOR_FRAME_LEN:
            return 0
        feat_extractor_output_len = (n_samples - FEAT_EXTRACTOR_FRAME_LEN) // FEAT_EXTRACTOR_HOP_LEN + 1

        # The feature extractor reduces the frames one more time by concating frames into chunks of 2 (stride)
        feat_extractor_output_len = feat_extractor_output_len // FEAT_EXTRACTOR_STRIDE

        if feat_extractor_output_len <= 0:
            return 0
        
        # The input size is further reduced after applying the convolution in the adapter layer
        # Reference: https://en.wikipedia.org/wiki/Convolutional_neural_network#Spatial_arrangement
        # eqn of conv output: (W-K+2P)/S + 1 where w = input length, k = kernel size, p = padding, s = stride
        init_len = feat_extractor_output_len
        for _ in range(ADAPTER_LAYERS):
            total_canvas = init_len + 2 * ADAPTER_PAD
            init_len = (total_canvas - ADAPTER_KERNEL) // ADAPTER_STRIDE + 1
            if init_len <= 0:
                return 0
        return init_len


class AdaptiveStreamingMuaalem:

    def __init__(self, nutq: Nutq, cfg: AdaptiveConfig):
        self.cfg = cfg
        self._nutq = copy.copy(nutq)
        self._nutq.decoder = _StreamingDecoderWrapper(self._nutq.decoder)
        self.reset()

        # Used for debugging. Attach a hook here
        self.on_milestone = None

    def reset(self):
        self._buffer = AdaptiveChunkBuffer(self.cfg)

        self._accumulated_ids    = []
        self._accumulated_frames = []
        
        # For sifat decoding
        self._global_logits_list = {}
        self._global_frames_len  = 0

        # Parameters tracking internal state
        self._local_reset()

        # For debugging
        self._n_h1_only_expansions   = 0
        self._n_h2_only_expansions   = 0
        self._n_h1h2_both_expansions = 0
        self._n_force_committed      = 0
        self._last_seam_ids          = []

    def _local_reset(self):
        # Track number of expansions
        self._expansion_count = 0
        
        # For early emissions on match before commit.
        self._tentative_ids     = []
        self._tentative_frames  = []
    
    @property
    def full_text(self) -> str:
        return self._nutq.level_id2txt("phonemes", self._accumulated_ids)

    @property
    def tentative_text(self) -> str: # committed text + tentative
        return self.full_text + self._nutq.level_id2txt("phonemes", self._tentative_ids)

    @property
    def full_sifat(self) -> List[Sifa]:
        return self._decode_global_sifat(include_tentative=False)

    @property
    def tentative_sifat(self) -> List[Sifa]:
        return self._decode_global_sifat(include_tentative=True)

    def process(self, audio: np.ndarray):
        self._buffer.push(self._normalize_audio(audio))
        self._maybe_decode_loop()

    def flush(self, no_pad: bool = False):
        win = self._buffer.build_window(base_chunk_samples=self._buffer.curr_chunk_size(), no_pad=no_pad)
        
        if win is None:
            return

        win.out = self._run_inference(win)
        self._commit_window(win)

    # ── Internals ────────────────────────────────────────────────────────

    def _maybe_decode_loop(self):

        # Keep processing until no more audio is in the buffer
        while True:
            target_len = self.cfg.base_chunk_samples + self._expansion_count * self.cfg.expansion_samples
            
            # Not enough audio for a window, wait for more
            if self._buffer.curr_chunk_size() < target_len + self.cfg.right_lookahead_samples:
                break 

            win = self._buffer.build_window(base_chunk_samples=target_len)

            win.out = self._run_inference(win)
            ids     = win.out.phonemes.ids
            frames  = win.out.phonemes.frames
            
            # Update current local state
            self._tentative_ids     = ids
            self._tentative_frames  = frames

            milestone_payload = self._should_commit(win)

            if self.on_milestone:
                self.on_milestone(milestone_payload)

            if milestone_payload["should_commit"]:
                self._commit_window(win)
            else:
                self._expansion_count += 1

        return
    
    def _run_inference(self, win: Window):
        """Run one inference call bounded by the window's frame range."""
        self._nutq.decoder.win = win
        try:
            output = self._nutq([win.audio], extract_sifat=False)[0]
        finally:
            self._nutq.decoder.win = None

        return output
    
    
    # o/p format commit, reason.
    def _should_commit(self, win: Window) -> dict:

        ids     = win.out.phonemes.ids
        frames  = win.out.phonemes.frames
        
        payload = {
            "text": win.out.phonemes.text,
            "last_frame": frames[-1] if frames else -1,
            "frame_end": win.frame_end,
            "expansion_count": self._expansion_count,
            "trailing_run": self._trailing_run_length(ids),
            "should_commit": True,
            "reason": "safely-inside"
        }

        if not ids:
            payload["reason"] = "no-tokens"
            return payload
        

        # H2, checks if the window ends with repeating tokens. e.g. madd, ghunna.
        # NOTE: I moved this block before the max-expansion cap to get the exact real-time physical state of the final window.
        # If it was after the cap, we would skip the check and export wrong state to the VAD.
        
        self.is_elongating = False # Intentionally excluded from _local_reset() so the state persists across forced commits.
        if self.cfg.min_trailing_run_to_expand > 0:
            if payload["trailing_run"] >= self.cfg.min_trailing_run_to_expand:
                self.is_elongating = True
        # End of H2 check
        
        # Max Cap, forced to commit if worst expansion-budget is consumed.
        if self._expansion_count >= self.cfg.max_expansions:
            payload["reason"] = "max-expansions"
            self._n_force_committed += 1
            return payload

        # H1: a decoded token (peak) near the right edge
        h1 = len(frames) > 0 and frames[-1] >= win.frame_end - self.cfg.edge_frames

        if h1 and self.is_elongating: 
            payload["should_commit"] = False
            payload["reason"] = "h1+h2"
            self._n_h1h2_both_expansions += 1
            return payload
            
        if h1: 
            payload["should_commit"] = False
            payload["reason"] = "h1"
            self._n_h1_only_expansions += 1
            return payload
            
        if self.is_elongating: 
            payload["should_commit"] = False
            payload["reason"] = "h2"
            self._n_h2_only_expansions += 1
            return payload
            
        return payload

    def _trailing_run_length(self, ids):
        if not ids:
            return 0
            
        last_token = ids[-1]
        run_length = 0
        
        for token in reversed(ids):
            if token != last_token:
                break
            run_length += 1
            
        return run_length

    def _commit_window(self, win: Window):
        # re-index relative window frames to absolute global frames and applies seam recovery
        full_ids, full_frames = self._stitch_window_to_global(win)
        full_text = self._nutq.level_id2txt("phonemes", full_ids)
        
        # Update global state (full audio stream)
        self._accumulated_ids.extend(full_ids)
        self._accumulated_frames.extend(full_frames)
        curr_win_logits = self._nutq.last_inference_logits
        if curr_win_logits is not None:
            if not self._global_logits_list:
                self._global_logits_list = {level: [] for level in curr_win_logits if level != "phonemes"}
            
            for level in self._global_logits_list.keys():
                self._global_logits_list[level].append(curr_win_logits[level][:, win.frame_start:win.frame_end, :].cpu())
                
            self._global_frames_len += (win.frame_end - win.frame_start)

        # Slide and reset local state (window) only after successful global stitch
        self._buffer.slide_by_frames(win.frame_end - win.frame_start)
        self._local_reset()

        return full_ids, full_text
    
    # start of post-processing functions #
    def _stitch_window_to_global(self, win: Window):
        
        def _map_frames(frames):
            new_frames = []
            for f in frames:
                new_f = self._global_frames_len + (f - win.frame_start) # the absolute global indicies
                new_frames.append(new_f)
            return new_frames

        full_ids    = win.out.phonemes.ids
        full_frames = _map_frames(win.out.phonemes.frames)

        if self.cfg.seam_overlap_frames > 0:
            seam_ids, seam_frames = self._recover_seam_tokens(win)
            if seam_ids:
                self._last_seam_ids = seam_ids
                full_ids = seam_ids + full_ids
                full_frames = _map_frames(seam_frames) + full_frames
            else:
                self._last_seam_ids = []
        else:
            self._last_seam_ids = []

        return full_ids, full_frames

    def _recover_seam_tokens(self, win: Window):
 
        if not self._accumulated_ids:
            return [], []  # nothing to anchor against

        phonemes_logits = self._nutq.last_inference_logits["phonemes"]

        ov_start = max(0, win.frame_start - self.cfg.seam_overlap_frames)

        if win.frame_start <= ov_start:
            return [], []

        # Extract tokens and their absolute frames for the overlap region
        ov_ids, _, ov_frames = self._nutq.decoder.base_decoder(phonemes_logits[0], ov_start, win.frame_start)
        
        if not ov_ids:
            return [], []

        seam_ids = self._suffix_prefix_recover(overlap_tokens=ov_ids)
        
        if not seam_ids:
            return [], []
            
        # The recovered seam corresponds exactly to the trailing N tokens of the overlap
        return seam_ids, ov_frames[-len(seam_ids):]

    def _suffix_prefix_recover(self, overlap_tokens: list) -> list:
        """
        Return overlap_tokens[k:] - the "newly recovered" tokens beyond the match.
        """
        for k in range(self.cfg.seam_match_window, 0, -1):
            if self._accumulated_ids[-k:] == overlap_tokens[:k]:
                return overlap_tokens[k:]
        return []
    
    def _decode_global_sifat(self, include_tentative: bool) -> List[Sifa]:
        ids = list(self._accumulated_ids)
        frames = list(self._accumulated_frames)
        text = self.full_text

        if include_tentative and self._tentative_ids:
            ids.extend(self._tentative_ids)
            frames.extend(self._tentative_frames)
            text = self.tentative_text

        if not ids:
            return []

        global_unit = Unit(text=text, probs=[], ids=ids, frames=frames)
        
        concat_logits = {}
        last_logits = self._nutq.last_inference_logits
        win_start = self._buffer.left_ctx_frames
            
        for level, level_tensors in self._global_logits_list.items():
            tensors = list(level_tensors)
            
            if include_tentative and last_logits is not None:
                tensors.append(last_logits[level][:, win_start:, :].cpu())

            if not tensors:
                return []
                
            concat_logits[level] = torch.cat(tensors, dim=1)
                
        sifat_nested = self._nutq.get_sifat_for_units(concat_logits, [global_unit])
        return sifat_nested[0] if sifat_nested else []
    
    # end of post-processing function #
    
    @classmethod
    def from_pretrained(cls, model_name_or_path = "obadx/muaalem-model-v3_2", cfg = None, device = "cpu"):
        nutq = Nutq(model_name_or_path=model_name_or_path, device=device)
        return cls(nutq, cfg or AdaptiveConfig())
    
    # Helpers
    # casts audio to float32 and normalize to range [-1, 1], the expected i/p format to feature extractor
    @staticmethod
    def _normalize_audio(audio: np.ndarray) -> np.ndarray:
        audio = np.asarray(audio)
        if audio.dtype == np.int16:
            return audio.astype(np.float32) / 32768.0
        return audio.astype(np.float32)

    # For debugging/benchmarking
    def batch_process(self, audio: np.ndarray):
        """Full-audio inference in one shot"""
        self.reset()
        self._buffer.push(self._normalize_audio(audio))
        self.flush(no_pad=True)
    
    def stream_file(self, audio: np.ndarray, push_chunk_size = None):
        """Simulates streaming"""
        self.reset()
        audio = self._normalize_audio(audio)
        push_chunk_size = push_chunk_size or self.cfg.expansion_samples
        for start in range(0, len(audio), push_chunk_size):
            yield self.process(audio[start:start + push_chunk_size])
        yield self.flush()
    
    def get_tentative_time_consumed(self, chars: int) -> float:
        """
        Calculate the audio time (in seconds) within the current chunk 
        corresponding to the first `chars` characters of `_tentative_text`.
        """
        if chars <= 0 or not self._tentative_ids or not self._tentative_frames:
            return 0.0

        idx = min(chars, len(self._tentative_frames)) - 1
        last_frame = self._tentative_frames[idx]

        frames_in_chunk = max(0, last_frame - self._buffer.left_ctx_frames)
        return frames_in_chunk * 0.04  # each CTC frame is 40ms


