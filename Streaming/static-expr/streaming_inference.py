"""
Model internals
─────────────────────────────────────────
quran_muaalem.Muaalem:
 - model                  Wav2Vec2BertForMultilevelCTC
 - processor              SeamlessM4TFeatureExtractor
 - multi_level_tokenizer  MultiLevelTokenizer

Forward: model(**features, return_dict=False)[0]
 - dict {"phonemes": Tensor(B, T_ctc, vocab), }

Blank id: PAD_TOKEN_IDX = 0

Adapter:  adapter_stride=2, adapter_kernel_size=3
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from nutq_core import Nutq, MuaalemOutput

import copy
import numpy as np
import torch

from ctc_decoder import GreedyCTCDecoder

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

BLANK_ID = 0

def samples_to_ctc_frames(n_samples):
    """
    For an input of n_samples, we return the size of the model output (CTC logits) to define the center frames
    """

    # The feature extractor converts from raw audio to feature frames (Log-Mel Spectrogram)
    # So we calculate the reduction in the input size with the values of 
    # window size (frame_length), the slide length (hop_length), and the stride
    # Reference for mel-spectogram calculations: https://www.youtube.com/watch?v=hF72sY70_IQ&list=PLoEMreTa9CNlQpbAJHTQSSym1Gw9mWNV3
    if n_samples < FEAT_EXTRACTOR_FRAME_LEN:
        return 0
    feat_extractor_output_len = (n_samples - FEAT_EXTRACTOR_FRAME_LEN) // FEAT_EXTRACTOR_HOP_LEN + 1

    # The feature extractor reduces the frames one more time by concating frames into chunks of 2 (stride)
    feat_extractor_output_len = (feat_extractor_output_len - 1) // FEAT_EXTRACTOR_STRIDE + 1

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

@dataclass
class StreamingConfig:
    """
    Used to configure the sliding window parameters (chunk_length, left_context, right_context)
    """
    left_context_s:  float = 0.75
    chunk_length_s:  float = 1.5
    right_context_s: float = 0.75
    sampling_rate:   int   = 16_000 # The default sampling rate

    @property
    def left_samples(self) -> int:
        return int(self.left_context_s * self.sampling_rate)

    @property
    def chunk_samples(self) -> int:
        return int(self.chunk_length_s * self.sampling_rate)

    @property
    def right_samples(self) -> int:
        return int(self.right_context_s * self.sampling_rate)

    @property
    def center_frame_start(self) -> int:
        # The center frame starts at the end of the left context frames
        # So we convert the number of samples in the left context to frames
        return samples_to_ctc_frames(self.left_samples)

    @property
    def center_frame_end(self) -> int:
        # The same logic applies but at the end of the (left + chunk)
        return samples_to_ctc_frames(self.left_samples + self.chunk_samples)

    @property
    def window_length_s(self) -> float:
        return self.left_context_s + self.chunk_length_s + self.right_context_s

    @property
    def window_samples(self) -> int:
        return int(self.window_length_s * self.sampling_rate)

@dataclass
class Window:
    """
    Used to hold data of one window including its center frame indicies.
    """
    audio:          np.ndarray
    frame_start:    int  # CTC-level, inclusive
    frame_end:      int  # CTC-level, exclusive
    is_final:       bool = False


class ChunkBuffer:
    """
    Used to buffer audio chunks and produce windows for inference.
    """
    def __init__(self, cfg: StreamingConfig):
        self.cfg      = cfg
        self.left_ctx = [0.0] * cfg.left_samples # holds history
        self.buffer   = [] # current unprocesed chunks (center + right)
        self._has_real_left_ctx = False  # flag for the first left window

    def push(self, audio: np.ndarray):
        self.buffer.extend(audio.tolist())
        return list(self.process())

    def flush(self):
        """
        Used to process the remaining audio in the buffer after the last push
        """
        if len(self.buffer) == 0:
            return None

        actual_len = len(self.buffer)
        center = self.buffer

        # If the len < chunk size, we pad with zeros
        if actual_len < self.cfg.chunk_samples:
            center.extend([0.0] * (self.cfg.chunk_samples - actual_len))

        if self._has_real_left_ctx:

            # Pad the right_ctx as well with zeros
            right_ctx = [0.0] * self.cfg.right_samples

            window = np.array(self.left_ctx + center + right_ctx, dtype=np.float32)
            frame_start = self.cfg.center_frame_start
            frame_end = max(
                samples_to_ctc_frames(self.cfg.left_samples + actual_len),
                frame_start + 1,
            )
            
        else:
            # First (and only) window: skip zero-padded left context
            right_ctx = [0.0] * self.cfg.right_samples

            window = np.array(center + right_ctx, dtype=np.float32)
            frame_start = 0
            frame_end = max(
                samples_to_ctc_frames(actual_len),
                1,
            )
        
        # TODO: We should reset in case a user presses on the mic icon.
        self.buffer = []
        return Window(
                    audio       = window, 
                    frame_start = frame_start,
                    frame_end   = frame_end, 
                    is_final    = True
                )

    def reset(self):
        self.left_ctx   = [0.0] * self.cfg.left_samples
        self.buffer     = []
        self._has_real_left_ctx = False

    def process(self):
        min_len = self.cfg.chunk_samples + self.cfg.right_samples

        while len(self.buffer) >= min_len:
            center = self.buffer[:self.cfg.chunk_samples]
            right  = self.buffer[self.cfg.chunk_samples:min_len]

            if self._has_real_left_ctx:
                # Normal window: left_ctx contains real audio from previous chunk
                window = self.left_ctx + center + right
                frame_start = self.cfg.center_frame_start
                frame_end   = self.cfg.center_frame_end
            else:
                # First window: skip zero-padded left context to avoid
                # encoder hallucinations from processing silence
                window = center + right
                frame_start = 0
                frame_end   = samples_to_ctc_frames(self.cfg.chunk_samples)
                self._has_real_left_ctx = True
            
            yield Window(
                audio       = np.array(window, dtype=np.float32),
                frame_start = frame_start,
                frame_end   = frame_end
            )
            
            # we remove the processed center chunk from the buffer
            self.buffer   = self.buffer[self.cfg.chunk_samples:]

            # new left ctx is the chunk seen starting from the end of the center of the previous window
            self.left_ctx = (self.left_ctx + center)[-self.cfg.left_samples:]

# ── StreamingResult ───────────────────────────────────────────────────────────

@dataclass
class StreamingResult:
    text: str
    token_ids: List[int]
    is_final: bool
    full_text: str
    window_time_s: float
    muaalem_outputs: List[MuaalemOutput] = field(default_factory=list)


# ── StreamingMuaalem ──────────────────────────────────────────────────────────

class _StreamingDecoderWrapper:
    """
    Stateful adapter that enforces dynamic decoding boundaries for a single
    streaming window. Defaults to full tensor bounds if inactive.
    """
    def __init__(self, base_decoder):
        self.base_decoder = base_decoder
        self.win = None
        
    def __call__(self, logits, start, end):
        if self.win is None:
            return self.base_decoder(logits, start, end)
        
        return self.base_decoder(logits, self.win.frame_start, self.win.frame_end)


class StreamingMuaalem:
    """
    Streaming wrapper for the Muaalem Wav2Vec2-BERT CTC model (Phase 1+2).

    Usage:
        nutq = Nutq(device="cuda")
        streamer = StreamingMuaalem.from_nutq(nutq)
        streamer.reset()
        for chunk in mic_stream():
            result = streamer.process(chunk)
            if result.text: print(result.text, end="", flush=True)
        final = streamer.flush()
    """

    def __init__(self, nutq: Nutq, cfg: StreamingConfig):
        # Shallow-copy so we can swap .decoder without mutating the caller's Nutq.
        self._nutq = copy.copy(nutq)
        self.cfg  = cfg
        self._buffer: Optional[ChunkBuffer] = None
        self._accumulated_ids: List[int] = []
        
        self._base_decoder = self._nutq.decoder
        self._decoder_wrapper = _StreamingDecoderWrapper(self._base_decoder)
        self._nutq.decoder = self._decoder_wrapper

    @classmethod
    def from_nutq(cls, nutq_instance, cfg=None, device=None):
        """
        Build from a Nutq instance.
        """
        return cls(nutq_instance, cfg or StreamingConfig())

    @classmethod
    def from_pretrained(cls, model_name_or_path="obadx/muaalem-model-v3_2",
                        cfg=None, device="cpu", dtype=None):
        dtype = dtype or torch.bfloat16
        nutq = Nutq(
            model_name_or_path=model_name_or_path,
            device=device,
            dtype=dtype,
        )
        return cls.from_nutq(nutq, cfg=cfg, device=device)

    def reset(self):
        self._buffer = ChunkBuffer(self.cfg)
        self._accumulated_ids = []

    def process(self, audio: np.ndarray) -> StreamingResult:
        if self._buffer is None:
            raise RuntimeError("Call reset() before process()")
        return self._decode_windows(self._buffer.push(_normalize_audio(audio)))

    def flush(self) -> StreamingResult:
        if self._buffer is None:
            raise RuntimeError("Call reset() before flush()")
        win = self._buffer.flush()
        r = self._decode_windows([win] if win else [])
        return StreamingResult(text=r.text, token_ids=r.token_ids,
                               is_final=True, full_text=r.full_text,
                               window_time_s=r.window_time_s)

    def stream_file(self, audio: np.ndarray, chunk_size=None):
        self.reset()
        audio = _normalize_audio(audio)
        chunk_size = chunk_size or self.cfg.chunk_samples
        for start in range(0, len(audio), chunk_size):
            yield self.process(audio[start: start + chunk_size])
        yield self.flush()
    
    def batch_process(self, audio: np.ndarray) -> StreamingResult:
        self.reset()
        audio = _normalize_audio(audio)
        self._decoder_wrapper.win = None
        outputs = self._nutq([audio])

        ids  = outputs[0].phonemes.ids
        text = outputs[0].phonemes.text

        self._accumulated_ids.extend(ids)

        return StreamingResult(
            text=text,
            token_ids=ids,
            is_final=True,
            full_text=self.full_text,
            window_time_s=len(audio) / self.cfg.sampling_rate,
            muaalem_outputs=outputs
        )

    @property
    def full_text(self) -> str:
        return self._nutq._level_id_to_text("phonemes", self._accumulated_ids)

    @property
    def latency_s(self) -> float:
        return self.cfg.chunk_length_s + self.cfg.right_context_s

    def _decode_windows(self, windows) -> StreamingResult:
        new_ids, covered_s = [], 0.0
        if not windows:
            return StreamingResult(
                text="", token_ids=[], is_final=False, full_text=self.full_text, window_time_s=0.0
            )

        outputs = []
        new_text = ""
        
        for win in windows:
            # Tell the injected decoder to use this window's bounds
            self._decoder_wrapper.win = win
            try:
                out = self._nutq([win.audio])[0]
            finally:
                self._decoder_wrapper.win = None
            outputs.append(out)
            
            new_ids.extend(out.phonemes.ids)
            new_text += out.phonemes.text
            covered_s += len(win.audio) / self.cfg.sampling_rate
            
        self._accumulated_ids.extend(new_ids)
        
        return StreamingResult(text=new_text, token_ids=new_ids,
                               is_final=False, full_text=self.full_text,
                               window_time_s=covered_s,
                               muaalem_outputs=outputs)

# ── Utilities ─────────────────────────────────────────────────────────────────

def _normalize_audio(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio)
    if audio.dtype == np.int16:
        return audio.astype(np.float32) / 32768.0
    return audio.astype(np.float32)


def pick_config_for_latency(target_latency_s: float, context_ratio: float = 0.5) -> StreamingConfig:
    """Build a StreamingConfig targeting a specific end-to-end latency budget."""
    if target_latency_s < 0.1:
        raise ValueError("target_latency_s must be >= 0.1 s")

    # right_s = 1
    # chunk_s = 3
    # left_s  = 0.25

    right_s = target_latency_s * context_ratio
    chunk_s = target_latency_s - right_s
    left_s  = min(chunk_s * 0.5, 0.25)

    return StreamingConfig(chunk_length_s=chunk_s, left_context_s=left_s,
                           right_context_s=right_s)
