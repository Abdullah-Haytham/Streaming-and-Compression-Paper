"""
streaming_event_tracer.py
═════════════════════════

Per-event tracing for AdaptiveStreamingMuaalem. Captures each
EXPAND / COMMIT / FLUSH decision the streamer makes during a run,
including the heuristic state at decision time (which of H1/H2 fired,
trailing-run length, last peak frame, frame_end).

This is the single source of truth used by:
  - sweep_adaptive_dataset.py — writes events.jsonl alongside results.csv
  - demo_adaptive_streaming.py — can register a print callback for live output

Why a shared module?
────────────────────
Both the sweep and the demo need to introspect the streamer's per-decision
state, but neither wants to fork streaming_inference.py to add hooks. The
tracer safely hooks the native `on_milestone` callback which now directly
emits the full decision payload constructed by `_should_commit`, including
all window metrics like run length, expansion counts, and heuristic reasons.

The tracer also collects sifat outputs across all commits so callers don't
need to re-iterate.

Usage
─────
    from streaming_event_tracer import run_with_event_capture

    events, hyp_text, all_sifat = run_with_event_capture(
        streamer, audio,
        on_event=lambda ev: print(ev)   # optional live callback
    )
    # `events` is a list of StreamingEvent (dataclass with `.as_dict()`)
    # `hyp_text` is streamer.full_text after the run
    # `all_sifat` is the concatenated list of MuaalemOutput.sifat entries
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Callable, List, Optional, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
#  Event model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StreamingEvent:
    """
    One per-decision event from AdaptiveStreamingMuaalem.

    Reason values
    ─────────────
    For EXPAND events:
        "h1"      — only H1 (peak near right edge) fired
        "h2"      — only H2 (trailing-run >= threshold) fired
        "h1+h2"   — both fired

    For COMMIT events:
        "safely-inside"  — model converged; peaks comfortably inside chunk
        "no-tokens"      — decoder emitted nothing for this window
        "max-expansions" — hit expansion cap, forced commit

    For FLUSH events:
        "flush"          — end-of-audio flush
    """
    t: float                      # audio cursor at the time of decision (seconds)
    type: str                     # "EXPAND" | "COMMIT" | "FLUSH"
    reason: str                   # see docstring above
    ec_after: int = 0             # expansion_count after this decision
    trailing_run: int = 0         # trailing-run length on decoded sequence
    last_frame: int = -1          # last peak frame in decoded sequence (or -1)
    frame_end: int = -1           # window.frame_end at decision time
    n_tok: int = 0                # tokens emitted (COMMIT / FLUSH only)
    text: str = ""                # emitted text (COMMIT / FLUSH only)
    seam_text: str = ""           # seam-recovered text prepended to commit, if any

    def as_dict(self) -> dict:
        return asdict(self)

    def to_persisted_dict(self) -> Optional[dict]:
        """
        Minimal dict for events.jsonl persistence.
        For EXPAND events, persists only type, reason, and t (no text/tokens).
        For COMMIT/FLUSH, keeps type, reason, text, t, and seam_text.

        The full dataclass (with timing, frame, and ec_after state) remains
        available in-memory for callbacks that want richer output (e.g. the
        demo's per-event print).
        """
        if self.type == "EXPAND":
            return {"type": self.type, "reason": self.reason, "t": self.t}
        out = {"type": self.type, "reason": self.reason, "text": self.text, "t": self.t}
        if self.seam_text:
            out["seam_text"] = self.seam_text
        return out


# ─────────────────────────────────────────────────────────────────────────────
#  Tracer
# ─────────────────────────────────────────────────────────────────────────────

def run_with_event_capture(
    streamer,
    audio: np.ndarray,
    push_size: Optional[int] = None,
    on_event: Optional[Callable[[StreamingEvent], None]] = None,
) -> Tuple[List[StreamingEvent], str, list]:
    cfg = streamer.cfg
    if push_size is None:
        push_size = max(int(cfg.expansion_samples), 1)

    events: List[StreamingEvent] = []
    audio_cursor_s = 0.0

    orig_on_milestone = getattr(streamer, "on_milestone", None)

    def tracer_callback(payload: dict):
        nonlocal audio_cursor_s
        is_commit = payload.get("should_commit", False)
        ev_type = "COMMIT" if is_commit else "EXPAND"
        
        # Calculate tokens emitted by comparing text lengths (since token_ids is gone from payload)
        # We can just count the text length roughly, or leave n_tok as 0 since only full text matters now
        # Actually, payload["text"] has the raw window text.
        raw_text = payload.get("text", "")
        
        ev = StreamingEvent(
            t=audio_cursor_s,
            type=ev_type,
            reason=payload.get("reason", "?"),
            ec_after=payload.get("expansion_count", 0) + (1 if not is_commit else 0),
            trailing_run=payload.get("trailing_run", 0),
            last_frame=payload.get("last_frame", -1),
            frame_end=payload.get("frame_end", -1),
            n_tok=len(raw_text) if is_commit else 0,
            text=raw_text if is_commit else "",
            seam_text="" # Seam text tracking removed as it's purely internal now
        )
        events.append(ev)
        if on_event is not None:
            on_event(ev)
            
        if orig_on_milestone:
            orig_on_milestone(payload)

    try:
        streamer.reset()
        streamer.on_milestone = tracer_callback

        for start in range(0, len(audio), push_size):
            sl = audio[start:start + push_size]
            audio_cursor_s += len(sl) / cfg.sampling_rate
            streamer.process(sl)

        # End-of-audio flush
        pre_flush_ec = streamer._expansion_count
        streamer.flush()
        
        # Add a flush event
        ev = StreamingEvent(
            t=audio_cursor_s, type="FLUSH", reason="flush",
            ec_after=pre_flush_ec,
            n_tok=0,
            text="",
        )
        events.append(ev)
        if on_event is not None:
            on_event(ev)

    finally:
        streamer.on_milestone = orig_on_milestone

    return events, streamer.full_text, streamer.full_sifat
