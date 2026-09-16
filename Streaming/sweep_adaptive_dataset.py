"""
sweep_adaptive_dataset.py
═════════════════════════
Sweep AdaptiveStreamingMuaalem configurations across the targeted golden
dataset built by sweep_streaming_advanced.build_targeted_golden_dataset().

This script focuses ONLY on AdaptiveStreamingMuaalem — it doesn't touch the
fixed-window StreamingMuaalem or KVCachedStreamingMuaalem. (KV-cache works
on the fixed-window pipeline and doesn't natively support adaptive's
expansion, lookahead, or seam-recovery features.)

Reference source
────────────────
Ground-truth phoneme + sifat references come from the manifest written by
build_targeted_golden_dataset, which carries the dataset's annotated
`phonemes` and `sifat` columns per row. These are the SAME canonical refs
gradio_app produces via
    quran_phonetizer(uthmani, MoshafAttributes(**moshaf_metadata),
                     remove_spaces=True)
PER measures streaming output vs. the Mushaf-derived expected pronunciation,
NOT vs. the model's own batch output. References are NOT re-fetched or
re-computed at sweep time — we trust the manifest.

Selection policy
────────────────
  primary metric : mean_per_inclusive (PER over ALL non-error samples,
                   fatals included with their actual PER — the "honest"
                   expected PER)
  tie-break      : worst_case_latency_s within `selection_per_epsilon`
                   of the best mean_per_inclusive
Rationale: ranking on conditional PER (the "what's PER when it works"
view) lets fatal-rich configs win by gaming the exclusion. Ranking on
inclusive PER is the right answer when accuracy is the priority.

Persistence
───────────
  - results.csv       : per-sample row appended after each run (Ctrl-C safe).
                        Includes H1/H2/H1+H2 expansion-reason counts and
                        force-commit count per sample.
  - events.jsonl      : one JSON object per run, containing the COMMIT and
                        FLUSH events (EXPAND events are dropped; their
                        aggregate H1/H2/H1+H2 counts go to results.csv).
                        Each event keeps only: type, reason, text, and
                        seam_text (when non-empty). This minimal schema
                        is sufficient for analyze_madd_accuracy.py's
                        per-commit fault attribution and is ~65% smaller
                        than dumping the full event state.
                        Self-contained — each line carries config_name,
                        sample_id, ref_phonemes, hyp_phonemes alongside
                        the events array.
  - aggregate.csv     : per-config aggregate. Mirrors
                        sweep_streaming_advanced.aggregate_global's columns
                        PLUS the following groups:
                          Behaviour — config-quality signal:
                            force_commit_mean   (pure undersizing indicator;
                                                 content-independent)
                          Behaviour — content-coupled diagnostics
                          (informative but NOT quality signals; don't rank
                          configs on these):
                            h1_only_mean, h2_only_mean, h1h2_both_mean,
                            expansion_rate    (total expansions per second)
                          Probabilistic failure representation:
                            n_total, n_fatal,
                            fatal_rate, fatal_rate_ci95_lo/hi  (Wilson 95%)
                            per_mean_inclusive, per_se,
                            per_mean_ci95_lo/hi
  - fatal_cases.jsonl : one JSON object per fatal sample
  - summary.json      : winner + selection policy + per-config aggregate

Usage
─────
    python sweep_adaptive_dataset.py --device cuda
    python sweep_adaptive_dataset.py --device cuda --quick       # smaller grid
    python sweep_adaptive_dataset.py --device cuda --max-samples 5

If a sample lacks ref_phonemes (older manifest), it is recorded as fatal
with error="no_reference_phonemes" and surfaces in fatal_cases.jsonl.
Re-build the targeted dataset to refresh the manifest.
"""


from __future__ import annotations

import torch
import argparse
import csv
import json
import logging
import statistics
import time
import traceback
import tracemalloc
import contextlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Generator
import soundfile as sf
import librosa
import os
import io
import random
import requests
from datasets import get_dataset_config_names

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

SIFAT_LEVELS: list[str] = [
    "hams_or_jahr", "shidda_or_rakhawa", "tafkheem_or_taqeeq",
    "itbaq", "safeer", "qalqla", "tikraar", "tafashie", "istitala", "ghonna",
]

# ── Data model (sweep-specific) ──────────────────────────────────────────────

@dataclass
class ConfigSpec:
    """A named AdaptiveConfig to evaluate."""
    name: str
    config: Any                  # AdaptiveConfig

    @property
    def worst_case_latency_s(self) -> float:
        return self.config.worst_case_latency_s


@dataclass
class SampleRecord:
    """One (config, sample) result row."""
    config_name: str
    sample_id: str
    mushaf: str
    audio_duration_s: float
    wall_time_s: float
    rtf: float
    e2e_latency_s: float
    peak_ram_mb: float
    worst_case_latency_s: float    # from config — bound, not measured
    apl_s: float                   # computed average phoneme latency
    acl_s: float                   # computed average chunk latency
    per: float
    S: int
    D: int
    I: int
    N: int
    sifat_accuracy: float          # macro-mean across SIFAT_LEVELS
    is_fatal: bool
    error: Optional[str]
    ref_phonemes: str
    hyp_phonemes: str
    # Expansion trigger breakdown (lifetime counts over this sample)
    n_expansions: int              # total expansions = h1_only+h2_only+h1h2
    n_h1_only_expansions: int      # H1 (edge) fired alone
    n_h2_only_expansions: int      # H2 (trailing run) fired alone
    n_h1h2_both_expansions: int    # both H1 and H2 fired
    n_force_committed: int         # commits caused by a safety cap

    @classmethod
    def fields(cls) -> List[str]:
        return [
            "config_name", "sample_id", "mushaf",
            "audio_duration_s", "wall_time_s", "rtf",
            "e2e_latency_s", "peak_ram_mb", "worst_case_latency_s",
            "apl_s", "acl_s",
            "per", "S", "D", "I", "N", "sifat_accuracy",
            "is_fatal", "error",
            "n_expansions", "n_h1_only_expansions",
            "n_h2_only_expansions", "n_h1h2_both_expansions",
            "n_force_committed",
            "ref_phonemes", "hyp_phonemes",
        ]


@dataclass
class SweepRunConfig:
    """Sweep-level policy knobs."""
    fatal_per_threshold:    float = 0.30
    early_stop_rate:        float = 0.25
    early_stop_min_samples: int   = 4
    selection_per_epsilon:  float = 0.005   # 0.5pp tolerance for winner tie-break
    pareto_per_epsilon:     float = 0.02    # within 2pp of best PER
    pareto_latency_tolerance_s: float = 0.1 # within 0.1s of min latency in band
    output_dir:             Path  = Path("sweep_results")
    csv_filename:           str   = "results.csv"
    aggregate_csv_filename: str   = "aggregate.csv"
    fatal_jsonl_filename:   str   = "fatal_cases.jsonl"
    events_jsonl_filename:  str   = "events.jsonl"
    summary_filename:       str   = "summary.json"


# ── HITLProfiler ──────────────────────────────────────────────────────────────

class HITLProfiler:
    """
    Hardware-In-The-Loop profiler.

    Wraps a streaming inference call to record:
      RTF          — wall_time / audio_duration
      Peak RAM     — peak memory allocation during the inference (MB, via tracemalloc)
      E2E Latency  — wall-clock time until final output is ready

    Usage:
        profiler = HITLProfiler(audio_duration_s=3.5)
        with profiler.profile():
            for result in streamer.stream_file(audio):
                profiler.record_chunk_output()
        print(f"RTF={profiler.rtf:.3f}  RAM={profiler.peak_ram_mb:.1f} MB")
    """

    def __init__(self, audio_duration_s: float):
        self.audio_duration_s   = max(audio_duration_s, 1e-9)
        self._wall_time:  float = 0.0
        self._peak_ram_mb: float = 0.0
        self._t_start:    float = 0.0
        self._t_end:      float = 0.0
        self._first_output_t: float = 0.0

    @contextlib.contextmanager
    def profile(self) -> Generator[None, None, None]:
        """Context manager: wrap the inference call here."""
        is_cuda = torch.cuda.is_available()
        if is_cuda:
            torch.cuda.reset_peak_memory_stats()
        else:
            tracemalloc.start()
            
        self._t_start = time.perf_counter()
        self._first_output_t = 0.0
        try:
            yield
        finally:
            self._t_end     = time.perf_counter()
            self._wall_time = self._t_end - self._t_start
            
            if is_cuda:
                peak_bytes = torch.cuda.max_memory_allocated()
            else:
                _, peak_bytes = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                
            self._peak_ram_mb = peak_bytes / (1024 ** 2)

    def record_chunk_output(self) -> None:
        """Call once when the first non-empty output chunk is yielded."""
        if self._first_output_t == 0.0:
            self._first_output_t = time.perf_counter()

    @property
    def rtf(self) -> float:
        return self._wall_time / self.audio_duration_s

    @property
    def peak_ram_mb(self) -> float:
        return self._peak_ram_mb

    @property
    def e2e_latency_s(self) -> float:
        """Wall-clock time from start to final output."""
        return self._wall_time

@dataclass
class GoldenSample:
    """One audio sample in the Golden Test Set."""
    sample_id:   str
    mushaf:      str
    audio:       np.ndarray
    sampling_rate: int   = 16_000
    duration_s:  float   = 0.0

    # Populated by compute_batch_references()
    ref_phonemes: str   = ""
    ref_sifat:    list  = field(default_factory=list)

    def __post_init__(self):
        if self.duration_s == 0.0:
            self.duration_s = len(self.audio) / self.sampling_rate

def _parse_sifat_list(sifat_list: list) -> list:
    """Parse JSON dictionaries from the manifest back into Sifa objects."""
    if not sifat_list:
        return []
    try:
        from imports.muaalem_typing import Sifa, SingleUnit
    except ImportError:
        return sifat_list
        
    parsed = []
    for sg_dict in sifat_list:
        if isinstance(sg_dict, Sifa):
            parsed.append(sg_dict)
            continue
        if not isinstance(sg_dict, dict):
            continue
            
        # The HF dataset stores the group in the 'phonemes' key.
        pg = sg_dict.get("phonemes_group") or sg_dict.get("phonemes", "")
        if isinstance(pg, str):
            pg = pg.replace(" ", "")
        kwargs = {"phonemes_group": pg}
        for k, v in sg_dict.items():
            if k in ("phonemes_group", "phonemes"): continue
            if isinstance(v, dict) and "text" in v:
                kwargs[k] = SingleUnit(text=v["text"], prob=v.get("prob", 1.0), idx=v.get("idx", 0))
            elif isinstance(v, str):
                kwargs[k] = SingleUnit(text=v, prob=1.0, idx=0)
            else:
                kwargs[k] = v
                
        # Sifa expects all the fields, so fill in missing ones with None
        for k in Sifa.__dataclass_fields__:
            if k not in kwargs:
                kwargs[k] = None
                
        parsed.append(Sifa(**kwargs))
    return parsed

def levenshtein_with_ops(s: str, t: str):
    m, n = len(s), len(t)

    # --- 1) Build DP table ---
    dp = [[0]*(n+1) for _ in range(m+1)]

    for i in range(m+1):
        dp[i][0] = i
    for j in range(n+1):
        dp[0][j] = j

    for i in range(1, m+1):
        for j in range(1, n+1):
            cost = 0 if s[i-1] == t[j-1] else 1

            dp[i][j] = min(
                dp[i-1][j] + 1,        # deletion
                dp[i][j-1] + 1,        # insertion
                dp[i-1][j-1] + cost    # substitution / match
            )

    # --- 2) Backtrack to compute S, D, I ---
    i, j = m, n
    S = D = I = 0

    while i > 0 or j > 0:
        # Case 1: match or substitution
        if i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + (s[i-1] != t[j-1]):
            if s[i-1] != t[j-1]:
                S += 1
            i -= 1
            j -= 1
        # Case 2: deletion
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            D += 1
            i -= 1
        # Case 3: insertion
        else:
            I += 1
            j -= 1

    distance = dp[m][n]
    return distance, S, D, I

def compute_per(ref: str, hyp: str) -> dict:
    dist, S, D, I = levenshtein_with_ops(ref, hyp)
    N = max(len(ref), 1)
    return {"per": dist / N, "S": S, "D": D, "I": I, "N": N}

def _align_sifat_groups(ref_sifat: list, hyp_sifat: list) -> list[tuple[int, int]]:
    ref_g = [s.phonemes_group for s in ref_sifat]
    hyp_g = [s.phonemes_group for s in hyp_sifat]
    n, m  = len(ref_g), len(hyp_g)
    dp    = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1): dp[i][0] = i
    for j in range(1, m + 1): dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = (
                dp[i-1][j-1] if ref_g[i-1] == hyp_g[j-1]
                else 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
            )
    aligned: list[tuple[int, int]] = []
    i, j = n, m
    while i > 0 and j > 0:
        if ref_g[i-1] == hyp_g[j-1] and dp[i][j] == dp[i-1][j-1]:
            aligned.append((i-1, j-1))
            i -= 1; j -= 1
        elif dp[i-1][j] <= dp[i][j-1] and dp[i-1][j] <= dp[i-1][j-1]:
            i -= 1
        elif dp[i][j-1] <= dp[i-1][j-1]:
            j -= 1
        else:
            i -= 1; j -= 1
    aligned.reverse()
    return aligned

def compute_sifat_accuracy(ref_sifat: list, hyp_sifat: list) -> dict:
    if not ref_sifat or not hyp_sifat:
        return {"macro_accuracy": 0.0, "per_level": {}, "n_aligned": 0}
    pairs = _align_sifat_groups(ref_sifat, hyp_sifat)
    if not pairs:
        return {"macro_accuracy": 0.0, "per_level": {}, "n_aligned": 0}
    per_level: dict[str, dict] = {}
    accs: list[float] = []
    for level in SIFAT_LEVELS:
        matches = total = 0
        for ri, hi in pairs:
            r = getattr(ref_sifat[ri], level, None)
            h = getattr(hyp_sifat[hi], level, None)
            if r is None and h is None:
                continue
            total += 1
            if r is not None and h is not None and r.text == h.text:
                matches += 1
        acc = matches / max(total, 1)
        per_level[level] = {"accuracy": acc, "matches": matches, "total": total}
        if total > 0:
            accs.append(acc)
    macro = statistics.mean(accs) if accs else 0.0
    return {"macro_accuracy": macro, "per_level": per_level, "n_aligned": len(pairs)}

def build_targeted_golden_dataset(
    dataset_name:    str   = "obadx/muaalem-annotated-v3",
    cache_dir:       str   = "targeted_golden_dataset",
    force_rebuild:   bool  = False,
    audio_column:    str   = "audio",
    min_duration_s:  float = 0.0,
    max_duration_s:  float = 300.0,
) -> list["GoldenSample"]:
    """
    Surgically builds a dataset by hitting the HF Datasets Server API.
    This "crawls the links" (viewer API) and uses Binary Search to find the exact 
    start offset for each Surah to eliminate all unnecessary bandwidth.
    
    Target Ayahs:
      - Surah 10 (Yunus):    Ayahs 77 to 88
      - Surah 46 (Al-Ahqaf): Ayahs 19 to 28
      - Surah 50 (Qaf):      Ayahs 1 to 18
    """
    
    cache_path  = Path(cache_dir)
    manifest_f  = cache_path / "manifest.json"
    audio_dir   = cache_path / "audio"
    batch_ref_f = cache_path / "batch_refs.pkl"

    # ── 1. Load from cache ───────────────────────────────────────────────────
    if manifest_f.exists() and not force_rebuild:
        logger.info("Loading Targeted Golden Test Set from cache: %s", cache_dir)
        with open(manifest_f, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        samples: list[GoldenSample] = []
        for entry in manifest:
            npy_path = audio_dir / f"{entry['sample_id']}.npy"
            if not npy_path.exists():
                logger.warning("Missing audio file %s — skipping", npy_path)
                continue
            audio = np.load(str(npy_path))
            samples.append(GoldenSample(
                sample_id  = entry["sample_id"],
                mushaf     = entry["mushaf"],
                audio      = audio,
                duration_s = entry["duration_s"],
                ref_phonemes = entry.get("phonemes", "").replace(" ", ""),
                ref_sifat    = _parse_sifat_list(entry.get("sifat", [])),
            ))

        if batch_ref_f.exists():
            import pickle
            with open(batch_ref_f, "rb") as f:
                refs: dict[str, dict] = pickle.load(f)
            for s in samples:
                if s.sample_id in refs:
                    s.ref_phonemes = refs[s.sample_id]["phonemes"]

        logger.info(
            "Loaded %d targeted samples from %d mushaf(s)",
            len(samples),
            len({s.mushaf for s in samples}),
        )
        return samples

    # ── 2. Build via HF Datasets Server API (Viewer Link Crawling) ───────────
    logger.info("Extracting targeted Ayahs via HF Datasets Server from '%s'", dataset_name)
    
    cache_path.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    moshafs = [c for c in get_dataset_config_names(dataset_name, trust_remote_code=True) if c.startswith("moshaf_") and "metadata" not in c]
    if not moshafs:
        raise RuntimeError(f"No moshaf configs found for {dataset_name}")

    manifest: list[dict] = []
    samples_out: list[GoldenSample] = []
    
    hf_token = os.environ.get("HF_TOKEN", "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")
    api_headers = {"Authorization": f"Bearer {hf_token}"}
    base_url = "https://datasets-server.huggingface.co"

    def fetch_api(endpoint, params):
        url = f"{base_url}{endpoint}"
        for attempt in range(5):
            resp = requests.get(url, params=params, headers=api_headers, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                time.sleep((2 ** attempt) + random.uniform(0.5, 1.5))
            else:
                resp.raise_for_status()
        raise Exception(f"Failed to fetch {url} after 5 retries")

    def get_sura_aya(mushaf, offset):
        try:
            data = fetch_api("/rows", {'dataset': dataset_name, 'config': mushaf, 'split': "train", 'offset': offset, 'length': 1})
            span = data['rows'][0]['row'].get('start_span')
            if not span: return int(data['rows'][0]['row'].get('sura_or_aya_index')), 0
            return span.get('sura_idx'), span.get('aya_idx')
        except Exception:
            return None, None

    def find_start_offset(mushaf, target_sura, target_aya, max_rows):
        low = 0
        high = max_rows - 1
        best = -1
        while low <= high:
            mid = (low + high) // 2
            sura, aya = get_sura_aya(mushaf, mid)
            if sura is None: return None
            if sura < target_sura or (sura == target_sura and aya < target_aya):
                low = mid + 1
            elif sura > target_sura or (sura == target_sura and aya > target_aya):
                high = mid - 1
            else:
                best = mid
                high = mid - 1 # continue searching left to find the VERY FIRST occurrence
        return best if best != -1 else low

    TARGET_RANGES = [
        (10, 77, 88),
        (46, 19, 28),
        (50, 1, 18),
    ]

    for mushaf in moshafs:
        logger.info("Crawling viewer API for %s...", mushaf)
        
        try:
            size_data = fetch_api("/size", {'dataset': dataset_name, 'config': mushaf})
            max_rows = size_data['size']['config']['num_rows']
            
            scanned_rows = 0
            extracted_rows = 0

            for (t_sura, t_start_aya, t_end_aya) in TARGET_RANGES:
                start_offset = find_start_offset(mushaf, t_sura, t_start_aya, max_rows)
                if start_offset is None:
                    continue
                
                current_offset = start_offset
                done_with_range = False
                
                while not done_with_range and current_offset < max_rows:
                    data = fetch_api("/rows", {
                        'dataset': dataset_name, 'config': mushaf, 'split': "train", 
                        'offset': current_offset, 'length': 100
                    })
                    rows = data.get('rows', [])
                    if not rows: break
                    
                    for row_item in rows:
                        row = row_item['row']
                        scanned_rows += 1
                        
                        span = row.get('start_span')
                        if not span or not isinstance(span, dict): continue
                        
                        sura = span.get('sura_idx')
                        aya = span.get('aya_idx')
                        if sura is None or aya is None: continue
                        sura, aya = int(sura), int(aya)
                        
                        if sura > t_sura or (sura == t_sura and aya > t_end_aya):
                            done_with_range = True
                            break
                            
                        if sura == t_sura and t_start_aya <= aya <= t_end_aya:
                            seg = row.get("segment_index")
                            if seg is None or pd.isna(seg): seg = f"a{aya}"
                            sample_id = f"{mushaf}_s{sura}_a{aya}_seg{seg}"
                            
                            audio_list = row.get(audio_column)
                            if not audio_list or not isinstance(audio_list, list) or not audio_list[0].get('src'):
                                continue
                                
                            audio_url = audio_list[0]['src']
                            try:
                                for attempt in range(3):
                                    audio_resp = requests.get(audio_url, timeout=15)
                                    if audio_resp.status_code == 200:
                                        break
                                    time.sleep(1)
                                
                                data_arr, sr = sf.read(io.BytesIO(audio_resp.content))
                                arr = data_arr.astype(np.float32)

                                if sr != 16_000:
                                    arr = librosa.resample(arr, orig_sr=sr, target_sr=16_000)
                                    sr = 16_000

                                duration_s = len(arr) / sr
                                if not (min_duration_s <= duration_s <= max_duration_s):
                                    continue
                                    
                                npy_path = audio_dir / f"{sample_id}.npy"
                                np.save(str(npy_path), arr)

                                manifest.append({
                                    "sample_id":  sample_id,
                                    "mushaf":     mushaf,
                                    "duration_s": duration_s,
                                    "sura_idx":   sura,
                                    "aya_idx":    aya,
                                    "audio_file": f"audio/{sample_id}.npy",
                                })

                                samples_out.append(GoldenSample(
                                    sample_id  = sample_id,
                                    mushaf     = mushaf,
                                    audio      = arr,
                                    duration_s = duration_s,
                                ))
                                extracted_rows += 1
                            except Exception as e:
                                logger.warning("Failed downloading %s: %s", sample_id, e)
                    
                    current_offset += 100

            logger.info("  → API Search Complete | Extracted %d target samples in %s", extracted_rows, mushaf)

        except Exception as e:
            logger.exception("Failed processing API stream for %s", mushaf)

    # ── 3. Persist manifest ──────────────────────────────────────────────────
    with open(manifest_f, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    logger.info(
        "Targeted set compiled: %d samples from %d mushaf(s) → %s",
        len(samples_out),
        len({s.mushaf for s in samples_out}),
        cache_dir,
    )
    return samples_out


def check_missing_segments(
    dataset_name: str = "obadx/muaalem-annotated-v3",
    cache_dir: str = "targeted_golden_dataset",
) -> list[dict]:
    """
    Parses the manifest.json file to verify that all target ayahs 
    were successfully extracted for each available mushaf.
    Returns a list of dictionaries detailing which ayahs are missing.
    """

    TARGET_RANGES = [
        (10, 77, 88),
        (46, 19, 28),
        (50, 1, 18),
        (19, 1, 1)
    ]

    manifest_f = Path(cache_dir) / "manifest.json"
    if not manifest_f.exists():
        logger.error("Manifest not found at %s", manifest_f)
        return []

    with open(manifest_f, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Gather all available configs for the dataset
    try:
        moshafs = [c for c in get_dataset_config_names(dataset_name, trust_remote_code=True) if c.startswith("moshaf_") and "metadata" not in c]
    except Exception as e:
        logger.error("Failed to fetch configs: %s", e)
        return []

    # Build a lookup set for O(1) checks
    present_ayahs = set()
    for entry in manifest:
        present_ayahs.add((entry["mushaf"], entry["sura_idx"], entry["aya_idx"]))

    missing_list = []

    for mushaf in moshafs:
        for (t_sura, t_start_aya, t_end_aya) in TARGET_RANGES:
            for aya in range(t_start_aya, t_end_aya + 1):
                if (mushaf, t_sura, aya) not in present_ayahs:
                    missing_list.append({
                        "mushaf": mushaf,
                        "sura_idx": t_sura,
                        "aya_idx": aya
                    })

    if missing_list:
        logger.warning("Found %d missing ayahs across %d mushaf(s).", len(missing_list), len(set(m['mushaf'] for m in missing_list)))
    else:
        logger.info("All target ayahs perfectly extracted across all %d mushaf(s)!", len(moshafs))
        
    return missing_list

def fetch_missing_segments(
    missing_list: list[dict],
    dataset_name: str = "obadx/muaalem-annotated-v3",
    cache_dir: str = "targeted_golden_dataset",
    audio_column: str = "audio",
):
    """
    Takes the output of check_missing_segments, searches specifically for the missing Ayahs 
    using the HF Datasets Server API, downloads them, and appends them to the manifest.
    """
    import soundfile as sf
    import librosa
    
    if not missing_list:
        logger.info("No missing segments to fetch.")
        return
        
    cache_path  = Path(cache_dir)
    manifest_f  = cache_path / "manifest.json"
    audio_dir   = cache_path / "audio"
    
    if not manifest_f.exists():
        logger.error("Manifest not found at %s", manifest_f)
        return
        
    with open(manifest_f, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    hf_token = os.environ.get("HF_TOKEN", "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")
    api_headers = {"Authorization": f"Bearer {hf_token}"}
    base_url = "https://datasets-server.huggingface.co"

    def fetch_api(endpoint, params):
        url = f"{base_url}{endpoint}"
        for attempt in range(5):
            resp = requests.get(url, params=params, headers=api_headers, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                time.sleep((2 ** attempt) + random.uniform(0.5, 1.5))
            else:
                resp.raise_for_status()
        raise Exception(f"Failed to fetch {url} after 5 retries")

    def get_sura_aya(mushaf, offset):
        try:
            data = fetch_api("/rows", {'dataset': dataset_name, 'config': mushaf, 'split': "train", 'offset': offset, 'length': 1})
            span = data['rows'][0]['row'].get('start_span')
            sura_aya_index = data['rows'][0]['row'].get('sura_or_aya_index')
            if not span: return int(sura_aya_index), 0
            return span.get('sura_idx'), span.get('aya_idx')
        except Exception:
            return None, None

    def find_start_offset(mushaf, target_sura, target_aya, max_rows):
        low = 0
        high = max_rows - 1
        best = -1
        while low <= high:
            mid = (low + high) // 2
            sura, aya = get_sura_aya(mushaf, mid)
            print(f"Mid: {mid}, Sura: {sura}, Aya: {aya}")
            if sura is None: return None
            if sura < target_sura or (sura == target_sura and aya < target_aya):
                low = mid + 1
            elif sura > target_sura or (sura == target_sura and aya > target_aya):
                high = mid - 1
            else:
                best = mid
                high = mid - 1
        print(low, high, best)
        return best if best != -1 else min(low, high)

    # Group missing by mushaf
    by_mushaf = {}
    for m in missing_list:
        by_mushaf.setdefault(m["mushaf"], []).append(m)
        
    extracted_rows = 0
    
    for mushaf, missing_items in by_mushaf.items():
        try:
            size_data = fetch_api("/size", {'dataset': dataset_name, 'config': mushaf})
            max_rows = size_data['size']['config']['num_rows']
            print(f"Max rows for {mushaf}: {max_rows}")
            
            # Group by Surah to avoid duplicate binary searches
            by_sura = {}
            for m in missing_items:
                by_sura.setdefault(m["sura_idx"], set()).add(m["aya_idx"])
                
            for t_sura, ayahs in by_sura.items():
                min_aya = min(ayahs)
                max_aya = max(ayahs)

                print(f"Processing mushaf: {mushaf}, sura: {t_sura}, ayahs: {ayahs}, min_aya: {min_aya}, max_aya: {max_aya}")
                
                start_offset = find_start_offset(mushaf, t_sura, min_aya, max_rows)
                print(f"Start offset: {start_offset}")
                if start_offset is None:
                    continue
                    
                current_offset = start_offset
                done_with_range = False
                
                while not done_with_range and current_offset < max_rows:
                    data = fetch_api("/rows", {
                        'dataset': dataset_name, 'config': mushaf, 'split': "train", 
                        'offset': current_offset, 'length': 100
                    })
                    rows = data.get('rows', [])
                    if not rows: break
                    
                    for row_item in rows:
                        row = row_item['row']
                        
                        span = row.get('start_span')
                        if not span or not isinstance(span, dict): continue
                        
                        sura = span.get('sura_idx')
                        aya = span.get('aya_idx')
                        if sura is None or aya is None: continue
                        sura, aya = int(sura), int(aya)
                        
                        if sura > t_sura or (sura == t_sura and aya > max_aya):
                            done_with_range = True
                            break
                            
                        if sura == t_sura and aya in ayahs:
                            seg = row.get("segment_index")
                            if seg is None or pd.isna(seg): seg = f"a{aya}"
                            sample_id = f"{mushaf}_s{sura}_a{aya}_seg{seg}"
                            
                            # Check if it was somehow already fetched
                            if any(m_entry['sample_id'] == sample_id for m_entry in manifest):
                                print(f"Already fetched sample: {sample_id}")
                                continue
                            
                            audio_list = row.get(audio_column)
                            if not audio_list or not isinstance(audio_list, list) or not audio_list[0].get('src'):
                                continue
                                
                            audio_url = audio_list[0]['src']
                            try:
                                for attempt in range(3):
                                    audio_resp = requests.get(audio_url, timeout=15)
                                    if audio_resp.status_code == 200:
                                        break
                                    time.sleep(1)
                                
                                data_arr, sr = sf.read(io.BytesIO(audio_resp.content))
                                arr = data_arr.astype(np.float32)

                                if sr != 16_000:
                                    arr = librosa.resample(arr, orig_sr=sr, target_sr=16_000)
                                    sr = 16_000

                                duration_s = len(arr) / sr
                                
                                npy_path = audio_dir / f"{sample_id}.npy"
                                np.save(str(npy_path), arr)

                                manifest.append({
                                    "sample_id":  sample_id,
                                    "mushaf":     mushaf,
                                    "duration_s": duration_s,
                                    "sura_idx":   sura,
                                    "aya_idx":    aya,
                                    "audio_file": f"audio/{sample_id}.npy",
                                })
                                extracted_rows += 1
                                logger.info("Successfully fetched missing segment: %s", sample_id)
                            except Exception as e:
                                logger.warning("Failed downloading %s: %s", sample_id, e)
                    current_offset += 100
        except Exception as e:
            logger.exception("Failed processing missing segments for %s", mushaf)
            
    if extracted_rows > 0:
        with open(manifest_f, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        logger.info("Appended %d missing samples to the manifest.", extracted_rows)
    else:
        logger.info("No missing samples could be resolved or fetched.")

def fetch_metadata_columns(
    dataset_name: str = "obadx/muaalem-annotated-v3",
    cache_dir: str = "targeted_golden_dataset",
):
    """
    Updates the manifest.json by adding the missing 'sifat' and 'phonemes' 
    columns from the HF Datasets Server API.
    """
    
    cache_path  = Path(cache_dir)
    manifest_f  = cache_path / "manifest.json"
    
    if not manifest_f.exists():
        logger.error("Manifest not found at %s", manifest_f)
        return
        
    with open(manifest_f, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    # Filter entries that are missing 'sifat' or 'phonemes'
    needs_update = [m for m in manifest if "sifat" not in m or "phonemes" not in m]
    if not needs_update:
        logger.info("All manifest entries already have 'sifat' and 'phonemes' columns.")
        return
        
    logger.info("Fetching metadata columns for %d manifest entries.", len(needs_update))
    
    hf_token = os.environ.get("HF_TOKEN", "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")
    api_headers = {"Authorization": f"Bearer {hf_token}"}
    base_url = "https://datasets-server.huggingface.co"

    def fetch_api(endpoint, params):
        url = f"{base_url}{endpoint}"
        for attempt in range(5):
            resp = requests.get(url, params=params, headers=api_headers, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                time.sleep((2 ** attempt) + random.uniform(0.5, 1.5))
            else:
                resp.raise_for_status()
        raise Exception(f"Failed to fetch {url} after 5 retries")

    def get_sura_aya(mushaf, offset):
        try:
            data = fetch_api("/rows", {'dataset': dataset_name, 'config': mushaf, 'split': "train", 'offset': offset, 'length': 1})
            span = data['rows'][0]['row'].get('start_span')
            sura_aya_index = data['rows'][0]['row'].get('sura_or_aya_index')
            if not span: return int(sura_aya_index), 0
            return span.get('sura_idx'), span.get('aya_idx')
        except Exception:
            return None, None

    def find_start_offset(mushaf, target_sura, target_aya, max_rows):
        low = 0
        high = max_rows - 1
        best = -1
        while low <= high:
            mid = (low + high) // 2
            sura, aya = get_sura_aya(mushaf, mid)
            if sura is None: return None
            if sura < target_sura or (sura == target_sura and aya < target_aya):
                low = mid + 1
            elif sura > target_sura or (sura == target_sura and aya > target_aya):
                high = mid - 1
            else:
                best = mid
                high = mid - 1
        return best if best != -1 else min(low, high)

    # Group missing by mushaf
    by_mushaf = {}
    for m in needs_update:
        by_mushaf.setdefault(m["mushaf"], []).append(m)
        
    updated_rows = 0
    
    for mushaf, missing_items in by_mushaf.items():
        try:
            size_data = fetch_api("/size", {'dataset': dataset_name, 'config': mushaf})
            max_rows = size_data['size']['config']['num_rows']
            
            # Group by Surah to avoid duplicate binary searches
            by_sura = {}
            for m in missing_items:
                by_sura.setdefault(m.get("sura_idx", 0), set()).add(m.get("aya_idx", 0))
                
            for t_sura, ayahs in by_sura.items():
                if t_sura == 0: continue
                min_aya = min(ayahs)
                max_aya = max(ayahs)

                start_offset = find_start_offset(mushaf, t_sura, min_aya, max_rows)
                if start_offset is None:
                    continue
                    
                current_offset = start_offset
                done_with_range = False
                
                while not done_with_range and current_offset < max_rows:
                    data = fetch_api("/rows", {
                        'dataset': dataset_name, 'config': mushaf, 'split': "train", 
                        'offset': current_offset, 'length': 100
                    })
                    rows = data.get('rows', [])
                    if not rows: break
                    
                    for row_item in rows:
                        row = row_item['row']
                        
                        span = row.get('start_span')
                        if not span or not isinstance(span, dict): continue
                        
                        sura = span.get('sura_idx')
                        aya = span.get('aya_idx')
                        if sura is None or aya is None: continue
                        sura, aya = int(sura), int(aya)
                        
                        if sura > t_sura or (sura == t_sura and aya > max_aya):
                            done_with_range = True
                            break
                            
                        if sura == t_sura and aya in ayahs:
                            seg = row.get("segment_index")
                            if seg is None or pd.isna(seg): seg = f"a{aya}"
                            sample_id = f"{mushaf}_s{sura}_a{aya}_seg{seg}"
                            
                            # Update corresponding manifest entries
                            for m_entry in manifest:
                                if m_entry['sample_id'] == sample_id:
                                    if 'sifat' in row:
                                        m_entry['sifat'] = row['sifat']
                                    if 'phonemes' in row:
                                        m_entry['phonemes'] = row['phonemes']
                                    updated_rows += 1
                                    logger.info("Successfully fetched metadata for: %s", sample_id)
                                    
                    current_offset += 100
        except Exception as e:
            logger.exception("Failed processing metadata columns for %s", mushaf)
            
    if updated_rows > 0:
        with open(manifest_f, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        logger.info("Updated %d samples with missing 'sifat' and 'phonemes' in the manifest.", updated_rows)
    else:
        logger.info("No metadata columns could be resolved or fetched.")


# ── Per-sample runner ────────────────────────────────────────────────────────

def run_one(spec: ConfigSpec, sample, nutq,
            sweep_cfg: SweepRunConfig) -> tuple:
    """
    Run one (config, sample) pair using HITLProfiler for measurement.
    Returns (SampleRecord, events_list).
    events_list is a list of dicts (StreamingEvent.as_dict()) — written to
    events.jsonl by the caller. On error, events_list is empty.

    Exceptions never escape — they are caught and recorded with
    is_fatal=True and error populated on the SampleRecord.
    """
    
    from streaming_inference import AdaptiveStreamingMuaalem
    from streaming_event_tracer import run_with_event_capture

    base = dict(
        config_name=spec.name,
        sample_id=sample.sample_id,
        mushaf=sample.mushaf,
        audio_duration_s=sample.duration_s,
        worst_case_latency_s=spec.worst_case_latency_s,
        apl_s=float("nan"), acl_s=float("nan"),
        wall_time_s=0.0, rtf=float("nan"),
        e2e_latency_s=0.0, peak_ram_mb=0.0,
        per=float("inf"), S=0, D=0, I=0, N=0,
        sifat_accuracy=0.0,
        is_fatal=True, error=None,
        ref_phonemes=sample.ref_phonemes or "",
        hyp_phonemes="",
        n_expansions=0,
        n_h1_only_expansions=0,
        n_h2_only_expansions=0,
        n_h1h2_both_expansions=0,
        n_force_committed=0,
    )

    if not sample.ref_phonemes:
        base["error"] = "no_reference_phonemes"
        return SampleRecord(**base), []

    try:
        streamer = AdaptiveStreamingMuaalem(nutq, cfg=spec.config)
    except Exception as e:
        base["error"] = f"streamer_init: {e}\n{traceback.format_exc()}"
        return SampleRecord(**base), []

    try:
        profiler = HITLProfiler(sample.duration_s)
        all_events: List = []   # full StreamingEvent objects, in-memory

        def on_event(ev):
            # Record to profiler when tokens are emitted
            if ev.type in ("COMMIT", "FLUSH") and ev.n_tok > 0:
                profiler.record_chunk_output()
            all_events.append(ev)

        with profiler.profile():
            _events, hyp, all_sifat = run_with_event_capture(
                streamer, sample.audio, on_event=on_event,
            )

        per_info  = compute_per(sample.ref_phonemes, hyp)
        sifat_acc = 0.0
        if sample.ref_sifat:
            try:
                sifat_info = compute_sifat_accuracy(sample.ref_sifat, all_sifat)
                sifat_acc  = sifat_info.get("macro_accuracy", 0.0)
            except Exception:
                sifat_acc = 0.0

        # Compute exact phoneme latency (APL) and chunk latency (ACL)
        t_prev = 0.0
        lookahead = spec.config.right_lookahead_s
        sum_Li = 0.0
        sum_word_latency = 0.0
        n_chunks = 0
        for ev in all_events:
            if ev.type in ("COMMIT", "FLUSH"):
                Li = ev.t - t_prev
                if Li > 0:
                    sum_Li += Li
                    sum_word_latency += Li * (Li / 2.0 + lookahead)
                    n_chunks += 1
                t_prev = ev.t
        
        apl_s = (sum_word_latency / sum_Li) if sum_Li > 0 else float("nan")
        acl_s = (sum_Li / n_chunks + lookahead) if n_chunks > 0 else float("nan")

        # Count H1/H2/force from the full in-memory event list. These counts
        # land in results.csv, which is why we can afford to DROP all EXPAND
        # events from events.jsonl — the aggregate signal is preserved.
        n_h1 = sum(1 for e in all_events if e.type == "EXPAND" and e.reason == "h1")
        n_h2 = sum(1 for e in all_events if e.type == "EXPAND" and e.reason == "h2")
        n_h1h2 = sum(1 for e in all_events if e.type == "EXPAND" and e.reason == "h1+h2")
        n_force = sum(
            1 for e in all_events
            if e.type == "COMMIT"
            and e.reason == "max-expansions"
        )
        n_total_expansions = sum(1 for e in all_events if e.type == "EXPAND")

        # Persist events to JSONL. EXPAND events keep only type/reason/t;
        # COMMIT/FLUSH keep type/reason/text/t/seam_text.
        persisted_events: List[dict] = []
        for ev in all_events:
            d = ev.to_persisted_dict()
            if d is not None:
                persisted_events.append(d)

        base.update(
            wall_time_s=profiler._wall_time,
            rtf=profiler.rtf,
            e2e_latency_s=profiler.e2e_latency_s,
            peak_ram_mb=profiler.peak_ram_mb,
            apl_s=apl_s,
            acl_s=acl_s,
            per=per_info["per"], S=per_info["S"],
            D=per_info["D"], I=per_info["I"], N=per_info["N"],
            sifat_accuracy=sifat_acc,
            is_fatal=per_info["per"] > sweep_cfg.fatal_per_threshold,
            error=None,
            hyp_phonemes=hyp,
            n_expansions=n_total_expansions,
            n_h1_only_expansions=n_h1,
            n_h2_only_expansions=n_h2,
            n_h1h2_both_expansions=n_h1h2,
            n_force_committed=n_force,
        )
        return SampleRecord(**base), persisted_events

    except Exception as e:
        base["error"] = f"stream_file: {e}\n{traceback.format_exc()}"
        return SampleRecord(**base), []


# ── Persistence ──────────────────────────────────────────────────────────────

def _csv_append(record: SampleRecord, path: Path) -> None:
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SampleRecord.fields())
        if new:
            w.writeheader()
        w.writerow({k: getattr(record, k) for k in SampleRecord.fields()})


def _jsonl_append(obj: dict, path: Path) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _config_snapshot(spec: ConfigSpec) -> dict:
    """Serializable snapshot of the AdaptiveConfig for fatal-case reproduction."""
    return {k: v for k, v in asdict(spec.config).items()}


# ── Rich aggregation (matches sweep_streaming_advanced.aggregate_global) ──

def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple:
    """
    Wilson score 95% CI for a binomial proportion k/n.

    Well-defined at boundary cases (k=0 or k=n) where the normal-approx
    interval collapses to zero width. Preferred over the normal approx
    for the small-sample sizes typical of this sweep.
    """
    if n <= 0:
        return (0.0, 1.0)
    p_hat = k / n
    denom  = 1.0 + (z * z) / n
    center = (p_hat + (z * z) / (2.0 * n)) / denom
    half   = z * ((p_hat * (1.0 - p_hat) / n
                   + (z * z) / (4.0 * n * n)) ** 0.5) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _build_aggregate(records: List[SampleRecord],
                     sweep_cfg: SweepRunConfig):
    """
    Aggregate per-sample records into one row per config_name. Returns a
    pandas DataFrame with the same column schema as
    sweep_streaming_advanced.aggregate_global, PLUS:

      Trigger breakdown (mean over non-fatal samples):
        h1_only_mean, h2_only_mean, h1h2_both_mean, force_commit_mean

      Sound probabilistic representation of failure:
        n_total          — fatal + non_fatal (errors not included)
        n_fatal          — count where per > fatal_per_threshold
        fatal_rate       — n_fatal / n_total
        fatal_rate_ci95_lo, fatal_rate_ci95_hi
                         — Wilson 95% CI on the failure proportion
        per_mean_inclusive — mean PER over ALL non-error samples
                              (including fatals). The "honest" expected
                              PER on a randomly drawn sample.
        per_se           — standard error of per_mean
        per_mean_ci95_lo, per_mean_ci95_hi
                         — normal-approx 95% CI for mean PER
    """

    # Build per-sample frames.
    # `non_fatal_rows` — feeds the conditional accuracy stats (per_mean,
    #   per_std, per_p99, sifat_*) — the "how good is the output when it
    #   doesn't fail" view.
    # `all_rows` — feeds the fatal-rate stats and per_mean_inclusive,
    #   the "honest expected performance" view (errors excluded since
    #   they're a separate failure mode tracked in fatal_cases.jsonl).
    non_fatal_rows = []
    all_rows = []
    for rec in records:
        if rec.error:
            continue
        total_exp = (rec.n_h1_only_expansions
                     + rec.n_h2_only_expansions
                     + rec.n_h1h2_both_expansions)
        base_row = {
            "config_name":           rec.config_name,
            "per":                   rec.per,
            "sifat_accuracy":        rec.sifat_accuracy,
            "rtf":                   rec.rtf,
            "peak_ram_mb":           rec.peak_ram_mb,
            "e2e_latency_s":         rec.e2e_latency_s,
            "worst_case_latency_s":  rec.worst_case_latency_s,
            "apl_s":                 rec.apl_s,
            "acl_s":                 rec.acl_s,
            "audio_duration_s":      rec.audio_duration_s,
            "n_h1_only_expansions":  rec.n_h1_only_expansions,
            "n_h2_only_expansions":  rec.n_h2_only_expansions,
            "n_h1h2_both_expansions":rec.n_h1h2_both_expansions,
            "n_total_expansions":    total_exp,
            "n_force_committed":     rec.n_force_committed,
            "is_fatal":              bool(rec.is_fatal),
        }
        all_rows.append(base_row)
        if not rec.is_fatal:
            non_fatal_rows.append(base_row)
    if not all_rows:
        return pd.DataFrame()

    df_nf  = pd.DataFrame(non_fatal_rows)
    df_all = pd.DataFrame(all_rows)

    def _p99(s):    return float(s.quantile(0.99))
    def _p95(s):    return float(s.quantile(0.95))
    def _median(s): return float(s.median())

    # ── Conditional (non-fatal) accuracy + behaviour stats ──────────────
    if not df_nf.empty:
        agg = (
            df_nf.groupby("config_name")
            .agg(
                theoretical_latency_s = ("worst_case_latency_s", "first"),
                per_mean              = ("per",            "mean"),
                per_std               = ("per",            "std"),
                per_p99               = ("per",            _p99),
                sifat_mean            = ("sifat_accuracy", "mean"),
                sifat_std             = ("sifat_accuracy", "std"),
                rtf_median            = ("rtf",            _median),
                rtf_p95               = ("rtf",            _p95),
                ram_mean_mb           = ("peak_ram_mb",    "mean"),
                ram_max_mb            = ("peak_ram_mb",    "max"),
                e2e_latency_mean_s    = ("e2e_latency_s",  "mean"),
                apl_s_mean            = ("apl_s",          "mean"),
                acl_s_mean            = ("acl_s",          "mean"),
                n_samples             = ("per",            "count"),
                # Behaviour
                # ─────────
                _sum_force            = ("n_force_committed",      "sum"),
                _sum_expansions       = ("n_total_expansions",     "sum"),
                _sum_h1               = ("n_h1_only_expansions",   "sum"),
                _sum_h2               = ("n_h2_only_expansions",   "sum"),
                _sum_h1h2             = ("n_h1h2_both_expansions", "sum"),
                _sum_duration         = ("audio_duration_s",       "sum"),
            )
            .reset_index()
            .fillna(0.0)
        )
        
        # Calculate rates
        dur = agg["_sum_duration"].replace(0.0, float("nan"))
        agg["expansion_rate"] = (agg["_sum_expansions"] / dur).fillna(0.0)
        agg["force_commit_rate_10s"] = (agg["_sum_force"] / dur * 10).fillna(0.0)
        agg["h1_rate_10s"] = (agg["_sum_h1"] / dur * 10).fillna(0.0)
        agg["h2_rate_10s"] = (agg["_sum_h2"] / dur * 10).fillna(0.0)
        agg["h1h2_rate_10s"] = (agg["_sum_h1h2"] / dur * 10).fillna(0.0)
        
        agg = agg.drop(columns=[
            "_sum_force", "_sum_expansions", "_sum_h1", "_sum_h2", 
            "_sum_h1h2", "_sum_duration"
        ])
    else:
        # No non-fatal samples for ANY config — degenerate but possible
        agg = pd.DataFrame(columns=[
            "config_name", "theoretical_latency_s",
            "per_mean", "per_std", "per_p99",
            "sifat_mean", "sifat_std",
            "rtf_median", "rtf_p95",
            "ram_mean_mb", "ram_max_mb",
            "e2e_latency_mean_s", "n_samples",
            "force_commit_rate_10s", "expansion_rate",
            "h1_rate_10s", "h2_rate_10s", "h1h2_rate_10s",
        ])

    # ── Probabilistic fatal-rate + inclusive-PER stats ──────────────────
    # Compute per-config to keep the row count aligned with `agg`.
    rows_prob: List[dict] = []
    for cname, sub in df_all.groupby("config_name"):
        n_total     = len(sub)
        n_fatal     = int(sub["is_fatal"].sum())
        n_non_fatal = n_total - n_fatal
        fatal_rate  = n_fatal / max(n_total, 1)
        fr_lo, fr_hi = _wilson_ci(n_fatal, n_total)
        # PER across ALL non-error samples (the "honest" expected PER)
        per_mean_inclusive = float(sub["per"].mean())
        per_std_inclusive  = float(sub["per"].std(ddof=1)) if n_total > 1 else 0.0
        per_se = per_std_inclusive / (n_total ** 0.5) if n_total > 0 else 0.0
        # 95% CI for mean PER (normal approx). For very small n the
        # Wilson-style trick doesn't help here — PER is continuous on
        # [0, ∞), not a proportion. Bootstrap would be more honest but
        # adds dependencies; we report SE alongside so the consumer can
        # judge confidence directly.
        per_ci_lo = max(0.0, per_mean_inclusive - 1.96 * per_se)
        per_ci_hi = per_mean_inclusive + 1.96 * per_se
        rows_prob.append({
            "config_name":          cname,
            "n_total":              n_total,
            "n_fatal":              n_fatal,
            "n_non_fatal":          n_non_fatal,
            "fatal_rate":           fatal_rate,
            "fatal_rate_ci95_lo":   fr_lo,
            "fatal_rate_ci95_hi":   fr_hi,
            "per_mean_inclusive":   per_mean_inclusive,
            "per_se":               per_se,
            "per_mean_ci95_lo":     per_ci_lo,
            "per_mean_ci95_hi":     per_ci_hi,
        })
    prob_df = pd.DataFrame(rows_prob)

    # Stitch the two views together. Use outer join so configs with ONLY
    # fatal samples still appear (their non-fatal stats will be NaN).
    if not agg.empty:
        agg = agg.merge(prob_df, on="config_name", how="outer").fillna(0.0)
    else:
        agg = prob_df.copy()
        # Fill in the columns expected downstream
        for c in ["theoretical_latency_s", "per_mean", "per_std", "per_p99",
                  "sifat_mean", "sifat_std", "rtf_median", "rtf_p95",
                  "ram_mean_mb", "ram_max_mb", "e2e_latency_mean_s", "n_samples",
                  "apl_s_mean", "acl_s_mean",
                  "force_commit_mean", "h1_only_mean", "h2_only_mean",
                  "h1h2_both_mean", "expansion_rate"]:
            if c not in agg.columns:
                agg[c] = 0.0

    agg = agg.sort_values(["per_mean", "theoretical_latency_s"]).reset_index(drop=True)

    # Pareto frontier definition
    # ──────────────────────────
    # Use `per_mean_inclusive` (the honest, fatals-included expected PER),
    # NOT `per_mean` (which is conditional on success). Otherwise a config
    # that fails 30% of the time but is excellent on the remaining 70%
    # would dominate the frontier — a clearly wrong answer when accuracy
    # is the priority. A config is Pareto if its inclusive PER is within
    # `pareto_per_epsilon` of the best inclusive PER AND its latency is
    # within `pareto_latency_tolerance_s` of the minimum latency in that band.
    agg["is_pareto"] = False
    if not agg.empty:
        # Only configs with at least one non-fatal sample are pareto-eligible
        # (an all-fatal config has no useful operating point).
        eligible = agg[agg["n_samples"] > 0]
        if not eligible.empty:
            best_inclusive = eligible["per_mean_inclusive"].min()
            threshold = best_inclusive + sweep_cfg.pareto_per_epsilon
            cands = eligible[eligible["per_mean_inclusive"] <= threshold]
            if not cands.empty:
                min_lat = cands["theoretical_latency_s"].min()
                pareto_mask = (
                    (agg["n_samples"] > 0)
                    & (agg["per_mean_inclusive"] <= threshold)
                    & (agg["theoretical_latency_s"]
                       <= min_lat + sweep_cfg.pareto_latency_tolerance_s)
                )
                agg.loc[pareto_mask, "is_pareto"] = True

    return agg


# ── Sweep driver ─────────────────────────────────────────────────────────────

def run_sweep(samples: list, configs: List[ConfigSpec], nutq,
              sweep_cfg: SweepRunConfig) -> Dict[str, Any]:
    """
    Run every config across the dataset. Returns the summary dict (also
    written to summary.json on disk).
    """
    sweep_cfg.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path       = sweep_cfg.output_dir / sweep_cfg.csv_filename
    aggregate_path = sweep_cfg.output_dir / sweep_cfg.aggregate_csv_filename
    jsonl_path     = sweep_cfg.output_dir / sweep_cfg.fatal_jsonl_filename
    events_path    = sweep_cfg.output_dir / sweep_cfg.events_jsonl_filename
    summary_path   = sweep_cfg.output_dir / sweep_cfg.summary_filename

    # Start fresh
    for p in (csv_path, jsonl_path, aggregate_path, events_path):
        if p.exists(): p.unlink()

    per_config_stats: Dict[str, dict] = {}
    all_records: List[SampleRecord] = []   # in-memory for aggregation

    for spec in configs:
        logger.info("─" * 70)
        logger.info("Config %-30s  (worst-case latency=%.2fs)",
                    spec.name, spec.worst_case_latency_s)
        per_list: List[float] = []
        rtf_list: List[float] = []
        e2e_list: List[float] = []
        n_fatal = 0
        early_stopped = False

        for idx, sample in enumerate(samples):
            logger.info("  [%3d/%3d] %s  (%.2fs)",
                        idx + 1, len(samples), sample.sample_id, sample.duration_s)
            rec, events = run_one(spec, sample, nutq, sweep_cfg)
            logger.info("ref: [%s]", rec.ref_phonemes)
            logger.info("hyp: [%s]", rec.hyp_phonemes)
            _csv_append(rec, csv_path)
            all_records.append(rec)

            # Append a single JSONL line per run with all events. This is the
            # input for analyze_madd_accuracy.py — self-contained so the
            # analyzer doesn't need to join against results.csv.
            if events:
                run_blob = {
                    "config_name":      spec.name,
                    "sample_id":        sample.sample_id,
                    "mushaf":           sample.mushaf,
                    "audio_duration_s": sample.duration_s,
                    "worst_case_latency_s": spec.worst_case_latency_s,
                    "ref_phonemes":     rec.ref_phonemes,
                    "hyp_phonemes":     rec.hyp_phonemes,
                    "per":              rec.per,
                    "is_fatal":         bool(rec.is_fatal),
                    "events":           events,
                }
                with events_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(run_blob, ensure_ascii=False) + "\n")

            if rec.error:
                logger.warning("    error: %s", rec.error.splitlines()[0])
            else:
                logger.info(
                    "    per=%.2f%% S=%d D=%d I=%d N=%d  sifat=%.1f%%  "
                    "rtf=%.3f  e2e=%.2fs%s",
                    rec.per * 100, rec.S, rec.D, rec.I, rec.N,
                    rec.sifat_accuracy * 100,
                    rec.rtf, rec.e2e_latency_s,
                    "  [FATAL]" if rec.is_fatal else "",
                )

            if rec.is_fatal:
                n_fatal += 1
                _jsonl_append({
                    "config_name":   spec.name,
                    "config_values": _config_snapshot(spec),
                    "sample_id":     rec.sample_id,
                    "mushaf":        rec.mushaf,
                    "audio_duration_s": rec.audio_duration_s,
                    "per":           rec.per,
                    "S": rec.S, "D": rec.D, "I": rec.I, "N": rec.N,
                    "sifat_accuracy": rec.sifat_accuracy,
                    "ref_phonemes":  rec.ref_phonemes,
                    "hyp_phonemes":  rec.hyp_phonemes,
                    "error":         rec.error,
                    "wall_time_s":   rec.wall_time_s,
                    "rtf":           rec.rtf,
                    "n_expansions":  rec.n_expansions,
                }, jsonl_path)
            else:
                per_list.append(rec.per)
                if rec.rtf == rec.rtf:           # not NaN
                    rtf_list.append(rec.rtf)
                if rec.e2e_latency_s > 0:
                    e2e_list.append(rec.e2e_latency_s)

            # Early-stop check
            n_processed = idx + 1
            if n_processed >= sweep_cfg.early_stop_min_samples:
                fatal_rate = n_fatal / n_processed
                if fatal_rate > sweep_cfg.early_stop_rate:
                    logger.warning(
                        "  EARLY STOP for '%s': %d/%d = %.1f%% fatal "
                        "(threshold %.1f%%) — skipping remaining %d sample(s)",
                        spec.name, n_fatal, n_processed,
                        fatal_rate * 100, sweep_cfg.early_stop_rate * 100,
                        len(samples) - n_processed,
                    )
                    early_stopped = True
                    break

        # Compute mean PER both ways for the per-config stats. The
        # "inclusive" form is what _select_best should rank on.
        all_per = per_list + [
            r.per for r in all_records
            if r.config_name == spec.name and r.is_fatal and not r.error
        ]
        per_config_stats[spec.name] = {
            "worst_case_latency_s": spec.worst_case_latency_s,
            "n_processed":     n_fatal + len(per_list),
            "n_fatal":         n_fatal,
            "fatal_rate":      n_fatal / max(1, n_fatal + len(per_list)),
            "mean_per":        statistics.mean(per_list) if per_list else float("inf"),
            "mean_per_inclusive":
                statistics.mean(all_per) if all_per else float("inf"),
            "median_per":      statistics.median(per_list) if per_list else float("inf"),
            "mean_rtf":        statistics.mean(rtf_list) if rtf_list else float("nan"),
            "mean_e2e_s":      statistics.mean(e2e_list) if e2e_list else float("nan"),
            "early_stopped":   early_stopped,
            "config_values":   _config_snapshot(spec),
        }

    # ── Rich per-config aggregate (matches sweep_streaming_advanced) ──────
    agg_df = _build_aggregate(all_records, sweep_cfg)
    aggregate_dict: Dict[str, dict] = {}
    if agg_df is not None and not agg_df.empty:
        agg_df.to_csv(aggregate_path, index=False)
        logger.info("Wrote per-config aggregate to %s", aggregate_path)
        # Index by config_name for the summary JSON
        aggregate_dict = {
            row["config_name"]: {k: v for k, v in row.items() if k != "config_name"}
            for _, row in agg_df.iterrows()
        }
        # Make Pareto flag JSON-friendly
        for v in aggregate_dict.values():
            v["is_pareto"] = bool(v["is_pareto"])

    summary = _select_best(per_config_stats, sweep_cfg,
                           aggregate=aggregate_dict)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


# ── Selection: lowest PER, tie-break by latency ──────────────────────────────

def _select_best(per_config_stats: Dict[str, dict],
                 sweep_cfg: SweepRunConfig,
                 aggregate: Optional[Dict[str, dict]] = None,
                 ) -> Dict[str, Any]:
    """
    Rank surviving configs on `mean_per_inclusive` (the honest PER that
    includes fatals at their actual PER). Tie-break on
    `worst_case_latency_s` within `selection_per_epsilon`.

    Rationale: conditional PER (`mean_per`) would let a config with a
    high fatal rate win by being excellent on the samples it doesn't
    fail on — that's a worse answer than a slightly less accurate but
    consistently reliable config. Inclusive PER aligns the winner with
    real-world expectation.
    """
    surviving = {
        name: s for name, s in per_config_stats.items()
        if not s["early_stopped"]
        and s.get("mean_per_inclusive", float("inf")) != float("inf")
    }

    ranking: List[dict] = []
    if surviving:
        best_per = min(s["mean_per_inclusive"] for s in surviving.values())
        eligible = [
            (name, s) for name, s in surviving.items()
            if s["mean_per_inclusive"] <= best_per + sweep_cfg.selection_per_epsilon
        ]
        # Within the eligible set, configs are "tied for accuracy" by policy.
        # Winner = lowest worst-case latency among them.
        eligible.sort(key=lambda kv: (kv[1]["worst_case_latency_s"],
                                      kv[1]["mean_per_inclusive"]))
        winner_name = eligible[0][0] if eligible else None
        # Display ranking: by mean_per_inclusive then latency (informative)
        for name, s in sorted(
            surviving.items(),
            key=lambda kv: (kv[1]["mean_per_inclusive"],
                            kv[1]["worst_case_latency_s"]),
        ):
            row = {"name": name, **s}
            if aggregate and name in aggregate:
                row["aggregate"] = aggregate[name]
            ranking.append(row)
    else:
        winner_name = None

    summary = {
        "winner": winner_name,
        "selection_policy": {
            "primary":                "min mean_per_inclusive",
            "tie_break":              "min worst_case_latency_s",
            "per_epsilon":            sweep_cfg.selection_per_epsilon,
            "fatal_per_threshold":    sweep_cfg.fatal_per_threshold,
            "early_stop_rate":        sweep_cfg.early_stop_rate,
            "early_stop_min_samples": sweep_cfg.early_stop_min_samples,
            "pareto_per_epsilon":         sweep_cfg.pareto_per_epsilon,
            "pareto_latency_tolerance_s": sweep_cfg.pareto_latency_tolerance_s,
        },
        "ranking_surviving":  ranking,
        "all_configs":        [{"name": n, **s} for n, s in per_config_stats.items()],
        "aggregate":          aggregate or {},
    }
    _print_summary(summary)
    return summary


def _print_summary(summary: dict) -> None:
    print()
    print("═" * 116)
    print("  SWEEP SUMMARY (AdaptiveStreamingMuaalem)")
    print("═" * 116)
    if summary["winner"] is None:
        print("  No surviving config (all early-stopped or errored).")
        return

    policy = summary["selection_policy"]
    print(f"  Selection : winner = min `mean_per_inclusive` "
          f"(±{policy['per_epsilon']*100:.2f}pp tie-band) → "
          f"tie-break = min worst_case_latency_s")
    print(f"  Pareto    : within {policy['pareto_per_epsilon']*100:.2f}pp of best "
          f"`mean_per_inclusive` AND within "
          f"{policy['pareto_latency_tolerance_s']:.2f}s of min latency in band")
    print(f"  Winner    : {summary['winner']}")
    print()

    surviving = summary["ranking_surviving"]
    best_per  = min(
        (r.get("mean_per_inclusive", float("inf")) for r in surviving),
        default=0.0,
    )
    eps = policy["per_epsilon"]

    def _fmt(x, w, fmt):
        if x is None: return f"{'—':>{w}}"
        try:
            if x != x: return f"{'—':>{w}}"   # NaN
        except TypeError:
            return f"{'—':>{w}}"
        return f"{x:>{w}{fmt}}"

    # ── Table 1: accuracy + reliability (the "is this config good?" view) ──
    print("  ── Accuracy & reliability ────────────────────────────────────"
          "────────────────────────────────────────────────")
    print(f"  {'config':<26} {'theo_lat':>8} {'PER_incl':>9} "
          f"{'± 95% CI':>17} {'PER_cond':>9} {'PER_p99':>8} {'sifat':>7} "
          f"{'fatal':>6} {'fatal 95% CI':>14} {'n':>3}  flags")
    print(f"  {'-'*26} {'-'*8} {'-'*9} "
          f"{'-'*17} {'-'*9} {'-'*8} {'-'*7} "
          f"{'-'*6} {'-'*14} {'-'*3}  -----")

    for row in surviving:
        in_eligible = row.get("mean_per_inclusive", 1e9) <= best_per + eps
        agg = row.get("aggregate") or {}
        is_pareto = bool(agg.get("is_pareto", False))

        flags = []
        if in_eligible: flags.append("*")
        if is_pareto:   flags.append("★")
        if row["name"] == summary["winner"]: flags.append("←")
        flags_str = "".join(flags)

        theo_lat       = agg.get("theoretical_latency_s", row["worst_case_latency_s"])
        per_mean       = agg.get("per_mean",            row.get("mean_per", float("nan")))
        per_incl       = agg.get("per_mean_inclusive",  row.get("mean_per_inclusive", float("nan")))
        per_ci_lo      = agg.get("per_mean_ci95_lo",    float("nan"))
        per_ci_hi      = agg.get("per_mean_ci95_hi",    float("nan"))
        per_p99        = agg.get("per_p99",             float("nan"))
        sifat_mean     = agg.get("sifat_mean",          float("nan"))
        fatal_rate     = agg.get("fatal_rate",          row.get("fatal_rate", 0.0))
        fr_lo          = agg.get("fatal_rate_ci95_lo",  float("nan"))
        fr_hi          = agg.get("fatal_rate_ci95_hi",  float("nan"))
        n_total        = agg.get("n_total",             row["n_processed"])

        per_ci_str = (f"[{per_ci_lo*100:>5.2f},{per_ci_hi*100:>5.2f}]%"
                      if per_ci_lo == per_ci_lo else f"{'—':>17}")
        fr_ci_str  = (f"[{fr_lo*100:>4.1f},{fr_hi*100:>4.1f}]%"
                      if fr_lo == fr_lo else f"{'—':>14}")

        print(f"  {row['name']:<26} "
              f"{_fmt(theo_lat,     8, '.2f')} "
              f"{_fmt(per_incl*100, 8, '.2f')}% "
              f"{per_ci_str:>17} "
              f"{_fmt(per_mean*100, 8, '.2f')}% "
              f"{_fmt(per_p99*100,  7, '.2f')}% "
              f"{_fmt(sifat_mean*100, 6, '.1f')}% "
              f"{_fmt(fatal_rate*100, 5, '.1f')}% "
              f"{fr_ci_str:>14} "
              f"{int(n_total):>3}  "
              f"{flags_str}")

    # ── Table 2: behaviour (cost & adequacy) ────────────────────────────
    # The metric ordering here reflects what's actually a QUALITY signal:
    #   force_commit_mean : pure config-quality (budget undersizing).
    #                       Content-independent. Should be 0 for a well-
    #                       sized config.
    #   expansion_rate    : expansions per second of audio. Content-
    #                       coupled but useful for comparing configs on
    #                       the SAME dataset — if config A has 2× the
    #                       rate at the same PER as config B, A is
    #                       paying compute for expansions B doesn't need.
    #   rtf_median, e2e_latency_mean_s, ram_max_mb : standard cost.
    # The H1/H2/H1+H2 breakdown is content-coupled and NOT a quality
    # signal. It appears in a small diagnostic footer at the end.
    print()
    print("  ── Behaviour: cost & budget adequacy ──────────────────────────────────────────")
    print(f"  {'config':<26} {'force':>7} {'exp/s':>7} {'rtf_med':>8} "
          f"{'rtf_p95':>8} {'ram_mb':>8} {'e2e_s':>8}")
    print(f"  {'-'*26} {'-'*7} {'-'*7} {'-'*8} "
          f"{'-'*8} {'-'*8} {'-'*8}")
    for row in surviving:
        agg = row.get("aggregate") or {}
        print(f"  {row['name']:<26} "
              f"{_fmt(agg.get('force_commit_rate_10s', 0.0),  7, '.2f')} "
              f"{_fmt(agg.get('expansion_rate',    0.0),  7, '.2f')} "
              f"{_fmt(agg.get('rtf_median',       float('nan')), 8, '.3f')} "
              f"{_fmt(agg.get('rtf_p95',           0.0),  8, '.3f')} "
              f"{_fmt(agg.get('ram_max_mb',        0.0),  8, '.1f')} "
              f"{_fmt(agg.get('e2e_latency_mean_s', float('nan')), 8, '.2f')}")

    # ── Diagnostic footer: trigger breakdown ────────────────────────────
    # Print this small table SEPARATELY so it's not mistaken for a
    # quality signal. These columns help diagnose *why* a config
    # underperforms, but the magnitudes are functions of audio content
    # as much as of the config — they shouldn't be used to rank.
    print()
    print("  ── Diagnostic: expansion-trigger breakdown (rate per 10s of audio) ────────────")
    print("     ⚠ Content-coupled — use to diagnose a specific config's failure mode, not to rank.")
    print(f"  {'config':<26} {'H1-only':>9} {'H2-only':>9} {'H1+H2':>9}")
    print(f"  {'-'*26} {'-'*9} {'-'*9} {'-'*9}")
    for row in surviving:
        agg = row.get("aggregate") or {}
        print(f"  {row['name']:<26} "
              f"{_fmt(agg.get('h1_rate_10s',   0.0), 9, '.2f')} "
              f"{_fmt(agg.get('h2_rate_10s',   0.0), 9, '.2f')} "
              f"{_fmt(agg.get('h1h2_rate_10s', 0.0), 9, '.2f')}")

    print()
    print("  Legend")
    print("  ──────")
    print("  Accuracy & reliability  (used for ranking)")
    print("    PER_incl  : mean PER over ALL non-error samples, fatals included")
    print("                at their actual PER. The HONEST expected PER on a")
    print("                random sample. Winner & Pareto rank on this.")
    print("    PER_cond  : mean PER over NON-FATAL samples only. 'How accurate")
    print("                when it works' — informative but biased by exclusion.")
    print("    ± 95% CI  : normal-approx CI for PER_incl. SE = std / sqrt(n).")
    print("    fatal     : P(per > fatal_per_threshold) on the dataset.")
    print("    fatal CI  : Wilson 95% CI for the fatal rate. Wilson is well-")
    print("                defined at 0/n and n/n, unlike normal approx.")
    print()
    print("  Cost & budget adequacy  (force is a config-quality signal; others are diagnostic)")
    print("    force     : rate of commits caused by a safety cap per 10s")
    print("                (max_expansions or max_chunk) per sample. PURE")
    print("                config-quality signal — content-independent. > 0")
    print("                means the budget was too small for the audio.")
    print("    exp/s     : total expansions / total audio seconds. Content-")
    print("                coupled, but a useful efficiency proxy when")
    print("                comparing configs on the same dataset: if A has")
    print("                2× the rate at similar PER as B, A is paying")
    print("                compute for expansions B doesn't need.")
    print("    rtf_med   : real-time factor (wall time / audio length), median.")
    print()
    print("  Diagnostic only  (CONTENT-COUPLED, not a quality signal)")
    print("    H1-only   : expansions where ONLY the edge heuristic fired")
    print("                (peak landed near chunk boundary, no trailing run).")
    print("                If a SPECIFIC config has bad PER, high H1-only is")
    print("                a hint that its base/lookahead is too small.")
    print("    H2-only   : expansions where ONLY the trailing-run heuristic")
    print("                fired (active elongation/ghunnah/ikhfa). For")
    print("                Quran content this is EXPECTED to be > 0 — a")
    print("                value of 2 doesn't mean the config is bad, it")
    print("                just means the audio had Madd / Ghunnah content.")
    print("    H1+H2     : both fired together. Compound boundary problems.")
    print()
    print("  Flags     : *=within-PER-epsilon (winner-eligible)  ★=Pareto-frontier  ←=winner")

    early = [c for c in summary["all_configs"] if c["early_stopped"]]
    if early:
        print()
        print("  Early-stopped configs (excluded from selection):")
        for c in early:
            print(f"    {c['name']:<26}  fatal {c['n_fatal']}/{c['n_processed']}")


# ── Default config grid (AdaptiveConfig only) ────────────────────────────────
#
# Edit / extend this function to change what gets compared. The grid is
# Quran-specific: every config is designed to handle a worst-case 6-beat
# Madd at mujawad (slowest) tempo, which lasts ~2.4–3.0 seconds.
#
# Madd encoding (from the paper):
#   - 6-beat Madd al-Mottasel    → 6 consecutive vowel symbols
#   - 6-beat Madd al-Aared       → up to 6 chars
#   - Stressed Ghunnah / Ikhfa   → 3 consecutive nasalized chars
# H2 (trailing-run, min_trailing_run_to_expand=2) catches all of these.
#
# Two latency notions
#   responsiveness_s    = base_chunk_s + right_lookahead_s
#     ↳ how long the streamer waits BEFORE the first commit can fire.
#       This is what "real-time" pertains to; a sub-1.0s value is the
#       practical bar for a live mic UX.
#   worst_case_latency_s = base + max_expansions*expansion_s + lookahead
#     ↳ the latency cap that fires only when EVERY expansion is used,
#       i.e. the audio is in the middle of a long Madd at chunk boundary.
#       This is rare in practice.
#
# Design constraints (paper-driven):
#   1. Total expansion budget (max_expansions * expansion_s) >= 3.0s
#      so a single decode-then-commit cycle can span a worst-case mujawad
#      6-beat Madd without being cut.
#   2. right_lookahead_s > 0 always — boundary recovery benefit is large
#      and the fixed latency cost is acceptable in this domain.
#   3. Empirical defaults stay fixed across the grid: right_pad=0.05,
#      left_ctx=1.0, seam_overlap=10, min_trailing_run_to_expand=2.

def default_configs(quick: bool = False) -> List[ConfigSpec]:
    from streaming_inference import AdaptiveConfig

    BASE = dict(
        right_pad_s=0.05,
        left_context_s=1.0,
        seam_overlap_frames=10,
        seam_match_window=6,
        min_trailing_run_to_expand=2,
    )

    def _spec(name, base_s, exp_s, max_exp, lookahead_s):
        return ConfigSpec(name, AdaptiveConfig(
            base_chunk_s=base_s, expansion_s=exp_s,
            max_expansions=max_exp, right_lookahead_s=lookahead_s, **BASE,
        ))

    if quick:
        # Three points along the latency/accuracy spectrum. Every config has
        # an expansion budget ≥ 3s. The real-time entry has sub-1.0s
        # responsiveness so it's truly live-mic-grade.
        return [
            #     name                     base  exp  m   la   → resp / worst
            # _spec("realtime_b0.5_la0.5",   0.5, 0.5, 6, 0.5),  # 1.0s / 4.0s
            # _spec("balanced_b1.5_la0.5",   1.5, 1.5, 4, 0.5),  # 2.0s / 8.0s
            # _spec("accuracy_b2.0_la1.0",   2.0, 1.5, 8, 1.0),  # 3.0s / 15.0s
            # _spec("nrt_b1.0_e1.0_m4_la0.5", 1.0, 1.0, 4, 0.5),  # resp 1.5s, worst 5.5s, budget 4.0s
            # _spec("nrt_b1.0_e1.0_m8_la0.5", 1.0, 1.0, 8, 0.5),  # resp 1.5s, worst 11.5s, budget 8.0s
            # _spec("nrt_b1.0_e1.5_m3_la0.5", 1.0, 1.5, 3, 0.5),  # resp 1.5s, worst 6.0s, budget 4.5s
            # _spec("nrt_b0.5_e1.0_m8_la1.0", 0.5, 1.0, 8, 1.0),
            # _spec("nrt_b0.3_e1.0_m8_la1.2", 0.3, 1.0, 8, 1.2), 
            _spec("nrt_b0.3_e0.8_m10_la1.0", 0.3, 0.8, 10, 1.0) # Worst-case 1.2+16*0.5 = 9.8s, responsiveness 1.2s
        ]

    grid: List[ConfigSpec] = []

    # ── Tier 1: Real-time (responsiveness ≤ 1.0s, base ≤ 0.75s) ──────────
    # Prioritized first because a live-mic UX is the most demanding
    # latency target. base=0.5 with la=0.5 hits 1.0s responsiveness
    # exactly; base=0.75 with la=0.25 hits the same ceiling via a
    # different mix. Every config still preserves the ≥ 3s expansion
    # budget — so even with a small base, a worst-case 6-beat Madd at
    # mujawad tempo can still be fully spanned by expansion.
    grid += [
        _spec("rt_b0.5_e0.5_m6_la0.5",  0.5, 0.5, 6, 0.5),  # resp 1.0s, worst 4.0s, budget 3.0s
        _spec("rt_b0.5_e1.0_m3_la0.5",  0.5, 1.0, 3, 0.5),  # resp 1.0s, worst 4.0s, budget 3.0s
        _spec("rt_b0.5_e1.5_m2_la0.5",  0.5, 1.5, 2, 0.5),  # resp 1.0s, worst 4.0s, budget 3.0s
        _spec("rt_b0.75_e0.5_m6_la0.25",0.75, 0.5, 6, 0.25),# resp 1.0s, worst 4.0s, budget 3.0s
        _spec("rt_b0.75_e1.0_m3_la0.25",0.75, 1.0, 3, 0.25),# resp 1.0s, worst 4.0s, budget 3.0s
    ]

    # ── Tier 2: Near-real-time (responsiveness 1.0–1.5s) ─────────────────
    # base=1.0 keeps initial commits responsive while giving the encoder
    # a fuller initial window than tier 1. Suitable for live mic when
    # the strict 1s ceiling can stretch slightly.
    grid += [
        _spec("nrt_b1.0_e0.5_m6_la0.5", 1.0, 0.5, 6, 0.5),  # resp 1.5s, worst 4.5s, budget 3.0s
        _spec("nrt_b1.0_e1.0_m3_la0.5", 1.0, 1.0, 3, 0.5),  # resp 1.5s, worst 4.5s, budget 3.0s
        _spec("nrt_b1.0_e1.0_m4_la0.5", 1.0, 1.0, 4, 0.5),  # resp 1.5s, worst 5.5s, budget 4.0s
        _spec("nrt_b1.0_e1.5_m3_la0.5", 1.0, 1.5, 3, 0.5),  # resp 1.5s, worst 6.0s, budget 4.5s
    ]

    # ── Tier 3: Balanced (responsiveness ~2s, worst case ~7–10s) ────────
    # base=1.5 with larger expansion budgets — handles back-to-back
    # maddat or madd at mujawad without ever capping at max_expansions.
    grid += [
        _spec("bal_b1.5_e1.0_m4_la0.5",  1.5, 1.0, 4, 0.5),  # resp 2.0s, worst 6.0s,  budget 4.0s
        _spec("bal_b1.5_e1.5_m4_la0.5",  1.5, 1.5, 4, 0.5),  # resp 2.0s, worst 8.0s,  budget 6.0s
        _spec("bal_b1.5_e1.5_m4_la1.0",  1.5, 1.5, 4, 1.0),  # resp 2.5s, worst 8.5s,  budget 6.0s
    ]

    # ── Tier 4: Accuracy-first (responsiveness ~2.5s, worst case ~10–15s)
    # 6–12s expansion budget; handles compound long elongations (e.g. a
    # Madd al-Mottasel followed by a Madd al-Aared) without rushing.
    grid += [
        _spec("acc_b2.0_e1.0_m6_la0.5",  2.0, 1.0, 6, 0.5),  # resp 2.5s, worst 8.5s,  budget 6.0s
        _spec("acc_b2.0_e1.5_m6_la0.5",  2.0, 1.5, 6, 0.5),  # resp 2.5s, worst 11.5s, budget 9.0s
        _spec("acc_b2.0_e1.5_m6_la1.0",  2.0, 1.5, 6, 1.0),  # resp 3.0s, worst 12.0s, budget 9.0s
        _spec("acc_b2.0_e1.5_m8_la1.0",  2.0, 1.5, 8, 1.0),  # resp 3.0s, worst 15.0s, budget 12.0s
    ]

    # ── Tier 5: Offline-like (worst case ~25s) ───────────────────────────
    # When accuracy is paramount and high latency is acceptable. The
    # entire mid-length segment can sit inside a single chunk if needed.
    grid += [
        _spec("ofl_b2.0_e1.5_m15_la1.0", 2.0, 1.5, 15, 1.0),  # resp 3.0s, worst 25.5s, budget 22.5s
    ]

    return grid


# ── CLI / main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Sweep AdaptiveStreamingMuaalem configs over the targeted "
                    "golden dataset (built via sweep_streaming_advanced)."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("adaptive_sweep_results_muaalem"),
                        help="Where CSV / JSONL / summary land (default adaptive_sweep_results_3/).")
    parser.add_argument("--device", type=str, default="cuda",
                        help="cpu / cuda / cuda:0 / mps (default cuda).")
    parser.add_argument("--model", type=str, default="obadx/muaalem-model-v3_2",
                        help="HF model name or local path (default obadx/muaalem-model-v3_2).")
    parser.add_argument("--dataset-cache", type=str, default="targeted_golden_dataset",
                        help="Cache dir used by build_targeted_golden_dataset "
                             "(default 'targeted_golden_dataset').")
    parser.add_argument("--force-rebuild-dataset", action="store_true",
                        help="Force re-download of the targeted golden dataset.")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Truncate the dataset to this many samples "
                             "(useful for development).")
    parser.add_argument("--quick", action="store_true",
                        help="Use a 3-config grid instead of the full grid.")
    parser.add_argument("--decoder", type=str, choices=["greedy", "beam"], default="greedy",
                        help="Decoder for phonemes (greedy or beam, default greedy).")

    # Selection / fatal policy
    parser.add_argument("--fatal-per", type=float, default=0.15,
                        help="PER above this counts as fatal (default 0.15).")
    parser.add_argument("--early-stop-rate", type=float, default=0.25,
                        help="Drop a config when fatal_rate > this "
                             "(default 0.25).")
    parser.add_argument("--early-stop-min-samples", type=int, default=50,
                        help="Don't early-stop before at least this many "
                             "samples (default 50).")
    parser.add_argument("--per-epsilon", type=float, default=0.005,
                        help="Mean-PER tolerance for the latency tie-break "
                             "(default 0.005 = 0.5pp).")
    parser.add_argument("--pareto-per-epsilon", type=float, default=0.02,
                        help="PER tolerance for Pareto-band membership "
                             "(default 0.02 = 2pp, matches sweep_streaming_advanced).")
    parser.add_argument("--pareto-latency-tolerance-s", type=float, default=0.1,
                        help="Latency tolerance for Pareto-band membership "
                             "(default 0.1s).")
    args = parser.parse_args()

    sweep_cfg = SweepRunConfig(
        fatal_per_threshold=args.fatal_per,
        early_stop_rate=args.early_stop_rate,
        early_stop_min_samples=args.early_stop_min_samples,
        selection_per_epsilon=args.per_epsilon,
        pareto_per_epsilon=args.pareto_per_epsilon,
        pareto_latency_tolerance_s=args.pareto_latency_tolerance_s,
        output_dir=args.output_dir,
    )

    logger.info("Selection policy: min mean_per, tie-break min worst_case_latency_s "
                "(epsilon=%.2fpp)", args.per_epsilon * 100)
    logger.info("Fatal policy: per>%.0f%%; early stop when fatal>%.0f%% of >=%d samples",
                args.fatal_per * 100, args.early_stop_rate * 100,
                args.early_stop_min_samples)

    # ── Dataset (reuses sweep_streaming_advanced's caching) ─────────

    logger.info("Loading targeted golden dataset (cache=%s) ...", args.dataset_cache)
    samples = build_targeted_golden_dataset(
        cache_dir=args.dataset_cache,
        force_rebuild=args.force_rebuild_dataset,
    )
    if args.max_samples is not None:
        samples = samples[: args.max_samples]
        logger.info("Truncated to first %d samples", len(samples))
    logger.info("Dataset: %d samples across %d mushaf(s)",
                len(samples), len({s.mushaf for s in samples}))

    # ── Ground-truth references ──────────────────────────────────────────────
    # As of the latest build_targeted_golden_dataset, the manifest stores
    # `phonemes` and `sifat` per row, and the cache-load path on lines
    # 405–406 of sweep_streaming_advanced.py already wires them onto
    # GoldenSample.ref_phonemes / ref_sifat. So references are available
    # for free here — no separate fetch needed. Samples without refs
    # (e.g. an older manifest) are skipped by run_one with the
    # "no_reference_phonemes" error string and surface in fatal_cases.jsonl.
    n_with_ref = sum(1 for s in samples if s.ref_phonemes)
    logger.info("Reference phonemes available for %d/%d samples",
                n_with_ref, len(samples))
    if n_with_ref < len(samples):
        logger.warning("  %d sample(s) have no `phonemes` field in the "
                       "manifest — they will be skipped (recorded as fatal "
                       "in fatal_cases.jsonl). Re-build the golden dataset "
                       "to refresh the manifest if this is unexpected.",
                       len(samples) - n_with_ref)

    # ── Model (one load, shared across configs) ──────────────────────────────
    logger.info("Loading model '%s' on %s ...", args.model, args.device)
    from nutq_core import Nutq
    from ctc_decoder import GreedyCTCDecoder, BeamCTCDecoder
    
    if args.decoder == "beam":
        decoder = BeamCTCDecoder(blank_id=0, beam_width=10)
        logger.info("Using BeamCTCDecoder (width=10)")
    else:
        decoder = GreedyCTCDecoder(blank_id=0)
        logger.info("Using GreedyCTCDecoder")
        
    if str(args.model).endswith(".onnx"):
        logger.info(f"Loading ONNX model from {args.model}")
        import onnxruntime as ort
        
        class ONNXModelWrapper:
            def __init__(self, onnx_path, device, dtype, length_fn):
                providers = ['CUDAExecutionProvider'] if 'cuda' in str(device) else ['CPUExecutionProvider']
                self.sess = ort.InferenceSession(onnx_path, providers=providers)
                self.output_names = [o.name for o in self.sess.get_outputs()]
                self.device = device
                self.dtype = dtype
                self.length_fn = length_fn

            def __call__(self, input_features, attention_mask=None, **kwargs):
                inputs = {"input_features": input_features.to(torch.float32).cpu().numpy()}
                outs = self.sess.run(None, inputs)
                
                # product/nutq_core.py unconditionally transposes from (T, B, V) to (B, T, V). 
                # Since ONNX returns (B, T, V), we transpose it to (T, B, V) here so it is reversed correctly later.
                logits = {name: torch.from_numpy(o).to(self.device, dtype=self.dtype).transpose(0, 1) for name, o in zip(self.output_names, outs)}
                
                # The convfix_final.onnx model was exported without adapter downsampling.
                # Therefore, the output frames T exactly match the feature extractor frames.
                if attention_mask is not None:
                    lengths = attention_mask.sum([-1]).long()
                else:
                    lengths = torch.tensor([outs[self.output_names.index('phonemes')].shape[1]] * input_features.shape[0])
                    
                return (logits, lengths)

        base_model = "obadx/muaalem-model-v3_2"
        nutq = Nutq(model_name_or_path=base_model, device=args.device, decoder=decoder)
        
        onnx_wrapper = ONNXModelWrapper(
            args.model, 
            torch.device(args.device), 
            torch.bfloat16, 
            nutq.model.encoder._get_feat_extract_output_lengths
        )
        
        nutq.model = onnx_wrapper
        
        # Monkey-patch the streaming inference timing calculations because the ONNX model 
        # (convfix) does NOT use the adapter stride=2 downsampling that the .pt model uses.
        import streaming_inference
        streaming_inference.ADAPTER_LAYERS = 0
    else:
        nutq = Nutq(model_name_or_path=args.model, device=args.device, decoder=decoder)

    # ── Configs to evaluate ──────────────────────────────────────────────────
    configs = default_configs(quick=args.quick)
    logger.info("Sweep grid: %d adaptive configurations%s",
                len(configs), " (quick mode)" if args.quick else "")

    # ── Run ──────────────────────────────────────────────────────────────────
    run_sweep(samples, configs, nutq, sweep_cfg)

    print()
    print(f"  Per-sample CSV  : {sweep_cfg.output_dir / sweep_cfg.csv_filename}")
    print(f"  Aggregate CSV   : {sweep_cfg.output_dir / sweep_cfg.aggregate_csv_filename}")
    print(f"  Per-event JSONL : {sweep_cfg.output_dir / sweep_cfg.events_jsonl_filename}")
    print(f"  Fatal cases     : {sweep_cfg.output_dir / sweep_cfg.fatal_jsonl_filename}")
    print(f"  Summary JSON    : {sweep_cfg.output_dir / sweep_cfg.summary_filename}")


if __name__ == "__main__":
    main()