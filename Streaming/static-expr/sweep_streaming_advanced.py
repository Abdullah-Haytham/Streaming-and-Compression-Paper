"""
sweep_streaming_advanced.py
════════════════════════════
Production-grade evaluation pipeline for streaming phoneme recognition
and Tajweed/Sifat detection over a stratified multi-reciter test set.

Goal:  sweep (chunk_s, left_ctx_s, right_ctx_s) configurations and find
       Pareto-optimal configs that balance PER, Sifat Accuracy, Latency,
       and Hardware resource consumption for edge deployment.

Outputs:
  global_sweep_metrics.csv     — one row per (config); aggregated statistics
  stratified_mushaf_metrics.csv — one row per (config × mushaf); per-reciter breakdown

Usage:
  # Build golden set and run full sweep
  python sweep_streaming_advanced.py --build-golden --sweep

  # Quick mode (small grid, 5 samples per mushaf)
  python sweep_streaming_advanced.py --build-golden --sweep --quick --n-per-mushaf 5

  # Re-use cached golden set; run sweep on GPU
  python sweep_streaming_advanced.py --sweep --device cuda

  # Only build the golden set (no sweep)
  python sweep_streaming_advanced.py --build-golden

References:
  PER formula:     https://obadx.github.io/quran-muaalem/en/training/evaluation.html
  Dataset:         https://huggingface.co/datasets/obadx/muaalem-annotated-v3
"""


import torch
import argparse
import contextlib
import json
import logging
import io, os
import pickle
import time
import random
import tracemalloc
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, Iterator, Optional
import soundfile as sf
import librosa
from datasets import load_dataset, load_dataset_builder, get_dataset_config_names, Audio as HF_Audio


import numpy as np
import pandas as pd

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sweep_advanced")


# ── Shared metric utilities from sweep_streaming.py ──────────────────────────
# Inline here so this file is self-contained.

SIFAT_LEVELS: list[str] = [
    "hams_or_jahr", "shidda_or_rakhawa", "tafkheem_or_taqeeq",
    "itbaq", "safeer", "qalqla", "tikraar", "tafashie", "istitala", "ghonna",
]


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


# ── Data Structures ───────────────────────────────────────────────────────────

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


@dataclass
class PerRunMetrics:
    """Metrics for one (sample, config) evaluation."""
    sample_id:    str
    mushaf:       str
    duration_s:   float
    chunk_s:      float
    left_ctx_s:   float
    right_ctx_s:  float
    decoder_name: str
    # Accuracy
    per:          float
    S:            int
    D:            int
    I:            int
    N:            int
    sifat_accuracy: float
    per_level_sifat: dict = field(default_factory=dict)
    # Resource
    rtf:          float = 0.0
    peak_ram_mb:  float = 0.0
    e2e_latency_s: float = 0.0
    # Transcripts (for per-sample CSV)
    ref_phonemes: str = ""
    hyp_phonemes: str = ""
    is_fatal:     bool = False
    error:        str  = ""

    @property
    def config_name(self) -> str:
        return f"chunk{self.chunk_s:.2f}_left{self.left_ctx_s:.2f}_right{self.right_ctx_s:.2f}_{self.decoder_name}"

    @property
    def worst_case_latency_s(self) -> float:
        return self.chunk_s + self.right_ctx_s

    @property
    def apl_s(self) -> float:
        return self.chunk_s / 2.0 + self.right_ctx_s

    @property
    def acl_s(self) -> float:
        return self.chunk_s + self.right_ctx_s


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


# ── Golden Dataset ─────────────────────────────────────────────────────────────

def _stratified_sample_by_length(
    samples: list[dict],
    n: int,
    duration_key: str = "duration_s",
) -> list[dict]:
    """
    Pick exactly n samples that span the duration distribution.

    Strategy: sort by duration, then pick at evenly-spaced percentile indices
    so the selected set covers short/medium/long recordings uniformly.
    If fewer than n samples are available, return all of them.
    """
    if len(samples) <= n:
        return samples
    sorted_s = sorted(samples, key=lambda x: x.get(duration_key, 0.0))
    idxs = [int(i * (len(sorted_s) - 1) / (n - 1)) for i in range(n)]
    return [sorted_s[i] for i in idxs]


def build_golden_dataset(
    dataset_name:    str   = "obadx/muaalem-annotated-v3",
    n_per_mushaf:    int   = 10,
    cache_dir:       str   = "golden_dataset",
    force_rebuild:   bool  = False,
    audio_column:    str   = "audio",
    mushaf_column:   str   = "mushaf",
    min_duration_s:  float = 1.0,
    max_duration_s:  float = 30.0,
    split:           str   = "train",
) -> list[GoldenSample]:
    """
    Build (or load from cache) a stratified Golden Test Set.

    Strategy
    ────────
    1. Stream the HuggingFace dataset to avoid full download.
    2. Collect up to max_collect samples per mushaf (to have a pool for stratification).
    3. Filter by duration [min_duration_s, max_duration_s].
    4. For each mushaf, stratify-sample n_per_mushaf items spanning the length
       distribution (short / medium / long).
    5. Save to cache_dir so subsequent runs skip the download.

    Cache Layout
    ────────────
    golden_dataset/
      manifest.json        — metadata (mushaf, sample_id, duration_s)
      audio/<sample_id>.npy — float32 audio arrays at 16 kHz
      batch_refs.pkl        — batch inference outputs (set later)
    """
    cache_path  = Path(cache_dir)
    manifest_f  = cache_path / "manifest.json"
    audio_dir   = cache_path / "audio"
    batch_ref_f = cache_path / "batch_refs.pkl"

    # ── Load from cache ──────────────────────────────────────────────────────
    if manifest_f.exists() and not force_rebuild:
        logger.info("Loading Golden Test Set from cache: %s", cache_dir)
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
                sample_id   = entry["sample_id"],
                mushaf      = entry["mushaf"],
                audio       = audio,
                duration_s  = entry["duration_s"],
            ))

        # Load batch refs if they exist
        if batch_ref_f.exists():
            with open(batch_ref_f, "rb") as f:
                refs: dict[str, dict] = pickle.load(f)
            for s in samples:
                if s.sample_id in refs:
                    s.ref_phonemes = refs[s.sample_id]["phonemes"]
                    s.ref_sifat    = refs[s.sample_id]["sifat"]

        logger.info(
            "Loaded %d samples from %d mushaf(s)",
            len(samples),
            len({s.mushaf for s in samples}),
        )
        return samples

    # ── Build from HuggingFace ────────────────────────────────────────────────
    logger.info("Building Golden Test Set from '%s' (split=%s)", dataset_name, split)

    # Discover available moshaf configs (skip metadata-only configs)
    all_configs = get_dataset_config_names(dataset_name, trust_remote_code=True)
    moshaf_configs = [c for c in all_configs if c.startswith("moshaf_") and "metadata" not in c]
    logger.info("Found %d moshaf configs: %s", len(moshaf_configs), moshaf_configs)

    # Collect into per-mushaf buckets by streaming each config
    max_collect_per_mushaf = max(n_per_mushaf * 10, 100)
    buckets: dict[str, list[dict]] = {}

    for cfg_name in moshaf_configs:
        if cfg_name in buckets and len(buckets[cfg_name]) >= max_collect_per_mushaf:
            continue

        logger.info("Streaming config '%s'...", cfg_name)
        try:
            # 1. First, quickly fetch metadata to see exactly how many segments this Moshaf has

            builder = load_dataset_builder(dataset_name, cfg_name, trust_remote_code=True)
            
            # The config might not have the requested split, fallback to 'train'
            actual_split = split
            if split not in builder.info.splits:
                if "train" in builder.info.splits:
                    actual_split = "train"
                    logger.info("Config '%s': no '%s' split, using 'train'", cfg_name, split)
                else:
                    raise ValueError(f"No valid split found for {cfg_name}")
            
            total_rows = builder.info.splits[actual_split].num_examples
            
            # 2. Pick a random starting offset so we aren't biased towards Al-Fatihah
            offset = 0
            # Buffer by 100 to ensure we don't skip so far that we can't collect enough candidates
            if total_rows and total_rows > (max_collect_per_mushaf + 100):
                import random
                offset = random.randint(0, total_rows - max_collect_per_mushaf - 100)

            # 3. Load the stream and execute the skip
            ds = load_dataset(
                dataset_name, cfg_name, split=actual_split,
                streaming=True, trust_remote_code=True,
            )
            
            if offset > 0:
                logger.info("  [%s] Found %d rows. Randomly skipping %d segments to avoid bias.", cfg_name, total_rows, offset)
                ds = ds.skip(offset)
                
        except Exception as exc:
            logger.warning("Could not load config '%s': %s — skipping", cfg_name, exc)
            continue

        # Cast audio column to 16 kHz resampled mono float32
        try:
            ds = ds.cast_column(audio_column, HF_Audio(sampling_rate=16_000, mono=True))
        except Exception:
            logger.warning("Could not cast audio column for '%s' — will use as-is.", cfg_name)

        n_seen = n_filtered = 0
        mushaf = cfg_name  # Use the HF config name as the mushaf identifier

        for row in ds:
            n_seen += 1

            # Extract audio
            audio_cell = row.get(audio_column, None)
            if audio_cell is None:
                continue
            if isinstance(audio_cell, dict):
                arr = np.asarray(audio_cell.get("array", []), dtype=np.float32)
                sr  = audio_cell.get("sampling_rate", 16_000)
            elif isinstance(audio_cell, np.ndarray):
                arr = audio_cell.astype(np.float32)
                sr  = 16_000
            else:
                continue

            if len(arr) == 0:
                continue

            # Resample if needed
            if sr != 16_000:
                try:
                    arr = librosa.resample(arr, orig_sr=sr, target_sr=16_000)
                except Exception:
                    logger.debug("Skipping row: cannot resample %d → 16000", sr)
                    continue

            duration_s = len(arr) / 16_000
            if not (min_duration_s <= duration_s <= max_duration_s):
                n_filtered += 1
                continue

            # Collect candidate
            buckets.setdefault(mushaf, [])
            if len(buckets[mushaf]) < max_collect_per_mushaf:
                sample_id = f"{mushaf}_{len(buckets[mushaf]):04d}"
                buckets[mushaf].append({
                    "sample_id":  sample_id,
                    "mushaf":     mushaf,
                    "audio":      arr,
                    "duration_s": duration_s,
                    "extra": {
                        k: v for k, v in row.items()
                        if k not in (audio_column,) and not isinstance(v, (np.ndarray, bytes))
                    },
                })

            # Early stopping: once this mushaf has enough candidates
            if len(buckets.get(mushaf, [])) >= max_collect_per_mushaf:
                logger.info("Config '%s' bucket full (%d) — moving to next", cfg_name, max_collect_per_mushaf)
                break

            if n_seen % 500 == 0:
                logger.info(
                    "  [%s] scanned %d rows | %d filtered | collected %d",
                    cfg_name, n_seen, n_filtered, len(buckets.get(mushaf, [])),
                )

        logger.info(
            "Config '%s': %d rows scanned, %d collected, %d filtered",
            cfg_name, n_seen, len(buckets.get(mushaf, [])), n_filtered,
        )

    # ── Stratified sampling ──────────────────────────────────────────────────
    cache_path.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []
    samples_out: list[GoldenSample] = []

    for mushaf, candidates in sorted(buckets.items()):
        selected = _stratified_sample_by_length(candidates, n_per_mushaf)
        logger.info(
            "mushaf=%s: %d candidates → %d selected (%.1f–%.1f s)",
            mushaf, len(candidates), len(selected),
            min(s["duration_s"] for s in selected),
            max(s["duration_s"] for s in selected),
        )
        for entry in selected:
            sid = entry["sample_id"]
            np.save(str(audio_dir / f"{sid}.npy"), entry["audio"])
            manifest.append({
                "sample_id":  sid,
                "mushaf":     entry["mushaf"],
                "duration_s": entry["duration_s"],
            })
            samples_out.append(GoldenSample(
                sample_id  = sid,
                mushaf     = entry["mushaf"],
                audio      = entry["audio"],
                duration_s = entry["duration_s"],
                ref_phonemes = entry.get("phonemes", "").replace(" ", ""),
                ref_sifat    = _parse_sifat_list(entry.get("sifat", [])),
            ))

    with open(manifest_f, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    logger.info(
        "Golden Test Set saved: %d samples across %d mushaf(s) → %s",
        len(samples_out), len(buckets), cache_dir,
    )
    return samples_out

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
    import requests
    
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
    import json
    from pathlib import Path
    from datasets import get_dataset_config_names

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
    import os
    import json
    import io
    import time
    import random
    import requests
    import numpy as np
    import pandas as pd
    from pathlib import Path
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
    import os
    import json
    import time
    import random
    import requests
    import pandas as pd
    from pathlib import Path
    
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

# ── Batch Reference Inference ─────────────────────────────────────────────────

def compute_batch_references(
    samples:    list[GoldenSample],
    nutq,
    batch_size: int   = 8,
    cache_dir:  str   = "golden_dataset",
) -> list[GoldenSample]:
    """
    Run offline (batch) inference on every sample to obtain reference
    phoneme transcripts and sifat annotations.

    Results are cached to golden_dataset/batch_refs.pkl so this expensive
    step only runs once across multiple sweep runs.
    """
    batch_ref_f = Path(cache_dir) / "batch_refs.pkl"

    # Load cached refs
    cached: dict[str, dict] = {}
    if batch_ref_f.exists():
        with open(batch_ref_f, "rb") as f:
            cached = pickle.load(f)
        logger.info("Loaded %d cached batch references", len(cached))

    needs_inference = [s for s in samples if s.sample_id not in cached]
    if needs_inference:
        logger.info(
            "Running batch inference for %d samples (batch_size=%d)…",
            len(needs_inference), batch_size,
        )
        for i in range(0, len(needs_inference), batch_size):
            batch = needs_inference[i : i + batch_size]
            waves = [s.audio for s in batch]
            try:
                with torch.inference_mode():
                    outputs = nutq(waves)
            except Exception as exc:
                logger.error("Batch inference failed for batch %d: %s", i, exc)
                for s in batch:
                    cached[s.sample_id] = {"phonemes": "", "sifat": []}
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                continue

            for s, out in zip(batch, outputs):
                cached[s.sample_id] = {
                    "phonemes": out.phonemes.text,
                    "sifat":    out.sifat,
                }

            logger.info(
                "  [%d/%d] %s → %d phoneme chars",
                i + len(batch), len(needs_inference),
                batch[0].sample_id, len(outputs[0].phonemes.text),
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Persist
        with open(batch_ref_f, "wb") as f:
            pickle.dump(cached, f)
        logger.info("Batch references saved to %s", batch_ref_f)

    # Attach refs to samples
    for s in samples:
        if s.sample_id in cached:
            s.ref_phonemes = cached[s.sample_id]["phonemes"]
            s.ref_sifat    = cached[s.sample_id]["sifat"]
        else:
            logger.warning("No batch reference for %s", s.sample_id)

    return samples


# ── Single Config Evaluation ──────────────────────────────────────────────────

def run_single_config(
    sample:       GoldenSample,
    chunk_s:      float,
    left_ctx_s:   float,
    right_ctx_s:  float,
    decoder_name: str,
    nutq,
    decoders:     dict,
) -> PerRunMetrics:
    """
    Evaluate one (sample, config) pair.

    Returns a PerRunMetrics with PER, sifat accuracy, RTF, peak RAM, and
    end-to-end latency.  The HITLProfiler wraps the entire streaming loop
    so RAM and time cover the full inference, not just model forward passes.
    """
    from streaming_inference import StreamingMuaalem, StreamingConfig

    # Hot-swap decoder on the Nutq instance
    nutq.decoder = decoders[decoder_name]

    cfg = StreamingConfig(
        chunk_length_s  = chunk_s,
        left_context_s  = left_ctx_s,
        right_context_s = right_ctx_s,
    )
    streamer  = StreamingMuaalem(nutq, cfg)
    profiler  = HITLProfiler(sample.duration_s)

    all_sifat: list = []

    with profiler.profile():
        for result in streamer.stream_file(sample.audio):
            if result.token_ids:
                profiler.record_chunk_output()
            for out in result.muaalem_outputs:
                all_sifat.extend(out.sifat)

    streaming_text = streamer.full_text
    per_result     = compute_per(sample.ref_phonemes, streaming_text)
    sifat_result   = compute_sifat_accuracy(sample.ref_sifat, all_sifat)

    return PerRunMetrics(
        sample_id      = sample.sample_id,
        mushaf         = sample.mushaf,
        duration_s     = sample.duration_s,
        chunk_s        = chunk_s,
        left_ctx_s     = left_ctx_s,
        right_ctx_s    = right_ctx_s,
        decoder_name   = decoder_name,
        per            = per_result["per"],
        S              = per_result["S"],
        D              = per_result["D"],
        I              = per_result["I"],
        N              = per_result["N"],
        sifat_accuracy = sifat_result["macro_accuracy"],
        per_level_sifat = sifat_result["per_level"],
        rtf            = profiler.rtf,
        peak_ram_mb    = profiler.peak_ram_mb,
        e2e_latency_s  = profiler.e2e_latency_s,
        ref_phonemes   = sample.ref_phonemes,
        hyp_phonemes   = streaming_text,
    )


# ── Main Sweep Loop ───────────────────────────────────────────────────────────

# Default configuration grids # 0.25, 0.5, 0.75, 1.0, 1.5, 2.0
CHUNK_LENGTHS_FULL  = [0.5, 0.75, 1.0, 1.5]
LEFT_CONTEXTS_FULL  = [0.25, 0.5, 0.75]
# 0.25, 
RIGHT_CONTEXTS_FULL = [0.5, 0.75]

CHUNK_LENGTHS_QUICK  = [0.5, 1.0, 1.5]
LEFT_CONTEXTS_QUICK  = [0.5, 0.75]
RIGHT_CONTEXTS_QUICK = [0.5, 0.75]


def run_sweep(
    golden_set:    list[GoldenSample],
    configs:       list[tuple[float, float, float, str]],
    nutq,
    skip_empty_ref: bool = True,
    fatal_per_threshold: float = 0.15,
    early_stop_rate: float = 0.25,
    early_stop_min_samples: int = 50,
) -> list[PerRunMetrics]:
    """
    Run the full grid sweep over golden_set × configs.

    Matches sweep_adaptive_dataset.py's tolerant fatal policy:
      - A sample is "fatal" when PER > fatal_per_threshold (default 0.15)
      - A config is early-stopped only when fatal_rate > early_stop_rate
        (default 0.25) after at least early_stop_min_samples (default 50)
      - Fatal samples are kept in metrics (not purged)

    Args:
        golden_set:     List of GoldenSample (with ref_phonemes populated).
        configs:        List of (chunk_s, left_ctx_s, right_ctx_s, decoder_name).
        nutq:           Nutq instance (decoder will be hot-swapped per config).
        skip_empty_ref: Skip samples with empty reference phonemes.
        fatal_per_threshold: PER above this counts as fatal.
        early_stop_rate:     Drop a config when fatal_rate > this.
        early_stop_min_samples: Don't early-stop before this many samples.

    Returns:
        List of PerRunMetrics — one per (sample, config) pair.
    """
    from ctc_decoder import GreedyCTCDecoder

    decoders = {
        "greedy": GreedyCTCDecoder(blank_id=0)
    }

    valid_samples = golden_set
    if skip_empty_ref:
        valid_samples = [s for s in golden_set if s.ref_phonemes]
        skipped = len(golden_set) - len(valid_samples)
        if skipped:
            logger.warning("Skipping %d samples with empty reference phonemes", skipped)

    total_runs = len(valid_samples) * len(configs)
    logger.info(
        "Sweep: %d samples × %d configs = %d evaluations",
        len(valid_samples), len(configs), total_runs,
    )

    all_metrics: list[PerRunMetrics] = []
    n_done = 0

    for cfg_idx, (chunk_s, left_s, right_s, dec_name) in enumerate(configs):
        config_label = f"chunk={chunk_s:.2f} left={left_s:.2f} right={right_s:.2f} [{dec_name[0].upper()}]"
        per_vals: list[float] = []
        n_fatal = 0
        early_stopped = False

        for s_idx, sample in enumerate(valid_samples, 1):
            try:
                m = run_single_config(
                    sample       = sample,
                    chunk_s      = chunk_s,
                    left_ctx_s   = left_s,
                    right_ctx_s  = right_s,
                    decoder_name = dec_name,
                    nutq         = nutq,
                    decoders     = decoders,
                )
            except Exception as exc:
                logger.error(
                    "Error on sample=%s config=%s: %s",
                    sample.sample_id, config_label, exc,
                    exc_info=True,
                )
                continue

            # Mark fatal on the metric object
            m.is_fatal = m.per > fatal_per_threshold
            all_metrics.append(m)
            per_vals.append(m.per)
            n_done += 1

            logger.info("  [%3d/%3d] %s  (%.2fs)",
                        s_idx, len(valid_samples), sample.sample_id, sample.duration_s)
            logger.info("ref: [%s]", m.ref_phonemes)
            logger.info("hyp: [%s]", m.hyp_phonemes)

            if m.is_fatal:
                n_fatal += 1
                logger.info(
                    "    per=%.2f%% S=%d D=%d I=%d N=%d  sifat=%.1f%%  "
                    "rtf=%.3f  e2e=%.2fs  [FATAL]",
                    m.per * 100, m.S, m.D, m.I, m.N,
                    m.sifat_accuracy * 100,
                    m.rtf, m.e2e_latency_s,
                )
            else:
                logger.info(
                    "    per=%.2f%% S=%d D=%d I=%d N=%d  sifat=%.1f%%  "
                    "rtf=%.3f  e2e=%.2fs",
                    m.per * 100, m.S, m.D, m.I, m.N,
                    m.sifat_accuracy * 100,
                    m.rtf, m.e2e_latency_s,
                )

            # Early-stop check (matches sweep_adaptive_dataset.py)
            if s_idx >= early_stop_min_samples:
                current_fatal_rate = n_fatal / s_idx
                if current_fatal_rate > early_stop_rate:
                    logger.warning(
                        "  EARLY STOP for '%s': %d/%d = %.1f%% fatal "
                        "(threshold %.1f%%) — skipping remaining %d sample(s)",
                        config_label, n_fatal, s_idx,
                        current_fatal_rate * 100, early_stop_rate * 100,
                        len(valid_samples) - s_idx,
                    )
                    early_stopped = True
                    break

        # Progress summary after each config
        if per_vals:
            logger.info(
                "[%d/%d configs] %s → PER mean=%.2f%% σ=%.2f%% n_fatal=%d/%d RTF_med=%.3f%s",
                cfg_idx + 1, len(configs),
                config_label,
                statistics.mean(per_vals) * 100,
                (statistics.stdev(per_vals) * 100 if len(per_vals) > 1 else 0.0),
                n_fatal, len(per_vals),
                statistics.median(
                    m.rtf for m in all_metrics
                    if m.chunk_s == chunk_s and m.left_ctx_s == left_s
                    and m.right_ctx_s == right_s and m.decoder_name == dec_name
                ),
                "  [EARLY-STOPPED]" if early_stopped else "",
            )

    logger.info("Sweep complete: %d / %d evaluations succeeded", n_done, total_runs)
    return all_metrics


# ── Metric Aggregation ────────────────────────────────────────────────────────

def _p99(series: pd.Series) -> float:
    return float(series.quantile(0.99))

def _p95(series: pd.Series) -> float:
    return float(series.quantile(0.95))

def _median(series: pd.Series) -> float:
    return float(series.median())


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple:
    """Wilson score 95% CI for a binomial proportion k/n."""
    if n <= 0:
        return (0.0, 1.0)
    p_hat = k / n
    denom  = 1.0 + (z * z) / n
    center = (p_hat + (z * z) / (2.0 * n)) / denom
    half   = z * ((p_hat * (1.0 - p_hat) / n
                   + (z * z) / (4.0 * n * n)) ** 0.5) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def aggregate_global(runs: list[PerRunMetrics],
                     fatal_per_threshold: float = 0.15,
                     ) -> pd.DataFrame:
    """
    Aggregate PerRunMetrics into one row per config_name.

    Matches sweep_adaptive_dataset.py's _build_aggregate with:
      - Conditional accuracy stats (per_mean, per_std, per_p99, sifat_*, rtf_*, ram_*)
        computed over non-fatal samples only
      - Probabilistic failure stats (fatal_rate + Wilson 95% CI,
        per_mean_inclusive + SE + 95% CI) computed over ALL samples
      - Pareto frontier based on per_mean_inclusive
    """
    if not runs:
        return pd.DataFrame()

    # Build per-sample rows with is_fatal flag
    non_fatal_rows: list[dict] = []
    all_rows: list[dict] = []
    for m in runs:
        base_row = {
            "config_name":           m.config_name,
            "chunk_s":               m.chunk_s,
            "left_ctx_s":            m.left_ctx_s,
            "right_ctx_s":           m.right_ctx_s,
            "decoder":               m.decoder_name,
            "theoretical_latency_s": m.worst_case_latency_s,
            "apl_s":                 m.apl_s,
            "acl_s":                 m.acl_s,
            "per":                   m.per,
            "sifat_accuracy":        m.sifat_accuracy,
            "rtf":                   m.rtf,
            "peak_ram_mb":           m.peak_ram_mb,
            "e2e_latency_s":         m.e2e_latency_s,
            "is_fatal":              m.is_fatal,
        }
        all_rows.append(base_row)
        if not m.is_fatal:
            non_fatal_rows.append(base_row)

    df_all = pd.DataFrame(all_rows)

    # ── Conditional (non-fatal) accuracy stats ─────────────────────────────
    if non_fatal_rows:
        df_nf = pd.DataFrame(non_fatal_rows)
        agg = (
            df_nf.groupby("config_name")
            .agg(
                theoretical_latency_s = ("theoretical_latency_s", "first"),
                per_mean              = ("per",             "mean"),
                per_std               = ("per",             "std"),
                per_p99               = ("per",             _p99),
                sifat_mean            = ("sifat_accuracy",  "mean"),
                sifat_std             = ("sifat_accuracy",  "std"),
                rtf_median            = ("rtf",             _median),
                rtf_p95               = ("rtf",             _p95),
                ram_mean_mb           = ("peak_ram_mb",     "mean"),
                ram_max_mb            = ("peak_ram_mb",     "max"),
                e2e_latency_mean_s    = ("e2e_latency_s",   "mean"),
                apl_s_mean            = ("apl_s",           "mean"),
                acl_s_mean            = ("acl_s",           "mean"),
                n_samples             = ("per",             "count"),
            )
            .reset_index()
            .fillna(0.0)
        )
    else:
        agg = pd.DataFrame(columns=[
            "config_name", "theoretical_latency_s",
            "per_mean", "per_std", "per_p99",
            "sifat_mean", "sifat_std",
            "rtf_median", "rtf_p95",
            "ram_mean_mb", "ram_max_mb",
            "e2e_latency_mean_s", "n_samples",
            "apl_s_mean", "acl_s_mean",
        ])

    # ── Probabilistic fatal-rate + inclusive-PER stats ──────────────────────
    rows_prob: list[dict] = []
    for cname, sub in df_all.groupby("config_name"):
        n_total     = len(sub)
        n_fatal     = int(sub["is_fatal"].sum())
        n_non_fatal = n_total - n_fatal
        fatal_rate  = n_fatal / max(n_total, 1)
        fr_lo, fr_hi = _wilson_ci(n_fatal, n_total)
        per_mean_inclusive = float(sub["per"].mean())
        per_std_inclusive  = float(sub["per"].std(ddof=1)) if n_total > 1 else 0.0
        per_se = per_std_inclusive / (n_total ** 0.5) if n_total > 0 else 0.0
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

    # Merge conditional + probabilistic stats
    if not agg.empty:
        agg = agg.merge(prob_df, on="config_name", how="outer").fillna(0.0)
    else:
        agg = prob_df.copy()
        for c in ["theoretical_latency_s", "per_mean", "per_std", "per_p99",
                  "sifat_mean", "sifat_std", "rtf_median", "rtf_p95",
                  "ram_mean_mb", "ram_max_mb", "e2e_latency_mean_s", "n_samples",
                  "apl_s_mean", "acl_s_mean"]:
            if c not in agg.columns:
                agg[c] = 0.0

    agg = agg.sort_values(["per_mean_inclusive", "theoretical_latency_s"]).reset_index(drop=True)

    # Pareto frontier using per_mean_inclusive (honest PER, matches adaptive sweep)
    agg["is_pareto"] = False
    if not agg.empty:
        eligible = agg[agg["n_samples"] > 0]
        if not eligible.empty:
            best_inclusive = eligible["per_mean_inclusive"].min()
            threshold = best_inclusive + 0.02
            cands = eligible[eligible["per_mean_inclusive"] <= threshold]
            if not cands.empty:
                min_lat = cands["theoretical_latency_s"].min()
                pareto_mask = (
                    (agg["n_samples"] > 0)
                    & (agg["per_mean_inclusive"] <= threshold)
                    & (agg["theoretical_latency_s"] <= min_lat + 0.1)
                )
                agg.loc[pareto_mask, "is_pareto"] = True

    return agg


def aggregate_mushaf(runs: list[PerRunMetrics]) -> pd.DataFrame:
    """
    Stratified breakdown: one row per (chunk_s, left_ctx_s, right_ctx_s, decoder, mushaf).

    Reveals reciter-specific degradation — e.g. a config that works on slow
    reciters (Murattal) but fails on fast reciters (Hadr).
    """
    if not runs:
        return pd.DataFrame()

    rows: list[dict] = []
    for m in runs:
        rows.append({
            "chunk_s":               m.chunk_s,
            "left_ctx_s":            m.left_ctx_s,
            "right_ctx_s":           m.right_ctx_s,
            "decoder":               m.decoder_name,
            "mushaf":                m.mushaf,
            "theoretical_latency_s": m.chunk_s + m.right_ctx_s,
            "per":                   m.per,
            "sifat_accuracy":        m.sifat_accuracy,
            "rtf":                   m.rtf,
            "peak_ram_mb":           m.peak_ram_mb,
            "duration_s":            m.duration_s,
        })

    df  = pd.DataFrame(rows)
    keys = ["chunk_s", "left_ctx_s", "right_ctx_s", "decoder", "mushaf"]

    agg = (
        df.groupby(keys)
        .agg(
            theoretical_latency_s = ("theoretical_latency_s", "first"),
            per_mean              = ("per",           "mean"),
            per_std               = ("per",           "std"),
            per_worst             = ("per",           "max"),
            sifat_mean            = ("sifat_accuracy", "mean"),
            rtf_median            = ("rtf",           _median),
            ram_mean_mb           = ("peak_ram_mb",   "mean"),
            avg_duration_s        = ("duration_s",    "mean"),
            n_samples             = ("per",           "count"),
        )
        .reset_index()
        .fillna(0.0)
        .sort_values(["chunk_s", "left_ctx_s", "right_ctx_s", "decoder", "mushaf"])
    )

    return agg


# ── Sifat per-level aggregation ───────────────────────────────────────────────

def aggregate_sifat_per_level(runs: list[PerRunMetrics]) -> pd.DataFrame:
    """
    One row per (config, sifat_level) with mean accuracy across all samples.
    Useful for identifying which Tajweed attribute degrades most under streaming.
    """
    rows: list[dict] = []
    for m in runs:
        base = {
            "chunk_s":    m.chunk_s,
            "left_ctx_s": m.left_ctx_s,
            "right_ctx_s": m.right_ctx_s,
            "decoder":    m.decoder_name,
            "sample_id":  m.sample_id,
        }
        for level, info in m.per_level_sifat.items():
            rows.append({**base, "sifat_level": level, "accuracy": info.get("accuracy", 0.0)})

    if not rows:
        return pd.DataFrame()

    df   = pd.DataFrame(rows)
    keys = ["chunk_s", "left_ctx_s", "right_ctx_s", "decoder", "sifat_level"]
    agg  = (
        df.groupby(keys)
        .agg(accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"))
        .reset_index()
        .sort_values(["chunk_s", "sifat_level"])
    )
    return agg


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_summary(global_df: pd.DataFrame) -> None:
    """Print a console summary matching sweep_adaptive_dataset.py's format."""
    if global_df.empty:
        logger.warning("No results to display.")
        return

    sep = "═" * 116
    print(f"\n{sep}")
    print("  SWEEP SUMMARY (Classical Sliding Window)")
    print(sep)

    def _fmt(x, w, fmt):
        if x is None: return f"{'—':>{w}}"
        try:
            if x != x: return f"{'—':>{w}}"
        except TypeError:
            return f"{'—':>{w}}"
        return f"{x:>{w}{fmt}}"

    # Table: accuracy & reliability
    print("  ── Accuracy & reliability ────────────────────────────────────"
          "────────────────────────────────────────────────")
    print(f"  {'config':<30} {'theo_lat':>8} {'PER_incl':>9} "
          f"{'± 95% CI':>17} {'PER_cond':>9} {'PER_p99':>8} {'sifat':>7} "
          f"{'fatal':>6} {'fatal 95% CI':>14} {'n':>4}  flags")
    print(f"  {'-'*30} {'-'*8} {'-'*9} "
          f"{'-'*17} {'-'*9} {'-'*8} {'-'*7} "
          f"{'-'*6} {'-'*14} {'-'*4}  -----")

    best_per = global_df["per_mean_inclusive"].min() if "per_mean_inclusive" in global_df.columns else 0.0
    eps = 0.005

    for _, row in global_df.iterrows():
        in_eligible = row.get("per_mean_inclusive", 1e9) <= best_per + eps
        is_pareto = bool(row.get("is_pareto", False))

        flags = []
        if in_eligible: flags.append("*")
        if is_pareto:   flags.append("★")
        flags_str = "".join(flags)

        per_incl  = row.get("per_mean_inclusive", float("nan"))
        per_ci_lo = row.get("per_mean_ci95_lo", float("nan"))
        per_ci_hi = row.get("per_mean_ci95_hi", float("nan"))
        per_mean  = row.get("per_mean", float("nan"))
        per_p99   = row.get("per_p99", float("nan"))
        sifat_m   = row.get("sifat_mean", float("nan"))
        fatal_r   = row.get("fatal_rate", 0.0)
        fr_lo     = row.get("fatal_rate_ci95_lo", float("nan"))
        fr_hi     = row.get("fatal_rate_ci95_hi", float("nan"))
        n_total   = int(row.get("n_total", 0))

        per_ci_str = (f"[{per_ci_lo*100:>5.2f},{per_ci_hi*100:>5.2f}]%"
                      if per_ci_lo == per_ci_lo else f"{'—':>17}")
        fr_ci_str  = (f"[{fr_lo*100:>4.1f},{fr_hi*100:>4.1f}]%"
                      if fr_lo == fr_lo else f"{'—':>14}")

        cname = row.get("config_name", "")
        print(f"  {cname:<30} "
              f"{_fmt(row.get('theoretical_latency_s'), 8, '.2f')} "
              f"{_fmt(per_incl*100, 8, '.2f')}% "
              f"{per_ci_str:>17} "
              f"{_fmt(per_mean*100, 8, '.2f')}% "
              f"{_fmt(per_p99*100, 7, '.2f')}% "
              f"{_fmt(sifat_m*100, 6, '.1f')}% "
              f"{_fmt(fatal_r*100, 5, '.1f')}% "
              f"{fr_ci_str:>14} "
              f"{n_total:>4}  "
              f"{flags_str}")

    # Table: cost
    print()
    print("  ── Behaviour: cost ──────────────────────────────────────────────────────────────")
    print(f"  {'config':<30} {'rtf_med':>8} {'rtf_p95':>8} {'ram_mb':>8} {'e2e_s':>8}")
    print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for _, row in global_df.iterrows():
        cname = row.get("config_name", "")
        print(f"  {cname:<30} "
              f"{_fmt(row.get('rtf_median', float('nan')), 8, '.3f')} "
              f"{_fmt(row.get('rtf_p95', 0.0), 8, '.3f')} "
              f"{_fmt(row.get('ram_max_mb', 0.0), 8, '.1f')} "
              f"{_fmt(row.get('e2e_latency_mean_s', float('nan')), 8, '.2f')}")

    print()
    print("  Flags: *=within-PER-epsilon (winner-eligible)  ★=Pareto-frontier")
    print(sep)


def write_per_sample_csv(runs: list[PerRunMetrics], path: Path) -> None:
    """Write a per-sample CSV matching sweep_adaptive_dataset.py's results.csv."""
    import csv
    fields = [
        "config_name", "sample_id", "mushaf",
        "audio_duration_s", "worst_case_latency_s", "apl_s", "acl_s",
        "per", "S", "D", "I", "N", "sifat_accuracy",
        "is_fatal", "error",
        "rtf", "e2e_latency_s", "peak_ram_mb",
        "ref_phonemes", "hyp_phonemes",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for m in runs:
            w.writerow({
                "config_name":         m.config_name,
                "sample_id":           m.sample_id,
                "mushaf":              m.mushaf,
                "audio_duration_s":    m.duration_s,
                "worst_case_latency_s": m.worst_case_latency_s,
                "apl_s":               m.apl_s,
                "acl_s":               m.acl_s,
                "per":                 m.per,
                "S":                   m.S,
                "D":                   m.D,
                "I":                   m.I,
                "N":                   m.N,
                "sifat_accuracy":      m.sifat_accuracy,
                "is_fatal":            m.is_fatal,
                "error":               m.error,
                "rtf":                 m.rtf,
                "e2e_latency_s":       m.e2e_latency_s,
                "peak_ram_mb":         m.peak_ram_mb,
                "ref_phonemes":        m.ref_phonemes,
                "hyp_phonemes":        m.hyp_phonemes,
            })


def write_fatal_cases(runs: list[PerRunMetrics], path: Path) -> None:
    """Write fatal cases as JSONL matching sweep_adaptive_dataset.py."""
    fatal_runs = [m for m in runs if m.is_fatal]
    if not fatal_runs:
        return
    with path.open("w", encoding="utf-8") as f:
        for m in fatal_runs:
            obj = {
                "config_name":      m.config_name,
                "config_values": {
                    "chunk_s": m.chunk_s,
                    "left_ctx_s": m.left_ctx_s,
                    "right_ctx_s": m.right_ctx_s,
                    "decoder": m.decoder_name,
                },
                "sample_id":        m.sample_id,
                "mushaf":           m.mushaf,
                "audio_duration_s": m.duration_s,
                "per":              m.per,
                "S": m.S, "D": m.D, "I": m.I, "N": m.N,
                "sifat_accuracy":   m.sifat_accuracy,
                "ref_phonemes":     m.ref_phonemes,
                "hyp_phonemes":     m.hyp_phonemes,
                "error":            m.error or None,
                "rtf":              m.rtf,
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def write_summary_json(global_df: pd.DataFrame, path: Path) -> None:
    """Write a summary.json matching sweep_adaptive_dataset.py's format."""
    if global_df.empty:
        path.write_text("{}", encoding="utf-8")
        return

    # Winner: min per_mean_inclusive, tie-break min theoretical_latency_s
    eps = 0.005
    best_per = global_df["per_mean_inclusive"].min()
    eligible = global_df[global_df["per_mean_inclusive"] <= best_per + eps]
    eligible_sorted = eligible.sort_values(["theoretical_latency_s", "per_mean_inclusive"])
    winner = eligible_sorted.iloc[0]["config_name"] if not eligible_sorted.empty else None

    ranking = []
    for _, row in global_df.iterrows():
        entry = {
            "name": row["config_name"],
            "worst_case_latency_s": row["theoretical_latency_s"],
            "n_processed": int(row.get("n_total", 0)),
            "n_fatal": int(row.get("n_fatal", 0)),
            "fatal_rate": row.get("fatal_rate", 0.0),
            "mean_per": row.get("per_mean", 0.0),
            "mean_per_inclusive": row.get("per_mean_inclusive", 0.0),
            "median_per": 0.0,  # not tracked per-config here
            "mean_rtf": row.get("rtf_median", 0.0),
            "mean_e2e_s": row.get("e2e_latency_mean_s", 0.0),
            "early_stopped": False,
            "config_values": {
                "chunk_s": float(row.get("config_name", "").split("_")[0].replace("chunk", "")) if "chunk" in str(row.get("config_name", "")) else 0.0,
            },
            "aggregate": {k: (bool(v) if k == "is_pareto" else v)
                         for k, v in row.to_dict().items()
                         if k != "config_name"},
        }
        ranking.append(entry)

    summary = {
        "winner": winner,
        "selection_policy": {
            "primary": "min mean_per_inclusive",
            "tie_break": "min worst_case_latency_s",
            "per_epsilon": eps,
            "fatal_per_threshold": 0.15,
        },
        "ranking_surviving": ranking,
    }
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


# ── CLI ───────────────────────────────────────────────────────────────────────

# Target latencies spanning the failure regime of classical + adaptive worst-cases
TARGET_LATENCIES = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.5, 6.0, 8.0, 8.5, 11.5, 12.0, 15.0, 25.5]

def build_configs(quick: bool) -> list[tuple]:
    latencies = TARGET_LATENCIES[:3] if quick else TARGET_LATENCIES
    rights = [0.5, 1.0]
    lefts  = [1.0]
    
    configs = []
    for lat in latencies:
        for r in rights:
            c = lat - r
            if c > 0:
                for l in lefts:
                    configs.append((c, l, r, "greedy"))
    return configs


def main():
    parser = argparse.ArgumentParser(
        description="Advanced streaming sweep: multi-reciter stratified evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset
    ds_grp = parser.add_argument_group("Dataset")
    ds_grp.add_argument("--build-golden", action="store_true",
                         help="Build (or rebuild) the Golden Test Set from HuggingFace")
    ds_grp.add_argument("--targeted", action="store_true",
                         help="Build or load the Targeted Ayahs Test Set instead of random stratification")
    ds_grp.add_argument("--force-rebuild", action="store_true",
                         help="Ignore cache and re-download the Golden Test Set")
    ds_grp.add_argument("--dataset", default="obadx/muaalem-annotated-v3",
                         help="HuggingFace dataset name")
    ds_grp.add_argument("--n-per-mushaf", type=int, default=10,
                         help="Number of samples to select per mushaf")
    ds_grp.add_argument("--cache-dir", default="golden_dataset",
                         help="Directory to cache the Golden Test Set")
    ds_grp.add_argument("--split", default="test",
                         help="Dataset split to load")
    ds_grp.add_argument("--min-duration", type=float, default=1.0,
                         help="Minimum sample duration (seconds)")
    ds_grp.add_argument("--max-duration", type=float, default=30.0,
                         help="Maximum sample duration (seconds)")

    # Model
    mdl_grp = parser.add_argument_group("Model")
    mdl_grp.add_argument("--model", default="obadx/muaalem-model-v3_2",
                          help="Local .pt bundle or HuggingFace model name")
    mdl_grp.add_argument("--device", default="cuda",
                          help="Torch device: cpu / cuda / mps")

    # Sweep
    sw_grp = parser.add_argument_group("Sweep")
    sw_grp.add_argument("--sweep", action="store_true",
                         help="Run the configuration sweep")
    sw_grp.add_argument("--quick", action="store_true",
                         help="Use reduced grid for fast iteration")
    sw_grp.add_argument("--target-latency", type=float, default=None,
                         help="Only sweep configs with latency ≤ this value (s)")

    # Output
    out_grp = parser.add_argument_group("Output")
    out_grp.add_argument("--out-dir", default="base_sweep_results_muaalem",
                          help="Directory to write CSV results")
    out_grp.add_argument("--global-csv", default="global_sweep_metrics.csv")
    out_grp.add_argument("--mushaf-csv", default="stratified_mushaf_metrics.csv")
    out_grp.add_argument("--sifat-csv", default="sifat_level_metrics.csv")

    args = parser.parse_args()

    # Default to targeted golden dataset to match adaptive sweep
    args.targeted = True
    if args.cache_dir == "golden_dataset":
        args.cache_dir = "targeted_golden_dataset"

    if not args.build_golden and not args.sweep:
        parser.error("Specify at least one of --build-golden or --sweep")

    # ── 1. Golden dataset ────────────────────────────────────────────────────
    golden: list[GoldenSample] = []
    
    # Determine which cache dir to use based on mode
    active_cache_dir = args.cache_dir
    if args.targeted and active_cache_dir == "golden_dataset":
        active_cache_dir = "targeted_golden_dataset"

    if args.build_golden or Path(active_cache_dir).exists():
        if args.targeted:
            golden = build_targeted_golden_dataset(
                dataset_name   = args.dataset,
                cache_dir      = active_cache_dir,
                force_rebuild  = args.force_rebuild,
                min_duration_s = args.min_duration,
                max_duration_s = args.max_duration,
            )
        else:
            golden = build_golden_dataset(
                dataset_name   = args.dataset,
                n_per_mushaf   = args.n_per_mushaf,
                cache_dir      = active_cache_dir,
                force_rebuild  = args.force_rebuild,
                split          = args.split,
                min_duration_s = args.min_duration,
                max_duration_s = args.max_duration,
            )
    else:
        parser.error(f"No dataset found at '{active_cache_dir}'. Run with --build-golden first.")

    if not golden:
        logger.error("Golden dataset is empty. Aborting.")
        return

    if not args.sweep:
        logger.info("Golden dataset built. Use --sweep to run the evaluation.")
        return

    # ── 2. Load model ────────────────────────────────────────────────────────
    from nutq_core import Nutq
    from ctc_decoder import GreedyCTCDecoder

    # Force torch.device conversion & verify GPU
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        device = torch.device("cpu")

    if str(args.model).endswith(".onnx"):
        logger.info(f"Loading ONNX model from {args.model}")
        import onnxruntime as ort
        
        class ONNXModelWrapper:
            def __init__(self, onnx_path, device, dtype):
                providers = ['CUDAExecutionProvider'] if 'cuda' in str(device) else ['CPUExecutionProvider']
                self.sess = ort.InferenceSession(onnx_path, providers=providers)
                self.output_names = [o.name for o in self.sess.get_outputs()]
                self.device = device
                self.dtype = dtype

            def __call__(self, input_features, attention_mask=None, **kwargs):
                inputs = {"input_features": input_features.to(torch.float32).cpu().numpy()}
                outs = self.sess.run(None, inputs)
                logits = {name: torch.from_numpy(o).to(self.device, dtype=self.dtype) for name, o in zip(self.output_names, outs)}
                return (logits,)

        base_model = "inference_ready.pt"
        nutq = Nutq(
            model_name_or_path = base_model,
            decoder            = GreedyCTCDecoder(blank_id=0),
            device             = device
        )
        onnx_wrapper = ONNXModelWrapper(args.model, device, torch.bfloat16)
        def _get_feat_extract_output_lengths(lengths):
            # The convfix_final.onnx model was exported without adapter downsampling.
            # So T exactly equals the feature extractor output frames.
            return lengths
        onnx_wrapper._get_feat_extract_output_lengths = _get_feat_extract_output_lengths
        nutq.model = onnx_wrapper
        
        # Monkey-patch the streaming inference timing calculations because the ONNX model 
        # (convfix) does NOT use the adapter stride=2 downsampling that the .pt model uses.
        import streaming_inference
        streaming_inference.ADAPTER_LAYERS = 0
    else:
        logger.info(f"Initializing model on {device} (dtype={torch.bfloat16})...")
        nutq = Nutq(
            model_name_or_path = args.model,
            decoder            = GreedyCTCDecoder(blank_id=0),
            device             = device,
            dtype              = torch.bfloat16,
        )

    # Verify model is actually on the target device
    if hasattr(nutq, "model") and hasattr(nutq.model, "device"):
        actual_device = nutq.model.device
        logger.info(f"Model verified on: {actual_device}")
    elif hasattr(nutq, "device"):
        logger.info(f"Nutq device attr: {nutq.device}")
    else:
        logger.warning("Could not verify model device. Ensure Nutq moves tensors correctly.")

    # ── 3. Ground-truth references (from manifest, no batch inference) ────
    # References are already loaded from the manifest's 'phonemes' and 'sifat'
    # columns by build_targeted_golden_dataset — same as sweep_adaptive_dataset.py.
    n_with_ref = sum(1 for s in golden if s.ref_phonemes)
    logger.info("%d / %d samples have reference phonemes (from manifest)", n_with_ref, len(golden))
    if n_with_ref < len(golden):
        logger.warning(
            "  %d sample(s) have no 'phonemes' field in the manifest — "
            "they will be skipped. Re-build the targeted dataset to refresh.",
            len(golden) - n_with_ref,
        )

    # ── 4. Build config grid ─────────────────────────────────────────────────
    configs = build_configs(args.quick)

    if args.target_latency is not None:
        before = len(configs)
        configs = [c for c in configs if (c[0] + c[2]) <= args.target_latency]
        logger.info(
            "Filtered configs by latency ≤ %.2f s: %d → %d",
            args.target_latency, before, len(configs),
        )

    if not configs:
        logger.error("No configs to sweep after filtering. Aborting.")
        return

    logger.info("Grid: %d configurations × %d samples = %d runs",
                len(configs), n_with_ref, len(configs) * n_with_ref)

    # ── 5. Run sweep ─────────────────────────────────────────────────────────
    all_metrics = run_sweep(
        golden_set = golden,
        configs    = configs,
        nutq       = nutq
    )

    if not all_metrics:
        logger.error("No successful evaluations. Check model and dataset.")
        return

    # ── 6. Save Raw Results First (Crash Protection) ─────────────────────────
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.csv"
    write_per_sample_csv(all_metrics, results_path)

    # ── 7. Aggregate ─────────────────────────────────────────────────────────
    global_df = aggregate_global(all_metrics)
    mushaf_df = aggregate_mushaf(all_metrics)
    sifat_df  = aggregate_sifat_per_level(all_metrics)

    print_summary(global_df)

    # Aggregate CSV
    aggregate_path = out_dir / args.global_csv
    global_df.to_csv(aggregate_path, index=False)

    # Mushaf breakdown
    mushaf_path = out_dir / args.mushaf_csv
    mushaf_df.to_csv(mushaf_path, index=False)

    # Sifat breakdown
    sifat_path = out_dir / args.sifat_csv
    if not sifat_df.empty:
        sifat_df.to_csv(sifat_path, index=False)

    # Fatal cases JSONL (matches sweep_adaptive_dataset.py's fatal_cases.jsonl)
    fatal_path = out_dir / "fatal_cases.jsonl"
    write_fatal_cases(all_metrics, fatal_path)

    # Summary JSON (matches sweep_adaptive_dataset.py's summary.json)
    summary_path = out_dir / "summary.json"
    write_summary_json(global_df, summary_path)

    n_fatal = sum(1 for m in all_metrics if m.is_fatal)
    logger.info("Results written:")
    logger.info("  %s  (%d per-sample rows)", results_path, len(all_metrics))
    logger.info("  %s  (%d configs)", aggregate_path, len(global_df))
    logger.info("  %s  (%d rows)", mushaf_path, len(mushaf_df))
    logger.info("  %s  (%d rows)", sifat_path, len(sifat_df) if not sifat_df.empty else 0)
    logger.info("  %s  (%d fatal cases)", fatal_path, n_fatal)
    logger.info("  %s", summary_path)


if __name__ == "__main__":
    main()
    # print(check_missing_segments())
    # missing = check_missing_segments()
    # if missing:
    #     logger.info("Triggering targeted fetch for missing segments...")
    #     fetch_missing_segments(missing)
    # fetch_metadata_columns()