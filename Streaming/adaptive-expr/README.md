# adaptive-expr

Comprehensive documentation for the adaptive streaming experiment track of the Muaalem/Quran phoneme pipeline.

## Entry point

- Source file: `/home/runner/work/Streaming-and-Compression-Paper/Streaming-and-Compression-Paper/Streaming/sweep_adaptive_dataset.py`

This script evaluates **AdaptiveStreamingMuaalem only**. It does not benchmark the fixed-window static pipeline or KV-cache static variants.

---

## Tested environment

| Component | Value |
|---|---|
| OS | Microsoft Windows 11 Pro |
| OS version | 10.0.26200 |
| CPU | 12th Gen Intel(R) Core(TM) i5-12450H |
| RAM | 16,891,633,664 bytes (~15.74 GiB / 16 GB installed) |
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU |
| NVIDIA driver | 537.70 |
| GPU memory | 4096 MiB |

> Timing and memory measurements are hardware-dependent.

---

## Setup and prerequisites

Run from:

- `/home/runner/work/Streaming-and-Compression-Paper/Streaming-and-Compression-Paper/Streaming`

The script imports and uses:

- `torch`
- `transformers`
- `datasets`
- `numpy`, `pandas`
- `requests`
- `soundfile`, `librosa`
- `onnxruntime` (only when `--model` points to `.onnx`)
- a `ctc_decoder` module providing `GreedyCTCDecoder` and `BeamCTCDecoder`

Install from the repository requirements file in `Streaming/`:

```powershell
cd "/home/runner/work/Streaming-and-Compression-Paper/Streaming-and-Compression-Paper/Streaming"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Hugging Face access

Set token before runs (dataset API + model pulls):

```powershell
$env:HF_TOKEN = "hf_..."
```

---

## Building the dataset DB (targeted golden dataset cache)

In this workflow, the “DB” is the cached targeted golden dataset built by `build_targeted_golden_dataset()`.

### Defaults

- Dataset: `obadx/muaalem-annotated-v3`
- Cache directory: `targeted_golden_dataset`
- Target ranges:
  - Surah 10, ayahs 77–88
  - Surah 46, ayahs 19–28
  - Surah 50, ayahs 1–18

### Config selection

Only dataset configs that:

- start with `moshaf_`
- and do **not** contain `metadata`

### Build flow

For each selected config, the script:

1. Calls Hugging Face Datasets Server API (`/size` and `/rows`).
2. Uses binary search to find the starting offset for each target range.
3. Crawls rows in batches (`length=100`).
4. Downloads audio URLs from row payloads.
5. Loads audio, converts to `float32` NumPy arrays.
6. Resamples to 16 kHz when needed.
7. Applies duration filtering.
8. Stores cached artifacts.

### Cached artifacts

Inside `--dataset-cache` (default `targeted_golden_dataset`):

- `manifest.json`
- `audio/*.npy`
- optional `batch_refs.pkl`

### Reuse vs rebuild

- Reuse cache: run normally.
- Force rebuild DB/cache:

```powershell
python sweep_adaptive_dataset.py --force-rebuild-dataset
```

### Canonical references and fatal behavior

- `manifest.json` phoneme/sifat references are canonical ground truth for PER/sifat metrics.
- References are **not recomputed** during sweep.
- Missing references in older/incomplete manifests produce fatal records with `error="no_reference_phonemes"`.
- Rebuilding refreshes manifest contents.

---

## Running the adaptive sweep

### Common commands

```powershell
# GPU (default device is cuda)
python sweep_adaptive_dataset.py --device cuda

# CPU
python sweep_adaptive_dataset.py --device cpu

# Quick grid
python sweep_adaptive_dataset.py --quick

# Limit samples
python sweep_adaptive_dataset.py --max-samples 10

# Decoder choice
python sweep_adaptive_dataset.py --decoder greedy
python sweep_adaptive_dataset.py --decoder beam

# Custom output folder
python sweep_adaptive_dataset.py --output-dir adaptive_sweep_results_custom
```

### Sweep behavior

- Adaptive configs only.
- Quick grid and full grid are both adaptive sets.
- Per-sample and aggregate metrics are written under the selected output directory.
- Fatal policy: sample is fatal when `PER > --fatal-per`.
- Early stopping: config is dropped when fatal rate exceeds `--early-stop-rate` after at least `--early-stop-min-samples` samples.
- Winner policy: minimum **inclusive PER** (`mean_per_inclusive`), then latency tie-break within `--per-epsilon`.
- Pareto policy uses `--pareto-per-epsilon` and `--pareto-latency-tolerance-s`.

### Output files

- `results.csv`
- `aggregate.csv`
- `events.jsonl`
- `fatal_cases.jsonl`
- `summary.json`

At sweep start, existing `results.csv`, `aggregate.csv`, `events.jsonl`, and `fatal_cases.jsonl` in the output directory are removed before new writes.

---

## CLI arguments (`main()` reference)

| Argument | Default | Meaning |
|---|---|---|
| `--output-dir` | `adaptive_sweep_results_muaalem` | Output directory for sweep artifacts. |
| `--device` | `cuda` | Device string (`cpu`, `cuda`, `cuda:0`, `mps`). |
| `--model` | `obadx/muaalem-model-v3_2` | HF model ID or local model path. |
| `--dataset-cache` | `targeted_golden_dataset` | Cache directory for targeted dataset DB artifacts. |
| `--force-rebuild-dataset` | `False` | Rebuild dataset cache even if manifest exists. |
| `--max-samples` | `None` | Truncate dataset to first N samples. |
| `--quick` | `False` | Use quick adaptive grid instead of full grid. |
| `--decoder` | `greedy` | Decoder type: `greedy` or `beam`. |
| `--fatal-per` | `0.15` | PER threshold above which a sample is fatal. |
| `--early-stop-rate` | `0.25` | Early-stop threshold for fatal rate. |
| `--early-stop-min-samples` | `50` | Minimum processed samples before early-stop can trigger. |
| `--per-epsilon` | `0.005` | Inclusive-PER tie-band for winner eligibility. |
| `--pareto-per-epsilon` | `0.02` | Inclusive-PER tolerance for Pareto membership. |
| `--pareto-latency-tolerance-s` | `0.1` | Latency tolerance (seconds) for Pareto membership. |

---

## Troubleshooting

### CUDA / device availability

- If CUDA fails, verify CUDA-compatible PyTorch install and visible GPU.
- Use `--device cpu` for fallback runs.

### HF auth / rate limits

- Ensure `HF_TOKEN` is set.
- The script retries on 429 responses, but sustained limits can still interrupt builds.

### Missing references

- `no_reference_phonemes` indicates missing manifest references.
- Rebuild with `--force-rebuild-dataset`.

### Audio download failures

- Network/server failures can skip samples during dataset build.
- Rebuild later to retry downloads.

### Windows path/command notes

- Quote paths with spaces.
- Use PowerShell env syntax (`$env:HF_TOKEN = "..."`).

