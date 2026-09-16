# Streaming and Compression Paper

This repository contains two related workstreams for the Muaalem/Quran phoneme system:

- `Streaming/`: adaptive streaming inference and evaluation tooling.
- `Compression/`: model compression pipeline (`Compression/README.md`).

The main entry point documented here is:

- `Streaming/sweep_adaptive_dataset.py`

It evaluates **adaptive streaming inference** quality/latency tradeoffs on a targeted Quran dataset slice. This script is adaptive-only (it does **not** benchmark fixed-window `StreamingMuaalem` or KV-cache variants).

---

## Tested Environment

The following machine specs were used by the project author:

| Component | Value |
|---|---|
| OS | Microsoft Windows 11 Pro |
| OS version | 10.0.26200 |
| CPU | 12th Gen Intel(R) Core(TM) i5-12450H |
| RAM | 16,891,633,664 bytes (~15.74 GiB / 16 GB installed) |
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU |
| NVIDIA driver | 537.70 |
| GPU memory | 4096 MiB |

> Timing and memory measurements are hardware/driver dependent.

---

## What the Adaptive Sweep Does

`sweep_adaptive_dataset.py`:

1. Builds or loads a targeted golden dataset cache.
2. Loads one model (`--model`, default `obadx/muaalem-model-v3_2`) and decoder (`--decoder`).
3. Runs an **adaptive configuration sweep** (`--quick` or full grid).
4. Writes per-sample + aggregate outputs and selects a winner.

### Adaptive-only scope

The script explicitly focuses on `AdaptiveStreamingMuaalem` and adaptive controls (base chunk, expansion, lookahead, seam recovery). Fixed-window and KV-cache paths are outside this sweep.

---

## Prerequisites and Installation

Run from the `Streaming` directory so local imports resolve correctly.

```powershell
cd "<repo-root>\Streaming"
```

The repository includes `Streaming/requirements.txt`. For this sweep script specifically, the key runtime dependencies visible from imports are:

- `torch`
- `transformers`
- `datasets`
- `numpy`, `pandas`
- `requests`
- `soundfile`, `librosa`
- `onnxruntime` (only if using an `.onnx` model)
- a module providing `ctc_decoder` (`GreedyCTCDecoder` / `BeamCTCDecoder`)

Typical setup pattern:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Hugging Face access

The script queries the Hugging Face Datasets Server and loads HF models. Set a token before running:

```powershell
$env:HF_TOKEN = "hf_..."
```

(Use your own token with dataset/model access permissions.)

---

## Entry Point Usage (`Streaming/sweep_adaptive_dataset.py`)

From `Streaming/`:

### GPU run (default device is `cuda`)

```powershell
python sweep_adaptive_dataset.py --device cuda
```

### CPU run

```powershell
python sweep_adaptive_dataset.py --device cpu
```

### Quick mode (reduced adaptive grid)

```powershell
python sweep_adaptive_dataset.py --quick
```

### Limit number of evaluated samples

```powershell
python sweep_adaptive_dataset.py --max-samples 10
```

### Rebuild targeted dataset cache

```powershell
python sweep_adaptive_dataset.py --force-rebuild-dataset
```

### Choose decoder

```powershell
python sweep_adaptive_dataset.py --decoder greedy
python sweep_adaptive_dataset.py --decoder beam
```

### Choose output directory

```powershell
python sweep_adaptive_dataset.py --output-dir adaptive_sweep_results_custom
```

---

## Targeted Golden Dataset Generation

Dataset build/load is handled inside `build_targeted_golden_dataset()`.

### Defaults

- Dataset: `obadx/muaalem-annotated-v3`
- Cache directory: `targeted_golden_dataset`
- Target ayah ranges:
  - Surah 10: ayahs 77–88
  - Surah 46: ayahs 19–28
  - Surah 50: ayahs 1–18

### Config discovery

The builder enumerates dataset configs where:

- config name starts with `moshaf_`
- and does **not** contain `metadata`

### Data acquisition flow

For each matching config:

1. Uses Hugging Face Datasets Server API (`/size`, `/rows`).
2. Uses binary search to locate the start offset of each target Surah/Ayah range.
3. Crawls rows in batches (`length=100`) to collect target rows.
4. Downloads row audio from URL fields.
5. Reads audio into NumPy, converts to `float32`, resamples to 16 kHz when needed.
6. Applies duration filtering (`min_duration_s` to `max_duration_s`, defaults 0.0 to 300.0).
7. Saves audio as `audio/<sample_id>.npy` and records metadata in `manifest.json`.

### Cache artifacts

Under `--dataset-cache` (default `targeted_golden_dataset`):

- `manifest.json`
- `audio/*.npy`
- optional `batch_refs.pkl` (if present, it is loaded and can override per-sample reference phonemes)

### Cache reuse vs rebuild

- If `manifest.json` exists and `--force-rebuild-dataset` is **not** used, cached data is loaded.
- `--force-rebuild-dataset` rebuilds from the remote dataset/API and refreshes manifest/audio cache.

### Canonical references for evaluation

The manifest’s `phonemes` and `sifat` entries are treated as canonical ground truth for PER and sifat scoring.

- References are **not recomputed** during the sweep.
- If references are missing (e.g., older manifest rows), samples are recorded as fatal with `error="no_reference_phonemes"` and appear in `fatal_cases.jsonl`.
- Rebuilding the dataset refreshes manifest contents.

---

## Sweep Behavior, Selection, and Outputs

### Adaptive grids

- `--quick`: reduced adaptive set (currently 1 config in source)
- full mode: full adaptive grid (17 configs in source)

### Runtime behavior per config

- Evaluates each sample and appends per-sample rows incrementally.
- Marks a sample as fatal when `PER > --fatal-per`.
- Performs early-stop for a config when processed samples reach `--early-stop-min-samples` and fatal rate exceeds `--early-stop-rate`.

### Selection policy

Winner selection uses:

1. **Primary**: minimum `mean_per_inclusive` (PER over all non-error samples, including fatals at their actual PER).
2. **Tie-break**: minimum `worst_case_latency_s` among configs within `--per-epsilon` of best inclusive PER.

Pareto flag policy uses:

- `--pareto-per-epsilon` band on inclusive PER
- plus `--pareto-latency-tolerance-s` around minimum latency in that band

### Output files

All outputs are written under `--output-dir`:

- `results.csv` (per-sample metrics)
- `aggregate.csv` (per-config aggregates)
- `events.jsonl` (per-run COMMIT/FLUSH event payloads)
- `fatal_cases.jsonl` (fatal/error records)
- `summary.json` (winner, policy, ranking, aggregates)

At sweep start, existing `results.csv`, `aggregate.csv`, `events.jsonl`, and `fatal_cases.jsonl` in the selected output directory are removed before new results are written. `summary.json` is overwritten at completion.

---

## CLI Reference

Defaults below are taken from `main()` in `Streaming/sweep_adaptive_dataset.py`.

| Argument | Default | Meaning |
|---|---|---|
| `--output-dir` | `adaptive_sweep_results_muaalem` | Output directory for sweep artifacts (`results.csv`, `aggregate.csv`, JSONL files, `summary.json`). |
| `--device` | `cuda` | Inference device string (examples: `cpu`, `cuda`, `cuda:0`, `mps`). |
| `--model` | `obadx/muaalem-model-v3_2` | HF model ID or local model path; `.onnx` paths trigger ONNXRuntime branch in code. |
| `--dataset-cache` | `targeted_golden_dataset` | Cache folder used by targeted golden dataset builder/loader. |
| `--force-rebuild-dataset` | `False` | Rebuild targeted dataset cache instead of reusing existing manifest/audio files. |
| `--max-samples` | `None` | If set, truncates dataset to first N samples for faster experiments. |
| `--quick` | `False` | Use reduced adaptive sweep grid instead of full adaptive grid. |
| `--decoder` | `greedy` | Decoder choice: `greedy` or `beam`. |
| `--fatal-per` | `0.15` | PER threshold above which a sample is counted as fatal. |
| `--early-stop-rate` | `0.25` | Early-stop a config if fatal rate exceeds this threshold. |
| `--early-stop-min-samples` | `50` | Minimum processed samples before early-stop logic is allowed. |
| `--per-epsilon` | `0.005` | Inclusive-PER tie band for winner eligibility before latency tie-break. |
| `--pareto-per-epsilon` | `0.02` | Inclusive-PER tolerance used for Pareto band membership. |
| `--pareto-latency-tolerance-s` | `0.1` | Latency tolerance (seconds) used for Pareto band membership. |

---

## Troubleshooting

### 1) CUDA / device errors

- If `--device cuda` fails, verify GPU visibility in PyTorch and installed CUDA-compatible wheels.
- Fallback to `--device cpu` for functional verification.

### 2) Hugging Face auth or rate limits

- Set `HF_TOKEN` explicitly.
- The dataset API logic retries on HTTP 429 with backoff, but persistent limits can still fail runs.
- Retry later or reduce repeated rebuilds when possible (reuse cache).

### 3) Missing references (`no_reference_phonemes`)

- Indicates manifest entries missing canonical `phonemes`.
- These are treated as fatal and logged to `fatal_cases.jsonl`.
- Rebuild cache with `--force-rebuild-dataset` to refresh manifest data.

### 4) Audio download failures

- Failed sample downloads are logged during dataset build.
- Network instability or remote URL failures can reduce extracted sample count.
- Re-run with rebuild to retry failed fetches.

### 5) Windows path and command tips

- Use quoted paths for directories containing spaces.
- Prefer PowerShell env syntax (`$env:HF_TOKEN = "..."`).
- Run commands from `Streaming/` so local module imports resolve as expected.

---

## Notes

- Measurements such as `rtf`, `e2e_latency_s`, and RAM/VRAM behavior depend on model, driver, and hardware.
- For compression workflow details, see `Compression/README.md`.
