# static-expr

Documentation for the static/fixed-window streaming experiment track and its relation to the adaptive track.

## Entry point

- Intended static entry script: `/home/runner/work/Streaming-and-Compression-Paper/Streaming-and-Compression-Paper/Streaming/static-expr/sweep_streaming_advanced.py`

This track covers non-adaptive/static workflows (fixed-window and static variants, including KV-cache variants where implemented in that script).

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

Use the same runtime environment as adaptive experiments:

- Working directory: `/home/runner/work/Streaming-and-Compression-Paper/Streaming-and-Compression-Paper/Streaming`
- Install dependencies from `requirements.txt`
- Set Hugging Face token before running sweeps

```powershell
cd "/home/runner/work/Streaming-and-Compression-Paper/Streaming-and-Compression-Paper/Streaming"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:HF_TOKEN = "hf_..."
```

---

## Building the dataset DB for static experiments

Use the same targeted golden dataset cache strategy used by streaming sweep scripts:

- cache root: `targeted_golden_dataset`
- core artifacts:
  - `manifest.json`
  - `audio/*.npy`
  - optional `batch_refs.pkl`

Recommended process:

1. Build/refresh cache once using the adaptive entry script:

```powershell
cd "/home/runner/work/Streaming-and-Compression-Paper/Streaming-and-Compression-Paper/Streaming"
python sweep_adaptive_dataset.py --force-rebuild-dataset
```

2. Run static experiments against the same cache to keep adaptive vs static comparison consistent.

---

## Running static sweep

```powershell
cd "/home/runner/work/Streaming-and-Compression-Paper/Streaming-and-Compression-Paper/Streaming/static-expr"
python sweep_streaming_advanced.py --help
python sweep_streaming_advanced.py
```

---

## Static script arguments

The static script file is referenced as `sweep_streaming_advanced.py`; use its `main()` / `--help` output as the source of truth for argument defaults and meanings in your current branch.

If your checkout does not currently contain `/home/runner/work/Streaming-and-Compression-Paper/Streaming-and-Compression-Paper/Streaming/static-expr/sweep_streaming_advanced.py`, sync to the branch/restructure commit that introduced it before documenting its exact CLI table.

When maintaining this README, document each argument exactly as defined in that script (name, default, and practical purpose), similar to the adaptive CLI table pattern.

---

## Troubleshooting

- CUDA/device errors: fallback to CPU and validate PyTorch/CUDA installation.
- HF auth/rate limits: ensure `HF_TOKEN` and retry after limits cool down.
- Dataset/reference issues: rebuild targeted cache to refresh manifests/audio files.
- Windows execution: prefer quoted paths and PowerShell env syntax.
