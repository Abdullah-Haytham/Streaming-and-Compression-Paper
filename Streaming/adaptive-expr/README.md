# adaptive-expr

Adaptive streaming experiment documentation for the Muaalem/Quran phoneme pipeline.

## Entry point

- Repository path: `Streaming/sweep_adaptive_dataset.py`
- Absolute path in this workspace: `/home/runner/work/Streaming-and-Compression-Paper/Streaming-and-Compression-Paper/Streaming/sweep_adaptive_dataset.py`

This workflow evaluates **AdaptiveStreamingMuaalem** configurations only (adaptive chunk expansion, lookahead, seam-recovery behavior). It is separate from fixed-window and KV-cache static workflows.

## What this experiment covers

- Build/load targeted golden dataset cache.
- Sweep adaptive configurations (`--quick` or full grid).
- Score PER and sifat metrics, track fatal cases, and select a winner using inclusive-PER-first policy.
- Write run artifacts (`results.csv`, `aggregate.csv`, `events.jsonl`, `fatal_cases.jsonl`, `summary.json`) into the selected output directory.

## Dataset focus

The adaptive workflow uses a targeted golden set built from:

- Dataset default: `obadx/muaalem-annotated-v3`
- Cache default: `targeted_golden_dataset`
- Target ranges:
  - Surah 10: ayahs 77–88
  - Surah 46: ayahs 19–28
  - Surah 50: ayahs 1–18

References (`phonemes`, `sifat`) are read from the manifest as canonical ground truth and are not recomputed during sweep.

## Quick start

Run from `Streaming/`:

```powershell
cd "/home/runner/work/Streaming-and-Compression-Paper/Streaming-and-Compression-Paper/Streaming"
python sweep_adaptive_dataset.py --device cuda
python sweep_adaptive_dataset.py --quick
python sweep_adaptive_dataset.py --max-samples 10
```

## Notes

- Timings and memory are hardware/driver dependent.
- Set Hugging Face token before execution (for dataset/model access), for example in PowerShell:

```powershell
$env:HF_TOKEN = "hf_..."
```
