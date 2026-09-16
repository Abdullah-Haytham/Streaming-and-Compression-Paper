# static-expr

Static streaming experiment documentation for the Muaalem/Quran phoneme pipeline.

## Entry point

- Script name: `sweep_streaming_advanced.py`
- Intended location in this structure: `Streaming/static-expr/sweep_streaming_advanced.py`

This workflow is for **static/windowed streaming evaluation** and is the companion to the adaptive workflow documented under `Streaming/adaptive-expr/README.md`.

## Goal

Use the static experiment track to benchmark non-adaptive streaming settings (including fixed-window behavior and, where implemented in the script, KV-cache variants) and compare quality/latency trade-offs under a consistent dataset/evaluation policy.

## Typical workflow

1. Select static sweep settings in `sweep_streaming_advanced.py`.
2. Run the sweep on the intended dataset split/cache.
3. Inspect produced per-sample and aggregate artifacts for accuracy, latency, and failure behavior.
4. Compare results against adaptive-expr outputs when choosing deployment policy.

## Running

From the `Streaming/static-expr` directory:

```powershell
cd "/home/runner/work/Streaming-and-Compression-Paper/Streaming-and-Compression-Paper/Streaming/static-expr"
python sweep_streaming_advanced.py --help
python sweep_streaming_advanced.py
```

Use `--help` as the source of truth for the currently supported static CLI options in your branch.
