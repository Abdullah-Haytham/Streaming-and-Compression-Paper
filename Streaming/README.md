# Streaming Experiments

This folder organizes streaming inference evaluation for the Muaalem/Quran phoneme pipeline into two experiment tracks:

- `adaptive-expr/` — adaptive streaming sweep documentation and usage (`sweep_adaptive_dataset.py`).
- `static-expr/` — static/fixed-window streaming sweep documentation and usage (`sweep_streaming_advanced.py`).

## Why two tracks?

- **adaptive-expr** focuses on adaptive chunk sizing/expansion and adaptive selection policy.
- **static-expr** focuses on fixed-window or non-adaptive streaming baselines (and KV-cache-style static variants where supported).

Together they provide comparable quality/latency experiments across adaptive and non-adaptive operating modes.

## Read next

- Adaptive docs: `Streaming/adaptive-expr/README.md`
- Static docs: `Streaming/static-expr/README.md`
