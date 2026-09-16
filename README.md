# Streaming and Compression Paper

This repository contains two main work areas:

- `Streaming/`: streaming inference experiments and evaluation docs.
- `Compression/`: model compression pipeline (`Compression/README.md`).

## Streaming structure

The streaming area is organized into two experiment tracks:

- `Streaming/adaptive-expr/` → adaptive workflow docs and entry point guidance.
- `Streaming/static-expr/` → static workflow docs and entry point guidance.

A general streaming overview is available at:

- `Streaming/README.md`

## Goals

- Evaluate Muaalem/Quran phoneme inference under adaptive and static streaming setups.
- Compare quality, latency, and failure behavior between the two approaches.
- Keep each experiment track documented independently while preserving a shared high-level overview.
