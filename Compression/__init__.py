"""Mualem pipeline: KD-aware structured pruning, fine-tuning, and QAT.

Heavy deps (torch, transformers, wandb, datasets, audiomentations) are imported
inside the functions that need them so ``--help`` stays fast and partial
environments can still run a subset of stages.
"""
