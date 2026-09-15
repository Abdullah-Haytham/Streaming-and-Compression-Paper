# mualem_pipeline_v4

End-to-end compression pipeline for the **muaalem** multi-level CTC speech model
(`obadx/muaalem-model-v3_2`, a Wav2Vec2-BERT backbone with one CTC head per
phonetic *sifa*). The pipeline turns a ~600 M parameter teacher into a
fine-tuned INT8 student through three sequential stages:

```
   prune  →  finetune  →  qat (+ INT8 export)
```

Each stage is checkpointed and skip-on-resume, optionally guided by Knowledge
Distillation, and optionally tracked in Weights & Biases (with full resumable
checkpoint upload).

---

## 1. What the package does

| Stage      | Module                  | Purpose                                                                                                    |
| ---------- | ----------------------- | ---------------------------------------------------------------------------------------------------------- |
| `prune`    | `pruning/`              | Structured pruning of attention heads, FFN neurons, encoder depth, and hidden size                          |
| `finetune` | `finetune.py`           | Multi-level CTC fine-tuning to recover accuracy lost during pruning (optionally KD-blended)                |
| `qat`      | `qat/`                  | Quantization-Aware Training of every (non-skip) `nn.Linear`, then export to real INT8 weights              |

Heavy dependencies (`torch`, `transformers`, `wandb`, `datasets`,
`audiomentations`) are imported **inside** the functions that need them, so
`python -m mualem_pipeline_v4 --help` stays fast and a partial environment can
still run a subset of the stages.

---

## 2. Installation & dependencies

```bash
pip install torch transformers datasets safetensors huggingface_hub \
            soundfile audiomentations wandb pynvml numpy
```

`audiomentations`, `wandb`, and `pynvml` are soft dependencies — the pipeline
auto-disables augmentation, W&B logging, and GPU-utilisation telemetry when
they are missing.

Host detection (`runtime.py`) picks a root working directory automatically:

| Host              | `ROOT`                                                |
| ----------------- | ----------------------------------------------------- |
| Google Colab      | `/content/drive/MyDrive/mualem_pipeline`              |
| Kaggle            | `/kaggle/working/mualem_pipeline`                     |
| Lightning AI      | `/teamspace/studios/this_studio/mualem_pipeline`      |
| Local / anywhere  | `./mualem_pipeline`                                   |

Each experiment lives under `ROOT/<--name>/`.

---

## 3. Quick start

```bash
# default end-to-end run (prune → finetune → qat)
python -m mualem_pipeline_v4 --name myrun

# pruning only
python -m mualem_pipeline_v4 --name myrun --stages prune

# tiny student: 12 heads, 3072 FFN, 18 layers, 768 hidden
python -m mualem_pipeline_v4 --name tiny \
    --head_target 12 --ffn_target 3072 \
    --layer_target 18 --hidden_target 768

# enable KD-guided pruning and KD-blended training, log to W&B
python -m mualem_pipeline_v4 --name kdrun \
    --kd_pruning --kd_alpha 0.4 \
    --wandb --wandb_project mualem-pipeline
```

Stage outputs (per experiment `--name`):

```
ROOT/<name>/
├── experiment_config.json
├── pruned_model/           # written by prune
│   ├── config.json
│   ├── model.safetensors
│   ├── vocab.json, preprocessor_config.json, …
│   └── pruning_summary.json
├── teacher_model/          # cached on the prune-stage download (used by KD)
├── finetune_checkpoints/   # written by finetune
│   ├── model_latest.safetensors, model_best.safetensors
│   ├── optimizer_latest.pt, scaler_latest.pt
│   ├── training_state.json
│   └── config.json, vocab.json, preprocessor_config.json   (side-cars)
├── qat_checkpoints/        # same layout as finetune_checkpoints
├── quantized_model/        # written by qat
│   ├── model_quantized.safetensors  ← INT8 export
│   └── config.json, vocab.json, preprocessor_config.json
└── .prune_done / .finetune_done / .qat_done   (skip-on-resume markers)
```

---

## 4. CLI reference

All flags are declared in `config.py`. `make_config()` returns a plain `dict`
that the stage code reads (and freely augments with derived paths).

### General

| Flag       | Default                  | Meaning                                                                |
| ---------- | ------------------------ | ---------------------------------------------------------------------- |
| `--name`   | `default`                | Experiment subdirectory under `ROOT`                                   |
| `--stages` | `prune finetune qat`     | Subset of stages to run. Already-done stages are skipped automatically |

### Pruning targets

| Flag                | Default                  | Meaning                                                                                                     |
| ------------------- | ------------------------ | ----------------------------------------------------------------------------------------------------------- |
| `--hf_repo`         | `obadx/muaalem-model-v3_2` | HuggingFace source model                                                                                  |
| `--head_target`     | `12`                     | Attention heads after pruning (originally 16)                                                               |
| `--ffn_target`      | `3072`                   | FFN intermediate width after pruning (originally 4096)                                                      |
| `--layer_target`    | `24`                     | Encoder layers after pruning. Default = no change. Min 6, max 24. Recommended floor 18                      |
| `--hidden_target`   | `1024`                   | Hidden size after pruning. Default = no change. Must be a multiple of `--head_target` *and* 64. Floor 256   |
| `--layer_score`     | `cosine`                 | Layer importance scoring. `cosine` (single forward, default) or `loss_delta` (N× slower, **not** implemented) |

### Knowledge Distillation

| Flag                  | Default | Meaning                                                                                                                  |
| --------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------ |
| `--kd_alpha`          | `0.0`   | Blend weight on `L_distill` in the training loss. `0` disables KD entirely. Recommended `0.3-0.5` when enabled            |
| `--kd_pruning`        | off     | Use KD-guided Taylor importance for head + FFN scoring during the prune stage                                            |
| `--kd_calib_batches`  | `30`    | Calibration samples used by KD pruning scoring and cosine layer scoring                                                  |
| `--no_grad_ckpt`      | off     | Disable encoder gradient checkpointing during FT/QAT. Faster but uses more VRAM; safe ≥ 40 GB                            |

### Fine-tuning

| Flag           | Default | Meaning                            |
| -------------- | ------- | ---------------------------------- |
| `--ft_epochs`  | `10`    | Number of FT epochs                |
| `--ft_lr`      | `1e-5`  | Base learning rate                 |
| `--ft_batch`   | `1`     | Micro-batch size                   |
| `--ft_accum`   | `16`    | Gradient-accumulation steps        |
| `--ft_samples` | `5000`  | Streamed samples per FT epoch      |
| `--ft_max_dur` | `15.0`  | Drop clips longer than N seconds   |

### QAT

| Flag            | Default | Meaning                          |
| --------------- | ------- | -------------------------------- |
| `--qat_epochs`  | `5`     | Number of QAT epochs             |
| `--qat_lr`      | `5e-6`  | Base learning rate               |
| `--qat_bits`    | `8`     | Quantisation bit-width           |
| `--qat_samples` | `5000`  | Streamed samples per QAT epoch   |
| `--qat_max_dur` | `10.0`  | Drop clips longer than N seconds |

### Learning-rate schedule (shared by FT and QAT)

Stepped once per epoch (the per-epoch step count isn't known up front because
the dataset is streamed). Implemented as `LambdaLR` so no scheduler state needs
persisting on resume — the lambda is deterministically replayed up to
`last_completed_epoch`.

| Flag                  | Default  | Meaning                                                                                       |
| --------------------- | -------- | --------------------------------------------------------------------------------------------- |
| `--lr_schedule`       | `cosine` | `constant`, `cosine`, or `linear`. Cosine/linear decay from the base lr to `--lr_min`         |
| `--lr_warmup_epochs`  | `1`      | Linear warmup epochs from 0 → base lr before the main schedule. `0` disables                  |
| `--lr_min`            | `1e-7`   | Floor learning rate at the final epoch (ignored for `constant`)                               |

### Dataset

| Flag        | Default              | Meaning                                                                                            |
| ----------- | -------------------- | -------------------------------------------------------------------------------------------------- |
| `--moshaf`  | all 27 subsets       | One or more `obadx/muaalem-annotated-v3` subsets to interleave (uniform probabilities, seed 42)    |

### Weights & Biases

| Flag                  | Default            | Meaning                                                                                                                                                                                  |
| --------------------- | ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--wandb`             | off                | Enable W&B tracking. Reads `WANDB_API_KEY` env or cached creds                                                                                                                           |
| `--wandb_project`     | `mualem-pipeline`  |                                                                                                                                                                                          |
| `--wandb_entity`      | `""`               | Team/user. Empty → personal default                                                                                                                                                      |
| `--wandb_run_name`    | `""`               | Override; default `"{name}-{stage}"`                                                                                                                                                     |
| `--wandb_group`       | `""`               | Override; default = `--name`                                                                                                                                                             |
| `--wandb_ckpt`        | `best`             | `none` / `best` / `all`. Controls upload + restore of the **full** resumable checkpoint (weights + optimizer + AMP scaler + training state + config/vocab side-cars). Restore is automatic on resume when no local checkpoint exists |
| `--wandb_log_every`   | `1`                | Log per-step metrics every N optimizer steps                                                                                                                                             |

---

## 5. Pipeline orchestration (`cli.py`)

1. `make_config(args)` builds the config dict, validates pruning shape constraints, derives experiment paths, and freezes the per-level loss-weight table.
2. The experiment directory is created and `experiment_config.json` is dumped (Paths stringified, level/loss tables stripped).
3. For each requested stage, in fixed order `prune → finetune → qat`:
   - If `.{stage}_done` already exists, the stage is skipped.
   - Otherwise the stage module is imported lazily (so e.g. `--stages prune` doesn't import `audiomentations`) and run.
4. Final disk check.

The `is_done` / `mark_done` pattern lives in `runtime.py` and lets the same
command be re-run safely on a restored experiment directory.

---

## 6. Stage details

### 6.1 Pruning (`pruning/`)

`pruning/pipeline.py:run_pruning` orchestrates these substeps:

1. Download the upstream HF checkpoint into a temp directory.
2. Copy the same files into `teacher_model/` if KD is enabled — saves a second download.
3. Load weights into a **numpy** state-dict (slicing is done in-place without torch to keep memory low).
4. Compute KD-guided Taylor importance scores (`kd_scores.compute_kd_pruning_scores`) when `--kd_pruning` is set. Otherwise log a magnitude-baseline head-score histogram to W&B.
5. **Head pruning** (`heads_ffn.prune_heads`): for each layer, drop the lowest-scoring `(16 - head_target)` heads. Scorer is KD Taylor when supplied, else `‖Q_h‖ + ‖K_h‖ + ‖V_h‖`. Slices `linear_q/k/v` rows and `linear_out` columns.
6. **FFN pruning** (`heads_ffn.prune_ffn`): same logic on the intermediate dimension. Handles the multiple naming conventions for the up/down linears (`intermediate_dense` / `feed_forward.intermediate` / `fc1`) and falls back to a shape-based finder.
7. **Layer pruning** (`layers.py`, only when `--layer_target < 24`): score every encoder layer by `mean cosine(input, output)` over a calibration set (high cosine = layer is nearly identity), drop the highest-scoring layers, and re-index the survivors to a contiguous `0..N-1`.
8. **Hidden-size pruning** (`hidden.py`, only when `--hidden_target < 1024`): score every hidden dimension by activation L2-norm on the calibration set (fallback: weight-norm). Slice every "hidden-tied" tensor across encoder + adapter. Validate post-slice shapes; raise on any mismatch.
9. Write the new `model.safetensors` + a `config.json` with updated `num_attention_heads / intermediate_size / num_hidden_layers / hidden_size` (and `output_hidden_size`, if it was tied to `hidden_size` originally — required so the adapter's freshly-built LayerNorm modules expect the right dim).
10. Copy tokenizer/vocab side-cars across, write `pruning_summary.json`, log W&B histograms + summary, delete the temp download, mark the stage done.

Important subtleties baked into the code:

- **Hidden-size tensor classification** (`hidden.classify_hidden_axis`) handles four families:
  - *row* (LayerNorm, output linears, depthwise convs, masked_spec_embed, …)
  - *col* (Q/K/V inputs, intermediate_dense, CTC heads, level_to_lm_head)
  - *doubled* (`pointwise_conv1`, adapter `residual_conv`, `self_attn_conv`) where the leading dim is `2*hidden` because a GLU halves it back
  - *both* (`pointwise_conv2`, square adapter attention linears whose `heads*head_dim == hidden_size`)
  Plus a hard skip list for `feature_extractor`, `pos_bias_*`, `logit_scale`.
- A pre-slice tensor-shape **audit** prints the shapes of critical tensors so silent miscompiles are loud.
- A post-slice **catch-all** scans for any non-skip tensor still containing the old hidden dim, surfacing missed tags.

### 6.2 Fine-tuning (`finetune.py` + `training.py`)

- Loads pruned model via `model.build_model`, which is the bridge that lets pruned shapes load cleanly: it rebinds `_parameters` and `_buffers` directly (so `state_dict` shape mismatch can't refuse the load) and patches `self_attn.num_heads / head_dim / head_size` from the live `linear_q` shape on every encoder + adapter attention module.
- The `feature_extractor` and `feature_projection` modules are frozen — the audio front-end is already well-trained and changes there would dwarf recovery signal.
- Encoder gradient-checkpointing is on by default (`--no_grad_ckpt` to disable).
- Optimizer: `AdamW(lr=ft_lr, weight_decay=0.01)`, mixed precision (`GradScaler`) when CUDA is available.
- Optional **KD**: if `--kd_alpha > 0`, the frozen teacher is loaded (`kd.load_teacher`, fp16 on GPU). For every batch, the teacher's hidden states are moved to CPU immediately after the no-grad forward so the GPU never holds two full activation graphs at once. `kd.compute_kd_loss` then pulls one teacher layer at a time back to the GPU.
- `training.train_one_epoch` is the loop body shared by FT and QAT:
  - Multi-level CTC loss with the per-level weights from `config.LOSS_WEIGHTS`.
  - KD-blended total `loss = L_CTC + α · L_distill` (eq. 5 of Li et al. *2510.04213*: `mean_l [ L1(h_t^l, h_s^l) - cos(h_t^l, h_s^l) ]`).
  - Per-step W&B metrics: loss totals, per-level losses, KD components (teacher fwd time/mem, per-layer cosine similarity), grad norm, AMP scale, GPU mem / util / throughput.
  - Per `--ft_accum` steps: AMP `unscale` → `clip_grad_norm_(1.0)` → `step` → `update`.
- LR scheduler is built **after** `load_checkpoint` so the `LambdaLR` fast-forwards to the resumed epoch (it seeds `initial_lr` per param group from the live `lr` for `last_epoch > -1`).
- Per-epoch checkpoint save (`checkpoint.save_checkpoint`): writes `model_latest.safetensors` (+ `model_best.safetensors` on improvement), optimizer + scaler state, and `training_state.json`. The static side-cars (`config.json`, `vocab.json`, `preprocessor_config.json`) are copied into the checkpoint directory so each W&B checkpoint artifact is fully self-contained — a clean machine with `wandb` access can resume from artifacts alone.
- W&B checkpoint policy: `--wandb_ckpt all` uploads every epoch; `best` uploads only on improvement; `none` disables. Plus a separate `<name>-ft-best` artifact carrying just `model_best.safetensors` for downstream consumers.

### 6.3 QAT (`qat/`)

`qat/pipeline.py:run_qat`:

1. Loads the best (or latest if no best exists) FT weights and `config.json` from `finetune_checkpoints/`.
2. Freezes `feature_extractor` and `feature_projection` again.
3. **`replace_with_qat`** walks the module tree and swaps every `nn.Linear` for a `QATLinear` (`qat/modules.py`), **except** those whose full path starts with `wav2vec2_bert.feature_extractor`, `wav2vec2_bert.feature_projection`, or `wav2vec2_bert.adapter` — those are quantisation-sensitive enough that INT8 hurts more than it helps.
4. Optional KD teacher (same as FT).
5. AdamW + LambdaLR + AMP + per-epoch checkpointing using the **same** `train_one_epoch` loop. Each epoch also logs observer min/max means and histograms of weight/activation ranges to W&B.
6. After training, `convert_to_int8` replaces every `QATLinear` with a `QuantizedLinear` (real INT8 weights + scale/zp/qmin/qmax buffers), `eval()`s the model, and saves to `quantized_model/model_quantized.safetensors`.

`QATLinear` forward (per call):
1. Update activation observer EMA-min/max on the live input.
2. Read scale/zp/qmin/qmax for both activation and weight.
3. Fake-quantize input and weight in **fp32** with `torch.cuda.amp.autocast(enabled=False)`. Round-to-nearest + clamp via `FakeQuantizeSTE`, whose backward is straight-through for in-range values and zero for saturated ones.
4. Update the weight observer with the *detached* weight (so the observer never enters the autograd graph).
5. `F.linear(xf, wf, b).to(x.dtype)`.

`MinMaxObserver` is EMA-based with `ema_decay=0.999`. Both symmetric (default) and asymmetric quantisation are supported; the pipeline uses symmetric.

---

## 7. Knowledge Distillation (`kd.py`)

Two helpers, used by both the pruning-stage scorer and the FT/QAT training loop:

- **`load_teacher(cfg, device)`**: loads the cached teacher from `teacher_model/`, downloads it on a cache miss. Cast to fp16 on CUDA (halves VRAM), parameters frozen, `eval()`.
- **`compute_kd_loss(teacher_hidden, student_hidden, device, return_per_layer=False)`**: eq. 5 of *2510.04213*. Teacher activations may live on CPU; each layer is moved to GPU for its term and dropped before the next, so the GPU only ever holds **one** teacher layer's activations at a time. Layers where teacher and student hidden sizes differ (i.e. `--hidden_target < 1024`) are silently skipped from the average — when **every** layer is mismatched the loss is zero and the caller should consider `--kd_alpha 0`. Setting `return_per_layer=True` returns a dict with `cos_per_layer` for W&B logging.

`pruning/kd_scores.compute_kd_pruning_scores` is a careful two-phase pipeline that fits in ~15 GB of VRAM:

- **Phase 1** — teacher (fp16, gradient checkpointing, no grad): forward each calibration sample, move every layer's hidden state to CPU, then **unload the teacher entirely** and free CUDA memory.
- **Phase 2** — student (fp16, gradient checkpointing, grad enabled): for each sample, move its stored teacher hiddens back to GPU, compute `L_distill`, `backward()` to accumulate gradients, then drop the tensors. After all samples, extract Taylor scores `|w · dL/dw|` per attention head (averaged over head_dim and input cols) and per FFN neuron (averaged over input cols).

The returned dicts are keyed by encoder layer prefix (`"wav2vec2_bert.encoder.layers.N"`) and consumed by `prune_heads` / `prune_ffn`.

---

## 8. Checkpoints & resume (`checkpoint.py`)

A "checkpoint" is the set of files in `<stage>_checkpoints/`:

```
model_latest.safetensors        (always)
model_best.safetensors          (only when an epoch was best so far)
optimizer_latest.pt
scaler_latest.pt                (only on CUDA + AMP)
training_state.json             ({completed_epoch, training_log})
config.json, vocab.json, preprocessor_config.json   (side-cars for clean-machine resume)
```

`save_checkpoint` writes all of the above. `load_checkpoint`:

1. If `training_state.json` is missing **and** a `wandb_logger` was passed, attempt `wandb_logger.restore_checkpoint(ckpt_dir)` — downloads the `<name>-<stage>-ckpt:latest` artifact.
2. If still missing, return `(0, [])` — start from scratch.
3. Otherwise load weights via the same `_parameters` / `_buffers` rebinding trick used in `model.build_model` (tolerates pruned shapes), restore optimizer state, attempt to restore GradScaler state.
4. Local always wins — an interrupted run on the same machine resumes from disk and never touches W&B.

LR scheduler state is **not** persisted because the `LambdaLR` lambda is deterministic given `last_completed_epoch`.

---

## 9. Module-by-module reference

| File                              | Lines | Summary                                                                                          |
| --------------------------------- | ----- | ------------------------------------------------------------------------------------------------ |
| `__init__.py` / `__main__.py`     |  10   | Package docstring; `python -m mualem_pipeline_v4 …` entry                                       |
| `cli.py`                          |  60   | Stage orchestrator; lazy stage imports; experiment_config.json; skip-on-done                     |
| `config.py`                       | 194   | Argparse, validation, `LOSS_WEIGHTS`, `ALL_MOSHAFS`, derived path resolution                     |
| `runtime.py`                      |  75   | Host detection (`ROOT`), `disk_check`, `mark_done`/`is_done`, `gpu_mem_mb`, `gpu_util_pct`       |
| `data.py`                         |  74   | `decode_audio`, `encode_labels`, `stream_samples` (interleaved streaming), `build_augmentation`  |
| `model.py`                        |  99   | `build_model` — shape-tolerant `Wav2Vec2BertForMultilevelCTC` loader. Patches `num_heads/head_dim/head_size` on every attention module from live linear shapes |
| `checkpoint.py`                   | 105   | Resumable checkpoint save/load with W&B-restore fallback                                         |
| `lr_schedule.py`                  |  69   | Per-epoch `LambdaLR` (cosine/linear/constant + warmup + floor)                                   |
| `kd.py`                           | 111   | Teacher loader + `compute_kd_loss` (eq. 5 of 2510.04213)                                          |
| `training.py`                     | 247   | Shared FT/QAT epoch loop; per-step W&B metrics; AMP + grad-accum + KD blending                   |
| `finetune.py`                     | 205   | Stage 2 driver                                                                                   |
| `wandb_logger.py`                 | 251   | Lazy W&B wrapper. Host detection for tags, log_step/epoch/histogram/summary, full-dir checkpoint artifact upload/restore |
| `pruning/__init__.py`             |   3   | Re-exports `run_pruning`                                                                          |
| `pruning/pipeline.py`             | 249   | Stage 1 driver                                                                                   |
| `pruning/_state.py`               |  67   | Shared helpers: layer prefix enumeration, calibration artifacts writer                           |
| `pruning/heads_ffn.py`            | 146   | Attention head + FFN neuron slicing (magnitude or KD Taylor)                                     |
| `pruning/layers.py`               | 165   | Layer scoring (mean cosine input/output) + drop + re-index                                       |
| `pruning/hidden.py`               | 395   | Hidden-size pruning: tag tables, axis classifier, activation-L2 scorer (weight-norm fallback), slicer, post-slice validator |
| `pruning/kd_scores.py`            | 175   | Two-phase KD-guided Taylor scoring for heads + FFN neurons                                       |
| `qat/__init__.py`                 |   3   | Re-exports `run_qat`                                                                              |
| `qat/pipeline.py`                 | 243   | Stage 3 driver + INT8 export                                                                     |
| `qat/modules.py`                  | 159   | `FakeQuantizeSTE`, `MinMaxObserver`, `QATLinear`, `QuantizedLinear`, replace/convert helpers     |

---

## 10. Levels & loss weights

The multi-level CTC head trains 11 parallel CTC sequences. Per-level weights
(thesis Eq 5.1) are frozen in `config.LOSS_WEIGHTS`:

| Level                  | Weight    |
| ---------------------- | --------- |
| `phonemes`             | 0.4       |
| `ghonna`               | 0.059875  |
| `hams_or_jahr`         | 0.059875  |
| `istitala`             | 0.059875  |
| `itbaq`                | 0.059875  |
| `qalqla`               | 0.059875  |
| `safeer`               | 0.059875  |
| `shidda_or_rakhawa`    | 0.0605    |
| `tafashie`             | 0.059875  |
| `tafkheem_or_taqeeq`   | 0.0605    |
| `tikraar`              | 0.059875  |

The dict order also defines `cfg["levels"]`. `encode_labels` (`data.py`) reads
`sample["phonemes"]` and `sample["sifat"]` from the HF dataset and maps each
character/sifa into the corresponding vocab id.

---

## 11. Worked configurations

### Recover the default teacher exactly (no real pruning, just FT + QAT)

```bash
python -m mualem_pipeline_v4 --name baseline \
    --head_target 16 --ffn_target 4096 \
    --layer_target 24 --hidden_target 1024
```

### Heavy pruning + KD recovery

```bash
python -m mualem_pipeline_v4 --name kd-tiny \
    --head_target 12 --ffn_target 3072 \
    --layer_target 18 --hidden_target 768 \
    --kd_pruning --kd_alpha 0.5 --kd_calib_batches 60 \
    --ft_epochs 15 --ft_lr 2e-5 \
    --qat_epochs 5 --qat_lr 5e-6 \
    --wandb --wandb_ckpt best
```

### Resume from W&B on a fresh machine

```bash
# same --name + --wandb + --wandb_ckpt as the original run is sufficient.
# load_checkpoint() will pull <name>-ft-ckpt:latest / <name>-qat-ckpt:latest
# automatically when nothing is on local disk.
python -m mualem_pipeline_v4 --name kd-tiny --wandb --wandb_ckpt best
```

---

## 12. Implementation notes worth knowing

- **Streaming data**: the HF dataset is streamed (`load_dataset(..., streaming=True)`) and `take(N)` is bounded per epoch by `--ft_samples` / `--qat_samples`. Multiple moshafs are interleaved with equal probability, seed 42, `stopping_strategy="all_exhausted"`.
- **The "stages are dict-driven" pattern**: every stage takes the same `cfg: dict` and reads its keys directly. No dataclasses, no defensive checks, no shared "Stage" base class — the orchestrator hands `cfg` over, stages mutate it freely.
- **Adapter and `output_hidden_size`**: the Wav2Vec2-BERT adapter builds its modules from `output_hidden_size`, not `hidden_size`. The pruning stage propagates `--hidden_target` into `output_hidden_size` only when the original config tied them — otherwise the adapter would build LayerNorms at the wrong dim and FT would die on `normalized_shape` errors.
- **State-dict naming remap**: the upstream checkpoint stores CTC heads under `level_to_lm_head.*`; `model.build_model` remaps these to `ctc_heads.*` on load. The hidden-size pruner therefore lists *both* names in `_HIDDEN_COL_TAGS`.
- **fp16 for KD scoring**: the student in `compute_kd_pruning_scores` runs in fp16 with grad enabled. Taylor importance `|w · dL/dw|` is robust to fp16 noise — gradient *direction* is what matters.
- **AMP scaler persistence**: the GradScaler is saved next to the optimizer, but only on CUDA runs. A CPU-only resume drops any stale scaler file so it doesn't confuse the loader.
