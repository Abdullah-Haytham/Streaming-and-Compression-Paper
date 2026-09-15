# mualem_pipeline_v4

End-to-end compression pipeline for the **Muaalem** multi-level CTC speech model.

The pipeline compresses the ~600M parameter teacher model through:

```text
prune → finetune → QAT → INT8 export
```

Each stage supports checkpointing and automatic resume.

## Installation

```bash
pip install torch transformers datasets safetensors huggingface_hub \
            soundfile audiomentations wandb pynvml numpy
```

## Quick Start

Run the full pipeline:

```bash
python -m mualem_pipeline_v4 --name myrun
```

Run only pruning:

```bash
python -m mualem_pipeline_v4 --name myrun --stages prune
```

Create a smaller **width-pruned** model:

```bash
python -m mualem_pipeline_v4 --name tiny \
    --head_target 12 \
    --ffn_target 3072 \
    --hidden_target 768
```

## Pipeline

### 1. Pruning

Structurally reduces the model by pruning:

* Attention heads
* FFN neurons
* Encoder layers
* Hidden dimensions

### 2. Fine-tuning

Fine-tunes the pruned model using multi-level CTC loss to recover accuracy lost during pruning.

### 3. QAT

Applies Quantization-Aware Training to supported linear layers and exports the final model with real **INT8 weights**.

## Knowledge Distillation

Knowledge Distillation (KD) is supported for pruning and fine-tuning, but **works best with width-only pruning**.

KD is **not recommended when pruning encoder depth** or combining width and depth pruning, as the teacher and student representations become harder to align.

For width-only pruning, KD can be enabled with:

```bash
python -m mualem_pipeline_v4 --name kdrun \
    --kd_pruning \
    --kd_alpha 0.4
```

## Outputs

Each experiment is stored under:

```text
ROOT/<name>/
├── pruned_model/
├── finetune_checkpoints/
├── qat_checkpoints/
└── quantized_model/
    └── model_quantized.safetensors
```

Completed stages are marked automatically, so interrupted experiments can be resumed by running the same command again.

## Useful Options

```text
--name              Experiment name
--stages             prune / finetune / qat
--head_target        Target attention heads
--ffn_target         Target FFN size
--layer_target       Target encoder layers
--hidden_target      Target hidden size
--kd_pruning         Enable KD-guided pruning
--kd_alpha           KD loss weight
--wandb              Enable W&B logging
```

See:

```bash
python -m mualem_pipeline_v4 --help
```

for all available options.
