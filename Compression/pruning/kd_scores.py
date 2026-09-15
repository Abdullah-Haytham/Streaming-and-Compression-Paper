"""KD-guided Taylor importance scoring for heads + FFN neurons.

Split into two phases that never put the teacher and student on the GPU at
the same time -- that's what fits this in 15 GB:

  Phase 1: teacher (no-grad, fp16, gradient checkpointing)
    Run every calibration sample through the teacher and copy each layer's
    hidden state to CPU. Unload the teacher entirely and free CUDA memory
    before phase 2.

  Phase 2: student (grad enabled, fp16, gradient checkpointing)
    Load the student in fp16, replay each stored teacher hidden one sample
    at a time, compute L_distill, ``backward()`` to accumulate gradients,
    then extract Taylor scores |w . dL/dw| from the parameters of interest.
"""

import gc


_HEAD_QKV_SUFFIXES = (
    "self_attn.linear_q.weight",
    "self_attn.linear_k.weight",
    "self_attn.linear_v.weight",
)
_FFN_INTERMEDIATE_TAGS = (
    "intermediate_dense.weight",
    "feed_forward.intermediate",
    "fc1.weight",
)
_FFN_LAYER_SEPARATORS = (".feed_forward.", ".ffn.")


def _classify_kd_param(name: str):
    """Return ``(kind, layer_prefix)`` for a parameter relevant to KD pruning.

    - ``("head", layer_prefix)`` for a Q/K/V weight matrix
    - ``("ffn",  layer_prefix)`` for an FFN intermediate weight matrix
    - ``(None, None)`` for anything else (bias tensors, output projections,
       LayerNorms, CTC heads, ...).

    ``layer_prefix`` matches ``layer_prefixes_from_state`` ordering, so callers
    can use it as a key into ``head_scores`` / ``ffn_scores``.
    """
    for qkv_suffix in _HEAD_QKV_SUFFIXES:
        if qkv_suffix in name:
            return "head", name.split(".self_attn.")[0]

    is_ffn_intermediate = (
        name.endswith(".weight")
        and any(tag in name for tag in _FFN_INTERMEDIATE_TAGS)
    )
    if is_ffn_intermediate:
        for separator in _FFN_LAYER_SEPARATORS:
            if separator in name:
                return "ffn", name.split(separator)[0]

    return None, None


def compute_kd_pruning_scores(cfg: dict, tmp_dir):
    """Return ``(head_scores, ffn_scores)``, both dicts keyed by layer prefix.

    Either or both may be ``None`` when calibration is impossible (missing
    dependency, zero usable samples).

    head_scores[layer_prefix] : np.ndarray of shape (16,)
    ffn_scores[layer_prefix]  : np.ndarray of shape (orig_ffn,)
    """
    print("\ncomputing KD-guided pruning importance (two-phase, memory-safe)")
    try:
        import torch
        from transformers import SeamlessM4TFeatureExtractor
    except ImportError as e:
        print(f"  missing dependency: {e} -- skipping KD pruning scoring")
        return None, None

    from ..data import decode_audio, stream_samples
    from ..kd import compute_kd_loss
    from ..model import build_model
    from ..runtime import disk_check

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}")
    HEAD_DIM = 64

    fe = SeamlessM4TFeatureExtractor.from_pretrained(str(tmp_dir))

    # Materialise audio inputs on CPU once so both phases see the same samples
    # without re-streaming (the dataset stream isn't rewindable).
    calib_stream = stream_samples(cfg["moshaf"], cfg["kd_calib_batches"] * 3)
    raw_inputs = []
    skipped = 0
    print(f"  collecting up to {cfg['kd_calib_batches']} calibration samples")
    for sample in calib_stream:
        if len(raw_inputs) >= cfg["kd_calib_batches"]:
            break
        try:
            audio, sr = decode_audio(sample["audio"])
        except Exception:
            skipped += 1
            continue
        feats = fe([audio], sampling_rate=16000, return_tensors="pt", padding=True)
        raw_inputs.append(feats["input_features"].cpu())
        del feats

    n_samples = len(raw_inputs)
    print(f"  collected {n_samples} samples ({skipped} skipped)")
    if n_samples == 0:
        return None, None

    print(f"\n  phase 1/2: teacher forward")
    disk_check("before loading teacher (KD scoring)")
    teacher, _ = build_model(tmp_dir / "config.json",
                              tmp_dir / "model.safetensors", device)
    if device.type == "cuda":
        teacher = teacher.half()
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()
    teacher.wav2vec2_bert.encoder.gradient_checkpointing = True
    t_size = sum(p.numel() for p in teacher.parameters()) / 1e6
    print(f"  teacher loaded ({t_size:.0f}M params, fp16)")

    teacher_hiddens = []
    with torch.no_grad():
        for i, inp_cpu in enumerate(raw_inputs):
            inp = inp_cpu.to(device)
            if device.type == "cuda":
                inp = inp.half()
            t_out = teacher.wav2vec2_bert(inp, output_hidden_states=True)
            cpu_hidden = tuple(h.detach().cpu().half() for h in t_out.hidden_states)
            teacher_hiddens.append(cpu_hidden)
            del inp, t_out
            if (i + 1) % 10 == 0:
                print(f"    teacher: {i+1}/{n_samples}", flush=True)

    del teacher
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        free_gb = (torch.cuda.get_device_properties(0).total_memory
                   - torch.cuda.memory_allocated()) / 1e9
        print(f"  teacher unloaded; GPU free: {free_gb:.1f} GB")

    print(f"\n  phase 2/2: student gradient accumulation")
    student, _ = build_model(tmp_dir / "config.json",
                              tmp_dir / "model.safetensors", device)
    if device.type == "cuda":
        # fp16 student halves VRAM; gradient direction is what matters here.
        student = student.half()
    student.wav2vec2_bert.encoder.gradient_checkpointing = True
    student.train()
    student.zero_grad()
    s_size = sum(p.numel() for p in student.parameters()) / 1e6
    print(f"  student loaded ({s_size:.0f}M params, fp16)")

    for i, (inp_cpu, t_hidden_cpu) in enumerate(zip(raw_inputs, teacher_hiddens)):
        inp = inp_cpu.to(device)
        if device.type == "cuda":
            inp = inp.half()

        # Move this sample's teacher hiddens to GPU only for the KD loss,
        # then drop them immediately after backward.
        t_hidden_gpu = tuple(h.to(device) for h in t_hidden_cpu)

        s_out = student.wav2vec2_bert(inp, output_hidden_states=True)
        s_hidden = s_out.hidden_states

        kd_loss = compute_kd_loss(t_hidden_gpu, s_hidden, device)
        kd_loss.backward()

        del inp, t_hidden_gpu, s_out, s_hidden, kd_loss
        if (i + 1) % 10 == 0:
            print(f"    student: {i+1}/{n_samples}", flush=True)

    print(f"  gradient accumulation done ({n_samples} samples)")

    head_scores = {}
    ffn_scores  = {}

    for name, param in student.named_parameters():
        if param.grad is None:
            continue
        # Taylor importance: |w . dL/dw|. Works identically in fp16.
        importance = (param.data.abs() * param.grad.abs()).detach().cpu().float().numpy()

        kind, layer_prefix = _classify_kd_param(name)
        if kind == "head":
            n_heads = param.shape[0] // HEAD_DIM
            per_head = importance.reshape(n_heads, HEAD_DIM, -1).mean(axis=(1, 2))
            head_scores[layer_prefix] = (
                head_scores.get(layer_prefix, 0.0) + per_head)
        elif kind == "ffn":
            per_neuron = importance.mean(axis=1)
            ffn_scores[layer_prefix] = (
                ffn_scores.get(layer_prefix, 0.0) + per_neuron)

    print(f"  head score layers: {len(head_scores)}, FFN score layers: {len(ffn_scores)}")

    del student, raw_inputs, teacher_hiddens
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return head_scores, ffn_scores
