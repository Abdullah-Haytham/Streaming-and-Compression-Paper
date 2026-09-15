"""ONNX export for Mualem Wav2Vec2-BERT Multilevel CTC.

Supports QAT (INT8 dequantized to FP32) and plain FP32 checkpoints.
Optional post-export dynamic and/or static INT8 quantization via onnxruntime.
"""

import argparse
import gc
import io
import json
import re
import shutil
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import onnx
from safetensors.torch import load_file
from transformers import Wav2Vec2BertConfig, Wav2Vec2BertModel, SeamlessM4TFeatureExtractor

HEAD_DIM = 64


class Wav2Vec2BertForMultilevelCTC(nn.Module):
    def __init__(self, config, vocab_sizes):
        super().__init__()
        self.wav2vec2_bert = Wav2Vec2BertModel(config)
        self.dropout = nn.Dropout(config.final_dropout)
        self.ctc_heads = nn.ModuleDict({
            name: nn.Linear(config.hidden_size, vs, bias=True)
            for name, vs in vocab_sizes.items()
        })

    def forward(self, input_features, attention_mask=None):
        out = self.wav2vec2_bert(input_features=input_features, attention_mask=attention_mask)
        h = self.dropout(out.last_hidden_state)
        return {name: head(h) for name, head in self.ctc_heads.items()}


class OnnxWrapper(nn.Module):
    def __init__(self, model, level_names):
        super().__init__()
        self.model = model
        self.level_names = sorted(level_names)

    def forward(self, input_features):
        logits = self.model(input_features)
        return tuple(logits[name] for name in self.level_names)


def _remap_keys(ckpt):
    return {
        (k.replace("level_to_lm_head.", "ctc_heads.") if k.startswith("level_to_lm_head.") else k): v
        for k, v in ckpt.items()
    }


def _load_qat_checkpoint(model, remapped):
    """Dequantize INT8 linear weights and load all remaining parameters."""
    rebuilt_deq, rebuilt_fp, matched, skipped_lin = 0, 0, 0, 0
    for module_path, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue

        w_key       = f"{module_path}.weight"
        w_int8_key  = f"{module_path}.weight_int8"
        w_scale_key = f"{module_path}.weight_scale"
        w_zp_key    = f"{module_path}.weight_zp"
        b_key       = f"{module_path}.bias"

        if w_key in remapped:
            weight_fp = remapped[w_key]
            source = "fp32"
        elif w_int8_key in remapped:
            w_int8  = remapped[w_int8_key]
            w_scale = remapped[w_scale_key]
            w_zp    = remapped[w_zp_key]
            weight_fp = (w_int8.float() - w_zp) * w_scale
            source = "int8"
        else:
            skipped_lin += 1
            continue

        bias_fp = remapped.get(b_key)
        new_linear = nn.Linear(weight_fp.shape[1], weight_fp.shape[0], bias=(bias_fp is not None))
        with torch.no_grad():
            new_linear.weight.copy_(weight_fp)
            if bias_fp is not None:
                new_linear.bias.copy_(bias_fp)

        parts = module_path.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], new_linear)

        if source == "int8":
            rebuilt_deq += 1
        elif module.weight.shape != weight_fp.shape:
            rebuilt_fp += 1
        else:
            matched += 1

    print(f"Phase 1: dequantized={rebuilt_deq}, rebuilt={rebuilt_fp}, matched={matched}, skipped={skipped_lin}")

    linear_handled_prefixes = {
        path for path, mod in model.named_modules() if isinstance(mod, nn.Linear)
    }

    skip_suffixes = (".weight", ".bias", ".weight_int8", ".weight_scale",
                     ".weight_zp", ".act_scale", ".act_zp", ".act_qmin", ".act_qmax")
    loaded, skipped_p = 0, 0
    for name, tensor in remapped.items():
        prefix = name
        for sfx in skip_suffixes:
            if name.endswith(sfx):
                prefix = name[:-len(sfx)]
                break
        if prefix in linear_handled_prefixes:
            continue

        parts = name.split(".")
        obj = model
        try:
            for p in parts[:-1]:
                obj = getattr(obj, p)
        except AttributeError:
            skipped_p += 1
            continue
        attr = parts[-1]

        if attr in obj._parameters and obj._parameters[attr] is not None:
            if obj._parameters[attr].shape == tensor.shape:
                with torch.no_grad():
                    obj._parameters[attr].data.copy_(tensor)
                loaded += 1
            else:
                print(f"  shape mismatch: {name} model={list(obj._parameters[attr].shape)} ckpt={list(tensor.shape)}")
                skipped_p += 1
        elif attr in obj._buffers:
            if obj._buffers[attr] is not None and obj._buffers[attr].shape == tensor.shape:
                with torch.no_grad():
                    obj._buffers[attr].copy_(tensor)
                loaded += 1
            else:
                skipped_p += 1
        else:
            skipped_p += 1

    print(f"Phase 2: loaded={loaded}, skipped={skipped_p}")


def _patch_attention_heads(model):
    """Fix num_heads on each encoder layer to match the pruned weight shapes."""
    for layer in model.wav2vec2_bert.encoder.layers:
        a = layer.self_attn
        actual_heads = a.linear_q.weight.shape[0] // HEAD_DIM
        a.num_heads = actual_heads
        a.head_dim  = HEAD_DIM
        if hasattr(a, "head_size"):
            a.head_size = HEAD_DIM
    if hasattr(model.wav2vec2_bert, "adapter") and model.wav2vec2_bert.adapter:
        for al in model.wav2vec2_bert.adapter.layers:
            if hasattr(al, "self_attn"):
                al.self_attn.num_heads = al.self_attn.linear_q.weight.shape[0] // HEAD_DIM
                al.self_attn.head_dim = HEAD_DIM


def load_model(model_dir: Path, weights_filename: str, use_qat: bool, device: torch.device):
    """Build and return the model loaded from *model_dir*."""
    with open(model_dir / "config.json", encoding="utf-8") as f:
        cfg = json.load(f)
    level_to_vocab_size = cfg.pop("level_to_vocab_size")
    for k in ("level_to_loss_weight", "architectures", "model_type", "transformers_version"):
        cfg.pop(k, None)

    model = Wav2Vec2BertForMultilevelCTC(Wav2Vec2BertConfig(**cfg), level_to_vocab_size)

    ckpt = load_file(str(model_dir / weights_filename))
    remapped = _remap_keys(ckpt)
    del ckpt

    if use_qat:
        _load_qat_checkpoint(model, remapped)
    else:
        missing, unexpected = model.load_state_dict(remapped, strict=False)
        print(f"Missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
    del remapped
    gc.collect()

    _patch_attention_heads(model)
    model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model loaded ({device}) — {n_params:.1f}M params")
    print(f"CTC heads: {list(level_to_vocab_size.keys())}")

    layer0 = model.wav2vec2_bert.encoder.layers[0]
    print(f"Layer 0: heads={layer0.self_attn.num_heads}, "
          f"q={list(layer0.self_attn.linear_q.weight.shape)}, "
          f"ffn1={list(layer0.ffn1.intermediate_dense.weight.shape)}, "
          f"ffn2={list(layer0.ffn2.intermediate_dense.weight.shape)}")

    return model, level_to_vocab_size, n_params



def export_onnx(wrapper, dummy_input, level_names, export_dir: Path, opset: int):
    """Trace and save the model to ONNX, inlining any external data."""
    onnx_path = export_dir / "mualem_multilevel_ctc.onnx"
    dynamic_axes = {"input_features": {0: "batch_size", 1: "seq_len"}}
    for name in level_names:
        dynamic_axes[name] = {0: "batch_size", 1: "time_steps"}

    t0 = time.time()
    torch.onnx.export(
        wrapper, (dummy_input,), str(onnx_path),
        input_names=["input_features"], output_names=level_names,
        dynamic_axes=dynamic_axes, opset_version=opset,
        do_constant_folding=True, export_params=True,
        dynamo=False,
    )
    print(f"Exported in {time.time()-t0:.1f}s")

    data_file = Path(str(onnx_path) + ".data")
    if data_file.exists():
        from onnx.external_data_helper import load_external_data_for_model
        _m = onnx.load(str(onnx_path), load_external_data=False)
        load_external_data_for_model(_m, str(export_dir))
        onnx.save_model(_m, str(onnx_path), save_as_external_data=False)
        data_file.unlink()
        del _m
        gc.collect()

    print(f"ONNX: {onnx_path.stat().st_size/1e6:.1f} MB")
    return onnx_path



def _make_calib_reader(fe, hf_dataset: str, hf_subset: str, n_samples: int):
    from onnxruntime.quantization import CalibrationDataReader
    from datasets import Audio, load_dataset

    class HFCalibReader(CalibrationDataReader):
        def __init__(self):
            ds = load_dataset(hf_dataset, hf_subset, split="train", streaming=True)
            ds = ds.cast_column("audio", Audio(decode=False))
            self._data = []
            self._idx = 0
            for sample in ds.take(n_samples):
                buf = io.BytesIO(sample["audio"]["bytes"])
                wav, _sr = sf.read(buf, dtype="float32")
                if wav.ndim > 1:
                    wav = wav.mean(axis=1)
                feats = fe([wav], sampling_rate=16000, return_tensors="np", padding=True)
                self._data.append({"input_features": feats["input_features"]})
            print(f"  Loaded {len(self._data)} calibration samples")

        def get_next(self):
            if self._idx >= len(self._data):
                return None
            d = self._data[self._idx]
            self._idx += 1
            return d

    return HFCalibReader()


def quantize_onnx(onnx_path: Path, export_dir: Path, mode: str, fe,
                  hf_dataset: str, hf_subset: str, n_calib: int):
    """Apply dynamic and/or static INT8 quantization to the exported ONNX model."""
    from onnxruntime.quantization import (
        quantize_dynamic, quantize_static, QuantType,
    )

    sz_fp = onnx_path.stat().st_size / 1e6
    q_dyn_path    = export_dir / "mualem_multilevel_ctc_int8_dynamic.onnx"
    q_static_path = export_dir / "mualem_multilevel_ctc_int8_static.onnx"

    if mode in ("dynamic", "both"):
        if q_dyn_path.exists():
            print(f"INT8 dynamic already exists: {q_dyn_path.stat().st_size/1e6:.1f} MB (skipping)")
        else:
            print("Running dynamic quantization...")
            quantize_dynamic(
                str(onnx_path), str(q_dyn_path),
                weight_type=QuantType.QInt8,
                extra_options={"MatMulConstBOnly": True},
            )
            sz_dyn = q_dyn_path.stat().st_size / 1e6
            print(f"  FP32: {sz_fp:.1f} MB  ->  INT8 dynamic: {sz_dyn:.1f} MB ({100*sz_dyn/sz_fp:.0f}%)")

    if mode in ("static", "both"):
        print(f"Running static quantization ({hf_dataset}/{hf_subset}, {n_calib} samples)...")
        torch.cuda.empty_cache()
        gc.collect()
        quantize_static(
            str(onnx_path), str(q_static_path),
            _make_calib_reader(fe, hf_dataset, hf_subset, n_calib),
            weight_type=QuantType.QInt8,
            activation_type=QuantType.QInt8,
            per_channel=False,
            optimize_model=False,
        )
        sz_st = q_static_path.stat().st_size / 1e6
        print(f"  FP32: {sz_fp:.1f} MB  ->  INT8 static: {sz_st:.1f} MB ({100*sz_st/sz_fp:.0f}%)")

    print("PTQ complete")



def validate_onnx(onnx_path: Path, wrapper, level_names, fe, device):
    """Graph check + numeric diff against the PyTorch model."""
    print("Validating ONNX graph...")
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    for inp in onnx_model.graph.input:
        dims = [d.dim_param or d.dim_value for d in inp.type.tensor_type.shape.dim]
        print(f"  in  {inp.name}: {dims}")
    for out in onnx_model.graph.output:
        dims = [d.dim_param or d.dim_value for d in out.type.tensor_type.shape.dim]
        print(f"  out {out.name}: {dims}")
    del onnx_model

    import onnxruntime as ort
    try:
        sess = ort.InferenceSession(str(onnx_path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    except Exception:
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    print(f"Provider: {sess.get_providers()[0]}")

    ok = True
    for dur in [0.5, 1.0, 2.0, 5.0]:
        a = np.random.randn(int(16000 * dur)).astype(np.float32)
        f = fe([a], sampling_rate=16000, return_tensors="pt", padding=True)
        inp = f["input_features"].to(device)
        with torch.no_grad():
            pt = wrapper(inp)
        ort_o = sess.run(None, {"input_features": inp.cpu().numpy()})
        diffs = [np.abs(pt[i].cpu().numpy() - ort_o[i]).max() for i in range(len(level_names))]
        md = max(diffs)
        status = "OK" if md < 1e-3 else ("warn" if md < 1e-2 else "FAIL")
        if md >= 1e-2:
            ok = False
        print(f"  {dur:.1f}s T={inp.shape[1]:>4d}  diff={md:.6f}  {status}")
    print("All passed." if ok else "Some diffs exceeded threshold.")
    return sess


def demo_decode(sess, fe, level_names, model_dir: Path):
    """Run a random sample through the ONNX session and print CTC-decoded text."""
    with open(model_dir / "vocab.json", encoding="utf-8") as f:
        vocab = json.load(f)

    id2tok = {}
    for lv in level_names:
        id2tok[lv] = {token_id: token for token, token_id in vocab[lv].items()}

    def ctc_decode(logits, lv):
        ids = logits.argmax(-1).flatten().tolist()
        out, prev = [], None
        for i in ids:
            if i != prev:
                out.append(i)
            prev = i
        return "".join(id2tok[lv].get(i, f"[{i}]") for i in out if i != 0)

    da = np.random.randn(16000).astype(np.float32)
    df = fe([da], sampling_rate=16000, return_tensors="np", padding=True)
    ort_o = sess.run(None, {"input_features": df["input_features"]})
    for name, logits in zip(level_names, ort_o):
        decoded = ctc_decode(logits, name)
        suffix = "..." if len(decoded) > 80 else ""
        print(f"  {name:30s} -> {decoded[:80]}{suffix}")



def save_metadata(export_dir: Path, model_dir: Path, weights_filename: str,
                  opset: int, feature_dim: int, level_names, level_to_vocab_size, n_params: float):
    for fname in ("vocab.json", "preprocessor_config.json", "config.json"):
        src = model_dir / fname
        if src.exists():
            shutil.copy2(str(src), str(export_dir / fname))

    meta = {
        "model": "Wav2Vec2BertForMultilevelCTC (QAT-dequantized)",
        "source": weights_filename,
        "opset": opset,
        "input": f"(B, T, {feature_dim})",
        "outputs": level_names,
        "vocab_sizes": level_to_vocab_size,
        "params_M": round(n_params, 2),
        "note": "INT8 QAT weights dequantized to FP32 for ONNX export",
    }
    (export_dir / "onnx_metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    print("ONNX Export Summary")
    print("-" * 40)
    for f in sorted(export_dir.iterdir()):
        print(f"  {f.name:45s} {f.stat().st_size/1e6:>8.1f} MB")
    print(f"\n{n_params:.1f}M params | Input: (B, T, {feature_dim}) | {len(level_names)} levels")



def parse_args():
    p = argparse.ArgumentParser(description="Export Mualem Wav2Vec2-BERT to ONNX")
    p.add_argument("--model_dir",  required=True, help="Directory with config.json, weights, vocab, preprocessor")
    p.add_argument("--export_dir", default="onnx_export", help="Output directory")
    p.add_argument("--weights",    default="model_best.safetensors", dest="weights_filename")
    p.add_argument("--opset",      type=int, default=17)
    p.add_argument("--qat",        action="store_true", dest="use_qat",
                   help="Checkpoint uses QAT INT8 -- dequantize before export")
    p.add_argument("--no_quantize", action="store_false", dest="apply_quantization")
    p.add_argument("--quant_mode",  default="both", choices=("dynamic", "static", "both"))
    p.add_argument("--calib_dataset", default="obadx/muaalem-annotated-v3")
    p.add_argument("--calib_subset",  default="moshaf_0.0")
    p.add_argument("--n_calib",       type=int, default=8)
    return p.parse_args()


def main():
    args = parse_args()

    model_dir  = Path(args.model_dir)
    export_dir = Path(args.export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, level_to_vocab_size, n_params = load_model(
        model_dir, args.weights_filename, args.use_qat, device
    )

    level_names = sorted(level_to_vocab_size.keys())
    wrapper = OnnxWrapper(model, level_names)
    wrapper.eval()

    fe = SeamlessM4TFeatureExtractor.from_pretrained(str(model_dir))
    dummy_audio = np.random.randn(16000).astype(np.float32)
    dummy_feats = fe([dummy_audio], sampling_rate=16000, return_tensors="pt", padding=True)
    dummy_input = dummy_feats["input_features"].to(device)
    feature_dim = dummy_input.shape[-1]

    with torch.no_grad():
        test_out = wrapper(dummy_input)
    for name, t in zip(level_names, test_out):
        print(f"  {name:30s} -> {list(t.shape)}")

    onnx_path = export_onnx(wrapper, dummy_input, level_names, export_dir, args.opset)

    if args.apply_quantization:
        quantize_onnx(
            onnx_path, export_dir, args.quant_mode, fe,
            args.calib_dataset, args.calib_subset, args.n_calib,
        )

    sess = validate_onnx(onnx_path, wrapper, level_names, fe, device)
    demo_decode(sess, fe, level_names, model_dir)
    save_metadata(export_dir, model_dir, args.weights_filename,
                  args.opset, feature_dim, level_names, level_to_vocab_size, n_params)


if __name__ == "__main__":
    main()
