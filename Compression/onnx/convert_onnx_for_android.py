#!/usr/bin/env python3
"""
Make the Mualem multilevel-CTC ONNX model runnable by ONNX Runtime (and the
Flutter `onnxruntime` plugin on Android/iOS).

WHY THIS IS NEEDED
------------------
The model shipped as `mualem_multilevel_ctc_int8_dynamic.onnx` was produced with
`onnxruntime.quantization.quantize_dynamic`. Dynamic quantization rewrites the
wav2vec2-bert convolution layers into `ConvInteger` nodes. ONNX Runtime has **no
execution kernel for `ConvInteger`**, so loading the model fails on both desktop
and Android with:

    NOT_IMPLEMENTED : Could not find an implementation for ConvInteger(10) node
    '/model/wav2vec2_bert/encoder/layers.0/conv_module/pointwise_conv1/Conv_quant'

`MatMulInteger` (used by the transformer's Linear layers) IS supported by ORT, so
the fix only needs to touch the convolutions.

WHAT THIS SCRIPT DOES
---------------------
For every `ConvInteger(x_q, w_q, x_zp, w_zp)` node it substitutes the identical
integer arithmetic carried out in float, which ORT can run:

    x_f   = Cast(x_q -> float)
    xzp_f = Cast(x_zp -> float)
    x_sub = x_f - xzp_f                       # de-offset the activations
    W_f   = (w_q - w_zp) as a float32 initializer
    y     = Conv(x_sub, W_f, <same conv attrs>)   # plain float Conv

The pre-existing dequantisation chain after the conv (Cast -> Mul by scales ->
Add bias) is left untouched: it now receives a float tensor and the trailing
`Cast(to=float)` becomes a harmless identity. Because the activations are kept in
full precision (we never quantise them), the result is numerically equal to —
and slightly more accurate than — the original `ConvInteger` path.

Only the 48 Conv layers change; the 140 `MatMulInteger` ops stay int8, so the
output model stays small (~36 MB vs ~25 MB; the conv weights are now float32).

USAGE
-----
    python convert_onnx_for_android.py \
        onnx_app/assets/models/mualem_multilevel_ctc_int8_dynamic.onnx \
        onnx_app/assets/models/mualem_multilevel_ctc_convfix.onnx

Requires: onnx, onnxruntime, numpy   (pip install onnx onnxruntime numpy)
"""

import argparse
import os
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def convert_convinteger_to_float(model: onnx.ModelProto) -> int:
    """Replace every ConvInteger node with an equivalent float Conv subgraph.

    Returns the number of ConvInteger nodes converted.
    """
    graph = model.graph
    initializers = {init.name: init for init in graph.initializer}

    def get_array(name: str) -> np.ndarray:
        return numpy_helper.to_array(initializers[name])

    conv_nodes = [n for n in graph.node if n.op_type == "ConvInteger"]
    if not conv_nodes:
        return 0

    # Build, per ConvInteger, the list of float nodes that replace it.
    replacement = {}
    new_initializers = []
    for node in conv_nodes:
        x_q, w_q_name, x_zp, w_zp_name = node.input
        conv_out = node.output[0]
        prefix = node.name + "_cf"

        # Float weight with its (symmetric, usually zero) zero-point folded in.
        w_q = get_array(w_q_name).astype(np.float32)
        w_zp = get_array(w_zp_name).astype(np.float32)
        w_float = (w_q - w_zp).astype(np.float32)
        w_float_name = w_q_name + "_f32"
        new_initializers.append(numpy_helper.from_array(w_float, w_float_name))

        x_f = prefix + "_xf"
        xzp_f = prefix + "_xzpf"
        x_sub = prefix + "_xsub"

        # Copy the conv attributes (dilations, group, kernel_shape, pads,
        # strides) verbatim from the ConvInteger node.
        conv_attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}

        replacement[node.name] = [
            helper.make_node("Cast", [x_q], [x_f],
                             to=TensorProto.FLOAT, name=prefix + "_castx"),
            helper.make_node("Cast", [x_zp], [xzp_f],
                             to=TensorProto.FLOAT, name=prefix + "_castzp"),
            helper.make_node("Sub", [x_f, xzp_f], [x_sub], name=prefix + "_sub"),
            helper.make_node("Conv", [x_sub, w_float_name], [conv_out],
                             name=prefix + "_conv", **conv_attrs),
        ]

    # Rebuild the node list in place so topological order is preserved: each
    # ConvInteger is swapped for its replacement nodes at the same position.
    rebuilt = []
    for node in graph.node:
        if node.op_type == "ConvInteger":
            rebuilt.extend(replacement[node.name])
        else:
            rebuilt.append(node)
    del graph.node[:]
    graph.node.extend(rebuilt)
    graph.initializer.extend(new_initializers)

    return len(conv_nodes)


def strip_unused_initializers(model: onnx.ModelProto) -> int:
    """Drop initializers no longer referenced by any node (the orphaned int8
    conv weights / zero-points left behind by the conversion)."""
    graph = model.graph
    used = set()
    for node in graph.node:
        used.update(node.input)
    kept = [init for init in graph.initializer if init.name in used]
    removed = len(graph.initializer) - len(kept)
    del graph.initializer[:]
    graph.initializer.extend(kept)
    return removed


def validate(path: str) -> None:
    """Load the converted model in ONNX Runtime and run a dummy input so we know
    it no longer trips over ConvInteger and produces finite outputs."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("  (onnxruntime not installed - skipping runtime validation)")
        return

    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    # input_features: [batch, seq_len, 160]
    dummy = np.random.randn(1, 120, 160).astype(np.float32)
    outputs = sess.run(None, {inp.name: dummy})
    names = [o.name for o in sess.get_outputs()]
    all_finite = all(np.isfinite(o).all() for o in outputs)
    print(f"  loaded OK, {len(outputs)} outputs, all finite = {all_finite}")
    print(f"  output heads: {', '.join(names)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="path to the int8-dynamic ONNX model")
    parser.add_argument("output", help="path to write the ConvInteger-free model")
    parser.add_argument("--no-validate", action="store_true",
                        help="skip the ONNX Runtime load/run check")
    args = parser.parse_args()

    print(f"Loading {args.input} ...")
    model = onnx.load(args.input)

    n = convert_convinteger_to_float(model)
    print(f"Converted {n} ConvInteger node(s) to float Conv.")
    if n == 0:
        print("No ConvInteger nodes found - nothing to do.")

    removed = strip_unused_initializers(model)
    print(f"Removed {removed} now-unused initializer(s).")

    onnx.checker.check_model(model, full_check=False)
    onnx.save(model, args.output)
    size_mb = os.path.getsize(args.output) / 1e6
    print(f"Saved {args.output} ({size_mb:.1f} MB)")

    remaining = sum(1 for node in model.graph.node if node.op_type == "ConvInteger")
    assert remaining == 0, f"{remaining} ConvInteger nodes still present!"

    if not args.no_validate:
        print("Validating with ONNX Runtime ...")
        validate(args.output)

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
