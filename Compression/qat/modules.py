"""QAT building blocks: fake-quantize STE, observer, training Linear, INT8 export."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FakeQuantizeSTE(torch.autograd.Function):
    """Round-to-nearest fake quant with a straight-through gradient.

    Gradient passes through unchanged for in-range values and is zeroed for
    values saturated against the (qmin, qmax) clamp.
    """

    @staticmethod
    def forward(ctx, x, scale, zp, qmin, qmax):
        xq  = torch.clamp(torch.round(x / scale + zp), qmin, qmax) # quantization
        xdq = (xq - zp) * scale # dequantization 
        ctx.save_for_backward((xq >= qmin) & (xq <= qmax))
        return xdq

    @staticmethod
    def backward(ctx, g):
        mask, = ctx.saved_tensors
        return g * mask.float(), None, None, None, None


class MinMaxObserver(nn.Module):
    """EMA min/max observer; produces (scale, zero_point, qmin, qmax) on demand."""

    def __init__(self, bits: int = 8, symmetric: bool = True, ema_decay: float = 0.999):
        super().__init__()
        self.bits = bits
        self.symmetric = symmetric
        self.ema_decay = ema_decay
        self.register_buffer("ema_min",      torch.tensor(float("inf")))
        self.register_buffer("ema_max",      torch.tensor(float("-inf")))
        self.register_buffer("num_observed", torch.tensor(0, dtype=torch.long))

    def forward(self, x):
        if self.training:
            mn, mx = x.detach().min(), x.detach().max()
            if self.num_observed == 0:
                self.ema_min.copy_(mn)
                self.ema_max.copy_(mx)
            else:
                self.ema_min.copy_(self.ema_decay * self.ema_min + (1 - self.ema_decay) * mn)
                self.ema_max.copy_(self.ema_decay * self.ema_max + (1 - self.ema_decay) * mx)
            self.num_observed += 1
        return x

    def scale_zp(self):
        if self.symmetric:
            qmax = 2 ** (self.bits - 1) - 1
            qmin = -qmax - 1
            amax = torch.clamp(torch.max(self.ema_min.abs(), self.ema_max.abs()), min=1e-8)
            return amax / qmax, torch.tensor(0.0, device=amax.device), qmin, qmax
        qmin = 0
        qmax = 2 ** self.bits - 1
        r = torch.clamp(self.ema_max - self.ema_min, min=1e-8)
        sc = r / (qmax - qmin)
        zp = torch.clamp(torch.round(qmin - self.ema_min / sc), qmin, qmax)
        return sc, zp, qmin, qmax


class QATLinear(nn.Module):
    """Training-time stand-in for nn.Linear that fake-quantizes weights and
    activations to ``bits`` precision while preserving gradient flow.

    Forward ordering:
      1. update activation observer (live input distribution)
      2. read scale/zp for activations and (current) weights
      3. fake-quantize both in fp32 under autocast-disabled
      4. update the weight observer with the *detached* weight
      5. linear + cast back to input dtype
    """

    def __init__(self, linear: nn.Linear,
                 bits: int = 8, symmetric: bool = True, ema_decay: float = 0.999):
        super().__init__()
        self.in_features  = linear.in_features
        self.out_features = linear.out_features
        self.weight = nn.Parameter(linear.weight.data.clone())
        self.bias   = nn.Parameter(linear.bias.data.clone()) if linear.bias is not None else None
        self.weight_obs = MinMaxObserver(bits, symmetric, ema_decay)
        self.act_obs    = MinMaxObserver(bits, symmetric, ema_decay)

    def forward(self, x):
        self.act_obs(x)
        a_sc, a_zp, a_qmin, a_qmax = self.act_obs.scale_zp()
        w_sc, w_zp, w_qmin, w_qmax = self.weight_obs.scale_zp()
        with torch.cuda.amp.autocast(enabled=False):
            xf = FakeQuantizeSTE.apply(x.float(),      a_sc, a_zp, a_qmin, a_qmax)
            wf = FakeQuantizeSTE.apply(self.weight.float(), w_sc, w_zp, w_qmin, w_qmax)
            self.weight_obs(self.weight.detach())
        b = self.bias.float() if self.bias is not None else None
        return F.linear(xf, wf, b).to(x.dtype)


class QuantizedLinear(nn.Module):
    """INT8 export of a trained ``QATLinear``.

    Stores int8 weights + the fp32 (de)quant parameters so on-device inference
    can run integer matmul where supported. Activations are dynamically
    quantised at forward time using the trained observers' scale/zp/range.
    """

    def __init__(self, ql: QATLinear):
        super().__init__()
        w_sc, w_zp, w_qmin, w_qmax = ql.weight_obs.scale_zp()
        a_sc, a_zp, a_qmin, a_qmax = ql.act_obs.scale_zp()
        w_int = torch.clamp(torch.round(ql.weight.data / w_sc + w_zp),
                            w_qmin, w_qmax).to(torch.int8)
        self.register_buffer("weight_int8",  w_int)
        self.register_buffer("weight_scale", w_sc)
        self.register_buffer("weight_zp",    w_zp)
        self.register_buffer("act_scale",    a_sc)
        self.register_buffer("act_zp",       a_zp)
        self.register_buffer("act_qmin",     torch.tensor(float(a_qmin)))
        self.register_buffer("act_qmax",     torch.tensor(float(a_qmax)))
        self.bias = ql.bias
        self.in_features, self.out_features = ql.in_features, ql.out_features

    def forward(self, x):
        xq  = torch.clamp(torch.round(x / self.act_scale + self.act_zp),
                          self.act_qmin.item(), self.act_qmax.item())
        xdq = (xq - self.act_zp) * self.act_scale
        wdq = (self.weight_int8.float() - self.weight_zp) * self.weight_scale
        return F.linear(xdq, wdq, self.bias)


def replace_with_qat(model, skip_prefixes, bits: int, symmetric: bool, ema_decay: float):
    """Swap every ``nn.Linear`` under ``model`` (except those whose path starts
    with one of ``skip_prefixes``) with a ``QATLinear`` carrying the same
    weights. Returns the number of replacements.
    """
    replaced = 0

    def _recurse(module, prefix=""):
        nonlocal replaced
        for name, child in list(module.named_children()):
            full = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear) and not any(full.startswith(s) for s in skip_prefixes):
                setattr(module, name, QATLinear(child, bits, symmetric, ema_decay))
                replaced += 1
            else:
                _recurse(child, full)

    _recurse(model)
    return replaced


def convert_to_int8(module):
    """Recursively swap every ``QATLinear`` for its ``QuantizedLinear`` export."""
    for name, child in list(module.named_children()):
        if isinstance(child, QATLinear):
            setattr(module, name, QuantizedLinear(child))
        else:
            convert_to_int8(child)
