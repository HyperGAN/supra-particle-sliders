"""Allocation-free learning-rate input admission; the shared recipe owns rates."""
import math


def recipe_overrides(*, lr=None, d_lr_mult=None, prior_lr_mult=None):
    if prior_lr_mult is not None:
        raise ValueError("prior_lr_mult is unavailable for this adapter-only GAN")
    values = dict(lr=lr, d_lr_mult=d_lr_mult, prior_lr_mult=prior_lr_mult)
    result = {}
    for key, value in values.items():
        if value is not None:
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(key + " must be finite and positive; booleans are unsupported")
            result[key] = float(value)
    return result
