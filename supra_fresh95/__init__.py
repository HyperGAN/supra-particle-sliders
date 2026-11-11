"""Explicit opt-in HNDL Supra policy; no registration or old cohort migration."""
PROFILE = 'supra-hndl-explicit-cached-fresh95-v1'

def open_training(*args, **kwargs):
    from .training import Training
    return Training(*args, **kwargs)

def open_inference(*args, **kwargs):
    from .inference import Inference
    return Inference(*args, **kwargs)

from .util import Checkpoint
__all__ = ['PROFILE', 'Checkpoint', 'open_training', 'open_inference']
