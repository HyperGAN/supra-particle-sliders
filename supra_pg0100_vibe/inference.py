"""Strict RAW inference loading for this separately identified native HNDL law."""
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from .native import Backend, NativeHNDLLoRALinear, frozen_digest, sha


def load_raw(runtime, path):
    path = Path(path)
    with safe_open(path, framework='pt', device='cpu') as stream:
        metadata = stream.metadata()
    if (metadata.get('format') != 'Supra-native256-vibe-FP32-HNDL-v1'
            or metadata.get('weights') != 'raw' or metadata.get('ema_decay') != 'null'):
        raise ValueError('explicit native FP32 HNDL RAW profile required')
    identity = json.loads(metadata['model'])
    stable = ('profile', 'model', 'model_revision', 'text_revision', 'vae_revision',
              'checkpoint_sha256', 'frozen_parameter_sha256', 'rank', 'alpha',
              'base_precision', 'builder', 'weight_selection', 'ema_decay')
    if any(identity.get(k) != runtime.vibe_identity.get(k) for k in stable):
        raise ValueError('RAW base, architecture or native math profile differs')
    if (not runtime.loras or any(type(m) is not NativeHNDLLoRALinear for m in runtime.loras)
            or any(p.device.type != 'cuda' for p in runtime.model.parameters())):
        raise ValueError('whole native CUDA model and this FP32 HNDL adapter class required')
    backend = Backend(runtime)
    names = list(backend.network.names)
    if json.loads(metadata['adapter_names']) != names:
        raise ValueError('RAW adapter ownership/order differs')
    tensors = load_file(path, device='cpu')
    if set(tensors) != set(names):
        raise ValueError('RAW tensor names differ')
    for name, parameter in zip(names, backend.network.weights, strict=True):
        value = tensors[name]
        if (value.shape != parameter.shape or value.dtype != torch.float32
                or parameter.dtype != torch.float32 or not torch.isfinite(value).all()):
            raise ValueError('RAW shape, dtype or finite-value contract differs')
    before = frozen_digest(runtime)
    if before != identity['frozen_parameter_sha256']:
        raise ValueError('pretrained weights differ before restore')
    with torch.no_grad():
        for name, parameter in zip(names, backend.network.weights, strict=True):
            parameter.copy_(tensors[name].to(parameter.device))
    if frozen_digest(runtime) != before:
        raise RuntimeError('RAW restore changed pretrained weights')
    return dict(path=str(path.resolve()), sha256=sha(path), step=int(metadata['step']),
                format=metadata['format'], selection='raw', ema_decay=None)
