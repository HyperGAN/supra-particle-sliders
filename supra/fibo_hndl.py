"""Fibo network bindings; all custom network operations are native HNDL graphs.

The frozen pretrained model and native sampler stay with the Fibo backend.
Vendored from HyperGAN/particle-sliders c13890f8 under its MIT license.
This module contains no GAN objective, regularizer, optimizer or update loop.
"""
from copy import deepcopy
import hashlib
from importlib.metadata import version
import inspect
import json
from pathlib import Path
import sys

import torch
from torch import nn

PROFILE = 'hypergan-hndl-v2'
ROOT = Path(__file__).parent


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verified_builder():
    """Require the reviewed common builder/source closure; no private fallback."""
    if not (3, 11) <= sys.version_info[:2] < (3, 15):
        raise RuntimeError('Fibo HNDL requires Python>=3.11,<3.15')
    try:
        import hndl
        from hypergan.hndl_networks import build_network
    except ModuleNotFoundError as exc:
        raise ImportError('Fibo HNDL requires the pinned HNDL and HyperGAN builder dependencies') from exc
    if version('hndl') != '0.10.0':
        raise ValueError('Fibo HNDL requires hndl==0.10.0')
    expected = json.loads((ROOT / 'fibo_hndl_010_sources.json').read_text())
    if (expected['schema'], expected['profile'], expected['hndl_version'],
            expected['hypergan_commit'], expected['particlegan_version']) != (
            'particle-sliders-hypergan-hndl-profile-v2', PROFILE, '0.10.0',
            '5f368a3770cf5bdae0448995c9f76e29af8049f7', '0.11.0'):
        raise ValueError('Fibo HNDL profile manifest contradicts the binding')
    hndl_root = Path(hndl.__file__).parent
    actual = {str(p.relative_to(hndl_root)): _sha(p) for p in hndl_root.rglob('*.py')}
    if actual != expected['hndl']:
        raise ValueError('Fibo HNDL source closure differs from the reviewed runtime')
    hypergan_root = Path(inspect.getsourcefile(build_network)).parent
    provenance = hypergan_root / '_build_provenance.json'
    if not provenance.is_file() or json.loads(provenance.read_text()) != {
            'schema_version': 1, 'commit': expected['hypergan_commit'], 'dirty': False}:
        raise ValueError('Fibo requires the exact clean HyperGAN develop build provenance')
    if any(_sha(hypergan_root / name) != digest for name, digest in expected['hypergan'].items()):
        raise ValueError('Fibo HyperGAN builder closure differs from the reviewed runtime')
    identity = dict(profile=PROFILE, hndl_version='0.10.0',
        hypergan_commit='5f368a3770cf5bdae0448995c9f76e29af8049f7',
        hypergan_build_provenance_sha256=_sha(provenance),
        runtime_sources_sha256=_sha(ROOT / 'fibo_hndl_010_sources.json'),
        projection_sha256=_sha(ROOT / 'fibo_projection.hndl'),
        critic_sha256=_sha(ROOT / 'fibo_critic.hndl'))
    return build_network, identity


class HNDLParticleProjection(nn.Module):
    """Shape, host-hook and strength binding for one HNDL routed branch."""
    def __init__(self, name, module, particles, rank, alpha, *, builder=None,
                 identity=None, attach=True):
        super().__init__()
        if rank != 8 or alpha != 8 or tuple(particles.shape) != (128, 4):
            raise ValueError('Fibo HNDL requires rank8/alpha8 and one128x4 bank')
        if builder is None:
            builder, identity = verified_builder()
        self.lora_name, self.rank, self.scale = name, rank, alpha / rank
        self.multiplier = 0.
        self.input_width, self.output_width = module.in_features, module.out_features
        self.graph = builder((ROOT / 'fibo_projection.hndl').read_text(),
            input_shape={'value': ['B', 1, self.input_width], 'particles': ['B', 128, 4]},
            output_shape=['B', 1, self.output_width], parameters={'output_width': self.output_width})
        # Initialization policy only. Every projection/route/MLP executes in HNDL.
        nn.init.kaiming_uniform_(self.graph['down'].weight, a=5 ** .5)
        nn.init.zeros_(self.graph['up'].weight)
        self._identity = deepcopy(identity)
        object.__setattr__(self, '_particles', particles)
        self.org_forward = module.forward
        if attach:
            module.forward = self.forward

    @property
    def lora_down(self):
        return self.graph['down']

    @property
    def lora_up(self):
        return self.graph['up']

    def extra_repr(self):
        return json.dumps(self._identity, sort_keys=True)

    def forward(self, x):
        if self.multiplier == 0:
            return self.org_forward(x)
        shape = x.shape
        value = x.to(device=self.lora_down.weight.device, dtype=self.lora_down.weight.dtype)
        value = value.reshape(-1, 1, self.input_width)
        # A broadcast view gives each independent token the same owned bank.
        bank = self._particles.unsqueeze(0).expand(value.shape[0], -1, -1)
        delta = self.graph(value=value, particles=bank).reshape(*shape[:-1], self.output_width)
        return self.org_forward(x) + delta.to(x) * (self.multiplier * self.scale)


class HNDLFieldCritic(nn.Module):
    """Native spatial velocity/condition binding, preserving the FieldCritic graph."""
    score_bound = 0

    def __init__(self, channels, condition_width, spatial_shape, width=32):
        super().__init__()
        builder, identity = verified_builder()
        if any(type(n) is not int or n < 1 for n in (channels, condition_width, width, *spatial_shape)):
            raise ValueError('Fibo critic dimensions must be positive integers')
        self.graph = builder((ROOT / 'fibo_critic.hndl').read_text(),
            input_shape={'response': ['B', channels, *spatial_shape], 'condition': ['B', condition_width]},
            output_shape=['B', 1], parameters={'width': width, 'double_width': 2 * width})
        self._identity = identity

    def extra_repr(self):
        return json.dumps(self._identity, sort_keys=True)

    def forward(self, response, condition):
        return self.graph(response=response, condition=condition).squeeze(-1)
