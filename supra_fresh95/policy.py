"""Official PG095 on SAME owned HNDL Parameters; no global host declarations."""
from dataclasses import dataclass
from importlib import metadata
import hashlib, math
import torch
from torch import nn
from particlegan import init as pg_init
from . import PROFILE
from .util import canonical, source_record

INIT_SHA = '3534ff23efa413cdf8b4125b91e0d30671d873b3e2958547d4e744b750454b9b'
SCHEMA = 'Supra-explicit-downQR-sqrt5-otherKEEP-initialization-v1'
MARKER = '_supra_explicit_fresh95_receipt'

@dataclass(frozen=True)
class Construction:
    mode: str
    cohort: str
    options_sha256: str

class DeclaredOwned(nn.Module):
    def __init__(self, rows):
        super().__init__()
        self.specs = {}
        for index, row in enumerate(rows):
            key = 'p'+str(index)
            self.register_parameter(key, row['parameter'])
            self.specs[key] = row['distribution']

pg_init.register(DeclaredOwned, lambda module: module.specs)

def tensor_sha(value):
    flat = torch.empty(value.numel(), dtype=value.dtype, device='cpu')
    flat.copy_(value.detach().reshape(-1).cpu())
    return hashlib.sha256(flat.view(torch.uint8).numpy().tobytes()).hexdigest()

def seed_for(seed, name):
    return int.from_bytes(hashlib.sha256(f'{PROFILE}:{seed}:{name}'.encode()).digest()[:8], 'big') % 2**63

def declarations(runtime, critic):
    from supra.hndl import HNDLLoRALinear
    from hndl.operators.linear import Linear
    rows, seen = [], set()
    if not runtime.loras or any(type(layer) is not HNDLLoRALinear for layer in runtime.loras):
        raise ValueError('only the exact original Supra HNDL projection lane is supported')
    for index, layer in enumerate(runtime.loras):
        for role, leaf in (('down', layer.down), ('up', layer.up)):
            if (type(leaf) is not Linear or leaf.equalized is not False or leaf.bias is not None
                    or any(leaf.children()) or list(leaf.buffers())
                    or nn.utils.parametrize.is_parametrized(leaf)
                    or set(leaf._parameters) != {'weight', 'bias'}
                    or tuple(leaf.weight.shape) != (leaf.out_features, leaf.in_features)):
                raise ValueError('exact original bias-free non-equalized HNDL down/up required')
            parameter = leaf.weight
            if role == 'up' and torch.count_nonzero(parameter).item():
                raise ValueError('protected UP must genuinely be zero before fresh construction')
            bound = 1 / math.sqrt(leaf.in_features)
            rows.append(dict(name=f'adapter.{index}.{role}.weight', parameter=parameter,
                distribution=pg_init.Uniform(-bound, bound) if role == 'down' else pg_init.KEEP,
                role='host-a-sqrt5-down-QR' if role == 'down' else 'protected-zero-UP-KEEP'))
    rows += [dict(name='critic.'+name, parameter=p, distribution=pg_init.KEEP,
                  role='original-HNDL-critic-constructor-KEEP')
             for name, p in critic.named_parameters(remove_duplicate=False) if p.requires_grad]
    for row in rows:
        p = row['parameter']
        if (type(p) is not nn.Parameter or not p.requires_grad or p.grad is not None
                or p.dtype != torch.float32 or not p.is_contiguous() or not torch.isfinite(p).all()
                or id(p) in seen or hasattr(p, MARKER)):
            raise ValueError('unique fresh FP32 owned Parameter required')
        seen.add(id(p))
    native = {id(p) for p in runtime.model.parameters() if p.requires_grad}
    owned = {id(p) for p in critic.parameters() if p.requires_grad}
    if seen != native | owned or native & owned:
        raise ValueError('scopes must cover exactly native adapter and critic, with no bank')
    return rows

def initialize(runtime, critic, construction, options, sources):
    if (not isinstance(construction, Construction) or construction.mode != 'fresh'
            or construction.options_sha256 != hashlib.sha256(canonical(options).encode()).hexdigest()):
        raise ValueError('explicit fresh intent before optimizers required')
    if metadata.version('particlegan') != '0.9.5' or source_record(pg_init.__file__)['sha256'] != INIT_SHA:
        raise ValueError('exact public PG095 initializer Source required')
    rows = declarations(runtime, critic)
    holder = DeclaredOwned(rows)
    before = [tensor_sha(row['parameter']) for row in rows]
    identities = [id(row['parameter']) for row in rows]
    seeds = {'p'+str(i): seed_for(options.get('seed', 7), row['name']) for i, row in enumerate(rows)}
    pg_init.deterministic_orthogonal_(holder, strict=True, parameter_seeds=seeds)
    if identities != [id(row['parameter']) for row in rows]:
        raise RuntimeError('official initializer replaced an owned Parameter')
    parameters = []
    for i, row in enumerate(rows):
        after = tensor_sha(row['parameter'])
        if row['distribution'] is pg_init.KEEP and before[i] != after:
            raise RuntimeError('KEEP constructor/zero-UP state changed')
        parameters.append(dict(name=row['name'], role=row['role'], seed=seeds['p'+str(i)],
            shape=list(row['parameter'].shape), dtype=str(row['parameter'].dtype),
            distribution=repr(row['distribution']), before_sha256=before[i], after_sha256=after))
    receipt = dict(schema=SCHEMA, profile=PROFILE, cohort=construction.cohort,
        options_sha256=construction.options_sha256, completed_steps=0, restored_from=None,
        source_identity=sources, initializer=source_record(pg_init.__file__), seed=options.get('seed', 7),
        parameters=parameters, parameter_identity_preserved=True, particle_bank=None,
        critic_policy='original-constructor-KEEP', selection='raw', ema_decay=None)
    marker = hashlib.sha256(canonical(receipt).encode()).hexdigest()
    for row in rows:
        setattr(row['parameter'], MARKER, marker)
    return receipt

def validate_receipt(receipt, construction, options, sources):
    if (receipt.get('schema') != SCHEMA or receipt.get('profile') != PROFILE
            or receipt.get('cohort') != construction.cohort
            or receipt.get('options_sha256') != construction.options_sha256
            or receipt.get('source_identity') != sources or receipt.get('completed_steps') != 0
            or receipt.get('restored_from') is not None or receipt.get('ema_decay') is not None
            or receipt.get('particle_bank') is not None or receipt.get('selection') != 'raw'
            or receipt.get('seed') != options.get('seed', 7)
            or receipt.get('initializer', {}).get('sha256') != INIT_SHA):
        raise ValueError('exact original new-profile initialization receipt required')
