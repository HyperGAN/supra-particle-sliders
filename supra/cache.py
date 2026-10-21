"""Supra frozen-context and native teacher-cache policy; no GAN implementation.

Selected Supra-only definitions relocated from the preserved fd89e1c campaign
module. See thin-product-provenance.json and licenses/NOTICE. Rows implement the
public native provider protocol; the shared game consumes this product object.
"""
from copy import deepcopy
import hashlib
import inspect
import json
import math
from pathlib import Path

import torch
from particle_sliders.pg09_backends import PreparedResponse, ResponseRow
from .shared import file_sha as file_digest


def canonical_digest(value):
    # Keep the exact historical JSON separator convention for asset receipts.
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


SUPRA_SCALE = 0.32844100447280516
SUPRA_MODEL_REVISION = '10dec6e4b4b5d1c44fd1d7d3fe5e50137333da5b'
SUPRA_TEXT_REVISION = '7bcac572ce56db69c1ea7c8af255c5d7c9672fc2'
SUPRA_CONFIG_SHA256 = 'd32639ad9084f60d13faea1ef0e4d92b4bfc2e0420bc0fd15f4992c6270edd5b'


def tensor_identity(tensor):
    value = tensor.detach().cpu().contiguous()
    return dict(shape=list(value.shape), dtype=str(value.dtype),
                sha256=hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())

def required_supra_prompts(config):
    prompts = {''} | set(config['preservation'])
    for row in config['rows']:
        prompts.update((row['neutral'], row['positive']))
    for row in config['verification']:
        prompts.update((row['prompt'], row.get('teacher', row['prompt'])))
    return sorted(prompts)

def export_admitted_supra_contexts(runtime, config, destination, *, runtime_identity):
    """Explicit preparation phase, called only after native runtime admission.

    Reuses native encode (including stored unconditional); never loads components.
    No implicit re-encoding is allowed by the training bridge below.
    """
    from safetensors.torch import save_file
    prompts = required_supra_prompts(config)
    if len(prompts) > 19:
        raise ValueError('Supra preparation exceeds the19 unique-context encode budget')
    if not runtime_identity.get('model_revision') or not runtime_identity.get('text_revision'):
        raise ValueError('verified model and native T5 revision identity required')
    runtime_source = inspect.getsourcefile(type(runtime))
    if runtime_source is None:
        raise ValueError('native runtime must have an inspectable pinned source')
    destination = Path(destination)
    if destination.exists() or destination.with_suffix('.json').exists():
        raise FileExistsError(destination)
    tensors, records = {}, []
    for index, prompt in enumerate(prompts):
        context, mask = runtime.encode(prompt)
        context, mask = context.detach().cpu().contiguous(), mask.detach().cpu().contiguous()
        if (context.ndim != 3 or context.shape[0] != 1 or mask.shape != context.shape[:2] or
                not torch.isfinite(context).all() or not torch.isfinite(mask).all() or not mask.any()):
            raise ValueError('invalid frozen native context')
        tensors[f'e{index}'], tensors[f'm{index}'] = context, mask
        records.append(dict(prompt=prompt, embedding=tensor_identity(context), mask=tensor_identity(mask)))
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(destination))
    receipt = dict(schema='supra-frozen-contexts-v1', sha256=file_digest(destination), records=records,
                   config_sha256=canonical_digest(config), unique_contexts=len(prompts), max_encode_calls=19,
                   stored_unconditional=runtime.stored_unconditional, runtime_identity=deepcopy(runtime_identity),
                   runtime_source=str(Path(runtime_source).resolve()),
                   runtime_source_sha256=file_digest(runtime_source), torch_version=str(torch.__version__),
                   policy='native pinned T5 encoding; original stored unconditional retained')
    destination.with_suffix('.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt

def validate_live_supra_context_preparation(config, runtime_identity):
    """Strict reviewed live contract; no encoding, output or runtime creation."""
    if len(required_supra_prompts(config)) != 19 or canonical_digest(config) != SUPRA_CONFIG_SHA256:
        raise ValueError('live Supra preparation requires the exact reviewed19-context config')
    if (runtime_identity.get('model_revision') != SUPRA_MODEL_REVISION or
            runtime_identity.get('text_revision') != SUPRA_TEXT_REVISION):
        raise ValueError('live Supra model/T5 revisions differ from reviewed preparation')
    return dict(config_sha256=SUPRA_CONFIG_SHA256, unique_contexts=19,
                model_revision=SUPRA_MODEL_REVISION, text_revision=SUPRA_TEXT_REVISION)

def export_live_supra_contexts(runtime, config, destination, *, runtime_identity):
    """Owner-admitted preparation only: exact config/revisions checked first."""
    validate_live_supra_context_preparation(config, runtime_identity)
    return export_admitted_supra_contexts(runtime, config, destination, runtime_identity=runtime_identity)

def load_supra_contexts(path, expected_sha256, config):
    from safetensors.torch import load_file
    path = Path(path)
    receipt = json.loads(path.with_suffix('.json').read_text())
    if (receipt.get('schema') != 'supra-frozen-contexts-v1' or
            receipt.get('sha256') != expected_sha256 or file_digest(path) != expected_sha256 or
            [r['prompt'] for r in receipt['records']] != required_supra_prompts(config)):
        raise ValueError('frozen context checksum/prompt identity differs')
    tensors = load_file(str(path), device='cpu')
    expected = {f'{kind}{i}' for i in range(len(receipt['records'])) for kind in ('e', 'm')}
    if set(tensors) != expected:
        raise ValueError('frozen context keys differ')
    result = {}
    for i, record in enumerate(receipt['records']):
        context, mask = tensors[f'e{i}'], tensors[f'm{i}']
        if tensor_identity(context) != record['embedding'] or tensor_identity(mask) != record['mask']:
            raise ValueError('frozen context tensor identity differs')
        result[record['prompt']] = (context, mask)
    return result

class SupraCampaignRows:
    """Small CPU teacher cache; selected native responses only move to the model.

    Two copies per edited state and four per preservation state retain the old
    4:1 update-frequency mixture under balanced sampling. The old optimized MSE
    coefficient0.1 is deliberately absent from this GAN-only proposal.
    """
    policy = dict(version='supra-native-cfg-context-time-v1',
                  suffix=['native_time', 'scale', 'strength', 'guidance/3', 'positive_path', 'preservation'],
                  pooling='masked neutral T5 mean / pooled RMS', row_id_feature=False,
                  edit_repetitions=2, preservation_repetitions=4,
                  validation_path_without_archived_label=-1,
                  cfg_adapter_branch='conditional_only', optimized_mse=False)

    def __init__(self, cache_path, *, cache_sha256, config, contexts, split='train', device='cpu'):
        if split not in ('train', 'validation'):
            raise ValueError('explicit train or validation split required')
        if file_digest(cache_path) != cache_sha256:
            raise ValueError('native teacher cache checksum differs')
        self.split, self.device = split, torch.device(device)
        self.config = deepcopy(config)
        config = self.config
        self.cache_path, self.cache_sha256 = str(Path(cache_path).resolve()), cache_sha256
        data = torch.load(cache_path, map_location='cpu', weights_only=True)
        if not data.get('records') or not data.get('holds'):
            raise ValueError('edited and preservation native pools required')
        self.records, self.holds = data['records'], data['holds']
        self.contexts = {p: tuple(t.detach().cpu().clone() for t in contexts[p])
                         for p in required_supra_prompts(config)}
        self.context_width = self.contexts[''][0].shape[-1]
        self.condition_width = self.context_width + 6
        self.response_shape = tuple(self.records[0]['z'].shape)
        if self.response_shape != (1, 4, 32, 32):
            raise ValueError('Supra requires native [1,4,32,32] fields')
        self._pooled = {}
        for prompt, (embedding, mask) in self.contexts.items():
            if (embedding.ndim != 3 or embedding.shape[0] != 1 or
                    embedding.shape[-1] != self.context_width or mask.shape != embedding.shape[:2] or
                    not torch.isfinite(embedding).all() or not torch.isfinite(mask).all() or
                    not torch.all((mask == 0) | (mask == 1)) or not mask.any()):
                raise ValueError('frozen T5 context/mask schema differs')
            pooled = (embedding.float() * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
            self._pooled[prompt] = pooled / pooled.square().mean().sqrt().clamp_min(1e-12)
        row_by_name = {r['name']: r for r in config['rows']}
        for hold, pool in ((False, self.records), (True, self.holds)):
            for row in pool:
                if (not math.isfinite(row['t']) or not 0 <= row['t'] < 1 or
                        any(not isinstance(row[k], torch.Tensor) or tuple(row[k].shape) != self.response_shape or
                            row[k].dtype != torch.float32 or not torch.isfinite(row[k]).all()
                            for k in ('z', 'neutral', 'target'))):
                    raise ValueError('cached native velocity/time schema differs')
                if hold:
                    if row['prompt'] not in config['preservation'] or not torch.equal(row['neutral'], row['target']):
                        raise ValueError('preservation row/target differs')
                elif (row.get('row') not in row_by_name or row['prompt'] != row_by_name[row['row']]['neutral'] or
                      (split == 'train' and row.get('context') not in ('neutral', 'positive'))):
                    raise ValueError('teacher prompt/row pairing differs')
        self.row_by_name = row_by_name
        self.routes = ([(False, i) for _ in range(2) for i in range(len(self.records))] +
                       [(True, i) for _ in range(4) for i in range(len(self.holds))])
        self._identity = dict(schema='supra-campaign-provider-v1', split=split,
                              cache_path=self.cache_path, cache_sha256=cache_sha256,
                              config=deepcopy(config), policy=deepcopy(self.policy), response_shape=list(self.response_shape),
                              rows=len(self.routes), edited_states=len(self.records), preservation_states=len(self.holds),
                              contexts={p: [tensor_identity(t) for t in pair] for p, pair in self.contexts.items()})

    def __len__(self):
        return len(self.routes)

    def identity(self):
        return deepcopy(self._identity)

    def _source(self, index):
        if type(index) is not int or not 0 <= index < len(self):
            raise IndexError(index)
        hold, offset = self.routes[index]
        return hold, (self.holds if hold else self.records)[offset]

    def __getitem__(self, index):
        hold, source = self._source(index)
        positive = source['prompt'] if hold else self.row_by_name[source['row']]['positive']
        row = ResponseRow(source['prompt'], positive, '', source['z'].to(self.device).clone(),
                          timestep=torch.tensor(source['t'], device=self.device), scale=1., guidance=3.)
        return PreparedResponse(row, source['target'].clone(), source['neutral'].clone())

    def conditions(self, indices):
        rows = []
        for index in indices:
            hold, source = self._source(int(index))
            role = source.get('context')
            path = -1. if hold or role is None else float(role == 'positive')
            suffix = torch.tensor([[source['t'], 1., 1., 1., path, float(hold)]])
            rows.append(torch.cat((self._pooled[source['prompt']], suffix), dim=1))
        return torch.cat(rows).detach() if rows else torch.empty((0, self.condition_width))

    def response_scale(self):
        if self.split != 'train':
            raise ValueError('training split required for response scale')
        energy = sum(float((r['target'] - r['neutral']).double().square().sum()) for r in self.records)
        count = sum(r['target'].numel() for r in self.records)
        value = math.sqrt(energy / count)
        if not math.isfinite(value) or value <= 0:
            raise ValueError('nonzero finite training contrast scale required')
        return value
