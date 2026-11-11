"""Distinct cached HNDL owner; original native adapter/game and formulas only."""
from contextlib import contextmanager
from copy import deepcopy
import hashlib, inspect, math
from pathlib import Path
import torch
from particle_sliders.field_critic import seed_initialization_rng
from particle_sliders.fibo_hndl import HNDLFieldCritic
from particle_sliders.pg095 import ConditionalPG095
from particle_sliders.pg09_backends import SupraRuntimeResponseAdapter
from supra import cache, hndl, hndl_lifecycle, native_io
from supra.runtime import MODEL_REV, T5_REV
from supra.shared import file_sha, source_cohort
from . import PROFILE
from . import policy, ownership
from .util import canonical, source_record, Checkpoint, file_pin, read_packet, save_packet, write_json, new_directory, equal

FULL_SCHEMA = 'Supra-explicit-cached-fresh95-v1-FULL'
RAW_SCHEMA = 'Supra-explicit-cached-fresh95-v1-RAW'

def guard():
    """No source-root/version shortcut; execute the unchanged imported guard."""
    from supra import __file__ as product_file
    from .baseline import ORIGINAL_SOURCES
    base = Path(product_file).absolute().parent
    for name, expected in ORIGINAL_SOURCES.items():
        if source_record(base/name) != expected:
            raise RuntimeError('captured original Supra Source differs: '+name)
    hndl.verify_sources()
    from importlib import metadata
    from particlegan import init as pg_init
    if metadata.version('particlegan') != '0.9.5' or source_record(pg_init.__file__)['sha256'] != policy.INIT_SHA:
        raise RuntimeError('exact original public initializer required')

def sources():
    from particle_sliders import pg09, pg09_backends, lifecycle
    return dict(profile=PROFILE,
        own={p.name: source_record(p) for p in sorted(Path(__file__).absolute().parent.glob('*.py'))},
        original={name: source_record(module.__file__) for name, module in
            (('hndl',hndl),('factory',hndl_lifecycle),('native_io',native_io),('cache',cache),
             ('pg09',pg09),('pg09_backends',pg09_backends),('lifecycle',lifecycle))})

def read_assets(options):
    """Original asset predicates, with no teacher, encoder, model or game call."""
    hndl_lifecycle.validate(options)
    config = options['native_config']
    receipt = native_io.read_context_receipt(options['contexts'])
    contexts = cache.load_supra_contexts(options['contexts'],options['contexts_sha256'],config)
    if (receipt.get('config_sha256') != cache.canonical_digest(config)
            or receipt.get('runtime_identity') != dict(model_revision=MODEL_REV,text_revision=T5_REV)
            or receipt.get('unique_contexts') != 19 or receipt.get('max_encode_calls') != 19
            or receipt.get('stored_unconditional') is not True):
        raise ValueError('original frozen context/config/revision receipt required')
    rows = cache.SupraCampaignRows(options['cache'],cache_sha256=options['cache_sha256'],config=config,contexts=contexts)
    if ({row['row'] for row in rows.records} != {row['name'] for row in config['rows']}
            or {row['prompt'] for row in rows.holds} != set(config['preservation'])
            or not math.isclose(rows.response_scale(),cache.SUPRA_SCALE,rel_tol=0,abs_tol=1e-12)):
        raise ValueError('original cached coverage/response scale required')
    return rows, contexts, receipt

def runtime_check(runtime,options,receipt):
    if (receipt.get('runtime_source_sha256') != file_sha(inspect.getsourcefile(type(runtime)))
            or str(runtime.device) != options['device'] or runtime.rank != options.get('rank',16)
            or (runtime.MODEL_REV,runtime.T5_REV) != (MODEL_REV,T5_REV)
            or not runtime.stored_unconditional or not hasattr(runtime,'vae')
            or native_io.frozen_digest(runtime) != options['frozen_base_sha256']
            or any(p.dtype != torch.float32 for p in runtime.model.parameters())
            or not runtime.loras or any(type(layer) is not hndl.HNDLLoRALinear for layer in runtime.loras)
            or any(torch.count_nonzero(layer.up.weight).item() for layer in runtime.loras)):
        raise ValueError('exact fresh HNDL native scaffold/base/source/zeroUP required')

class ReceiptBackend:
    def __init__(self,backend,receipt,identity):
        self.backend=backend
        self.network,self.particles=backend.network,None
        self.receipt,self.sources=deepcopy(receipt),deepcopy(identity)
    def __getattr__(self,name):return getattr(self.backend,name)
    def metadata(self):
        return dict(self.backend.metadata(),fresh_initialization=dict(profile=PROFILE,
            receipt=deepcopy(self.receipt),sources=deepcopy(self.sources)),
            cached_only=True,teacher_parity_forward_performed=False,particle_bank=None)

class Training(ownership.Owner):
    def __init__(self,config,directory,*,mode,resume=None,runtime_factory=None,owner_admission=None,ema_decay=None):
        super().__init__()
        # Intent and typed saved metadata are resolved BEFORE invoking a runtime
        # factory, moving cached data, constructing a critic or any optimizer.
        if mode not in ('fresh','resume') or (mode=='fresh' and resume is not None) or (mode=='resume' and resume is None):
            raise ValueError('explicit fresh without state or strict resume with FULL required')
        if ema_decay is not None or config.options.get('ema_decay') is not None:
            raise ValueError('no weight EMA; raw current weights only')
        if config.backend != 'supra_hndl' or not isinstance(config.cohort,str) or not config.cohort.strip():
            raise ValueError('explicit original HNDL adapter-only config/cohort required')
        if not callable(runtime_factory) or not callable(owner_admission):
            raise ValueError('explicit runtime factory and resource-owner admission required')
        self.config=deepcopy(config.as_dict());options=deepcopy(config.options)
        self.construction=policy.Construction(mode,config.cohort,hashlib.sha256(canonical(options).encode()).hexdigest())
        packet=read_packet(resume) if mode=='resume' else None
        guard();self.sources=sources()
        if packet is not None:
            expected={'schema','profile','config','sources','initialization','game','ema_decay'}
            if (set(packet)!=expected or packet['schema']!=FULL_SCHEMA or packet['profile']!=PROFILE
                    or packet['config']!=self.config or packet['sources']!=self.sources or packet['ema_decay'] is not None):
                raise ValueError('only exact new-profile typed FULL; no old artifact migration')
            policy.validate_receipt(packet['initialization'],self.construction,options,self.sources)
        self.directory=Path(directory).absolute()
        if self.directory.exists():raise FileExistsError('new immutable owner directory required')
        rows,contexts,context_receipt=read_assets(options)
        request=dict(profile=PROFILE,operation=mode,cohort=config.cohort,options=deepcopy(options),
            teacher_forwards=0,initialization_calls=int(mode=='fresh'),particle_bank=None,ema_decay=None)
        if owner_admission(deepcopy(request)) is not True:
            raise ValueError('resource owner did not admit this explicit operation')
        device=torch.device(options['device']);devices=[device.index] if device.type=='cuda' else []
        try:
            with torch.random.fork_rng(devices=devices):
                seed_initialization_rng(options.get('seed',7),devices)
                self.runtime=runtime_factory(deepcopy(request))
                runtime_check(self.runtime,options,context_receipt)
                self.cleanup=ownership.snapshot(self.runtime)
                self.original_encode=self.runtime.encode
                self.runtime.cache={p:tuple(v.to(device) for v in pair)for p,pair in contexts.items()}
                admitted=self.runtime.cache
                def frozen_encode(prompt):
                    if prompt not in admitted:raise ValueError('outside original cached Supra contexts')
                    return admitted[prompt]
                self.runtime.encode=frozen_encode
                identity=hndl.model_identity(self.runtime)
                identity.update(context_receipt_sha256=file_sha(Path(options['contexts']).with_suffix('.json')),
                    training_sources={name:file_sha(Path(hndl_lifecycle.__file__).with_name(name))
                        for name in ('cache.py','campaign.py','hndl_lifecycle.py')})
                native=SupraRuntimeResponseAdapter(self.runtime,model_identity=identity)
                rows.device=device
                with torch.random.fork_rng(devices=devices):
                    seed_initialization_rng(options.get('seed',7),devices)
                    critic=HNDLFieldCritic(4,rows.condition_width,(32,32),options.get('critic_width',32)).to(device)
                self.initialization=(policy.initialize(self.runtime,critic,self.construction,options,self.sources)
                    if mode=='fresh' else deepcopy(packet['initialization']))
                self.backend=ReceiptBackend(native,self.initialization,self.sources)
                # Receipt already exists in actual backend metadata/signature.
                self.game=ConditionalPG095(self.backend,rows,critic,conditions=None,particles=None,
                    response_domain='continuous_field',response_scale=rows.response_scale(),preset='bispf_optimized',
                    batch_size=4,seed=options.get('seed',7),total_steps=options.get('steps',4),sampling='balanced')
                if self.game.prior is not None:raise RuntimeError('adapter-only Supra must not acquire a prior/bank')
                if packet is not None:
                    self.game.load_state_dict(packet['game'])
                    if not equal(self.game.state_dict(),packet['game']):raise RuntimeError('strict whole game/optimizers/streams/RNG restore differs')
                self.runtime.cfg_adapter_branch='conditional_only'
            self.parameter_flags=[(p,p.requires_grad)for p in self.runtime.model.parameters()]
            new_directory(self.directory)
            self.inherited_steps=self.game.completed_steps
            if mode=='fresh':
                write_json(self.directory/'initialization.json',self.initialization)
                self.save(self.directory/'FULL-000000.pt')
                self.export(self.directory/'RAW-000000.pt')
        except BaseException as primary:
            try:self.close()
            except BaseException as cleanup:raise BaseExceptionGroup('Supra construction/cleanup failed',[primary,cleanup])
            raise
    def update(self):
        self.require_open()
        if self.runtime.cfg_adapter_branch!='conditional_only' or any(p.requires_grad!=flag for p,flag in self.parameter_flags):
            raise RuntimeError('native branch/Parameter ownership changed')
        return self.game.step()
    def save(self,path):
        self.require_open()
        return save_packet(path,dict(schema=FULL_SCHEMA,profile=PROFILE,config=deepcopy(self.config),
            sources=deepcopy(self.sources),initialization=deepcopy(self.initialization),
            game=self.game.state_dict(),ema_decay=None))
    def export(self,path):
        self.require_open()
        state=self.game.export_state_dict()
        return save_packet(path,dict(schema=RAW_SCHEMA,profile=PROFILE,config=deepcopy(self.config),sources=deepcopy(self.sources),
            initialization=deepcopy(self.initialization),model_identity=hndl.model_identity(self.runtime),
            network=state['network'],completed_steps=self.game.completed_steps,selection='raw',ema_decay=None))
    def render(self,*,prompt,strength,seed,steps=50,cfg=3.,output_type='pil'):
        self.require_open()
        with observed(self,seed):
            return native_io.render(self.backend,self.runtime,prompt=prompt,strength=strength,seed=seed,steps=steps,cfg=cfg,output_type=output_type)

@contextmanager
def observation(owner):
    restore=ownership.snapshot(owner.runtime)
    owner.runtime.cache=dict(owner.runtime.cache)
    owner.runtime.encode=owner.original_encode
    try:
        with source_cohort(hndl.LOCK):yield
    finally:
        try:restore.close()
        except BaseException:
            # Retain the failed observation restorer before permanent close.
            owner.closing=True
            owner.cleanup.pending.insert(0,restore.close)
            raise

@contextmanager
def observed(owner,seed):
    from particle_sliders.lifecycle import _network_observation
    device=torch.device(owner.runtime.device)
    devices=[device.index]if device.type=='cuda'else[]
    with _network_observation(owner.backend.network,seed,devices=devices,observation=lambda:observation(owner)):
        yield
