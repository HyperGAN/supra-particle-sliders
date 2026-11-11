"""RAW-only native inference; no campaign data, critic, game or initializer."""
from copy import deepcopy
import hashlib
import torch
from supra import hndl, native_io
from particle_sliders.pg09_backends import SupraRuntimeResponseAdapter
from . import PROFILE
from . import ownership, policy
from .util import canonical, read_packet, equal
from .training import RAW_SCHEMA, guard, sources, observed

def finite(value):
    if isinstance(value,torch.Tensor):
        return bool(torch.isfinite(value).all()) if value.is_floating_point() or value.is_complex() else True
    if isinstance(value,dict):return all(finite(item)for item in value.values())
    if isinstance(value,(list,tuple)):return all(finite(item)for item in value)
    return type(value)in(bool,int,float,str,type(None))

class Inference(ownership.Owner):
    def __init__(self,raw,*,runtime_factory=None,owner_admission=None,ema_decay=None):
        super().__init__()
        if ema_decay is not None:raise ValueError('only raw current weights, no EMA')
        if not callable(runtime_factory) or not callable(owner_admission):
            raise ValueError('explicit runtime factory and resource-owner admission required')
        packet=read_packet(raw)
        keys={'schema','profile','config','sources','initialization','model_identity','network','completed_steps','selection','ema_decay'}
        if (set(packet)!=keys or packet['schema']!=RAW_SCHEMA or packet['profile']!=PROFILE
                or packet['selection']!='raw' or packet['ema_decay'] is not None
                or type(packet['completed_steps'])is not int or packet['completed_steps']<0 or not finite(packet['network'])):
            raise ValueError('finite exact new-profile RAW required; no FULL/legacy/EMA migration')
        guard();identity=sources()
        config=packet['config'];options=config['options']
        if config['backend']!='supra_hndl' or packet['sources']!=identity:
            raise ValueError('exact new cached HNDL Source identity required')
        construction=policy.Construction('resume',config['cohort'],hashlib.sha256(canonical(options).encode()).hexdigest())
        policy.validate_receipt(packet['initialization'],construction,options,identity)
        request=dict(profile=PROFILE,operation='inference',cohort=config['cohort'],
            options=dict(device=options['device'],rank=options.get('rank',16),base_precision='float32',
                         frozen_base_sha256=packet['model_identity']['frozen_parameter_sha256']),
            teacher_forwards=0,initialization_calls=0,particle_bank=None,ema_decay=None)
        if owner_admission(deepcopy(request))is not True:raise ValueError('resource owner did not admit RAW inference')
        device=torch.device(options['device']);devices=[device.index]if device.type=='cuda'else[]
        try:
            with torch.random.fork_rng(devices=devices):
                self.runtime=runtime_factory(deepcopy(request))
                if (str(self.runtime.device)!=options['device'] or self.runtime.rank!=options.get('rank',16)
                        or not self.runtime.stored_unconditional or not self.runtime.loras
                        or any(type(layer)is not hndl.HNDLLoRALinear for layer in self.runtime.loras)
                        or any(p.dtype!=torch.float32 for p in self.runtime.model.parameters())
                        or hndl.model_identity(self.runtime)!=packet['model_identity']):
                    raise ValueError('exact RAW original native HNDL model/source/precision required')
                self.cleanup=ownership.snapshot(self.runtime)
                self.original_encode=self.runtime.encode
                self.backend=SupraRuntimeResponseAdapter(self.runtime,model_identity=packet['model_identity'])
                self.backend.network.load_state_dict(packet['network'],strict=True)
                if not equal(self.backend.network.state_dict(),packet['network']):
                    raise RuntimeError('strict RAW network restoration differs')
                self.runtime.cfg_adapter_branch='conditional_only'
            self.completed_steps=packet['completed_steps']
            self.initialization=deepcopy(packet['initialization'])
            self.selection='raw'
        except BaseException as primary:
            try:self.close()
            except BaseException as cleanup:raise BaseExceptionGroup('Supra RAW construction/cleanup failed',[primary,cleanup])
            raise
    def render(self,*,prompt,strength,seed,steps=50,cfg=3.,output_type='pil'):
        self.require_open()
        with observed(self,seed):
            return native_io.render(self.backend,self.runtime,prompt=prompt,strength=strength,seed=seed,
                                   steps=steps,cfg=cfg,output_type=output_type)
    def response(self,row,*,strength=1.,seed=0):
        self.require_open()
        # Diagnostic native field; original strength0/base and CFG are untouched.
        with observed(self,seed),torch.no_grad():
            return self.backend._response(row,scale=strength).detach().clone()
