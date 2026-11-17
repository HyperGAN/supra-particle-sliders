"""Unchanged native DiT/LoRA graph, conditional CFG and positive Euler law."""
from contextlib import nullcontext
import hashlib
from pathlib import Path

import torch
from PIL import Image
from particle_sliders.pg09_backends import AdapterParameters,FrozenResponseAdapter,ResponseRow
from particle_sliders.fibo_hndl import verified_builder
from supra import native_model
from supra.hndl import HNDLLoRALinear
from supra.runtime import SupraRuntime,MODEL_ID,MODEL_REV,T5_REV,VAE_REV
from . import PROFILE


class NativeHNDLLoRALinear(HNDLLoRALinear):
    """Native CUDA graph contract and FP32 delta, cast to native base output."""
    def __init__(self,original,**kwargs):
        super().__init__(original,**kwargs)
        ids=[id(p) for p in self._graph.parameters()]
        self._graph.to(device=original.down.weight.device,dtype=torch.float32)
        assert ids==[id(p) for p in self._graph.parameters()]

    def forward(self,value):
        base=self.base(value)
        with torch.autocast(device_type=value.device.type,enabled=False):
            delta=self._graph(value=value.float().reshape(-1,1,value.shape[-1]))
        delta=delta.reshape(*value.shape[:-1],self.base.out_features)
        return base+(self.multiplier*self.scale*delta).to(base.dtype)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def frozen_digest(runtime):
    from supra.native_io import frozen_digest as original
    return original(runtime)


def load_runtime(*,device='cuda:0',seed=1):
    if not device.startswith('cuda:'):
        raise ValueError('native model math requires CUDA')
    builder,identity=verified_builder()
    with torch.random.fork_rng(devices=[torch.device(device).index]):
        torch.manual_seed(seed)
        runtime=SupraRuntime(device,rank=16,allow_hub=False,native_model=native_model)
        before=frozen_digest(runtime)
        originals=list(runtime.loras)
        found=[(n,m) for n,m in runtime.model.named_modules() if type(m) is native_model.LoRALinear]
        assert {id(m) for _,m in found}=={id(m) for m in originals}
        replacements={id(m):NativeHNDLLoRALinear(m,builder=builder,identity=identity) for _,m in found}
        for name,original in found:
            parent,attr=native_model._module_by_path(runtime.model,name)
            replacement=replacements[id(original)]
            if attr.isdigit():parent[int(attr)]=replacement
            else:setattr(parent,attr,replacement)
        runtime.loras=[replacements[id(m)] for m in originals]
        assert before==frozen_digest(runtime),'pretrained weights changed during fresh scaffold construction'
    runtime.vibe_identity=dict(profile=PROFILE,model=MODEL_ID,model_revision=MODEL_REV,
        text_revision=T5_REV,vae_revision=VAE_REV,checkpoint_sha256=sha(runtime.checkpoint),
        frozen_parameter_sha256=before,rank=16,alpha=16,base_precision='float32',
        native_sources={n:sha(Path(native_model.__file__).with_name(n)) for n in ('runtime.py','native_model.py','hndl.py','supra_lora.hndl')},
        builder=identity,weight_selection='raw',ema_decay=None)
    runtime.vibe_identity['adapter_precision']='FP32 HNDL graph with residual cast to native base output dtype'
    return runtime


def load_teacher_runtime(context_directory,*,device='cuda:0',seed=1):
    """Whole native DiT on CUDA; completed conditioning replaces unused encoders."""
    from huggingface_hub import hf_hub_download
    from collections import OrderedDict
    if not device.startswith('cuda:'):raise ValueError('CUDA required')
    builder,identity=verified_builder()
    checkpoint=hf_hub_download(MODEL_ID,'model_final_ema.pt',revision=MODEL_REV,local_files_only=True)
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    runtime=SupraRuntime.__new__(SupraRuntime)
    runtime.device=torch.device(device);runtime.rank=16;runtime.cfg_adapter_branch='both'
    runtime.checkpoint=checkpoint;runtime.ctx_len=int(state.get('config',{}).get('ctx_len',128))
    with torch.random.fork_rng(devices=[runtime.device.index]):
        torch.manual_seed(seed)
        runtime.model=native_model.SupraDiT();native_model.load_supra_weights(runtime.model,state,strict=True)
        runtime.base_params=sum(p.numel() for p in runtime.model.parameters())
        runtime.loras=native_model.attach_supra_lora(runtime.model,rank=16,alpha=16,targets=__import__('supra.runtime',fromlist=['TARGETS']).TARGETS)
        runtime.model.to(runtime.device).eval()
        before=frozen_digest(runtime)
        found=[(n,m) for n,m in runtime.model.named_modules() if type(m) is native_model.LoRALinear]
        replacements={id(m):NativeHNDLLoRALinear(m,builder=builder,identity=identity) for _,m in found}
        originals=runtime.loras
        for name,original in found:
            parent,attr=native_model._module_by_path(runtime.model,name)
            if attr.isdigit():parent[int(attr)]=replacements[id(original)]
            else:setattr(parent,attr,replacements[id(original)])
        runtime.loras=[replacements[id(m)] for m in originals]
        assert before==frozen_digest(runtime)
    del state
    runtime.cache={};runtime.token_counts={};runtime.stored_unconditional=True
    for path in Path(context_directory).glob('*.pt'):
        data=torch.load(path,map_location='cpu',weights_only=True)
        assert data['checkpoint_sha256']==sha(checkpoint) and data['text_revision']==T5_REV
        runtime.cache[data['prompt']]=tuple(v.to(runtime.device) for v in data['context'])
    assert '' in runtime.cache
    def encode(prompt):
        if prompt not in runtime.cache:raise ValueError('outside completed frozen conditioning')
        return runtime.cache[prompt]
    runtime.encode=encode
    runtime.vibe_identity=dict(profile=PROFILE,model=MODEL_ID,model_revision=MODEL_REV,text_revision=T5_REV,vae_revision=VAE_REV,
        checkpoint_sha256=sha(checkpoint),frozen_parameter_sha256=before,rank=16,alpha=16,
        base_precision='float32',builder=identity,weight_selection='raw',ema_decay=None,
        adapter_precision='FP32 HNDL graph; residual cast to native output',
        residency='whole original DiT CUDA; completed contexts; unused encoder/decoder not loaded')
    return runtime


class Backend(FrozenResponseAdapter):
    family='supra_runtime'
    time_coordinates='native flow time i/50, noise0 to data1; positive Euler velocity/50'
    def __init__(self,runtime):
        self.runtime=runtime
        network=AdapterParameters(((n,p) for n,p in runtime.model.named_parameters() if p.requires_grad),runtime.scale)
        super().__init__(network,decoder=self.decode,model_identity=runtime.vibe_identity)

    def _velocity(self,condition,latent,timestep):
        assert not torch.any((timestep<0)|(timestep>1))
        embedding,mask=self.runtime.encode(condition) if isinstance(condition,str) else condition
        context=torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False) if latent.device.type=='cuda' else nullcontext()
        with context:
            return self.runtime.model(latent,timestep.reshape(-1).to(latent.device,torch.float32),
                embedding.to(latent.device),mask.to(latent.device)).float()

    def decode(self,latent):
        with torch.autocast('cuda',dtype=torch.bfloat16):
            return self.runtime.vae.decode(latent/.18215).sample

    def metadata(self):
        return dict(super().metadata(),profile=PROFILE,particle_bank=None,
            sampler='native positive Euler, velocity/50,50 steps',ema_decay=None)


@torch.no_grad()
def sample_native(backend,prompt,seed,*,strength):
    runtime=backend.runtime
    latent=runtime.noise(seed)
    row=ResponseRow(prompt,prompt,'',latent,guidance=3.)
    for i in range(50):
        timestep=latent.new_tensor(i/50)
        latent=latent+backend.cfg(row,latent,timestep,scale=strength)/50
    return latent

@torch.no_grad()
def render_native(backend,prompt,seed,*,strength):
    decoded=backend.decode(sample_native(backend,prompt,seed,strength=strength))
    pixels=((decoded.float().clamp(-1,1)+1)*127.5).round().byte()[0]
    return Image.fromarray(pixels.permute(1,2,0).cpu().numpy())


def close_runtime(runtime):
    import gc
    runtime.cache.clear()
    runtime.loras.clear()
    for key in ('model','text_encoder','vae'):
        if hasattr(runtime,key):delattr(runtime,key)
    gc.collect()
    torch.cuda.empty_cache()
