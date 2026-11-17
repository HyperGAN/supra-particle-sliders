"""Content-bound native paired velocity fields and training-only edit scaling."""
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
import hashlib,json
from pathlib import Path

import torch
from particle_sliders.pg09_backends import PreparedResponse,ResponseRow

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

class Rows:
    response_shape=(1,4,32,32)
    def __init__(self,directory,device):
        self.directory=Path(directory).resolve();self.device=torch.device(device)
        self.index=json.loads((self.directory/'index.json').read_text())
        self.index_sha=sha(self.directory/'index.json');self.shards=OrderedDict()
        self._offsets=[]
        for s in self.index['shards']:
            path=(self.directory/s['path']).resolve()
            assert path.is_relative_to(self.directory) and sha(path)==s['sha256']
            assert s['count']==100
            self._offsets.extend((s,i) for i in range(100))
        assert len(self)==800
        self.condition_width=774
        self._identity=dict(schema='Supra-vibe-native-paired-fields-v1',index=deepcopy(self.index),
            index_sha256=self.index_sha,provider_sha256=sha(__file__),response_shape=list(self.response_shape),
            conditions='masked unit-RMS neutral T5 mean,time,scale1,strength1,guidance/3,positive_trajectory,hold0')

    def __len__(self):return len(self._offsets)
    def identity(self):
        assert sha(self.directory/'index.json')==self.index_sha
        return deepcopy(self._identity)
    def record(self,index):
        shard,position=self._offsets[index];key=shard['path']
        if key not in self.shards:
            assert sha(self.directory/key)==shard['sha256']
            self.shards[key]=torch.load(self.directory/key,map_location='cpu',weights_only=True)
            if len(self.shards)>2:self.shards.popitem(last=False)
        self.shards.move_to_end(key)
        data=self.shards[key];row=data['records'][position]
        assert row['position']==index%50
        for name in ('latent','neutral','positive'):
            v=row[name];assert tuple(v.shape)==self.response_shape and v.dtype==torch.float32 and not v.requires_grad and torch.isfinite(v).all()
        return row,data
    def __getitem__(self,index):
        r,d=self.record(index)
        condition=lambda side:tuple(v.to(self.device).clone() for v in d[side])
        row=ResponseRow(condition('neutral_context'),condition('positive_context'),condition('unconditional_context'),
            r['latent'].to(self.device).clone(),timestep=torch.tensor(r['timestep'],device=self.device),guidance=3.)
        return PreparedResponse(row,r['positive'].to(self.device).clone(),r['neutral'].to(self.device).clone())
    def conditions(self,indices):
        out=[]
        for index in indices.tolist():
            r,d=self.record(index);embedding,mask=[v.to(self.device) for v in d['neutral_context']]
            pooled=(embedding.float()*mask[...,None]).sum(1)/mask.sum(1,keepdim=True)
            pooled=pooled/pooled.square().mean().sqrt().clamp_min(1e-12)
            suffix=pooled.new_tensor([[r['timestep'],1.,1.,1.,float(r['trajectory']=='positive'),0.]])
            out.append(torch.cat((pooled,suffix),1))
        return torch.cat(out)
    def frozen_conditioning(self,runtime):
        contexts={}
        for index in range(0,len(self),100):
            _,d=self.record(index)
            for name,key in [('neutral','neutral_context'),('positive','positive_context')]:
                contexts[d['prompt'][name]]=tuple(v.to(runtime.device) for v in d[key])
            contexts['']=tuple(v.to(runtime.device) for v in d['unconditional_context'])
        runtime.cache=contexts
        def encode(prompt):
            if prompt not in contexts:raise ValueError('outside frozen conditioning; no implicit encoding')
            return contexts[prompt]
        runtime.encode=encode
        return len(contexts)

@dataclass
class Edit:
    native: PreparedResponse
    scale: torch.Tensor
    target: torch.Tensor
    neutral: torch.Tensor
    transform: str

class NormalizedRows:
    def __init__(self,raw,normalization):
        self.raw=raw;self.response_shape=raw.response_shape
        assert normalization['data_identity_sha256']==digest(raw.identity())
        self.scales=normalization['scale'].detach().clone()
        assert tuple(self.scales.shape)==(50,4096) and torch.isfinite(self.scales).all() and (self.scales>0).all()
        self._identity=dict(schema='Supra-vibe-train-only-normalized-edit-v1',raw=raw.identity(),
            scale_sha256=hashlib.sha256(self.scales.numpy().tobytes()).hexdigest(),
            generated='(student-neutral)/train_scale[position]',target='(positive-neutral)/train_scale[position]',
            native_export_and_render='untransformed',ema_decay=None)
        self.transform=digest(self._identity)
    def __len__(self):return len(self.raw)
    def identity(self):
        assert self.raw.identity()==self._identity['raw']
        return deepcopy(self._identity)
    def conditions(self,indices):return self.raw.conditions(indices)
    def __getitem__(self,index):
        native=self.raw[index];scale=self.scales[index%50].reshape(self.response_shape).to(native.target.device)
        target=((native.target.float()-native.neutral.float())/scale).detach()
        return Edit(native,scale,target,torch.zeros_like(target),self.transform)

class NormalizedBackend:
    def __init__(self,backend,rows):
        self.backend,self.rows=backend,rows;self.network=backend.network;self.runtime=backend.runtime
    def metadata(self):return dict(self.backend.metadata(),training_transform=self.rows.identity())
    def generated(self,prepared):
        assert prepared.transform==self.rows.transform
        return (self.backend.generated(prepared.native)-prepared.native.neutral)/prepared.scale

def fit_normalization(rows,check):
    moments={};values=[[] for _ in range(50)]
    for i in range(len(rows)):
        check();r,_=rows.record(i);position=i%50;edit=(r['positive']-r['neutral']).flatten()
        n,mean,m2=moments.setdefault(position,[0,torch.zeros_like(edit),torch.zeros_like(edit)])
        n+=1;delta=edit-mean;mean+=delta/n;m2+=delta*(edit-mean);moments[position][0]=n
    scales=[(moments[p][2]/(moments[p][0]-1)).clamp_min(0).sqrt().clamp_min(1e-4) for p in range(50)]
    for i in range(len(rows)):
        check();r,_=rows.record(i);v=(r['positive']-r['neutral']).flatten()/scales[i%50]
        values[i%50].append(v.square().mean().sqrt())
    scales=torch.stack([s*torch.stack(v).median().clamp_min(1e-4) for s,v in zip(scales,values)])
    return dict(scale=scales,data_identity_sha256=digest(rows.identity()),
        method='FP32 Welford sample std per position/coordinate, floor1e-4, median normalized-edit RMS gain',
        split='train',teacher_records=len(rows),historical_scale_reused=False)
