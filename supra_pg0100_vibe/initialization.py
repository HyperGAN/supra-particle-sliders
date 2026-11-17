"""Official scoped down-QR; actual zero UP and critic constructor values KEEP."""
import hashlib,math
import torch
from torch import nn
from particlegan import init as pg_init
from .native import frozen_digest

class Declared(nn.Module):
    def __init__(self,rows):
        super().__init__();self.specs={}
        for i,row in enumerate(rows):
            self.register_parameter(f'p{i}',row['p']);self.specs[f'p{i}']=row['distribution']
pg_init.register(Declared,lambda m:m.specs)

def tensor_sha(p):
    return hashlib.sha256(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

def initialize(runtime,critic,*,seed,cohort):
    before_base=frozen_digest(runtime);rows=[]
    for i,layer in enumerate(runtime.loras):
        assert torch.count_nonzero(layer.up.weight)==0
        for role,leaf in [('down',layer.down),('up',layer.up)]:
            bound=1/math.sqrt(leaf.in_features)
            rows.append(dict(name=f'adapter.{i}.{role}.weight',p=leaf.weight,
                distribution=pg_init.Uniform(-bound,bound) if role=='down' else pg_init.KEEP))
    rows.extend(dict(name='critic.'+n,p=p,distribution=pg_init.KEEP) for n,p in critic.named_parameters())
    ids=[id(r['p']) for r in rows];assert len(set(ids))==len(ids)
    assert all(not hasattr(r['p'],'_supra_vibe_initialization_sha') for r in rows),'initialize fresh owned components only once'
    assert set(ids)=={id(p) for p in runtime.model.parameters() if p.requires_grad}|{id(p) for p in critic.parameters()}
    before=[tensor_sha(r['p']) for r in rows]
    seeds={f'p{i}':int.from_bytes(hashlib.sha256(f'Supra-vibe-EMPF:{seed}:{r["name"]}'.encode()).digest()[:8],'big')%2**63 for i,r in enumerate(rows)}
    pg_init.deterministic_orthogonal_(Declared(rows),strict=True,parameter_seeds=seeds)
    assert ids==[id(r['p']) for r in rows] and before_base==frozen_digest(runtime)
    parameters=[]
    for i,r in enumerate(rows):
        after=tensor_sha(r['p'])
        if r['distribution'] is pg_init.KEEP:assert before[i]==after
        parameters.append(dict(name=r['name'],before_sha256=before[i],after_sha256=after,seed=seeds[f'p{i}'],distribution=repr(r['distribution'])))
    assert all(torch.count_nonzero(layer.up.weight)==0 for layer in runtime.loras)
    receipt=dict(schema='Supra-vibe-EMPF-owned-downQR-otherKEEP-v1',cohort=cohort,seed=seed,
        completed_steps=0,restored_from=None,pretrained_unchanged=True,parameter_identity_preserved=True,
        protected_zero_UP=True,critic_policy='constructor-KEEP',particle_bank=None,ema_decay=None,
        initializer_sha256=hashlib.sha256(open(pg_init.__file__,'rb').read()).hexdigest(),parameters=parameters)
    marker=hashlib.sha256(str(receipt).encode()).hexdigest()
    for row in rows:setattr(row['p'],'_supra_vibe_initialization_sha',marker)
    return receipt
