"""FOUR NEW authored/UNRUN cases; tiny random native model, synthetic cache.

Only the live19-context DATA reader is replaced for this explicit CPU fixture.
Original HNDL/core/PG guards, original HNDL graph/critic and shared game execute.
No pretrained runtime, real teacher, encoder, old node or old cohort is used.
"""
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import hashlib, inspect, json, math
import pytest
import torch
from particle_sliders.lifecycle import SessionConfig
from supra import cache, campaign, hndl, hndl_lifecycle, native_io, native_model
from supra.runtime import MODEL_REV, T5_REV
from supra.shared import file_sha
from supra_fresh95 import open_training, open_inference, Checkpoint
from supra_fresh95 import training, policy, ownership
from supra_fresh95.util import equal, read_packet

class TinyVAE(torch.nn.Module):
    def decode(self,latent):return SimpleNamespace(sample=latent[:,:3])

class TinyRuntime:
    MODEL_REV, T5_REV = MODEL_REV, T5_REV
    def __init__(self):
        self.device=torch.device('cpu');self.rank=2
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(123)
            self.model=native_model.SupraDiT(d_model=8,depth=1,n_heads=1,ctx_dim=4,
                mlp_ratio=2.,num_tokens=256)
            self.loras=native_model.attach_supra_lora(self.model,2,2.,('cross_attn.q',))
            hndl.bind_native_loras(self)
        self.cache={};self.token_counts={};self.stored_unconditional=True
        self.cfg_adapter_branch='both';self.text_encoder=None
        self.vae=TinyVAE()
        for index,prompt in enumerate(('', 'n','p','hold')):
            self.cache[prompt]=(torch.full((1,3,4),1.+index/10),torch.ones(1,3))
    def parameters(self):return [p for p in self.model.parameters()if p.requires_grad]
    def encode(self,prompt):
        if prompt not in self.cache:raise AssertionError('software fixture must never encode new text')
        return self.cache[prompt]
    @contextmanager
    def scale(self,value):
        old=[layer.multiplier for layer in self.loras]
        for layer in self.loras:layer.multiplier=float(value)
        try:yield
        finally:
            for layer,saved in zip(self.loras,old):layer.multiplier=saved

@pytest.fixture
def staged(tmp_path,monkeypatch):
    runtime=TinyRuntime()
    native_config=dict(rows=[dict(name='row',neutral='n',positive='p')],preservation=['hold'],verification=[])
    z=torch.linspace(-.3,.3,4096).reshape(1,4,32,32)
    record=dict(row='row',prompt='n',t=.25,z=z,neutral=torch.zeros_like(z),target=torch.full_like(z,.2))
    path=tmp_path/'synthetic-prepared.pt'
    torch.save(dict(records=[dict(record,context='neutral'),dict(record,context='positive')],
        holds=[dict(prompt='hold',t=.5,z=z,neutral=torch.zeros_like(z),target=torch.zeros_like(z))]),path)
    context_path=tmp_path/'fixture-contexts.source';context_path.write_text('explicit synthetic prepared contexts only\n')
    context_receipt=dict(schema='Supra-NEW-synthetic-context-provenance-v2',
        runtime_source_sha256=file_sha(inspect.getsourcefile(TinyRuntime)),
        synthetic_contexts_only=True,context_prompts=sorted(runtime.cache),real_teacher_calls=0)
    context_path.with_suffix('.json').write_text(json.dumps(context_receipt,sort_keys=True)+'\n')
    options=dict(native_config=native_config,cache=str(path),cache_sha256=file_sha(path),
        contexts=str(context_path),contexts_sha256=file_sha(context_path),frozen_base_sha256=native_io.frozen_digest(runtime),
        device='cpu',rank=2,steps=3,batch=4,seed=7,critic_width=4,ema_decay=None)
    def fixture_assets(actual):
        assert actual==options
        rows=cache.SupraCampaignRows(path,cache_sha256=actual['cache_sha256'],config=native_config,contexts=runtime.cache)
        assert json.loads(context_path.with_suffix('.json').read_text())==context_receipt
        return rows,deepcopy(runtime.cache),deepcopy(context_receipt)
    monkeypatch.setattr(training,'read_assets',fixture_assets)
    def forbidden(*args,**kwargs):raise AssertionError('no model loading or real teachers in NEW fixtures')
    monkeypatch.setattr(hndl,'load_native',forbidden)
    monkeypatch.setattr(hndl_lifecycle,'prepare',forbidden)
    monkeypatch.setattr(campaign,'teacher_parity',forbidden)
    return SessionConfig('supra_hndl','NEW-synthetic-Supra-cached-fresh95-v1',**options),tmp_path

def factory(request):
    assert request['operation']in('fresh','resume','inference')and request['particle_bank']is None
    return TinyRuntime()

def open_new(staged,name='fresh',**kwargs):
    config,root=staged
    return open_training(config,root/name,mode='fresh',runtime_factory=factory,owner_admission=lambda request:True,**kwargs)

def test_new_fresh_policy_precedes_optimizers_and_preserves_HNDL_identities(staged,monkeypatch):
    real=training.ConditionalPG095;seen=[];original=policy.initialize
    def initialize(runtime,critic,*args,**kwargs):
        before={id(p)for p in runtime.parameters()}|{id(p)for p in critic.parameters()}
        critic_before=deepcopy(critic.state_dict())
        receipt=original(runtime,critic,*args,**kwargs)
        assert before=={id(p)for p in runtime.parameters()}|{id(p)for p in critic.parameters()}
        assert equal(critic_before,critic.state_dict())
        assert all(torch.count_nonzero(layer.up.weight)==0 for layer in runtime.loras)
        seen.append(receipt);return receipt
    def game(backend,*args,**kwargs):
        assert seen and backend.metadata()['fresh_initialization']['receipt']==seen[-1]
        assert kwargs['particles']is None
        return real(backend,*args,**kwargs)
    monkeypatch.setattr(policy,'initialize',initialize);monkeypatch.setattr(training,'ConditionalPG095',game)
    with open_new(staged)as owner:
        assert owner.game.prior is None and owner.game.completed_steps==0
        assert owner.game.state_dict()['config']['backend']['fresh_initialization']['receipt']==seen[0]
        assert (owner.directory/'FULL-000000.pt').is_file()and(owner.directory/'RAW-000000.pt').is_file()
        assert all(row['role']!='particle-bank'for row in owner.initialization['parameters'])

def test_new_deterministic_host_scale_KEEP_and_caller_RNG(staged):
    before=torch.get_rng_state().clone()
    with open_new(staged,'a')as a:
        first=deepcopy(a.game.network.state_dict());receipt=deepcopy(a.initialization)
        for layer in a.runtime.loras:
            down=layer.down.weight.detach();rows,columns=down.shape
            # Official QR preserves the declared Uniform RMS/Gram, not support.
            variance=1/(3*layer.down.in_features)
            gram=down@down.T if rows<=columns else down.T@down
            expected=torch.eye(min(rows,columns),device=down.device,dtype=down.dtype)*(variance*max(rows,columns))
            assert torch.allclose(gram,expected,rtol=1e-5,atol=1e-6)
            assert math.isclose(float(down.square().mean()),variance,rel_tol=1e-5,abs_tol=1e-7)
            assert torch.count_nonzero(layer.up.weight)==0
    assert torch.equal(before,torch.get_rng_state())
    with open_new(staged,'b')as b:
        assert equal(first,b.game.network.state_dict())and receipt==b.initialization
        assert all(row['before_sha256']==row['after_sha256']for row in receipt['parameters']if row['role']!='host-a-sqrt5-down-QR')
    assert torch.equal(before,torch.get_rng_state())

def test_new_strict_FULL_optimizer_RNG_resume_and_RAW_infer_skip_init(staged,monkeypatch):
    config,root=staged
    with open_new(staged)as original:
        original.update()  # ONE genuinely new tiny random-model software update.
        full=original.save(root/'saved-FULL1.pt');raw=original.export(root/'saved-RAW1.pt')
        state=deepcopy(original.game.state_dict());network=deepcopy(original.game.network.state_dict())
        row=original.game.rows[0].row
    def forbidden(*args,**kwargs):raise AssertionError('resume/inference initializer or teacher/game replay')
    monkeypatch.setattr(policy,'initialize',forbidden)
    with open_training(config,root/'resume',mode='resume',resume=full,runtime_factory=factory,owner_admission=lambda request:True)as resumed:
        assert resumed.inherited_steps==1 and equal(state,resumed.game.state_dict())
    monkeypatch.setattr(training,'read_assets',forbidden);monkeypatch.setattr(training,'ConditionalPG095',forbidden)
    from supra_fresh95 import inference
    with open_inference(raw,runtime_factory=factory,owner_admission=lambda request:True)as loaded:
        assert not hasattr(loaded,'game')and equal(network,loaded.backend.network.state_dict())
        before=torch.get_rng_state().clone();response=loaded.response(row,strength=1.,seed=9)
        assert response.shape==(1,4,32,32)and torch.isfinite(response).all()
        assert torch.equal(before,torch.get_rng_state())

def test_new_early_intent_legacy_EMA_refusal_and_pending_cleanup_retry(staged,monkeypatch):
    config,root=staged;calls=[]
    def forbidden(request):calls.append(request);raise AssertionError('invalid request reached runtime factory')
    for mode,resume,ema in(('fresh',Checkpoint('never',1,'0'*64),None),('resume',None,None),('fresh',None,.9)):
        with pytest.raises(ValueError):
            open_training(config,root/'invalid',mode=mode,resume=resume,ema_decay=ema,runtime_factory=forbidden,owner_admission=lambda request:True)
    legacy=root/'legacy.pt';torch.save(dict(schema='old-original-Supra-FULL'),legacy)
    from supra_fresh95.util import file_pin
    with pytest.raises(ValueError):
        open_training(config,root/'legacy',mode='resume',resume=file_pin(legacy),runtime_factory=forbidden,owner_admission=lambda request:True)
    assert not calls
    with open_new(staged,'cleanup')as owner:
        complete=[];attempts=[]
        def once():
            attempts.append(1)
            if len(attempts)==1:raise RuntimeError('deliberate pending callback restoration')
        owner.cleanup.pending[:0]=[lambda:complete.append('one-success'),once]
        with pytest.raises(RuntimeError,match='pending callback'):owner.close()
        assert owner.closing and not owner.closed and len(owner.cleanup.pending)==1
        with pytest.raises(RuntimeError,match='cleanup'):owner.update()
        owner.close();assert owner.closed and complete==['one-success']and len(attempts)==2
