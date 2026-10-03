"""Synthetic CPU wrapper qualification, not actual Supra data or science.

The TinyBackbone declarations below are copied unchanged from the already
qualified test_particle_final_precision.py fixture; no older test is collected
or executed. Exactly THREE real native updates occur in the last case:
fresh0->1->2 and a separate public saved1->2 replay. No fabricated clock12800,
full14-block host, teacher-cache generation, accuracy gate, or CUDA is used.
"""
from copy import deepcopy
import hashlib

import pytest
import torch
from torch import nn
from particlegan import init
from supra import particle_final_precision as precision
from supra import particle_latent_coverage as coverage
from supra.particle_pilot import state_digest

SOFTWARE_NATIVE_UPDATES = 3

class TinyOrdinaryLoRA(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Linear(16, 16)
        self.down = nn.Linear(16, 2, bias=False)
        self.up = nn.Linear(2, 16, bias=False)
        self.multiplier = 1.

    def forward(self, x):
        return self.base(x) + self.multiplier * self.up(self.down(x))

class TinyFinal(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(16, 16)

    def forward(self, x):
        return self.linear(x)

class TinyBackbone(nn.Module):
    """One routed site, native physical patch layout, no downloaded backbone."""
    def __init__(self):
        super().__init__()
        self.projection = TinyOrdinaryLoRA()
        self.final = TinyFinal()

    def forward(self, z, t, context, mask):
        batch = len(z)
        patches = z.reshape(batch, 4, 16, 2, 16, 2).permute(0, 2, 4, 1, 3, 5).reshape(batch, 256, 16)
        text = (context[:, :, :16] * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        hidden = (patches + .125 * text[:, None] + .05 * t[:, None, None]).tanh()
        output = self.final(self.projection(hidden))
        return output.reshape(batch, 16, 16, 4, 2, 2).permute(0, 3, 1, 4, 2, 5).reshape(batch, 4, 32, 32)


def _pool(seeds, *, test=False):
    contexts=[]; prompts=[]; targets=[]
    for source,target in zip(coverage.SOURCES,coverage.TARGETS):
        for j,_ in enumerate(seeds):
            for path in range(2):
                for k in range(10):
                    # Deliberately analytic software values, not claimed Gaussian
                    # trajectories or a substitute for the 1080-call builder.
                    z=torch.linspace(-.17,.23,4096)+.013*j+.003*path+.001*k+(1. if test else 0.)
                    contexts.append(torch.cat((z,torch.tensor([k/10.,source,target,1.]))))
                    prompts.append("synthetic-caption-"+str(source))
                    targets.append("synthetic-caption-"+str(target))
    n=len(contexts)
    return dict(context=torch.stack(contexts),base=torch.zeros(n,4,32,32),
        targets=torch.zeros(n,4,32,32),hold=torch.zeros(n,dtype=torch.bool),
        source_indices=torch.arange(n),prompts=prompts,target_prompts=targets)


def software_inputs():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(901)
        base=TinyBackbone()
        streams={name:torch.Generator(device="cpu").manual_seed(int.from_bytes(
            hashlib.sha256(("final-precision-tiny:"+name).encode()).digest()[:8],"little")%(2**63-1))
            for name,p in base.named_parameters() if p.requires_grad}
        init.initialize_(base,method="sample_distributions_v1",parameter_generators=streams)
        base.requires_grad_(False).eval()
    fit=_pool(coverage.BASE_SEEDS); guard=fit["context"][:64].clone()
    guard[:,0]+=2.; guard[-7:,4098]=guard[-7:,4097]
    holds=guard[-7:].clone()
    original=dict(fit=fit,test=_pool(coverage.TEST_SEEDS,test=True),
        guard=dict(context=guard,targets=torch.zeros(64,4,32,32)),holds=dict(context=holds),
        coordinate_scale=torch.ones(16),
        text_contexts=torch.linspace(-.13,.19,19*3*768).reshape(19,3,768),
        text_masks=torch.ones(19,3))
    expanded=deepcopy(original); expanded["fit"]=_pool(coverage.TRAIN_SEEDS)
    indices=[160*s+20*j+10*p+k for s in range(6) for j in range(2) for p in range(2) for k in range(10)]
    for key in ("context","base","targets","hold"):
        expanded["fit"][key][indices]=original["fit"][key]
    for key in ("prompts","target_prompts"):
        for new,old in zip(indices,original["fit"][key]):expanded["fit"][key][new]=old
    counts=dict(teacher_B4_forwards=0,Euler_B4_forwards=0,opposite_caption_B4_forwards=0,
        private_seed_draws=0,native_updates=0,optimizer_steps=0)
    manifest=dict(schema=coverage.SOFTWARE_DATA_SCHEMA,software_only=True,
        source_order=list(coverage.SOURCES),target_order=list(coverage.TARGETS),
        original_seeds=list(coverage.BASE_SEEDS),train_seeds=list(coverage.TRAIN_SEEDS),
        TEST_seeds=list(coverage.TEST_SEEDS),index_law="160*s+20*j+10*p+k",
        original_indices=indices,paths=["neutral","positive"],Euler_steps=0,
        recorded_steps=list(coverage.TIMES),cfg=3.,builder_batch=0,counts=counts,
        original_dataset_digest=state_digest(original),expanded_dataset_digest=state_digest(expanded),
        provenance={"scope":"synthetic CPU software fixture; no teacher generation or trained12800"})
    return base,original,dict(schema=coverage.SOFTWARE_DATA_SCHEMA,data=expanded,manifest=manifest)


def fresh_software(seeds=coverage.TRAIN_SEEDS):
    base,original,artifact=software_inputs()
    loop=precision.make_precision_training_loop(base,original,precision=precision.FP32_FINAL,
        device="cpu",probe_interval=1000)
    initial=precision.checkpoint_precision(loop)
    assert loop.policy.completed_steps==0
    precision.restore_precision(loop,initial)
    proof=dict(software_only=True,public_restore_exact=state_digest(
        precision.checkpoint_precision(loop))==state_digest(initial),completed_steps=0,
        initialized_after_restore=False,scope="fresh synthetic CPU0, not pretrained12800")
    coverage.attach_software_latent_coverage(loop,original,artifact,seeds=seeds,bootstrap_proof=proof)
    return loop,original,artifact,initial


@pytest.fixture(autouse=True)
def caller_scope():
    rng,threads=torch.get_rng_state().clone(),torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.autograd.set_multithreading_enabled(False):yield
    finally:
        torch.set_num_threads(threads);torch.set_rng_state(rng)


def test_synthetic_schema_rejects_production_and_corrupt_lineage_before_attach():
    loop,original,artifact,_=fresh_software()
    assert coverage.validate_software_latent_coverage_data(original,artifact)
    before=state_digest(coverage.checkpoint_latent_coverage(loop))
    with pytest.raises(ValueError):coverage.validate_latent_coverage_data(original,artifact)
    with pytest.raises(ValueError):coverage.attach_latent_coverage(loop,original,artifact,
        seeds=coverage.TRAIN_SEEDS,bootstrap_proof={})
    assert state_digest(coverage.checkpoint_latent_coverage(loop))==before
    for change in ("subset","guard","seeds","counts","dtype"):
        bad=deepcopy(artifact)
        if change=="subset":bad["data"]["fit"]["context"][0,0]+=.125
        elif change=="guard":bad["data"]["guard"]["context"][0,0]+=.125
        elif change=="seeds":bad["manifest"]["train_seeds"][-1]=39001
        elif change=="counts":bad["manifest"]["counts"]["teacher_B4_forwards"]=1080
        else:bad["data"]["fit"]["context"]=bad["data"]["fit"]["context"].double()
        # Recompute the envelope digest so schema/value checks, not a stale
        # checksum alone, must reject these destructive fixtures.
        bad["manifest"]["expanded_dataset_digest"]=state_digest(bad["data"])
        with pytest.raises(ValueError):coverage.validate_software_latent_coverage_data(original,bad)


def test_attachment_retains_native_owners_guards_scales_and_uniform_paired_addresses():
    base,original,artifact=software_inputs()
    states=[]
    for seeds in (coverage.BASE_SEEDS,coverage.TRAIN_SEEDS):
        loop=precision.make_precision_training_loop(base,original,precision=precision.FP32_FINAL,
            device="cpu",probe_interval=1000)
        origin=precision.checkpoint_precision(loop);precision.restore_precision(loop,origin)
        proof=dict(software_only=True,public_restore_exact=True,completed_steps=0,initialized_after_restore=False)
        tag=coverage.attach_software_latent_coverage(loop,original,artifact,seeds=seeds,bootstrap_proof=proof)
        after=precision.checkpoint_precision(loop)
        assert state_digest(origin["pilot"]["policy"])==state_digest(after["pilot"]["policy"])
        assert state_digest(origin["caller"])==state_digest(after["caller"])
        assert torch.equal(origin["pilot"]["data_rng"],after["pilot"]["data_rng"])
        assert torch.equal(origin["pilot"]["paired_noise_rng"],after["pilot"]["paired_noise_rng"])
        assert loop.config["dataset_digest"]==state_digest(original)
        assert torch.equal(loop.guard_context,original["guard"]["context"])
        assert torch.equal(loop.policy.D.scale,original["coordinate_scale"])
        assert tag["software_only"] and tag["origin_step"]==0
        assert tag["sampling"]["historical_sequence"]==coverage.SAMPLING["historical_sequence"]
        states.append(loop)
    common=torch.arange(960);s=common//160;j=(common%160)//20;offset=common%20
    baseline=40*s+20*(j//4)+offset
    assert torch.equal(torch.bincount(baseline,minlength=240),torch.full((240,),4))
    assert torch.equal(states[0].fit_context[baseline,4096:],states[1].fit_context[common,4096:])
    assert torch.equal(states[0].data_rng.get_state(),states[1].data_rng.get_state())
    assert torch.equal(states[0].paired_noise_rng.get_state(),states[1].paired_noise_rng.get_state())


def test_native_three_update_split_replay_and_wrong_state_rejection(tmp_path,monkeypatch):
    loop,_,_,_=fresh_software()
    first=coverage.coverage_update(loop)
    assert first["step"]==1 and first["coverage_rows"]==960
    assert first["batch_indices"]==first["coverage_addresses"]
    midpoint=coverage.checkpoint_latent_coverage(loop)
    path=tmp_path/"software-clock1.pt";torch.save(midpoint,path)
    row=coverage.coverage_update(loop)
    assert row["step"]==2
    expected=state_digest(coverage.checkpoint_latent_coverage(loop))
    resumed,_,_,_=fresh_software()
    def forbidden(*args,**kwargs):raise AssertionError("learned restore must never initialize")
    monkeypatch.setattr(init,"initialize_",forbidden)
    saved=torch.load(path,map_location="cpu",weights_only=False)
    before=state_digest(coverage.checkpoint_latent_coverage(resumed))
    for change in ("support","order","dtype","precision"):
        bad=deepcopy(saved)
        if change=="support":bad["tag"]["seeds"]=list(coverage.BASE_SEEDS)
        elif change=="order":bad["fit_context"]=bad["fit_context"].flip(0)
        elif change=="dtype":bad["fit_context"]=bad["fit_context"].double()
        else:bad["precision_checkpoint"]["precision"]=precision.BF16_FINAL
        with pytest.raises(ValueError):coverage.restore_latent_coverage(resumed,bad)
        assert state_digest(coverage.checkpoint_latent_coverage(resumed))==before
    coverage.restore_latent_coverage(resumed,saved)
    assert state_digest(coverage.checkpoint_latent_coverage(resumed))==state_digest(saved)
    replay=coverage.coverage_update(resumed)
    assert replay==row and replay["step"]==2
    assert state_digest(coverage.checkpoint_latent_coverage(resumed))==expected
    precision.assert_policy_precision(resumed.policy,precision.FP32_FINAL)
