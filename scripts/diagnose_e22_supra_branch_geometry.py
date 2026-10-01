#!/usr/bin/env python3
"""Read-only fresh/trained native branch activation geometry, with API init."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from particlegan import get_recipe,init
from supra.runtime import model_module,TARGETS
from supra.particle_adapter import SupraParticleHost,SupraParticleRouter,adapter_sites,adapter_input_dims
from supra.particle_export import CleanParticleAdapter,load_particle_adapter
from supra.particle_training_data import FrozenSliderContexts
from diagnose_e22_supra_noise_gap import ordinary_velocity


def emit(**row):
    print(json.dumps(row,default=str),flush=True)


def rms(value):
    return float(value.float().square().mean().sqrt())


def features(value):
    value = value.float().reshape(-1,value.shape[-1])
    centered = value-value.mean(0)
    covariance = centered.T@centered/len(value)
    trace = covariance.trace()
    return dict(rms=rms(value),centered_rms=rms(centered),
                covariance_effective_rank=float(trace.square()/covariance.square().sum().clamp_min(1e-30)))


@torch.no_grad()
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',type=Path,default=ROOT/'outputs/e22-full-f459cb6d-6400')
    ap.add_argument('--source',type=Path,default=ROOT/'outputs/e22-convergence-gap/native-noise/residuals.pt')
    ap.add_argument('--output',type=Path,default=ROOT/'outputs/e22-convergence-gap/branch-geometry')
    ap.add_argument('--device',default='cuda:0')
    args=ap.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        ap.error('choose a fresh output')
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    started=time.perf_counter()
    device=torch.device(args.device)
    data=torch.load(args.run/'data.pt',map_location='cpu',weights_only=False)
    saved=torch.load(args.run/'final.pt',map_location='cpu',weights_only=False,mmap=True)
    previous=torch.load(args.source,map_location='cpu',weights_only=False)
    state=saved['policy']
    mod=model_module()
    # Original runtime constructs the DiT then attaches LoRA after CPU seed 7.
    # Loading frozen checkpoint values consumes no RNG. Preserve the original
    # constructor's fresh down matrices, rather than its trained old adapter.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        base=mod.SupraDiT()
        mod.attach_supra_lora(base,rank=16,alpha=16,targets=TARGETS)
    teacher_state={k.removeprefix('teacher.'):v for k,v in state['models']['encoder'].items()
                   if k.startswith('teacher.') and not k.endswith(('.down.weight','.up.weight'))}
    missing=base.load_state_dict(teacher_state,strict=False,assign=True)
    assert not missing.unexpected_keys
    assert len(missing.missing_keys)==142 and all(k.endswith(('.down.weight','.up.weight')) for k in missing.missing_keys)
    base.to(device).eval().requires_grad_(False)
    sites=adapter_sites(base)
    encoder=FrozenSliderContexts(data['text_contexts'],data['text_masks'],base).to(device)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        fresh_host=SupraParticleHost(base,sites=sites).to(device)
        fresh_router=SupraParticleRouter(site_input_dims=adapter_input_dims(base,sites)).to(device)
        init.deterministic_orthogonal_(fresh_host,seed=0)
        init.deterministic_orthogonal_(fresh_router,seed=3)
        fresh_host.zero_particle_outputs()
        fresh_table=init.deterministic_orthogonal_(get_recipe('e22_routed',num_particles=128,z_dim=4).make_prior()).z.to(device)
    fresh=CleanParticleAdapter(fresh_host,fresh_router,fresh_table,dict(served_source='fast'))
    trained=load_particle_adapter(base,args.run/'final-boss-particlegan.safetensors',device=device)
    report=dict(metadata=dict(checkpoint_step=6400,particle_gan=str(Path(sys.modules['particlegan'].__file__).resolve()),
        precision='native CUDA BF16 host; FP32 particle branches',evaluation_only=True,
        fresh_particle_initialization='public deterministic_orthogonal_: full generator=0, router=3, R2 table; zero up',
        fresh_old_initialization='original DiT + rank16 LoRALinear construction under CPU seed7; zero up',
        saturation_threshold='abs(bridge preactivation)>3; tanh derivative below .00987',
        interpretation='activation conditioning, not an accuracy comparison or optimizer intervention'),groups={})
    for group in ('fit','holds'):
        context=previous['groups'][group]['context'].to(device)
        inputs=encoder.unpack(context)
        group_report={}
        old_records={}
        handles=[]
        for site in sites:
            branch=base.get_submodule(site)
            def old_hook(_,inputs,value,site=site):
                old_records[site]=dict(input_rms=rms(inputs[0]),features=features(value))
            handles.append(branch.down.register_forward_hook(old_hook))
        old_velocity=ordinary_velocity(base,inputs)
        for handle in handles:handle.remove()
        group_report['fresh_original_lora']=dict(sites=old_records)
        for arm,adapter in (('fresh_particle',fresh),('trained_particle',trained)):
            records={}
            handles=[]
            for site,branch in zip(sites,adapter.generator.particle_branches()):
                def particle_hook(module,inputs,value,site=site,rank=adapter.generator.rank):
                    joined=inputs[0].float()
                    hidden,code=joined[...,:rank],joined[...,rank:]
                    hidden_term=hidden@module.weight[:,:rank].T
                    code_term=code@module.weight[:,rank:].T
                    activated=value.tanh()
                    records[site]=dict(down_rms=rms(hidden),code_rms=rms(code),
                        hidden_preactivation_rms=rms(hidden_term),code_preactivation_rms=rms(code_term),
                        bias_rms=rms(module.bias),bridge_preactivation_rms=rms(value),
                        saturation_fraction=float((value.abs()>3).float().mean()),
                        tanh_derivative_mean=float((1-activated.square()).mean()),features=features(activated))
                handles.append(branch.bridge.register_forward_hook(particle_hook))
            velocity=adapter.velocity(*inputs[:6],strength=inputs[6],cfg=3.)
            for handle in handles:handle.remove()
            if arm=='fresh_particle':
                assert torch.equal(velocity,old_velocity),'fresh zero branches do not preserve original base forward'
            else:
                teacher=encoder.teacher_velocity(context)
                assert torch.equal((velocity-teacher).cpu(),previous['groups'][group]['new']),'trained replay differs from previous panel'
            ratios=[records[site]['features']['rms']/max(old_records[site]['features']['rms'],1e-30) for site in sites]
            rows=list(records.values())
            group_report[arm]=dict(sites=records,summary=dict(
                mean_site_saturation_fraction=sum(x['saturation_fraction'] for x in rows)/len(rows),
                mean_site_tanh_derivative=sum(x['tanh_derivative_mean'] for x in rows)/len(rows),
                mean_site_feature_rms_ratio_to_original=sum(ratios)/len(ratios),
                feature_rms_ratio_range=[min(ratios),max(ratios)],
                saturation_sites=sorted([dict(site=k,**v) for k,v in records.items()],key=lambda x:x['saturation_fraction'],reverse=True)[:5]))
            emit(event='arm_complete',group=group,arm=arm,summary={k:v for k,v in group_report[arm]['summary'].items() if k!='saturation_sites'})
        report['groups'][group]=group_report
        (args.output/'branch-geometry.json').write_text(json.dumps(report,indent=2)+'\n')
    report.update(seconds=time.perf_counter()-started,
        limitations='Identical frozen contexts and exact zero-start/parity are established. Feature RMS/rank and saturation alone do not prove the cause of long-run error.')
    (args.output/'branch-geometry.json').write_text(json.dumps(report,indent=2)+'\n')
    emit(event='complete',seconds=report['seconds'],output=args.output)


if __name__=='__main__':
    main()
