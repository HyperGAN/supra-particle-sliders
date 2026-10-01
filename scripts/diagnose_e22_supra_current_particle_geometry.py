#!/usr/bin/env python3
"""Read-only matched V1/V2 key and modulation geometry on the current task.

No optimizer, structural decision, or reconstruction objective is run. Unit
keys are only a frozen selection/Jacobian diagnostic; host routing is unchanged.
The latest residual payload can be reused for independent critic diagnostics.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch


def emit(**row):
    print(json.dumps(row,default=str),flush=True)


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def stream_sha(value):
    return hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()


def move(value,device):
    if isinstance(value,torch.Tensor):return value.to(device)
    if isinstance(value,dict):return {k:move(v,device) for k,v in value.items()}
    if isinstance(value,list):return [move(v,device) for v in value]
    if isinstance(value,tuple):return tuple(move(v,device) for v in value)
    return value


def feature_geometry(value):
    value=value.detach().double().reshape(-1,value.shape[-1])
    centered=value-value.mean(0)
    covariance=centered.T@centered/len(value)
    return dict(rms=float(value.square().mean().sqrt()),
        centered_rms=float(centered.square().mean().sqrt()),
        effective_rank=float(covariance.trace().square()/covariance.square().sum().clamp_min(1e-30)))


def query_geometry(query,table,mass):
    q=query.detach().float().reshape(-1,table.shape[-1])
    norms=table.norm(dim=-1)
    keys=table/norms[:,None].clamp_min(1e-30)*math.sqrt(table.shape[-1])
    logits=q@table.T/math.sqrt(table.shape[-1])+mass
    normalized_logits=q@keys.T/math.sqrt(table.shape[-1])+mass
    weights=logits.softmax(-1)
    unit_weights=normalized_logits.softmax(-1)
    def jacobian_norm(w,k):
        w=w.double();w=w/w.sum(-1,keepdim=True)
        values=table.double();k=k.double()
        vc=values[None]-(w@values)[:,None]
        kc=k[None]-(w@k)[:,None]
        jac=torch.einsum('qn,qni,qnj->qij',w,vc,kc)/math.sqrt(table.shape[-1])
        return jac.square().sum((-2,-1)).sqrt()
    native_jac=jacobian_norm(weights,table)
    unit_jac=jacobian_norm(unit_weights,keys)
    selected=logits.argmax(-1)
    unit_selected=normalized_logits.argmax(-1)
    directions=table/norms[:,None].clamp_min(1e-30)
    cosine=(q@directions.T)/q.norm(dim=-1,keepdim=True).clamp_min(1e-30)
    usage=weights.mean(0);top=usage.topk(3)
    maximum=int(norms.argmax())
    return dict(tokens=len(q),query_norm_mean=float(q.norm(dim=-1).mean()),
        native_max_weight_mean=float(weights.max(-1).values.mean()),
        native_entropy_mean=float(-(weights*weights.clamp_min(1e-30).log()).sum(-1).mean()),
        native_onehot_fraction=float((weights.max(-1).values==1).float().mean()),
        native_token_ESS_mean=float(weights.square().sum(-1).reciprocal().mean()),
        max_norm_row_selection_fraction=float((selected==maximum).float().mean()),
        selected_norm_mean=float(norms[selected].mean()),
        selected_direction_cosine_mean=float(cosine.gather(1,selected[:,None]).mean()),
        best_direction_cosine_mean=float(cosine.max(-1).values.mean()),
        equal_norm_key_argmax_changed_fraction=float((selected!=unit_selected).float().mean()),
        equal_norm_key_max_weight_mean=float(unit_weights.max(-1).values.mean()),
        equal_norm_key_entropy_mean=float(-(unit_weights*unit_weights.clamp_min(1e-30).log()).sum(-1).mean()),
        native_query_jacobian_frobenius_median=float(native_jac.median()),
        native_query_jacobian_frobenius_mean=float(native_jac.mean()),
        native_query_jacobian_below_1e_minus8_fraction=float((native_jac<1e-8).float().mean()),
        equal_norm_key_query_jacobian_frobenius_median=float(unit_jac.median()),
        top_usage=[dict(row=int(i),mass=float(u),norm=float(norms[i])) for i,u in zip(top.indices,top.values)])


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--latest',type=Path,default=ROOT/'outputs/e22-particle-noise-update-256/latest')
    parser.add_argument('--v1',type=Path,default=ROOT/'outputs/e22-full-f459cb6d')
    parser.add_argument('--v2',type=Path,default=ROOT/'outputs/e22-particle-v2-1600')
    parser.add_argument('--particlegan-root',type=Path,default=ROOT/'outputs/e22-convergence-gap/particlegan-cabe2084-source')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/e22-particle-current-geometry')
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--draws',type=int,default=4)
    args=parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):parser.error('choose a fresh output directory')
    sys.path.insert(0,str(args.particlegan_root.resolve()))
    from particlegan.continuous import DataDriftController
    from supra.particle_adapter import NONLINEAR_V1,LINEAR_MODULATED_V2
    from supra.particle_export import load_particle_adapter,_EncodedCondition
    from supra.particle_pilot import state_digest
    from supra.particle_training_data import FrozenSliderContexts
    from supra.runtime import model_module,TARGETS
    import particlegan
    if Path(particlegan.__file__).resolve().parent.parent!=args.particlegan_root.resolve():
        raise RuntimeError('ParticleGAN import differs from the declared frozen source')
    torch.set_num_threads(4)
    device=torch.device(args.device)
    args.output.mkdir(parents=True,exist_ok=True)
    latest=torch.load(args.latest/'final.pt',map_location='cpu',weights_only=False,mmap=True)
    data=torch.load(args.latest/'data.pt',map_location='cpu',weights_only=False)
    dataset_digest=state_digest(data)
    if dataset_digest!=latest['config']['dataset_digest']:raise RuntimeError('cached task data changed')
    data_stream=torch.Generator();data_stream.set_state(latest['data_rng'].cpu())
    contexts={};ids_by_group={};step=latest['policy']['completed_steps']
    for upcoming in range(step+1,step+6):
        pool='holds' if upcoming%5==0 else 'fit'
        ids=torch.randint(len(data[pool]['context']),(4,),generator=data_stream)
        if pool not in contexts:
            contexts[pool]=data[pool]['context'][ids].to(device)
            ids_by_group[pool]=dict(proposed_step=upcoming,indices=ids.tolist())
    contexts['test']=data['test']['context'][:4].to(device)
    ids_by_group['test']=dict(indices=list(range(4)))
    payload=dict(metadata=dict(checkpoint=str((args.latest/'final.pt').resolve()),
        checkpoint_sha256=sha(args.latest/'final.pt'),checkpoint_step=step,dataset_digest=dataset_digest,
        imported_particlegan=str(Path(particlegan.__file__).resolve()),
        source_commit=json.loads((args.latest/'run.json').read_text())['particlegan_commit'],
        source_data_stream_sha256=stream_sha(latest['data_rng']),
        source_dv12_stream_sha256=stream_sha(latest['policy']['streams']['noise_generator']),
        source_paired_stream_sha256=stream_sha(latest['paired_noise_rng']),
        precision='native CUDA BF16 host; FP32 clean routing and bridge; FP64 geometry',
        selection=ids_by_group,noise_draws=args.draws,evaluation_only=True,
        noise_semantics='Private DV12 stream cloned from saved boundary for each pool; no begin_step or intervening training updates are advanced.'),groups={},comparison_residuals={})
    report=dict(metadata=payload['metadata'],points={})
    began=time.perf_counter()
    for label,run in (('v1_1600',args.v1),('v2_1600',args.v2),('v2_latest',args.latest)):
        saved=torch.load(run/'final.pt',map_location='cpu',weights_only=False,mmap=True)
        if saved['config']['dataset_digest']!=dataset_digest:raise RuntimeError('comparison task data differ')
        module=model_module()
        with torch.random.fork_rng(devices=[]),torch.device('meta'):
            base=module.SupraDiT();module.attach_supra_lora(base,rank=16,alpha=16,targets=TARGETS)
        base.load_state_dict({k.removeprefix('teacher.'):v for k,v in saved['policy']['models']['encoder'].items()
                             if k.startswith('teacher.')},strict=True,assign=True)
        base.to(device).eval().requires_grad_(False)
        with torch.random.fork_rng(devices=[device.index or 0]):
            encoder=FrozenSliderContexts(data['text_contexts'],data['text_masks'],base).to(device)
            export=run/('final-boss-particlegan.safetensors' if label=='v1_1600' else 'final.safetensors')
            adapter=load_particle_adapter(base,export,device=device)
        expected=saved['config'].get('architecture',NONLINEAR_V1)
        if adapter.generator.architecture!=expected:raise RuntimeError('clean export uses another architecture')
        before=state_digest(adapter.state_dict());encoder_before=state_digest(encoder.state_dict())
        norms=adapter.table.norm(dim=-1)
        point=dict(completed_steps=saved['policy']['completed_steps'],architecture=expected,
            served_source=adapter.metadata['served_source'],table=dict(
                max_row=int(norms.argmax()),max_norm=float(norms.max()),median_norm=float(norms.median()),
                mean_norm=float(norms.mean())),groups={})
        payload['comparison_residuals'][label]={}
        for group,context in contexts.items():
            records={site:{} for site in adapter.generator.sites};handles=[]
            for branch in adapter.generator.particle_branches():
                def pre_hook(_,inputs,site=branch.site):
                    with torch.autocast(device.type,enabled=False):
                        q=adapter.router.query_for_site(site)(inputs[0].float())
                        records[site]['routing']=query_geometry(q,adapter.table,adapter.router.log_mass)
                def bridge_hook(mod,inputs,value,branch=branch):
                    joined=inputs[0].float();rank=adapter.generator.rank
                    hidden,codes=joined[...,:rank],joined[...,rank:]
                    activated=value.tanh()
                    actual_up=hidden+activated if expected==LINEAR_MODULATED_V2 else activated
                    records[branch.site]['bridge']=dict(code_rms=float(codes.square().mean().sqrt()),
                        hidden_preactivation_rms=float((hidden@mod.weight[:,:rank].T).square().mean().sqrt()),
                        code_preactivation_rms=float((codes@mod.weight[:,rank:].T).square().mean().sqrt()),
                        saturation_fraction=float((value.abs()>3).float().mean()),
                        tanh_derivative_mean=float((1-activated.square()).mean()),
                        modulation_features=feature_geometry(activated),native_up_features=feature_geometry(actual_up))
                handles.append(branch.register_forward_pre_hook(pre_hook))
                handles.append(branch.bridge.register_forward_hook(bridge_hook))
            inputs=encoder.unpack(context)
            clean=adapter.velocity(*inputs[:6],strength=inputs[6],cfg=3.)
            for handle in handles:handle.remove()
            teacher=encoder.teacher_velocity(context)
            residual=(clean-teacher).cpu()
            payload['comparison_residuals'][label][group]=residual
            routing=[r['routing'] for r in records.values()];bridges=[r['bridge'] for r in records.values()]
            summary=dict(mean_site_max_weight=sum(r['native_max_weight_mean'] for r in routing)/len(routing),
                mean_site_entropy=sum(r['native_entropy_mean'] for r in routing)/len(routing),
                mean_site_norm_key_argmax_changed_fraction=sum(r['equal_norm_key_argmax_changed_fraction'] for r in routing)/len(routing),
                mean_site_saturation=sum(b['saturation_fraction'] for b in bridges)/len(bridges),
                mean_site_modulation_rank=sum(b['modulation_features']['effective_rank'] for b in bridges)/len(bridges),
                mean_site_native_up_rank=sum(b['native_up_features']['effective_rank'] for b in bridges)/len(bridges))
            point['groups'][group]=dict(sites=records,summary=summary)
            if label=='v2_latest':
                controller=DataDriftController('dv12');controller.load_state_dict(move(latest['policy']['controller'],device))
                control_before=state_digest(controller.state_dict())
                candidate=adapter.routing.candidate_for(dict(router=adapter.router),adapter.table,
                    averaged=adapter.metadata['served_source']=='averaged')
                prior=controller.routed_prior(candidate.table,candidate.log_mass)
                stream=torch.Generator(device=device);stream.set_state(latest['policy']['streams']['noise_generator'].cpu())
                condition=_EncodedCondition(*[inputs[i] for i in (0,2,3,4,5)],inputs[6],3.)
                packed=torch.cat((inputs[0].flatten(1),inputs[1][:,None]),1)
                models=dict(generator=adapter.generator,router=adapter.router,conditioning=condition)
                noisy=[]
                for draw in range(args.draws):
                    value=adapter.routing.forward(models,packed,candidate,perturb_fn=lambda codes:
                        controller.perturb_latent(codes,stream,prior,record=False))
                    noisy.append((value-teacher).cpu())
                if state_digest(controller.state_dict())!=control_before:raise RuntimeError('private DV12 replay changed controller')
                payload['groups'][group]=dict(context=context.cpu(),new=residual,
                    noisy=torch.stack(noisy),condition=encoder.condition(context).cpu())
                paired=torch.Generator(device=device);paired.set_state(latest['paired_noise_rng'].cpu())
                payload['groups'][group]['critic_base_noise']=torch.randn(len(context),256,16,device=device,generator=paired).cpu()
                payload['groups'][group]['generator_base_noise']=torch.randn(len(context),256,16,device=device,generator=paired).cpu()
                payload['groups'][group]['output_sigma']=float(latest['policy']['last_output_sigma'])
                torch.save(payload,args.output/'residuals.pt')
            emit(event='geometry_group',point=label,group=group,**summary)
        if before!=state_digest(adapter.state_dict()) or encoder_before!=state_digest(encoder.state_dict()):
            raise RuntimeError('frozen geometry modified a model')
        point['frozen_models_unchanged']=True
        report['points'][label]=point
        (args.output/'geometry.json').write_text(json.dumps(report,indent=2)+'\n')
        del adapter,encoder,base,saved
        torch.cuda.empty_cache()
    report.update(seconds=time.perf_counter()-began,
        limitations='Equal-norm keys are selection/Jacobian counterfactuals at the frozen point, never host routing changes. They do not prove better training. All comparisons use the same latest-selected diagnostic contexts.')
    (args.output/'geometry.json').write_text(json.dumps(report,indent=2)+'\n')
    torch.save(payload,args.output/'residuals.pt')
    emit(event='complete',output=args.output,seconds=report['seconds'])


if __name__=='__main__':main()
