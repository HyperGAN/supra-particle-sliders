#!/usr/bin/env python3
"""Frozen tied-key/value usage and query Jacobian diagnostics; no updates."""
import argparse
import json
import math
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from supra.runtime import model_module,TARGETS
from supra.particle_export import load_particle_adapter
from supra.particle_training_data import FrozenSliderContexts


def emit(**row):
    print(json.dumps(row,default=str),flush=True)


@torch.no_grad()
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',type=Path,default=ROOT/'outputs/e22-full-f459cb6d-6400')
    ap.add_argument('--source',type=Path,default=ROOT/'outputs/e22-convergence-gap/native-noise/residuals.pt')
    ap.add_argument('--output',type=Path,default=ROOT/'outputs/e22-convergence-gap/key-geometry')
    ap.add_argument('--device',default='cuda:0')
    args=ap.parse_args()
    if args.output.exists() and any(args.output.iterdir()):ap.error('choose a fresh output')
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    started=time.perf_counter()
    device=torch.device(args.device)
    data=torch.load(args.run/'data.pt',map_location='cpu',weights_only=False)
    saved=torch.load(args.run/'final.pt',map_location='cpu',weights_only=False,mmap=True)
    previous=torch.load(args.source,map_location='cpu',weights_only=False)
    state=saved['policy']
    mod=model_module()
    with torch.random.fork_rng(devices=[]),torch.device('meta'):
        base=mod.SupraDiT()
        mod.attach_supra_lora(base,rank=16,alpha=16,targets=TARGETS)
    teacher_state={k.removeprefix('teacher.'):v for k,v in state['models']['encoder'].items() if k.startswith('teacher.')}
    base.load_state_dict(teacher_state,strict=True,assign=True)
    base.to(device).eval().requires_grad_(False)
    encoder=FrozenSliderContexts(data['text_contexts'],data['text_masks'],base).to(device)
    new=load_particle_adapter(base,args.run/'final-boss-particlegan.safetensors',device=device)
    table=new.table
    mass=new.router.log_mass
    norms=table.norm(dim=-1)
    unit_table=table/norms[:,None].clamp_min(1e-30)
    max_row=int(norms.argmax())
    report=dict(metadata=dict(checkpoint_step=6400,evaluation_only=True,
        precision='native CUDA BF16 host; native FP32 clean query/routing; FP64 covariance diagnostic',
        key_counterfactual='only compare logit argmax with per-row unit keys; never apply different weights to the host',
        covariance='d clean mixed code / d query = weighted table covariance / sqrt(z_dim)',
        interpretation='selection/gradient conditioning, not an optimizer or structural criterion'),
        table=dict(max_norm_row=max_row,max_norm=float(norms.max()),median_norm=float(norms.median()),
                   mean_norm=float(norms.mean()),max_norm_coordinates=table[max_row].cpu().tolist()),groups={})
    for group in ('fit','holds'):
        context=previous['groups'][group]['context'].to(device)
        inputs=encoder.unpack(context)
        records={}
        handles=[]
        for site,branch in zip(new.generator.sites,new.generator.particle_branches()):
            def hook(_,inputs,site=site):
                x=inputs[0]
                with torch.autocast(device_type=device.type,enabled=False):
                    q=new.router.query_for_site(site)(x.float()).reshape(-1,table.shape[-1])
                    logits=q@table.T/math.sqrt(table.shape[-1])+mass
                    weights=logits.softmax(-1)
                    codes=weights@table
                    selected=logits.argmax(-1)
                    cosine_logits=q@unit_table.T
                    # Active represented masses are uniform in this run;
                    # rowwise key normalization only tests norm preference.
                    unit_selected=(cosine_logits+mass).argmax(-1)
                    qnorm=q.norm(dim=-1)
                    selected_cosine=(q*unit_table[selected]).sum(-1)/qnorm.clamp_min(1e-30)
                    max_cosine=cosine_logits.max(-1).values/qnorm.clamp_min(1e-30)
                    wd=weights.double()
                    wd=wd/wd.sum(-1,keepdim=True)
                    td=table.double()
                    centered=td[None]-((wd@td)[:,None])
                    jac=torch.einsum('qn,qni,qnj->qij',wd,centered,centered)/math.sqrt(table.shape[-1])
                    jacnorm=jac.square().sum((-2,-1)).sqrt()
                    usage=weights.mean(0)
                    top=usage.topk(5)
                    gaps=logits.topk(2,dim=-1).values.diff(dim=-1).abs().squeeze(-1)
                    entropy=-(weights*weights.clamp_min(1e-30).log()).sum(-1)
                    records[site]=dict(codes=len(q),query_norm_mean=float(qnorm.mean()),
                        query_norm_max=float(qnorm.max()),code_rms=float(codes.square().mean().sqrt()),
                        mean_selected_row_norm=float(norms[selected].mean()),
                        max_norm_row_selection_fraction=float((selected==max_row).float().mean()),
                        max_norm_row_average_weight=float(usage[max_row]),
                        selected_cosine_mean=float(selected_cosine.mean()),best_unit_key_cosine_mean=float(max_cosine.mean()),
                        unit_key_argmax_changed_fraction=float((selected!=unit_selected).float().mean()),
                        unit_key_max_norm_row_selection_fraction=float((unit_selected==max_row).float().mean()),
                        softmax_max_weight_mean=float(weights.max(-1).values.mean()),
                        softmax_onehot_fraction=float((weights.max(-1).values==1).float().mean()),
                        token_weight_ess_mean=float(weights.square().sum(-1).reciprocal().mean()),
                        softmax_entropy_mean=float(entropy.mean()),mean_top_logit_gap=float(gaps.mean()),
                        query_jacobian_frobenius_mean=float(jacnorm.mean()),
                        query_jacobian_frobenius_median=float(jacnorm.median()),
                        query_jacobian_below_1e_minus8_fraction=float((jacnorm<1e-8).float().mean()),
                        top_usage=[dict(row=int(i),usage=float(u),norm=float(norms[i])) for i,u in zip(top.indices,top.values)])
            handles.append(branch.register_forward_pre_hook(hook))
        value=new.velocity(*inputs[:6],strength=inputs[6],cfg=3.)
        for handle in handles:handle.remove()
        teacher=encoder.teacher_velocity(context)
        assert torch.equal((value-teacher).cpu(),previous['groups'][group]['new']),'clean replay changed output'
        summary=dict(sites=len(records),
            mean_site_max_norm_row_selection_fraction=sum(x['max_norm_row_selection_fraction'] for x in records.values())/len(records),
            mean_site_onehot_fraction=sum(x['softmax_onehot_fraction'] for x in records.values())/len(records),
            mean_site_unit_key_argmax_changed_fraction=sum(x['unit_key_argmax_changed_fraction'] for x in records.values())/len(records))
        report['groups'][group]=dict(summary=summary,sites=records,exact_clean_replay=True)
        (args.output/'key-geometry.json').write_text(json.dumps(report,indent=2)+'\n')
        emit(event='group_complete',group=group,**summary)
    report.update(seconds=time.perf_counter()-started,
        limitations='Only frozen diagnostic batches were measured. Norm preference and small query Jacobians do not establish the causal benefit of changing routing or table learning.')
    (args.output/'key-geometry.json').write_text(json.dumps(report,indent=2)+'\n')
    emit(event='complete',seconds=report['seconds'],output=args.output)


if __name__=='__main__':main()
