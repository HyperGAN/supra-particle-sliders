#!/usr/bin/env python3
"""Read-only native table profiles with an exact-addition DV12 offset control.

For five actual late native updates, hold G/router at their pre-step values and
evaluate predetermined fractions of the actual table displacement. Native DV12
recomputes support geometry; fixed-offset replay retains each actual G-pass
site's displacement using the native addition order. A clean control describes
local host sensitivity without the latent perturbation.
No profile selects a learning rate, checkpoint, structural move or output loss.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
FRACTIONS = (0., .01, .1, .25, .5, 1.)
MODES = ('native_dv12', 'fixed_offset_dv12', 'clean')


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):
            value.update(block)
    return value.hexdigest()


def emit(**row):
    print(json.dumps(row), flush=True)


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=ROOT/'outputs/e22-particle-v2-6400')
    parser.add_argument('--particlegan-root',type=Path,
                        default=ROOT/'outputs/e22-convergence-gap/particlegan-cabe2084-source')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/e22-table-dv12-exact-offset-6400')
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('choose a fresh output directory')
    sys.path.insert(0,str(args.particlegan_root.resolve()))
    sys.path.insert(1,str(ROOT))
    import torch
    import torch.nn.functional as F
    import particlegan
    from supra.runtime import model_module,TARGETS
    from supra.particle_game import patchify
    from supra.particle_pilot import checkpoint,restore,state_digest,frozen_digest
    from supra.particle_training import make_training_loop,training_update
    from monitor_e22_supra_particle_convergence import evaluation_modes
    if Path(particlegan.__file__).resolve().parent.parent != args.particlegan_root.resolve():
        raise RuntimeError('unexpected ParticleGAN import')
    torch.set_num_threads(4)
    device=torch.device(args.device)
    torch.cuda.set_device(device)
    began=time.perf_counter()
    saved=torch.load(args.run/'final.pt',map_location='cpu',weights_only=False,mmap=True)
    data=torch.load(args.run/'data.pt',map_location='cpu',weights_only=False)
    review=json.loads((args.run/'qualification-review.json').read_text())
    if not review.get('qualified') or saved['policy']['completed_steps']!=6400:
        raise RuntimeError('requires qualified V2 step6400')
    initial=state_digest(saved)
    if initial != review['final_native_digest'] or state_digest(data)!=saved['config']['dataset_digest']:
        raise RuntimeError('qualified starting state or dataset differs')
    args.output.mkdir(parents=True)
    source_files=[Path(__file__),ROOT/'scripts/monitor_e22_supra_particle_convergence.py']
    source_files+=sorted((ROOT/'supra').glob('particle*.py'))+[ROOT/'supra/runtime.py']
    sources={str(path.relative_to(ROOT)):sha(path) for path in source_files}
    pg_sources={str(path.relative_to(args.particlegan_root)):sha(path)
                for path in sorted((args.particlegan_root/'particlegan').rglob('*.py'))}
    for path in source_files:
        target=args.output/'source'/path.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target)
    inputs={name:sha(args.run/name) for name in ('final.pt','data.pt','qualification-review.json')}
    plan=dict(schema='supra_table_dv12_exact_offset_v1',start_step=6400,updates=5,fractions=list(FRACTIONS),modes=list(MODES),
        initial_native_digest=initial,input_sha256=inputs,application_source_sha256=sources,
        particlegan_source_sha256=pg_sources,imported_particlegan=str(Path(particlegan.__file__).resolve()),
        objective='actual post-D paired RpGAN game, unweighted reporting; no output metric',
        intervention='only table displacement; G/router fixed before actual optimizer step; no optimizer or training changes',
        fixed_offset_replay='per-site new_codes + actual_Gpass_displacement, using identical native FP32 addition order',
        controls='all six predeclared fractions under native/fixed-offset DV12 and clean; same actual G-pass Gaussian',
        selection='five predetermined updates, all profiles; no variant/checkpoint/hyperparameter selection',
        limits='Local game profiles on one stream. BF16 forward rounding and detached DV12 support geometry can both differ from autograd; a finite displacement alone does not prove an incorrect derivative.')
    write(args.output/'plan.json',plan)
    emit(event='plan',**plan)
    with torch.random.fork_rng(devices=[device.index or 0]):
        mod=model_module()
        with torch.device('meta'):
            base=mod.SupraDiT()
            mod.attach_supra_lora(base,rank=16,alpha=16,targets=TARGETS)
        base.load_state_dict({key.removeprefix('teacher.'):value for key,value in
            saved['policy']['models']['encoder'].items() if key.startswith('teacher.')},strict=True,assign=True)
        base.to(device).eval().requires_grad_(False)
        loop=make_training_loop(base,data,device=device,architecture=saved['config']['architecture'],
            probe_interval=saved['config']['probe_interval'],branch_lr=saved['config']['branch_lr'])
    restore(loop,saved)
    if state_digest(checkpoint(loop))!=initial:
        raise RuntimeError('native initial restore differs')
    frozen=frozen_digest(loop)
    reference=[training_update(loop) for _ in range(5)]
    reference_final=state_digest(checkpoint(loop))
    restore(loop,saved)
    policy=loop.policy
    original_generate,original_step,original_after=policy.routed_generate,policy.opt_g.step,policy.after_generator_step
    controller_type=type(policy.controller)
    original_perturb=controller_type.perturb_latent
    capture={}
    records=[]
    evidence=dict(plan=plan,rows=[])

    def in_step_digest():
        modules=policy._training_modules()
        return state_digest(dict(models={key:m.state_dict() for key,m in modules.items()},
            averages={key:m.state_dict() for key,m in policy._average_modules().items()},
            gradients={key:{name:p.grad for name,p in m.named_parameters()} for key,m in modules.items()},
            training={key:[m.training for m in owner.modules()] for key,owner in modules.items()},
            requires_grad={key:[p.requires_grad for p in m.parameters()] for key,m in modules.items()},
            table=policy.table,table_grad=policy.table.grad,averaged_table=policy.averaged_table,
            optimizers=[o.state_dict() for o in policy.optimizers],controller=policy.controller.state_dict(),
            routing=policy.routed_control.state_dict(),lr_settle=policy.lr_settle.state_dict(),
            penalty=policy.penalty.regularizer.state_dict(),penalty_stats=policy.penalty.last_stats,
            phase=policy._phase,completed_steps=policy.completed_steps,data_rng=loop.data_rng.get_state(),
            paired_rng=loop.paired_noise_rng.get_state(),dv12_rng=policy.noise_generator.get_state(),
            cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(device)))

    def perturb(owner,codes,*a,**kw):
        value=original_perturb(owner,codes,*a,**kw)
        if owner is policy.controller and capture.get('in_actual_Gpass'):
            # Read the application just recorded by the original call. Do not
            # replay noise, request another record or change diagnostic state.
            application=owner._latent_application_records[-1]
            offset=application.displacement
            if not torch.equal(codes.detach()+offset,value.detach()):
                raise RuntimeError('actual native displacement does not reproduce the native addition')
            capture['sites'].append(dict(codes=codes.detach().clone(),perturbed=value.detach().clone(),
                                        displacement=offset.detach().clone()))
        return value

    def generated(context,**kw):
        capture['calls']+=1
        actual=capture['calls']==2
        if actual:
            capture['context']=context.detach().clone()
            capture['dv12_state']=policy.noise_generator.get_state().clone()
            capture['controller_state']=deepcopy(policy.controller.state_dict())
            capture['sigma']=float(policy._noise.output_sigma.detach())
            capture['in_actual_Gpass']=True
        try:
            value=original_generate(context,**kw)
        finally:
            capture['in_actual_Gpass']=False
        if actual:
            capture['prediction']=value.detach().clone()
        return value

    def stepped(*a,**kw):
        groups={role:list(group['params']) for group,role in zip(policy.opt_g.param_groups,policy.roles[0])
                if role in ('generator','router','table')}
        if len(groups.get('table',[]))!=1 or groups['table'][0] is not policy.table:
            raise RuntimeError('unexpected table role owner')
        capture['groups']=groups
        capture['before']={role:[p.detach().clone() for p in params] for role,params in groups.items()}
        capture['table_gradient']=policy.table.grad.detach().clone()
        return original_step(*a,**kw)

    @torch.no_grad()
    def after():
        original_after()
        native_before=in_step_digest()
        current={role:[p.detach().clone() for p in params] for role,params in capture['groups'].items()}
        table_before=capture['before']['table'][0]
        displacement=current['table'][0]-table_before
        hold=(policy.completed_steps+1)%5==0
        weight=.1 if hold else 1.
        dot=float((capture['table_gradient'].double()*displacement.double()).sum())
        if len(capture['sites'])!=len(policy.G.sites):
            raise RuntimeError('actual Gpass did not capture every ordered native site exactly once')
        result=dict(step=policy.completed_steps+1,hold=hold,game_weight=weight,
            weighted_gradient_dot_displacement=dot,unweighted_gradient_dot_displacement=dot/weight,
            displacement_norm=float(displacement.double().norm()),
            table_gradient_norm=float(capture['table_gradient'].double().norm()),site_calls=len(capture['sites']),profiles={})
        raw=dict(step=result['step'],table_before=table_before.cpu(),displacement=displacement.cpu(),
                 table_gradient=capture['table_gradient'].cpu(),actual_prediction=capture['prediction'].cpu(),
                 sites=[dict(codes=site['codes'].cpu(),perturbed=site['perturbed'].cpu(),
                             displacement=site['displacement'].cpu()) for site in capture['sites']],profiles={})
        try:
            with evaluation_modes(policy):
                for role,params in capture['groups'].items():
                    for parameter,value in zip(params,capture['before'][role]):
                        parameter.copy_(value)
                context=capture['context']
                condition=policy.encoder.condition(context)
                real=capture['sigma']*capture['generator_base']
                real_logits=policy.D(real,condition)
                for mode in MODES:
                    result['profiles'][mode]=[]
                    raw['profiles'][mode]=[]
                    base_prediction=None
                    for fraction in FRACTIONS:
                        # Preserve native endpoints bit exactly; intermediate
                        # floating additions are explicit diagnostic laws.
                        policy.table.copy_(table_before if fraction==0 else
                            current['table'][0] if fraction==1 else table_before+displacement*fraction)
                        candidate=policy.routed_control.candidate()
                        private=torch.Generator(device=device).set_state(capture['dv12_state'].cpu())
                        from particlegan.continuous import DataDriftController
                        controller=DataDriftController('dv12')
                        controller.load_state_dict(deepcopy(capture['controller_state']))
                        controller_before=state_digest(controller.state_dict())
                        prior=controller.routed_prior(candidate.table,candidate.log_mass)
                        site_cursor=0
                        changed_codes=total_codes=changed_perturb=total_perturb=0
                        def applied(codes):
                            nonlocal site_cursor,changed_codes,total_codes,changed_perturb,total_perturb
                            saved_site=capture['sites'][site_cursor]
                            site_cursor+=1
                            if codes.shape!=saved_site['codes'].shape:
                                raise RuntimeError('native site order/shape changed during replay')
                            fixed_offset=codes+saved_site['displacement']
                            value=(controller.perturb_latent(codes,private,prior,record=False)
                                   if mode=='native_dv12' else fixed_offset)
                            changed_codes+=int((codes!=saved_site['codes']).sum())
                            total_codes+=codes.numel()
                            changed_perturb+=int((value!=fixed_offset).sum())
                            total_perturb+=value.numel()
                            return value
                        prediction=policy.routed_control.spec.forward(policy._training_modules(),context,candidate,
                            perturb_fn=None if mode=='clean' else applied)
                        if mode!='clean' and site_cursor!=len(capture['sites']):
                            raise RuntimeError('replay did not visit every captured site')
                        if state_digest(controller.state_dict())!=controller_before:
                            raise RuntimeError('private DV12 replay changed diagnostics')
                        if fraction==0:
                            base_prediction=prediction.clone()
                            if mode!='clean' and not torch.equal(prediction,capture['prediction']):
                                raise RuntimeError('native/fixed-offset zero-displacement Gpass did not replay bit exactly')
                        gap=real_logits-policy.D(real+patchify(prediction).float()/policy.D.scale,condition)
                        score=float(F.softplus(gap).mean())
                        if not torch.isfinite(torch.tensor(score)):
                            raise RuntimeError('nonfinite native game profile')
                        record=dict(fraction=fraction,g_game=score,
                            prediction_changed_fraction=float((prediction!=base_prediction).float().mean()),
                            prediction_equal_to_mode_baseline=torch.equal(prediction,base_prediction),
                            codes_changed_fraction=None if mode=='clean' else changed_codes/total_codes,
                            native_vs_fixed_offset_site_output_changed_fraction=None if mode=='clean' else changed_perturb/total_perturb)
                        result['profiles'][mode].append(record)
                        raw['profiles'][mode].append(dict(fraction=fraction,gap=gap.cpu(),prediction=prediction.cpu()))
                    baseline=result['profiles'][mode][0]['g_game']
                    for item in result['profiles'][mode]:
                        item['delta_from_mode_baseline']=item['g_game']-baseline
        finally:
            for role,params in capture['groups'].items():
                for parameter,value in zip(params,current[role]):
                    parameter.copy_(value)
        if in_step_digest()!=native_before:
            raise RuntimeError('displacement profiles altered active native state')
        result['native_state_unchanged_by_profiles']=True
        records.append(result)
        evidence['rows'].append(raw)
        write(args.output/'profiles.json',dict(plan=plan,rows=records,completed=False))
        torch.save(evidence,args.output/'profile-evidence.pt')
        emit(event='profile_step',step=result['step'],hold=hold,
             table_dot_displacement_unweighted=dot/weight,
             game_deltas={mode:[item['delta_from_mode_baseline'] for item in result['profiles'][mode]] for mode in MODES})

    # The controller checkpoints its instance dictionary. A class wrapper
    # preserves that strict schema and delegates every original application;
    # private replay controllers are not included in G-pass capture.
    controller_type.perturb_latent=perturb
    policy.routed_generate=generated
    policy.opt_g.step=stepped
    policy.after_generator_step=after
    rows=[]
    try:
        for index in range(5):
            capture.clear()
            capture.update(calls=0,sites=[],in_actual_Gpass=False)
            paired=torch.Generator(device=device).set_state(loop.paired_noise_rng.get_state().cpu())
            torch.randn((4,256,16),device=device,generator=paired)
            capture['generator_base']=torch.randn((4,256,16),device=device,generator=paired)
            row=training_update(loop)
            if row!=reference[index]:
                raise RuntimeError('instrumented native update differs from exact reference replay')
            rows.append(row)
    finally:
        controller_type.perturb_latent=original_perturb
        policy.routed_generate=original_generate
        policy.opt_g.step=original_step
        policy.after_generator_step=original_after
    final=state_digest(checkpoint(loop))
    if final!=reference_final or frozen_digest(loop)!=frozen:
        raise RuntimeError('instrumented complete native state differs from reference replay')
    unchanged=(all(sha(args.run/name)==digest for name,digest in inputs.items())
        and all(sha(ROOT/name)==digest for name,digest in sources.items())
        and all(sha(args.particlegan_root/name)==digest for name,digest in pg_sources.items()))
    if not unchanged:
        raise RuntimeError('diagnostic inputs or executing sources changed')
    complete=dict(plan=plan,rows=records,completed=True,exact_native_rows=True,exact_native_full_replay=True,
        frozen_unchanged=True,source_and_inputs_unchanged=True,final_native_digest=final,
        reference_final_native_digest=reference_final,evidence_sha256=sha(args.output/'profile-evidence.pt'),
        seconds=time.perf_counter()-began)
    write(args.output/'native-training-rows.json',rows)
    write(args.output/'profiles.json',complete)
    emit(event='complete',seconds=complete['seconds'],final_native_digest=final)


if __name__=='__main__':
    main()
