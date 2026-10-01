#!/usr/bin/env python3
"""One true-game-gradient linear-path witness at the identical FAST6400 point.

The intervention adds U(Ax)-U0(A0x) to each existing nonlinear particle
branch. U0/A0 are frozen copies of FAST6400; inputs remain differentiable.
There are no new trainable parameters, output losses, metric guards, or metric
selection. Native update 6401 is the sole optimization step in each arm.
"""
import argparse
from contextlib import contextmanager
from copy import deepcopy
import gc
import json
import math
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
START = 6400
PIN = 'f459cb6d6aaaabeb1af076ec53ad7a963618de90'
REFERENCE_NAMES = ('linear_reference_down', 'linear_reference_up')


def attach_references(policy):
    """Identical frozen FAST references on live and averaged G owners."""
    before = {id(p) for group in policy.opt_g.param_groups for p in group['params']}
    added = 0
    for live, averaged in zip(policy.G.particle_branches(), policy.ema_G.particle_branches()):
        for target in (live, averaged):
            target.add_module(REFERENCE_NAMES[0], deepcopy(live.down).eval().requires_grad_(False))
            target.add_module(REFERENCE_NAMES[1], deepcopy(live.up).eval().requires_grad_(False))
            added += sum(p.numel() for name in REFERENCE_NAMES for p in getattr(target, name).parameters())
    after = {id(p) for group in policy.opt_g.param_groups for p in group['params']}
    if before != after or any(p.requires_grad for owner in (policy.G, policy.ema_G)
                             for branch in owner.particle_branches()
                             for name in REFERENCE_NAMES for p in getattr(branch, name).parameters()):
        raise RuntimeError('Frozen reference attachment changed optimizer ownership')
    return added


@contextmanager
def linear_path_forward(enabled=True):
    """Full FP32 projection math; no detached-input or straight-through path."""
    from supra.particle_adapter import _ParticleProjection
    original = _ParticleProjection.forward
    def forward(self, x):
        if not hasattr(self, REFERENCE_NAMES[0]):
            return original(self, x)
        frame = self.frame
        if frame is None:
            raise RuntimeError('particle projections require forward_routed()')
        expected_batch = frame.batch_size * (2 if frame.guided else 1)
        if x.ndim != 3 or x.shape[0] != expected_batch:
            raise ValueError('Supra projection activations must be [batch, tokens, channels]')
        base_output = self.base(x)
        with torch.autocast(device_type=x.device.type, enabled=False):
            projected_input = x.float()
            query = frame.router.query_for_site(self.site)(projected_input)
            table = frame.candidate.table
            logits = query @ table.T / math.sqrt(table.shape[-1])
            if frame.guided:
                conditional, unconditional = logits.chunk(2, dim=0)
                logits = torch.stack((conditional, unconditional), dim=1)
            codes = frame.routing.mix(self.site, logits)
            if frame.guided:
                codes = torch.cat((codes[:,0], codes[:,1]), dim=0)
            if frame.strength == 0:
                return base_output
            hidden = self.down(projected_input)
            nonlinear = self.up(self.bridge(torch.cat((hidden, codes.float()), dim=-1)).tanh())
            # Keep both input Jacobians in the graph. At the initial point
            # their difference is zero; existing U/A parameter derivatives
            # acquire a genuine additional linear path.
            linear_difference = self.up(hidden) - self.linear_reference_up(
                self.linear_reference_down(projected_input))
            delta = nonlinear + linear_difference
            return (base_output.float() + frame.strength*delta).to(base_output.dtype)
    if enabled:
        _ParticleProjection.forward = forward
    try:
        yield
    finally:
        _ParticleProjection.forward = original


def original_frozen_digest(loop):
    from supra.particle_pilot import state_digest
    return state_digest({role: {name: value for name, value in owner.named_parameters()
                        if not value.requires_grad and not any(part in name.split('.') for part in REFERENCE_NAMES)}
                        for role, owner in (('generator', loop.policy.G), ('encoder', loop.policy.encoder))})


def parameter_group(name):
    for group in ('down', 'bridge', 'up'):
        if group in name.split('.'):
            return group
    return 'other'


@torch.no_grad()
def evaluate_fast(loop, data, next_fit, fixed_critic):
    """FAST G and fixed saved D; all metrics are observational."""
    from supra.particle_game import patchify
    from scripts.diagnose_e22_supra_training_controls import emit
    stream = torch.Generator(device=loop.policy.device).manual_seed(72)
    metrics, outputs = {}, {}
    pools = dict(next_fit=next_fit, test=data['test']['context'], preservation=data['preservation']['context'])
    for name, pool in pools.items():
        records, predictions = [], []
        for begin in range(0, len(pool), 4):
            context = pool[begin:begin+4].to(loop.policy.device)
            residual = loop.policy.routed_generate(context, sigma=0, perturb=False, averaged=False)
            predictions.append(residual.cpu())
            condition = loop.policy.encoder.condition(context)
            teacher = loop.policy.encoder.teacher_velocity(context)
            base = .125*torch.randn((4,len(context),256,16), device=loop.policy.device, generator=stream)
            real = base.flatten(0,1)
            fake = (base + (patchify(residual).float()/fixed_critic.scale).unsqueeze(0)).flatten(0,1)
            cond = condition.repeat(4,1)
            real_scores, fake_scores = fixed_critic(real,cond), fixed_critic(fake,cond)
            g = torch.nn.functional.softplus(real_scores-fake_scores).reshape(4,len(context)).mean(0)
            d = torch.nn.functional.softplus(fake_scores-real_scores).reshape(4,len(context)).mean(0)
            mse = residual.square().flatten(1).mean(1)
            trms = teacher.square().flatten(1).mean(1).sqrt()
            for row in range(len(context)):
                records.append(dict(index=begin+row, g_game=float(g[row]), d_game=float(d[row]),
                                    mse=float(mse[row]), teacher_rms=float(trms[row])))
        outputs[name] = torch.cat(predictions)
        mean_mse = sum(r['mse'] for r in records)/len(records)
        metrics[name] = dict(count=len(records), rmse=mean_mse**.5,
                            relative_rms=(mean_mse/(sum(r['teacher_rms']**2 for r in records)/len(records)))**.5,
                            g_game=sum(r['g_game'] for r in records)/len(records),
                            d_game=sum(r['d_game'] for r in records)/len(records), records=records)
        emit(event='fast_game_evaluation', pool=name, **{k:v for k,v in metrics[name].items() if k!='records'})
    return metrics, outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT/'outputs/e22-full-f459cb6d-6400')
    parser.add_argument('--particlegan-root', type=Path,
                        default=ROOT/'outputs/e22-convergence-gap/particlegan-f459-source')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/e22-convergence-gap/linear-path-one-step')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    sys.path.insert(0,str(args.particlegan_root.resolve()))
    sys.path.insert(1,str(ROOT))
    import particlegan
    from scripts.diagnose_e22_supra_training_controls import emit, write_json, sha, dv12_instrumentation
    from supra.particle_pilot import checkpoint, restore, state_digest
    from supra.particle_training import make_training_loop, training_update
    from supra.particle_adapter import NONLINEAR_V1
    from supra.runtime import model_module, TARGETS
    imported = Path(particlegan.__file__).resolve()
    declared = json.loads((args.run/'run.json').read_text())
    sources = {p.name:sha(p) for p in sorted(imported.parent.glob('*.py'))}
    if imported.parent.parent != args.particlegan_root.resolve() or declared['particlegan_commit'] != PIN:
        raise RuntimeError('This witness requires the original archived f459cb6d source')
    if state_digest(sources) != declared['particlegan_source_digest']:
        raise RuntimeError('ParticleGAN source digest differs from the native run')
    manifest = json.loads((args.run/'source/sha256.json').read_text())
    if any(sha(ROOT/name)!=value for name,value in manifest.items() if name.startswith('supra/')):
        raise RuntimeError('Supra training source changed')
    saved = torch.load(args.run/'final.pt',map_location='cpu',weights_only=False,mmap=True)
    data = torch.load(args.run/'data.pt',map_location='cpu',weights_only=False)
    if saved['policy']['completed_steps']!=START or str(torch.device(args.device))!=saved['policy']['device']:
        raise RuntimeError('Strict source checkpoint must be completed step6400 on the original device')
    if state_digest(data)!=saved['config']['dataset_digest'] or saved['config']['output_error_guard']:
        raise RuntimeError('Dataset or output guard differs from the native contract')
    generator = torch.Generator();generator.set_state(saved['data_rng'])
    next_ids = torch.randint(len(data['fit']['context']),(4,),generator=generator)
    next_fit = data['fit']['context'][next_ids]
    plan = dict(start_step=START,end_step=START+1,updates=1,arms=['native_control','linear_path'],
                source_commit=PIN,source_digest=declared['particlegan_source_digest'],script_sha256=sha(__file__),
                forward='Existing nonlinear branch + U(Ax)-U0(A0x); references frozen FAST6400, inputs differentiated',
                optimization='One complete unchanged native RpGAN/KA2 update6401; existing trainable parameters only',
                evaluation='FAST training G only; fixed saved D; next fit batch, all test and preservation contexts',
                output_errors='Reporting only; no loss, structural criterion, guard or selection',
                averaging='Same FAST references attached to live/EMA owners for native schema; initial EMA forward equality not claimed',
                next_fit_indices=next_ids.tolist(),limits='One update is a conditioning witness, not proof of long-term quality repair or a supported export')
    emit(event='checked_predeclared_witness',**plan)
    if args.check_only:
        return
    if not torch.cuda.is_available():
        raise RuntimeError('Native BF16 witness requires CUDA')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('choose a fresh output directory')
    args.output.mkdir(parents=True,exist_ok=True)
    write_json(args.output/'plan.json',plan)
    torch.set_num_threads(4);torch.cuda.set_device(torch.device(args.device))
    expected = state_digest(saved)
    starting_outputs = starting_noisy = reference_streams = reference_critic = None
    reports = {}
    for arm in plan['arms']:
        module = model_module()
        with torch.random.fork_rng(devices=[]),torch.device('meta'):
            base = module.SupraDiT();module.attach_supra_lora(base,rank=16,alpha=16,targets=TARGETS)
        teacher = {k.removeprefix('teacher.'):v for k,v in saved['policy']['models']['encoder'].items()
                   if k.startswith('teacher.')}
        base.load_state_dict(teacher,strict=True,assign=True)
        base.to(args.device).eval().requires_grad_(False)
        loop = make_training_loop(base,data,device=args.device,probe_interval=saved['config']['probe_interval'],
                                  branch_lr=saved['config']['branch_lr'],
                                  architecture=saved['config'].get('architecture', NONLINEAR_V1))
        restore(loop,saved)
        if state_digest(checkpoint(loop))!=expected:
            raise RuntimeError('Initial full native checkpoint restore is not exact')
        frozen_before = original_frozen_digest(loop)
        trainable_before = {n:p.detach().cpu().clone() for n,p in loop.policy.G.named_parameters() if p.requires_grad}
        existing_count = sum(p.numel() for p in loop.policy.G.parameters() if p.requires_grad)
        added_frozen = attach_references(loop.policy)
        if sum(p.numel() for p in loop.policy.G.parameters() if p.requires_grad)!=existing_count:
            raise RuntimeError('Intervention added trainable parameters')
        fixed_critic = deepcopy(loop.policy.D).eval().requires_grad_(False)
        raw_gradient = {name:[] for name in ('down','bridge','up','other')}
        def record_gradient(name):
            def hook(value):
                raw_gradient[parameter_group(name)].append(value.detach().double().square().sum())
            return hook
        hooks = [p.register_hook(record_gradient(n)) for n,p in loop.policy.G.named_parameters() if p.requires_grad]
        native_model_outputs = []
        with linear_path_forward(enabled=arm=='linear_path'):
            before_hash = state_digest(checkpoint(loop))
            before_metrics, before_outputs = evaluate_fast(loop,data,next_fit,fixed_critic)
            if before_hash!=state_digest(checkpoint(loop)):
                raise RuntimeError('Read-only pre-update evaluation changed native state')
            if starting_outputs is None:
                starting_outputs = before_outputs
            elif any(not torch.equal(before_outputs[k],starting_outputs[k]) for k in before_outputs):
                raise RuntimeError('Intervention changes the native BF16 clean FAST6400 forward point')
            recorder = loop.policy.G.model.register_forward_hook(
                lambda _,__,value:native_model_outputs.append(value.detach().cpu().clone()))
            with dv12_instrumentation(loop.policy.controller) as calls:
                row = training_update(loop)
            recorder.remove()
            if len(native_model_outputs)!=2 or row['step']!=START+1:
                raise RuntimeError('Witness must contain exactly the D/G host forwards and one native update')
            streams = {k:row[k] for k in ('batch_indices','base_noise_sums','paired_rng_digest','dv12_rng_digest')}
            critic_hash = state_digest(dict(critic=loop.policy.D.state_dict(),optimizer=loop.policy.opt_d.state_dict()))
            if reference_streams is None:
                reference_streams,reference_critic,starting_noisy = streams,critic_hash,native_model_outputs
            elif streams!=reference_streams or critic_hash!=reference_critic or any(
                    not torch.equal(a,b) for a,b in zip(native_model_outputs,starting_noisy)):
                raise RuntimeError('Initial native noisy D/G outputs, streams or critic step differ between arms')
            after_hash = state_digest(checkpoint(loop))
            after_metrics,after_outputs = evaluate_fast(loop,data,next_fit,fixed_critic)
            if after_hash!=state_digest(checkpoint(loop)):
                raise RuntimeError('Read-only post-update evaluation changed native state')
        for hook in hooks:hook.remove()
        if original_frozen_digest(loop)!=frozen_before:
            raise RuntimeError('Original frozen base/teacher weights changed')
        gradient_norms = {k:(float(torch.stack(v).sum().sqrt()) if v else 0.) for k,v in raw_gradient.items()}
        step_squares = {k:0. for k in raw_gradient}
        for name,value in loop.policy.G.named_parameters():
            if name in trainable_before:
                step_squares[parameter_group(name)] += float((value.detach().cpu()-trainable_before[name]).double().square().sum())
        report = dict(arm=arm,initial_restore_exact=True,initial_clean_and_native_noisy_forward_equal=True,
                      critic_step_identical=True,training_streams_identical=True,original_frozen_unchanged=True,
                      existing_trainable_G_parameters=existing_count,additional_trainable_parameters=0,
                      added_frozen_reference_parameters_live_and_ema=added_frozen,
                      raw_game_G_gradient_norms=gradient_norms,native_G_step_norms={k:v**.5 for k,v in step_squares.items()},
                      native_update=row,native_dv12_calls=calls['calls'],evaluated_G_source='FAST training G',
                      before={k:{n:v for n,v in value.items() if n!='records'} for k,value in before_metrics.items()},
                      after={k:{n:v for n,v in value.items() if n!='records'} for k,value in after_metrics.items()},
                      fixed_D_game_change={k:after_metrics[k]['g_game']-before_metrics[k]['g_game'] for k in before_metrics},
                      original_full_state_digest=expected,intervention_final_state_digest=after_hash)
        reports[arm]=report
        write_json(args.output/(arm+'-evaluation.json'),dict(before=before_metrics,after=after_metrics))
        torch.save(dict(before=before_outputs,after=after_outputs,native_D_G_noisy_host_outputs=native_model_outputs),
                   args.output/(arm+'-outputs.pt'))
        write_json(args.output/'receipt.json',dict(plan=plan,arms=reports))
        emit(event='arm_complete',**report)
        del loop,base,fixed_critic,trainable_before,before_outputs,after_outputs,native_model_outputs
        gc.collect();torch.cuda.empty_cache()
    emit(event='complete',output=str(args.output),one_step_only=True)


if __name__=='__main__':
    main()
