#!/usr/bin/env python3
"""Matched, read-only native split-direction test at a qualified Supra boundary.

Recreates the saved last proposal exactly. Changes only antisymmetric direction:
the actual cached preservation backward versus contemporary preservation/edit
native-game backwards at identical post-update weights and private noise. Native
parents, children, radii, fit-only selection, FAST/EMA zero-harm guards remain.
All candidate guards are recorded for diagnosis, never used to select a candidate.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def emit(**row):
    print(json.dumps(row, default=str), flush=True)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    temporary = Path(path).with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str) + '\n')
    temporary.replace(path)


def guard_record(baselines, candidate, margin):
    fit = baselines['fit'] - candidate['fit']
    fast = baselines['guard'] - candidate['guard']
    average = baselines['average_guard'] - candidate['average_guard']
    gain, average_gain = float(fast.mean()), float(average.mean())
    harm, average_harm = float(-fast.min()), float(-average.min())
    fit_gain = float(baselines['fit'].mean()) - float(candidate['fit'].mean())
    fit_pass = fit_gain >= margin
    guard_pass = gain >= margin and average_gain >= -1e-12 and max(harm, average_harm) <= 1e-12
    return dict(fit_gain=fit_gain, guard_gain=gain, average_guard_gain=average_gain,
                max_context_harm=harm, average_max_context_harm=average_harm,
                fit_pass=fit_pass, guard_pass=guard_pass, accepted=fit_pass and guard_pass,
                harmed_fast=int((fast < -1e-12).sum()), harmed_average=int((average < -1e-12).sum()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT/'outputs/e22-particle-v2-6400')
    parser.add_argument('--particlegan-root', type=Path,
                        default=ROOT/'outputs/e22-convergence-gap/particlegan-cabe2084-source')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/e22-structural-phase-causal-6400')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('choose a fresh output directory')
    sys.path.insert(0, str(args.particlegan_root.resolve()))
    sys.path.insert(1, str(ROOT))
    import torch
    import particlegan
    from particlegan.continuous import DataDriftController
    from particlegan.routing import RoutedRowControl
    from supra.runtime import model_module, TARGETS
    from supra.particle_adapter import LINEAR_MODULATED_V2
    from supra.particle_game import patchify
    from supra.particle_pilot import checkpoint, restore, state_digest
    from supra.particle_training import make_training_loop
    device = torch.device(args.device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('native BF16/full-model diagnostic requires CUDA')
    if Path(particlegan.__file__).resolve().parent.parent != args.particlegan_root.resolve():
        raise RuntimeError('ParticleGAN source provenance mismatch')
    torch.set_num_threads(4)
    torch.cuda.set_device(device)
    began = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    inputs = {name: args.run / name for name in ('final.pt', 'data.pt', 'receipt.json', 'run.json', 'train.jsonl')}
    input_hashes = {name: sha(path) for name, path in inputs.items()}
    sources = [Path(__file__)] + [ROOT/'supra'/name for name in
        ('particle_adapter.py', 'particle_game.py', 'particle_training.py', 'particle_pilot.py',
         'particle_training_data.py', 'runtime.py')]
    sources += list((args.particlegan_root/'particlegan').glob('*.py'))
    source_hashes = {str(path.resolve()): sha(path) for path in sources}
    saved = torch.load(inputs['final.pt'], map_location='cpu', weights_only=False, mmap=True)
    data = torch.load(inputs['data.pt'], map_location='cpu', weights_only=False)
    receipt = json.loads(inputs['receipt.json'].read_text())
    start_digest = state_digest(saved)
    if receipt['final_native_digest'] != start_digest or receipt['final_checkpoint_sha256'] != input_hashes['final.pt']:
        raise RuntimeError('starting checkpoint differs from qualified receipt')
    step = saved['policy']['completed_steps']
    if step % 100 or step % 5 or saved['config'].get('architecture') != LINEAR_MODULATED_V2:
        raise RuntimeError('this witness requires a V2 preservation/probe boundary')
    if data and state_digest(data) != saved['config']['dataset_digest']:
        raise RuntimeError('starting data digest mismatch')
    last = saved['policy']['routing']['last']
    if last['step'] != step or last.get('accepted') is not False or 'parent' not in last:
        raise RuntimeError('saved native event must be an evaluated rejected split')
    trace = [json.loads(line) for line in inputs['train.jsonl'].read_text().splitlines() if line.strip()]
    recent_hold = next(row for row in reversed(trace) if row['hold'])
    recent_edit = next(row for row in reversed(trace) if not row['hold'])
    if recent_hold['step'] != step or recent_edit['step'] != step-1:
        raise RuntimeError('last game batches are not the native adjacent edit/preservation updates')
    with torch.random.fork_rng(devices=[device.index or 0]):
        module = model_module()
        with torch.device('meta'):
            base = module.SupraDiT()
            module.attach_supra_lora(base, rank=16, alpha=16, targets=TARGETS)
        base.load_state_dict({key.removeprefix('teacher.'): value
            for key, value in saved['policy']['models']['encoder'].items()
            if key.startswith('teacher.')}, strict=True, assign=True)
        base.to(device).eval().requires_grad_(False)
        loop = make_training_loop(base, data, device=device, architecture=LINEAR_MODULATED_V2,
            branch_lr=saved['config']['branch_lr'], probe_interval=saved['config']['probe_interval'])
    restore(loop, saved)
    before = state_digest(checkpoint(loop))
    if before != start_digest:
        raise RuntimeError('full native checkpoint did not restore exactly')
    policy, control = loop.policy, loop.policy.routed_control
    if control.spec.output_error_guard or control.spec.max_context_harm != 0:
        raise RuntimeError('native zero feature harm/no-output-guard contract changed')
    global_cpu = torch.get_rng_state().clone()
    global_cuda = torch.cuda.get_rng_state(device).clone()
    gradients_before = [(parameter, None if parameter.grad is None else parameter.grad.clone())
        for optimizer in policy.optimizers for group in optimizer.param_groups for parameter in group['params']]
    fast, average = control.candidate(copy=True), control.candidate(averaged=True, copy=True)
    effect = control.evidence.effect
    enough = control.evidence.effective_contexts >= control.spec.min_observations
    fresh = control.evidence.last_probe >= max(1, control.counters['evals'] - math.ceil(len(policy.table)/control.spec.probe_budget))
    children = (enough & fresh & (effect > control.spec.min_effect)).nonzero().flatten()
    parents = (enough & fresh & (effect < -control.spec.min_effect)).nonzero().flatten()
    children = children[effect[children].argsort(descending=True, stable=True)]
    parents = parents[effect[parents].argsort(stable=True)]
    pairs = [(int(child), int(parent)) for child in children for parent in parents][:control.spec.candidate_budget]
    if len(pairs) != control.spec.candidate_budget:
        raise RuntimeError('native candidate budget cannot be reconstructed')
    payload = dict(baselines={}, candidates=[], gradients={}, source_table=fast.table.cpu(),
                   source_averaged_table=average.table.cpu(), pairs=pairs, native_last=deepcopy(last))
    plan = dict(checkpoint_step=step, native_state_digest=before, input_sha256=input_hashes,
        source_sha256=source_hashes, particlegan_commit=json.loads(inputs['run.json'].read_text())['particlegan_commit'],
        split_scale=control.spec.split_scale, improvement_margin=control.spec.improvement_margin,
        max_context_harm=control.spec.max_context_harm, output_error_guard=control.spec.output_error_guard,
        pairs=pairs, gradient_batches=dict(hold={key:recent_hold[key] for key in ('step','batch_indices')},
                                          edit={key:recent_edit[key] for key in ('step','batch_indices')}),
        selection='same native pairs/radii; fit-only winner per direction arm, antisymmetric tie preference',
        interventions=['actual_cached_hold', 'current_hold', 'current_edit'],
        common_noise='private copies of saved outgoing paired and DV12 streams, reset identically for each contemporary gradient',
        guard_diagnostics='all candidate guards are recorded; only fit-selected candidates count as arm decisions',
        limitations='One fixed boundary. Cached hold gradient precedes the final G step; contemporary hold controls that timing difference. No training, guard-based candidate selection, or output metric is used.')
    write(args.output/'plan.json', plan)
    emit(event='plan', **plan)

    def current_gradient(pool_name, row):
        context = data[pool_name]['context'][row['batch_indices']].to(device)
        candidate = control.candidate()
        private = torch.Generator(device=device).set_state(saved['policy']['streams']['noise_generator'].cpu())
        paired = torch.Generator(device=device).set_state(saved['paired_noise_rng'].cpu())
        controller = DataDriftController('dv12')
        controller.load_state_dict(deepcopy(policy.controller.state_dict()))
        controller_before = state_digest(controller.state_dict())
        prior = controller.routed_prior(candidate.table, candidate.log_mass)
        real = policy.output_sigma() * torch.randn((len(context),256,16), device=device, generator=paired)
        with torch.enable_grad(), control._evaluating():
            prediction = control.spec.forward(control.models, context, candidate,
                perturb_fn=lambda codes: controller.perturb_latent(codes, private, prior, record=False))
            condition = policy.encoder.condition(context)
            with torch.no_grad():
                real_logits = policy.D(real, condition)
            fake = real + patchify(prediction).float() / policy.D.scale
            loss = policy.recipe.make_loss().g_loss(policy.D(fake, condition), real_logits)
            gradient, = torch.autograd.grad(loss, policy.table)
        if state_digest(controller.state_dict()) != controller_before:
            raise RuntimeError('private gradient replay mutated DV12 controller')
        return gradient.detach(), float(loss.detach())

    directions = {'actual_cached_hold': control.latest_gradient.detach().clone()}
    for label, pool, row in (('current_hold', 'holds', recent_hold), ('current_edit', 'fit', recent_edit)):
        gradient, loss = current_gradient(pool, row)
        directions[label] = gradient
        emit(event='game_gradient', direction=label, loss_g_unweighted=loss,
             gradient_norm=float(gradient.double().norm()))
    payload['gradients'] = {name:value.cpu() for name,value in directions.items()}
    fit_context, fit_targets = control.fit_context[:control.fit_fill], control.fit_targets[:control.fit_fill]
    guard_context, guard_targets = control.guard_context[:control.guard_fill], control.guard_targets[:control.guard_fill]
    with torch.no_grad(), control._evaluating():
        baselines = dict(fit=control._measure(fit_context, fit_targets, fast),
                         guard=control._measure(guard_context, guard_targets, fast),
                         average_guard=control._measure(guard_context, guard_targets, average))
        payload['baselines'] = {name:value.cpu() for name,value in baselines.items()}
        if float(baselines['fit'].mean()) != last['fit_error']:
            raise RuntimeError('saved native fit baseline did not replay bit exactly')
        entries = []
        for child, parent in pairs:
            variant_list = [('duplicate', 'shared', torch.zeros_like(fast.table[parent]))]
            for direction, gradient in directions.items():
                if not bool(gradient[parent].norm() > 0):
                    raise RuntimeError('matched direction comparison cannot include random fallback directions')
                holder = SimpleNamespace(spec=control.spec, latest_gradient=gradient)
                delta = RoutedRowControl._delta(holder, parent, fast, average)
                variant_list.append(('antisymmetric', direction, delta))
            for variant, direction, delta in variant_list:
                proposed = control._split(fast, child, parent, delta)
                proposed_average = control._split(average, child, parent, delta)
                values = dict(fit=control._measure(fit_context, fit_targets, proposed),
                              guard=control._measure(guard_context, guard_targets, proposed),
                              average_guard=control._measure(guard_context, guard_targets, proposed_average))
                record = dict(child=child, parent=parent, variant=variant, direction=direction,
                    fit_error=float(values['fit'].mean()), **guard_record(baselines, values, control.spec.improvement_margin))
                entries.append(record)
                payload['candidates'].append(dict(record=record, delta=delta.cpu(),
                                                   values={name:value.cpu() for name,value in values.items()}))
                emit(event='candidate', **record)
                torch.save(payload, args.output/'candidate-losses.pt')
    decisions = {}
    for direction in directions:
        eligible = [entry for entry in entries if entry['direction'] in ('shared', direction)]
        selected = sorted(eligible, key=lambda entry:(entry['fit_error'], entry['variant']!='antisymmetric'))[0]
        decisions[direction] = selected
    actual = decisions['actual_cached_hold']
    replay_fields = ('child','parent','variant','guard_gain','average_guard_gain','max_context_harm',
                     'average_max_context_harm','accepted')
    if any(actual[key] != last[key] for key in replay_fields):
        raise RuntimeError(f'native selected candidate/guard did not replay exactly: {actual} versus {last}')
    after = state_digest(checkpoint(loop))
    if before != after:
        raise RuntimeError('read-only candidate/gradient diagnostic mutated native checkpoint state')
    if not torch.equal(global_cpu, torch.get_rng_state()) or not torch.equal(global_cuda, torch.cuda.get_rng_state(device)):
        raise RuntimeError('diagnostic advanced a global random stream')
    if any((old is None and parameter.grad is not None) or (old is not None and not torch.equal(old,parameter.grad))
           for parameter,old in gradients_before):
        raise RuntimeError('diagnostic mutated optimizer-owned gradient buffers')
    if any(sha(path) != input_hashes[name] for name,path in inputs.items()) or any(sha(path) != digest for path,digest in source_hashes.items()):
        raise RuntimeError('inputs or executing sources changed during capture')
    cosines = {}
    for parent in sorted({parent for _,parent in pairs}):
        row = {}
        for first,second in (('actual_cached_hold','current_hold'), ('current_hold','current_edit'), ('actual_cached_hold','current_edit')):
            a,b = directions[first][parent].double(), directions[second][parent].double()
            row[f'{first}__{second}'] = float((a@b)/(a.norm()*b.norm()).clamp_min(1e-30))
        cosines[str(parent)] = row
    result = dict(plan=plan, decisions=decisions, entries=entries, parent_gradient_cosines=cosines,
                  qualified_gpu_capture=True, native_baseline_and_selected_guard_replay_exact=True,
                  full_checkpoint_unchanged=True, global_rng_unchanged=True, gradient_buffers_unchanged=True,
                  input_and_source_unchanged=True, payload_sha256=sha(args.output/'candidate-losses.pt'),
                  seconds=time.perf_counter()-began)
    write(args.output/'result.json', result)
    emit(event='complete', decisions=decisions, seconds=result['seconds'], output=args.output)


if __name__ == '__main__':
    main()
