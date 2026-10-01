#!/usr/bin/env python3
"""Five predeclared 256-update native-game branches from the same step 6400.

This diagnostic never optimizes output MSE, changes structural acceptance to
an output metric, selects checkpoints by evaluation, or changes training keys.
It imports the archived ParticleGAN source used by the original checkpoint.
"""
import argparse
from contextlib import contextmanager
from copy import deepcopy
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
START, UPDATES = 6400, 256
ARMS = ('native_control', 'clean_dv12', 'hold_game_units', 'no_anchor', 'generator_no_amsgrad')
PIN = 'f459cb6d6aaaabeb1af076ec53ad7a963618de90'


def emit(**value):
    print(json.dumps(value, default=str), flush=True)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str)+'\n')
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_copy(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(v) for v in value)
    return deepcopy(value)


@contextmanager
def dv12_instrumentation(controller, *, disable_application=False):
    """Record native draw shapes without adding state to its owner."""
    owner = type(controller)
    original = owner.perturb_latent
    count = {'calls': 0, 'signature': hashlib.sha256()}
    def sampled_but_unapplied(self, latent, stream, prior=None, record=False):
        value = original(self, latent, stream, prior, record)
        if self is controller:
            count['calls'] += 1
            count['signature'].update(repr((tuple(latent.shape), str(latent.dtype), str(latent.device))).encode())
            if disable_application:
                return latent
        return value
    owner.perturb_latent = sampled_but_unapplied
    try:
        yield count
    finally:
        owner.perturb_latent = original


def hold_units_update(loop):
    """Native application lifecycle; D/controller see unweighted payoffs.

    Only G's preservation gradient keeps the task's .1 weight. D's native
    payoff and full KA2 penalty have identical relative units on both pools.
    """
    from particlegan import RoutedBatch
    from supra.particle_game import apply_critic_penalty, patchify, _rng_digest
    policy = loop.policy
    hold = (policy.completed_steps+1) % 5 == 0
    pool = loop.hold_context if hold else loop.fit_context
    indices = torch.randint(len(pool), (4,), generator=loop.data_rng)
    context = pool[indices.to(pool.device)]
    target = torch.zeros(4, 4, 32, 32, device=pool.device)
    game_weight = .1 if hold else 1.
    condition = loop.condition(context) if hasattr(loop, 'condition') else policy.encoder.condition(context)
    critic_base = torch.randn((4, 256, 16), device=policy.device, dtype=torch.float32,
                              generator=loop.paired_noise_rng)
    generator_base = torch.randn((4, 256, 16), device=policy.device, dtype=torch.float32,
                                 generator=loop.paired_noise_rng)
    noise = policy.begin_step(target, routed=RoutedBatch(context, target, loop.guard_context, loop.guard_targets))
    loss = policy.recipe.make_loss()
    policy.G.eval()
    policy.encoder.eval()
    policy.router.eval()
    policy.D.train()
    with torch.no_grad():
        prediction = policy.routed_generate(context, sigma=0, perturb=True)
        real = noise.output_sigma * critic_base
        fake = real + patchify(prediction-target).float() / policy.D.scale
    policy.observe_critic_pair(real, fake)
    penalty = apply_critic_penalty(policy.penalty, policy.D, real, fake, condition)
    loss_d_game = loss.d_loss(policy.D(real, condition), policy.D(fake, condition))
    loss_d = loss_d_game+penalty
    policy.opt_d.zero_grad(set_to_none=True)
    policy.before_critic_backward()
    loss_d.backward()
    policy.opt_d.step()
    policy.after_critic_step()
    policy.D.eval()
    policy.G.train()
    policy.encoder.train()
    policy.router.train()
    flags = [p.requires_grad for p in policy.D.parameters()]
    try:
        policy.D.requires_grad_(False)
        prediction = policy.routed_generate(context, sigma=0, perturb=True)
        real = noise.output_sigma * generator_base
        with torch.no_grad():
            real_logits = policy.D(real.detach(), condition)
        fake = real + patchify(prediction-target).float() / policy.D.scale
        loss_g_unweighted = loss.g_loss(policy.D(fake, condition), real_logits)
        loss_g = game_weight * loss_g_unweighted
        policy.opt_g.zero_grad(set_to_none=True)
        policy.before_generator_backward()
        loss_g.backward()
        gradient = policy.table.grad
        dense_rows = 0 if gradient is None else int(gradient.norm(dim=-1).gt(0).sum())
        gradient_norm = 0. if gradient is None else float(gradient.detach().double().norm())
        policy.after_generator_backward(loss_gan=loss_g_unweighted.detach(), loss_critic=loss_d_game.detach())
        policy.opt_g.step()
        policy.after_generator_step()
    finally:
        for parameter, flag in zip(policy.D.parameters(), flags):
            parameter.requires_grad_(flag)
    movement = policy.finish_step()
    result = dict(step=policy.completed_steps, loss_d=float(loss_d.detach()), loss_d_game=float(loss_d_game.detach()),
                  loss_g=float(loss_g.detach()), loss_g_unweighted=float(loss_g_unweighted.detach()),
                  penalty=float(penalty.detach()), penalty_phase=policy.penalty.last_stats.get('phase', 'lazy_skip'),
                  penalty_calls=policy.opt_d.record.calls, output_sigma=policy.output_sigma(), move=movement,
                  bank_grad_norm=gradient_norm, dense_gradient_rows=dense_rows,
                  optimizer_surprise_fires=0 if policy.surprise is None else policy.surprise.fires,
                  optimizer_surprise_ratio=None if policy.surprise is None else policy.surprise.last_ratio,
                  anchor_release_events=0 if policy.surprise is None else policy.surprise.anchor_events,
                  batch_indices=indices.tolist(), base_noise_sums=[float(critic_base.sum()), float(generator_base.sum())],
                  paired_rng_digest=_rng_digest(loop.paired_noise_rng), dv12_rng_digest=_rng_digest(policy.noise_generator),
                  hold=hold, game_weight=game_weight, critic_game_weight=1., controller_payoff_weight=1.)
    for name in ('loss_d', 'loss_g', 'penalty', 'bank_grad_norm', 'output_sigma'):
        if not torch.isfinite(torch.tensor(result[name])):
            raise RuntimeError(f'nonfinite {name}: {result}')
    return result


@torch.no_grad()
def evaluate_final(served, data, fixed_critic, arm_critic, device):
    """Complete fixed pools, clean native BF16 host, two fixed critics."""
    from supra.particle_game import patchify
    results = {}
    stream = torch.Generator(device=device).manual_seed(72)
    for name in ('fit', 'test', 'holds', 'preservation'):
        pool = data[name]
        records = []
        for start in range(0, len(pool['context']), 4):
            context = pool['context'][start:start+4].to(device)
            residual = served.routed_forward(context)
            condition = served.encoder.condition(context)
            teacher = served.encoder.teacher_velocity(context)
            bases = .125 * torch.randn((4, len(context), 256, 16), device=device, dtype=torch.float32, generator=stream)
            real = bases.flatten(0, 1)
            cond = condition.repeat(4, 1)
            logits = {}
            for label, critic in (('fixed_start_D', fixed_critic), ('arm_final_D', arm_critic)):
                fake = (bases + (patchify(residual).float()/critic.scale).unsqueeze(0)).flatten(0, 1)
                real_scores, fake_scores = critic(real, cond), critic(fake, cond)
                logits[label] = dict(
                    g_game=torch.nn.functional.softplus(real_scores-fake_scores).reshape(4, len(context)).mean(0),
                    d_game=torch.nn.functional.softplus(fake_scores-real_scores).reshape(4, len(context)).mean(0),
                    score_gap=(real_scores-fake_scores).reshape(4, len(context)).mean(0))
            mse = residual.float().square().flatten(1).mean(1)
            teacher_rms = teacher.square().flatten(1).mean(1).sqrt()
            for row in range(len(context)):
                record = dict(index=start+row, source_caption_id=int(context[row,4097]), time=float(context[row,4096]),
                              mse=float(mse[row]), rmse=float(mse[row].sqrt()), teacher_rms=float(teacher_rms[row]),
                              relative_rmse=float(mse[row].sqrt()/teacher_rms[row].clamp_min(1e-12)))
                for label, values in logits.items():
                    record[label] = {key: float(value[row]) for key, value in values.items()}
                records.append(record)
        total_mse = sum(r['mse'] for r in records)/len(records)
        results[name] = dict(records=records, count=len(records), rmse=total_mse**.5,
                            relative_rms=(total_mse/(sum(r['teacher_rms']**2 for r in records)/len(records)))**.5)
        for label in ('fixed_start_D', 'arm_final_D'):
            results[name][label] = {key: sum(r[label][key] for r in records)/len(records)
                                   for key in ('g_game', 'd_game', 'score_gap')}
        emit(event='evaluation_pool', pool=name, count=len(records), rmse=results[name]['rmse'],
             fixed_g=results[name]['fixed_start_D']['g_game'], final_g=results[name]['arm_final_D']['g_game'])
    return results


def save_delta(loop, path):
    """Diagnostic state without the unchanged frozen hosts; not a resume file."""
    from supra.particle_pilot import checkpoint
    complete = checkpoint(loop)
    native = complete['policy']
    trainable = {name for name, parameter in loop.policy.G.named_parameters() if parameter.requires_grad}
    delta = {key: value for key, value in native.items() if key not in ('models', 'averages')}
    for family in ('models', 'averages'):
        delta[family] = {}
        for role, values in native[family].items():
            if role == 'generator':
                delta[family][role] = {name: value for name, value in values.items() if name in trainable}
            elif role != 'encoder':
                delta[family][role] = values
    torch.save(cpu_copy(dict(format='diagnostic_delta_without_frozen_hosts_v1', policy=delta,
                             data_rng=complete['data_rng'], paired_noise_rng=complete['paired_noise_rng'],
                             config=complete['config'], resumeable=False)), path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT/'outputs/e22-full-f459cb6d-6400')
    parser.add_argument('--particlegan-root', type=Path,
                        default=ROOT/'outputs/e22-convergence-gap/particlegan-f459-source')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/e22-convergence-gap/training-controls-256')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS))
    parser.add_argument('--check-only', action='store_true', help='Check pinned sources/configuration on CPU; no model or updates')
    args = parser.parse_args()
    if len(set(args.arms)) != len(args.arms):
        parser.error('duplicate arms')
    sys.path.insert(0, str(args.particlegan_root.resolve()))
    sys.path.insert(1, str(ROOT))
    import particlegan
    from supra.particle_pilot import checkpoint, restore, state_digest, frozen_digest
    from supra.particle_training import make_training_loop, training_update
    from supra.particle_adapter import NONLINEAR_V1
    from supra.particle_export import export_served_adapter
    from supra.runtime import model_module, TARGETS
    imported = Path(particlegan.__file__).resolve()
    if imported.parent.parent != args.particlegan_root.resolve():
        raise RuntimeError('ParticleGAN was not imported from the declared archived source')
    declared = json.loads((args.run/'run.json').read_text())
    if declared['particlegan_commit'] != PIN:
        raise RuntimeError('This diagnostic is predeclared for the f459cb6d native run')
    sources = {path.name: sha(path) for path in sorted(imported.parent.glob('*.py'))}
    if state_digest(sources) != declared['particlegan_source_digest']:
        raise RuntimeError('ParticleGAN source digest differs from the original run')
    manifest = json.loads((args.run/'source/sha256.json').read_text())
    supra_sources = {name: sha(ROOT/name) for name in manifest if name.startswith('supra/')}
    if any(manifest[name] != value for name, value in supra_sources.items()):
        raise RuntimeError('Supra training sources differ from the saved run archive')
    saved = torch.load(args.run/'final.pt', map_location='cpu', weights_only=False, mmap=True)
    data = torch.load(args.run/'data.pt', map_location='cpu', weights_only=False)
    if saved['policy']['completed_steps'] != START or declared['fixed_updates'] != START:
        raise RuntimeError('Every arm must start at the same completed step 6400')
    if str(torch.device(args.device)) != saved['policy']['device']:
        raise RuntimeError('Strict native restore requires the original device; no silent device remapping')
    if saved['config']['output_error_guard'] or saved['config']['batch_size'] != 4:
        raise RuntimeError('Unexpected output guard or native batch contract')
    if state_digest(data) != saved['config']['dataset_digest']:
        raise RuntimeError('Dataset content differs from the saved native configuration')
    plan = dict(start_step=START, updates=UPDATES, end_step=START+UPDATES, predeclared_arms=list(ARMS),
                selected_arms=args.arms, checkpoint=str((args.run/'final.pt').resolve()),
                original_particlegan_commit=PIN, imported_particlegan=str(imported),
                original_particlegan_source_digest=declared['particlegan_source_digest'],
                supra_source_sha256=supra_sources, script_sha256=sha(__file__),
                paired_streams='Exact initial native/data/paired-noise restore for every arm; no new training keys',
                interventions=dict(native_control='Existing native training_update without changes',
                    clean_dv12='Native perturbation draws consumed; return unperturbed routed codes',
                    hold_game_units='Unweighted D payoff+KA2; G preservation weight .1; unweighted controller payoffs',
                    no_anchor='Native penalty anchor_weight=0 only; caps, EMA and Adam retained',
                    generator_no_amsgrad='Only restored opt_g groups amsgrad=False; moments, LR and D untouched'),
                evaluation='Final only, all original fit/test/holds/preservation records, batch4, native clean residual, '
                           'fixed start critic plus each final critic; four private paired Gaussian draws at sigma .125',
                output_metrics='Reporting only; never optimization, structural acceptance, guard, or checkpoint selection',
                limits='A fixed 256-update local branch test does not by itself establish the cause of the original 6400-update gap')
    emit(event='source_and_configuration_checks_passed', **plan)
    if args.check_only:
        return
    if not torch.cuda.is_available():
        raise RuntimeError('Native BF16 training requires CUDA; --check-only is CPU-only')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('choose a fresh diagnostic output directory')
    args.output.mkdir(parents=True)
    write_json(args.output/'plan.json', plan)
    torch.set_num_threads(4)
    torch.cuda.set_device(torch.device(args.device))
    expected_digest = state_digest(saved)
    reference_sampling = reference_dv12 = None
    reports = {}
    for arm in args.arms:
        output = args.output/arm
        output.mkdir()
        started = time.perf_counter()
        module = model_module()
        with torch.random.fork_rng(devices=[]), torch.device('meta'):
            base = module.SupraDiT()
            module.attach_supra_lora(base, rank=16, alpha=16, targets=TARGETS)
        teacher = {key.removeprefix('teacher.'): value for key, value in saved['policy']['models']['encoder'].items()
                   if key.startswith('teacher.')}
        base.load_state_dict(teacher, strict=True, assign=True)
        base.to(args.device).eval().requires_grad_(False)
        loop = make_training_loop(base, data, device=args.device,
                                  probe_interval=saved['config']['probe_interval'], branch_lr=saved['config']['branch_lr'],
                                  architecture=saved['config'].get('architecture', NONLINEAR_V1))
        restore(loop, saved)
        actual_digest = state_digest(checkpoint(loop))
        if actual_digest != expected_digest:
            raise RuntimeError(f'{arm}: native starting state is not exactly the same full checkpoint')
        original_frozen = frozen_digest(loop)
        fixed_critic = deepcopy(loop.policy.D).eval().requires_grad_(False)
        write_json(output/'initial-state.json', dict(full_checkpoint_exact=True, state_digest=actual_digest,
                   frozen_digest=original_frozen, completed_steps=START, native_device=str(loop.policy.device)))
        emit(event='arm_restored', arm=arm, full_checkpoint_exact=True, step=START)
        if arm == 'no_anchor':
            loop.policy.attach_penalty(loop.policy.recipe.make_critic_penalty(
                loop.policy.opt_d, collect_stats=True, anchor_weight=0.))
        elif arm == 'generator_no_amsgrad':
            for group in loop.policy.opt_g.param_groups:
                group['amsgrad'] = False
        rows = []
        def train():
            with (output/'train.jsonl').open('w') as trace:
                for step in range(START+1, START+UPDATES+1):
                    if arm == 'hold_game_units':
                        with torch.autograd.set_multithreading_enabled(False):
                            row = hold_units_update(loop)
                    else:
                        row = training_update(loop)
                    if row['step'] != step:
                        raise RuntimeError('Native completed-step clock changed')
                    row['arm'] = arm
                    row['dv12_call_count'] = perturbation['calls']
                    row['dv12_draw_shape_signature'] = perturbation['signature'].hexdigest()
                    rows.append(row)
                    trace.write(json.dumps(row, default=str)+'\n')
                    if (step-START) % 16 == 0:
                        trace.flush()
                        emit(event='training', seconds=time.perf_counter()-started, **row)
        with dv12_instrumentation(loop.policy.controller, disable_application=arm=='clean_dv12') as perturbation:
            train()
        perturbation_calls = perturbation['calls']
        sampling = [{key: row[key] for key in ('step', 'hold', 'batch_indices', 'base_noise_sums', 'paired_rng_digest')}
                    for row in rows]
        if reference_sampling is None:
            reference_sampling = sampling
        elif sampling != reference_sampling:
            raise RuntimeError(f'{arm}: original data or paired Gaussian stream differs from the native control')
        draw_program = [{key: row[key] for key in ('step', 'dv12_call_count', 'dv12_draw_shape_signature', 'dv12_rng_digest')}
                        for row in rows]
        if reference_dv12 is None:
            reference_dv12 = draw_program
        same_program_checks = []
        for actual, reference in zip(draw_program, reference_dv12):
            same_program = (actual['dv12_call_count'] == reference['dv12_call_count'] and
                            actual['dv12_draw_shape_signature'] == reference['dv12_draw_shape_signature'])
            if same_program and actual['dv12_rng_digest'] != reference['dv12_rng_digest']:
                raise RuntimeError(f"{arm}: equal native DV12 draw shapes but different stream at {actual['step']}")
            same_program_checks.append(same_program)
        if frozen_digest(loop) != original_frozen:
            raise RuntimeError(f'{arm}: frozen base or teacher changed')
        before_evaluation = state_digest(checkpoint(loop))
        # Capture the final live training critic before creating a serving
        # snapshot. Serving may select averaged G/R/table owners.
        final_critic = deepcopy(loop.policy.D).eval().requires_grad_(False)
        served = loop.policy.served_model()
        export = export_served_adapter(served, output/'final.safetensors', extra_metadata={
            'diagnostic_arm': arm, 'starting_step': START, 'fixed_updates': UPDATES,
            'selection': 'final fixed horizon; output metrics evaluation only'})
        metrics = evaluate_final(served, data, fixed_critic, final_critic, loop.policy.device)
        del served, final_critic
        after_evaluation = state_digest(checkpoint(loop))
        if before_evaluation != after_evaluation:
            raise RuntimeError(f'{arm}: export or evaluation changed native training state')
        save_delta(loop, output/'training-delta.pt')
        write_json(output/'evaluation.json', metrics)
        report = dict(arm=arm, completed_steps=loop.policy.completed_steps, updates=UPDATES,
                      start_state_exact=True, data_and_paired_noise_streams_matched=True, frozen_unchanged=True,
                      evaluation_state_unchanged=True, final_state_digest=after_evaluation,
                      native_dv12_calls=perturbation_calls,
                      clean_dv12_consumed_calls=perturbation_calls if arm=='clean_dv12' else None,
                      dv12_stream_checked_on_equal_call_shapes=True,
                      dv12_equal_call_shape_steps=sum(same_program_checks),
                      dv12_call_shape_program_entirely_matched=all(same_program_checks),
                      sampling_reference_arm=args.arms[0], final_critic_source='fast live training critic',
                      served_source=export['config']['served_source'],
                      final_dv12_rng_digest=rows[-1]['dv12_rng_digest'], final_paired_rng_digest=rows[-1]['paired_rng_digest'],
                      native_penalty_calls=loop.policy.opt_d.record.calls,
                      actual_anchor_weight=loop.policy.penalty.regularizer.anchor_weight,
                      generator_amsgrad=[group['amsgrad'] for group in loop.policy.opt_g.param_groups],
                      critic_amsgrad=[group['amsgrad'] for group in loop.policy.opt_d.param_groups],
                      accepted_moves=[row['move'] for row in rows if isinstance(row['move'], dict) and row['move'].get('moves',0)>0],
                      learning_rates=dict(generator=[group['lr'] for group in loop.policy.opt_g.param_groups],
                                          critic=[group['lr'] for group in loop.policy.opt_d.param_groups]),
                      controller=loop.policy.controller.diagnostics(), export=export,
                      metrics={name: {key: value for key, value in values.items() if key!='records'}
                               for name, values in metrics.items()}, seconds=time.perf_counter()-started)
        reports[arm] = report
        write_json(output/'receipt.json', report)
        write_json(args.output/'receipt.json', dict(plan=plan, arms=reports))
        emit(event='arm_complete', **report)
        del loop, base, fixed_critic
        gc.collect()
        torch.cuda.empty_cache()
    emit(event='complete', arms=list(reports), output=str(args.output), checkpoint_selection='final fixed horizon')


if __name__ == '__main__':
    main()
