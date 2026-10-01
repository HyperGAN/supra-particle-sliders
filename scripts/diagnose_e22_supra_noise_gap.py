#!/usr/bin/env python3
"""Frozen-checkpoint native BF16 diagnostics; no optimizer or row decisions."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from safetensors.torch import load_file
from particlegan import RoutedCandidate
from particlegan.continuous import DataDriftController
from supra.runtime import model_module, TARGETS, load_adapter_state
from supra.particle_export import load_particle_adapter, _EncodedCondition
from supra.particle_training_data import FrozenSliderContexts
from supra.particle_game import ConditionalTokenCritic, patchify


def emit(**row):
    print(json.dumps(row, default=str), flush=True)


def move(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: move(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [move(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(move(v, device) for v in value)
    return value


@torch.no_grad()
def ordinary_velocity(model, inputs):
    z, t, ctx, mask, uctx, umask, _ = inputs
    b = len(z)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        v = model(torch.cat((z, z)), torch.cat((t, t)),
                  torch.cat((ctx, uctx.expand(b, -1, -1))),
                  torch.cat((mask, umask.expand(b, -1))))
    conditional, unconditional = v.float().chunk(2)
    return unconditional + 3 * (conditional - unconditional)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', type=Path, default=ROOT/'outputs/e22-full-f459cb6d-6400')
    ap.add_argument('--output', type=Path, default=ROOT/'outputs/e22-convergence-gap/native-noise')
    ap.add_argument('--draws', type=int, default=8)
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        ap.error('choose a fresh diagnostic output')
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    start = time.perf_counter()
    device = torch.device(args.device)
    data = torch.load(args.run/'data.pt', map_location='cpu', weights_only=False)
    saved = torch.load(args.run/'final.pt', map_location='cpu', weights_only=False, mmap=True)
    state = saved['policy']
    emit(event='loaded_state', step=state['completed_steps'])
    mod = model_module()
    # Use the checkpoint's frozen pure-base teacher. No T5/VAE loads or edits
    # to the trained source files are needed for this velocity diagnostic.
    with torch.random.fork_rng(devices=[]), torch.device('meta'):
        base = mod.SupraDiT()
        mod.attach_supra_lora(base, rank=16, alpha=16, targets=TARGETS)
    teacher_state = {k.removeprefix('teacher.'): v for k, v in state['models']['encoder'].items()
                     if k.startswith('teacher.')}
    base.load_state_dict(teacher_state, strict=True, assign=True)
    base.to(device).eval().requires_grad_(False)
    encoder = FrozenSliderContexts(data['text_contexts'], data['text_masks'], base).to(device)
    new = load_particle_adapter(base, args.run/'final-boss-particlegan.safetensors', device=device)
    declared = json.loads((args.run/'run.json').read_text())
    load_adapter_state(base, load_file(declared['original_baseline_path']))
    for m in base.modules():
        if hasattr(m, 'multiplier'):
            m.multiplier = 1.
    critic = ConditionalTokenCritic(data['coordinate_scale']).to(device).eval().requires_grad_(False)
    critic.load_state_dict(state['models']['critic'])
    controller = DataDriftController('dv12')
    controller.load_state_dict(move(state['controller'], device))
    controller_before = deepcopy(controller.state_dict())
    data_stream = torch.Generator()
    data_stream.set_state(saved['data_rng'])
    indices = {}
    for step in range(6401, 6406):
        hold = step % 5 == 0
        ids = torch.randint(len(data['holds' if hold else 'fit']['context']), (4,), generator=data_stream)
        if step in (6401, 6405):
            indices['holds' if hold else 'fit'] = ids
    indices['validation'] = torch.arange(4)
    payload = dict(groups={}, metadata=dict(precision='native CUDA BF16 host; FP32 routing and critic',
                   checkpoint_step=6400, imported_particlegan=str(Path(sys.modules['particlegan'].__file__).resolve()),
                   evaluation_only=True, noise_draws=args.draws,
                   selection='next edit batch6401, next preservation batch6405, first4 validation contexts',
                   noise_sampling='one private CUDA generator72; draws are evaluations, not training replicates'))
    report = dict(metadata=payload['metadata'], groups={})
    branch_output = []
    handle = new.generator.model.register_forward_hook(lambda _, __, value: branch_output.append(value.detach().clone()))
    candidate = RoutedCandidate(new.table, new.router.log_mass, {'log_mass': new.router.log_mass}, averaged=False)
    prior = controller.routed_prior(new.table, new.router.log_mass)
    stream = torch.Generator(device=device).manual_seed(72)
    shared_output_noise = .125 * torch.randn((4, 256, 16), device=device, generator=stream)
    old_receipt = json.loads((args.run/'evaluation/comparison.json').read_text())
    for name, ids in indices.items():
        pool = data[{'validation':'test'}.get(name, name)]
        context = pool['context'][ids].to(device)
        inputs = encoder.unpack(context)
        teacher = encoder.teacher_velocity(context)
        branch_output.clear()
        clean = new.velocity(*inputs[:6], strength=inputs[6], cfg=3.)
        clean_branches = branch_output.pop()
        old = ordinary_velocity(base, inputs)
        cond = encoder.condition(context)
        payload['groups'][name] = dict(context=context.cpu(), new=(clean-teacher).cpu(), old=(old-teacher).cpu())
        torch.save(payload, args.output/'residuals.pt')
        row = dict(indices=ids.tolist(), clean_new_rmse=float((clean-teacher).square().mean().sqrt()),
                   old_rmse=float((old-teacher).square().mean().sqrt()))
        if name == 'validation':
            row['native_receipt_max_mse_difference'] = {
                arm: max(abs(float((pred-teacher).square().flatten(1).mean(1)[i]) - old_receipt['validation'][arm]['records'][i]['mse'])
                         for i in range(4)) for arm, pred in (('new_particlegan', clean), ('old_lora', old))}
            report['groups'][name] = row
            emit(event='validation_parity', **row)
            continue
        # Avoid changing native policy clocks or streams. This is the public
        # routed replay with a private perturbation stream and diagnostics off.
        condition = _EncodedCondition(*[inputs[i] for i in (0, 2, 3, 4, 5)], inputs[6], 3.)
        packed = torch.cat((inputs[0].flatten(1), inputs[1][:, None]), dim=1)
        models = dict(generator=new.generator, router=new.router, conditioning=condition)
        noisy, branch_deltas, atom_records = [], [], []
        def perturb(codes):
            perturbed = controller.perturb_latent(codes, stream, prior, record=False)
            if len(noisy) == 0:
                distances = (codes[:, None] - prior._mass_support[None]).square().sum(-1)
                nearest = distances.min(1).values.sqrt()
                atom_records.append(dict(codes=len(codes), exact_atom_codes=int((nearest == 0).sum()),
                                         near_atom_codes=int((nearest < 1e-6).sum()),
                                         code_rms=float(codes.square().mean().sqrt()),
                                         perturbation_rms=float((perturbed-codes).square().mean().sqrt())))
            return perturbed
        for draw in range(args.draws):
            branch_output.clear()
            value = new.routing.forward(models, packed, candidate, perturb_fn=perturb)
            noisy.append(value)
            branch_deltas.append(branch_output.pop()-clean_branches)
            emit(event='noise_draw', group=name, draw=draw+1, draws=args.draws,
                 residual_rmse=float((value-teacher).square().mean().sqrt()))
        values, branches = torch.stack(noisy), torch.stack(branch_deltas)
        mean = values.mean(0)
        centered = values - mean
        bc, bu = branches[:, :len(context)], branches[:, len(context):]
        cc, cu = bc-bc.mean(0), bu-bu.mean(0)
        vc, vu, cov = cc.square().mean(), cu.square().mean(), (cc*cu).mean()
        error = patchify(clean-teacher)/critic.scale
        real_logits = critic(shared_output_noise, cond)
        clean_game = torch.nn.functional.softplus(real_logits-critic(shared_output_noise+error, cond)).mean()
        noisy_losses = [float(torch.nn.functional.softplus(real_logits-critic(shared_output_noise+patchify(v-teacher)/critic.scale, cond)).mean()) for v in values]
        row.update(noisy_mean_bias_rmse=float((mean-clean).square().mean().sqrt()),
                   noisy_centered_rms=float(centered.square().mean().sqrt()),
                   noisy_single_draw_clean_difference_rms=float((values-clean).square().mean().sqrt()),
                   noisy_mean_residual_rmse=float((mean-teacher).square().mean().sqrt()),
                   clean_g_game_loss=float(clean_game), noisy_g_game_losses=noisy_losses,
                   conditional_variance=float(vc), unconditional_variance=float(vu), branch_covariance=float(cov),
                   guided_variance_observed=float(centered.square().mean()),
                   guided_variance_identity=float(9*vc+4*vu-12*cov),
                   cfg_amplification_over_mean_branch_variance=float(centered.square().mean()/((vc+vu)/2).clamp_min(1e-30)),
                   first_draw_site_noise=atom_records)
        payload['groups'][name]['noisy'] = (values-teacher).cpu()
        torch.save(payload, args.output/'residuals.pt')
        report['groups'][name] = row
        (args.output/'noise-gap.json').write_text(json.dumps(report, indent=2)+'\n')
        emit(event='group_complete', group=name, **{k:v for k,v in row.items() if k!='first_draw_site_noise'})
    handle.remove()
    assert all(torch.equal(v, controller.state_dict()[k]) if isinstance(v, torch.Tensor) else v == controller.state_dict()[k]
               for k,v in controller_before.items()), 'diagnostic changed controller state'
    report.update(controller_unchanged=True, seconds=time.perf_counter()-start,
                  limitations='Eight private noise draws estimate conditional perturbation bias/variance; they do not establish causality without a matched training ablation.')
    (args.output/'noise-gap.json').write_text(json.dumps(report, indent=2)+'\n')
    emit(event='complete', seconds=report['seconds'], output=str(args.output))


if __name__ == '__main__':
    main()
