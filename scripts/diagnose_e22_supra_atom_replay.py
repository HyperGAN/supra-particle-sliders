#!/usr/bin/env python3
"""Frozen native DV12 support-atom counterfactuals; no training decisions."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from particlegan import RoutedCandidate, get_recipe
from particlegan.continuous import DataDriftController
from supra.runtime import model_module, TARGETS
from supra.particle_export import load_particle_adapter, _EncodedCondition
from supra.particle_training_data import FrozenSliderContexts
from supra.particle_game import ConditionalTokenCritic, patchify
from diagnose_e22_supra_noise_gap import move


def emit(**value):
    print(json.dumps(value, default=str), flush=True)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', type=Path, default=ROOT/'outputs/e22-full-f459cb6d-6400')
    ap.add_argument('--source', type=Path, default=ROOT/'outputs/e22-convergence-gap/native-noise')
    ap.add_argument('--output', type=Path, default=ROOT/'outputs/e22-convergence-gap/native-atom-replay')
    ap.add_argument('--draws', type=int, default=8)
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        ap.error('choose a fresh diagnostic output')
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    started = time.perf_counter()
    device = torch.device(args.device)
    data = torch.load(args.run/'data.pt', map_location='cpu', weights_only=False)
    saved = torch.load(args.run/'final.pt', map_location='cpu', weights_only=False, mmap=True)
    state = saved['policy']
    previous = torch.load(args.source/'residuals.pt', map_location='cpu', weights_only=False)
    mod = model_module()
    with torch.random.fork_rng(devices=[]), torch.device('meta'):
        base = mod.SupraDiT()
        mod.attach_supra_lora(base, rank=16, alpha=16, targets=TARGETS)
    teacher_state = {k.removeprefix('teacher.'):v for k,v in state['models']['encoder'].items()
                     if k.startswith('teacher.')}
    base.load_state_dict(teacher_state, strict=True, assign=True)
    base.to(device).eval().requires_grad_(False)
    encoder = FrozenSliderContexts(data['text_contexts'], data['text_masks'], base).to(device)
    new = load_particle_adapter(base, args.run/'final-boss-particlegan.safetensors', device=device)
    critic = ConditionalTokenCritic(data['coordinate_scale']).to(device).eval().requires_grad_(False)
    critic.load_state_dict(state['models']['critic'])
    controller = DataDriftController('dv12')
    controller.load_state_dict(move(state['controller'], device))
    controller_before = deepcopy(controller.state_dict())
    candidate = RoutedCandidate(new.table, new.router.log_mass, {'log_mass':new.router.log_mass})
    prior = controller.routed_prior(new.table, new.router.log_mass)
    loss = get_recipe('e22_routed').make_loss()
    native_stream = torch.Generator(device=device).manual_seed(72)
    paired_noise = .125*torch.randn((4,256,16),device=device,generator=native_stream)
    hot = 'blocks.8.cross_attn.q'
    report = dict(metadata=dict(checkpoint_step=state['completed_steps'],
        particlegan=str(Path(sys.modules['particlegan'].__file__).resolve()),
        draws=args.draws, precision='native CUDA BF16 host; FP32 routing and critic',
        evaluation_only=True, hot_site=hot,
        intervention='compute every native draw, then suppress selected displacement; matched private RNG state',
        contexts='saved noise-panel next fit and preservation batches; no new samples'),groups={})
    payload = dict(metadata=report['metadata'],groups={})
    arms = ('native', 'suppress_exact_atoms', 'suppress_hot_site')
    for group in ('fit','holds'):
        context = previous['groups'][group]['context'].to(device)
        inputs = encoder.unpack(context)
        teacher = encoder.teacher_velocity(context)
        clean = new.velocity(*inputs[:6],strength=inputs[6],cfg=3.)
        condition = _EncodedCondition(*[inputs[i] for i in (0,2,3,4,5)],inputs[6],3.)
        packed = torch.cat((inputs[0].flatten(1),inputs[1][:,None]),dim=1)
        models = dict(generator=new.generator,router=new.router,conditioning=condition)
        cond = encoder.condition(context)
        real_logits = critic(paired_noise,cond)
        payoff = lambda v: float(loss.g_loss(critic(paired_noise+patchify(v-teacher)/critic.scale,cond),real_logits))
        clean_game = payoff(clean)
        start_state = native_stream.get_state()
        group_report = dict(clean_rmse=float((clean-teacher).square().mean().sqrt()),
                            clean_g_game_loss=clean_game,arms={})
        group_payload = dict(context=context.cpu(),clean=(clean-teacher).cpu(),arms={})
        native_values = None
        native_end_state = None
        for arm in arms:
            stream = torch.Generator(device=device)
            stream.set_state(start_state)
            values,counts = [],[]
            for draw in range(args.draws):
                position = 0
                exact_total,hot_exact,total,suppressed = 0,0,0,0
                def perturb(codes):
                    nonlocal position,exact_total,hot_exact,total,suppressed
                    site = new.generator.sites[position]
                    position += 1
                    # Calling the original law first preserves each native
                    # randn draw and its local-radius computation.
                    result = controller.perturb_latent(codes,stream,prior,record=False)
                    distance = (codes[:,None]-prior._mass_support[None]).square().sum(-1)
                    exact = distance.min(1).values == 0
                    number = int(exact.sum())
                    exact_total += number
                    total += len(codes)
                    if site == hot:
                        hot_exact = number
                    if arm == 'suppress_exact_atoms':
                        suppressed += number
                        return torch.where(exact[:,None],codes,result)
                    if arm == 'suppress_hot_site' and site == hot:
                        suppressed += len(codes)
                        return codes
                    return result
                value = new.routing.forward(models,packed,candidate,perturb_fn=perturb)
                assert position == len(new.generator.sites)
                values.append(value)
                counts.append(dict(exact_atom_codes=exact_total,hot_site_exact_codes=hot_exact,
                                   total_codes=total,suppressed_codes=suppressed))
                emit(event='draw',group=group,arm=arm,draw=draw+1,
                     g_game_loss=payoff(value),**counts[-1])
            values = torch.stack(values)
            if arm == 'native':
                native_values = values
                native_end_state = stream.get_state()
            assert torch.equal(stream.get_state(),native_end_state), 'arm changed native draw progression'
            mean = values.mean(0)
            payoffs = [payoff(value) for value in values]
            row = dict(noisy_mean_bias_rmse=float((mean-clean).square().mean().sqrt()),
                noisy_centered_rms=float((values-mean).square().mean().sqrt()),
                single_draw_clean_difference_rms=float((values-clean).square().mean().sqrt()),
                noisy_mean_residual_rmse=float((mean-teacher).square().mean().sqrt()),
                mean_single_draw_residual_mse=float((values-teacher).square().mean()),
                paired_native_velocity_difference_rms=float((values-native_values).square().mean().sqrt()),
                g_game_losses=payoffs,mean_g_game_loss=sum(payoffs)/len(payoffs),counts=counts,
                native_rng_progression_equal=True)
            if arm == 'native':
                actual = (values-teacher).cpu()
                row['previous_panel_max_residual_difference'] = float((actual-previous['groups'][group]['noisy']).abs().max())
                assert row['previous_panel_max_residual_difference'] == 0., 'native replay failed previous-panel parity'
            group_report['arms'][arm] = row
            group_payload['arms'][arm] = (values-teacher).cpu()
            emit(event='arm_complete',group=group,arm=arm,**{k:v for k,v in row.items() if k not in ('counts','g_game_losses')})
        native_stream.set_state(native_end_state)
        report['groups'][group] = group_report
        payload['groups'][group] = group_payload
        (args.output/'atom-replay.json').write_text(json.dumps(report,indent=2)+'\n')
        torch.save(payload,args.output/'residuals.pt')
    assert all(torch.equal(v,controller.state_dict()[k]) if isinstance(v,torch.Tensor) else v == controller.state_dict()[k]
               for k,v in controller_before.items()), 'diagnostic changed controller state'
    report.update(controller_unchanged=True,seconds=time.perf_counter()-started,
        limitations='Frozen replay identifies function/payoff effects, not long-run training causality or an accepted replacement DV12 law.')
    (args.output/'atom-replay.json').write_text(json.dumps(report,indent=2)+'\n')
    emit(event='complete',seconds=report['seconds'],output=args.output)


if __name__ == '__main__':
    main()
