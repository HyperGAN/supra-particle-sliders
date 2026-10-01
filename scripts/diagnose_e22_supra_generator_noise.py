#!/usr/bin/env python3
"""Read-only game-gradient replay through the saved complete Supra host.

Uses a frozen native critic and private DV12 evaluation draws. There is no
parameter update, reconstruction loss, policy decision, or checkpoint choice.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import torch.nn.functional as F
from particlegan import RoutedCandidate
from particlegan.continuous import DataDriftController
from supra.runtime import model_module, TARGETS
from supra.particle_export import load_particle_adapter, _EncodedCondition
from supra.particle_training_data import FrozenSliderContexts
from supra.particle_game import ConditionalTokenCritic, patchify
from diagnose_e22_supra_noise_gap import move


def cosine(a, b):
    return float(torch.dot(a, b) / (a.norm()*b.norm()).clamp_min(1e-30))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', type=Path, default=ROOT/'outputs/e22-full-f459cb6d-6400')
    ap.add_argument('--payload', type=Path, default=ROOT/'outputs/e22-convergence-gap/native-noise/residuals.pt')
    ap.add_argument('--output', type=Path, default=ROOT/'outputs/e22-convergence-gap/generator-noise.json')
    ap.add_argument('--draws', type=int, default=8)
    args = ap.parse_args()
    if args.output.exists():
        ap.error('choose a fresh diagnostic output')
    torch.set_num_threads(4)
    device = torch.device('cuda:0')
    started = time.perf_counter()
    data = torch.load(args.run/'data.pt', map_location='cpu', weights_only=False)
    saved = torch.load(args.run/'final.pt', map_location='cpu', weights_only=False, mmap=True)
    state = saved['policy']
    payload = torch.load(args.payload, map_location='cpu', weights_only=False)
    with torch.random.fork_rng(devices=[]), torch.device('meta'):
        base = model_module().SupraDiT()
        model_module().attach_supra_lora(base, rank=16, alpha=16, targets=TARGETS)
    base.load_state_dict({k.removeprefix('teacher.'): v for k,v in state['models']['encoder'].items()
                         if k.startswith('teacher.')}, strict=True, assign=True)
    base.to(device).eval().requires_grad_(False)
    encoder = FrozenSliderContexts(data['text_contexts'], data['text_masks'], base).to(device)
    new = load_particle_adapter(base, args.run/'final-boss-particlegan.safetensors', device=device)
    for branch in new.generator.particle_branches():
        for module in (branch.down, branch.bridge, branch.up):
            module.requires_grad_(True)
    new.router.requires_grad_(True)
    new.table.requires_grad_(True)
    critic = ConditionalTokenCritic(data['coordinate_scale']).to(device).eval().requires_grad_(False)
    critic.load_state_dict(state['models']['critic'])
    controller = DataDriftController('dv12')
    controller.load_state_dict(move(state['controller'], device))
    controller_before = deepcopy(controller.state_dict())
    parameters = ([p for p in new.generator.parameters() if p.requires_grad]
                  + list(new.router.parameters()) + [new.table])
    roles = (["branch"] * sum(p.requires_grad for p in new.generator.parameters())
             + ["router"] * len(list(new.router.parameters())) + ["bank"])
    dimensions = [p.numel() for p in parameters]
    role_ids = {role: torch.cat([torch.full((n,), r == role, dtype=torch.bool)
                               for n,r in zip(dimensions, roles)]) for role in set(roles)}
    candidate = RoutedCandidate(new.table, new.router.log_mass, {'log_mass':new.router.log_mass}, averaged=False)
    prior = controller.routed_prior(new.table, new.router.log_mass)
    stream = torch.Generator(device=device).manual_seed(72)
    paired = torch.load(args.payload.parent/'paired-noise.pt', map_location='cpu', weights_only=False)
    output_noise = .125 * paired['generator_base'].to(device)
    report = dict(metadata=dict(checkpoint_step=6400, native_game_only=True, no_optimizer_step=True,
                               private_noise_key=72, draws=args.draws,
                               imported_particlegan=str(Path(sys.modules['particlegan'].__file__).resolve())), groups={})
    for name in ('fit','holds'):
        context = payload['groups'][name]['context'].to(device)
        inputs = encoder.unpack(context)
        with torch.no_grad():
            teacher = encoder.teacher_velocity(context)
            cond = encoder.condition(context)
            real_logits = critic(output_noise, cond)
        condition = _EncodedCondition(*[inputs[i] for i in (0,2,3,4,5)], inputs[6], 3.)
        packed = torch.cat((inputs[0].flatten(1), inputs[1][:,None]), dim=1)
        models = dict(generator=new.generator, router=new.router, conditioning=condition)
        activation_records, handles = {}, []
        def record_activation(site, value):
            v = value.detach()
            activation_records[site] = dict(rms=float(v.square().mean().sqrt()),
                abs_over_3_fraction=float((v.abs()>3).float().mean()),
                tanh_derivative_mean=float((1-v.tanh().square()).mean()))
        for branch in new.generator.particle_branches():
            handles.append(branch.bridge.register_forward_hook(
                lambda _, __, v, site=branch.site: record_activation(site, v)))
        def gradient(perturb):
            prediction = new.routing.forward(models, packed, candidate, perturb_fn=perturb)
            residual = prediction-teacher
            payoff = F.softplus(real_logits-critic(output_noise+patchify(residual)/critic.scale, cond)).mean()
            objective = payoff * (.1 if name == 'holds' else 1.)
            grads = torch.autograd.grad(objective, parameters, allow_unused=True)
            flat = torch.cat([torch.zeros_like(p).flatten() if g is None else g.detach().flatten()
                              for p,g in zip(parameters,grads)]).cpu()
            return flat, float(payoff.detach()), float(residual.detach().square().mean().sqrt())
        clean, clean_payoff, clean_rmse = gradient(None)
        for h in handles:
            h.remove()
        noisy, payoffs = [], []
        for draw in range(args.draws):
            g,p,_ = gradient(lambda codes: controller.perturb_latent(codes,stream,prior,record=False))
            noisy.append(g)
            payoffs.append(p)
            print(json.dumps(dict(event='gradient_draw',group=name,draw=draw+1,
                                  clean_cosine=cosine(g,clean), game_payoff=p)),flush=True)
        noisy = torch.stack(noisy)
        mean = noisy.mean(0)
        def role_stats(ids):
            c,n,m = clean[ids],noisy[:,ids],mean[ids]
            std = (n-m).square().mean(0).sum().sqrt()
            return dict(clean_norm=float(c.norm()), noisy_mean_norm=float(m.norm()),
                noisy_mean_clean_cosine=cosine(c,m),
                draw_clean_cosines=[cosine(v,c) for v in n],
                draw_norms=[float(v.norm()) for v in n],
                centered_noise_norm=float(std),
                mean_to_centered_noise_norm=float(m.norm()/std.clamp_min(1e-30)),
                mean_minus_clean_relative_norm=float((m-c).norm()/c.norm().clamp_min(1e-30)))
        report['groups'][name] = dict(clean_rmse_evaluation_only=clean_rmse,
            clean_game_payoff=clean_payoff,noisy_game_payoffs=payoffs,
            applied_task_weight=.1 if name=='holds' else 1.,
            gradient=role_stats(torch.ones(len(clean),dtype=torch.bool)),
            roles={r:role_stats(ids) for r,ids in role_ids.items()},
            clean_bridge_activations=activation_records)
        args.output.write_text(json.dumps(report,indent=2)+'\n')
    assert all(torch.equal(v,controller.state_dict()[k]) if isinstance(v,torch.Tensor) else v==controller.state_dict()[k]
               for k,v in controller_before.items()), 'controller changed'
    report['controller_unchanged'] = True
    report['seconds'] = time.perf_counter()-started
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(event='complete',output=str(args.output),seconds=report['seconds'])),flush=True)


if __name__ == '__main__':
    main()
