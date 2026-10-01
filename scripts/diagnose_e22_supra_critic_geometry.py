#!/usr/bin/env python3
"""Read-only CPU geometry of the actual frozen final RpGAN critic.

No optimizer, structural move, model forward, checkpoint choice, or training
seed change is performed. Optional payload groups contain packed ``context``
and native paired residuals named ``new``/``old``. Without a payload, the
cached pure-base edit residual is a realistic scale/shape witness only.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from supra.particle_game import ConditionalTokenCritic, patchify


def condition(data, context):
    ids = context[:, 4097].long()
    text, mask = data['text_contexts'][ids].float(), data['text_masks'][ids].float()
    pooled = (text * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
    return torch.cat((pooled, context[:, 4096:4097].float()), dim=-1)


def summary(values):
    values = torch.as_tensor(values).detach().double().flatten()
    return dict(mean=float(values.mean()), minimum=float(values.min()), maximum=float(values.max()),
                nonpositive=int((values <= 0).sum()), count=values.numel())


def cosine(left, right):
    left, right = left.flatten(1), right.flatten(1)
    return (left * right).sum(1) / (left.norm(dim=1) * right.norm(dim=1)).clamp_min(1e-30)


def scalar_gradient_metrics(critic, error, cond, noise):
    error = error.detach().float().requires_grad_(True)
    draws, batch = len(noise), len(error)
    real = noise.flatten(0, 1)
    fake = (noise + error.unsqueeze(0)).flatten(0, 1)
    replicated_condition = cond.repeat(draws, 1)
    real_logits = critic(real, replicated_condition).detach()
    fake_logits = critic(fake, replicated_condition)
    per_context_loss = F.softplus(real_logits - fake_logits).reshape(draws, batch).mean(0)
    gradient = torch.autograd.grad(per_context_loss.sum(), error)[0]
    centered_error = error - error.mean(1, keepdim=True)
    centered_gradient = gradient - gradient.mean(1, keepdim=True)
    centered_rms_fraction = centered_error.flatten(1).norm(dim=1) / error.flatten(1).norm(dim=1).clamp_min(1e-30)
    centered_gradient_fraction = centered_gradient.flatten(1).norm(dim=1) / gradient.flatten(1).norm(dim=1).clamp_min(1e-30)
    with torch.no_grad():
        h = critic.error_input(fake) + critic.condition_input(replicated_condition).unsqueeze(1)
        f = critic.feature_output(h.tanh())
        local_fake = critic.score(critic.features(fake, replicated_condition)).reshape(draws, batch, 256)
        local_real = critic.score(critic.features(real, replicated_condition)).reshape(draws, batch, 256)
        token_g = F.softplus(local_real - local_fake).mean((0, 2))
        paired_gap = real_logits - fake_logits
        permutation = torch.arange(255, -1, -1)
        permutation_score_difference = (critic(fake[:, permutation], replicated_condition) - fake_logits).abs().max()
    return dict(
        native_g_loss=summary(per_context_loss),
        tokenwise_g_loss_observational_only=summary(token_g),
        real_minus_fake_gap=summary(paired_gap),
        native_g_gradient_norm=summary(gradient.flatten(1).norm(dim=1)),
        descent_reduces_normalized_squared_residual_cosine=summary(cosine(gradient, error)),
        descent_reduces_raw_squared_residual_cosine=summary(cosine(gradient / critic.scale, error * critic.scale)),
        descent_reduces_centered_residual_cosine=summary(cosine(centered_gradient, centered_error)),
        residual_centered_norm_fraction=summary(centered_rms_fraction),
        gradient_centered_norm_fraction=summary(centered_gradient_fraction),
        normalized_residual_rms=summary(error.square().mean((1, 2)).sqrt()),
        first_tanh_saturated_fraction=float((h.abs() > 3).float().mean()),
        first_tanh_derivative_mean=float((1 - h.tanh().square()).mean()),
        second_tanh_saturated_fraction=float((f.abs() > 3).float().mean()),
        second_tanh_derivative_mean=float((1 - f.tanh().square()).mean()),
        patch_permutation_score_max_difference=float(permutation_score_difference),
        interpretation='Positive cosine means the frozen native G output-space descent reduces this residual. '
                       'This is not the parameter-space Adam update and does not establish an architecture ablation.')


def penalty_gradients(critic, ema, error, cond, noise, coefficient, anchor_weight):
    # Exact final blended KA2 formula in the existing repeated-score token
    # view: grad(sum(repeated pooled scores)) = grad(local patch score).
    real = noise[0].detach().requires_grad_(True)
    fake = (real.detach() + error).detach().requires_grad_(True)
    real_local = critic.score(critic.features(real, cond))
    fake_local = critic.score(critic.features(fake, cond))
    gr = torch.autograd.grad(real_local.sum(), real, create_graph=True)[0]
    gf = torch.autograd.grad(fake_local.sum(), fake, create_graph=True)[0]
    anchor_input = real.detach().clone().requires_grad_(True)
    ga = torch.autograd.grad(ema.score(ema.features(anchor_input, cond)).sum(), anchor_input)[0].detach()
    sqr, sqf = gr.square().sum(-1), gf.square().sum(-1)
    nr, nf = (sqr + 1e-12).sqrt(), (sqf + 1e-12).sqrt()
    a = (sqr / 16).mean() + F.relu(nf / 4 - 1).square().mean()
    prox = (gr - ga).square().sum(-1).mean() / 16
    b = F.relu(nr - 1).square().mean() + F.relu(nf - 1).square().mean() + anchor_weight * prox
    penalty = coefficient / 2 * (.5 * a + .5 * b)
    adversarial = F.softplus(critic(fake.detach(), cond) - critic(real.detach(), cond)).mean()
    parameters = list(critic.parameters())
    def parameter_gradient(loss):
        gs = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
        return torch.cat([torch.zeros_like(p).flatten() if g is None else g.flatten() for p, g in zip(parameters, gs)])
    gp, gd = parameter_gradient(penalty), parameter_gradient(adversarial)
    return dict(adversarial_d=float(adversarial.detach()), final_blended_penalty=float(penalty.detach()),
                penalty_parameter_gradient_norm=float(gp.norm()), adversarial_parameter_gradient_norm=float(gd.norm()),
                penalty_to_unweighted_adversarial_gradient_norm=float(gp.norm() / gd.norm().clamp_min(1e-30)),
                penalty_to_hold_weighted_adversarial_gradient_norm=float(gp.norm() / (.1 * gd.norm()).clamp_min(1e-30)),
                penalty_adversarial_gradient_cosine=float(torch.dot(gp, gd) / (gp.norm() * gd.norm()).clamp_min(1e-30)),
                real_r1=float((sqr / 16).mean().detach()), fake_rms_cap_fraction=float((nf > 4).float().mean()),
                blend_real_l2_cap_fraction=float((nr > 1).float().mean()),
                blend_fake_l2_cap_fraction=float((nf > 1).float().mean()), ema_gradient_prox=float(prox.detach()),
                anchor_weight=float(anchor_weight),
                interpretation='Frozen final record with exact 16-coordinate A/B/EMA units; not an optimizer step. '
                               'Hold weighting multiplies the payoff only, making this ratio exactly ten times larger.')


def local_origin_geometry(critic, cond, noise):
    # Derivatives averaged over one fixed antithetic Gaussian quadrature; no
    # model fitting or seed comparison. Cross-token score Hessians are zero.
    base = noise.reshape(-1, 16)[:256].detach()
    z = torch.cat((base, -base)).requires_grad_(True)
    records = []
    selected = torch.linspace(0, len(cond)-1, min(6, len(cond))).round().long()
    for c in cond[selected]:
        c = c.unsqueeze(0)
        def local_mean(delta):
            x = (z + delta).unsqueeze(0)
            return critic(x, c).sum()
        zero = torch.zeros(16, requires_grad=True)
        grad = torch.autograd.functional.jacobian(local_mean, zero)
        hess = torch.autograd.functional.hessian(local_mean, zero)
        eig = torch.linalg.eigvalsh(hess.double())
        records.append(dict(mean_noise_score_gradient_norm=float(grad.norm()),
                            mean_noise_score_hessian_eigenvalues=eig.tolist(),
                            positive_score_curvature_directions=int((eig > 1e-7).sum()),
                            negative_score_curvature_directions=int((eig < -1e-7).sum())))
    return dict(records=records, gaussian_quadrature_samples=len(z), antithetic=True,
                interpretation='For balanced +/- patch residuals the linear score term cancels. Positive local '
                               'score curvature lets a frozen G lower its native loss by departing zero in that '
                               'direction, to second order; the full loss also has the logistic outer-product term. '
                               'This is a critic shape observation, not proof of a different minimax optimum.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=ROOT/'outputs/e22-full-f459cb6d-6400/final.pt')
    parser.add_argument('--data', type=Path, default=ROOT/'outputs/e22-full-f459cb6d-6400/data.pt')
    parser.add_argument('--payload', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)['policy']
    data = torch.load(args.data, map_location='cpu', weights_only=False, mmap=True)
    critic = ConditionalTokenCritic(data['coordinate_scale']).float()
    critic.load_state_dict({k: v.clone() for k, v in state['models']['critic'].items()})
    critic.eval()
    ema = deepcopy(critic)
    regularizer = state['optimizers'][1]['regularizer']
    ema.load_state_dict({k: v.clone() for k, v in regularizer['ema'].items()})
    record = regularizer['record']
    sigma = state['last_output_sigma']
    if args.payload:
        payload = torch.load(args.payload, map_location='cpu', weights_only=False)
        groups, provenance = payload['groups'], payload.get('metadata', {})
    else:
        groups = {}
        for name in ['fit', 'test', 'holds', 'preservation']:
            pool = data[name]
            ids = torch.linspace(0, len(pool['context'])-1, 24).round().long()
            groups[name] = dict(context=pool['context'][ids], cached_pure_base=pool['base'][ids]-pool['targets'][ids])
        provenance = dict(witness='Cached pure-base edit residual, not the trained new/old student residual')
    results = {}
    for name, group in groups.items():
        cond = condition(data, group['context'])
        rng = torch.Generator().manual_seed(72)
        noise = sigma * torch.randn((4, len(cond), 256, 16), generator=rng)
        results[name] = {}
        for arm, residual in group.items():
            if arm == 'context':
                continue
            arms = [(arm, residual, cond, noise)]
            if residual.ndim == 5:
                arms = [(arm, residual.flatten(0, 1), cond.repeat(len(residual), 1),
                         noise.repeat(1, len(residual), 1, 1)),
                        (arm+'_mean', residual.mean(0), cond, noise)]
            for arm_name, native_error, arm_cond, arm_noise in arms:
                error = patchify(native_error.float()) / critic.scale
                results[name][arm_name] = scalar_gradient_metrics(critic, error, arm_cond, arm_noise)
                results[name][arm_name]['penalty_gradient_geometry'] = penalty_gradients(
                    critic, ema, error, arm_cond, arm_noise, state['recipe']['reg_coeff'], record['w'])
        results[name]['origin_geometry'] = local_origin_geometry(critic, cond, noise)
    result = dict(checkpoint=str(args.checkpoint), data=str(args.data), payload=None if args.payload is None else str(args.payload),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  critic_state_sha256=hashlib.sha256(b''.join(v.detach().contiguous().numpy().tobytes()
                      for k, v in sorted(critic.state_dict().items()))).hexdigest(),
                  updates=state['completed_steps'], sigma=sigma, native_penalty_record_calls=record['calls'],
                  provenance=provenance, device='CPU float32', fixed_private_noise_key=72,
                  payload_sha256=None if args.payload is None else hashlib.sha256(args.payload.read_bytes()).hexdigest(),
                  no_training_or_optimizer_step=True, results=results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(output=str(args.output), script_sha256=result['script_sha256'],
                         critic_state_sha256=result['critic_state_sha256'], groups=list(results))))


if __name__ == '__main__':
    main()
