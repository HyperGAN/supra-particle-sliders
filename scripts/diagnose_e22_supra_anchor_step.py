"""Matched native critic-step counterfactual from saved CUDA Supra residuals.

Only the critic and its EMA/Adam state are reconstructed on CPU. The sole
intervention is the native KA2 anchor_weight override; host outputs are fixed.
"""
from argparse import ArgumentParser
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import torch


def flatten(values):
    return torch.cat([value.detach().reshape(-1).double() for value in values])


def gradients(value, parameters):
    result = torch.autograd.grad(value, parameters, retain_graph=True, allow_unused=True)
    return flatten([torch.zeros_like(parameter) if gradient is None else gradient
                    for parameter, gradient in zip(parameters, result)])


def cosine(left, right):
    denominator = left.norm() * right.norm()
    return None if denominator == 0 else float(left.dot(right) / denominator)


def norm_ratio(numerator, denominator):
    return None if denominator.norm() == 0 else float(numerator.norm() / denominator.norm())


def g_game(critic, residual, condition, base, sigma, weight, loss, patchify):
    residual = residual.detach().clone().requires_grad_(True)
    flags = [parameter.requires_grad for parameter in critic.parameters()]
    try:
        critic.requires_grad_(False)
        real = sigma * base
        with torch.no_grad():
            real_scores = critic(real, condition)
        fake = real + patchify(residual) / critic.scale
        value = weight * loss.g_loss(critic(fake, condition), real_scores)
        gradient = torch.autograd.grad(value, residual)[0]
        return float(value.detach()), gradient.detach().flatten().double()
    finally:
        for parameter, flag in zip(critic.parameters(), flags):
            parameter.requires_grad_(flag)


def arm(state, context, d_residual, g_residual, d_base, g_base, sigma, game_weight, anchor_weight):
    from particlegan.recipes import Recipe
    from particlegan.continuous import DataDriftController, OptimizerSurprise
    from supra.particle_game import ConditionalTokenCritic, apply_critic_penalty, patchify

    recipe = Recipe(**state["recipe"])
    critic_state = state["models"]["critic"]
    critic = ConditionalTokenCritic(critic_state["scale"]).train()
    critic.load_state_dict(critic_state)
    optimizer = recipe.make_critic_optimizer(critic, ema_critic=deepcopy(critic), foreach=False)
    optimizer.load_state_dict(deepcopy(state["optimizers"][1]))
    controller = DataDriftController("dv12")
    controller.load_state_dict(deepcopy(state["controller"]))
    controller.observe_game(optimizer.record)
    controller.observe_blind()
    surprise = OptimizerSurprise()
    surprise.load_state_dict(deepcopy(state["surprise"]))
    assert not surprise.decide(state["completed_steps"]), "This diagnostic does not implement a pending R1 intervention"
    scale = max(state["lr_settle"][1][0]["s"], .75 * state["lr_settle"][0][2]["s"])
    for group, initial_lr in zip(optimizer.param_groups, state["initial_lrs"][1]):
        group["lr"] = initial_lr * scale * controller.critic_scale()
    penalty_fn = recipe.make_critic_penalty(optimizer, collect_stats=True, anchor_weight=anchor_weight)
    penalty_fn.regularizer.continuous_controller = controller

    encoder = state["models"]["encoder"]
    ids = context[:, 4097].long()
    text, mask = encoder["contexts"][ids], encoder["masks"][ids]
    pooled = (text * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
    condition = torch.cat((pooled, context[:, 4096:4097]), dim=-1)
    real = sigma * d_base
    fake = real + patchify(d_residual) / critic.scale
    loss = recipe.make_loss()
    penalty = apply_critic_penalty(penalty_fn, critic, real, fake, condition)
    game_unscaled = loss.d_loss(critic(real, condition), critic(fake, condition))
    game = game_weight * game_unscaled
    parameters = list(critic.parameters())
    game_grad = gradients(game, parameters)
    unscaled_game_grad = gradients(game_unscaled, parameters)
    penalty_grad = gradients(penalty, parameters)
    before = flatten(parameters)
    g_before, g_grad_before = g_game(critic, g_residual, condition, g_base, sigma, game_weight, loss, patchify)
    total = game + penalty
    optimizer.zero_grad(set_to_none=True)
    total.backward()
    total_grad_before_guard = flatten([parameter.grad for parameter in parameters])
    initial_clipped = optimizer.guard.clipped_tensors
    optimizer.step()
    step = flatten(parameters) - before
    with torch.no_grad():
        game_after = game_weight * loss.d_loss(critic(real, condition), critic(fake, condition))
    g_after, g_grad_after = g_game(critic, g_residual, condition, g_base, sigma, game_weight, loss, patchify)
    report = dict(anchor_weight=anchor_weight, game_weight=game_weight,
                  critic_lr=optimizer.param_groups[0]["lr"],
                  controller_game_trust=controller.game_trust,
                  controller_data_drive=controller.data_drive,
                  loss_d_game_unscaled=float(game_unscaled.detach()), loss_d_game=float(game.detach()),
                  penalty=float(penalty.detach()), native_penalty_stats=penalty_fn.last_stats,
                  game_gradient_norm=float(game_grad.norm()),
                  unscaled_game_gradient_norm=float(unscaled_game_grad.norm()),
                  penalty_gradient_norm=float(penalty_grad.norm()),
                  total_gradient_norm=float(total_grad_before_guard.norm()),
                  cosine_penalty_game=cosine(penalty_grad, game_grad),
                  penalty_to_weighted_game_gradient_ratio=norm_ratio(penalty_grad, game_grad),
                  penalty_to_unscaled_game_gradient_ratio=norm_ratio(penalty_grad, unscaled_game_grad),
                  native_critic_step_norm=float(step.norm()),
                  cosine_step_adversarial_descent=cosine(step, -game_grad),
                  adversarial_first_order_change=float(game_grad.dot(step)),
                  loss_d_game_after_step=float(game_after),
                  loss_d_game_change=float(game_after - game.detach()),
                  newly_guard_clipped_tensors=optimizer.guard.clipped_tensors-initial_clipped,
                  loss_g_before=g_before, loss_g_after=g_after,
                  g_residual_gradient_norm_before=float(g_grad_before.norm()),
                  g_residual_gradient_norm_after=float(g_grad_after.norm()),
                  g_residual_gradient_change_ratio=norm_ratio(g_grad_after-g_grad_before, g_grad_before),
                  cosine_g_residual_gradient_before_after=cosine(g_grad_before, g_grad_after),
                  ka2_alpha_after=optimizer.record.alpha,
                  native_ka2_ema_updated=optimizer.record.ema_updates != state["optimizers"][1]["regularizer"]["record"]["ema_updates"])
    vectors = dict(game=game_grad, unscaled_game=unscaled_game_grad, penalty=penalty_grad,
                   step=step, g_before=g_grad_before, g_after=g_grad_after,
                   parameters_before=before, optimizer_state=optimizer.state_dict())
    return report, vectors


def main():
    parser = ArgumentParser()
    parser.add_argument("--particlegan-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--residuals", type=Path, required=True)
    parser.add_argument("--paired-noise", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.particlegan_root.resolve()))
    sys.path.insert(1, str(Path(__file__).resolve().parents[1]))
    torch.set_num_threads(1)
    import particlegan
    assert Path(particlegan.__file__).resolve().is_relative_to(args.particlegan_root.resolve())
    state = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=False)["policy"]
    payload = torch.load(args.residuals, map_location="cpu", weights_only=False)
    paired = torch.load(args.paired_noise, map_location="cpu", weights_only=False)
    d_base, g_base = paired["critic_base"], paired["generator_base"]
    sigma = paired["metadata"].get("sigma", state["last_output_sigma"])
    results = {}
    for name, game_weight in (("fit", 1.), ("holds", .1)):
        group = payload["groups"][name]
        reports, vectors = {}, {}
        for label, anchor_weight in (("native", 1.), ("without_anchor", 0.)):
            reports[label], vectors[label] = arm(
                state, group["context"], group["noisy"][0], group["noisy"][1],
                d_base, g_base, sigma, game_weight, anchor_weight)
        native, off = vectors["native"], vectors["without_anchor"]
        assert torch.equal(native["parameters_before"], off["parameters_before"])
        assert torch.equal(native["game"], off["game"])
        assert torch.equal(native["g_before"], off["g_before"])
        caps, prox = off["penalty"], native["penalty"] - off["penalty"]
        decomposition = dict(
            method="Difference of two native KA2 penalties at identical critic state; anchor_weight is the only override",
            cap_gradient_norm=float(caps.norm()), prox_gradient_norm=float(prox.norm()),
            cap_to_weighted_game_gradient_ratio=norm_ratio(caps, native["game"]),
            prox_to_weighted_game_gradient_ratio=norm_ratio(prox, native["game"]),
            cosine_caps_game=cosine(caps, native["game"]),
            cosine_prox_game=cosine(prox, native["game"]),
            cosine_caps_prox=cosine(caps, prox),
            prox_projection_on_game=float(prox.dot(native["game"])),
            step_difference_norm=float((native["step"]-off["step"]).norm()),
            step_difference_ratio=norm_ratio(native["step"]-off["step"], native["step"]),
            cosine_native_off_step=cosine(native["step"], off["step"]),
            g_gradient_after_step_difference_norm=float((native["g_after"]-off["g_after"]).norm()),
            g_gradient_after_step_difference_ratio=norm_ratio(native["g_after"]-off["g_after"], native["g_after"]),
            cosine_native_off_g_gradient=cosine(native["g_after"], off["g_after"]),
        )
        results[name] = dict(arms=reports, decomposition=decomposition)
        print(json.dumps(dict(group=name, native_game_change=reports["native"]["loss_d_game_change"],
                              without_anchor_game_change=reports["without_anchor"]["loss_d_game_change"],
                              cosine_prox_game=decomposition["cosine_prox_game"])), flush=True)
    report = dict(
        checkpoint_step=state["completed_steps"],
        imported_particlegan=str(Path(particlegan.__file__).resolve()),
        input_sha256={name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in
                      (("residuals", args.residuals), ("paired_noise", args.paired_noise))},
        precision="Saved native CUDA BF16 host / FP32 routing residuals; FP32 CPU critic and native Adam step",
        pairing="Same frozen critic/EMA/Adam, context, DV12 residual samples, paired CUDA Gaussian bases and effective next LR",
        selection="Predeclared first DV12 sample for D and second for G; no metric selection",
        paired_noise_scope="Exact next6401 D/G bases used on both fit and holds; holds does not claim actual6405 noise replay",
        gradient_scope="G adversarial gradient with respect to native student velocity residual; no host parameter Jacobian reconstructed",
        results=results,
        limits=["Two fixed batches and one critic update identify local effects, not persistent game repair.",
                "CPU critic arithmetic can differ slightly from CUDA FP32.",
                "Only anchor_weight differs between arms; no output-MSE loss, guard or selection is used.",
                "The noisy residuals are independent recorded DV12 diagnostic draws, not a claimed training RNG replay."],
        output_metrics_used=False, gpu_used=False, saved_training_state_modified=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(dict(output=str(args.output))), flush=True)


if __name__ == "__main__":
    main()
