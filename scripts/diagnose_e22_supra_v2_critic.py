#!/usr/bin/env python3
"""Read-only V2 critic geometry on captured native clean/DV12 residuals.

The payload comes from full native Supra forwards; the critic-only derivatives
run on CPU. Residual alignment is diagnostic only. No optimizer update, output
loss, structural decision or checkpoint selection is performed.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "outputs/e22-particle-noise-update-256/latest")
    parser.add_argument("--particlegan-root", type=Path,
                        default=ROOT / "outputs/e22-convergence-gap/particlegan-cabe2084-source")
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("choose a fresh diagnostic output file")
    sys.path.insert(0, str(args.particlegan_root.resolve()))
    sys.path.insert(1, str(ROOT))
    import torch
    import torch.nn.functional as F
    import particlegan
    from particlegan import Recipe
    from particlegan.continuous import DataDriftController
    from supra.particle_game import ConditionalTokenCritic, apply_critic_penalty, patchify
    from supra.particle_pilot import state_digest
    from diagnose_e22_supra_critic_geometry import (
        condition, cosine, local_origin_geometry, scalar_gradient_metrics, summary,
    )

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    imported = Path(particlegan.__file__).resolve()
    if imported.parent.parent != args.particlegan_root.resolve():
        raise RuntimeError("ParticleGAN import differs from the declared archived source")
    pg_hashes = {path.name: sha(path) for path in sorted(imported.parent.glob("*.py"))}
    declared = json.loads((args.run / "run.json").read_text())
    if state_digest(pg_hashes) != declared["particlegan_source_digest"]:
        raise RuntimeError("native source differs from the checkpoint's training source")
    complete = torch.load(args.run / "final.pt", map_location="cpu", weights_only=False, mmap=True)
    state = complete["policy"]
    data = torch.load(args.run / "data.pt", map_location="cpu", weights_only=False)
    payload = torch.load(args.payload, map_location="cpu", weights_only=False)
    if state_digest(data) != complete["config"]["dataset_digest"] or complete["config"].get("architecture") != "linear_modulated_v2":
        raise RuntimeError("diagnostic requires the checkpoint's matched V2 data")
    checkpoint_sha = sha(args.run / "final.pt")
    provenance = payload.get("metadata", {})
    expected_payload = dict(checkpoint_sha256=checkpoint_sha, checkpoint_step=state["completed_steps"],
                            dataset_digest=complete["config"]["dataset_digest"],
                            source_commit=declared["particlegan_commit"], evaluation_only=True)
    if any(provenance.get(key) != value for key, value in expected_payload.items()):
        raise RuntimeError("native payload does not match this V2 checkpoint/source/data")
    if set(payload["groups"]) != {"fit", "holds", "test"} or any(
            not {"context", "new", "noisy"} <= set(group) for group in payload["groups"].values()):
        raise RuntimeError("native clean/noisy payload is incomplete")
    recipe = Recipe(**state["recipe"])
    with torch.random.fork_rng(devices=[]):
        critic = ConditionalTokenCritic(data["coordinate_scale"]).float().eval()
        critic.load_state_dict(state["models"]["critic"], strict=True)
        optimizer = recipe.make_critic_optimizer(critic, ema_critic=deepcopy(critic), foreach=False)
        optimizer.load_state_dict(deepcopy(state["optimizers"][1]))
        controller = DataDriftController("dv12")
        controller.load_state_dict(state["controller"])
        optimizer.continuous_controller = controller
        penalty = recipe.make_critic_penalty(optimizer, collect_stats=True)
    critic_digest = state_digest(critic.state_dict())
    ema = deepcopy(critic)
    ema.load_state_dict(state["optimizers"][1]["regularizer"]["ema"], strict=True)
    record = deepcopy(state["optimizers"][1]["regularizer"]["record"])
    native_regularizer = deepcopy(state["optimizers"][1])
    sigma = state["last_output_sigma"]
    payload_sha = sha(args.payload)
    before_global = torch.get_rng_state().clone()

    def flat_grad(value):
        parameters = list(critic.parameters())
        gradients = torch.autograd.grad(value, parameters, retain_graph=True, allow_unused=True)
        return torch.cat([torch.zeros_like(parameter).flatten() if gradient is None else gradient.flatten()
                          for parameter, gradient in zip(parameters, gradients)])

    def dot_cos(left, right):
        return float(torch.dot(left, right) / (left.norm() * right.norm()).clamp_min(1e-30))

    def decomposition(error, cond, noise):
        real = noise[0].detach().requires_grad_(True)
        fake = (real.detach() + error).detach().requires_grad_(True)
        # Summed local patch scores are exactly the repeated-score penalty
        # view's input-gradient units, while game logits remain pooled.
        gr = torch.autograd.grad(critic.score(critic.features(real, cond)).sum(), real, create_graph=True)[0]
        gf = torch.autograd.grad(critic.score(critic.features(fake, cond)).sum(), fake, create_graph=True)[0]
        anchor_input = real.detach().clone().requires_grad_(True)
        ga = torch.autograd.grad(ema.score(ema.features(anchor_input, cond)).sum(), anchor_input)[0].detach()
        sqr, sqf = gr.square().sum(-1), gf.square().sum(-1)
        nr, nf = (sqr + 1e-12).sqrt(), (sqf + 1e-12).sqrt()
        coefficient = recipe.reg_coeff / 4
        pieces = dict(real_r1=coefficient * (sqr / 16).mean(),
                      fake_rms_cap=coefficient * F.relu(nf / 4 - recipe.reg_kappa).square().mean(),
                      real_l2_cap=coefficient * F.relu(nr - recipe.reg_kappa).square().mean(),
                      fake_l2_cap=coefficient * F.relu(nf - recipe.reg_kappa).square().mean(),
                      anchor=coefficient * record["w"] * recipe.reg_anchor_weight * (gr - ga).square().sum(-1).mean() / 16)
        game = F.softplus(critic(fake.detach(), cond) - critic(real.detach(), cond)).mean()
        gradients = {key: flat_grad(value) for key, value in pieces.items()}
        game_gradient = flat_grad(game)
        manual = sum(pieces.values())
        # Native replay on an independent record validates this decomposition
        # without advancing a training controller or running an optimizer.
        optimizer.load_state_dict(deepcopy(native_regularizer))
        native = apply_critic_penalty(penalty, critic, real.detach(), fake.detach(), cond)
        native_gradient = flat_grad(native)
        combined = sum(gradients.values())
        if not torch.allclose(manual.detach(), native.detach(), atol=1e-6, rtol=1e-5):
            raise RuntimeError("manual KA2 decomposition differs from native penalty value")
        if not torch.allclose(combined, native_gradient, atol=1e-6, rtol=1e-4):
            raise RuntimeError("manual KA2 decomposition differs from native parameter gradient")
        result = {key: dict(value=float(pieces[key].detach()), gradient_norm=float(value.norm()),
                           cosine_with_unweighted_d_game=dot_cos(value, game_gradient))
                  for key, value in gradients.items()}
        result.update(total_penalty=float(manual.detach()), unweighted_d_game=float(game.detach()),
                      penalty_to_game_gradient_ratio=float(combined.norm() / game_gradient.norm().clamp_min(1e-30)),
                      penalty_game_gradient_cosine=dot_cos(combined, game_gradient),
                      native_value_and_gradient_verified=True,
                      local_real_gradient_norm=summary(nr), local_fake_gradient_norm=summary(nf))
        return result

    def geometry(error, cond, noise):
        result = scalar_gradient_metrics(critic, error, cond, noise)
        # The native game may have a nonzero gradient even at perfect paired
        # matching for a finite trained critic. Compare it with the actual
        # residual gradient; neither diagnostic is used to update parameters.
        def gradient_at(value, panels=noise):
            value = value.detach().clone().requires_grad_(True)
            draws, batch = len(panels), len(value)
            replicated = cond.repeat(draws, 1)
            real_scores = critic(panels.flatten(0, 1), replicated).detach()
            fake_scores = critic((panels + value.unsqueeze(0)).flatten(0, 1), replicated)
            payoff = F.softplus(real_scores - fake_scores).reshape(draws, batch).mean(0)
            return torch.autograd.grad(payoff.sum(), value)[0]
        actual = gradient_at(error)
        origin = gradient_at(torch.zeros_like(error))
        noise_free = gradient_at(error, torch.zeros_like(noise))
        with torch.no_grad():
            draws = len(noise)
            fake = (noise + error.unsqueeze(0)).flatten(0, 1)
            c = cond.repeat(draws, 1)
            local = critic.error_input(fake)
            contextual = critic.condition_input(c).unsqueeze(1)
            preactivation = local + contextual
            norms = dict(error_path_rms=float(local.square().mean().sqrt()),
                         conditioning_path_rms=float(contextual.square().mean().sqrt()),
                         joint_rms=float(preactivation.square().mean().sqrt()),
                         condition_to_error_rms_ratio=float(contextual.square().mean().sqrt() / local.square().mean().sqrt().clamp_min(1e-30)))
            features = critic.features(fake, c)
            score_weight = critic.score.weight.flatten()
            token_scores = (features * score_weight).sum(-1)
            token_score_centered = token_scores - token_scores.mean(1, keepdim=True)
            norms["within_context_local_score_std"] = float(token_score_centered.square().mean().sqrt())
        result["conditioning_geometry"] = norms
        result["perfect_match_gradient_to_actual_norm_ratio"] = summary(
            origin.flatten(1).norm(dim=1) / actual.flatten(1).norm(dim=1).clamp_min(1e-30))
        result["perfect_match_gradient_residual_cosine"] = summary(cosine(origin, error))
        result["actual_gradient_minus_origin_residual_cosine"] = summary(cosine(actual - origin, error))
        raw_error = (error * critic.scale).detach().reshape(-1, 16)
        raw_gradient = (actual / critic.scale).detach().reshape(-1, 16)
        order = raw_error.norm(dim=1).argsort()
        allocation = {}
        for label, low, high in (("lower_half", 0., .5), ("middle_40pct", .5, .9), ("largest_10pct", .9, 1.)):
            indices = order[int(low * len(order)):int(high * len(order))]
            e, g = raw_error[indices], raw_gradient[indices]
            allocation[label] = dict(patches=len(indices),
                residual_power_fraction=float(e.square().sum() / raw_error.square().sum().clamp_min(1e-30)),
                gradient_power_fraction=float(g.square().sum() / raw_gradient.square().sum().clamp_min(1e-30)),
                descent_residual_cosine=float((e * g).sum() / (e.norm() * g.norm()).clamp_min(1e-30)),
                gradient_to_residual_norm=float(g.norm() / e.norm().clamp_min(1e-30)))
        result["patch_gradient_allocation_observational_only"] = allocation
        result["noise_free_gradient_observational_only"] = dict(
            norm_relative_to_native_noise=summary(noise_free.flatten(1).norm(dim=1) / actual.flatten(1).norm(dim=1).clamp_min(1e-30)),
            cosine_with_native_noise_gradient=summary(cosine(noise_free, actual)),
            residual_cosine=summary(cosine(noise_free, error)),
            interpretation="Frozen-critic output-noise sensitivity only; no noise mode or training floor is changed.")
        result["ka2_component_parameter_gradients"] = decomposition(error, cond, noise)
        return result

    results = {}
    for name, group in payload["groups"].items():
        cond = condition(data, group["context"])
        if "condition" in group:
            native_condition = group["condition"].float()
            if not torch.allclose(cond, native_condition, atol=1e-6, rtol=1e-5):
                raise RuntimeError("payload conditioning differs from the checkpoint's source/time conditioning")
            cond = native_condition
        stream = torch.Generator().manual_seed(72)
        noise = sigma * torch.randn((4, len(cond), 256, 16), generator=stream)
        results[name] = {}
        for arm in ("new", "noisy"):
            if arm not in group:
                continue
            value = group[arm]
            if value.ndim == 5:
                errors = [(arm, value.flatten(0, 1), cond.repeat(len(value), 1), noise.repeat(1, len(value), 1, 1)),
                          (arm + "_mean", value.mean(0), cond, noise)]
            else:
                errors = [(arm, value, cond, noise)]
            for label, residual, contexts, draws in errors:
                error = patchify(residual.float()) / critic.scale
                results[name][label] = geometry(error, contexts, draws)
        results[name]["origin_geometry"] = local_origin_geometry(critic, cond, noise)
        results[name]["origin_geometry"]["interpretation"] = (
            "These are mean score derivatives under a shared-coordinate Gaussian quadrature, not the full "
            "RpGAN or parameter-space Hessian. At paired origin H_G = .25 grad(S)grad(S)^T - .5 H_S; "
            "positive score curvature contributes a negative term but the logistic positive term may dominate. "
            "Balanced patch perturbations cancel linear drift in expectation, not on every fixed panel.")
    if critic_digest != state_digest(critic.state_dict()) or not torch.equal(before_global, torch.get_rng_state()):
        raise RuntimeError("diagnostic mutated critic parameters or global CPU RNG")
    if sha(args.run / "final.pt") != checkpoint_sha or sha(args.payload) != payload_sha:
        raise RuntimeError("diagnostic input files changed")
    report = dict(checkpoint=str(args.run / "final.pt"), checkpoint_sha256=checkpoint_sha,
                  payload=str(args.payload), payload_sha256=payload_sha, step=state["completed_steps"],
                  architecture=complete["config"]["architecture"], particlegan_commit=declared["particlegan_commit"],
                  particlegan_source_digest=state_digest(pg_hashes), script_sha256=sha(__file__),
                  precision="CPU float32 critic on native BF16/CFG3 captured residuals",
                  noise="four private CPU72 paired draws; diagnostic sampling, not training replicates", sigma=sigma,
                  native_penalty_record={key: value for key, value in record.items() if key != "sur_hist"},
                  payload_metadata=payload.get("metadata", {}), no_training_or_optimizer_update=True,
                  critic_parameters_and_input_files_unchanged=True, results=results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(output=str(args.output), step=state["completed_steps"], groups=list(results))), flush=True)


if __name__ == "__main__":
    main()
