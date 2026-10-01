#!/usr/bin/env python3
"""Matched CPU critic-only capacity test on native step6400 residual captures.

This is a mechanistic stress test, not a full-model training continuation.
Only private D copies learn on captured DV12 residuals. G, particles, router,
structural guards, data and all original training state remain untouched.
Native RpGAN, saved D Adam moments/LR, KA2/R1/caps/EMA/spike guard are retained;
the application controller and output sigma are frozen at the saved boundary.
No output metric enters updates, stopping or selection. The two predeclared
arms run exactly128 updates, with the same paired panels and 4:1 task phase.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
UPDATES = 128


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def emit(**row):
    print(json.dumps(row, default=str), flush=True)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "outputs/e22-particle-v2-6400")
    parser.add_argument("--particlegan-root", type=Path,
                        default=ROOT / "outputs/e22-convergence-gap/particlegan-cabe2084-source")
    parser.add_argument("--payload", type=Path,
                        default=ROOT / "outputs/e22-particle-final-critic-causal/residuals.pt")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs/e22-particle-final-critic-causal/capacity")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("choose a fresh isolated diagnostic directory")
    sys.path.insert(0, str(args.particlegan_root.resolve()))
    sys.path.insert(1, str(ROOT))
    import torch
    import torch.nn.functional as F
    import particlegan
    from particlegan import Recipe
    from particlegan.continuous import DataDriftController
    from supra.particle_game import ConditionalTokenCritic, apply_critic_penalty, patchify
    from supra.particle_pilot import state_digest
    from diagnose_e22_supra_critic_geometry import scalar_gradient_metrics, condition, summary, cosine
    from experimental_e22_bounded_critic_features import (
        BoundedFeatureCritic, FORMULATION, extend_native_state, checkpoint, restore,
    )
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    source_path = Path(particlegan.__file__).resolve().parent
    if source_path.parent != args.particlegan_root.resolve():
        raise RuntimeError("imported ParticleGAN differs from the declared source")
    source_hashes = {path.name: sha(path) for path in sorted(source_path.glob("*.py"))}
    source_digest = state_digest(source_hashes)
    declared = json.loads((args.run / "run.json").read_text())
    if source_digest != declared["particlegan_source_digest"]:
        raise RuntimeError("source differs from the training checkpoint")
    saved = torch.load(args.run / "final.pt", map_location="cpu", weights_only=False, mmap=True)
    native = saved["policy"]
    data = torch.load(args.run / "data.pt", map_location="cpu", weights_only=False)
    payload = torch.load(args.payload, map_location="cpu", weights_only=False)
    native_sha, payload_sha = sha(args.run / "final.pt"), sha(args.payload)
    if native["completed_steps"] != 6400 or saved["config"].get("architecture") != "linear_modulated_v2":
        raise RuntimeError("diagnostic requires qualified native V2 step6400")
    metadata = payload["metadata"]
    expected = dict(checkpoint_sha256=native_sha, checkpoint_step=6400,
                    dataset_digest=saved["config"]["dataset_digest"],
                    source_commit=declared["particlegan_commit"], evaluation_only=True,
                    frozen_models_and_global_rng_unchanged=True)
    if any(metadata.get(key) != value for key, value in expected.items()) or set(payload["groups"]) != {"fit", "holds", "test"}:
        raise RuntimeError("native payload provenance/groups differ")
    if state_digest(data) != saved["config"]["dataset_digest"]:
        raise RuntimeError("dataset differs from the trained checkpoint")
    args.output.mkdir(parents=True, exist_ok=True)
    before_global = torch.get_rng_state().clone()
    sigma = float(native["last_output_sigma"])
    recipe = Recipe(**native["recipe"])
    groups = {}
    evaluation_panels = {}
    evaluation_stream = torch.Generator().manual_seed(72)
    for name in ("fit", "holds", "test"):
        group = payload["groups"][name]
        cond = condition(data, group["context"])
        if not torch.allclose(cond, group["condition"], atol=1e-6, rtol=1e-5):
            raise RuntimeError("CPU condition reconstruction differs from native capture")
        if group["new"].shape != (4, 4, 32, 32) or group["noisy"].shape != (4, 4, 4, 32, 32):
            raise RuntimeError("payload requires four contexts and four native DV12 draws")
        groups[name] = dict(condition=group["condition"].float(),
            clean=patchify(group["new"].float()) / data["coordinate_scale"],
            dv12=torch.stack([patchify(value.float()) / data["coordinate_scale"] for value in group["noisy"]]))
        evaluation_panels[name] = sigma * torch.randn((4, 4, 256, 16), generator=evaluation_stream)
    # These private CPU panels are a common quadrature, not recreated native
    # CUDA training draws. No training seed or saved RNG is changed.
    training_stream = torch.Generator().manual_seed(72)
    training_panels = [sigma * torch.randn((4, 256, 16), generator=training_stream) for _ in range(UPDATES)]
    panel_digest = state_digest(training_panels)
    plan = dict(step=6400, updates=UPDATES, arms=["native_critic_v1", FORMULATION],
        only_critic_copies_updated=True, output_sigma=sigma, batch_contexts=4,
        data_sampling="Repeated captured next edit/hold minibatches; no test contexts in any update.",
        residual_sampling="Cycle the four captured native DV12 residuals in each task.",
        gaussian_sampling="Identical private CPU72 paired panels in both arms; no training RNG advance.",
        controller="Frozen saved application controller; native private KA2/EMA/spike guard evolve.",
        new_parameters=16, bounded_features="tanh(a)+tanh(gain)*tanh(a/4), absolute feature bound2",
        metric_policy="Output norms/alignment are read-only diagnostics; RpGAN+KA2 alone train D.",
        selection="Fixed budget, no checkpoint selection or automatic early stopping.",
        limits="This isolates critic response capacity, not whole-model convergence. A gain parameter also changes native KA2 tensorwise surprise topology once active.")
    write_json(args.output / "plan.json", plan)
    began = time.perf_counter()

    def build(arm):
        with torch.random.fork_rng(devices=[]):
            cls = ConditionalTokenCritic if arm == "native_critic_v1" else BoundedFeatureCritic
            critic = cls(data["coordinate_scale"]).float()
            critic_state, optimizer_state = deepcopy(native["models"]["critic"]), deepcopy(native["optimizers"][1])
            if arm == FORMULATION:
                critic_state, optimizer_state = extend_native_state(critic_state, optimizer_state, critic)
            critic.load_state_dict(critic_state, strict=True)
            optimizer = recipe.make_critic_optimizer(critic, ema_critic=deepcopy(critic), foreach=False)
            optimizer.load_state_dict(optimizer_state)
            controller = DataDriftController("dv12")
            controller.load_state_dict(deepcopy(native["controller"]))
            optimizer.continuous_controller = controller
            penalty = recipe.make_critic_penalty(optimizer, collect_stats=True)
        return critic, optimizer, controller, penalty

    def allocation(critic, error, cond, panels):
        x = error.detach().clone().requires_grad_(True)
        condition_repeated = cond.repeat(len(panels), 1)
        real_scores = critic(panels.flatten(0, 1), condition_repeated).detach()
        fake_scores = critic((panels + x.unsqueeze(0)).flatten(0, 1), condition_repeated)
        game = F.softplus(real_scores - fake_scores).reshape(len(panels), len(x)).mean(0)
        gradient = torch.autograd.grad(game.sum(), x)[0]
        raw_error, raw_gradient = (x.detach() * critic.scale).reshape(-1, 16), (gradient.detach() / critic.scale).reshape(-1, 16)
        order = raw_error.norm(dim=-1).argsort()
        bins = {}
        for label, lo, hi in (("lower_half", 0., .5), ("middle_40pct", .5, .9), ("largest_10pct", .9, 1.)):
            indices = order[int(len(order) * lo):int(len(order) * hi)]
            e, g = raw_error[indices], raw_gradient[indices]
            bins[label] = dict(patches=len(indices), residual_power_fraction=float(e.square().sum() / raw_error.square().sum()),
                gradient_power_fraction=float(g.square().sum() / raw_gradient.square().sum().clamp_min(1e-30)),
                gradient_to_residual_norm=float(g.norm() / e.norm().clamp_min(1e-30)),
                residual_descent_cosine=float((e * g).sum() / (e.norm() * g.norm()).clamp_min(1e-30)))
        bins["tail_gain_relative_to_lower_half"] = (bins["largest_10pct"]["gradient_to_residual_norm"]
            / max(bins["lower_half"]["gradient_to_residual_norm"], 1e-30))
        return bins

    def geometry(critic):
        before = state_digest(critic.state_dict())
        result = {}
        for name, group in groups.items():
            result[name] = {}
            for label, error, cond, panels in (
                ("clean", group["clean"], group["condition"], evaluation_panels[name]),
                ("dv12", group["dv12"].flatten(0, 1), group["condition"].repeat(4, 1), evaluation_panels[name].repeat(1, 4, 1, 1)),
            ):
                result[name][label] = scalar_gradient_metrics(critic, error, cond, panels)
                result[name][label]["allocation_diagnostic_only"] = allocation(critic, error, cond, panels)
                with torch.no_grad():
                    actual = critic.features((panels + error.unsqueeze(0)).flatten(0, 1), cond.repeat(4, 1))
                    result[name][label]["feature_absolute_max"] = float(actual.abs().max())
                if isinstance(critic, BoundedFeatureCritic) and not bool(actual.abs().max() <= 2):
                    raise RuntimeError("critic capacity violated the declared feature bound")
        if before != state_digest(critic.state_dict()):
            raise RuntimeError("geometry evaluation mutated critic state")
        return result

    def update(critic, optimizer, penalty, index):
        step = 6401 + index
        name = "holds" if step % 5 == 0 else "fit"
        weight = .1 if name == "holds" else 1.
        group = groups[name]
        error = group["dv12"][index % 4]
        real, fake = training_panels[index], training_panels[index] + error
        critic.train()
        game = weight * recipe.make_loss().d_loss(critic(real, group["condition"]), critic(fake, group["condition"]))
        reg = apply_critic_penalty(penalty, critic, real, fake, group["condition"])
        total = game + reg
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        optimizer.step()
        critic.eval()
        if not bool(torch.isfinite(total)) or any(not bool(torch.isfinite(value).all()) for value in critic.state_dict().values()):
            raise RuntimeError("nonfinite isolated critic continuation")
        return dict(index=index, task=name, game_weight=weight, d_game=float(game.detach()),
            penalty=float(reg.detach()), total=float(total.detach()),
            penalty_phase=penalty.last_stats.get("phase"), penalty_calls=optimizer.record.calls,
            controller_alpha=optimizer.record.alpha, controller_weight=optimizer.record.w,
            controller_surprise_ratio=optimizer.record.last_ratio,
            ema_updates=optimizer.record.ema_updates, ema_skips=optimizer.record.ema_skips,
            gain=(None if not isinstance(critic, BoundedFeatureCritic) else critic.bypass[0].detach().tolist()))

    # Initial equality checks cover native scores, structural feature payloads,
    # input derivatives, all old parameter derivatives and the paired EMA.
    native_model, native_opt, _, native_penalty = build("native_critic_v1")
    capacity_model, capacity_opt, _, capacity_penalty = build(FORMULATION)
    initial_checks = {}
    for name, group in groups.items():
        error = group["clean"].detach().clone().requires_grad_(True)
        cond = group["condition"]
        base_features, capacity_features = native_model.features(error, cond), capacity_model.features(error, cond)
        base_score, capacity_score = native_model(error, cond), capacity_model(error, cond)
        if not torch.equal(base_features, capacity_features) or not torch.equal(base_score, capacity_score):
            raise RuntimeError("capacity extension changed the initial critic function")
        old_parameters, new_parameters = tuple(native_model.parameters()), tuple(capacity_model.parameters())
        g0 = torch.autograd.grad(base_score.sum(), (error,) + old_parameters)
        g1 = torch.autograd.grad(capacity_score.sum(), (error,) + new_parameters)
        if any(not torch.equal(a, b) for a, b in zip(g0, g1[:-1])):
            raise RuntimeError("capacity extension changed an existing initial derivative")
        base_ema, cap_ema = native_opt.ema_critic, capacity_opt.ema_critic
        if not torch.equal(base_ema.features(error, cond), cap_ema.features(error, cond)):
            raise RuntimeError("capacity extension changed the initial KA2 anchor")
        initial_checks[name] = dict(features_scores_input_and_old_parameter_derivatives_exact=True,
                                    ema_features_exact=True, initial_gain_derivative_norm=float(g1[-1].norm()))
    initial_geometry = geometry(native_model)
    if geometry(capacity_model) != initial_geometry:
        raise RuntimeError("initial native/capacity geometry differed despite zero gain")
    write_json(args.output / "initial-geometry.json", initial_geometry)
    results = {}
    for arm in ("native_critic_v1", FORMULATION):
        critic, optimizer, controller, penalty = build(arm)
        controller_digest = state_digest(controller.state_dict())
        initial = checkpoint(critic, optimizer, formulation=arm)
        # Exact four-update continuation replay uses tagged diagnostic states.
        first = [update(critic, optimizer, penalty, index) for index in range(4)]
        replay_digest = state_digest(checkpoint(critic, optimizer, formulation=arm))
        restore(critic, optimizer, initial, formulation=arm)
        again = [update(critic, optimizer, penalty, index) for index in range(4)]
        if first != again or replay_digest != state_digest(checkpoint(critic, optimizer, formulation=arm)):
            raise RuntimeError("isolated native critic resume failed exact replay")
        restore(critic, optimizer, initial, formulation=arm)
        rows = []
        with (args.output / (arm + ".jsonl")).open("w") as log:
            for index in range(UPDATES):
                row = update(critic, optimizer, penalty, index)
                rows.append(row)
                log.write(json.dumps(row) + "\n")
                log.flush()
                if (index + 1) % 16 == 0:
                    emit(event="critic_only_progress", arm=arm, completed=index + 1,
                         d_game=row["d_game"], penalty=row["penalty"])
        if state_digest(controller.state_dict()) != controller_digest:
            raise RuntimeError("critic-only diagnostic advanced the application controller")
        final = checkpoint(critic, optimizer, formulation=arm)
        torch.save(final, args.output / (arm + ".pt"))
        result = dict(initial_state_digest=state_digest(initial), final_state_digest=state_digest(final),
            exact_four_update_tagged_resume=True, application_controller_unchanged=True,
            updates=UPDATES, edit_updates=sum(row["task"] == "fit" for row in rows),
            preservation_updates=sum(row["task"] == "holds" for row in rows),
            final_ka2_record=optimizer.record.state_dict(), final_geometry=geometry(critic),
            gain=None if arm == "native_critic_v1" else critic.bypass[0].detach().tolist())
        results[arm] = result
        write_json(args.output / (arm + "-result.json"), result)
    if not torch.equal(before_global, torch.get_rng_state()):
        raise RuntimeError("private diagnostic changed global CPU RNG")
    if sha(args.run / "final.pt") != native_sha or sha(args.payload) != payload_sha:
        raise RuntimeError("diagnostic input files changed")
    if state_digest({path.name: sha(path) for path in sorted(source_path.glob("*.py"))}) != source_digest:
        raise RuntimeError("native ParticleGAN source changed")
    report = dict(plan=plan, checkpoint_sha256=native_sha, payload_sha256=payload_sha,
        script_sha256=sha(__file__), helper_sha256=sha(ROOT / "scripts/experimental_e22_bounded_critic_features.py"),
        particlegan_commit=declared["particlegan_commit"], particlegan_source_digest=source_digest,
        dataset_digest=saved["config"]["dataset_digest"], training_panel_digest=panel_digest,
        initial_checks=initial_checks, initial_geometry=initial_geometry, results=results,
        native_training_inputs_source_and_global_rng_unchanged=True, seconds=time.perf_counter() - began,
        interpretation="Differences measure adaptation of a frozen-residual native critic under added bounded capacity. They cannot by themselves establish faster or more accurate Supra training.")
    write_json(args.output / "receipt.json", report)
    emit(event="complete", output=args.output, seconds=report["seconds"])


if __name__ == "__main__":
    main()
