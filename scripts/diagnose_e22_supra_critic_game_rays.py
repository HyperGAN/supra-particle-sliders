#!/usr/bin/env python3
"""Read-only conditional RpGAN rays on qualified native step6400 residuals.

All rays and four private CPU72 paired Gaussian panels are declared in advance.
No model training, optimizer, native controller/RNG advance, output-MSE loss,
checkpoint selection, or seed comparison occurs. Tokenwise softplus is an
observational game-shape comparison, not a replacement training criterion.
"""
from argparse import ArgumentParser
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ALPHAS = (-1., -.5, 0., .25, .5, .75, 1., 1.5, 2., 4.)
TOL = 1e-6


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def emit(**value):
    print(json.dumps(value, allow_nan=False), flush=True)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "outputs/e22-particle-v2-6400")
    parser.add_argument("--payload", type=Path, default=ROOT / "outputs/e22-particle-final-critic-causal/residuals.pt")
    parser.add_argument("--earlier-run", type=Path, default=ROOT / "outputs/e22-particle-noise-update-256/latest")
    parser.add_argument("--particlegan-root", type=Path,
                        default=ROOT / "outputs/e22-convergence-gap/particlegan-cabe2084-source")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/e22-particle-final-critic-causal/game-rays")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("preserve existing evidence; choose a fresh output directory")
    sys.path.insert(0, str(args.particlegan_root.resolve()))
    sys.path.insert(1, str(ROOT))
    import torch
    import torch.nn.functional as F
    import particlegan
    from particlegan import Recipe
    from supra.particle_game import ConditionalTokenCritic, patchify
    from supra.particle_pilot import state_digest
    from diagnose_e22_supra_critic_geometry import condition, summary

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    began = time.perf_counter()
    cpu_before = torch.get_rng_state().clone()
    pg_dir = Path(particlegan.__file__).resolve().parent
    if pg_dir.parent != args.particlegan_root.resolve():
        raise RuntimeError("imported ParticleGAN differs from declared archived source")
    source_hashes = {path.name: sha(path) for path in sorted(pg_dir.glob("*.py"))}
    declared = json.loads((args.run / "run.json").read_text())
    if state_digest(source_hashes) != declared["particlegan_source_digest"]:
        raise RuntimeError("ParticleGAN source differs from qualified native training source")
    checkpoint_path = args.run / "final.pt"
    inputs = {"checkpoint6400": checkpoint_path, "data": args.run / "data.pt", "payload": args.payload,
              "run": args.run / "run.json", "qualification": args.run / "qualification-review.json"}
    hashes = {name: sha(path) for name, path in inputs.items()}
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    data = torch.load(inputs["data"], map_location="cpu", weights_only=False)
    payload = torch.load(args.payload, map_location="cpu", weights_only=False)
    qualification = json.loads(inputs["qualification"].read_text())
    native = saved["policy"]
    if (native["completed_steps"] != 6400 or saved["config"].get("architecture") != "linear_modulated_v2"
            or qualification["qualified"] is not True or state_digest(saved) != qualification["final_native_digest"]):
        raise RuntimeError("requires the complete qualified native V2 step6400 checkpoint")
    if state_digest(data) != saved["config"]["dataset_digest"]:
        raise RuntimeError("cached dataset differs from native trained inputs")
    expected = dict(checkpoint_sha256=hashes["checkpoint6400"], checkpoint_step=6400,
        dataset_digest=saved["config"]["dataset_digest"], source_commit=declared["particlegan_commit"],
        evaluation_only=True, frozen_models_and_global_rng_unchanged=True)
    if any(payload["metadata"].get(key) != value for key, value in expected.items()):
        raise RuntimeError("capture provenance differs from the qualified native endpoint")
    if set(payload["groups"]) != {"fit", "holds", "test"}:
        raise RuntimeError("requires all three native capture pools")
    sigma = float(native["last_output_sigma"])
    loss = Recipe(**native["recipe"]).make_loss()
    critics = {}
    with torch.random.fork_rng(devices=[]):
        critic = ConditionalTokenCritic(data["coordinate_scale"]).float().eval().requires_grad_(False)
        critic.load_state_dict(native["models"]["critic"], strict=True)
        critics["D6400"] = critic
        if args.earlier_run is not None:
            earlier_path = args.earlier_run / "final.pt"
            inputs["checkpoint1856"] = earlier_path
            hashes["checkpoint1856"] = sha(earlier_path)
            earlier = torch.load(earlier_path, map_location="cpu", weights_only=False, mmap=True)
            if (earlier["policy"]["completed_steps"] != 1856
                    or earlier["config"]["dataset_digest"] != saved["config"]["dataset_digest"]
                    or earlier["config"].get("architecture") != "linear_modulated_v2"):
                raise RuntimeError("optional earlier critic must be the matching V2 step1856")
            prior = deepcopy(critic)
            prior.load_state_dict(earlier["policy"]["models"]["critic"], strict=True)
            critics["D1856_on_final6400_residuals"] = prior
    before_critic = {name: state_digest(model.state_dict()) for name, model in critics.items()}
    stream = torch.Generator().manual_seed(72)
    groups = {}
    for name in ("fit", "holds", "test"):
        group = payload["groups"][name]
        reconstructed = condition(data, group["context"])
        if not torch.allclose(reconstructed, group["condition"], atol=1e-6, rtol=1e-5):
            raise RuntimeError("CPU condition reconstruction differs from the native capture")
        if group["new"].shape != (4, 4, 32, 32) or group["noisy"].shape != (4, 4, 4, 32, 32):
            raise RuntimeError("requires four contexts and four native DV12 samples")
        panels = sigma * torch.randn((4, 4, 256, 16), generator=stream)
        groups[name] = dict(context=group["context"], condition=group["condition"].float(),
            clean=patchify(group["new"].float()) / data["coordinate_scale"],
            dv12=patchify(group["noisy"].flatten(0, 1).float()) / data["coordinate_scale"], noise=panels)
    args.output.mkdir(parents=True)
    plan = dict(schema="supra_frozen_critic_game_rays_v1", device="CPU float32", alphas=list(ALPHAS),
        critics=list(critics), residual_boundary=6400, contexts_per_pool=4, dv12_draws=4, gaussian_panels=4,
        gaussian="One private CPU72 stream, sequential fit/holds/test panels; no training seed or stream change.",
        criteria="Read-only native paired G softplus and its derivatives; tokenwise softplus observational only.",
        output_metrics_used_for_optimizer_or_selection=False, training_or_optimizer_updates=0,
        tolerance=TOL, alpha_zero_identity="real equals fake, so every G game equals log2",
        limits="Twelve captured source/time contexts, plus48 correlated DV12 variants; not full-pool quality or a trainable optimizer intervention.")
    (args.output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    raw, results = {}, {}

    def evaluate(model, error, cond, panels):
        batch, draws, tokens = len(error), len(panels), error.shape[1]
        repeated = cond.repeat(draws, 1)
        real = panels.flatten(0, 1)
        with torch.no_grad():
            real_logits = model(real, repeated)
            real_features = model.features(real, repeated)
            real_local = model.score(real_features).reshape(draws, batch, tokens)
        points, arrays = [], []
        for alpha in ALPHAS:
            x = (alpha * error).detach().requires_grad_(True)
            fake = (panels + x.unsqueeze(0)).flatten(0, 1)
            fake_features = model.features(fake, repeated)
            fake_logits = model.score(fake_features.mean(1))
            fake_local = model.score(fake_features).reshape(draws, batch, tokens)
            gaps = (real_logits - fake_logits).reshape(draws, batch)
            local_gap = real_local - fake_local
            pooled = F.softplus(gaps).mean(0)
            tokenwise = F.softplus(local_gap).mean((0, 2))
            native_value = loss.g_loss(fake_logits, real_logits)
            if not torch.equal(native_value, F.softplus(real_logits - fake_logits).mean()):
                raise RuntimeError("native loss differs from the explicit paired-softplus audit")
            with torch.no_grad():
                # Pooling before/after the linear score is algebraically equal.
                # Native FP32 scores have a common offset near6 on preservation,
                # so subtracting them exposes ~1e-6 accumulation roundoff. Prove
                # the identity in FP64 and separately bound FP32 cancellation.
                weight, bias = model.score.weight.double(), model.score.bias.double()
                r64, f64 = real_features.double(), fake_features.detach().double()
                local64 = (F.linear(r64, weight, bias) - F.linear(f64, weight, bias)).reshape(draws, batch, tokens)
                gap64 = (F.linear(r64.mean(1), weight, bias) - F.linear(f64.mean(1), weight, bias)).reshape(draws, batch)
                if not torch.allclose(local64.mean(-1), gap64, atol=1e-12, rtol=1e-12):
                    raise RuntimeError("FP64 token-local and pooled gap identity failed")
                score_magnitude = torch.stack((real_local.abs().max(), fake_local.detach().abs().max(),
                    real_logits.abs().max(), fake_logits.detach().abs().max(), torch.tensor(1.))).max()
                rounding_bound = 8 * torch.finfo(torch.float32).eps * score_magnitude
                if bool(((local_gap.detach().mean(-1) - gaps.detach()).abs() > rounding_bound).any()):
                    raise RuntimeError("FP32 pooled/local gap difference exceeds score-scaled rounding bound")
            if bool((tokenwise + TOL < pooled).any()):
                raise RuntimeError("tokenwise softplus violates the expected Jensen inequality")
            gradient = torch.autograd.grad(pooled.sum(), x, retain_graph=True)[0]
            token_gradient = torch.autograd.grad(tokenwise.sum(), x)[0]
            with torch.no_grad():
                h = model.error_input(fake) + model.condition_input(repeated).unsqueeze(1)
                a = model.feature_output(h.tanh())
                first_derivative = (1 - h.tanh().square()).reshape(draws, batch, tokens, -1).mean((0, 3))
                second_derivative = (1 - a.tanh().square()).reshape(draws, batch, tokens, -1).mean((0, 3))
                first_saturation = (h.abs() > 3).float().reshape(draws, batch, tokens, -1).mean((0, 3))
                second_saturation = (a.abs() > 3).float().reshape(draws, batch, tokens, -1).mean((0, 3))
                native_radial_patch = (gradient * error).sum(-1)
                token_radial_patch = (token_gradient * error).sum(-1)
                raw_error, raw_gradient = error * model.scale, gradient / model.scale
                token_raw_gradient = token_gradient / model.scale
                pooled_drift = local_gap.mean(-1) - gaps
                values = dict(alpha=alpha, native_G_game=pooled.detach(), tokenwise_G_game=tokenwise.detach(),
                    real_minus_fake_gap=gaps.detach(), patch_real_minus_fake_gap=local_gap.detach(),
                    native_radial_derivative=native_radial_patch.sum(1),
                    tokenwise_radial_derivative=token_radial_patch.sum(1),
                    native_patch_radial_derivative=native_radial_patch,
                    tokenwise_patch_radial_derivative=token_radial_patch,
                    native_patch_raw_gradient_power=raw_gradient.square().sum(-1),
                    tokenwise_patch_raw_gradient_power=token_raw_gradient.square().sum(-1),
                    first_tanh_mean_derivative=first_derivative, second_tanh_mean_derivative=second_derivative,
                    first_tanh_saturation_fraction=first_saturation, second_tanh_saturation_fraction=second_saturation,
                    pooled_gap_identity_max_error=pooled_drift.abs().max(),
                    pooled_gap_identity_FP64_max_error=(local64.mean(-1) - gap64).abs().max(),
                    pooled_gap_identity_FP32_rounding_bound=rounding_bound)
                point = dict(alpha=alpha, native_G_game=summary(pooled), tokenwise_G_game=summary(tokenwise),
                    native_radial_derivative=summary(values["native_radial_derivative"]),
                    tokenwise_radial_derivative=summary(values["tokenwise_radial_derivative"]),
                    raw_gradient_norm=summary(raw_gradient.flatten(1).norm(dim=1)),
                    negative_native_patch_radial_fraction=float((native_radial_patch < -1e-12).float().mean()),
                    first_tanh_saturation_fraction=float(first_saturation.mean()),
                    second_tanh_saturation_fraction=float(second_saturation.mean()),
                    first_tanh_derivative_mean=float(first_derivative.mean()),
                    second_tanh_derivative_mean=float(second_derivative.mean()))
                if alpha == 0. and not torch.allclose(pooled, torch.full_like(pooled, __import__("math").log(2)), atol=1e-7, rtol=0):
                    raise RuntimeError("paired origin must equal log2 in every context")
                arrays.append(values)
                points.append(point)
        positive = [i for i, alpha in enumerate(ALPHAS) if alpha >= 0]
        violations = []
        for left, right in zip(positive, positive[1:]):
            difference = arrays[right]["native_G_game"] - arrays[left]["native_G_game"]
            token_difference = arrays[right]["tokenwise_G_game"] - arrays[left]["tokenwise_G_game"]
            violations.append(dict(from_alpha=ALPHAS[left], to_alpha=ALPHAS[right],
                native_game_reversals=(difference < -TOL).nonzero().flatten().tolist(),
                tokenwise_game_reversals=(token_difference < -TOL).nonzero().flatten().tolist(),
                native_new_minus_old=difference.tolist(), tokenwise_new_minus_old=token_difference.tolist()))
        one = arrays[ALPHAS.index(1.)]
        power = (error * model.scale).square().sum(-1)
        order = power.flatten().argsort()
        bins = {}
        for label, lo, hi in (("lower_half", 0., .5), ("middle40pct", .5, .9), ("largest10pct", .9, 1.)):
            indices = order[int(len(order) * lo):int(len(order) * hi)]
            subset = lambda name: one[name].flatten()[indices]
            bins[label] = dict(patches=len(indices), diagnostic_residual_power_fraction=float(power.flatten()[indices].sum() / power.sum()),
                native_raw_gradient_power_fraction=float(subset("native_patch_raw_gradient_power").sum() / one["native_patch_raw_gradient_power"].sum()),
                tokenwise_raw_gradient_power_fraction=float(subset("tokenwise_patch_raw_gradient_power").sum() / one["tokenwise_patch_raw_gradient_power"].sum()),
                first_tanh_derivative_mean=float(subset("first_tanh_mean_derivative").mean()),
                second_tanh_derivative_mean=float(subset("second_tanh_mean_derivative").mean()),
                first_tanh_saturation_fraction=float(subset("first_tanh_saturation_fraction").mean()),
                second_tanh_saturation_fraction=float(subset("second_tanh_saturation_fraction").mean()),
                negative_native_patch_radial_fraction=float((subset("native_patch_radial_derivative") < -1e-12).float().mean()))
        return dict(points=points, positive_ray_adjacent_comparisons=violations,
                    residual_bins_read_only_at_alpha1=bins), arrays

    with (args.output / "progress.jsonl").open("w") as log:
        for critic_name, model in critics.items():
            results[critic_name], raw[critic_name] = {}, {}
            for pool, group in groups.items():
                results[critic_name][pool], raw[critic_name][pool] = {}, {}
                for mode in ("clean", "dv12"):
                    repeated = 1 if mode == "clean" else 4
                    error, cond, panels = group[mode], group["condition"].repeat(repeated, 1), group["noise"].repeat(1, repeated, 1, 1)
                    result, values = evaluate(model, error, cond, panels)
                    results[critic_name][pool][mode], raw[critic_name][pool][mode] = result, values
                    row = dict(critic=critic_name, pool=pool, mode=mode, contexts=len(error),
                        positive_ray_reversal_pairs=sum(len(point["native_game_reversals"]) for point in result["positive_ray_adjacent_comparisons"]),
                        alpha1=result["points"][ALPHAS.index(1.)], seconds=time.perf_counter() - began)
                    emit(**row)
                    log.write(json.dumps(row, allow_nan=False) + "\n")
                    log.flush()
    if any(state_digest(model.state_dict()) != before_critic[name] for name, model in critics.items()):
        raise RuntimeError("read-only ray diagnostics mutated a critic")
    if not torch.equal(cpu_before, torch.get_rng_state()):
        raise RuntimeError("private CPU diagnostics advanced global RNG")
    if any(sha(path) != hashes[name] for name, path in inputs.items()):
        raise RuntimeError("diagnostic input file changed")
    raw_path = args.output / "raw-rays.pt"
    torch.save(dict(alphas=ALPHAS, groups=groups, rays=raw), raw_path)
    receipt = dict(plan=plan, input_paths={name: str(path.resolve()) for name, path in inputs.items()},
        input_sha256=hashes, script_sha256=sha(__file__), source_sha256=source_hashes,
        application_source_sha256={name: sha(ROOT / name) for name in
            ("supra/particle_game.py", "scripts/diagnose_e22_supra_critic_geometry.py", "supra/particle_pilot.py")},
        native_source_commit=declared["particlegan_commit"], checkpoint_native_digest=qualification["final_native_digest"],
        critic_state_digests=before_critic, panel_digest=state_digest({name: value["noise"] for name, value in groups.items()}),
        raw_sha256=sha(raw_path), results=results, critic_parameters_and_input_files_unchanged=True,
        global_CPU_RNG_unchanged=True, no_training_or_optimizer_step=True, output_metrics_used_for_optimizer_or_selection=False,
        seconds=time.perf_counter() - began)
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    emit(event="complete", output=str(args.output), seconds=receipt["seconds"])


if __name__ == "__main__":
    main()
