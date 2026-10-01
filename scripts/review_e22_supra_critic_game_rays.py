#!/usr/bin/env python3
"""Independent CPU artifact and analytic-chain review of frozen game rays."""
from argparse import ArgumentParser
from copy import deepcopy
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
ALPHAS = (-1., -.5, 0., .25, .5, .75, 1., 1.5, 2., 4.)
TOL = 1e-6


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "outputs/e22-particle-final-critic-causal/game-rays")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.run / "independent-review.json"
    if output.exists():
        parser.error("preserve previous review; choose another output")
    receipt = json.loads((args.run / "receipt.json").read_text())
    pg_root = Path(receipt["input_paths"]["checkpoint6400"]).parents[1] / "e22-convergence-gap/particlegan-cabe2084-source"
    sys.path.insert(0, str(pg_root))
    sys.path.insert(1, str(ROOT))
    import torch
    import torch.nn.functional as F
    import particlegan
    from supra.particle_game import ConditionalTokenCritic, patchify
    from supra.particle_pilot import state_digest
    from scripts.review_e22_supra_pr223 import require, read, sha, finite
    from scripts.diagnose_e22_supra_critic_geometry import summary

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    before_rng = torch.get_rng_state().clone()
    plan = receipt["plan"]
    require(read(args.run / "plan.json") == plan and plan["schema"] == "supra_frozen_critic_game_rays_v1", "exact ray plan")
    require(plan["alphas"] == list(ALPHAS) and plan["tolerance"] == TOL and plan["residual_boundary"] == 6400
            and plan["gaussian_panels"] == 4 and plan["contexts_per_pool"] == 4 and plan["dv12_draws"] == 4, "fixed ray/panel/capture controls")
    require(plan["training_or_optimizer_updates"] == 0 and plan["output_metrics_used_for_optimizer_or_selection"] is False,
            "read-only game diagnostic")
    for flag in ("critic_parameters_and_input_files_unchanged", "global_CPU_RNG_unchanged", "no_training_or_optimizer_step"):
        require(receipt[flag] is True, f"runtime witness {flag}")
    for name, path in receipt["input_paths"].items():
        require(sha(path) == receipt["input_sha256"][name], f"input hash {name}")
    require(sha(ROOT / "scripts/diagnose_e22_supra_critic_game_rays.py") == receipt["script_sha256"], "executed diagnostic source")
    for name, digest in receipt["application_source_sha256"].items():
        require(sha(ROOT / name) == digest, f"application source {name}")
    pg_dir = Path(particlegan.__file__).resolve().parent
    require({path.name: sha(path) for path in sorted(pg_dir.glob("*.py"))} == receipt["source_sha256"], "native archived source")
    raw_path = args.run / "raw-rays.pt"
    require(sha(raw_path) == receipt["raw_sha256"], "raw ray file hash")
    raw = torch.load(raw_path, map_location="cpu", weights_only=False)
    require(raw["alphas"] == ALPHAS, "raw declared alpha controls")
    paths = receipt["input_paths"]
    saved = torch.load(paths["checkpoint6400"], map_location="cpu", weights_only=False, mmap=True)
    data = torch.load(paths["data"], map_location="cpu", weights_only=False)
    payload = torch.load(paths["payload"], map_location="cpu", weights_only=False)
    qualification = read(paths["qualification"])
    require(qualification["qualified"] is True and state_digest(saved) == receipt["checkpoint_native_digest"]
            == qualification["final_native_digest"], "complete qualified6400 native owner content")
    require(saved["policy"]["completed_steps"] == 6400 and state_digest(data) == saved["config"]["dataset_digest"], "native boundary/data")
    declared = read(paths["run"])
    require(state_digest(receipt["source_sha256"]) == declared["particlegan_source_digest"]
            and receipt["native_source_commit"] == declared["particlegan_commit"], "source-qualified native game")
    sigma = float(saved["policy"]["last_output_sigma"])
    panels = torch.Generator().manual_seed(72)
    for pool in ("fit", "holds", "test"):
        group, capture = raw["groups"][pool], payload["groups"][pool]
        require(torch.equal(group["context"], capture["context"]) and torch.equal(group["condition"], capture["condition"].float()), "native conditioning capture")
        require(torch.equal(group["clean"], patchify(capture["new"].float()) / data["coordinate_scale"])
                and torch.equal(group["dv12"], patchify(capture["noisy"].flatten(0, 1).float()) / data["coordinate_scale"]), "normalized clean/DV12 residual units")
        expected = sigma * torch.randn((4, 4, 256, 16), generator=panels)
        require(torch.equal(group["noise"], expected), "independent privateCPU72 panel reconstruction")
    require(state_digest({name: group["noise"] for name, group in raw["groups"].items()}) == receipt["panel_digest"], "panel content witness")
    models = {}
    with torch.random.fork_rng(devices=[]):
        critic = ConditionalTokenCritic(data["coordinate_scale"]).float().eval().requires_grad_(False)
        critic.load_state_dict(saved["policy"]["models"]["critic"], strict=True)
        models["D6400"] = critic
        if "checkpoint1856" in paths:
            earlier = torch.load(paths["checkpoint1856"], map_location="cpu", weights_only=False, mmap=True)
            prior = deepcopy(critic)
            prior.load_state_dict(earlier["policy"]["models"]["critic"], strict=True)
            require(earlier["policy"]["completed_steps"] == 1856 and earlier["config"]["dataset_digest"] == saved["config"]["dataset_digest"], "matched earlier native critic")
            models["D1856_on_final6400_residuals"] = prior
    require(list(models) == plan["critics"] == list(raw["rays"]), "critic identities")
    checks = dict(cases=0, ray_points=0, context_ray_points=0, positive_ray_adjacent_context_comparisons=0,
        positive_ray_reversals=0, positive_radial_sign_failures=0, origin_outward_derivatives=0,
        FP32_pooling_roundoff_max=0., FP64_pooling_identity_error_max=0., analytic_radial_max_error=0.,
        analytic_raw_gradient_power_max_error=0., FP64_finite_difference_radial_max_error=0.)
    per_critic = {}
    with torch.no_grad():
        for name, model in models.items():
            require(state_digest(model.state_dict()) == receipt["critic_state_digests"][name], "scoring critic content")
            per_critic[name] = dict(positive_reversals=0, positive_radial_sign_failures=0, origin_outward_derivatives=0)
            double = deepcopy(model).double()
            for pool in ("fit", "holds", "test"):
                group = raw["groups"][pool]
                for mode in ("clean", "dv12"):
                    count = 1 if mode == "clean" else 4
                    e = group[mode]
                    cond = group["condition"].repeat(count, 1)
                    noise = group["noise"].repeat(1, count, 1, 1)
                    draws, batch, tokens = len(noise), len(e), e.shape[1]
                    repeated = cond.repeat(draws, 1)
                    real = noise.flatten(0, 1)
                    real_logits = model(real, repeated).reshape(draws, batch)
                    values = raw["rays"][name][pool][mode]
                    report = receipt["results"][name][pool][mode]
                    require([v["alpha"] for v in values] == list(ALPHAS), "complete predeclared rays")
                    checks["cases"] += 1
                    for point, metric in zip(values, report["points"]):
                        alpha = point["alpha"]
                        finite(metric, "ray metric")
                        for value in point.values():
                            if isinstance(value, torch.Tensor):
                                require(bool(torch.isfinite(value).all()), "finite raw ray tensors")
                        fake = (noise + alpha * e.unsqueeze(0)).flatten(0, 1)
                        logits = model(fake, repeated).reshape(draws, batch)
                        gap = real_logits - logits
                        game = F.softplus(gap).mean(0)
                        require(torch.equal(gap, point["real_minus_fake_gap"]) and torch.equal(game, point["native_G_game"]), "native score/game independent rerun")
                        local = point["patch_real_minus_fake_gap"]
                        token_game = F.softplus(local).mean((0, 2))
                        require(torch.equal(token_game, point["tokenwise_G_game"]), "tokenwise observational game arithmetic")
                        for field, source in (("native_G_game", game), ("tokenwise_G_game", token_game),
                                             ("native_radial_derivative", point["native_patch_radial_derivative"].sum(1)),
                                             ("tokenwise_radial_derivative", point["tokenwise_patch_radial_derivative"].sum(1))):
                            require(summary(source) == metric[field], f"context summary/patch derivative sum {field}")
                        require(bool((token_game + TOL >= game).all()), "paired Jensen comparison")
                        delta = (local.mean(-1) - gap).abs().max()
                        require(float(delta) == float(point["pooled_gap_identity_max_error"])
                                and bool(delta <= point["pooled_gap_identity_FP32_rounding_bound"]), "FP32 pooling cancellation bound")
                        checks["FP32_pooling_roundoff_max"] = max(checks["FP32_pooling_roundoff_max"], float(delta))
                        checks["FP64_pooling_identity_error_max"] = max(checks["FP64_pooling_identity_error_max"], float(point["pooled_gap_identity_FP64_max_error"]))
                        require(float(point["pooled_gap_identity_FP64_max_error"]) <= 1e-12, "FP64 linear score pooling identity")
                        # Analytic local score derivative: wS * sech²(a) * WF * sech²(h) * WE.
                        # Native G then contributes -sigmoid(pooled gap)/(panels*tokens).
                        h = F.linear(fake, model.error_input.weight, model.error_input.bias)
                        h += F.linear(repeated, model.condition_input.weight).unsqueeze(1)
                        a = F.linear(h.tanh(), model.feature_output.weight, model.feature_output.bias)
                        chain = ((1 - a.tanh().square()) * model.score.weight) @ model.feature_output.weight
                        chain = (chain * (1 - h.tanh().square())) @ model.error_input.weight
                        chain = chain.reshape(draws, batch, tokens, 16)
                        analytic = -(gap.sigmoid().unsqueeze(-1).unsqueeze(-1) * chain).mean(0) / tokens
                        token_analytic = -(local.sigmoid().unsqueeze(-1) * chain).mean(0) / tokens
                        radial = (analytic * e).sum((1, 2))
                        token_radial = (token_analytic * e).sum((1, 2))
                        require(torch.allclose(radial, point["native_radial_derivative"], atol=2e-6, rtol=2e-5)
                                and torch.allclose(token_radial, point["tokenwise_radial_derivative"], atol=2e-6, rtol=2e-5), "independent analytic paired-game radial chain")
                        power = (analytic / model.scale).square().sum(-1)
                        require(torch.allclose(power, point["native_patch_raw_gradient_power"], atol=1e-10, rtol=3e-5), "raw-coordinate chain/scale units")
                        checks["analytic_radial_max_error"] = max(checks["analytic_radial_max_error"], float((radial - point["native_radial_derivative"]).abs().max()))
                        checks["analytic_raw_gradient_power_max_error"] = max(checks["analytic_raw_gradient_power_max_error"], float((power - point["native_patch_raw_gradient_power"]).abs().max()))
                        if alpha in (0., 1.):
                            nr, ed, cd = noise.double(), e.double(), repeated.double()
                            real64 = double(nr.flatten(0, 1), cd)
                            def value(offset):
                                pred = double((nr + offset * ed.unsqueeze(0)).flatten(0, 1), cd)
                                return F.softplus(real64 - pred).reshape(draws, batch).mean(0)
                            eps = 1e-4
                            fd = (value(alpha + eps) - value(alpha - eps)) / (2 * eps)
                            difference = (fd - radial.double()).abs().max()
                            require(torch.allclose(fd, radial.double(), atol=2e-6, rtol=3e-5), "independent FP64 radial finite difference")
                            checks["FP64_finite_difference_radial_max_error"] = max(checks["FP64_finite_difference_radial_max_error"], float(difference))
                        if alpha > 0.:
                            failures = int((radial < -TOL).sum())
                            checks["positive_radial_sign_failures"] += failures
                            per_critic[name]["positive_radial_sign_failures"] += failures
                        if alpha == 0.:
                            require(torch.allclose(game, torch.full_like(game, math.log(2)), atol=1e-7, rtol=0), "paired origin identity")
                            outward = int((radial < -TOL).sum())
                            checks["origin_outward_derivatives"] += outward
                            per_critic[name]["origin_outward_derivatives"] += outward
                        checks["ray_points"] += 1
                        checks["context_ray_points"] += batch
                    positive = [v for v in values if v["alpha"] >= 0]
                    for left, right, comparison in zip(positive, positive[1:], report["positive_ray_adjacent_comparisons"]):
                        difference = right["native_G_game"] - left["native_G_game"]
                        reverse = (difference < -TOL).nonzero().flatten().tolist()
                        require(comparison["native_game_reversals"] == reverse
                                and comparison["native_new_minus_old"] == difference.tolist(), "positive-ray ranking arithmetic")
                        checks["positive_ray_reversals"] += len(reverse)
                        per_critic[name]["positive_reversals"] += len(reverse)
                        checks["positive_ray_adjacent_context_comparisons"] += batch
    require(torch.equal(before_rng, torch.get_rng_state()), "review private/no-global-RNG advance")
    report = dict(schema="supra_frozen_critic_game_rays_independent_cpu_review_v1", qualified=True,
        gpu_used=False, run=str(args.run.resolve()), checks=checks, per_critic=per_critic,
        source_files_verified=len(receipt["source_sha256"]) + len(receipt["application_source_sha256"]) + 1,
        input_files_verified=len(paths), native_source_commit=receipt["native_source_commit"],
        no_training_or_optimizer_step=True, output_metrics_used_for_optimizer_or_selection=False,
        interpretation="No sampled positive-ray rank reversal; nonzero origin drift is reported, never discarded. This is a shape/units diagnostic, not an intervention win.",
        limits=["Native model/capture immutability is a qualified runtime witness; this review checks frozen critic/data/record arithmetic on CPU.",
                "Twelve selected contexts plus correlated DV12 variants. First-order rays do not test the whole model's parameter update or minimax convergence.",
                "Tokenwise softplus and raw residual bins are observational only; no training objective or guard has changed."],
        reviewer_sha256=sha(__file__), receipt_sha256=sha(args.run / "receipt.json"), raw_sha256=sha(raw_path))
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
