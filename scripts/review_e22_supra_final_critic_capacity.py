#!/usr/bin/env python3
"""Independent CPU replay and KA2-gradient review of the isolated capacity trial."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=Path, default=ROOT / "outputs/e22-particle-final-critic-causal/capacity")
    parser.add_argument("--run", type=Path, default=ROOT / "outputs/e22-particle-v2-6400")
    parser.add_argument("--payload", type=Path, default=ROOT / "outputs/e22-particle-final-critic-causal/residuals.pt")
    parser.add_argument("--particlegan-root", type=Path,
                        default=ROOT / "outputs/e22-convergence-gap/particlegan-cabe2084-source")
    args = parser.parse_args()
    sys.path.insert(0, str(args.particlegan_root.resolve()))
    sys.path.insert(1, str(ROOT))
    import torch
    import torch.nn.functional as F
    import particlegan
    from particlegan import Recipe
    from particlegan.continuous import DataDriftController
    from supra.particle_game import ConditionalTokenCritic, apply_critic_penalty, patchify
    from supra.particle_pilot import state_digest
    from experimental_e22_bounded_critic_features import BoundedFeatureCritic, FORMULATION
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    receipt = json.loads((args.trial / "receipt.json").read_text())
    raw = torch.load(args.run / "final.pt", map_location="cpu", weights_only=False, mmap=True)
    native = raw["policy"]
    data = torch.load(args.run / "data.pt", map_location="cpu", weights_only=False)
    payload = torch.load(args.payload, map_location="cpu", weights_only=False)
    source = Path(particlegan.__file__).resolve().parent
    source_digest = state_digest({p.name: sha(p) for p in sorted(source.glob("*.py"))})
    checks = {}

    def check(name, condition):
        checks[name] = bool(condition)
        if not condition:
            raise RuntimeError("independent critic review failed: " + name)

    check("source_path_exact", source.parent == args.particlegan_root.resolve())
    check("source_checkpoint_payload_and_dataset_digests", source_digest == receipt["particlegan_source_digest"]
          and sha(args.run / "final.pt") == receipt["checkpoint_sha256"]
          and sha(args.payload) == receipt["payload_sha256"]
          and state_digest(data) == receipt["dataset_digest"])
    check("runner_and_helper_digests", sha(ROOT / "scripts/diagnose_e22_supra_final_critic_capacity.py") == receipt["script_sha256"]
          and sha(ROOT / "scripts/experimental_e22_bounded_critic_features.py") == receipt["helper_sha256"])
    check("step_architecture_and_budget", native["completed_steps"] == 6400
          and raw["config"]["architecture"] == "linear_modulated_v2"
          and receipt["plan"]["updates"] == 128)
    recipe = Recipe(**native["recipe"])
    groups = {}
    for name in ("fit", "holds", "test"):
        group = payload["groups"][name]
        groups[name] = dict(cond=group["condition"].float(), clean=patchify(group["new"]) / data["coordinate_scale"],
                           noisy=torch.stack([patchify(r) / data["coordinate_scale"] for r in group["noisy"]]))
    panels_stream = torch.Generator().manual_seed(72)
    panels = [float(native["last_output_sigma"]) * torch.randn((4, 256, 16), generator=panels_stream)
              for _ in range(128)]
    check("same_private_training_panels", state_digest(panels) == receipt["training_panel_digest"])

    def build(arm, final=None):
        with torch.random.fork_rng(devices=[]):
            cls = ConditionalTokenCritic if arm == "native_critic_v1" else BoundedFeatureCritic
            critic = cls(data["coordinate_scale"]).float()
            values, opt_values = deepcopy(native["models"]["critic"]), deepcopy(native["optimizers"][1])
            if arm == FORMULATION:
                # Independent explicit extension, rather than calling the
                # runner's migration/checkpoint implementation.
                names = list(dict(critic.named_parameters()))
                check("gain_is_appended_after_old_parameters", names[-1] == "bypass.0")
                values["bypass.0"] = torch.zeros(16)
                opt_values["param_groups"][0]["params"].append(len(names) - 1)
                opt_values["regularizer"]["ema"]["bypass.0"] = torch.zeros(16)
            if final is not None:
                check("isolated_checkpoint_tag_" + arm,
                    final["format"] == "isolated_e22_critic_diagnostic_v1" and final["formulation"] == arm)
                values, opt_values = final["critic"], final["optimizer"]
            critic.load_state_dict(values, strict=True)
            optimizer = recipe.make_critic_optimizer(critic, ema_critic=deepcopy(critic), foreach=False)
            optimizer.load_state_dict(deepcopy(opt_values))
            controller = DataDriftController("dv12")
            controller.load_state_dict(deepcopy(native["controller"]))
            optimizer.continuous_controller = controller
            penalty = recipe.make_critic_penalty(optimizer, collect_stats=True)
        return critic, optimizer, controller, penalty

    def manual_feature(critic, error, cond):
        first = (F.linear(error, critic.error_input.weight, critic.error_input.bias)
                 + F.linear(cond, critic.condition_input.weight).unsqueeze(1)).tanh()
        a = F.linear(first, critic.feature_output.weight, critic.feature_output.bias)
        feature = a.tanh()
        if isinstance(critic, BoundedFeatureCritic):
            feature = feature + critic.bypass[0].tanh() * (a / 4.).tanh()
        return feature

    final_checks = {}
    for arm in ("native_critic_v1", FORMULATION):
        model, optimizer, controller, penalty = build(arm)
        initial = dict(format="isolated_e22_critic_diagnostic_v1", formulation=arm,
                       critic=deepcopy(model.state_dict()), optimizer=deepcopy(optimizer.state_dict()))
        check("initial_state_exact_" + arm, state_digest(initial) == receipt["results"][arm]["initial_state_digest"])
        rows = [json.loads(line) for line in (args.trial / (arm + ".jsonl")).read_text().splitlines()]
        check("trace_budget_" + arm, len(rows) == 128)
        controller_before = state_digest(controller.state_dict())
        for index, row in enumerate(rows):
            name = "holds" if (6401 + index) % 5 == 0 else "fit"
            weight = .1 if name == "holds" else 1.
            group = groups[name]
            real, fake = panels[index], panels[index] + group["noisy"][index % 4]
            check("no_test_contexts_in_updates", name in ("fit", "holds"))
            # Preserve native real-first graph construction and subtraction
            # order: algebraically equivalent reversed expressions can change
            # shared-parameter float32 gradient accumulation after several
            # Adam steps. This remains a separate update implementation.
            real_scores = model(real, group["cond"])
            fake_scores = model(fake, group["cond"])
            game = weight * F.softplus(-(real_scores - fake_scores)).mean()
            reg = apply_critic_penalty(penalty, model, real, fake, group["cond"])
            if float(game.detach()) != row["d_game"] or float(reg.detach()) != row["penalty"]:
                raise RuntimeError(f"independent trace mismatch {arm}/{index}: game {float(game.detach())!r}/{row['d_game']!r}; penalty {float(reg.detach())!r}/{row['penalty']!r}")
            check("game_and_penalty_trace_exact_" + arm, True)
            optimizer.zero_grad(set_to_none=True)
            (game + reg).backward()
            optimizer.step()
            check("native_ka2_clocks_trace_exact_" + arm,
                optimizer.record.calls == row["penalty_calls"] and optimizer.record.alpha == row["controller_alpha"]
                and optimizer.record.w == row["controller_weight"] and optimizer.record.last_ratio == row["controller_surprise_ratio"])
        actual = dict(format="isolated_e22_critic_diagnostic_v1", formulation=arm,
                      critic=deepcopy(model.state_dict()), optimizer=deepcopy(optimizer.state_dict()))
        final = torch.load(args.trial / (arm + ".pt"), map_location="cpu", weights_only=False)
        check("complete_128_update_full_state_replay_" + arm,
              state_digest(actual) == state_digest(final) == receipt["results"][arm]["final_state_digest"])
        check("application_controller_frozen_" + arm, state_digest(controller.state_dict()) == controller_before)
        for name, group in groups.items():
            with torch.no_grad():
                independent = manual_feature(model, group["clean"], group["cond"])
                check("feature_law_exact_" + arm, torch.equal(independent, model.features(group["clean"], group["cond"])))
                if arm == FORMULATION:
                    check("bounded_features_" + arm, bool(independent.abs().max() <= 2))
        final_checks[arm] = dict(full_state_digest=state_digest(actual), updates=128)

    # Native paired KA2's value and full parameter gradient are decomposed on
    # a fresh initial extended model, including its new gain. Game/penalty
    # gain directions explain what the proposed extra capacity actually learns.
    gain_competition = {}
    for name, group in groups.items():
        model, optimizer, _, penalty = build(FORMULATION)
        real, fake = panels[0], panels[0] + group["noisy"][0]
        cond, weight = group["cond"], (.1 if name == "holds" else 1.)
        game = weight * F.softplus(model(fake, cond) - model(real, cond)).mean()
        native_penalty = apply_critic_penalty(penalty, model, real, fake, cond)
        real_variable, fake_variable = real.detach().requires_grad_(True), fake.detach().requires_grad_(True)
        gr = torch.autograd.grad(model.score(model.features(real_variable, cond)).sum(), real_variable, create_graph=True)[0]
        gf = torch.autograd.grad(model.score(model.features(fake_variable, cond)).sum(), fake_variable, create_graph=True)[0]
        anchor_variable = real.detach().requires_grad_(True)
        ga = torch.autograd.grad(optimizer.ema_critic.score(optimizer.ema_critic.features(anchor_variable, cond)).sum(), anchor_variable)[0].detach()
        sr, sf = gr.square().sum(-1), gf.square().sum(-1)
        nr, nf = (sr + 1e-12).sqrt(), (sf + 1e-12).sqrt()
        coefficient = recipe.reg_coeff / 4.
        pieces = dict(real_r1=coefficient * (sr / 16.).mean(),
                      fake_rms_cap=coefficient * F.relu(nf / 4. - recipe.reg_kappa).square().mean(),
                      real_l2_cap=coefficient * F.relu(nr - recipe.reg_kappa).square().mean(),
                      fake_l2_cap=coefficient * F.relu(nf - recipe.reg_kappa).square().mean(),
                      anchor=coefficient * optimizer.record.w * recipe.reg_anchor_weight * (gr - ga).square().sum(-1).mean() / 16.)
        parameters = tuple(model.parameters())

        def flat_gradient(value):
            gradients = torch.autograd.grad(value, parameters, retain_graph=True, allow_unused=True)
            return torch.cat([(torch.zeros_like(p) if g is None else g).flatten()
                              for p, g in zip(parameters, gradients)])

        manual = sum(pieces.values())
        check("full_extended_ka2_value_" + name, torch.allclose(manual, native_penalty, atol=1e-6, rtol=1e-5))
        check("full_extended_ka2_parameter_gradient_" + name,
              torch.allclose(flat_gradient(manual), flat_gradient(native_penalty), atol=1e-6, rtol=1e-4))
        dg = torch.autograd.grad(game, model.bypass[0], retain_graph=True)[0]
        dp = torch.autograd.grad(native_penalty, model.bypass[0], retain_graph=True)[0]
        piece_gradients = {key: torch.autograd.grad(value, model.bypass[0], retain_graph=True, allow_unused=True)[0]
                          for key, value in pieces.items()}
        gain_competition[name] = dict(game_weight=weight,
            d_game_gain_gradient=dg.tolist(), ka2_gain_gradient=dp.tolist(), total_gain_gradient=(dg + dp).tolist(),
            d_game_gradient_norm=float(dg.norm()), ka2_gradient_norm=float(dp.norm()),
            game_penalty_gradient_cosine=float((dg * dp).sum() / (dg.norm() * dp.norm()).clamp_min(1e-30)),
            total_gradient_pushes_gain_negative=int(((dg + dp) > 0).sum()),
            piece_gain_gradients={key: (None if value is None else value.tolist()) for key, value in piece_gradients.items()},
            native_ka2_value_and_all_parameter_gradients_verified=True)
    report = dict(qualified=all(checks.values()), checks=checks, results=final_checks,
        initial_gain_competition=gain_competition, receipt_sha256=sha(args.trial / "receipt.json"),
        reviewer_sha256=sha(__file__),
        limits="Independent CPU replay qualifies only the predeclared frozen-residual D-only trial. No full-model convergence or native CUDA panel equivalence is claimed.")
    (args.trial / "independent-review.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(qualified=report["qualified"], checks=len(checks), output=str(args.trial / "independent-review.json"))), flush=True)


if __name__ == "__main__":
    main()
