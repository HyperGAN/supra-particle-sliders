#!/usr/bin/env python3
"""Fixed saved-critic game-force audit; no Supra forward or model update.

Pooled and per-token payoffs share critic weights, conditions and existing
Gaussian panels. All directional comparisons use learned games. Residual-power
bins describe force allocation only and never determine the audit decision.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

STARTED = time.monotonic()
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.diagnose_e22_supra_endpoint_game_rays import module_witness
from scripts.review_e22_supra_pr223 import state_digest

CARD = ROOT / "docs/e22_supra_token_game_force_v1.json"
ARMS = ("sampled_control", "sampled_hb_neutral")
INDICES = (55, 66, 120, 131, 213, 235, 173, 179, 98, 106, 20, 39)
COMMON = ("D1856", "D6400")
SIGMA, BATCH, LIMIT, ZERO_NORM, SLOPE_TOLERANCE = .125, 4, 120, 1e-20, 1e-8


def require(condition, label):
    if not condition:
        raise ValueError(label)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def budget(started=STARTED, clock=time.monotonic):
    elapsed = clock() - started
    if elapsed > LIMIT:
        raise TimeoutError("fixed CPU1/120-second saved-force budget exhausted")
    return elapsed


def finish(output, report, started=STARTED, clock=time.monotonic):
    """Include report serialization, hashes and completion write in the budget."""
    report = {**report, "completion_receipt_required": True}
    try:
        write(output / "report.json", report)
        budget(started, clock)
        digest = sha(output / "report.json")
        budget(started, clock)
        write(output / "completion.json", dict(complete=True, qualified=True,
            report_sha256=digest, wall_seconds=clock() - started,
            clock_scope="After report write/hash; deadline also checked after completion write."))
        return budget(started, clock)
    except TimeoutError:
        report.update(qualified=False, complete=False, budget_overrun=True)
        write(output / "report.json", report)
        write(output / "completion.json", dict(complete=False, qualified=False,
            budget_overrun=True, report_sha256=sha(output / "report.json"), wall_seconds=clock() - started))
        raise


def validate_card(card):
    require(card["schema"] == "supra_token_game_force_card_v1", "fixed card schema")
    require(tuple(card["arms"]) == ARMS and tuple(card["fit_indices"]) == INDICES,
        "both fixed6400 arms and all12 score-independent fit contexts")
    require(card["checkpoint"] == 6400 and tuple(card["common_judges"]) == COMMON,
        "fixed endpoint and both common judges")
    require(card["execution_budget_seconds"] == LIMIT and card["cpu_threads"] == 1
        and card["training_updates"] == 0 and card["full_supra_forward_calls"] == 0,
        "fixed CPU-only zero-update budget")
    require(card["sigma"] == SIGMA and card["draws"] == 4 and card["context_batch"] == BATCH,
        "native paired-noise/context units")
    require(card["slope_tolerance"] == SLOPE_TOLERANCE and card["undefined_norm_at_or_below"] == ZERO_NORM,
        "predeclared game comparison and undefined norm conventions")


def panel_subset(panel_batches, indices=INDICES):
    """Select the captured draw/context panel; never regenerate or reorder draws."""
    return torch.stack([panel_batches[index // BATCH][:, index % BATCH] for index in indices], dim=1)


def force(critic, physical_patches, condition, unscaled_panels, loss, *, tokenwise=False, origin=False):
    """Per-context four-draw public RpGAN game and physical-coordinate derivative.

    Summing independent context losses obtains each unscaled context derivative;
    there is no accidental 1/B factor. The separately reported B4 mean includes it.
    """
    residual = (torch.zeros_like(physical_patches) if origin else physical_patches).detach().clone().requires_grad_(True)
    draws, contexts = unscaled_panels.shape[:2]
    condition_repeated = condition.repeat(draws, 1)
    real = SIGMA * unscaled_panels
    fake = real + (residual / critic.scale).unsqueeze(0)
    with torch.no_grad():
        if tokenwise:
            real_logits = critic.score(critic.features(real.flatten(0, 1), condition_repeated)).reshape(draws, contexts, -1, 1)
        else:
            real_logits = critic(real.flatten(0, 1), condition_repeated).reshape(draws, contexts, 1)
    if tokenwise:
        fake_logits = critic.score(critic.features(fake.flatten(0, 1), condition_repeated)).reshape(draws, contexts, -1, 1)
    else:
        fake_logits = critic(fake.flatten(0, 1), condition_repeated).reshape(draws, contexts, 1)
    games = torch.stack([loss.g_loss(fake_logits[:, index].reshape(-1, 1),
        real_logits[:, index].reshape(-1, 1)) for index in range(contexts)])
    gradient = torch.autograd.grad(games.sum(), residual)[0].detach()
    with torch.no_grad():
        # Exact algebra plus an explicit FP32 reduction bound, not a payoff change.
        local_real = critic.score(critic.features(real.flatten(0, 1), condition_repeated)).reshape(draws, contexts, -1)
        local_fake = critic.score(critic.features(fake.flatten(0, 1), condition_repeated)).reshape(draws, contexts, -1)
        pooled_fake = critic(fake.flatten(0, 1), condition_repeated).reshape(draws, contexts)
        mean_fake = local_fake.mean(-1)
        require(bool(((pooled_fake - mean_fake).abs() <= 2e-6 + 2e-6 * pooled_fake.abs()).all()),
            "linear score/feature pooling identity within fixed FP32 bound")
        gap = local_real - local_fake
    require(bool(torch.isfinite(games).all()) and bool(torch.isfinite(gradient).all()), "finite learned game/force")
    if origin:
        torch.testing.assert_close(games.detach(), torch.full_like(games, math.log(2)), atol=2e-7, rtol=0)
    return dict(games=games.detach(), gradient=gradient,
        radial=(gradient.double() * physical_patches.double()).flatten(1).sum(1),
        local_gap_standard_deviation=gap.double().std(dim=-1, unbiased=False).mean(0),
        pooling_identity_max_absolute=float((pooled_fake - mean_fake).abs().max()))


def unit_descent_slopes(judge_gradient, native_gradient, token_gradient):
    """Equal physical Euclidean norm directions; no finite parameter step taken."""
    j, n, t = [x.double().flatten(1) for x in (judge_gradient, native_gradient, token_gradient)]
    nn, tn = n.norm(dim=1), t.norm(dim=1)
    rows = []
    for index in range(len(j)):
        native = None if nn[index] <= ZERO_NORM else -float(torch.dot(j[index], n[index]) / nn[index])
        token = None if tn[index] <= ZERO_NORM else -float(torch.dot(j[index], t[index]) / tn[index])
        cosine = None if nn[index] <= ZERO_NORM or tn[index] <= ZERO_NORM else float(torch.dot(n[index], t[index]) / (nn[index] * tn[index]))
        rows.append(dict(native_direction_slope=native, token_direction_slope=token,
            token_minus_native_slope=None if native is None or token is None else token - native,
            native_token_force_cosine=cosine, native_force_norm=float(nn[index]), token_force_norm=float(tn[index])))
    return rows


def self_cauchy(native_gradient, token_gradient):
    rows = unit_descent_slopes(native_gradient, native_gradient, token_gradient)
    for row in rows:
        if row["native_direction_slope"] is not None:
            require(abs(row["native_direction_slope"] + row["native_force_norm"]) <= 1e-12 + 1e-10 * row["native_force_norm"],
                "native pooled self-gradient norm identity")
        if row["token_minus_native_slope"] is not None:
            require(row["token_minus_native_slope"] >= -1e-12 - 1e-10 * row["native_force_norm"],
                "native pooled negative-gradient direction is steepest under its own frozen game")
    return rows


def allocation(physical_patches, gradient):
    """Descriptive power bins only. No bin, raw error or norm gates a decision."""
    power = physical_patches.double().square().sum(-1)
    force_power = gradient.double().square().sum(-1)
    order = power.argsort(dim=-1, stable=True)
    groups = dict(lower_half=order[:, :power.shape[1] // 2], largest_10_percent=order[:, -math.ceil(.1 * power.shape[1]):])
    rows = []
    for row in range(len(power)):
        result = {}
        for label, index in groups.items():
            error_total, force_total = float(power[row].sum()), float(force_power[row].sum())
            result[label] = dict(descriptive_residual_power_fraction=None if error_total <= ZERO_NORM else float(power[row, index[row]].sum()) / error_total,
                game_force_power_fraction=None if force_total <= ZERO_NORM else float(force_power[row, index[row]].sum()) / force_total)
        rows.append(result)
    return rows


def mechanism_decision(observations):
    """A local stop/check decision, never quality or output-error selection."""
    own_restoring = all(row["own_native_radial_derivative"] > SLOPE_TOLERANCE for row in observations)
    cells = {}
    for arm in ARMS:
        for judge in COMMON:
            values = [row["common_games"][judge]["token_minus_native_slope"] for row in observations if row["arm"] == arm]
            cells[arm + "/" + judge] = dict(defined=all(value is not None for value in values),
                mean_token_minus_native_slope=None if any(value is None for value in values) else sum(values) / len(values),
                contexts_improved=sum(value is not None and value < -SLOPE_TOLERANCE for value in values), contexts=len(values))
    supported = all(cell["defined"] and cell["mean_token_minus_native_slope"] < -SLOPE_TOLERANCE for cell in cells.values())
    return dict(own_native_restoring_all24contexts=own_restoring, common_game_cells=cells,
        tokenwise_local_common_descent_supported=supported,
        next_action=("eligible_only_for_a_separately_fixed_tiny_acquisition_check" if own_restoring and supported
                     else "stop_this_token_granularity_mechanism_before_quality_training"),
        decision_uses_output_error=False, optimizer_or_model_updates=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-report-sha256", required=True)
    parser.add_argument("--capture-sha256", required=True)
    parser.add_argument("--qualified-review-sha256", required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/e22-supra-token-game-force-v1")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    require(not args.output.exists(), "preserve previous evidence")
    card = read(CARD)
    validate_card(card)
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU-only launch requires CUDA_VISIBLE_DEVICES=''")
    torch.set_num_threads(1)
    require(not torch.cuda.is_initialized(), "no CUDA initialization")
    global_rng = torch.get_rng_state().clone()
    capture, parent, pg = [Path(card[key]) for key in ("capture_run", "parent_run", "particlegan_root")]
    files = {str(ROOT / name): digest for name, digest in card["source_sha256"].items()}
    files[str(CARD)] = sha(CARD)
    files.update({str(capture / name): card["capture_sha256"][name] for name in ("report.json", "residuals.pt", "completion.json", "rays.json")})
    files[str(parent / "independent-review.json")] = card["parent_review_sha256"]
    require(args.capture_report_sha256 == card["capture_sha256"]["report.json"]
        and args.capture_sha256 == card["capture_sha256"]["residuals.pt"]
        and args.qualified_review_sha256 == card["parent_review_sha256"], "root-frozen actual capture and full-review identities")
    for path, expected in files.items():
        budget()
        require(sha(path) == expected, "exact fixed input/source SHA: " + path)
    report, completion, review = read(capture / "report.json"), read(capture / "completion.json"), read(parent / "independent-review.json")
    require(report["qualified"] is True and report["training_updates"] == 0 and all(report["checks"].values()), "qualified zero-update endpoint capture")
    require(completion["complete"] is True and completion["qualified"] is True
        and completion["report_sha256"] == args.capture_report_sha256 and report["raw_residual_sha256"] == args.capture_sha256,
        "qualified post-write capture completion")
    require(report["per_context_rays_sha256"] == card["capture_sha256"]["rays.json"], "qualified per-context ray results")
    captured_rays = read(capture / "rays.json")
    require(review["qualified"] is True and review["partial"] is False
        and review["native_source_digest"] == card["native_source_digest"], "exact completed native parent qualification")
    require(subprocess.check_output(["git", "-C", str(pg), "rev-parse", "HEAD"], text=True).strip() == card["particlegan_commit"], "fixed native revision")
    parent_plan = read(parent / "plan.json")
    require(sha(parent / "plan.json") == review["plan_sha256"], "exact qualified parent plan")
    files[str(parent / "plan.json")] = review["plan_sha256"]
    require(not subprocess.check_output(["git", "-C", str(pg), "status", "--porcelain", "--", "particlegan"], text=True).strip(), "native package must be clean")
    for name, expected in parent_plan["particlegan_source_sha256"].items():
        path = pg / name
        require(sha(path) == expected, "qualified native source: " + name)
        files[str(path)] = expected
    checkpoint_paths = {name: Path(parent_plan["input_paths"][name]) for name in COMMON}
    checkpoint_paths.update({arm: parent / arm / "checkpoint-06400.pt" for arm in ARMS})
    declared = {name: parent_plan["input_sha256"][name] for name in COMMON}
    declared.update({arm: review["checkpoint_artifacts"][arm]["6400"]["file_sha256"] for arm in ARMS})
    for name, path in checkpoint_paths.items():
        budget()
        require(sha(path) == declared[name], "actual qualified critic checkpoint: " + name)
        files[str(path)] = declared[name]
    raw = torch.load(capture / "residuals.pt", map_location="cpu", weights_only=False)
    require(raw["plan"] == report["plan"], "capture tensor/plan binding")
    metadata = raw["context_metadata"]["fit"][list(INDICES)].clone()
    require(tuple(metadata[:, 1].long().tolist()) == tuple(card["context_subjects"])
        and metadata[:, 0].tolist() == card["context_times"], "fixed subject/time-balanced fit metadata")
    panels = panel_subset(raw["unscaled_panel_batches"]["fit"]).clone()
    require(tuple(panels.shape) == (4, 12, 256, 16), "exact retained four-draw panels")
    before_raw, before_panels = state_digest(raw), state_digest(panels)
    if args.preflight_only:
        require(torch.equal(torch.get_rng_state(), global_rng) and not torch.cuda.is_initialized(), "RNG/CUDA unchanged in preflight")
        print(json.dumps(dict(preflight_passed=True, critic_checkpoint_loads=0, full_supra_forward_calls=0, training_updates=0, seconds=budget())))
        return
    sys.path.insert(0, str(pg))
    from particlegan.gan_loss import GANLoss
    from supra.particle_game import ConditionalTokenCritic, patchify
    require(Path(sys.modules["particlegan"].__file__).resolve().parent.parent == pg.resolve(), "fixed native public loss import")
    args.output.mkdir(parents=True)
    checks, observations, identities = {}, [], {}
    plan = dict(card=card, card_sha256=sha(CARD), input_sha256=files,
        selected_unscaled_panels_digest=before_panels, context_metadata_digest=state_digest(metadata),
        torch_version=torch.__version__, device="cpu", cpu_threads=1)
    write(args.output / "plan.json", plan)
    try:
        critics, originals, states = {}, {}, {}
        for name, path in checkpoint_paths.items():
            budget()
            # mmap metadata + critic tensors only; no G/base/teacher tensor traversal.
            saved = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
            require(saved["policy"]["completed_steps"] == (6400 if name in ARMS else int(name[1:])), "actual trained critic step")
            states[name] = saved["policy"]["models"]["critic"]
            if name in ARMS:
                require(saved["config"]["particle_init"] == ("sampled_v1" if name == ARMS[0] else "sampled_hb_neutral_v1")
                    and saved["config"]["architecture"] == "gated_particle_v3", "actual fresh trained ownD cohort")
            originals[name] = state_digest(states[name])
            if name in COMMON:
                require(originals[name] == review["judges"][name], "exact qualified common judge tensors")
            with torch.random.fork_rng(devices=[]):
                critic = ConditionalTokenCritic(states[name]["scale"]).float().eval().requires_grad_(False)
            critic.load_state_dict(states[name], strict=True)
            require(state_digest(critic.state_dict()) == originals[name], "exact copied critic weights")
            critics[name] = critic
            identities[name] = dict(checkpoint_sha256=declared[name], critic_tensor_digest=originals[name],
                module_witness=module_witness(critic))
            del saved
        loss = GANLoss()
        for arm in ARMS:
            selected = raw["captures"][arm + "@6400"]["fit"]
            patches = patchify(selected["physical_residual"][list(INDICES)]).float()
            condition = selected["condition"][list(INDICES)].float()
            for start in range(0, len(INDICES), BATCH):
                budget()
                residual, cond, panel = patches[start:start+BATCH], condition[start:start+BATCH], panels[:, start:start+BATCH]
                native = force(critics[arm], residual, cond, panel, loss)
                token = force(critics[arm], residual, cond, panel, loss, tokenwise=True)
                origin = force(critics[arm], residual, cond, panel, loss, origin=True)
                self_control = self_cauchy(native["gradient"], token["gradient"])
                common = {name: force(critics[name], residual, cond, panel, loss) for name in COMMON}
                for name, value in common.items():
                    expected = captured_rays[name][arm + "@6400"]["fit"]
                    for position, fit_index in enumerate(INDICES[start:start+BATCH]):
                        for field, observed in (("games", value["games"]), ("radial_derivatives", value["radial"])):
                            require(math.isclose(float(observed[position]), expected[field]["1"][fit_index],
                                abs_tol=2e-6, rel_tol=1e-5), "original captured alpha1 common native game/radial identity")
                directions = {name: unit_descent_slopes(value["gradient"], native["gradient"], token["gradient"])
                              for name, value in common.items()}
                allocations = [allocation(residual, value["gradient"]) for value in (native, token)]
                for index in range(len(residual)):
                    observations.append(dict(arm=arm, fit_index=INDICES[start+index],
                        source_caption_id=int(metadata[start+index, 1]), time=float(metadata[start+index, 0]),
                        critic_batch_fit_indices=list(INDICES[start:start+BATCH]),
                        own_native_game=float(native["games"][index]), own_token_game=float(token["games"][index]),
                        own_native_radial_derivative=float(native["radial"][index]),
                        own_token_radial_derivative=float(token["radial"][index]),
                        own_native_origin_radial_derivative=float(origin["radial"][index]),
                        local_gap_standard_deviation=float(native["local_gap_standard_deviation"][index]),
                        native_self_cauchy_control=self_control[index],
                        common_games={name: {**directions[name][index], "native_common_game": float(common[name]["games"][index]),
                            "alpha1_game_replay_difference": float(common[name]["games"][index]) - captured_rays[name][arm + "@6400"]["fit"]["games"]["1"][INDICES[start+index]],
                            "alpha1_radial_replay_difference": float(common[name]["radial"][index]) - captured_rays[name][arm + "@6400"]["fit"]["radial_derivatives"]["1"][INDICES[start+index]]} for name in COMMON},
                        descriptive_power_allocation=dict(native=allocations[0][index], tokenwise=allocations[1][index]),
                        pooling_identity_max_absolute=native["pooling_identity_max_absolute"]))
                print(json.dumps(dict(phase="observe", arm=arm, fit_indices=list(INDICES[start:start+BATCH]), seconds=budget())), flush=True)
        for name, critic in critics.items():
            checks["copied_critic_immutable:" + name] = module_witness(critic) == identities[name]["module_witness"]
            checks["checkpoint_critic_tensors_immutable:" + name] = state_digest(states[name]) == originals[name]
        checks["retained_tensors_immutable"] = state_digest(raw) == before_raw and state_digest(panels) == before_panels
        checks["global_CPU_RNG_immutable"] = torch.equal(torch.get_rng_state(), global_rng)
        checks["CUDA_not_initialized"] = not torch.cuda.is_initialized()
        checks["all24_contexts_both_critics_retained"] = len(observations) == 24
        for path, expected in files.items():
            budget()
            checks["input_source_immutable:" + path] = sha(path) == expected
        require(all(checks.values()), "state/input/RNG/source immutability")
        finish(args.output, dict(schema="supra_token_game_force_report_v1", complete=True, qualified=True,
            plan=plan, critic_identities=identities, checks=checks, observations=observations,
            decision=mechanism_decision(observations), training_updates=0, optimizer_updates=0,
            full_supra_forward_calls=0, critic_force_evaluations=30, full_checkpoint_tensor_traversals=0,
            output_metrics_used_for_optimizer_or_selection=False,
            per_context_gradient_units="four-draw mean; no1/B4 scaling; physical [256,16] coordinates",
            B4_mean_gradient_units="Each recorded per-context gradient divided by4, combined in disjoint residual rows; no cross-context cancellation is inferred.",
            limits=card["limits"]))
    except Exception as error:
        write(args.output / "failure.json", dict(qualified=False, complete=False, error=type(error).__name__ + ": " + str(error),
            seconds=time.monotonic() - STARTED, completed_observations=observations, checks=checks, training_updates=0, full_supra_forward_calls=0))
        raise


if __name__ == "__main__":
    main()
