#!/usr/bin/env python3
"""Predeclared Supra recovery comparison, with output metrics reporting only."""
import argparse
from copy import deepcopy
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from supra.runtime import SupraRuntime
from supra.particle_game import ConditionalTokenCritic, patchify, update
from supra.particle_pilot import checkpoint, frozen_digest, initial_digest, make_loop, restore, state_digest, switch_teacher
from scripts.verify_e22_supra import optimizer_provenance


def emit(value):
    print(json.dumps(value, sort_keys=True, default=str), flush=True)


def reopen_count(policy):
    return sum(tester.counts.get("reopens", 0) for row in policy.lr_settle.testers
               for tester in row if tester is not None)


def game_controls(policy):
    stats = deepcopy(policy.penalty.last_stats)
    record = policy.opt_d.record
    return dict(optimizer_surprise=None if policy.surprise is None else policy.surprise.diagnostics(),
                group_surprise={} if policy.surprise is None else dict(policy.surprise.last_ratios),
                ladder_reopens=reopen_count(policy), penalty_calls=record.calls,
                ka2_ratio=record.last_ratio, penalty_stats=stats,
                lrs=[[group["lr"] for group in optimizer.param_groups] for optimizer in policy.optimizers])


@torch.no_grad()
def evaluate_live(loop, data):
    """Use the current teacher policy, including after the controlled switch."""
    served = loop.policy.served_model()
    device = loop.policy.device
    metrics = {}
    snapshots = {}
    for name in ("fit", "guard", "test"):
        residuals, targets = [], []
        for context in data[name]["context"].split(1):
            context = context.to(device)
            residuals.append(served.routed_forward(context, perturb=False, output_noise=False).cpu())
            targets.append(served.encoder.teacher_velocity(context).cpu())
        residual, target = torch.cat(residuals), torch.cat(targets)
        # The cached partial background is immutable and excludes precisely the
        # two changing branches. It remains a valid baseline for both signs.
        baseline = data[name]["partial"]
        mse = float(residual.square().mean())
        baseline_mse = float((baseline - target).square().mean())
        metrics[name] = dict(velocity_rmse=mse**.5, relative_missing_branch_mse=mse/max(baseline_mse, 1e-30),
                             missing_branch_rmse=baseline_mse**.5, contexts=len(residual))
        if name == "test":
            stream = torch.Generator(device=device).manual_seed(71)
            noisy = torch.cat([served.routed_forward(context.to(device), generator=stream,
                                                    perturb=True, output_noise=False).cpu()
                               for context in data[name]["context"].split(1)])
            snapshots = dict(clean_residual=residual, dv12_residual=noisy, live_targets=target,
                             condition=served.encoder.condition(data[name]["context"].to(device)).cpu(),
                             critic={key: value.cpu().clone() for key, value in served.critic.state_dict().items()},
                             source=served.source, output_sigma=served.output_sigma,
                             teacher_site_scale=float(served.encoder.teacher_site_scale))
    return metrics, snapshots


def recovery_auc(records, shift):
    curve = [(row["step"]-shift, row["evaluation"]["test"]["velocity_rmse"])
             for row in records if row["stage"] in ("shift", "recovery")]
    return sum((right[0]-left[0])*(left[1]+right[1])/2 for left, right in zip(curve, curve[1:]))


def train(args):
    output = args.output / args.arm
    output.mkdir(parents=True, exist_ok=True)
    if (output / "results.json").exists():
        raise ValueError("completed evidence already exists; choose a new output directory")
    data = torch.load(args.data, map_location="cpu", weights_only=False)
    teacher_sha = hashlib.sha256(args.teacher.read_bytes()).hexdigest()
    if teacher_sha != data["metadata"]["adapter_sha256"]:
        raise ValueError("teacher weights differ from the original paired cache")
    overrides = ({"reopen_signal": "none", "reopen_anchor": "hold"} if args.arm == "old"
                 else {"reopen_signal": "optimizer", "reopen_anchor": "release"})
    provenance = dict(**optimizer_provenance(), data_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),
                      teacher_sha256=teacher_sha, task="two-branch teacher reversal on fixed original trajectory contexts",
                      warmup=args.warmup, recovery=args.recovery, probe_interval=args.probe_interval,
                      evaluation_interval=args.eval_every, output_sigma_for_critic_panel=.125,
                      selection="final fixed horizon, no metric selection", arm=args.arm,
                      gpu=torch.cuda.get_device_name(0))
    provenance["teacher_final_site_scale"] = args.site_scale
    runtime = SupraRuntime(device="cuda:0", rank=16, allow_hub=False)
    runtime.load(args.teacher)
    runtime.model.requires_grad_(False)
    runtime.text_encoder.to("cpu")
    loop = make_loop(runtime.model, data, mode="full", probe_interval=args.probe_interval,
                     recipe_overrides=overrides)
    if args.resume is not None:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        loop.config["teacher_schedule"] = deepcopy(saved["config"]["teacher_schedule"])
        restore(loop, saved)
        del saved
        gc.collect()
        provenance["resume_path"] = str(args.resume.resolve())
    shift_step = loop.policy.completed_steps + args.warmup
    provenance["warmup"] = shift_step
    loop.config["teacher_schedule"] = dict(initial_site_scale=float(loop.policy.encoder.teacher_site_scale),
                                           final_site_scale=args.site_scale, switch_after=shift_step,
                                           recovery_updates=args.recovery)
    frozen = frozen_digest(loop)
    initial = initial_digest(loop)
    records, traces, snapshots = [], [], {}
    warm_input, warm_behavior, recovery_input = [], [], []

    def report(stage):
        evaluation, snapshot = evaluate_live(loop, data)
        row = dict(event="evaluate", stage=stage, step=loop.policy.completed_steps, arm=args.arm,
                   evaluation=evaluation, controls=game_controls(loop.policy),
                   elapsed_seconds=time.perf_counter()-started)
        records.append(row)
        if stage in ("initial", "pre_shift", "shift", "final") or stage == "recovery":
            snapshots[f"{stage}_{row['step']}"] = snapshot
        emit(row)

    started = time.perf_counter()
    emit(dict(event="start", provenance=provenance, config=loop.config, initial_weights_digest=initial))
    report("initial")
    previous_fires = 0
    for iteration in range(1, args.warmup+args.recovery+1):
        step = loop.policy.completed_steps+1
        if iteration == args.warmup+1:
            if records[-1]["stage"] != "pre_shift":
                report("pre_shift")
            pre_shift_digest = state_digest({"models": {role: module.state_dict() for role, module in
                                                         loop.policy._training_modules().items()},
                                            "table": loop.policy.table})
            switch_teacher(loop, args.site_scale)
            report("shift")
        with torch.autograd.set_multithreading_enabled(False):
            row = update(loop)
        for name in ("loss_d", "loss_g", "penalty", "bank_grad_norm", "output_sigma"):
            if not torch.isfinite(torch.tensor(row[name])):
                raise RuntimeError(f"nonfinite {name} at {step}")
        traces.append(row)
        inputs = (row["batch_indices"], row["base_noise_sums"], row["paired_rng_digest"], row["dv12_rng_digest"])
        if step <= shift_step:
            warm_input.append(inputs)
            warm_behavior.append({key: row[key] for key in
                                  ("loss_d", "loss_d_game", "loss_g", "penalty", "output_sigma", "move", "bank_grad_norm")})
        else:
            recovery_input.append(inputs)
        fires = row["optimizer_surprise_fires"]
        if step % 50 == 0 or fires > previous_fires or (row["move"] and row["move"].get("moves", 0)):
            emit(dict(event="train", arm=args.arm, controls=game_controls(loop.policy), **row))
        previous_fires = fires
        if step % args.eval_every == 0:
            report("warmup" if step < shift_step else "pre_shift" if step == shift_step else "recovery")
    report("final")
    if frozen != frozen_digest(loop):
        raise RuntimeError("a frozen student/teacher parameter changed")
    torch.save(checkpoint(loop), output / "final.pt")
    torch.save(snapshots, output / "game-snapshots.pt")
    (output / "trace.json").write_text(json.dumps(traces, indent=2)+"\n")
    result = dict(provenance=provenance, config=loop.config, records=records,
                  initial_weights_digest=initial, pre_shift_model_digest=pre_shift_digest,
                  warmup_input_digest=state_digest(warm_input), recovery_input_digest=state_digest(recovery_input),
                  warmup_behavior_digest=state_digest(warm_behavior), frozen_weights_unchanged=True,
                  recovery_auc=recovery_auc(records, shift_step), controls=game_controls(loop.policy),
                  accepted_moves=[row for row in traces if row["move"] and row["move"].get("moves", 0)],
                  final_evaluation=records[-1]["evaluation"])
    (output / "results.json").write_text(json.dumps(result, indent=2, default=str)+"\n")
    emit(dict(event="complete", path=str(output / "results.json"), arm=args.arm,
              final=result["final_evaluation"], recovery_auc=result["recovery_auc"], controls=result["controls"]))


def critic_panel(old, new):
    # A fixed private paired-noise stream evaluates every generator under every
    # critic. It never feeds a training decision, checkpoint or row controller.
    draws = torch.randn(4, len(old["condition"]), 256, 16, generator=torch.Generator().manual_seed(72))
    conditions = old["condition"].repeat(4, 1)
    if not torch.equal(old["condition"], new["condition"]) or not torch.equal(old["live_targets"], new["live_targets"]):
        raise RuntimeError("critic panel contexts/teacher fields differ")
    from particlegan import get_recipe
    loss = get_recipe("e22_routed").make_loss()
    result = {}
    for critic_name, critic_snapshot in (("old", old), ("new", new)):
        state = critic_snapshot["critic"]
        critic = ConditionalTokenCritic(state["scale"]).eval().requires_grad_(False)
        critic.load_state_dict(state)
        real = (.125*draws).flatten(0, 1)
        with torch.no_grad():
            real_logits = critic(real, conditions)
            for generator_name, generator_snapshot in (("old", old), ("new", new)):
                for law in ("clean", "dv12"):
                    error = patchify(generator_snapshot[f"{law}_residual"]) / critic.scale
                    fake = real + error.repeat(4, 1, 1)
                    fake_logits = critic(fake, conditions)
                    result[f"critic_{critic_name}_generator_{generator_name}_{law}"] = dict(
                        loss_g=float(loss.g_loss(fake_logits, real_logits)),
                        loss_d_game=float(loss.d_loss(real_logits, fake_logits)),
                        score_gap=float((fake_logits-real_logits).mean()))
    return result


def summarize(args):
    reports = {arm: json.loads((args.output / arm / "results.json").read_text()) for arm in ("old", "new")}
    snapshots = {arm: torch.load(args.output / arm / "game-snapshots.pt", map_location="cpu", weights_only=False)
                 for arm in ("old", "new")}
    old, new = reports["old"], reports["new"]
    integrity = {key: old[key] == new[key] for key in
                 ("initial_weights_digest", "pre_shift_model_digest", "warmup_input_digest", "recovery_input_digest", "warmup_behavior_digest")}
    integrity["data_and_teacher"] = all(old["provenance"][key] == new["provenance"][key]
                                         for key in ("data_sha256", "teacher_sha256", "warmup", "recovery", "probe_interval"))
    integrity["teacher_schedule"] = old["config"]["teacher_schedule"] == new["config"]["teacher_schedule"]
    panels = {key: critic_panel(snapshots["old"][key], snapshots["new"][key]) for key in snapshots["old"]
              if key.startswith(("shift_", "recovery_", "final_"))}
    result = dict(integrity=integrity, arms=reports, critic_panels=panels,
                  evaluation_only_auc_ratio=new["recovery_auc"]/old["recovery_auc"],
                  evaluation_only_final_rmse_ratio=new["final_evaluation"]["test"]["velocity_rmse"]/old["final_evaluation"]["test"]["velocity_rmse"],
                  limits=["One fixed initialization and two held-out prompts, eight original trajectory contexts.",
                          "Controlled two-branch teacher reversal; not a general image-quality benchmark.",
                          "Only two particle sites; 69 published ordinary LoRA branches remain frozen.",
                          "All output metrics and critic panels are reporting only."])
    (args.output / "comparison.json").write_text(json.dumps(result, indent=2)+"\n")
    emit(dict(event="comparison", path=str(args.output / "comparison.json"), integrity=integrity,
              auc_ratio=result["evaluation_only_auc_ratio"], final_rmse_ratio=result["evaluation_only_final_rmse_ratio"]))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for arm, report in reports.items():
        curve = [row for row in report["records"] if row["stage"] in ("shift", "recovery")]
        axes[0].plot([row["step"]-report["provenance"]["warmup"] for row in curve],
                     [row["evaluation"]["test"]["velocity_rmse"] for row in curve], "o-", label=arm)
        trace = json.loads((args.output / arm / "trace.json").read_text())
        shift = report["provenance"]["warmup"]
        selected = [row for row in trace if row["step"] > shift]
        axes[1].plot([row["step"]-shift for row in selected], [row["loss_g"] for row in selected], alpha=.6, label=arm)
    axes[0].set_ylabel("Held-out velocity RMSE (evaluation only)")
    axes[1].set_ylabel("Native G game loss (arm-specific critic)")
    axes[1].set_xlabel("Updates after teacher reversal")
    for axis in axes:
        axis.legend()
        axis.grid(alpha=.2)
    figure.tight_layout()
    figure.savefig(args.output / "recovery.png", dpi=160)
    plt.close(figure)
    if not integrity["initial_weights_digest"] or not integrity["warmup_input_digest"] or not integrity["recovery_input_digest"] or not integrity["data_and_teacher"]:
        raise RuntimeError("matched comparison integrity failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("old", "new", "summarize"), required=True)
    parser.add_argument("--data", type=Path, default=Path("outputs/e22-pilot/data.pt"))
    parser.add_argument("--teacher", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/e22-moving-f459cb6d"))
    parser.add_argument("--warmup", type=int, default=900)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--site-scale", type=float, default=-1.)
    parser.add_argument("--recovery", type=int, default=300)
    parser.add_argument("--probe-interval", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=50)
    args = parser.parse_args()
    torch.set_num_threads(8)
    if args.arm == "summarize":
        summarize(args)
    else:
        if (args.teacher is None or args.warmup < 0 or (args.warmup == 0 and args.resume is None)
                or min(args.recovery, args.probe_interval, args.eval_every) <= 0):
            parser.error("training needs a teacher and positive update intervals")
        train(args)


if __name__ == "__main__":
    main()
