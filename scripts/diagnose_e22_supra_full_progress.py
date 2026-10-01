#!/usr/bin/env python3
"""Reporting-only progress across existing full Supra ParticleGAN checkpoints."""
import argparse
from copy import deepcopy
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

from particlegan import get_recipe
from supra.particle_adapter import NONLINEAR_V1
from supra.particle_game import ConditionalTokenCritic, patchify
from supra.particle_pilot import restore
from supra.particle_training import make_training_loop, raw_velocity
from supra.runtime import SupraRuntime


def emit(value):
    print(json.dumps(value, sort_keys=True), flush=True)


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unchanged_state(loop):
    policy = loop.policy
    return dict(step=policy.completed_steps, penalty_calls=policy.opt_d.record.calls,
                fires=None if policy.surprise is None else policy.surprise.fires,
                streams={name: getattr(policy, name).get_state().cpu().clone() for name in
                         ("latent_generator", "penalty_generator", "eval_generator", "noise_generator")},
                data_rng=loop.data_rng.get_state().cpu().clone(),
                paired_rng=loop.paired_noise_rng.get_state().cpu().clone(),
                cpu_rng=torch.get_rng_state().clone(),
                cuda_rng=torch.cuda.get_rng_state(policy.device).cpu().clone())


def same_state(left, right):
    for key in left:
        if isinstance(left[key], dict):
            if not same_state(left[key], right[key]):
                return False
        elif isinstance(left[key], torch.Tensor):
            if not torch.equal(left[key], right[key]):
                return False
        elif left[key] != right[key]:
            return False
    return True


@torch.no_grad()
def evaluate_pool(pool, served, fixed_critic, paired_draws, *, name, step):
    mse_values, gap_values, payoff_values, d_payoff_values, score_gaps = [], [], [], [], []
    loss = get_recipe("e22_routed").make_loss()
    device = served.table.device
    for start in range(0, len(pool["context"]), 4):
        context = pool["context"][start:start+4].to(device)
        target = served.encoder.teacher_velocity(context)
        prediction = raw_velocity(served, context)
        residual = prediction - target
        if not bool(torch.isfinite(residual).all()):
            raise RuntimeError("nonfinite checkpoint validation residual")
        mse_values.extend(residual.square().flatten(1).mean(1).cpu().tolist())
        base_context = context.clone()
        base_context[:, 4098] = base_context[:, 4097]
        base = served.encoder.teacher_velocity(base_context)
        gap_values.extend((base-target).square().flatten(1).mean(1).cpu().tolist())
        condition = served.encoder.condition(context)
        real = (.125 * paired_draws[:, start:start+len(context)]).flatten(0, 1).to(device)
        fake = real + (patchify(residual) / fixed_critic.scale).repeat(4, 1, 1)
        conditions = condition.repeat(4, 1)
        real_logits, fake_logits = fixed_critic(real, conditions), fixed_critic(fake, conditions)
        payoff_values.append((len(context), float(loss.g_loss(fake_logits, real_logits))))
        d_payoff_values.append((len(context), float(loss.d_loss(real_logits, fake_logits))))
        score_gaps.append((len(context), float((fake_logits-real_logits).mean())))
        if (start // 4 + 1) % 20 == 0 or start + 4 >= len(pool["context"]):
            emit(dict(event="progress", checkpoint_step=step, pool=name,
                      contexts=min(start+4, len(pool["context"])), total=len(pool["context"])))
    count = len(mse_values)
    mse = sum(mse_values) / count
    gap = sum(gap_values) / count
    weighted_mean = lambda values: sum(size * value for size, value in values) / count
    return dict(contexts=count, subjects=len(set(pool["prompts"])), rmse=math.sqrt(mse), mse=mse,
                base_gap_mse=gap, velocity_ratio=None if not gap else mse/gap,
                fixed_final_critic_g_loss=weighted_mean(payoff_values),
                fixed_final_critic_d_game_loss=weighted_mean(d_payoff_values),
                fixed_final_critic_score_gap=weighted_mean(score_gaps))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "outputs/e22-full-f459cb6d")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    output = args.out or args.run / "progress-diagnostic.json"
    if output.exists():
        parser.error("diagnostic evidence already exists")
    started = time.perf_counter()
    torch.set_num_threads(8)
    data = torch.load(args.run / "data.pt", map_location="cpu", weights_only=False)
    final = torch.load(args.run / "checkpoint-01600.pt", map_location="cpu", weights_only=False, mmap=True)
    config = deepcopy(final["config"])
    fixed_state = {key: value.clone() for key, value in final["policy"]["models"]["critic"].items()}
    fixed_critic = ConditionalTokenCritic(fixed_state["scale"]).eval().requires_grad_(False)
    fixed_critic.load_state_dict(fixed_state)
    fixed_critic.to(args.device)
    del final
    stream = torch.Generator().manual_seed(72)
    draws = {name: torch.randn(4, len(data[name]["context"]), 256, 16, generator=stream)
             for name in ("test", "preservation")}
    runtime = SupraRuntime(device=args.device, rank=16, allow_hub=False)
    runtime.model.requires_grad_(False)
    runtime.text_encoder.to("cpu")
    loop = make_training_loop(runtime.model, data, device=args.device,
                              probe_interval=config["probe_interval"], branch_lr=config["branch_lr"],
                              architecture=config.get("architecture", NONLINEAR_V1))
    rows = []
    for step in (400, 800, 1200, 1600):
        path = args.run / f"checkpoint-{step:05d}.pt"
        emit(dict(event="loading", step=step, elapsed_seconds=time.perf_counter()-started))
        saved = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        restore(loop, saved)
        del saved
        gc.collect()
        before = unchanged_state(loop)
        served = loop.policy.served_model()
        row = dict(step=step, served_source=served.source,
                   validation=evaluate_pool(data["test"], served, fixed_critic, draws["test"], name="validation", step=step),
                   preservation=evaluate_pool(data["preservation"], served, fixed_critic, draws["preservation"], name="preservation", step=step),
                   counters_and_streams_unchanged=same_state(before, unchanged_state(loop)))
        if not row["counters_and_streams_unchanged"]:
            raise RuntimeError("diagnostic changed training counters or random streams")
        rows.append(row)
        emit(dict(event="checkpoint_complete", elapsed_seconds=time.perf_counter()-started, **row))
        del served
        gc.collect()
    existing = json.loads((args.run / "evaluation/comparison.json").read_text())
    reproduction = {name: rows[-1][key]["rmse"] == existing[name]["new_particlegan"]["rmse"]
                    for name, key in (("validation", "validation"), ("preservation", "preservation"))}
    if not all(reproduction.values()):
        raise RuntimeError("1600 checkpoint did not reproduce existing final evaluation")
    report = dict(records=rows, final_evaluation_exactly_reproduced=reproduction,
                  fixed_critic_step=1600, paired_noise_sigma=.125, paired_noise_draws=4,
                  paired_noise_stream="private CPU generator 72; identical draws for every checkpoint",
                  data_sha256=sha_file(args.run / "data.pt"), evaluation_only=True,
                  checkpoint_selection="none; every preexisting400-update checkpoint evaluated",
                  native_model="pure base; no released adapter loaded",
                  validation_scope="24 cached trajectories of the same six training subjects: six subjects x two noise draws x neutral/positive paths; not unseen subjects.",
                  limitations=["Four stored checkpoints; no new training or convergence guarantee.",
                               "Held-final-critic game diagnostic measures a fixed critic, not a re-trained game at each checkpoint."],
                  seconds=time.perf_counter()-started)
    output.write_text(json.dumps(report, indent=2)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for key in ("validation", "preservation"):
        axes[0].plot([row["step"] for row in rows], [row[key]["rmse"] for row in rows], "o-", label=key)
        axes[1].plot([row["step"] for row in rows], [row[key]["fixed_final_critic_g_loss"] for row in rows], "o-", label=key)
    axes[0].set_ylabel("Velocity RMSE (reporting only)")
    axes[1].set_ylabel("G game loss under fixed final critic")
    axes[1].set_xlabel("Stored training update")
    for axis in axes:
        axis.legend()
        axis.grid(alpha=.2)
    figure.tight_layout()
    figure.savefig(output.with_suffix(".png"), dpi=150)
    plt.close(figure)
    emit(dict(event="complete", path=str(output), seconds=time.perf_counter()-started,
              final_evaluation_exactly_reproduced=reproduction))


if __name__ == "__main__":
    main()
