#!/usr/bin/env python3
"""Read-only whole-pool particle contribution under two fixed native critics.

The three predeclared interventions are evaluations, never optimizer choices.
The primary observations are paired RpGAN payoff changes; output intervention
RMS is reporting only. Native checkpoints and their RNGs remain unchanged.
"""
import argparse
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import torch.nn.functional as F
from supra.particle_adapter import NONLINEAR_V1
from supra.particle_game import ConditionalTokenCritic, patchify
from supra.particle_pilot import checkpoint, restore, state_digest
from supra.particle_training import make_training_loop
from supra.runtime import model_module, TARGETS

ARMS = ("clean", "zero_particle_codes", "mass_only_routing")
DRAWS = 4


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def emit(**row):
    print(json.dumps(row, default=str), flush=True)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2)+"\n")
    temporary.replace(path)


@contextmanager
def mass_only_routing(router):
    """Remove activation-dependent logits while retaining bank values/masses."""
    handles = []
    calls = [0]
    def zero_query(_, __, value):
        calls[0] += 1
        return torch.zeros_like(value)
    try:
        for site in router.sites:
            handles.append(router.query_for_site(site).register_forward_hook(zero_query))
        yield calls
    finally:
        for handle in handles:
            handle.remove()


@torch.no_grad()
def evaluate_pool(served, pool, critics, *, sigma=.125, pool_name="test", batch_size=4):
    """Evaluate every context, with identical paired panels across all arms/Ds."""
    records = {arm: [] for arm in ARMS}
    stream = torch.Generator(device="cpu").manual_seed(72)
    noise_hash = hashlib.sha256()
    candidate = served.routing.candidate_for(served.models, served.table,
                                             averaged=served.source=="averaged")
    device = served.table.device
    for start in range(0, len(pool["context"]), batch_size):
        context = pool["context"][start:start+batch_size].to(device)
        condition = served.encoder.condition(context)
        clean = served.routing.forward(served.models, context, candidate)
        code_calls = [0]
        def zero_code(codes):
            code_calls[0] += 1
            return torch.zeros_like(codes)
        zero_codes = served.routing.forward(served.models, context, candidate, perturb_fn=zero_code)
        if code_calls[0] != len(served.generator.sites):
            raise RuntimeError("code ablation did not visit every declared native site")
        with mass_only_routing(served.router) as routing_calls:
            mass_only = served.routing.forward(served.models, context, candidate)
        if routing_calls[0] != len(served.generator.sites):
            raise RuntimeError("mass-only routing did not visit every declared native site")
        if any(branch.frame is not None for branch in served.generator.particle_branches()) or served.generator._in_forward:
            raise RuntimeError("particle intervention retained a completed host frame")
        predictions = dict(clean=clean, zero_particle_codes=zero_codes, mass_only_routing=mass_only)
        noise = torch.randn(DRAWS, len(context), 256, 16, generator=stream, dtype=torch.float32)
        noise_hash.update(noise.numpy().tobytes())
        real = (sigma*noise.to(device)).flatten(0, 1)
        repeated_condition = condition.repeat(DRAWS, 1)
        logits = {}
        for label, critic in critics.items():
            real_scores = critic(real, repeated_condition)
            logits[label] = {}
            for arm, residual in predictions.items():
                if residual.shape != (len(context), 4, 32, 32) or not bool(torch.isfinite(residual).all()):
                    raise RuntimeError(f"invalid full-model residual in {arm}")
                standardized = patchify(residual).float()/critic.scale
                fake = real + standardized.repeat(DRAWS, 1, 1)
                fake_scores = critic(fake, repeated_condition)
                gap = (real_scores-fake_scores).reshape(DRAWS, len(context))
                logits[label][arm] = dict(g_game=F.softplus(gap), d_game=F.softplus(-gap), score_gap=gap)
        for arm, residual in predictions.items():
            change_square = (residual-clean).float().square().flatten(1).mean(1)
            for index in range(len(context)):
                record = dict(index=start+index, source_caption_id=int(context[index, 4097]),
                              time=float(context[index, 4096]), output_change_mse_evaluation_only=float(change_square[index]))
                for label in critics:
                    values = logits[label][arm]
                    clean_values = logits[label]["clean"]
                    record[label] = dict(
                        g_game=float(values["g_game"][:,index].mean()),
                        d_game=float(values["d_game"][:,index].mean()),
                        score_gap=float(values["score_gap"][:,index].mean()),
                        per_draw_g_game=values["g_game"][:,index].cpu().tolist(),
                        per_draw_g_game_delta_from_clean=(values["g_game"][:,index]-clean_values["g_game"][:,index]).cpu().tolist(),
                        g_game_delta_from_clean=float((values["g_game"][:,index]-clean_values["g_game"][:,index]).mean()))
                records[arm].append(record)
        if (start//batch_size+1)%10==0 or start+batch_size>=len(pool["context"]):
            emit(event="particle_contribution_pool_progress", pool=pool_name,
                 completed=min(start+batch_size, len(pool["context"])), total=len(pool["context"]))

    def summarize(rows):
        total = len(rows)
        result = dict(contexts=total,
            output_change_rms_evaluation_only=(sum(row["output_change_mse_evaluation_only"] for row in rows)/total)**.5)
        for label in critics:
            result[label] = dict(
                g_game=sum(row[label]["g_game"] for row in rows)/total,
                d_game=sum(row[label]["d_game"] for row in rows)/total,
                score_gap=sum(row[label]["score_gap"] for row in rows)/total,
                g_game_delta_from_clean=sum(row[label]["g_game_delta_from_clean"] for row in rows)/total,
                per_draw_g_game=[sum(row[label]["per_draw_g_game"][draw] for row in rows)/total for draw in range(DRAWS)],
                per_draw_g_game_delta_from_clean=[sum(row[label]["per_draw_g_game_delta_from_clean"][draw] for row in rows)/total for draw in range(DRAWS)],
                contexts_game_worsened=sum(row[label]["g_game_delta_from_clean"]>0 for row in rows),
                contexts_game_improved=sum(row[label]["g_game_delta_from_clean"]<0 for row in rows),
                contexts_game_unchanged=sum(row[label]["g_game_delta_from_clean"]==0 for row in rows))
        return result
    result = dict(contexts=len(pool["context"]), gaussian_panels=DRAWS,
                  paired_noise_sha256=noise_hash.hexdigest(), output_sigma=sigma,
                  per_arm={})
    for arm, rows in records.items():
        subject_ids = sorted({row["source_caption_id"] for row in rows})
        result["per_arm"][arm] = {**summarize(rows),
            "per_subject":{str(subject):summarize([row for row in rows if row["source_caption_id"]==subject])
                           for subject in subject_ids},
            "records":rows}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT/"outputs/e22-particle-v2-1600")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--reference", type=Path, default=ROOT/"outputs/e22-full-f459cb6d")
    parser.add_argument("--reference-checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--include-fit", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("choose a fresh output file")
    torch.set_num_threads(4)
    path = args.checkpoint or args.run/"final.pt"
    reference_path = args.reference_checkpoint or args.reference/"final.pt"
    pools = ("test", "preservation", "fit", "holds") if args.include_fit else ("test", "preservation")
    started = time.perf_counter()
    saved = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    reference = torch.load(reference_path, map_location="cpu", weights_only=False, mmap=True)
    data = torch.load(args.run/"data.pt", map_location="cpu", weights_only=False)
    architecture = saved["config"].get("architecture", NONLINEAR_V1)
    if reference["config"].get("architecture", NONLINEAR_V1) != NONLINEAR_V1 or reference["policy"]["completed_steps"] != 1600:
        raise RuntimeError("fixed reference critic must come from the archived V1-1600 checkpoint")
    if saved["config"]["dataset_digest"] != reference["config"]["dataset_digest"] or state_digest(data) != saved["config"]["dataset_digest"]:
        raise RuntimeError("contribution panels require the same unchanged cached context data")
    if not torch.equal(saved["policy"]["models"]["critic"]["scale"], reference["policy"]["models"]["critic"]["scale"]):
        raise RuntimeError("current and archived critics use different error-coordinate units")
    if str(torch.device(args.device)) != saved["policy"]["device"]:
        raise RuntimeError("exact native restore requires the chosen checkpoint's logical device")
    emit(event="contribution_predeclared", checkpoint=path, architecture=architecture,
         completed_steps=saved["policy"]["completed_steps"], pools=pools, arms=ARMS,
         critics=["fixed_v1_1600_D", "current_final_D"], draws=DRAWS,
         output_metrics="intervention reporting only; no training, selection or optimizer steps")
    module = model_module()
    with torch.random.fork_rng(devices=[]), torch.device("meta"):
        base = module.SupraDiT()
        module.attach_supra_lora(base, rank=16, alpha=16, targets=TARGETS)
    base.load_state_dict({key.removeprefix("teacher."):value for key,value in
        saved["policy"]["models"]["encoder"].items() if key.startswith("teacher.")}, strict=True, assign=True)
    base.to(args.device).eval().requires_grad_(False)
    loop = make_training_loop(base, data, device=args.device,
        probe_interval=saved["config"]["probe_interval"], branch_lr=saved["config"]["branch_lr"],
        architecture=architecture)
    restore(loop, saved)
    before = state_digest(checkpoint(loop))
    if before != state_digest(saved):
        raise RuntimeError("contribution evaluator did not restore the chosen complete native checkpoint")
    served = loop.policy.served_model()
    served_before = state_digest({name:model.state_dict() for name,model in served.models.items()})
    served_table_before = state_digest(served.table)
    if served.generator.architecture != architecture:
        raise RuntimeError("public serving changed particle architecture")
    with torch.random.fork_rng(devices=[]):
        fixed = ConditionalTokenCritic(reference["policy"]["models"]["critic"]["scale"])
    fixed.load_state_dict(reference["policy"]["models"]["critic"], strict=True)
    fixed = fixed.to(args.device).eval().requires_grad_(False)
    critics = dict(fixed_v1_1600_D=fixed, current_final_D=deepcopy(loop.policy.D).eval().requires_grad_(False))
    critic_hashes = {name:state_digest(critic.state_dict()) for name,critic in critics.items()}
    report = dict(metadata=dict(checkpoint=str(path.resolve()), checkpoint_sha256=sha(path),
        reference_checkpoint=str(reference_path.resolve()), reference_checkpoint_sha256=sha(reference_path),
        completed_steps=loop.policy.completed_steps, architecture=architecture,
        served_source=served.source, bank_shape=list(served.table.shape), native_sites=len(served.generator.sites),
        critic_state_sha256=critic_hashes, arms=list(ARMS), draws=DRAWS, output_sigma=.125,
        perturbation="all forwards clean; code ablation delivered at every native routing site",
        mass_only_routing="zero every site query; retain active bank values and represented row masses",
        noise="one private CPU generator 72 per pool, four matched Gaussian panels for every arm and critic",
        native_game="paired RpGAN softplus(real_score - fake_score), unweighted diagnostic payoffs",
        evaluation_only=True, optimizer_updates=0,
        limitations="Both critics are trained on this task and are endogenous game witnesses, not independent image-quality judges. Contribution depends on the fixed context pools and interventions."), pools={})
    for name in pools:
        report["pools"][name] = evaluate_pool(served, data[name], critics, pool_name=name)
        summary = {arm:{label:row[label]["g_game_delta_from_clean"] for label in critics}
                   for arm,row in report["pools"][name]["per_arm"].items()}
        emit(event="contribution_pool_complete", pool=name, deltas=summary)
        write_json(args.output, report)
    if before != state_digest(checkpoint(loop)):
        raise RuntimeError("whole-pool contribution evaluation changed native checkpoint or RNG")
    if served_before != state_digest({name:model.state_dict() for name,model in served.models.items()}) or served_table_before != state_digest(served.table):
        raise RuntimeError("particle interventions changed the served model or bank")
    if critic_hashes != {name:state_digest(critic.state_dict()) for name,critic in critics.items()}:
        raise RuntimeError("evaluation changed a fixed critic")
    report.update(full_checkpoint_and_RNG_unchanged=True, served_model_and_bank_unchanged=True,
                  fixed_critics_unchanged=True, seconds=time.perf_counter()-started)
    write_json(args.output, report)
    emit(event="particle_contribution_complete", output=args.output, seconds=report["seconds"],
         full_native_state_unchanged=True)


if __name__ == "__main__":
    main()
