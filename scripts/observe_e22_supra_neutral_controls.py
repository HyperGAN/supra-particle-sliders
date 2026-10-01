#!/usr/bin/env python3
"""Bounded saved-control observation; no model, update, or bulk tensor hash."""
import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("sampled_control", "sampled_hb_neutral")


def small(value, depth=0):
    """Read scalar/small control tensors only, never bulk owner tensors."""
    if isinstance(value, torch.Tensor):
        if value.numel() <= 64:
            return value.detach().cpu().tolist()
        return dict(omitted_tensor_shape=list(value.shape), dtype=str(value.dtype))
    if isinstance(value, dict):
        return {str(key): small(item, depth + 1) for key, item in value.items()} if depth < 8 else "omitted deep state"
    if isinstance(value, (list, tuple)):
        return [small(item, depth + 1) for item in value] if len(value) <= 256 else dict(omitted_sequence_length=len(value))
    if isinstance(value, float) and not math.isfinite(value):
        return dict(undefined_control_value=repr(value))
    return value


def observe(path):
    state = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    policy = state["policy"]
    groups = []
    for index, (optimizer, bases, roles, testers) in enumerate(zip(
            policy["optimizers"], policy["initial_lrs"], policy["roles"], policy["lr_settle"])):
        for group_index, (group, base, role, tester) in enumerate(zip(optimizer["param_groups"], bases, roles, testers)):
            controls = {name: tester.get(name) for name in (
                "s", "b", "tau", "blocks_in_window", "last_decisive", "last_decisive_scale",
                "last", "counts", "windows", "log")} if tester is not None else None
            groups.append(dict(optimizer=index, group=group_index, role=role, base_lr=base,
                saved_applied_lr=group["lr"], saved_applied_scale=group["lr"] / base,
                tester=small(controls)))
    record = policy["optimizers"][1]["regularizer"]["record"]
    controller = policy["controller"]
    noise_raw = float(policy["output_noise"])
    table_scales = [item["tester"]["s"] for item in groups if item["role"] == "table" and item["tester"]["s"] is not None]
    noncritic = [item["tester"]["s"] for item in groups if item["optimizer"] != 1 and item["role"] != "noise"]
    noise_settle = 1. if noncritic and all(value is not None for value in noncritic) and any(value > 1. / 64. for value in noncritic) else controller["mobility"]
    stat = path.stat()
    result = dict(checkpoint=str(path), checkpoint_file_metadata=dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns),
        step=policy["completed_steps"], groups=groups,
        noise=dict(log_parameter=noise_raw, unconstrained_sigma=math.exp(noise_raw),
            last_saved_sigma=policy["last_output_sigma"], nominal_sigma=policy["recipe"]["output_noise_std"],
            model_table_settlement_floor=policy["recipe"]["output_noise_std"] * noise_settle),
        critic=dict(table_support_floor=.75 * max(table_scales) if table_scales else None,
            current_payoff_damping=1. / (1. + controller["payoff_error"] ** 2)),
        ka2=small(record), controller=small(controller), surprise=small(policy["surprise"]),
        reopen_guard=small(policy["reopen_guard"]), routing_counters=small(policy["routing"]["counters"]),
        application_initialization=state["config"]["particle_init"])
    del state, policy, optimizer, testers, tester, group, record, controller
    gc.collect()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "outputs/e22-supra-neutral-initialization-6400")
    parser.add_argument("--particlegan-root", type=Path, default=ROOT.parent / "ParticleGAN-supra-neutral-develop")
    parser.add_argument("--steps", type=int, nargs="+", default=[2000, 2400])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    started = time.monotonic()
    snapshots = {arm: {str(step): observe(args.run / arm / f"checkpoint-{step:05d}.pt") for step in args.steps} for arm in ARMS}
    transitions = {}
    for arm, states in snapshots.items():
        earlier, later = (states[str(step)] for step in (min(args.steps), max(args.steps)))
        transitions[arm] = dict(group_changes={a["role"]: dict(
            earlier_raw_scale=a["tester"]["s"] if a["tester"] else None,
            later_raw_scale=b["tester"]["s"] if b["tester"] else None,
            earlier_applied_lr=a["saved_applied_lr"], later_applied_lr=b["saved_applied_lr"],
            earlier_counts=a["tester"]["counts"] if a["tester"] else None,
            later_counts=b["tester"]["counts"] if b["tester"] else None,
            new_log_events=[event for event in (b["tester"]["log"] if b["tester"] else [])
                            if event not in (a["tester"]["log"] if a["tester"] else [])])
            for a, b in zip(earlier["groups"], later["groups"])},
            earlier_noise=earlier["noise"], later_noise=later["noise"],
            earlier_routing=earlier["routing_counters"], later_routing=later["routing_counters"])
    native_names = ("particlegan/policy.py", "particlegan/continuous.py", "particlegan/ka2.py")
    result = dict(schema="supra_neutral_controls_observation_v1", snapshots=snapshots, transitions=transitions,
        reviewer_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        native_source_sha256={name: hashlib.sha256((args.particlegan_root / name).read_bytes()).hexdigest() for name in native_names},
        scope="CPU mmap scalar/small control metadata only; no full file or tensor hash, model, optimizer update, or replay.",
        native_or_generator_updates=0, model_forward_calls=0, bulk_file_or_tensor_hashes=0,
        observation_seconds=time.monotonic() - started,
        limits=["Saved group LRs were applied at begin_step; raw tester/controller fields include the end-of-step observations.",
                "Two saved boundaries plus their stored event histories identify control transitions, not their causal effect.",
                "Stored Gaussian/noise equality does not establish unchanged model gradients, critic payoffs, or routing geometry."])
    target = args.output or args.run / "control-observation-02000-02400.json"
    target.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(output=str(target), snapshots=sum(map(len, snapshots.values())), observation_seconds=result["observation_seconds"])))


if __name__ == "__main__":
    main()
