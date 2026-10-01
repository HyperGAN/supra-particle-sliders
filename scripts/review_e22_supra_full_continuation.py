#!/usr/bin/env python3
"""CPU-only audit of preserved sampling and native continuation controls."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_trace(path):
    text = Path(path).read_text()
    lines = text.splitlines()
    if text and not text.endswith("\n"):
        lines.pop()
    return [json.loads(line) for line in lines if line]


def rng_digest(state):
    return hashlib.sha256(state.cpu().contiguous().numpy().tobytes()).hexdigest()


def checkpoint_controls(path):
    state = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    policy = state["policy"]
    roles = []
    for role_names, testers, optimizer, initial in zip(policy["roles"], policy["lr_settle"],
                                                      policy["optimizers"], policy["initial_lrs"]):
        for role, tester, group, initial_lr in zip(role_names, testers, optimizer["param_groups"], initial):
            roles.append(dict(role=role, lr=group["lr"], initial_lr=initial_lr,
                              stationarity=None if tester is None else {
                                  key: tester[key] for key in ("s", "b", "tau", "last_decisive", "counts")},
                              last_decision=None if tester is None else tester["last"].get("decision")))
    record = policy["optimizers"][1]["regularizer"]["record"]
    surprise = policy.get("surprise")
    mass = policy["models"]["router"]["log_mass"]
    weights = mass.softmax(0)
    return dict(step=policy["completed_steps"], roles=roles,
                served_source=policy["served_source"],
                application_data_rng_digest=rng_digest(state["data_rng"]),
                paired_rng_digest=rng_digest(state["paired_noise_rng"]),
                native_dv12_rng_digest=rng_digest(policy["streams"]["noise_generator"]),
                native_ka2={key: record[key] for key in
                            ("calls", "observed_steps", "w", "alpha", "last_ratio",
                             "ema_updates", "ema_skips", "ema_reseeds")},
                optimizer_surprise=None if surprise is None else
                    {key: surprise[key] for key in ("fires", "armed", "streak", "last_ratio", "anchor_events", "log")},
                routing_counters=policy["routing"]["counters"],
                routing_probe_clock=policy["routing"]["probe_clock"],
                controller={key: value for key, value in policy["controller"].items()
                            if isinstance(value, (str, int, float, bool, type(None)))},
                bank_finite=bool(torch.isfinite(policy["table"]).all()),
                represented_active_rows=int(torch.isfinite(mass).sum()),
                represented_row_mass_ess=float(1 / weights.square().sum()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "outputs/e22-full-f459cb6d-6400")
    parser.add_argument("--initial", type=Path, default=ROOT / "outputs/e22-full-f459cb6d")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    original, continued = read_trace(args.initial / "train.jsonl"), read_trace(args.run / "train.jsonl")
    combined = original + continued
    if not continued:
        raise ValueError("continued training has not produced a complete trace row")
    data = torch.load(args.run / "data.pt", map_location="cpu", weights_only=False)
    saved = torch.load(args.initial / "final.pt", map_location="cpu", weights_only=False, mmap=True)
    source_step = saved["policy"]["completed_steps"]
    controls = checkpoint_controls(args.checkpoint) if args.checkpoint else None
    checkpoint_data_rng_digest = None
    stream = torch.Generator().manual_seed(7)
    errors, original_stream_match = [], False
    for step, row in enumerate(combined, 1):
        hold = step % 5 == 0
        pool = data["holds"] if hold else data["fit"]
        indices = torch.randint(len(pool["context"]), (4,), generator=stream).tolist()
        if row["step"] != step or row["batch_indices"] != indices or row["hold"] != hold:
            errors.append(step)
        if row["game_weight"] != (.1 if hold else 1.):
            errors.append(step)
        if step == source_step:
            original_stream_match = torch.equal(stream.get_state(), saved["data_rng"])
        if controls and step == controls["step"]:
            checkpoint_data_rng_digest = rng_digest(stream.get_state())
    del saved
    fields = ("loss_d", "loss_d_game", "loss_g", "penalty", "output_sigma", "bank_grad_norm")
    finite = all(math.isfinite(row[field]) for row in continued for field in fields)
    report = dict(source_step=source_step, first_continued_step=continued[0]["step"],
                  last_continued_step=continued[-1]["step"], continued_updates=len(continued),
                  continued_holds=sum(row["hold"] for row in continued),
                  sampler="Replayed the single declared CPU generator7 stream; no alternative initialization or training.",
                  original_data_rng_matches_after_1600=original_stream_match,
                  absolute_step_hold_and_batch_sequence_match=not errors,
                  sequence_mismatch_steps=sorted(set(errors)),
                  data_bytes_identical=digest_file(args.run / "data.pt") == digest_file(args.initial / "data.pt"),
                  all_training_trace_values_finite=finite,
                  penalty_calls_equal_absolute_steps=all(row["penalty_calls"] == row["step"] for row in continued),
                  penalty_phases=sorted(set(row["penalty_phase"] for row in continued)),
                  dense_bank_gradient_rows_min=min(row["dense_gradient_rows"] for row in continued),
                  dense_bank_gradient_rows_max=max(row["dense_gradient_rows"] for row in continued),
                  bank_gradient_norm_min=min(row["bank_grad_norm"] for row in continued),
                  bank_gradient_norm_max=max(row["bank_grad_norm"] for row in continued),
                  zero_bank_gradient_updates=sum(row["bank_grad_norm"] == 0 for row in continued),
                  r1_fires_end=continued[-1]["optimizer_surprise_fires"],
                  anchor_release_events_end=continued[-1]["anchor_release_events"],
                  proposal_records=sum(row["move"] is not None for row in continued),
                  accepted_move_updates=[row["step"] for row in continued if row["move"] and row["move"].get("moves", 0)],
                  output_metrics_used=False, gpu_used=False)
    if controls:
        report["checkpoint"] = controls
        traced_checkpoint = next((row for row in combined if row["step"] == controls["step"]), None)
        report["checkpoint_streams_match_trace_and_declared_sampler"] = dict(
            data_rng=checkpoint_data_rng_digest == controls["application_data_rng_digest"],
            paired_rng=traced_checkpoint is not None and traced_checkpoint["paired_rng_digest"] == controls["paired_rng_digest"],
            dv12_rng=traced_checkpoint is not None and traced_checkpoint["dv12_rng_digest"] == controls["native_dv12_rng_digest"])
    output = args.out or args.run / f"continuation-review-{continued[-1]['step']:05d}.json"
    output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(dict(path=str(output), **report), sort_keys=True), flush=True)
    if (errors or not original_stream_match or not report["data_bytes_identical"] or not finite
            or (controls and not all(report["checkpoint_streams_match_trace_and_declared_sampler"].values()))):
        raise RuntimeError("continuation audit failed")


if __name__ == "__main__":
    main()
