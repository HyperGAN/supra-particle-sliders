#!/usr/bin/env python3
"""Read-only CPU row-space comparison; no Supra forwards or training updates.

Compare both fresh sampled particle states with the fixed historical ordinary
LoRA endpoint. These weight-space observations are not an activation-weighted
task-span estimate or causal evidence about convergence. Constructing the
small reference toy performs its frozen teacher/base fixture forwards.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
from safetensors import safe_open
import torch

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def tensor_digest(values):
    result = hashlib.sha256()
    for name, value in sorted(values.items()):
        result.update(str((name, str(value.dtype), tuple(value.shape))).encode())
        result.update(value.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


def rng_digest():
    return hashlib.sha256(repr((random.getstate(), np.random.get_state(),
                               torch.get_rng_state().tolist())).encode()).hexdigest()


def summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {"n": int(values.size), "min": float(values.min()),
            "q25": float(np.quantile(values, .25)), "median": float(np.median(values)),
            "mean": float(values.mean()), "q75": float(np.quantile(values, .75)),
            "max": float(values.max())}


def row_basis(matrix):
    matrix = matrix.detach().double()
    singular = torch.linalg.svdvals(matrix)
    if singular[-1] <= singular[0] * 1e-10:
        raise AssertionError("declared down factor does not have full row rank")
    return torch.linalg.qr(matrix.T, mode="reduced").Q


def compare(initial, learned, up):
    left, right = row_basis(initial), row_basis(learned)
    cosines = torch.linalg.svdvals(left.T @ right).clamp(0., 1.)
    up, learned = up.detach().double(), learned.detach().double()
    gram_up = up.T @ up
    projected = learned @ left
    numerator = ((gram_up @ projected) * projected).sum()
    denominator = ((gram_up @ learned) * learned).sum()
    if denominator <= 0:
        raise AssertionError("historical ordinary adapter has zero linear delta")
    return {"cosines": cosines.tolist(),
            "principal_angles_degrees": torch.rad2deg(torch.acos(cosines)).tolist(),
            "mean_squared_cosine": float(cosines.square().mean()),
            "ordinary_linear_delta_energy_in_initial_span": float(numerator / denominator),
            "initial_down_norm": float(initial.double().norm()),
            "ordinary_down_norm": float(learned.norm()),
            "ordinary_up_norm": float(up.norm())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "outputs/e22-supra-neutral-initialization-6400")
    parser.add_argument("--ordinary", type=Path, default=ROOT.parent /
                        "supra-concept-sliders/outputs/final-boss-supra-converged/checkpoint-06400/final-boss-supra.safetensors")
    parser.add_argument("--toy-root", type=Path, default=ROOT.parent / "ParticleGAN-convergence-toy-develop")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/e22-supra-initial-basis-audit-v1")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("preserve existing artifacts; choose a fresh output directory")
    started = time.monotonic()
    torch.set_num_threads(1)
    rng_before = rng_digest()
    paths = {arm: args.run / arm / "checkpoint-00000.pt"
             for arm in ("sampled_control", "sampled_hb_neutral")}
    paths.update(ordinary=args.ordinary, plan=args.run / "plan.json")
    input_before = {name: sha(path) for name, path in paths.items()}
    plan = json.loads(paths["plan"].read_text())
    if input_before["ordinary"] != plan["input_sha256"]["original6400"]:
        raise AssertionError("ordinary endpoint differs from the declared full-Supra comparison")
    for relative, expected in plan["application_source_sha256"].items():
        if sha(ROOT / relative) != expected:
            raise AssertionError("held application source changed: " + relative)
    pg = Path(plan["protocol"]["particlegan"]["root"])
    for relative, expected in plan["particlegan_source_sha256"].items():
        if sha(pg / relative) != expected:
            raise AssertionError("held native source changed: " + relative)
    saved = {name: torch.load(path, map_location="cpu", mmap=True, weights_only=False)
             for name, path in paths.items() if name.startswith("sampled_")}
    generators = {arm: state["policy"]["models"]["generator"] for arm, state in saved.items()}
    for state in saved.values():
        if state["policy"]["completed_steps"] != 0:
            raise AssertionError("audit requires fresh initial states")
    selected = {arm: {name: value for name, value in values.items()
                     if name.endswith(("down.weight", "up.weight", "bridge.weight", "bridge.bias"))}
                for arm, values in generators.items()}
    state_before = {arm: tensor_digest(values) for arm, values in selected.items()}
    initial = generators["sampled_control"]
    neutral = generators["sampled_hb_neutral"]
    sites = [name.removeprefix("model.").removesuffix(".down.weight")
             for name in initial if name.endswith(".down.weight")]
    if len(sites) != 71:
        raise AssertionError("expected all 71 actual Supra adapter sites")
    report_sites = {}
    with safe_open(args.ordinary, framework="pt", device="cpu") as ordinary:
        expected_keys = {site + suffix for site in sites for suffix in (".down.weight", ".up.weight")}
        if set(ordinary.keys()) != expected_keys:
            raise AssertionError("ordinary and particle projection topology differ")
        for site in sites:
            prefix = "model." + site
            down, up = initial[prefix + ".down.weight"], initial[prefix + ".up.weight"]
            bridge, bias = initial[prefix + ".bridge.weight"], initial[prefix + ".bridge.bias"]
            rank = down.shape[0]
            if rank != 16 or bridge.shape != (16, 20):
                raise AssertionError("unexpected actual rank/bank dimensions")
            for suffix in (".down.weight", ".up.weight"):
                if not torch.equal(initial[prefix + suffix], neutral[prefix + suffix]):
                    raise AssertionError("neutral initialization changed a down/up factor")
            if up.count_nonzero() or neutral[prefix + ".up.weight"].count_nonzero():
                raise AssertionError("both fresh output factors must be zero")
            if not torch.equal(bridge[:, rank:], neutral[prefix + ".bridge.weight"][:, rank:]):
                raise AssertionError("neutral initialization changed C")
            if not bridge[:, rank:].count_nonzero():
                raise AssertionError("particle code path must remain nonzero")
            if (neutral[prefix + ".bridge.weight"][:, :rank].count_nonzero()
                    or neutral[prefix + ".bridge.bias"].count_nonzero()):
                raise AssertionError("neutral H/b are not zero")
            values = compare(down, ordinary.get_tensor(site + ".down.weight"),
                             ordinary.get_tensor(site + ".up.weight"))
            values.update(input_width=int(down.shape[1]), output_width=int(up.shape[0]), rank=rank,
                          initial_H_operator_norm=float(torch.linalg.svdvals(bridge[:, :rank].double())[0]),
                          initial_C_operator_norm=float(torch.linalg.svdvals(bridge[:, rank:].double())[0]),
                          initial_bias_rms=float(bias.double().square().mean().sqrt()))
            report_sites[site] = values
    toy_source = args.toy_root / "examples/e22_routed_convergence.py"
    sys.path.insert(0, str(args.toy_root / "examples"))
    sys.path.insert(1, str(pg))
    import e22_routed_convergence as toy
    data = toy.make_data()
    toy_before = toy.digest(data)
    toy_sites = {site: compare(data["initial_ordinary"][site + ".down.weight"],
                              data["teacher"][site + ".down.weight"], data["teacher"][site + ".up.weight"])
                 for site in ("first", "second")}
    input_after = {name: sha(path) for name, path in paths.items()}
    state_after = {arm: tensor_digest(values) for arm, values in selected.items()}
    rng_after = rng_digest()
    if input_before != input_after or state_before != state_after or rng_before != rng_after:
        raise AssertionError("read-only audit changed source artifacts, selected state, or global RNG")
    if toy_before != toy.digest(data):
        raise AssertionError("audit changed the fresh toy data")
    metrics = ("mean_squared_cosine", "ordinary_linear_delta_energy_in_initial_span",
               "initial_down_norm", "ordinary_down_norm", "ordinary_up_norm",
               "initial_H_operator_norm", "initial_C_operator_norm", "initial_bias_rms")
    report = {"schema": "supra_initial_basis_transfer_audit_v1", "status": "completed",
              "optimizer_updates": 0, "full_supra_model_forward_calls": 0,
              "toy_fixture_forwards_performed": True, "device": "cpu", "threads": 1,
              "scope": "weight-space observation; no full-Supra quality or causal qualification",
              "comparison": "actual fresh sampled particle down span versus historical ordinary LoRA at 6400",
              "limits": ["Historical ordinary down is a learned reference, not the exact frozen-caption target span.",
                         "Projection inputs and their covariance are not measured.",
                         "Different initializers and optimized targets can explain misalignment; this is not a sign bug.",
                         "The linear-delta energy statistic ignores activation weighting and full-model coupling."],
              "inputs": {name: {"path": str(paths[name].resolve()), "sha256": value}
                         for name, value in input_before.items()},
              "sources": {"audit": {"path": str(Path(__file__).resolve()), "sha256": sha(__file__)},
                          "toy": {"path": str(toy_source.resolve()), "sha256": sha(toy_source)},
                          "held_application_sources": plan["application_source_sha256"],
                          "held_native_sources": plan["particlegan_source_sha256"]},
              "immutability": {"input_files_unchanged": True, "selected_state_unchanged": True,
                               "global_cpu_rng_unchanged": True, "cuda_not_initialized_by_audit": True,
                               "selected_initial_state_digests": state_before, "global_cpu_rng_digest": rng_before,
                               "toy_data_digest": toy_before},
              "summary": {metric: summary([value[metric] for value in report_sites.values()]) for metric in metrics},
              "principal_angle_degrees": summary([angle for values in report_sites.values()
                                                    for angle in values["principal_angles_degrees"]]),
              "pr227_aligned_teacher": toy_sites, "sites": report_sites,
              "elapsed_seconds": time.monotonic() - started}
    args.out.mkdir(parents=True)
    (args.out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "report": str(args.out / "report.json"),
                      "summary": report["summary"], "elapsed_seconds": report["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
