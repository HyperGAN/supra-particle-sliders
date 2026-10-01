#!/usr/bin/env python3
"""Read-only native split-direction witness from saved CPU checkpoints.

The cached gradient is compared with Adam's beta1=0 first moment. Native
_delta is evaluated on existing tables with g and .1*g; no controller, model,
optimizer, reservoir, or random stream is advanced.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs/e22-particle-current-geometry-cpu/structural-phase-witness.json")
    parser.add_argument("--particlegan-root", type=Path,
                        default=ROOT / "outputs/e22-convergence-gap/particlegan-cabe2084-source")
    args = parser.parse_args()
    sys.path.insert(0, str(args.particlegan_root.resolve()))
    import torch
    from particlegan.routing import RoutedRowControl
    import particlegan
    if Path(particlegan.__file__).resolve().parent.parent != args.particlegan_root.resolve():
        raise RuntimeError("native source provenance mismatch")
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    points = {}
    for label, run in (
        ("v1_1600", ROOT / "outputs/e22-full-f459cb6d"),
        ("v2_1600", ROOT / "outputs/e22-particle-v2-1600"),
        ("v2_1856", ROOT / "outputs/e22-particle-noise-update-256/latest"),
    ):
        path = run / "final.pt"
        saved = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        state = saved["policy"]
        routing = state["routing"]
        gradient = routing["latest_gradient"]
        group = state["optimizers"][0]["param_groups"][2]
        if len(group["params"]) != 1 or group["betas"][0] != 0:
            raise RuntimeError("the saved optimizer no longer has the audited table role or beta1")
        moment = state["optimizers"][0]["state"][group["params"][0]]["exp_avg"]
        before_gradient = gradient.clone()
        before_fast = state["table"].clone()
        before_average = state["averaged_table"].clone()
        fast = SimpleNamespace(table=state["table"])
        average = SimpleNamespace(table=state["averaged_table"])
        holder = SimpleNamespace(spec=SimpleNamespace(split_scale=routing["config"]["split_scale"]),
                                 latest_gradient=gradient)
        active = (gradient.norm(dim=1) > 0).nonzero().flatten().tolist()
        deltas = torch.stack([RoutedRowControl._delta(holder, row, fast, average) for row in active])
        holder.latest_gradient = gradient * .1
        scaled = torch.stack([RoutedRowControl._delta(holder, row, fast, average) for row in active])
        relative = ((deltas - scaled).norm(dim=1) / deltas.norm(dim=1).clamp_min(1e-30)).max()
        unchanged = (torch.equal(before_gradient, gradient) and torch.equal(before_fast, fast.table)
                     and torch.equal(before_average, average.table))
        if not unchanged or relative > 1e-6 or not torch.equal(gradient, moment):
            raise RuntimeError("native direction provenance/scale/immutability witness failed")
        points[label] = dict(checkpoint=str(path.resolve()), checkpoint_sha256=sha(path),
            completed_steps=state["completed_steps"],
            boundary_is_preservation=state["completed_steps"] % 5 == 0,
            probe_interval=routing["config"]["probe_interval"], preservation_interval=5,
            all_periodic_proposals_on_preservation=routing["config"]["probe_interval"] % 5 == 0,
            beta1=group["betas"][0], cached_gradient_equals_Adam_exp_avg=True,
            cached_gradient_norm=float(gradient.norm()), active_rows=len(active),
            g_versus_point1g_max_relative_delta_difference=float(relative),
            native_split_radius_median=float(deltas.norm(dim=1).median()),
            native_split_radius_max=float(deltas.norm(dim=1).max()), saved_tensors_unchanged=unchanged)
    report = dict(imported_source=str(Path(particlegan.__file__).resolve()), evaluation_only=True,
        native_method="RoutedRowControl._delta", points=points,
        limitations=["This verifies current cached-gradient provenance and scale normalization; it does not reconstruct unrecorded historical proposal gradients.",
                     "Probe-phase coupling follows the fixed 100-update clock and every-fifth preservation schedule. It does not prove a phase-neutral schedule would pass the unchanged zero-harm feature guard."])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
