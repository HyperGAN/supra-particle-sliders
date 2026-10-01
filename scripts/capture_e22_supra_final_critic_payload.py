#!/usr/bin/env python3
"""Read-only native 6400 clean/DV12 residual capture for critic diagnostics.

The next edit and preservation minibatches are reconstructed from a private
copy of the saved data stream. Four test contexts are diagnostic only. Neither
the policy lifecycle nor any training RNG/controller/optimizer is advanced.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def emit(**row):
    print(json.dumps(row, default=str), flush=True)


def move(value, device):
    import torch
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move(item, device) for item in value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "outputs/e22-particle-v2-6400")
    parser.add_argument("--particlegan-root", type=Path,
                        default=ROOT / "outputs/e22-convergence-gap/particlegan-cabe2084-source")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs/e22-particle-final-critic-causal/residuals.pt")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("choose a fresh diagnostic payload")
    sys.path.insert(0, str(args.particlegan_root.resolve()))
    sys.path.insert(1, str(ROOT))
    import torch
    import particlegan
    from particlegan.continuous import DataDriftController
    from supra.particle_export import load_particle_adapter, _EncodedCondition
    from supra.particle_pilot import state_digest
    from supra.particle_training_data import FrozenSliderContexts
    from supra.runtime import model_module, TARGETS
    torch.set_num_threads(4)
    device = torch.device(args.device)
    if Path(particlegan.__file__).resolve().parent.parent != args.particlegan_root.resolve():
        raise RuntimeError("ParticleGAN import differs from the declared frozen source")
    declared = json.loads((args.run / "run.json").read_text())
    source_digest = state_digest({p.name: sha(p) for p in sorted(Path(particlegan.__file__).parent.glob("*.py"))})
    if source_digest != declared["particlegan_source_digest"]:
        raise RuntimeError("ParticleGAN source differs from the training source")
    saved = torch.load(args.run / "final.pt", map_location="cpu", weights_only=False, mmap=True)
    data = torch.load(args.run / "data.pt", map_location="cpu", weights_only=False)
    step = saved["policy"]["completed_steps"]
    if step != 6400 or saved["config"].get("architecture") != "linear_modulated_v2":
        raise RuntimeError("this diagnostic requires the qualified V2 step6400 state")
    if state_digest(data) != saved["config"]["dataset_digest"]:
        raise RuntimeError("cached task data differs from the checkpoint")
    checkpoint_sha = sha(args.run / "final.pt")
    export_sha = sha(args.run / "final.safetensors")
    data_sha = sha(args.run / "data.pt")
    before_cpu = torch.get_rng_state().clone()
    before_cuda = torch.cuda.get_rng_state(device).clone()
    stream = torch.Generator()
    stream.set_state(saved["data_rng"].cpu())
    selected, selection = {}, {}
    for upcoming in range(step + 1, step + 6):
        name = "holds" if upcoming % 5 == 0 else "fit"
        indices = torch.randint(len(data[name]["context"]), (4,), generator=stream)
        if name not in selected:
            selected[name] = data[name]["context"][indices].to(device)
            selection[name] = dict(proposed_step=upcoming, indices=indices.tolist())
    selected["test"] = data["test"]["context"][:4].to(device)
    selection["test"] = dict(indices=list(range(4)))
    began = time.perf_counter()
    module = model_module()
    with torch.random.fork_rng(devices=[]), torch.device("meta"):
        base = module.SupraDiT()
        module.attach_supra_lora(base, rank=16, alpha=16, targets=TARGETS)
    base.load_state_dict({key.removeprefix("teacher."): value
                         for key, value in saved["policy"]["models"]["encoder"].items()
                         if key.startswith("teacher.")}, strict=True, assign=True)
    base.to(device).eval().requires_grad_(False)
    with torch.random.fork_rng(devices=[device.index or 0]):
        encoder = FrozenSliderContexts(data["text_contexts"], data["text_masks"], base).to(device)
        adapter = load_particle_adapter(base, args.run / "final.safetensors", device=device)
    before_adapter = state_digest(adapter.state_dict())
    before_encoder = state_digest(encoder.state_dict())
    payload = dict(metadata=dict(
        checkpoint=str((args.run / "final.pt").resolve()), checkpoint_sha256=checkpoint_sha,
        checkpoint_step=step, dataset_digest=saved["config"]["dataset_digest"],
        source_commit=declared["particlegan_commit"], particlegan_source_digest=source_digest,
        script_sha256=sha(__file__), export_sha256=export_sha,
        precision="native CUDA BF16 host; FP32 particle routing/modulation",
        selection=selection, evaluation_only=True, noise_draws=4,
        noise_semantics="Four private DV12 draws per pool from the saved boundary; no lifecycle advance."), groups={})
    with torch.no_grad():
        for name, context in selected.items():
            inputs = encoder.unpack(context)
            teacher = encoder.teacher_velocity(context)
            clean = adapter.velocity(*inputs[:6], strength=inputs[6], cfg=3.)
            controller = DataDriftController("dv12")
            controller.load_state_dict(move(saved["policy"]["controller"], device))
            controller_before = state_digest(controller.state_dict())
            candidate = adapter.routing.candidate_for(dict(router=adapter.router), adapter.table,
                averaged=adapter.metadata["served_source"] == "averaged")
            prior = controller.routed_prior(candidate.table, candidate.log_mass)
            private_stream = torch.Generator(device=device)
            private_stream.set_state(saved["policy"]["streams"]["noise_generator"].cpu())
            condition = _EncodedCondition(*[inputs[i] for i in (0, 2, 3, 4, 5)], inputs[6], 3.)
            packed = torch.cat((inputs[0].flatten(1), inputs[1][:, None]), 1)
            models = dict(generator=adapter.generator, router=adapter.router, conditioning=condition)
            noisy = []
            for _ in range(4):
                value = adapter.routing.forward(models, packed, candidate, perturb_fn=lambda codes:
                    controller.perturb_latent(codes, private_stream, prior, record=False))
                noisy.append((value - teacher).cpu())
            if controller_before != state_digest(controller.state_dict()):
                raise RuntimeError("private replay changed native controller diagnostics")
            payload["groups"][name] = dict(context=context.cpu(), condition=encoder.condition(context).cpu(),
                new=(clean - teacher).cpu(), noisy=torch.stack(noisy),
                output_sigma=float(saved["policy"]["last_output_sigma"]))
            emit(event="captured", pool=name, contexts=len(context), draws=4)
    if before_adapter != state_digest(adapter.state_dict()) or before_encoder != state_digest(encoder.state_dict()):
        raise RuntimeError("read-only capture changed a model")
    if not torch.equal(before_cpu, torch.get_rng_state()) or not torch.equal(before_cuda, torch.cuda.get_rng_state(device)):
        raise RuntimeError("read-only capture changed a global RNG")
    for path, expected in ((args.run / "final.pt", checkpoint_sha),
                           (args.run / "final.safetensors", export_sha), (args.run / "data.pt", data_sha)):
        if sha(path) != expected:
            raise RuntimeError("capture input file changed")
    payload["metadata"].update(frozen_models_and_global_rng_unchanged=True, seconds=time.perf_counter() - began)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    emit(event="complete", output=args.output, step=step, seconds=payload["metadata"]["seconds"])


if __name__ == "__main__":
    main()
