#!/usr/bin/env python3
"""Replay all existing original-LoRA snapshots on the live fixed-game probes.

This is a read-only curve of historical checkpoints, scored by the exact
frozen step-1856 critic, training probe indices, CPU-generator-72 Gaussian
panels, native batch-four BF16 and CFG3 used by the current progress monitor.
No output-MSE optimization, stopping or checkpoint selection is performed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
PIN = "cabe2084284db923d525918cbf3e18de6f20faac"
FINAL_ORIGINAL_SHA = "24a98ba055d0b5298136d67dc7491a339684da03416c510d1d443fe730eb7069"
ORIGINAL_ROOT = Path("/ml2/hypergan/supra-concept-sliders/outputs/final-boss-supra-converged")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    temporary = Path(path).with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "outputs/e22-particle-noise-update-256/latest")
    parser.add_argument("--live-run", type=Path, default=ROOT / "outputs/e22-particle-v2-6400")
    parser.add_argument("--original-root", type=Path, default=ORIGINAL_ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/e22-dashboard-baselines")
    parser.add_argument("--particlegan-root", type=Path,
                        default=ROOT / "outputs/e22-convergence-gap/particlegan-cabe2084-source")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("choose a fresh output directory")
    sys.path.insert(0, str(args.particlegan_root.resolve()))
    sys.path.insert(1, str(ROOT))
    import torch
    import torch.nn.functional as F
    import particlegan
    from safetensors import safe_open
    from safetensors.torch import load_file
    from supra.runtime import MODEL_ID, MODEL_REV, T5_REV, VAE_REV, TARGETS, adapter_state, load_adapter_state, model_module
    from supra.particle_game import ConditionalTokenCritic, patchify
    from supra.particle_pilot import state_digest
    from supra.particle_training_data import FrozenSliderContexts
    from monitor_e22_supra_particle_convergence import GameProgressMonitor

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("native BF16 evaluation requires CUDA")
    torch.cuda.set_device(device)
    torch.set_num_threads(8)
    started = time.perf_counter()
    source_dir = Path(particlegan.__file__).resolve().parent
    run = json.loads((args.run / "run.json").read_text())
    pg_digest = state_digest({path.name: sha(path) for path in sorted(source_dir.glob("*.py"))})
    if (source_dir.parent != args.particlegan_root.resolve() or run["particlegan_commit"] != PIN
            or run["particlegan_source_digest"] != pg_digest):
        raise RuntimeError("frozen critic and archived ParticleGAN source disagree")
    if (run["model_revision"], run["text_revision"], run["vae_revision"]) != (MODEL_REV, T5_REV, VAE_REV):
        raise RuntimeError("model/text/VAE pins differ from the live task")
    checkpoints = []
    for path in sorted(args.original_root.glob("checkpoint-*/final-boss-supra.safetensors")):
        step = int(path.parent.name.split("-")[-1])
        if 0 < step <= 6400:
            checkpoints.append((step, path))
    if [step for step, _ in checkpoints] != list(range(400, 6401, 400)):
        raise RuntimeError("historical curve must contain every fixed 400-step snapshot through 6400")
    inputs = {"critic_checkpoint": args.run / "final.pt", "data": args.run / "data.pt", "run": args.run / "run.json"}
    inputs.update({f"original_{step}": path for step, path in checkpoints})
    for name in ("edit", "preservation"):
        inputs[f"live_{name}_indices"] = args.live_run / f"progress-{name}-indices.json"
    input_hashes = {name: sha(path) for name, path in inputs.items()}
    if input_hashes["original_6400"] != FINAL_ORIGINAL_SHA:
        raise RuntimeError("original fixed-6400 benchmark checkpoint changed")
    source_paths = [Path(__file__), ROOT / "scripts/monitor_e22_supra_particle_convergence.py"]
    source_paths += [ROOT / "supra" / name for name in
                     ("runtime.py", "particle_game.py", "particle_pilot.py", "particle_training_data.py")]
    source_hashes = {str(path.relative_to(ROOT)): sha(path) for path in source_paths}
    args.output.mkdir(parents=True)
    for path in source_paths:
        destination = args.output / "source" / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    write_json(args.output / "source/sha256.json", source_hashes)
    data = torch.load(inputs["data"], map_location="cpu", weights_only=False)
    saved = torch.load(inputs["critic_checkpoint"], map_location="cpu", weights_only=False, mmap=True)
    if saved["policy"]["completed_steps"] != 1856 or saved["config"]["dataset_digest"] != state_digest(data):
        raise RuntimeError("the frozen judge is not the owned step-1856 training state")
    encoded = saved["policy"]["models"]["encoder"]
    if not torch.equal(encoded["contexts"], data["text_contexts"]) or not torch.equal(encoded["masks"], data["text_masks"]):
        raise RuntimeError("teacher text buffers differ from cached task")
    teacher_state = {key.removeprefix("teacher."): value for key, value in encoded.items() if key.startswith("teacher.")}
    critic_state = saved["policy"]["models"]["critic"]
    if not torch.equal(critic_state["scale"], data["coordinate_scale"]):
        raise RuntimeError("critic coordinate units differ from task")
    with torch.random.fork_rng(devices=[device.index or 0]):
        module = model_module()
        with torch.device("meta"):
            original = module.SupraDiT()
            module.attach_supra_lora(original, rank=16, alpha=16, targets=TARGETS)
        original.load_state_dict(teacher_state, strict=True, assign=True)
        original.to(device).eval().requires_grad_(False)
        encoder = FrozenSliderContexts(data["text_contexts"], data["text_masks"], original).to(device)
        critic = ConditionalTokenCritic(critic_state["scale"]).to(device)
        critic.load_state_dict(critic_state, strict=True)
        critic.eval().requires_grad_(False)
    # Reuse the monitor constructor itself, so context ordering and both private
    # CPU Gaussian panels are identical. This private DV12 RNG is unused by LoRA.
    stub = SimpleNamespace(policy=SimpleNamespace(device=device, noise_generator=torch.Generator(device=device)))
    probes = GameProgressMonitor(stub, data, critic, args.output)
    for name in probes.pools:
        generated = json.loads((args.output / f"progress-{name}-indices.json").read_text())
        live = json.loads(inputs[f"live_{name}_indices"].read_text())
        if generated != live:
            raise RuntimeError(f"{name} probe indices differ from live monitor")
    expected_adapter = adapter_state(original)
    frozen = {key: value for key, value in original.state_dict().items() if key not in expected_adapter}
    before = dict(frozen=state_digest(frozen), teacher=state_digest(encoder.state_dict()),
                  critic=state_digest(critic.state_dict()), data=state_digest(data),
                  cpu_rng=torch.get_rng_state().clone(), cuda_rng=torch.cuda.get_rng_state(device).clone())
    ordinary = [branch for branch in original.modules() if hasattr(branch, "multiplier")]
    multipliers = [branch.multiplier for branch in ordinary]
    batches = {}
    with torch.no_grad():
        for name, (contexts, noise) in probes.pools.items():
            batches[name] = []
            for start in range(0, len(contexts), 4):
                context = contexts[start:start + 4]
                condition = encoder.condition(context).repeat(4, 1)
                real = (.125 * noise[:, start:start + len(context)]).flatten(0, 1)
                batches[name].append((context, encoder.teacher_velocity(context), condition, real, critic(real, condition)))
    result = dict(complete=False, label="Original formulation benchmark to beat", evaluation_only=True,
                  output_metrics_used=False, optimization=False,
                  reference=dict(critic_step=1856, critic_checkpoint_sha256=input_hashes["critic_checkpoint"],
                                 data_sha256=input_hashes["data"], particlegan_commit=PIN,
                                 particlegan_source_digest=pg_digest, output_sigma=.125, noise_generator="private CPU72",
                                 gaussian_panels=4, native_batch=4, cfg=3., precision="native BF16 host; FP32 critic/residual",
                                 probe_contexts={name: len(contexts) for name, (contexts, _) in probes.pools.items()},
                                 noise_panel_digest={name: state_digest(noise) for name, (_, noise) in probes.pools.items()},
                                 probe_indices={name: json.loads((args.output / f"progress-{name}-indices.json").read_text())
                                                for name in probes.pools}),
                  checkpoint_selection="all historical fixed 400-step snapshots through6400, independent of outcomes",
                  limitation="Original LoRA uses AdamW/MSE; live model uses particles and native game. This curve compares complete formulations, not isolated optimizer changes.",
                  input_sha256=input_hashes, source_sha256=source_hashes,
                  curves=[dict(label="Original LoRA (benchmark to beat)", architecture="ordinary rank16 LoRA", formulation="AdamW with paired velocity MSE and preservation", points=[])])
    output_path = args.output / "fixed-reference-baselines.json"
    write_json(output_path, result)
    with torch.no_grad():
        for step, path in checkpoints:
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                metadata = {key: json.loads(value) for key, value in (handle.metadata() or {}).items()}
            expected_metadata = dict(format="supra-native-lora-v1", rank=16, alpha=16, targets=list(TARGETS),
                                     model_id=MODEL_ID, model_revision=MODEL_REV, text_encoder_revision=T5_REV,
                                     vae_revision=VAE_REV, step=step, prompts_sha256=run["prompts_sha256"])
            if any(metadata.get(key) != value for key, value in expected_metadata.items()):
                raise RuntimeError(f"historical checkpoint {step} metadata differs from live task")
            weights = load_file(str(path), device="cpu")
            if set(weights) != set(expected_adapter) or len(weights) != 142:
                raise RuntimeError("original LoRA must contain all 71 projection sites")
            for key, value in weights.items():
                if value.shape != expected_adapter[key].shape or value.dtype != expected_adapter[key].dtype or not torch.isfinite(value).all():
                    raise RuntimeError(f"invalid historical adapter tensor {key}")
            load_adapter_state(original, weights)
            point = dict(step=step, checkpoint=str(path.resolve()), checkpoint_sha256=input_hashes[f"original_{step}"], probes={})
            for name, fixed_batches in batches.items():
                values = []
                for context, target, condition, real, real_score in fixed_batches:
                    z, t, ctx, mask, uctx, umask, strength = encoder.unpack(context)
                    batch = len(z)
                    try:
                        for branch in ordinary:
                            branch.multiplier = strength
                        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                            output = original(torch.cat((z, z)), torch.cat((t, t)),
                                              torch.cat((ctx, uctx.expand(batch, -1, -1))),
                                              torch.cat((mask, umask.expand(batch, -1))))
                        conditional, unconditional = output.float().chunk(2)
                        residual = unconditional + 3 * (conditional - unconditional) - target
                    finally:
                        for branch, multiplier in zip(ordinary, multipliers):
                            branch.multiplier = multiplier
                    fake = real + (patchify(residual).float() / critic.scale).repeat(4, 1, 1)
                    values.extend(F.softplus(real_score - critic(fake, condition)).reshape(4, batch).mean(0).cpu().tolist())
                score = sum(values) / len(values)
                if not torch.isfinite(torch.tensor(score)):
                    raise RuntimeError("historical game probe is nonfinite")
                point["probes"][name] = dict(contexts=len(values), clean=dict(frozen_start_D=score))
            result["curves"][0]["points"].append(point)
            write_json(output_path, result)
            print(json.dumps(dict(event="reference_checkpoint", **point)), flush=True)
    unchanged = (before["frozen"] == state_digest({key: value for key, value in original.state_dict().items() if key not in expected_adapter})
                 and before["teacher"] == state_digest(encoder.state_dict()) and before["critic"] == state_digest(critic.state_dict())
                 and before["data"] == state_digest(data) and torch.equal(before["cpu_rng"], torch.get_rng_state())
                 and torch.equal(before["cuda_rng"], torch.cuda.get_rng_state(device))
                 and multipliers == [branch.multiplier for branch in ordinary])
    if not unchanged:
        raise RuntimeError("historical evaluation changed frozen weights/data/global RNG state")
    if any(sha(path) != input_hashes[name] for name, path in inputs.items()):
        raise RuntimeError("historical evaluation input changed")
    if any(sha(ROOT / name) != value for name, value in source_hashes.items()):
        raise RuntimeError("historical evaluation source changed")
    result.update(complete=True, evaluation_state_unchanged=True, input_files_unchanged=True,
                  seconds=time.perf_counter() - started)
    write_json(output_path, result)
    print(json.dumps(dict(event="reference_curves_complete", points=len(checkpoints), seconds=result["seconds"])), flush=True)


if __name__ == "__main__":
    main()
