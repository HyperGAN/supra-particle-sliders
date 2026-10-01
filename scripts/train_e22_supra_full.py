#!/usr/bin/env python3
"""Train every Supra adapter site on the original final-boss prompt task."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import yaml
from supra.runtime import SupraRuntime, MODEL_REV, T5_REV, VAE_REV
from supra.particle_training_data import build_slider_data
from supra.particle_adapter import (
    LINEAR_MODULATED_V2, NONLINEAR_V1, PARTICLE_ARCHITECTURES, validate_particle_architecture,
)
from supra.particle_training import make_training_loop, training_update, raw_velocity
from supra.particle_pilot import checkpoint, restore, frozen_digest, state_digest
from verify_e22_supra import optimizer_provenance

ORIGINAL = Path("/ml2/hypergan/supra-concept-sliders")


def emit(**row):
    print(json.dumps(row, sort_keys=True, default=str), flush=True)


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, default=str) + "\n")
    tmp.replace(path)


def save_checkpoint(loop, path):
    tmp = path.with_suffix(".tmp")
    torch.save(checkpoint(loop), tmp)
    tmp.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


IMMUTABLE_PROVENANCE = (
    "particlegan_commit", "particlegan_root", "particlegan_dirty",
    "particlegan_source_digest", "model_revision", "text_revision", "vae_revision",
    "prompts_sha256", "teacher_cache_sha256", "validation_cache_sha256", "config",
    "output_metrics",
)


def resolve_training_architecture(config=None, requested=None):
    """Fresh runs use V2; resume always retains the checkpoint's exact formula."""
    if requested is not None:
        requested = validate_particle_architecture(requested)
    if config is None:
        return LINEAR_MODULATED_V2 if requested is None else requested
    architecture = validate_particle_architecture(config.get("architecture", NONLINEAR_V1))
    if requested is not None and requested != architecture:
        raise ValueError("resume particle architecture differs from the saved configuration")
    return architecture


def validate_resume(previous, provenance, state):
    """A longer external budget must not change the game or its sampled stream."""
    # JSON receipts represent native recipe tuples as arrays. Strict native
    # configuration equality is additionally enforced by restore(loop, state).
    canonical = lambda value: json.loads(json.dumps(value))
    changed = [key for key in IMMUTABLE_PROVENANCE
               if canonical(previous.get(key)) != canonical(provenance.get(key))]
    if changed:
        raise ValueError(f"resume provenance changed: {', '.join(changed)}")
    if canonical(state["config"]) != canonical(provenance["config"]):
        raise ValueError("resume checkpoint configuration changed")
    completed = int(state["policy"]["completed_steps"])
    if not 0 < completed < provenance["fixed_updates"]:
        raise ValueError("resume step must precede the new fixed update horizon")
    return completed


def archive_training_sources(output):
    paths = [Path(__file__), ROOT / "scripts/verify_e22_supra.py"]
    paths += [ROOT / "supra" / name for name in (
        "particle_adapter.py", "particle_game.py", "particle_pilot.py",
        "particle_training.py", "particle_training_data.py", "runtime.py",
    )]
    hashes = {}
    for source in paths:
        relative = source.relative_to(ROOT)
        target = output / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[str(relative)] = sha(target)
    write_json(output / "source/sha256.json", hashes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/e22-full-f459cb6d"))
    parser.add_argument("--steps", type=int, default=1600)
    parser.add_argument("--probe-interval", type=int, default=100)
    parser.add_argument("--branch-lr", type=float, default=5e-5)
    parser.add_argument("--architecture", choices=PARTICLE_ARCHITECTURES,
                        help="fresh default: linear_modulated_v2; resume: retain the saved architecture")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--baseline", type=Path,
                        help="original LoRA adapter at the same declared update horizon")
    parser.add_argument("--checkpoint-every", type=int, default=400)
    args = parser.parse_args()
    if args.steps < 4:
        parser.error("the exact-resume preflight needs at least four updates")
    if args.checkpoint_every < 1:
        parser.error("--checkpoint-every must be positive")
    previous = None
    if args.resume:
        source_run = args.resume.resolve().parent
        previous = json.loads((source_run / "run.json").read_text())
    architecture = resolve_training_architecture(None if previous is None else previous["config"],
                                                 args.architecture)
    args.output.mkdir(parents=True, exist_ok=True)
    existing_run = args.output / "run.json"
    if existing_run.exists():
        if args.resume is None:
            parser.error("output already contains a run; choose a new output or resume")
        if json.loads(existing_run.read_text())["fixed_updates"] != args.steps:
            parser.error("use a separate output directory when extending a run's update horizon")
    baseline_path = args.baseline or (
        ORIGINAL / "outputs/final-boss-supra-1600/final-boss-supra.safetensors"
        if args.steps == 1600 else ORIGINAL /
        f"outputs/final-boss-supra-converged/checkpoint-{args.steps:05d}/final-boss-supra.safetensors"
    )
    if not baseline_path.is_file():
        parser.error(f"matched original baseline missing: {baseline_path}")
    torch.set_num_threads(8)
    started = time.perf_counter()
    status_path = args.output / "status.json"
    write_json(status_path, dict(phase="preparing", step=0, steps=args.steps))
    config_path = ORIGINAL / "configs/supra/prompts-final-boss.yaml"
    train_path = ORIGINAL / "outputs/final-boss-supra-1600/teacher_cache.pt"
    validation_path = ORIGINAL / "outputs/final-boss-supra-converged/validation_cache.pt"
    config = yaml.safe_load(config_path.read_text())
    runtime = SupraRuntime(device="cuda:0", rank=16, allow_hub=False)
    # No published trained adapter is loaded into either student or teacher.
    runtime.model.requires_grad_(False)
    if args.resume:
        source_data = source_run / "data.pt"
        data = torch.load(source_data, map_location="cpu", weights_only=False)
        if (args.output / "data.pt").exists():
            if sha(args.output / "data.pt") != sha(source_data):
                raise ValueError("resume data changed")
        elif args.output.resolve() != source_run:
            shutil.copy2(source_data, args.output / "data.pt")
    else:
        all_prompts = {""}
        for row in config["rows"]:
            all_prompts.update((row["neutral"], row["positive"]))
        all_prompts.update(config["preservation"])
        for row in config["verification"]:
            all_prompts.add(row["prompt"])
            if "teacher" in row:
                all_prompts.add(row["teacher"])
        prompts = [""] + sorted(all_prompts - {""})
        for prompt in prompts:
            runtime.encode(prompt)
        text = torch.cat([runtime.cache[p][0].cpu() for p in prompts])
        masks = torch.cat([runtime.cache[p][1].cpu() for p in prompts])
        train_cache = torch.load(train_path, map_location="cpu", weights_only=True)
        validation_cache = torch.load(validation_path, map_location="cpu", weights_only=True)
        data = build_slider_data(train_cache, validation_cache, config, text, masks, prompts)
        torch.save(data, args.output / "data.pt")
    runtime.text_encoder.to("cpu")
    loop = make_training_loop(runtime.model, data, probe_interval=args.probe_interval,
                              branch_lr=args.branch_lr, architecture=architecture)
    provenance = dict(**optimizer_provenance(), model_revision=MODEL_REV, text_revision=T5_REV,
                      vae_revision=VAE_REV, prompts_sha256=sha(config_path),
                      teacher_cache_sha256=sha(train_path), validation_cache_sha256=sha(validation_path),
                      original_baseline_sha256=sha(baseline_path),
                      original_baseline_path=str(baseline_path.resolve()),
                      config=loop.config, fixed_updates=args.steps,
                      comparison=f"full native ParticleGAN formulation versus original {args.steps}-update LoRA recipe",
                      output_metrics="evaluation only; never affect optimization, guards or checkpoint selection")
    resumed_step = 0
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False, mmap=True)
        resumed_step = validate_resume(previous, provenance, state)
        if existing_run.exists():
            declared = json.loads(existing_run.read_text())
            for key in ("original_baseline_sha256", "original_baseline_path"):
                if declared[key] != provenance[key]:
                    raise ValueError(f"resume comparison changed: {key}")
            provenance = declared
        else:
            provenance["resumed_from"] = dict(
                checkpoint=str(args.resume.resolve()), checkpoint_sha256=sha(args.resume),
                completed_steps=resumed_step, run_sha256=sha(source_run / "run.json"),
                data_sha256=sha(source_data), trace=str(source_run / "train.jsonl"),
            )
        expected_state = state_digest(state)
        restore(loop, state)
        if loop.policy.completed_steps != resumed_step:
            raise RuntimeError("native restore changed the completed step")
        if state_digest(checkpoint(loop)) != expected_state:
            raise RuntimeError("native continuation did not restore the complete state exactly")
        del state
        emit(event="resumed", step=resumed_step, fixed_updates=args.steps,
             full_state_exact=True,
             sampling="restored CPU data RNG and native/private paired-noise RNG states")
    if not existing_run.exists():
        write_json(args.output / "run.json", provenance)
        archive_training_sources(args.output)
    emit(event="ready", provenance=provenance, dataset=data["metadata"])
    frozen = frozen_digest(loop)
    trace_path = args.output / "train.jsonl"
    training_started = time.perf_counter()
    with trace_path.open("a") as trace:
        if not args.resume:
            served = loop.policy.served_model()
            probe = data["fit"]["context"][:4].to(loop.policy.device).clone()
            probe[:,4099] = 0
            native = runtime.velocity(data["fit"]["prompts"][:4], probe[:,:4096].reshape(-1,4,32,32),
                                      probe[:,4096], scale=0, cfg=3)
            zero = raw_velocity(served, probe)
            if not torch.equal(zero, native):
                raise RuntimeError("initial zero-strength native forward mismatch")
            del served, probe, native, zero
            initial_rows = [training_update(loop), training_update(loop)]
            if initial_rows[-1]["dense_gradient_rows"] != 128:
                raise RuntimeError("shared bank did not receive dense gradients by update two")
            smoke_path = args.output / "smoke-resume.pt"
            save_checkpoint(loop, smoke_path)
            reference = [training_update(loop), training_update(loop)]
            expected = state_digest(checkpoint(loop))
            restore(loop, torch.load(smoke_path, map_location="cuda:0", weights_only=False))
            replay = [training_update(loop), training_update(loop)]
            exact_resume = expected == state_digest(checkpoint(loop)) and reference == replay
            if not exact_resume or frozen != frozen_digest(loop):
                raise RuntimeError("full-model exact resume or frozen weights failed")
            branch_gradient_counts = {branch.site: sum(p.grad is not None and bool(p.grad.abs().sum()>0)
                                                     for p in branch.parameters() if p.requires_grad)
                                      for branch in loop.policy.G.particle_branches()}
            if any(count == 0 for count in branch_gradient_counts.values()):
                raise RuntimeError("one or more full adapter sites has no gradient")
            smoke = dict(exact_resume=exact_resume, zero_strength_exact=True, frozen_unchanged=True,
                         sites_with_gradients=len(branch_gradient_counts),
                         dense_gradient_rows=initial_rows[-1]["dense_gradient_rows"])
            write_json(args.output / "smoke.json", smoke)
            emit(event="smoke", **smoke)
            for row in initial_rows + reference:
                trace.write(json.dumps(row, default=str)+"\n")
            trace.flush()
        for _ in range(loop.policy.completed_steps, args.steps):
            row = training_update(loop)
            trace.write(json.dumps(row, default=str)+"\n")
            if row["step"] % 25 == 0 or row["step"] == args.steps:
                trace.flush()
                torch.cuda.synchronize()
                elapsed = time.perf_counter()-training_started
                write_json(status_path, dict(phase="training", step=row["step"], steps=args.steps,
                                             seconds=elapsed, final_checkpoint="final.pt"))
                emit(event="train", seconds=elapsed, **row)
            if row["step"] % args.checkpoint_every == 0:
                save_checkpoint(loop, args.output / f"checkpoint-{row['step']:05d}.pt")
        torch.cuda.synchronize()
        training_seconds = time.perf_counter()-training_started
    save_checkpoint(loop, args.output / "final.pt")
    if frozen != frozen_digest(loop):
        raise RuntimeError("frozen base/teacher weights changed")
    summary = dict(phase="trained", step=loop.policy.completed_steps, steps=args.steps,
                   resumed_step=resumed_step, updates_this_process=loop.policy.completed_steps-resumed_step,
                   training_seconds=training_seconds, total_seconds=time.perf_counter()-started,
                   final_checkpoint=str(args.output / "final.pt"), frozen_unchanged=True,
                   r1=None if loop.policy.surprise is None else loop.policy.surprise.diagnostics(),
                   ladder_reopens=loop.policy.controller.reopens if hasattr(loop.policy.controller,"reopens") else None,
                   checkpoint_selection="final fixed update horizon")
    write_json(status_path, summary)
    emit(event="complete", **summary)


if __name__ == "__main__":
    main()
