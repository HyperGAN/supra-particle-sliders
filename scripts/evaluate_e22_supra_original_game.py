#!/usr/bin/env python3
"""Evaluate the original step-1600 Supra LoRA under both particle-game critics.

This is a read-only evaluation of the existing ordinary-LoRA checkpoint. It
uses the V1/V2 experiment's complete pools, batch-four BF16/CFG3 forwards and
the same private noise panels. No optimizer, structural decision, stopping
rule or checkpoint selection is introduced. The original recipe differs in
both architecture and its training objective, so this comparison cannot
attribute a difference to the ParticleGAN optimizer alone.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
PIN = "f459cb6d6aaaabeb1af076ec53ad7a963618de90"
BASELINE_SHA = "134f5a9d12206b74f39a2c3f9c90fdb38477a1c3204b5deea5152b1cbe49a47f"
BASELINE = Path("/ml2/hypergan/supra-concept-sliders/outputs/final-boss-supra-1600/final-boss-supra.safetensors")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "outputs/e22-particle-v2-1600")
    parser.add_argument("--reference", type=Path, default=ROOT / "outputs/e22-full-f459cb6d")
    parser.add_argument("--particlegan-root", type=Path,
                        default=ROOT / "outputs/e22-convergence-gap/particlegan-f459-source")
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.output = args.output or args.run / "evaluation-original-game"
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("choose a fresh evaluation output directory")
    sys.path.insert(0, str(args.particlegan_root.resolve()))
    sys.path.insert(1, str(ROOT))

    import torch
    import particlegan
    from safetensors import safe_open
    from safetensors.torch import load_file
    from supra.runtime import (MODEL_ID, MODEL_REV, T5_REV, VAE_REV, TARGETS,
                               adapter_state, load_adapter_state, model_module)
    from supra.particle_adapter import LINEAR_MODULATED_V2, NONLINEAR_V1
    from supra.particle_game import ConditionalTokenCritic
    from supra.particle_pilot import state_digest
    from supra.particle_training_data import FrozenSliderContexts
    from diagnose_e22_supra_training_controls import evaluate_final

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("matched native Supra/BF16 evaluation requires CUDA")
    torch.set_num_threads(8)
    torch.cuda.set_device(device)
    started = time.perf_counter()
    imported = Path(particlegan.__file__).resolve()
    old_run = json.loads((args.reference / "run.json").read_text())
    new_run = json.loads((args.run / "run.json").read_text())
    experiment = json.loads((args.run / "receipt.json").read_text())
    if imported.parent.parent != args.particlegan_root.resolve():
        raise RuntimeError("ParticleGAN import differs from the declared archived source")
    pg_hashes = {path.name: sha(path) for path in sorted(imported.parent.glob("*.py"))}
    for run in (old_run, new_run):
        if run["particlegan_commit"] != PIN or run["particlegan_source_digest"] != state_digest(pg_hashes):
            raise RuntimeError("ParticleGAN source differs from the qualified particle experiment")
        if run["fixed_updates"] != 1600 or run["original_baseline_sha256"] != BASELINE_SHA:
            raise RuntimeError("reference is not the matched original step-1600 task")
        if (run["model_revision"], run["text_revision"], run["vae_revision"]) != (MODEL_REV, T5_REV, VAE_REV):
            raise RuntimeError("native model/text/VAE pins differ from this runtime")
    for field in ("full_initial_native_state_identical", "legacy_step_two_replay_exact",
                  "native_resume_exact", "frozen_unchanged", "export_reload_exact",
                  "evaluation_state_unchanged", "data_and_paired_streams_matched"):
        if experiment.get(field) is not True:
            raise RuntimeError(f"particle architecture experiment lacks qualification: {field}")
    if old_run["config"].get("architecture", NONLINEAR_V1) != NONLINEAR_V1:
        raise RuntimeError("the reference critic must come from legacy V1")
    if new_run["config"].get("architecture") != LINEAR_MODULATED_V2:
        raise RuntimeError("the new critic must come from V2")

    inputs = dict(original_adapter=args.baseline, v1_final=args.reference / "final.pt",
                  v2_final=args.run / "final.pt", data=args.run / "data.pt",
                  v1_data=args.reference / "data.pt", v1_run=args.reference / "run.json",
                  v2_run=args.run / "run.json", qualification=args.run / "receipt.json")
    input_hashes = {label: sha(path) for label, path in inputs.items()}
    if input_hashes["original_adapter"] != BASELINE_SHA:
        raise RuntimeError("original published-example adapter differs from the declared baseline")
    if input_hashes["data"] != input_hashes["v1_data"]:
        raise RuntimeError("V1 and V2 evaluation pools differ")
    if experiment["plan"]["reference_checkpoint_sha256"] != input_hashes["v1_final"]:
        raise RuntimeError("V1 critic checkpoint differs from the qualified comparator")

    source_paths = [Path(__file__), ROOT / "scripts/diagnose_e22_supra_training_controls.py"]
    source_paths += [ROOT / "supra" / name for name in (
        "runtime.py", "particle_training_data.py", "particle_game.py", "particle_pilot.py")]
    source_hashes = {str(path.relative_to(ROOT)): sha(path) for path in source_paths}
    args.output.mkdir(parents=True, exist_ok=True)
    for source in source_paths:
        target = args.output / "source" / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    write_json(args.output / "source/sha256.json", source_hashes)

    data = torch.load(inputs["data"], map_location="cpu", weights_only=False)
    v1 = torch.load(inputs["v1_final"], map_location="cpu", weights_only=False, mmap=True)
    v2 = torch.load(inputs["v2_final"], map_location="cpu", weights_only=False, mmap=True)
    for saved, run in ((v1, old_run), (v2, new_run)):
        if saved["policy"]["completed_steps"] != 1600 or json.loads(json.dumps(saved["config"])) != run["config"]:
            raise RuntimeError("checkpoint horizon/config differs from its run receipt")
        if saved["config"]["dataset_digest"] != state_digest(data):
            raise RuntimeError("checkpoint does not own the declared evaluation data")
        encoded = saved["policy"]["models"]["encoder"]
        if not torch.equal(encoded["contexts"], data["text_contexts"]) or not torch.equal(encoded["masks"], data["text_masks"]):
            raise RuntimeError("checkpoint text buffers differ from the matched evaluation data")
    if state_digest(v2) != experiment["final_state_digest"]:
        raise RuntimeError("V2 final native state differs from the qualified experiment")
    teacher_state = {key.removeprefix("teacher."): value for key, value in v2["policy"]["models"]["encoder"].items()
                     if key.startswith("teacher.")}
    old_teacher = {key.removeprefix("teacher."): value for key, value in v1["policy"]["models"]["encoder"].items()
                   if key.startswith("teacher.")}
    if state_digest(teacher_state) != state_digest(old_teacher):
        raise RuntimeError("V1 and V2 frozen teachers differ")

    with safe_open(str(args.baseline), framework="pt", device="cpu") as handle:
        metadata = {key: json.loads(value) for key, value in (handle.metadata() or {}).items()}
    expected_metadata = dict(format="supra-native-lora-v1", rank=16, alpha=16,
                             targets=list(TARGETS), model_id=MODEL_ID, model_revision=MODEL_REV,
                             text_encoder_revision=T5_REV, vae_revision=VAE_REV, step=1600,
                             prompts_sha256=old_run["prompts_sha256"])
    if any(metadata.get(key) != value for key, value in expected_metadata.items()):
        raise RuntimeError("original LoRA metadata differs from the matched native task")

    # Constructor randomness is quarantined. Copy the teacher before loading
    # the original adapter so target-caption velocities retain pure base weights.
    with torch.random.fork_rng(devices=[device.index or 0]):
        module = model_module()
        with torch.device("meta"):
            base = module.SupraDiT()
            module.attach_supra_lora(base, rank=16, alpha=16, targets=TARGETS)
        base.load_state_dict(teacher_state, strict=True, assign=True)
        base.to(device).eval().requires_grad_(False)
        encoder = FrozenSliderContexts(data["text_contexts"], data["text_masks"], base).to(device)
        original = deepcopy(base).eval().requires_grad_(False)
        state = load_file(str(args.baseline), device="cpu")
        expected = adapter_state(original)
        if set(state) != set(expected) or len(state) != 142:
            raise RuntimeError("original LoRA must provide all 71 native projection sites")
        for key, value in state.items():
            if value.shape != expected[key].shape or value.dtype != expected[key].dtype:
                raise RuntimeError(f"original LoRA shape/dtype mismatch: {key}")
            if not bool(torch.isfinite(value).all()):
                raise RuntimeError(f"original LoRA has nonfinite weights: {key}")
        load_adapter_state(original, state)
        critics = []
        for saved in (v1, v2):
            critic_state = saved["policy"]["models"]["critic"]
            if not torch.equal(critic_state["scale"], data["coordinate_scale"]):
                raise RuntimeError("critic coordinate units differ from the original task")
            critic = ConditionalTokenCritic(critic_state["scale"]).to(device)
            critic.load_state_dict(critic_state, strict=True)
            critics.append(critic.eval().requires_grad_(False))

    class OriginalServed(torch.nn.Module):
        """Only the interfaces consumed by the shared read-only evaluator."""

        def __init__(self, model, contexts):
            super().__init__()
            self.model, self.encoder = model, contexts

        @torch.no_grad()
        def velocity(self, context, *, strength=None):
            z, t, ctx, mask, uctx, umask, packed_strength = self.encoder.unpack(context)
            batch = len(z)
            ordinary = [branch for branch in self.model.modules() if hasattr(branch, "multiplier")]
            previous = [branch.multiplier for branch in ordinary]
            try:
                for branch in ordinary:
                    branch.multiplier = packed_strength if strength is None else strength
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    value = self.model(torch.cat((z, z)), torch.cat((t, t)),
                                       torch.cat((ctx, uctx.expand(batch, -1, -1))),
                                       torch.cat((mask, umask.expand(batch, -1))))
                conditional, unconditional = value.float().chunk(2)
                return unconditional + 3 * (conditional - unconditional)
            finally:
                for branch, multiplier in zip(ordinary, previous):
                    branch.multiplier = multiplier

        @torch.no_grad()
        def routed_forward(self, context):
            return self.velocity(context) - self.encoder.teacher_velocity(context)

    served = OriginalServed(original, encoder).eval().requires_grad_(False)
    with torch.no_grad():
        context = data["test"]["context"][:4].to(device)
        source = context.clone()
        source[:, 4098] = source[:, 4097]
        if not torch.equal(served.velocity(context, strength=0.), encoder.teacher_velocity(source)):
            raise RuntimeError("original strength-zero output differs from the independent frozen base")
    before = dict(served=state_digest(served.state_dict()),
                  critics=state_digest([critic.state_dict() for critic in critics]),
                  data=state_digest(data), cpu_rng=torch.get_rng_state().clone(),
                  cuda_rng=torch.cuda.get_rng_state(device).clone(),
                  multipliers=[branch.multiplier for branch in served.modules() if hasattr(branch, "multiplier")])
    plan = dict(task="original published-example LoRA versus fixed particle-game critics", step=1600,
                original_recipe="ordinary rank16 LoRA; AdamW; paired velocity MSE with preservation",
                inputs={label: str(path.resolve()) for label, path in inputs.items()}, input_sha256=input_hashes,
                particlegan_commit=PIN, particlegan_source_digest=state_digest(pg_hashes), source_sha256=source_hashes,
                precision="native BF16 host and CFG3; FP32 residual and critics; batches of four",
                panels="shared evaluate_final private CUDA generator72, four paired draws per context",
                pools=["fit", "test", "holds", "preservation"], optimization=False,
                checkpoint_selection="existing original final fixed-1600 snapshot; no selection by this evaluation",
                critic_labels={"fixed_start_D": "archived particle V1 final1600 critic",
                               "arm_final_D": "particle V2 final1600 critic"},
                limitation="original recipe differs in architecture and objective; critic scores are learned/endogenous; "
                           "no isolated attribution to optimizer and no general superiority claim")
    write_json(args.output / "plan.json", plan)
    print(json.dumps(dict(event="original_evaluation_start", **plan)), flush=True)
    metrics = evaluate_final(served, data, *critics, device)
    unchanged = (before["served"] == state_digest(served.state_dict())
                 and before["critics"] == state_digest([critic.state_dict() for critic in critics])
                 and before["data"] == state_digest(data)
                 and torch.equal(before["cpu_rng"], torch.get_rng_state())
                 and torch.equal(before["cuda_rng"], torch.cuda.get_rng_state(device))
                 and before["multipliers"] == [branch.multiplier for branch in served.modules() if hasattr(branch, "multiplier")])
    if not unchanged:
        raise RuntimeError("evaluation mutated model/critic/data/global RNG state")
    if any(sha(path) != input_hashes[label] for label, path in inputs.items()):
        raise RuntimeError("an evaluation input changed during execution")
    if any(sha(ROOT / name) != value for name, value in source_hashes.items()):
        raise RuntimeError("evaluation source changed during execution")
    write_json(args.output / "evaluation.json", metrics)
    summary = {name: {key: value for key, value in pool.items() if key != "records"} for name, pool in metrics.items()}
    receipt = dict(plan=plan, evaluation_state_unchanged=True, input_files_unchanged=True,
                   strength_zero_exact=True, seconds=time.perf_counter() - started, metrics=summary)
    write_json(args.output / "receipt.json", receipt)
    print(json.dumps(dict(event="original_evaluation_complete", **receipt)), flush=True)


if __name__ == "__main__":
    main()
