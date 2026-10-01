#!/usr/bin/env python3
"""Fixed-6400 original LoRA versus native particles under two shared judges.

Both frozen critics, all complete pools and private paired Gaussian panels
are exactly those used by the native V2 continuation's final evaluation.
This loads the declared ordinary-LoRA step-6400 snapshot without training,
stopping or selection. Output errors are diagnostics only.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
PIN = "cabe2084284db923d525918cbf3e18de6f20faac"
BASELINE_SHA = "24a98ba055d0b5298136d67dc7491a339684da03416c510d1d443fe730eb7069"
BASELINE = Path("/ml2/hypergan/supra-concept-sliders/outputs/final-boss-supra-converged/checkpoint-06400/final-boss-supra.safetensors")


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
    parser.add_argument("--run", type=Path, default=ROOT / "outputs/e22-particle-v2-6400")
    parser.add_argument("--reference", type=Path, default=ROOT / "outputs/e22-particle-noise-update-256/latest")
    parser.add_argument("--particlegan-root", type=Path,
                        default=ROOT / "outputs/e22-convergence-gap/particlegan-cabe2084-source")
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.output = args.output or args.run / "evaluation-original-6400-game"
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("choose a fresh output directory")
    sys.path.insert(0, str(args.particlegan_root.resolve()))
    sys.path.insert(1, str(ROOT))
    import torch
    import particlegan
    from safetensors import safe_open
    from safetensors.torch import load_file
    from supra.runtime import MODEL_ID, MODEL_REV, T5_REV, VAE_REV, TARGETS, adapter_state, load_adapter_state, model_module
    from supra.particle_adapter import LINEAR_MODULATED_V2
    from supra.particle_game import ConditionalTokenCritic
    from supra.particle_pilot import state_digest
    from supra.particle_training_data import FrozenSliderContexts
    from diagnose_e22_supra_training_controls import evaluate_final

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("matched native Supra/BF16 evaluation requires CUDA")
    torch.cuda.set_device(device)
    torch.set_num_threads(8)
    started = time.perf_counter()
    source_dir = Path(particlegan.__file__).resolve().parent
    pg_digest = state_digest({path.name: sha(path) for path in sorted(source_dir.glob("*.py"))})
    if source_dir.parent != args.particlegan_root.resolve():
        raise RuntimeError("ParticleGAN import differs from declared source archive")
    inputs = dict(original=args.baseline, initial=args.reference / "final.pt", final=args.run / "final.pt",
                  initial_data=args.reference / "data.pt", data=args.run / "data.pt",
                  initial_run=args.reference / "run.json", final_run=args.run / "run.json",
                  initial_receipt=args.reference / "receipt.json", final_receipt=args.run / "receipt.json",
                  native_evaluation=args.run / "evaluation.json", native_plan=args.run / "plan.json",
                  native_sources=args.run / "source/sha256.json")
    input_hashes = {name: sha(path) for name, path in inputs.items()}
    if input_hashes["original"] != BASELINE_SHA or input_hashes["data"] != input_hashes["initial_data"]:
        raise RuntimeError("original checkpoint or complete paired pools differ from declared benchmark")
    initial_run = json.loads(inputs["initial_run"].read_text())
    final_run = json.loads(inputs["final_run"].read_text())
    initial_receipt = json.loads(inputs["initial_receipt"].read_text())
    final_receipt = json.loads(inputs["final_receipt"].read_text())
    native_plan = json.loads(inputs["native_plan"].read_text())
    for run, step in ((initial_run, 1856), (final_run, 6400)):
        if (run["particlegan_commit"] != PIN or run["particlegan_source_digest"] != pg_digest
                or run["fixed_updates"] != step or run["config"].get("architecture") != LINEAR_MODULATED_V2):
            raise RuntimeError("qualified native run/source/horizon/architecture differs")
        if (run["model_revision"], run["text_revision"], run["vae_revision"]) != (MODEL_REV, T5_REV, VAE_REV):
            raise RuntimeError("native model/text/VAE pins differ")
    for receipt in (initial_receipt, final_receipt):
        for field in ("initial_native_state_exact", "two_update_replay_exact", "frozen_unchanged",
                      "export_reload_exact", "evaluation_state_unchanged", "source_and_inputs_unchanged"):
            if receipt.get(field) is not True:
                raise RuntimeError(f"native run lacks required qualification: {field}")
    if final_receipt.get("data_and_paired_streams_matched") is not True:
        raise RuntimeError("fixed native continuation lacks sampled-program qualification")
    if final_receipt["final_checkpoint_sha256"] != input_hashes["final"]:
        raise RuntimeError("native final checkpoint differs from its receipt")
    if (native_plan["start_step"] != 1856 or native_plan["fixed_updates"] != 6400
            or native_plan["input_sha256"]["final.pt"] != input_hashes["initial"]
            or native_plan["original_baseline_sha256"] != BASELINE_SHA):
        raise RuntimeError("native continuation does not own the declared judges and original benchmark")
    if final_receipt["plan"] != native_plan:
        raise RuntimeError("native final receipt differs from its original fixed plan")
    source_paths = [Path(__file__), ROOT / "scripts/diagnose_e22_supra_training_controls.py"]
    source_paths += [ROOT / "supra" / name for name in
                     ("runtime.py", "particle_adapter.py", "particle_game.py", "particle_pilot.py", "particle_training_data.py")]
    source_hashes = {str(path.relative_to(ROOT)): sha(path) for path in source_paths}
    native_source_hashes = json.loads(inputs["native_sources"].read_text())["application"]
    for name, value in source_hashes.items():
        if name != str(Path(__file__).relative_to(ROOT)) and native_source_hashes.get(name) != value:
            raise RuntimeError(f"shared evaluation source differs from the qualified native run: {name}")
    args.output.mkdir(parents=True, exist_ok=True)
    for path in source_paths:
        destination = args.output / "source" / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    write_json(args.output / "source/sha256.json", source_hashes)
    data = torch.load(inputs["data"], map_location="cpu", weights_only=False)
    initial = torch.load(inputs["initial"], map_location="cpu", weights_only=False, mmap=True)
    final = torch.load(inputs["final"], map_location="cpu", weights_only=False, mmap=True)
    for saved, run, step, receipt in ((initial, initial_run, 1856, initial_receipt), (final, final_run, 6400, final_receipt)):
        if saved["policy"]["completed_steps"] != step or json.loads(json.dumps(saved["config"])) != run["config"]:
            raise RuntimeError("native checkpoint configuration/horizon differs from receipt")
        if state_digest(saved) != receipt["final_native_digest"]:
            raise RuntimeError("full native checkpoint digest differs from qualified receipt")
        if saved["config"]["dataset_digest"] != state_digest(data):
            raise RuntimeError("native checkpoint does not own complete evaluation pools")
        encoded = saved["policy"]["models"]["encoder"]
        if not torch.equal(encoded["contexts"], data["text_contexts"]) or not torch.equal(encoded["masks"], data["text_masks"]):
            raise RuntimeError("native checkpoint text buffers differ from paired data")
    teacher_state = {key.removeprefix("teacher."): value for key, value in initial["policy"]["models"]["encoder"].items()
                     if key.startswith("teacher.")}
    final_teacher = {key.removeprefix("teacher."): value for key, value in final["policy"]["models"]["encoder"].items()
                     if key.startswith("teacher.")}
    if state_digest(teacher_state) != state_digest(final_teacher):
        raise RuntimeError("frozen teacher differs between the two owned critics")
    if any(bool(value.count_nonzero()) for key, value in teacher_state.items() if key.endswith(".up.weight")):
        raise RuntimeError("paired teacher must contain pure-base zero-up ordinary adapters")
    with safe_open(str(args.baseline), framework="pt", device="cpu") as handle:
        metadata = {key: json.loads(value) for key, value in (handle.metadata() or {}).items()}
    expected_metadata = dict(format="supra-native-lora-v1", rank=16, alpha=16, targets=list(TARGETS),
                             model_id=MODEL_ID, model_revision=MODEL_REV, text_encoder_revision=T5_REV,
                             vae_revision=VAE_REV, step=6400, prompts_sha256=final_run["prompts_sha256"])
    if any(metadata.get(key) != value for key, value in expected_metadata.items()):
        raise RuntimeError("original LoRA metadata differs from the native paired task")
    # Copy the pure-base teacher before installing the historical learned LoRA.
    with torch.random.fork_rng(devices=[device.index or 0]):
        module = model_module()
        with torch.device("meta"):
            original = module.SupraDiT()
            module.attach_supra_lora(original, rank=16, alpha=16, targets=TARGETS)
        original.load_state_dict(teacher_state, strict=True, assign=True)
        original.to(device).eval().requires_grad_(False)
        encoder = FrozenSliderContexts(data["text_contexts"], data["text_masks"], original).to(device)
        weights = load_file(str(args.baseline), device="cpu")
        expected = adapter_state(original)
        if set(weights) != set(expected) or len(weights) != 142:
            raise RuntimeError("original LoRA must provide all 71 native sites")
        for key, value in weights.items():
            if value.shape != expected[key].shape or value.dtype != expected[key].dtype or not bool(torch.isfinite(value).all()):
                raise RuntimeError(f"invalid original adapter tensor: {key}")
        load_adapter_state(original, weights)
        critics = []
        for saved in (initial, final):
            critic_state = saved["policy"]["models"]["critic"]
            if not torch.equal(critic_state["scale"], data["coordinate_scale"]):
                raise RuntimeError("critic coordinate units differ from task")
            critic = ConditionalTokenCritic(critic_state["scale"]).to(device)
            critic.load_state_dict(critic_state, strict=True)
            critics.append(critic.eval().requires_grad_(False))

    class OriginalServed(torch.nn.Module):
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
                    output = self.model(torch.cat((z, z)), torch.cat((t, t)),
                                        torch.cat((ctx, uctx.expand(batch, -1, -1))),
                                        torch.cat((mask, umask.expand(batch, -1))))
                conditional, unconditional = output.float().chunk(2)
                return unconditional + 3 * (conditional - unconditional)
            finally:
                for branch, multiplier in zip(ordinary, previous):
                    branch.multiplier = multiplier

        @torch.no_grad()
        def routed_forward(self, context):
            return self.velocity(context) - self.encoder.teacher_velocity(context)

    served = OriginalServed(original, encoder).eval().requires_grad_(False)
    context = data["test"]["context"][:4].to(device)
    source_context = context.clone()
    source_context[:, 4098] = source_context[:, 4097]
    if not torch.equal(served.velocity(context, strength=0.), encoder.teacher_velocity(source_context)):
        raise RuntimeError("original strength-zero output differs from independent frozen base")
    before = dict(served=state_digest(served.state_dict()), critics=state_digest([critic.state_dict() for critic in critics]),
                  data=state_digest(data), cpu_rng=torch.get_rng_state().clone(), cuda_rng=torch.cuda.get_rng_state(device).clone(),
                  multipliers=[branch.multiplier for branch in served.modules() if hasattr(branch, "multiplier")])
    plan = dict(task="fixed6400 original ordinary LoRA versus native V2 particles", fixed_updates=6400,
                original_recipe="ordinary rank16 LoRA; AdamW; paired velocity MSE with preservation",
                particle_recipe="linear-modulated V2; shared128x4 bank; 71 routing sites; native E22 game/DV12/structural controls",
                inputs={name: str(path.resolve()) for name, path in inputs.items()}, input_sha256=input_hashes,
                particlegan_commit=PIN, particlegan_source_digest=pg_digest, source_sha256=source_hashes,
                precision="native BF16 host and CFG3; FP32 residual and critics; batches of four",
                panels="identical evaluate_final private CUDA generator72; four paired draws per context; output sigma.125",
                pools=["fit", "test", "holds", "preservation"], optimization=False,
                output_metrics="evaluation only; no loss, optimizer, structural criterion, stopping or selection",
                checkpoint_selection="declared original fixed6400 snapshot and native fixed6400 final; no outcome selection",
                critic_labels={"fixed_start_D": "common V2 step1856 critic", "arm_final_D": "native V2 final6400 critic"},
                limitation="Complete formulations differ in architecture and objective; learned endogenous judges and one task/stream cannot isolate optimizer effects or prove general superiority")
    write_json(args.output / "plan.json", plan)
    print(json.dumps(dict(event="original6400_evaluation_start", **plan)), flush=True)
    metrics = evaluate_final(served, data, *critics, device)
    unchanged = (before["served"] == state_digest(served.state_dict())
                 and before["critics"] == state_digest([critic.state_dict() for critic in critics])
                 and before["data"] == state_digest(data) and torch.equal(before["cpu_rng"], torch.get_rng_state())
                 and torch.equal(before["cuda_rng"], torch.cuda.get_rng_state(device))
                 and before["multipliers"] == [branch.multiplier for branch in served.modules() if hasattr(branch, "multiplier")])
    if not unchanged:
        raise RuntimeError("original evaluation mutated served weights/critic/data/global RNG state")
    if any(sha(path) != input_hashes[name] for name, path in inputs.items()):
        raise RuntimeError("an original evaluation input changed")
    if any(sha(ROOT / name) != value for name, value in source_hashes.items()):
        raise RuntimeError("original evaluation source changed")
    native_metrics = json.loads(inputs["native_evaluation"].read_text())
    comparison = {}
    for name, original_pool in metrics.items():
        native_pool = native_metrics[name]
        if original_pool["count"] != native_pool["count"] or len(original_pool["records"]) != len(native_pool["records"]):
            raise RuntimeError("native and original complete evaluation pool sizes differ")
        for a, b in zip(original_pool["records"], native_pool["records"]):
            if any(a[key] != b[key] for key in ("index", "source_caption_id", "time", "teacher_rms")):
                raise RuntimeError("native and original per-context coordinates/teacher differ")
        row = dict(count=original_pool["count"], original_rmse=original_pool["rmse"], particle_rmse=native_pool["rmse"],
                   output_errors_diagnostic_only=True, original_relative_rms=original_pool["relative_rms"],
                   particle_relative_rms=native_pool["relative_rms"])
        for judge in ("fixed_start_D", "arm_final_D"):
            old, new = original_pool[judge]["g_game"], native_pool[judge]["g_game"]
            subjects = sorted({record["source_caption_id"] for record in original_pool["records"]})
            row[judge] = dict(original_g_game=old, particle_g_game=new, particle_minus_original=new-old,
                              particle_better_contexts=sum(b[judge]["g_game"] < a[judge]["g_game"] for a, b in
                                                          zip(original_pool["records"], native_pool["records"])),
                              per_source_caption={str(subject): dict(
                                  original_g_game=sum(r[judge]["g_game"] for r in original_pool["records"] if r["source_caption_id"] == subject)
                                      / sum(r["source_caption_id"] == subject for r in original_pool["records"]),
                                  particle_g_game=sum(r[judge]["g_game"] for r in native_pool["records"] if r["source_caption_id"] == subject)
                                      / sum(r["source_caption_id"] == subject for r in native_pool["records"])) for subject in subjects})
        comparison[name] = row
    write_json(args.output / "evaluation.json", metrics)
    write_json(args.output / "comparison.json", comparison)
    receipt = dict(plan=plan, evaluation_state_unchanged=True, input_files_unchanged=True, source_files_unchanged=True,
                   strength_zero_exact=True, complete_pools_and_teachers_matched=True, seconds=time.perf_counter()-started,
                   metrics={name: {key: value for key, value in pool.items() if key != "records"} for name, pool in metrics.items()},
                   comparison=comparison)
    write_json(args.output / "receipt.json", receipt)
    print(json.dumps(dict(event="original6400_evaluation_complete", **receipt)), flush=True)


if __name__ == "__main__":
    main()
