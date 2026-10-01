#!/usr/bin/env python3
"""Read-only final6400 particle ablations on the native full-pool game panels.

The declared clean/zero-code/mass-only interventions use identical private
CUDA72 draws in fit/test/holds/preservation order, matching evaluate_final.
No game update, output-MSE optimization, stopping or selection is performed.
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
PIN = "cabe2084284db923d525918cbf3e18de6f20faac"
ARMS = ("clean", "zero_particle_codes", "mass_only_routing")
DRAWS = 4


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


def emit(**row):
    print(json.dumps(row), flush=True)


def evaluate_pool(served, pool, critics, noise_batches, native_pool, *, pool_name):
    import torch
    import torch.nn.functional as F
    from supra.particle_game import patchify
    from evaluate_e22_supra_particle_contribution import mass_only_routing
    records = {arm: [] for arm in ARMS}
    candidate = served.routing.candidate_for(served.models, served.table, averaged=served.source == "averaged")
    noise_hash = hashlib.sha256()
    if native_pool["count"] != len(pool["context"]):
        raise RuntimeError("native final evaluation pool size differs")
    with torch.no_grad():
        for batch_index, start in enumerate(range(0, len(pool["context"]), 4)):
            context = pool["context"][start:start + 4].to(served.table.device)
            clean = served.routing.forward(served.models, context, candidate)
            code_calls = [0]
            def zero_code(codes):
                code_calls[0] += 1
                return torch.zeros_like(codes)
            zero_codes = served.routing.forward(served.models, context, candidate, perturb_fn=zero_code)
            if code_calls[0] != len(served.generator.sites):
                raise RuntimeError("code ablation did not visit all native sites")
            with mass_only_routing(served.router) as routing_calls:
                mass_only = served.routing.forward(served.models, context, candidate)
            if routing_calls[0] != len(served.generator.sites):
                raise RuntimeError("mass-only intervention did not visit all native sites")
            if any(branch.frame is not None for branch in served.generator.particle_branches()) or served.generator._in_forward:
                raise RuntimeError("intervention retained a completed host frame")
            predictions = dict(clean=clean, zero_particle_codes=zero_codes, mass_only_routing=mass_only)
            noise = noise_batches[batch_index]
            if noise.shape != (DRAWS, len(context), 256, 16):
                raise RuntimeError("private paired panel has incorrect shape")
            noise_hash.update(noise.cpu().numpy().tobytes())
            bases = .125 * noise
            real = bases.flatten(0, 1)
            condition = served.encoder.condition(context).repeat(DRAWS, 1)
            logits = {}
            for label, critic in critics.items():
                real_scores = critic(real, condition)
                logits[label] = {}
                for arm, residual in predictions.items():
                    if residual.shape != (len(context), 4, 32, 32) or not bool(torch.isfinite(residual).all()):
                        raise RuntimeError(f"invalid full-model residual in {arm}")
                    fake = (bases + (patchify(residual).float() / critic.scale).unsqueeze(0)).flatten(0, 1)
                    fake_scores = critic(fake, condition)
                    # Match evaluate_final's elementwise operations and reduce
                    # the complete [draw,batch] matrix before indexing. A mean
                    # of a strided column can use a different CUDA reduction.
                    gap = real_scores - fake_scores
                    values = dict(g_game=F.softplus(gap).reshape(DRAWS, len(context)),
                                  d_game=F.softplus(fake_scores - real_scores).reshape(DRAWS, len(context)),
                                  score_gap=gap.reshape(DRAWS, len(context)))
                    values["mean"] = {key: value.mean(0) for key, value in values.items()}
                    logits[label][arm] = values
            for arm, residual in predictions.items():
                change = (residual - clean).float().square().flatten(1).mean(1)
                for index in range(len(context)):
                    row = dict(index=start + index, source_caption_id=int(context[index, 4097]), time=float(context[index, 4096]),
                               output_change_mse_evaluation_only=float(change[index]))
                    for label in critics:
                        values, baseline = logits[label][arm], logits[label]["clean"]
                        delta = values["g_game"] - baseline["g_game"]
                        row[label] = dict(g_game=float(values["mean"]["g_game"][index]),
                                          d_game=float(values["mean"]["d_game"][index]),
                                          score_gap=float(values["mean"]["score_gap"][index]),
                                          per_draw_g_game=values["g_game"][:, index].cpu().tolist(),
                                          per_draw_g_game_delta_from_clean=(values["g_game"][:, index] - baseline["g_game"][:, index]).cpu().tolist(),
                                          g_game_delta_from_clean=float(delta.mean(0)[index]))
                    if arm == "clean":
                        native = native_pool["records"][row["index"]]
                        if any(row[key] != native[key] for key in ("index", "source_caption_id", "time")):
                            raise RuntimeError("native and audit clean coordinates differ")
                        for label in critics:
                            for key in ("g_game", "d_game", "score_gap"):
                                actual, expected = row[label][key], native[label][key]
                                if actual != expected:
                                    raise RuntimeError("clean game is not bit-exact with native evaluate_final: "
                                                       f"{pool_name}/{row['index']}/{label}/{key}; "
                                                       f"actual={actual!r} ({actual.hex()}), expected={expected!r} ({expected.hex()})")
                    records[arm].append(row)
            if (batch_index + 1) % 10 == 0 or start + 4 >= len(pool["context"]):
                emit(event="particle_contribution_progress", pool=pool_name, completed=min(start + 4, len(pool["context"])), total=len(pool["context"]))

    def summarize(rows):
        count = len(rows)
        result = dict(contexts=count, output_change_rms_evaluation_only=(sum(row["output_change_mse_evaluation_only"] for row in rows) / count) ** .5)
        for label in critics:
            result[label] = dict(g_game=sum(row[label]["g_game"] for row in rows) / count,
                                 d_game=sum(row[label]["d_game"] for row in rows) / count,
                                 score_gap=sum(row[label]["score_gap"] for row in rows) / count,
                                 g_game_delta_from_clean=sum(row[label]["g_game_delta_from_clean"] for row in rows) / count,
                                 per_draw_g_game=[sum(row[label]["per_draw_g_game"][draw] for row in rows) / count for draw in range(DRAWS)],
                                 per_draw_g_game_delta_from_clean=[sum(row[label]["per_draw_g_game_delta_from_clean"][draw] for row in rows) / count for draw in range(DRAWS)],
                                 contexts_game_worsened=sum(row[label]["g_game_delta_from_clean"] > 0 for row in rows),
                                 contexts_game_improved=sum(row[label]["g_game_delta_from_clean"] < 0 for row in rows),
                                 contexts_game_unchanged=sum(row[label]["g_game_delta_from_clean"] == 0 for row in rows))
        return result
    result = dict(contexts=len(pool["context"]), gaussian_panels=DRAWS, paired_noise_sha256=noise_hash.hexdigest(),
                  output_sigma=.125, clean_native_final_evaluation_exact=True, per_arm={})
    for arm, rows in records.items():
        subjects = sorted({row["source_caption_id"] for row in rows})
        result["per_arm"][arm] = {**summarize(rows), "per_subject": {
            str(subject): summarize([row for row in rows if row["source_caption_id"] == subject]) for subject in subjects}, "records": rows}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "outputs/e22-particle-v2-6400")
    parser.add_argument("--reference", type=Path, default=ROOT / "outputs/e22-particle-noise-update-256/latest")
    parser.add_argument("--particlegan-root", type=Path, default=ROOT / "outputs/e22-convergence-gap/particlegan-cabe2084-source")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.output = args.output or args.run / "particle-contribution-final6400.json"
    if args.output.exists():
        parser.error("choose a fresh output file")
    sys.path.insert(0, str(args.particlegan_root.resolve()))
    sys.path.insert(1, str(ROOT))
    import torch
    import particlegan
    from supra.particle_adapter import LINEAR_MODULATED_V2
    from supra.particle_game import ConditionalTokenCritic
    from supra.particle_pilot import checkpoint, restore, state_digest
    from supra.particle_training import make_training_loop
    from supra.runtime import model_module, TARGETS
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("native BF16 game contribution evaluation requires CUDA")
    torch.cuda.set_device(device)
    torch.set_num_threads(8)
    started = time.perf_counter()
    source_dir = Path(particlegan.__file__).resolve().parent
    pg_hashes = {path.name: sha(path) for path in sorted(source_dir.glob("*.py"))}
    pg_digest = state_digest(pg_hashes)
    if source_dir.parent != args.particlegan_root.resolve():
        raise RuntimeError("ParticleGAN import differs from source archive")
    inputs = dict(final=args.run / "final.pt", reference=args.reference / "final.pt", data=args.run / "data.pt",
                  reference_data=args.reference / "data.pt", run=args.run / "run.json", receipt=args.run / "receipt.json",
                  reference_run=args.reference / "run.json", reference_receipt=args.reference / "receipt.json",
                  evaluation=args.run / "evaluation.json", sources=args.run / "source/sha256.json")
    input_hashes = {name: sha(path) for name, path in inputs.items()}
    if input_hashes["data"] != input_hashes["reference_data"]:
        raise RuntimeError("current and fixed-reference evaluation pools differ")
    runs = [json.loads(inputs[name].read_text()) for name in ("run", "reference_run")]
    receipts = [json.loads(inputs[name].read_text()) for name in ("receipt", "reference_receipt")]
    for run, receipt, step in zip(runs, receipts, (6400, 1856)):
        if run["particlegan_commit"] != PIN or run["particlegan_source_digest"] != pg_digest or run["fixed_updates"] != step:
            raise RuntimeError("native source pin/horizon differs from the qualified comparison")
        if run["config"].get("architecture") != LINEAR_MODULATED_V2:
            raise RuntimeError("final contribution audit requires qualified V2")
        for field in ("initial_native_state_exact", "two_update_replay_exact", "frozen_unchanged", "export_reload_exact",
                      "evaluation_state_unchanged", "source_and_inputs_unchanged"):
            if receipt.get(field) is not True:
                raise RuntimeError(f"native run lacks qualification: {field}")
    if receipts[0]["final_checkpoint_sha256"] != input_hashes["final"] or receipts[0]["plan"]["input_sha256"]["final.pt"] != input_hashes["reference"]:
        raise RuntimeError("critic lineage does not match the native final receipt")
    if receipts[0].get("data_and_paired_streams_matched") is not True:
        raise RuntimeError("native continuation lacks matched sampled-program qualification")
    sources = [Path(__file__), ROOT / "scripts/evaluate_e22_supra_particle_contribution.py", ROOT / "scripts/diagnose_e22_supra_training_controls.py"]
    sources += [ROOT / "supra" / name for name in ("runtime.py", "particle_adapter.py", "particle_game.py", "particle_pilot.py", "particle_training.py", "particle_training_data.py")]
    source_hashes = {str(path.relative_to(ROOT)): sha(path) for path in sources}
    native_sources = json.loads(inputs["sources"].read_text())["application"]
    for name, value in source_hashes.items():
        if name in native_sources and native_sources[name] != value:
            raise RuntimeError(f"shared contribution source differs from native final run: {name}")
    source_output = args.output.parent / "particle-contribution-final6400-source"
    if source_output.exists():
        parser.error("choose a fresh output directory for this audit source")
    for path in sources:
        destination = source_output / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    write_json(source_output / "sha256.json", source_hashes)
    saved = torch.load(inputs["final"], map_location="cpu", weights_only=False, mmap=True)
    reference = torch.load(inputs["reference"], map_location="cpu", weights_only=False, mmap=True)
    data = torch.load(inputs["data"], map_location="cpu", weights_only=False)
    for saved_state, run, receipt, step in zip((saved, reference), runs, receipts, (6400, 1856)):
        if saved_state["policy"]["completed_steps"] != step or state_digest(saved_state) != receipt["final_native_digest"]:
            raise RuntimeError("native checkpoint differs from its complete qualified state")
        if json.loads(json.dumps(saved_state["config"])) != run["config"] or saved_state["config"]["dataset_digest"] != state_digest(data):
            raise RuntimeError("native configuration or dataset differs")
    if str(device) != saved["policy"]["device"]:
        raise RuntimeError("exact restore requires the owned logical device")
    teacher = {key.removeprefix("teacher."): value for key, value in saved["policy"]["models"]["encoder"].items() if key.startswith("teacher.")}
    reference_teacher = {key.removeprefix("teacher."): value for key, value in reference["policy"]["models"]["encoder"].items() if key.startswith("teacher.")}
    if state_digest(teacher) != state_digest(reference_teacher):
        raise RuntimeError("fixed and final native teacher states differ")
    with torch.random.fork_rng(devices=[device.index or 0]):
        module = model_module()
        with torch.device("meta"):
            base = module.SupraDiT()
            module.attach_supra_lora(base, rank=16, alpha=16, targets=TARGETS)
        base.load_state_dict(teacher, strict=True, assign=True)
        base.to(device).eval().requires_grad_(False)
        loop = make_training_loop(base, data, device=device, probe_interval=saved["config"]["probe_interval"],
                                  branch_lr=saved["config"]["branch_lr"], architecture=LINEAR_MODULATED_V2)
    restore(loop, saved)
    before = state_digest(checkpoint(loop))
    if before != receipts[0]["final_native_digest"]:
        raise RuntimeError("final native checkpoint did not restore bit-exactly")
    served = loop.policy.served_model()
    if served.generator.architecture != LINEAR_MODULATED_V2 or served.source != "fast":
        raise RuntimeError("unexpected public architecture or served model")
    with torch.random.fork_rng(devices=[device.index or 0]):
        fixed = ConditionalTokenCritic(reference["policy"]["models"]["critic"]["scale"]).to(device)
        fixed.load_state_dict(reference["policy"]["models"]["critic"], strict=True)
        fixed.eval().requires_grad_(False)
    critics = dict(fixed_start_D=fixed, arm_final_D=deepcopy(loop.policy.D).eval().requires_grad_(False))
    if any(not torch.equal(critic.scale, data["coordinate_scale"].to(device)) for critic in critics.values()):
        raise RuntimeError("critic coordinate units differ from owned task")
    def served_digest():
        return state_digest(dict(models={name: model.state_dict() for name, model in served.models.items()},
                                 table=served.table, controller=served.controller.state_dict(), stream=served.stream.get_state()))
    def gradient_digest():
        return state_digest({role: {name: parameter.grad for name, parameter in model.named_parameters()}
                             for role, model in loop.policy._training_modules().items()} | {"table": loop.policy.table.grad})
    served_before = served_digest()
    gradients_before = gradient_digest()
    critic_hashes = {name: state_digest(critic.state_dict()) for name, critic in critics.items()}
    global_rng_before = (torch.get_rng_state().clone(), torch.cuda.get_rng_state(device).clone())
    private_stream = torch.Generator(device=device).manual_seed(72)
    panels = {}
    for pool_name in ("fit", "test", "holds", "preservation"):
        pool_panels = [torch.randn((DRAWS, min(4, len(data[pool_name]["context"]) - start), 256, 16),
                                  device=device, dtype=torch.float32, generator=private_stream)
                       for start in range(0, len(data[pool_name]["context"]), 4)]
        if pool_name in ("test", "preservation"):
            panels[pool_name] = pool_panels
    native_evaluation = json.loads(inputs["evaluation"].read_text())
    report = dict(complete=False, metadata=dict(checkpoint=str(inputs["final"].resolve()), checkpoint_sha256=input_hashes["final"],
                  reference_checkpoint=str(inputs["reference"].resolve()), reference_checkpoint_sha256=input_hashes["reference"],
                  completed_steps=6400, architecture=LINEAR_MODULATED_V2, particlegan_commit=PIN, particlegan_source_digest=pg_digest,
                  served_source=served.source, bank_shape=list(served.table.shape), native_sites=len(served.generator.sites),
                  critic_state_sha256=critic_hashes, arms=list(ARMS), draws=DRAWS, output_sigma=.125,
                  perturbation="clean native full-model forwards; code ablation at every routing site",
                  mass_only_routing="zero all site queries; retain all active bank values and row masses",
                  noise="one private CUDA72 generator, advancing in fit/test/holds/preservation order exactly as evaluate_final",
                  native_game="paired RpGAN softplus(real_score - fake_score), unweighted diagnostic payoffs",
                  precision="native BF16 host/CFG3, FP32 residual and critics, batch4",
                  evaluation_only=True, optimizer_updates=0, output_metrics="intervention RMS only; no optimizer or selection",
                  checkpoint_selection="declared final fixed6400 snapshot, no outcome selection",
                  input_sha256=input_hashes, source_sha256=source_hashes,
                  limitations="Endogenous critics and fixed fixture ablations establish contribution here; they are not independent image-quality judges or proof of general superiority."), pools={})
    write_json(args.output, report)
    for pool_name in ("test", "preservation"):
        report["pools"][pool_name] = evaluate_pool(served, data[pool_name], critics, panels[pool_name], native_evaluation[pool_name], pool_name=pool_name)
        emit(event="particle_contribution_pool_complete", pool=pool_name,
             deltas={arm: {label: row[label]["g_game_delta_from_clean"] for label in critics}
                     for arm, row in report["pools"][pool_name]["per_arm"].items()})
        write_json(args.output, report)
    if before != state_digest(checkpoint(loop)) or served_before != served_digest() or gradients_before != gradient_digest():
        raise RuntimeError("particle contribution audit changed native/served/gradient state or owned RNGs")
    if critic_hashes != {name: state_digest(critic.state_dict()) for name, critic in critics.items()}:
        raise RuntimeError("particle contribution audit changed a frozen critic")
    if not torch.equal(global_rng_before[0], torch.get_rng_state()) or not torch.equal(global_rng_before[1], torch.cuda.get_rng_state(device)):
        raise RuntimeError("particle contribution audit advanced global RNGs")
    if any(sha(path) != input_hashes[name] for name, path in inputs.items()) or any(sha(ROOT / name) != value for name, value in source_hashes.items()):
        raise RuntimeError("particle contribution audit input or source changed")
    if any(sha(source_dir / name) != value for name, value in pg_hashes.items()):
        raise RuntimeError("ParticleGAN archived source changed during audit")
    report.update(complete=True, full_checkpoint_and_RNG_unchanged=True, served_model_and_bank_unchanged=True,
                  gradients_unchanged=True, fixed_critics_unchanged=True, global_RNG_unchanged=True,
                  input_and_sources_unchanged=True, clean_native_final_evaluation_exact=True, seconds=time.perf_counter() - started)
    write_json(args.output, report)
    emit(event="particle_contribution_complete", output=str(args.output), seconds=report["seconds"], full_native_state_unchanged=True)


if __name__ == "__main__":
    main()
