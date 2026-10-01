#!/usr/bin/env python3
"""Fixed-horizon particle-bank game trust test against native continuation.

Both arms start at the qualified V2 step6400 boundary on its pinned source.
The experimental arm projects only the proposed bank displacement using its
own training game. Output metrics remain final, read-only diagnostics.
"""
import argparse
from contextlib import nullcontext
from copy import deepcopy
import gc
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PIN = "cabe2084284db923d525918cbf3e18de6f20faac"
ARMS = ("native", "bank_game_trust_v1")
ORIGINAL = Path("/ml2/hypergan/supra-concept-sliders/outputs/final-boss-supra-converged/checkpoint-06400/final-boss-supra.safetensors")
ORIGINAL_SHA = "24a98ba055d0b5298136d67dc7491a339684da03416c510d1d443fe730eb7069"


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def finite_json(value):
    import math
    import torch
    if isinstance(value, torch.Tensor):
        return finite_json(value.detach().cpu().tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [finite_json(item) for item in value]
    return value


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(finite_json(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def emit(**row):
    print(json.dumps(finite_json(row), allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "outputs/e22-particle-v2-6400")
    parser.add_argument("--particlegan-root", type=Path,
                        default=ROOT / "outputs/e22-convergence-gap/particlegan-cabe2084-source")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/e22-bank-game-trust-256")
    parser.add_argument("--updates", type=int, default=256)
    parser.add_argument("--probe-every", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("choose a fresh output directory")
    if args.updates < 5 or args.probe_every < 1:
        parser.error("use at least five fixed updates and a positive probe interval")
    pg_root = args.particlegan_root.resolve()
    sys.path.insert(0, str(pg_root))
    sys.path.insert(1, str(ROOT))
    import torch
    import particlegan
    from supra.runtime import model_module, TARGETS, load_adapter_state
    from supra.particle_pilot import checkpoint, restore, state_digest, frozen_digest
    from supra.particle_training import make_training_loop, training_update
    from supra.particle_game import ConditionalTokenCritic, _rng_digest, patchify
    from monitor_e22_supra_particle_convergence import GameProgressMonitor, rolling_losses
    from diagnose_e22_supra_training_controls import evaluate_final
    from experimental_e22_bank_game_trust import bank_game_trust, FORMULATION, FRACTIONS

    torch.set_num_threads(4)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    imported = Path(particlegan.__file__).resolve()
    if imported.parent.parent != pg_root or FORMULATION != ARMS[1]:
        raise RuntimeError("unexpected native source or experimental law")
    run = json.loads((args.run / "run.json").read_text())
    review = json.loads((args.run / "qualification-review.json").read_text())
    saved = torch.load(args.run / "final.pt", map_location="cpu", weights_only=False, mmap=True)
    data = torch.load(args.run / "data.pt", map_location="cpu", weights_only=False)
    initial = state_digest(saved)
    pg_files = sorted(imported.parent.rglob("*.py"))
    pg_hashes = {str(path.relative_to(imported.parent)): sha(path) for path in pg_files}
    native_source_digest = state_digest({path.name: sha(path) for path in imported.parent.glob("*.py")})
    if (not review.get("qualified") or initial != review["final_native_digest"]
            or run["particlegan_commit"] != PIN or native_source_digest != run["particlegan_source_digest"]
            or saved["policy"]["completed_steps"] != 6400
            or state_digest(data) != saved["config"]["dataset_digest"]):
        raise RuntimeError("qualified initial checkpoint/source/data changed")
    if saved["config"]["output_error_guard"] or saved["config"]["max_feature_context_harm"] != 0:
        raise RuntimeError("this experiment requires the unchanged game-feature structural law")
    sources = [Path(__file__), ROOT / "scripts/experimental_e22_bank_game_trust.py",
               ROOT / "scripts/monitor_e22_supra_particle_convergence.py",
               ROOT / "scripts/diagnose_e22_supra_training_controls.py", ROOT / "supra/runtime.py"]
    sources += sorted((ROOT / "supra").glob("particle*.py"))
    app_hashes = {str(path.relative_to(ROOT)): sha(path) for path in sources}
    inputs = {name: args.run / name for name in ("final.pt", "data.pt", "run.json", "qualification-review.json")}
    inputs["original_reference"] = ORIGINAL
    inputs["qualified_five_step_reference"] = ROOT / "outputs/e22-table-dv12-exact-offset-6400/profiles.json"
    five_step_reference = json.loads(inputs["qualified_five_step_reference"].read_text())
    if not five_step_reference.get("exact_native_full_replay"):
        raise RuntimeError("the earlier native five-step replay lacks qualification")
    input_hashes = {name: sha(path) for name, path in inputs.items()}
    if input_hashes["original_reference"] != ORIGINAL_SHA:
        raise RuntimeError("original ordinary LoRA reference differs from its declared checkpoint")
    args.output.mkdir(parents=True)
    for path in sources:
        target = args.output / "source" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    for path in pg_files:
        target = args.output / "source/particlegan" / path.relative_to(imported.parent)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    plan = dict(schema="supra_bank_game_trust_continuation_v1", arms=list(ARMS), start_step=6400,
                updates=args.updates, final_step=6400 + args.updates, fractions=list(FRACTIONS),
                initial_native_digest=initial, particlegan_commit=PIN, native_source_digest=native_source_digest,
                input_paths={name: str(path.resolve()) for name, path in inputs.items()}, input_sha256=input_hashes,
                application_source_sha256=app_hashes, particlegan_source_sha256=pg_hashes,
                selection="final predeclared fixed horizon; no output-based stopping, selection or optimizer criterion",
                optimizer="native moments retained; experimental bank projection precedes native after_generator_step",
                objective="own training-batch paired RpGAN game, with replayed native perturbations and a clean game check",
                controls="same initial complete state, native source, context sampling, Gaussian draws and DV12 stream",
                structural="native learned feature criterion, zero per-context feature harm, no output guard",
                reference="ordinary LoRA remains overall reference; this causal test compares matched particle continuations",
                limitation="late continuation on one task/stream, not a fresh-start or SOTA benchmark")
    write(args.output / "plan.json", plan)
    write(args.output / "status.json", dict(phase="loading", arm="native", step=6400, steps=plan["final_step"]))
    emit(event="plan", **plan)
    began = time.perf_counter()
    with torch.random.fork_rng(devices=[device.index or 0]):
        module = model_module()
        with torch.device("meta"):
            base = module.SupraDiT()
            module.attach_supra_lora(base, rank=16, alpha=16, targets=TARGETS)
        base.load_state_dict({key.removeprefix("teacher."): value for key, value in
                              saved["policy"]["models"]["encoder"].items() if key.startswith("teacher.")},
                            strict=True, assign=True)
        base.to(device).eval().requires_grad_(False)
        fixed = ConditionalTokenCritic(data["coordinate_scale"]).to(device)
        fixed.load_state_dict(saved["policy"]["models"]["critic"], strict=True)
        fixed.eval().requires_grad_(False)
        loop = make_training_loop(base, data, device=device, architecture=saved["config"]["architecture"],
                                  probe_interval=saved["config"]["probe_interval"], branch_lr=saved["config"]["branch_lr"])
    restore(loop, saved)
    if state_digest(checkpoint(loop)) != initial:
        raise RuntimeError("native initial restore is not exact")
    frozen = frozen_digest(loop)
    common_dv12 = saved["policy"]["streams"]["noise_generator"].clone()
    from safetensors.torch import load_file
    with torch.random.fork_rng(devices=[device.index or 0]):
        original_model = deepcopy(base)
        load_adapter_state(original_model, load_file(str(ORIGINAL), device="cpu"))
        original_model.eval().requires_grad_(False)

    class OriginalReference:
        def __init__(self, model, encoder):
            self.model, self.encoder = model, encoder

        @torch.no_grad()
        def routed_forward(self, context):
            z, t, ctx, mask, uctx, umask, strength = self.encoder.unpack(context)
            batch = len(z)
            branches = [module for module in self.model.modules() if hasattr(module, "multiplier")]
            previous = [module.multiplier for module in branches]
            try:
                for module in branches:
                    module.multiplier = strength
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    output = self.model(torch.cat((z, z)), torch.cat((t, t)),
                                        torch.cat((ctx, uctx.expand(batch, -1, -1))),
                                        torch.cat((mask, umask.expand(batch, -1))))
                conditional, unconditional = output.float().chunk(2)
                velocity = unconditional + 3 * (conditional - unconditional)
                return velocity - self.encoder.teacher_velocity(context)
            finally:
                for module, multiplier in zip(branches, previous):
                    module.multiplier = multiplier

    original_reference = OriginalReference(original_model, loop.policy.encoder)
    original_digest = state_digest(original_model.state_dict())
    reference_monitor = GameProgressMonitor(loop, data, fixed, args.output / "reference-probes")
    reference_point = dict(step=6400, probes={})
    reference_native_before = state_digest(checkpoint(loop))
    with torch.no_grad():
        for name, (contexts, panels) in reference_monitor.pools.items():
            values = []
            for start in range(0, len(contexts), 4):
                context = contexts[start:start + 4]
                residual = original_reference.routed_forward(context)
                condition = loop.policy.encoder.condition(context).repeat(4, 1)
                real = (.125 * panels[:, start:start + len(context)]).flatten(0, 1)
                fake = real + (patchify(residual) / fixed.scale).repeat(4, 1, 1)
                gap = fixed(real, condition) - fixed(fake, condition)
                values.extend(torch.nn.functional.softplus(gap).reshape(4, len(context)).mean(0).cpu().tolist())
            reference_point["probes"][name] = dict(contexts=len(contexts), clean=dict(frozen_start_D=sum(values) / len(values)))
    if reference_native_before != state_digest(checkpoint(loop)) or original_digest != state_digest(original_model.state_dict()):
        raise RuntimeError("original reference probes changed frozen or native state")
    write(args.output / "original-reference-progress.json", dict(label="Original ordinary LoRA, trained6400, reference to beat",
          judge="common qualified particle V2 step6400 critic", reference_update=6400,
          comparison="constant historical quality target; new continuation ends6400+fixed budget",
          evaluation_only=True, points=[reference_point], original_sha256=ORIGINAL_SHA))
    del reference_monitor
    common_final = None
    results = {}
    endpoint_states = {}
    for arm in ARMS:
        arm_output = args.output / arm
        arm_output.mkdir()
        restore(loop, saved)
        if state_digest(checkpoint(loop)) != initial:
            raise RuntimeError("arm does not begin from the same native state")
        rows, trust_rows = [], []
        monitor = GameProgressMonitor(loop, data, fixed, arm_output)
        monitor.dv12_state = common_dv12.clone()
        emit(event="game_progress", arm=arm, **monitor.evaluate(loop, rows))
        expected_data = torch.Generator().set_state(saved["data_rng"].cpu())
        expected_gaussian = torch.Generator(device=device).set_state(saved["paired_noise_rng"].cpu())
        manager = bank_game_trust(loop) if arm == FORMULATION else nullcontext(None)
        with manager as helper:
            first_rows, first_trust = [], []
            for _ in range(2):
                if helper is not None:
                    helper.begin_update()
                first_rows.append(training_update(loop))
                if helper is not None:
                    first_trust.append(deepcopy(helper.last_record))
            replay_digest = state_digest(checkpoint(loop))
        restore(loop, saved)
        manager = bank_game_trust(loop) if arm == FORMULATION else nullcontext(None)
        started = time.perf_counter()
        with manager as helper, (arm_output / "train.jsonl").open("w") as trace:
            for offset in range(1, args.updates + 1):
                step = 6400 + offset
                hold = step % 5 == 0
                pool = data["holds" if hold else "fit"]["context"]
                indices = torch.randint(len(pool), (4,), generator=expected_data)
                panels = [torch.randn((4, 256, 16), device=device, generator=expected_gaussian) for _ in range(2)]
                if helper is not None:
                    helper.begin_update()
                row = training_update(loop)
                expected = dict(step=step, hold=hold, batch_indices=indices.tolist(),
                                base_noise_sums=[float(value.sum()) for value in panels],
                                paired_rng_digest=_rng_digest(expected_gaussian))
                if any(row[key] != value for key, value in expected.items()):
                    raise RuntimeError("training contexts/paired Gaussian stream changed")
                rows.append(row)
                if helper is not None:
                    trust_rows.append(deepcopy(helper.last_record))
                trace.write(json.dumps(finite_json(dict(row, bank_trust=None if helper is None else helper.last_record)),
                                       allow_nan=False) + "\n")
                if offset == 2:
                    if rows != first_rows or trust_rows != first_trust or state_digest(checkpoint(loop)) != replay_digest:
                        raise RuntimeError("experimental/native exact two-update resume failed")
                    write(arm_output / "replay-review.json", dict(initial_native_state_exact=True,
                                                                  two_update_replay_exact=True))
                    emit(event="resume_passed", arm=arm, step=step)
                if arm == "native" and offset == 5 and state_digest(checkpoint(loop)) != five_step_reference["final_native_digest"]:
                    raise RuntimeError("unchanged V2 native control differs from its qualified five-step reference")
                if offset % 16 == 0 or offset == args.updates:
                    trace.flush()
                    seconds = time.perf_counter() - started
                    status = dict(phase="training", arm=arm, step=step, steps=plan["final_step"], seconds=seconds,
                                  rolling=rolling_losses(rows), bank_trust=None if helper is None else helper.last_record)
                    write(args.output / "status.json", status)
                    write(arm_output / "status.json", status)
                    emit(event="training", **status)
                if offset % args.probe_every == 0 or offset == args.updates:
                    emit(event="game_progress", arm=arm, **monitor.evaluate(loop, rows))
        final_state = checkpoint(loop)
        if frozen_digest(loop) != frozen:
            raise RuntimeError("frozen host or teacher changed")
        if not torch.equal(expected_data.get_state(), loop.data_rng.get_state()) or not torch.equal(
                expected_gaussian.get_state(), loop.paired_noise_rng.get_state()):
            raise RuntimeError("final owned application streams differ")
        if common_final is None:
            common_final = deepcopy(loop.policy.D).eval().requires_grad_(False)
            original_before = state_digest(checkpoint(loop))
            original_metrics = evaluate_final(original_reference, data, fixed, common_final, device)
            if original_before != state_digest(checkpoint(loop)) or original_digest != state_digest(original_model.state_dict()):
                raise RuntimeError("original final evaluation changed frozen or native state")
            write(args.output / "evaluation-original.json", original_metrics)
        endpoint_states[arm] = state_digest(final_state)
        wrapper = dict(schema="supra_game_trust_experimental_checkpoint_v1", formulation=arm,
                       plan=plan, native=final_state)
        torch.save(wrapper, arm_output / "final.pt")
        before_evaluation = state_digest(checkpoint(loop))
        metrics = evaluate_final(loop.policy.served_model(), data, fixed, common_final, device)
        if before_evaluation != state_digest(checkpoint(loop)):
            raise RuntimeError("final evaluation changed native training state")
        write(arm_output / "evaluation.json", metrics)
        results[arm] = dict(final_native_digest=endpoint_states[arm], final_checkpoint_sha256=sha(arm_output / "final.pt"),
                           data_rng_digest=_rng_digest(loop.data_rng), paired_rng_digest=_rng_digest(loop.paired_noise_rng),
                           dv12_rng_digest=_rng_digest(loop.policy.noise_generator),
                           trust_rows=trust_rows, training_seconds=time.perf_counter() - started,
                           metrics={name: {key: value for key, value in pool.items() if key != "records"}
                                    for name, pool in metrics.items()})
        write(arm_output / "receipt.json", results[arm])
        write(arm_output / "status.json", dict(phase="complete", arm=arm, step=plan["final_step"], steps=plan["final_step"]))
        fraction_counts = {str(fraction): sum(record["selected_fraction"] == fraction for record in trust_rows)
                           for fraction in FRACTIONS} if trust_rows else {}
        emit(event="arm_complete", arm=arm, final_native_digest=endpoint_states[arm],
             selected_bank_fractions=fraction_counts, metrics=results[arm]["metrics"])
        del final_state, wrapper, monitor
        gc.collect()
    if any(sha(ROOT / name) != value for name, value in app_hashes.items()) or any(
            sha(imported.parent / name) != value for name, value in pg_hashes.items()) or any(
            sha(inputs[name]) != value for name, value in input_hashes.items()):
        raise RuntimeError("source or immutable inputs changed during the test")
    if any(results[ARMS[0]][key] != results[ARMS[1]][key] for key in (
            "data_rng_digest", "paired_rng_digest", "dv12_rng_digest")):
        raise RuntimeError("the two arms did not preserve matching owned random streams")
    deltas = {pool: {judge: results[ARMS[1]]["metrics"][pool][judge]["g_game"]
                    - results[ARMS[0]]["metrics"][pool][judge]["g_game"]
                    for judge in ("fixed_start_D", "arm_final_D")}
              for pool in ("fit", "test", "holds", "preservation")}
    reference_gaps = {arm: {pool: {judge: results[arm]["metrics"][pool][judge]["g_game"]
                                  - original_metrics[pool][judge]["g_game"]
                                  for judge in ("fixed_start_D", "arm_final_D")}
                           for pool in ("fit", "test", "holds", "preservation")} for arm in ARMS}
    receipt = dict(plan=plan, results=results, game_deltas_candidate_minus_native=deltas,
                   game_gaps_to_original_6400_reference=reference_gaps,
                   judges={"fixed_start_D": "common qualified particle V2 step6400 critic",
                           "arm_final_D": "common native-control final critic for both endpoints"},
                   initial_native_state_exact=True, two_update_replay_exact=True,
                   frozen_unchanged=True, evaluation_state_unchanged=True, source_and_inputs_unchanged=True,
                   matched_owned_streams=True, seconds=time.perf_counter() - began)
    write(args.output / "receipt.json", receipt)
    write(args.output / "status.json", dict(phase="complete", step=plan["final_step"], steps=plan["final_step"]))
    emit(event="complete", game_deltas_candidate_minus_native=deltas, seconds=receipt["seconds"])


if __name__ == "__main__":
    main()
