#!/usr/bin/env python3
"""Final-only evaluation of full ParticleGAN Supra versus the original LoRA.

All output metrics are reporting only. This process performs no training,
structural decisions, stopping decisions or checkpoint selection.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import yaml
from PIL import Image, ImageDraw
from PIL.PngImagePlugin import PngInfo
from safetensors import safe_open

from supra.particle_pilot import checkpoint, restore, state_digest
from supra.particle_adapter import NONLINEAR_V1
from supra.particle_training import make_training_loop, raw_velocity
from supra.particle_training_data import SOURCE_COLUMN, TARGET_COLUMN, STRENGTH_COLUMN
from supra.particle_export import export_served_adapter, load_particle_adapter
from supra.particle_game import patchify
from supra.runtime import MODEL_REV, T5_REV, VAE_REV, SupraRuntime

ORIGINAL_ROOT = Path("/ml2/hypergan/supra-concept-sliders")


def emit(value):
    print(json.dumps(value, sort_keys=True), flush=True)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def tensor_sha256(value):
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def pack(z, time_value, source_id, target_id, strength=1.):
    extra = z.new_tensor([[time_value, source_id, target_id, strength]]).expand(len(z), -1)
    return torch.cat((z.flatten(1), extra), dim=1)


def comparison_horizon(run, completed_steps, baseline_path, *, expected_steps=None):
    """Require the final declared horizon and an equally trained baseline."""
    declared = run.get("fixed_updates")
    for label, value in (("checkpoint step", completed_steps), ("declared horizon", declared),
                         ("explicit horizon", expected_steps)):
        if value is not None and (type(value) is not int or value <= 0):
            raise ValueError(f"{label} must be a positive integer")
    if declared is not None and expected_steps is not None and declared != expected_steps:
        raise ValueError("explicit horizon differs from the declared training horizon")
    horizon = declared if declared is not None else expected_steps
    if horizon is None:
        raise ValueError("a declared run horizon or --expected-steps is required")
    if completed_steps != horizon:
        raise ValueError(f"comparison expects fixed step {horizon}, got {completed_steps}")
    with safe_open(str(baseline_path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    try:
        baseline_steps = json.loads(metadata["step"])
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("original baseline must declare its training step in adapter metadata") from error
    if type(baseline_steps) is not int or baseline_steps != horizon:
        raise ValueError(f"original baseline step {baseline_steps} differs from the comparison horizon {horizon}")
    return horizon


def render_labels(horizon):
    return ("Off / strength 0", f"Original LoRA / {horizon}",
            f"New ParticleGAN / {horizon}", "Frozen positive prompt")


def baseline_probe_reproduction(receipt_path, current_probes):
    """A receipt without original probes cannot certify their reproduction."""
    receipt_path = Path(receipt_path)
    report = dict(available=False, receipt_path=str(receipt_path))
    if not receipt_path.is_file():
        return {**report, "reason": "original baseline has no velocity-probe receipt"}
    report["receipt_sha256"] = file_sha256(receipt_path)
    published = json.loads(receipt_path.read_text()).get("velocity_probes")
    if not published:
        return {**report, "reason": "original receipt has no published velocity_probes; common probes are evaluated separately"}
    if len(published) != len(current_probes):
        return {**report, "reason": "original and current probe counts differ"}
    key = lambda row: (row["name"], round(row["t"], 7))
    current = {key(row): row for row in current_probes}
    if len(current) != len(current_probes) or {key(row) for row in published} != set(current):
        return {**report, "reason": "original and current probe identities differ"}
    differences = []
    for row in published:
        actual = current[key(row)]
        for name, value in row.items():
            if isinstance(value, (int, float)):
                if name not in actual or not isinstance(actual[name], (int, float)):
                    return {**report, "reason": "original and current numeric probe fields differ"}
                differences.append(abs(value - actual[name]))
    if not differences:
        return {**report, "reason": "original receipt contains no comparable numeric probe fields"}
    return {**report, "available": True, "compared_records": len(published),
            "maximum_absolute_metric_difference": max(differences)}


def summarize(rows, *, preservation=False):
    if not rows:
        raise ValueError("evaluation pools must contain records")
    mean = lambda key: sum(row[key] for row in rows) / len(rows)
    result = dict(contexts=len(rows), mse=mean("mse"), rmse=mean("mse") ** .5,
                  base_power=mean("base_power"), relative_rms=(mean("mse") / max(mean("base_power"), 1e-12)) ** .5)
    if not preservation:
        result.update(base_gap_mse=mean("base_gap_mse"),
                      velocity_ratio=mean("mse") / max(mean("base_gap_mse"), 1e-12),
                      direction_cosine=mean("direction_cosine"))
    groups = {}
    for row in rows:
        groups.setdefault(row["prompt"], []).append(row)
    result["per_subject"] = {
        prompt: dict(contexts=len(group), mse=sum(row["mse"] for row in group) / len(group),
                     **({} if preservation else {
                         "velocity_ratio": sum(row["mse"] for row in group) / max(sum(row["base_gap_mse"] for row in group), 1e-12)}))
        for prompt, group in groups.items()
    }
    result["records"] = rows
    return result


@torch.no_grad()
def evaluate_pool(pool, served, runtime, *, preservation=False, batch_size=4, game_loss=None):
    records = {"old_lora": [], "new_particlegan": []}
    draws = 4
    panel_totals = {arm: [0.] * draws for arm in records}
    panel_stream = torch.Generator(device="cpu").manual_seed(72)
    noise_digest = hashlib.sha256()
    for start in range(0, len(pool["context"]), batch_size):
        context = pool["context"][start:start + batch_size].to(runtime.device)
        prompts = pool["prompts"][start:start + batch_size]
        target = served.encoder.teacher_velocity(context)
        base_context = context.clone()
        base_context[:, TARGET_COLUMN] = base_context[:, SOURCE_COLUMN]
        base = served.encoder.teacher_velocity(base_context)
        predictions = {
            "old_lora": runtime.velocity(prompts, context[:, :4096].reshape(-1, 4, 32, 32),
                                         context[:, 4096], scale=1., cfg=3.),
            "new_particlegan": raw_velocity(served, context),
        }
        if game_loss is not None:
            condition = served.encoder.condition(context)
            residuals = {arm: patchify(prediction - target).float() / served.critic.scale
                         for arm, prediction in predictions.items()}
            for draw in range(draws):
                noise = torch.randn((len(context), 256, 16), generator=panel_stream)
                noise_digest.update(noise.numpy().tobytes())
                real = served.output_sigma * noise.to(runtime.device)
                real_logits = served.critic(real, condition)
                for arm, residual in residuals.items():
                    value = game_loss.g_loss(served.critic(real + residual, condition), real_logits)
                    panel_totals[arm][draw] += float(value) * len(context)
        for arm, prediction in predictions.items():
            if prediction.shape != target.shape or not bool(torch.isfinite(prediction).all()):
                raise RuntimeError(f"invalid {arm} evaluation velocity")
            mse = (prediction - target).square().flatten(1).mean(1)
            power = base.square().flatten(1).mean(1)
            gap = (base - target).square().flatten(1).mean(1)
            cosine = torch.nn.functional.cosine_similarity((prediction - base).flatten(1),
                                                            (target - base).flatten(1), dim=1)
            for index, prompt in enumerate(prompts):
                records[arm].append(dict(index=start + index, prompt=prompt, t=float(context[index, 4096]),
                                         mse=float(mse[index]), base_power=float(power[index]),
                                         base_gap_mse=float(gap[index]), direction_cosine=float(cosine[index])))
        if (start // batch_size + 1) % 10 == 0 or start + batch_size >= len(pool["context"]):
            emit(dict(event="velocity_progress", preservation=preservation,
                      completed=min(start + batch_size, len(pool["context"])), total=len(pool["context"])))
    result = {arm: summarize(rows, preservation=preservation) for arm, rows in records.items()}
    if game_loss is not None:
        for arm, report in result.items():
            values = [total / len(pool["context"]) for total in panel_totals[arm]]
            report["common_new_critic"] = dict(generator_loss=sum(values) / draws, per_draw_generator_loss=values,
                                               draws=draws, contexts=len(pool["context"]),
                                               output_sigma=float(served.output_sigma),
                                               critic_state_sha256=state_digest(served.critic.state_dict()),
                                               paired_noise_sha256=noise_digest.hexdigest(),
                                               noise_rng="private CPU generator 72",
                                               law="clean paired residual; native RpGAN under one fixed new critic",
                                               evaluation_only=True,
                                               limitation="This critic was trained with the new model. The old LoRA has no critic; this is not independent quality evidence or an evaluation ranking score.")
    return result


@torch.no_grad()
def new_trajectory(served, runtime, prompt_ids, prompt, target_prompt, seed, *, strength=1., steps=50):
    z = runtime.noise(seed)
    initial_hash = tensor_sha256(z)
    for index in range(steps):
        context = pack(z, index / steps, prompt_ids[prompt], prompt_ids[target_prompt], strength)
        velocity = raw_velocity(served, context)
        if not bool(torch.isfinite(velocity).all()):
            raise RuntimeError("nonfinite new served trajectory")
        z = z + velocity / steps
        if (index + 1) % 10 == 0:
            emit(dict(event="trajectory_progress", arm="new_particlegan", seed=seed,
                      prompt=prompt, step=index + 1, steps=steps))
    return z, initial_hash


@torch.no_grad()
def evaluate_endpoints(endpoints, served, runtime, prompt_ids, positive_by_prompt):
    result = {"old_lora": [], "new_particlegan": []}
    for index, row in enumerate(endpoints):
        prompt, seed = row["prompt"], row["seed"]
        target_prompt = positive_by_prompt[prompt]
        target, base = row["target"].to(runtime.device), row["neutral"].to(runtime.device)
        old = runtime.trajectory(prompt, seed, 50, 3., scale=1.)
        new, initial_hash = new_trajectory(served, runtime, prompt_ids, prompt, target_prompt, seed)
        gap = float((base - target).square().mean())
        for arm, endpoint in (("old_lora", old), ("new_particlegan", new)):
            mse = float((endpoint - target).square().mean())
            result[arm].append(dict(name=row["name"], prompt=prompt, seed=seed, mse=mse,
                                    base_gap_mse=gap, ratio=mse / max(gap, 1e-12),
                                    endpoint_sha256=tensor_sha256(endpoint), initial_latent_sha256=initial_hash))
        emit(dict(event="endpoint_complete", completed=index + 1, total=len(endpoints),
                  name=row["name"], seed=seed,
                  old_ratio=result["old_lora"][-1]["ratio"], new_ratio=result["new_particlegan"][-1]["ratio"]))
    return {arm: dict(trajectories=len(rows), mean_endpoint_ratio=sum(r["ratio"] for r in rows) / len(rows),
                      mse=sum(r["mse"] for r in rows) / len(rows), records=rows)
            for arm, rows in result.items()}


@torch.no_grad()
def published_probes(config, served, runtime, prompt_ids, positive_by_prompt):
    cases = [("trained_knight", config["rows"][0]["neutral"], config["rows"][0]["positive"]),
             ("heldout_bridge", config["verification"][2]["prompt"], config["verification"][2]["teacher"]),
             ("fruit_control", config["verification"][3]["prompt"], None)]
    result = {"old_lora": [], "new_particlegan": []}
    for name, prompt, target_prompt in cases:
        _, states = runtime.trajectory(prompt, 29001, 50, 3., keep=True)
        contexts = torch.cat([pack(states[index].to(runtime.device), index / 50,
                                   prompt_ids[prompt], prompt_ids[target_prompt or prompt])
                              for index in (0, 10, 25, 40)])
        pool = dict(context=contexts.cpu(), prompts=[prompt] * 4)
        measurements = evaluate_pool(pool, served, runtime, preservation=target_prompt is None, batch_size=1)
        for arm, report in measurements.items():
            for row in report["records"]:
                z = contexts[row["index"]:row["index"] + 1]
                base_context = z.clone()
                base_context[:, TARGET_COLUMN] = base_context[:, SOURCE_COLUMN]
                base = served.encoder.teacher_velocity(base_context)
                prediction = (raw_velocity(served, z) if arm == "new_particlegan" else
                              runtime.velocity(prompt, z[:, :4096].reshape(-1, 4, 32, 32), z[:, 4096], scale=1., cfg=3.))
                item = dict(name=name, t=row["t"], relative_drift=float((prediction - base).norm() / base.norm().clamp_min(1e-8)))
                if target_prompt:
                    item.update(gap_mse=row["base_gap_mse"], trained_mse=row["mse"],
                                gap_fraction=row["mse"] / max(row["base_gap_mse"], 1e-12),
                                direction_cosine=row["direction_cosine"])
                result[arm].append(item)
        emit(dict(event="published_probe_complete", name=name))
    return result


@torch.no_grad()
def decode(runtime, endpoint):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pixels = runtime.vae.decode(endpoint / .18215).sample
    pixels = ((pixels.float().clamp(-1, 1) + 1) * 127.5).round().byte()[0]
    return Image.fromarray(pixels.permute(1, 2, 0).cpu().numpy())


@torch.no_grad()
def export_and_verify(served, runtime, context, path, *, metadata):
    """Check the lean adapter on real native BF16 before loading the old LoRA."""
    path = Path(path)
    sidecar = path.with_suffix(".json")
    if path.exists() or sidecar.exists():
        raise ValueError("particle export already exists; preserve it and choose another evaluation output parent")
    receipt = export_served_adapter(served, path, extra_metadata=metadata)
    # Reconstruction makes fresh modules before strictly loading their tensors.
    # Keep those constructor draws out of the training RNG stored in the loop.
    devices = [runtime.device.index] if runtime.device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        loaded = load_particle_adapter(runtime.model, path, device=runtime.device)
        inputs = served.encoder.unpack(context)
        reference = raw_velocity(served, context)
        reloaded = loaded.velocity(*inputs[:6], strength=inputs[6], cfg=3.)
        receipt["reload_exact"] = torch.equal(reference, reloaded)
        receipt["reload_max_absolute_difference"] = float((reference - reloaded).abs().max())
        zero_context = context.clone()
        zero_context[:, STRENGTH_COLUMN] = 0.
        zero_context[:, TARGET_COLUMN] = zero_context[:, SOURCE_COLUMN]
        inputs = served.encoder.unpack(zero_context)
        zero = loaded.velocity(*inputs[:6], strength=inputs[6], cfg=3.)
        receipt["strength_zero_exact"] = torch.equal(zero, served.encoder.teacher_velocity(zero_context))
        del loaded
    if not receipt["reload_exact"] or not receipt["strength_zero_exact"]:
        raise RuntimeError(f"native particle adapter export failed exact checks: {receipt}")
    receipt["sha256"] = file_sha256(path)
    receipt["contexts_checked"] = len(context)
    receipt["forward_precision"] = "native BF16 autocast, FP32 frozen weights and particle projections"
    write_json(sidecar, receipt)
    emit(dict(event="export_verified", path=str(path), bytes=receipt["bytes"], sha256=receipt["sha256"],
              tensors=receipt["tensors"], reload_exact=True, strength_zero_exact=True))
    gc.collect()
    return receipt


@torch.no_grad()
def render_all(config, served, runtime, prompt_ids, positive_by_prompt, output, provenance):
    folder = output / "samples"
    folder.mkdir()
    grid = Image.new("RGB", (1024, 50 + len(config["verification"]) * 2 * 282), "#151921")
    draw = ImageDraw.Draw(grid)
    labels = render_labels(provenance["declared_updates"])
    for column, label in enumerate(labels):
        draw.text((column * 256 + 8, 18), label, fill="white")
    samples = []
    for row_index, row in enumerate(config["verification"]):
        prompt = row["prompt"]
        target_prompt = row.get("teacher", positive_by_prompt.get(prompt, prompt))
        for seed_index, seed in enumerate((42, 1234)):
            y = 50 + (row_index * 2 + seed_index) * 282
            draw.text((8, y + 5), f"{row['name']} | seed {seed}", fill="#77d8c1")
            for column, arm in enumerate(("off", "old_lora", "new_particlegan", "positive_teacher")):
                started = time.perf_counter()
                if arm == "new_particlegan":
                    endpoint, initial_hash = new_trajectory(served, runtime, prompt_ids, prompt, target_prompt, seed)
                else:
                    endpoint = runtime.trajectory(target_prompt if arm == "positive_teacher" else prompt,
                                                  seed, 50, 3., scale=1. if arm == "old_lora" else 0.)
                    initial_hash = tensor_sha256(runtime.noise(seed))
                image = decode(runtime, endpoint)
                filename = f"{row['name']}_seed{seed}_{arm}.png"
                sample = dict(file=filename, name=row["name"], arm=arm,
                              prompt=target_prompt if arm == "positive_teacher" else prompt, seed=seed,
                              strength=1. if arm in ("old_lora", "new_particlegan") else 0.,
                              initial_latent_sha256=initial_hash, endpoint_sha256=tensor_sha256(endpoint),
                              steps=50, cfg=3., seconds=time.perf_counter() - started)
                info = PngInfo()
                info.add_text("supra_comparison", json.dumps({**provenance, **sample}, sort_keys=True))
                image.save(folder / filename, pnginfo=info)
                sample["image_sha256"] = file_sha256(folder / filename)
                samples.append(sample)
                grid.paste(image, (column * 256, y + 26))
                emit(dict(event="render_complete", completed=len(samples), total=32, **sample))
    grid.save(output / "grid.png")
    groups = {(sample["name"], sample["seed"]): set() for sample in samples}
    for sample in samples:
        groups[sample["name"], sample["seed"]].add(sample["initial_latent_sha256"])
    if not all(len(hashes) == 1 for hashes in groups.values()):
        raise RuntimeError("comparison renders did not share original initial latents")
    write_json(folder / "metadata.json", dict(**provenance, samples=samples, matched_initial_latents=True))
    return dict(images=len(samples), matched_initial_latents=True, grid=str(output / "grid.png"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=ORIGINAL_ROOT / "outputs/final-boss-supra-1600/final-boss-supra.safetensors")
    parser.add_argument("--prompts", type=Path, default=ROOT / "configs/supra/prompts-final-boss.yaml")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--expected-steps", type=int,
                        help="Expected fixed horizon when no run receipt is available; must agree with a declared receipt")
    args = parser.parse_args()
    for path in (args.data, args.checkpoint, args.baseline, args.prompts):
        if not path.is_file():
            parser.error(f"missing input {path}")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("evaluation output must be empty to preserve existing evidence")
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    config = yaml.safe_load(args.prompts.read_text())
    data = torch.load(args.data, map_location="cpu", weights_only=False)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    run_path = args.checkpoint.parent / "run.json"
    run = {}
    if run_path.is_file():
        run = json.loads(run_path.read_text())
        if (file_sha256(args.baseline) != run["original_baseline_sha256"]
                or file_sha256(args.prompts) != run["prompts_sha256"]):
            raise ValueError("baseline or prompt recipe differs from the declared training comparison")
    horizon = comparison_horizon(run, state["policy"]["completed_steps"], args.baseline,
                                 expected_steps=args.expected_steps)
    if data["test"] is None or data["preservation"] is None or not data["endpoints"]:
        raise ValueError("final evaluation requires validation and endpoint records")
    torch.set_num_threads(8)
    runtime = SupraRuntime(device=args.device, rank=16, allow_hub=False)
    # The particle model and its teacher must start from the pure base. Load the
    # old release only after creating/restoring those independent owners.
    runtime.model.requires_grad_(False)
    runtime.text_encoder.to("cpu")
    loop = make_training_loop(runtime.model, data, device=args.device,
                              probe_interval=state["config"]["probe_interval"],
                              branch_lr=state["config"]["branch_lr"],
                              architecture=state["config"].get("architecture", NONLINEAR_V1))
    restore(loop, state)
    before_digest = state_digest(checkpoint(loop))
    del state
    gc.collect()
    served = loop.policy.served_model()
    if served.completed_steps != horizon:
        raise ValueError("restored serving snapshot differs from the declared final horizon")
    prompt_ids = {prompt: index for index, prompt in enumerate(data["prompts"])}
    runtime.cache = {prompt: (data["text_contexts"][index:index + 1].to(runtime.device),
                              data["text_masks"][index:index + 1].to(runtime.device))
                     for prompt, index in prompt_ids.items()}
    positive_by_prompt = {row["neutral"]: row["positive"] for row in config["rows"]}
    positive_by_prompt.update({row["prompt"]: row["teacher"] for row in config["verification"] if "teacher" in row})
    export_receipt = export_and_verify(
        served, runtime, data["test"]["context"][:4].to(runtime.device),
        args.output.parent / "final-boss-particlegan.safetensors",
        metadata=dict(training=run, checkpoint_sha256=file_sha256(args.checkpoint), data_sha256=file_sha256(args.data)))
    runtime.load(args.baseline)
    provenance = dict(schema="supra-full-particlegan-vs-original-lora-v1",
                      checkpoint_sha256=file_sha256(args.checkpoint), data_sha256=file_sha256(args.data),
                      baseline_sha256=file_sha256(args.baseline), prompts_sha256=file_sha256(args.prompts),
                      model_revision=MODEL_REV, text_encoder_revision=T5_REV, vae_revision=VAE_REV,
                      completed_steps=served.completed_steps, declared_updates=horizon, baseline_completed_steps=horizon,
                      served_source=served.source,
                      source_sha256={name: file_sha256(ROOT / name) for name in
                                     ("scripts/evaluate_e22_supra_full.py", "supra/particle_training.py",
                                      "supra/particle_training_data.py", "supra/particle_adapter.py", "supra/particle_export.py", "supra/runtime.py")},
                      evaluation_only=True, live_target_batch_size=4,
                      task=loop.config, limitation="Different adapter architectures, initialization and game; equal update budget, not an isolated optimizer comparison or image-quality proof.")
    write_json(args.output / "status.json", dict(phase="velocity", **provenance))
    emit(dict(event="evaluation_started", **provenance))
    report = dict(**provenance)
    report["adapter_export"] = export_receipt
    native_loss = loop.policy.recipe.make_loss()
    report["validation"] = evaluate_pool(data["test"], served, runtime, game_loss=native_loss)
    report["preservation"] = evaluate_pool(data["preservation"], served, runtime, preservation=True, game_loss=native_loss)
    zero_context = data["test"]["context"][:4].to(runtime.device).clone()
    zero_context[:, STRENGTH_COLUMN] = 0.
    zero_context[:, TARGET_COLUMN] = zero_context[:, SOURCE_COLUMN]
    report["strength_zero_exact"] = torch.equal(raw_velocity(served, zero_context), served.encoder.teacher_velocity(zero_context))
    if not report["strength_zero_exact"]:
        raise RuntimeError("full particle model strength-zero replay differed from pure base")
    write_json(args.output / "status.json", dict(phase="endpoints", **provenance))
    report["endpoints"] = evaluate_endpoints(data["endpoints"], served, runtime, prompt_ids, positive_by_prompt)
    report["published_probes"] = published_probes(config, served, runtime, prompt_ids, positive_by_prompt)
    report["published_baseline_reproduction"] = baseline_probe_reproduction(
        args.baseline.parent / "validation.json", report["published_probes"]["old_lora"])
    write_json(args.output / "comparison.json", report)
    write_json(args.output / "status.json", dict(phase="rendering", **provenance))
    report["renders"] = render_all(config, served, runtime, prompt_ids, positive_by_prompt, args.output, provenance)
    report["training_state_unchanged"] = before_digest == state_digest(checkpoint(loop))
    if not report["training_state_unchanged"]:
        raise RuntimeError("evaluation changed native training checkpoint state")
    report["evaluation_seconds"] = time.perf_counter() - started
    write_json(args.output / "comparison.json", report)
    write_json(args.output / "status.json", dict(phase="complete", seconds=report["evaluation_seconds"]))
    emit(dict(event="complete", output=str(args.output), validation={arm: r["velocity_ratio"] for arm, r in report["validation"].items()},
              endpoints={arm: r["mean_endpoint_ratio"] for arm, r in report["endpoints"].items()},
              strength_zero_exact=True, training_state_unchanged=True))


if __name__ == "__main__":
    main()
