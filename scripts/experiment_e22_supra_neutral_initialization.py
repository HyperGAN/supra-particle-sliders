#!/usr/bin/env python3
"""Two fresh editing-only Supra arms: sampled H/b versus neutral H/b.

Fixed external horizons and fixed learned critics. Reporting never enters the
optimizer, structural guards, checkpoint selection, or stopping decisions.
"""
import argparse
from copy import deepcopy
import gc
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT.parent / "supra-e22-verification"
PIN = "6ec7e5788e14ea15ddc3e16ac71110458108b6a6"
ARMS = ("sampled_control", "sampled_hb_neutral")
STEPS = 6400
ENDPOINTS = (5120, 6400)
sys.path.insert(0, str(ROOT / "scripts"))
from experiment_e22_supra_particle_gated import sha, write, emit, diagnostic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--particlegan-root", type=Path, default=ROOT.parent / "ParticleGAN-supra-neutral-develop")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/e22-supra-neutral-initialization-6400")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("preserve existing artifacts; use a fresh output directory")
    pg = args.particlegan_root.resolve()
    if subprocess.check_output(["git", "-C", str(pg), "rev-parse", "HEAD"], text=True).strip() != PIN:
        parser.error("native revision differs from the declared protocol")
    if subprocess.check_output(["git", "-C", str(pg), "status", "--porcelain", "--", "particlegan"], text=True).strip():
        parser.error("native package source is dirty")
    sys.path.insert(0, str(pg))
    sys.path.insert(1, str(ROOT))
    import torch
    import torch.nn.functional as F
    import particlegan
    from safetensors.torch import load_file
    from supra.runtime import TARGETS, MODEL_SOURCE, model_module, load_adapter_state
    from supra.particle_adapter import GATED_PARTICLE_V3
    from supra.particle_export import load_particle_adapter, export_served_adapter
    from supra.particle_game import ConditionalTokenCritic, patchify, update
    from supra.particle_pilot import checkpoint, restore, state_digest, frozen_digest
    from supra.particle_training import SHARED_ROUTED_PROFILE, make_training_loop
    from monitor_e22_supra_particle_convergence import GameProgressMonitor, evaluation_modes, rolling_losses
    from evaluate_e22_supra_particle_contribution import mass_only_routing

    if Path(particlegan.__file__).resolve().parent.parent != pg:
        raise RuntimeError("wrong native package import")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("full BF16 Supra requires CUDA")
    torch.cuda.set_device(device)
    torch.set_num_threads(8)
    args.output.mkdir(parents=True)
    started = time.monotonic()
    checks, loops, trace_files = {}, {}, {}
    charged = {arm: 0. for arm in ARMS}
    card = ROOT / "docs/e22_supra_neutral_initialization_protocol.json"
    protocol = json.loads(card.read_text())
    inputs = {
        "data": ARCHIVE / "outputs/e22-particle-v2-6400/data.pt",
        "teacher": ARCHIVE / "outputs/e22-particle-v2-6400/final.pt",
        "D1856": ARCHIVE / "outputs/e22-particle-noise-update-256/latest/final.pt",
        "D6400": ARCHIVE / "outputs/e22-particle-v2-6400/final.pt",
        "original6400": ROOT.parent / "supra-concept-sliders/outputs/final-boss-supra-converged/checkpoint-06400/final-boss-supra.safetensors",
        "historical_particle_v2": ARCHIVE / "outputs/e22-particle-v2-6400/final.safetensors",
        "historical_particle_v3": ARCHIVE / "outputs/e22-particle-gated-v3-editing-only-6400/final.safetensors",
    }
    app_files = sorted((ROOT / "supra").glob("particle*.py")) + [ROOT / "supra/runtime.py", Path(__file__), card,
        ROOT / "scripts/experiment_e22_supra_particle_gated.py", ROOT / "scripts/monitor_e22_supra_particle_convergence.py",
        ROOT / "scripts/evaluate_e22_supra_particle_contribution.py"]
    sources = {str(path.relative_to(ROOT)): sha(path) for path in app_files}
    native_files = sorted((pg / "particlegan").rglob("*.py"))
    native_sources = {str(path.relative_to(pg)): sha(path) for path in native_files}
    input_hashes = {key: sha(path) for key, path in inputs.items()}
    aliases = dict(data="data", teacher="frozen_teacher", D1856="common_D1856", D6400="common_V2_D6400",
        original6400="historical_ordinary_6400", historical_particle_v2="historical_V2_6400_export",
        historical_particle_v3="historical_V3_6400_export")
    for name, declared in protocol["inputs"].items():
        if sha(Path(declared["path"])) != declared["sha256"]:
            raise ValueError("frozen protocol input changed: " + name)
    for alias, name in aliases.items():
        if str(inputs[alias].resolve()) != protocol["inputs"][name]["path"] or input_hashes[alias] != protocol["inputs"][name]["sha256"]:
            raise ValueError("runner input differs from frozen protocol: " + alias)
    native_hash = hashlib.sha256()
    for name in sorted(native_sources):
        native_hash.update(name.encode())
        native_hash.update(json.dumps(native_sources[name], sort_keys=True).encode())
    if native_hash.hexdigest() != protocol["particlegan"]["python_source_digest"] or sha(MODEL_SOURCE) != protocol["inputs"]["backend_model_source"]["sha256"]:
        raise ValueError("native/backend source differs from frozen protocol")
    plan = dict(schema="supra_sampled_hb_neutral_matched_v1", protocol=json.loads(card.read_text()),
        particlegan_commit=PIN, application_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        fixed_updates=STEPS, endpoints=list(ENDPOINTS), device=str(device), torch_version=torch.__version__,
        gpu=torch.cuda.get_device_name(device), input_paths={key: str(path.resolve()) for key, path in inputs.items()},
        input_sha256=input_hashes, application_source_sha256=sources, particlegan_source_sha256=native_sources)
    write(args.output / "plan.json", plan)
    for path in app_files + native_files:
        dest = args.output / "source" / (path.relative_to(ROOT) if path in app_files else Path("native") / path.relative_to(pg))
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)

    def require(name, condition):
        checks[name] = bool(condition)
        if not condition:
            raise AssertionError(name)

    def budget():
        if time.monotonic() - started > 14400 or any(value > 7200 for value in charged.values()):
            raise TimeoutError("declared execution budget exhausted")
        require("application_sources_unchanged", all(sha(ROOT / name) == value for name, value in sources.items()))
        require("native_sources_unchanged", all(sha(pg / name) == value for name, value in native_sources.items()))

    def save_state(loop, path):
        temporary = path.with_suffix(".tmp")
        torch.save(checkpoint(loop), temporary)
        temporary.replace(path)

    def immutable_owners(loop):
        p = loop.policy
        owners = {"G": p.G, "D": p.D, "E": p.encoder,
                  "average_G": p.ema_G, "average_D": p.opt_d.ema_critic, "average_E": p.ema_encoder}
        frozen_names = [name for name, value in p.G.named_parameters() if not value.requires_grad]
        average_G = dict(p.ema_G.named_parameters())
        return state_digest(dict(parameters=frozen_digest(loop), average_frozen_G={name: average_G[name] for name in frozen_names},
            average_frozen_encoder=p.ema_encoder.state_dict(), buffers={
            role + "." + name: value for role, owner in owners.items() for name, value in owner.named_buffers()}))

    def all_edit(loop):
        indices = torch.randint(len(loop.fit_context), (4,), generator=loop.data_rng)
        context = loop.fit_context[indices.to(device)]
        with torch.autograd.set_multithreading_enabled(False):
            row = update(loop, context=context, target=torch.zeros(4, 4, 32, 32, device=device),
                         batch_indices=indices, game_weight=1.)
        row.update(hold=False, game_weight=1.)
        for key in ("loss_g", "loss_d", "loss_d_game", "penalty", "bank_grad_norm", "output_sigma"):
            require("finite_" + key, torch.isfinite(torch.tensor(row[key])).item())
        return row

    class FastReference:
        def __init__(self, loop, ablate=False):
            self.loop, self.encoder, self.ablate = loop, loop.policy.encoder, ablate
        def routed_forward(self, context):
            p = self.loop.policy
            return p.routed_control.spec.forward(p._training_modules(), context, p.routed_control.candidate(),
                perturb_fn=(lambda codes: torch.zeros_like(codes)) if self.ablate else None)

    @torch.no_grad()
    def full_evaluation(reference, data, judges):
        stream, results = torch.Generator().manual_seed(72), {}
        for pool_name in ("fit", "test", "holds", "preservation"):
            pool, records, panel_hash = data[pool_name], [], hashlib.sha256()
            for start in range(0, len(pool["context"]), 4):
                context = pool["context"][start:start + 4].to(device)
                residual = reference.routed_forward(context)
                condition = reference.encoder.condition(context).repeat(4, 1)
                raw_panel = torch.randn(4, len(context), 256, 16, generator=stream)
                panel_hash.update(raw_panel.contiguous().numpy().tobytes(order="C"))
                bases = .125 * raw_panel.to(device)
                values = {}
                for name, judge in judges.items():
                    fake = (bases + (patchify(residual).float() / judge.scale).unsqueeze(0)).flatten(0, 1)
                    values[name] = F.softplus(judge(bases.flatten(0, 1), condition) - judge(fake, condition)).reshape(4, len(context)).mean(0)
                mse = residual.float().square().flatten(1).mean(1)
                for index in range(len(context)):
                    records.append(dict(index=start + index, source_caption_id=int(context[index, 4097]),
                        time=float(context[index, 4096]), mse_diagnostic=float(mse[index]),
                        **{name: float(scores[index]) for name, scores in values.items()}))
            results[pool_name] = dict(count=len(records), records=records,
                rmse_diagnostic=(sum(row["mse_diagnostic"] for row in records) / len(records)) ** .5,
                **{name: sum(row[name] for row in records) / len(records) for name in judges})
            require("private_panel_identity_" + pool_name, panel_hash.hexdigest() ==
                protocol["evaluation"]["gaussian_panels"]["unscaled_gaussian_sha256_by_pool"][pool_name])
            emit(event="full_pool", pool=pool_name, **{name: results[pool_name][name] for name in judges})
        return results

    def report(phase, step, **extra):
        write(args.output / "status.json", dict(phase=phase, step=step, steps=STEPS,
            editing_updates=step, preservation_updates=0, seconds=time.monotonic() - started,
            arms={arm: dict(charged_seconds=charged[arm], **extra.get(arm, {})) for arm in ARMS}))

    try:
        report("preparing", 0)
        data = torch.load(inputs["data"], map_location="cpu", weights_only=False)
        teacher = torch.load(inputs["teacher"], map_location="cpu", weights_only=False, mmap=True)
        require("cached_data_identity", state_digest(data) == teacher["config"]["dataset_digest"])
        require("frozen_protocol_data_identity", state_digest(data) == protocol["data"]["digest"])
        teacher_state = {key.removeprefix("teacher."): value for key, value in teacher["policy"]["models"]["encoder"].items() if key.startswith("teacher.")}
        with torch.random.fork_rng(devices=[]), torch.device("meta"):
            module = model_module()
            base = module.SupraDiT()
            module.attach_supra_lora(base, rank=16, alpha=16, targets=TARGETS)
        base.load_state_dict(teacher_state, strict=True, assign=True)
        base.to(device).eval().requires_grad_(False)
        require("pure_base_zero_ordinary_up", all(not value.count_nonzero() for name, value in teacher_state.items() if name.endswith(".up.weight")))
        del teacher, teacher_state
        gc.collect()
        mode_names = dict(sampled_control="sampled_v1", sampled_hb_neutral="sampled_hb_neutral_v1")
        def fresh(arm):
            with torch.random.fork_rng(devices=[device.index or 0]):
                loop = make_training_loop(base, data, device=device, architecture=GATED_PARTICLE_V3,
                    profile=SHARED_ROUTED_PROFILE, particle_init=mode_names[arm])
            loop.config.update(training_schedule="fresh_editing_only_v1", sampling="CPU generator7; fit only from update1", preservation_game_weight=0.)
            return loop
        for arm in ARMS:
            loops[arm] = fresh(arm)
            (args.output / arm).mkdir()
            p = loops[arm].policy
            require("particle_shape_" + arm, len(p.G.sites) == 71 and p.G.rank == 16 and tuple(p.table.shape) == (128, 4))
            require("initial_live_C_zero_up_" + arm, all(not branch.up.weight.count_nonzero()
                and branch.bridge.weight[:, 16:].count_nonzero() > 0 and branch.bridge.weight.requires_grad
                and branch.bridge.bias.requires_grad for branch in p.G.particle_branches()))
        control, neutral = (loops[arm] for arm in ARMS)
        plan["configs"] = {arm: deepcopy(loop.config) for arm, loop in loops.items()}
        write(args.output / "plan.json", plan)
        a, b = (deepcopy(checkpoint(loops[arm])) for arm in ARMS)
        for family in ("models", "averages"):
            for name, value in b["policy"][family]["generator"].items():
                expected = a["policy"][family]["generator"][name]
                if name.endswith("bridge.weight"):
                    expected[:, :16].zero_()
                elif name.endswith("bridge.bias"):
                    expected.zero_()
                require("initial_generator_" + family + "_" + name, torch.equal(value, expected))
            a["policy"][family].pop("generator")
            b["policy"][family].pop("generator")
        a.pop("config"); b.pop("config")
        require("all_other_initial_native_owners_equal", state_digest(a) == state_digest(b))
        del a, b
        source = data["fit"]["context"][:4].to(device).clone()
        source[:, 4098] = source[:, 4097]
        with torch.no_grad(), evaluation_modes(control.policy), evaluation_modes(neutral.policy):
            x, y = FastReference(control).routed_forward(source), FastReference(neutral).routed_forward(source)
        require("both_initial_common_base_exact", torch.equal(x, y) and not x.count_nonzero())
        judges = {}
        for name in ("D1856", "D6400"):
            state = torch.load(inputs[name], map_location="cpu", weights_only=False, mmap=True)
            with torch.random.fork_rng(devices=[device.index or 0]):
                judge = ConditionalTokenCritic(data["coordinate_scale"]).to(device)
            judge.load_state_dict(state["policy"]["models"]["critic"], strict=True)
            require("judge_clock_" + name, state["policy"]["completed_steps"] == (1856 if name == "D1856" else 6400))
            judge_key = "D1856" if name == "D1856" else "V2D6400"
            require("judge_tensor_identity_" + name, state_digest(judge.state_dict()) == protocol["evaluation"]["judges"][judge_key]["critic_tensor_digest"])
            judges[name] = judge.eval().requires_grad_(False)
        write(args.output / "judges.json", {name: state_digest(judge.state_dict()) for name, judge in judges.items()})
        frozen = {arm: immutable_owners(loop) for arm, loop in loops.items()}
        initial_bank = {arm: state_digest(loop.policy.table) for arm, loop in loops.items()}
        initial_router = {arm: state_digest(loop.policy.router.state_dict()) for arm, loop in loops.items()}
        indices = []
        test = data["test"]["context"]
        for subject in test[:, 4097].unique(sorted=True):
            available = (test[:, 4097] == subject).nonzero().flatten()
            offsets = torch.linspace(0, len(available) - 1, 2).round().long()
            indices.extend(available[offsets].tolist())
        probe_context = test[indices].to(device)
        probe_panels = torch.randn(4, len(indices), 256, 16, generator=torch.Generator().manual_seed(72)).to(device)
        @torch.no_grad()
        def clean_probe(reference):
            scores = []
            judge = judges["D1856"]
            for start in range(0, len(probe_context), 4):
                context = probe_context[start:start + 4]
                residual = reference.routed_forward(context)
                condition = reference.encoder.condition(context).repeat(4, 1)
                real = .125 * probe_panels[:, start:start + len(context)]
                fake = real + (patchify(residual).float() / judge.scale).unsqueeze(0)
                scores.extend(F.softplus(judge(real.flatten(0, 1), condition) - judge(fake.flatten(0, 1), condition)).reshape(4, len(context)).mean(0).cpu().tolist())
            return sum(scores) / len(scores)
        # Rescore historical references on the new CPU72 panel cohort.
        class OrdinaryReference:
            def __init__(self):
                self.encoder = control.policy.encoder
                self.model = deepcopy(base).eval().requires_grad_(False)
                load_adapter_state(self.model, load_file(str(inputs["original6400"]), device="cpu"))
            @torch.no_grad()
            def routed_forward(self, context):
                z, t, ctx, mask, uctx, umask, strength = self.encoder.unpack(context)
                for layer in self.model.modules():
                    if hasattr(layer, "multiplier"):
                        layer.multiplier = strength
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    outputs = self.model(torch.cat((z, z)), torch.cat((t, t)),
                        torch.cat((ctx, uctx.expand(len(z), -1, -1))), torch.cat((mask, umask.expand(len(z), -1))))
                positive, negative = outputs.float().chunk(2)
                return negative + 3 * (positive - negative) - self.encoder.teacher_velocity(context)
        ordinary = OrdinaryReference()
        references = {"ordinary_lora_6400": full_evaluation(ordinary, data, judges)}
        reference_probes = {"ordinary_lora_6400": clean_probe(ordinary)}
        del ordinary
        for name in ("historical_particle_v2", "historical_particle_v3"):
            with torch.random.fork_rng(devices=[device.index or 0]):
                adapter = load_particle_adapter(base, inputs[name], device=device)
            class ExportReference:
                encoder = control.policy.encoder
                @torch.no_grad()
                def routed_forward(self, context):
                    z, t, ctx, mask, uctx, umask, strength = self.encoder.unpack(context)
                    return adapter.velocity(z, t, ctx, mask, uctx, umask, strength=strength) - self.encoder.teacher_velocity(context)
            references[name] = full_evaluation(ExportReference(), data, judges)
            reference_probes[name] = clean_probe(ExportReference())
            del adapter
            gc.collect()
        write(args.output / "historical-references.json", references)
        write(args.output / "historical-probes.json", dict(scores=reference_probes, indices=indices,
            context_digest=state_digest(probe_context), panel_digest=state_digest(probe_panels),
            panel_law="private CPU72 four panels, fixed test subset; same as live clean probes",
            meaning="historical final checkpoints are reference levels, not matched fresh-init controls"))
        monitors, rows, coverage = {}, {arm: [] for arm in ARMS}, {arm: dict(live_bank_updates=0, dense_128_row_updates=0, moves=0) for arm in ARMS}
        for arm, loop in loops.items():
            save_state(loop, args.output / arm / "checkpoint-00000.pt")
            monitor = GameProgressMonitor(loop, data, judges["D1856"], args.output / arm / "monitor")
            # Fixed held-out progress contexts, never used by training/guards.
            monitor.pools = {"edit": (probe_context, probe_panels)}
            # The new dashboard renders JSON; the legacy PNG assumes two pools.
            monitor.plot = lambda: None
            write(args.output / arm / "monitor/progress-edit-indices.json", indices)
            monitors[arm] = monitor
            trace_files[arm] = (args.output / arm / "train.jsonl").open("w", buffering=1)
            monitor.evaluate(loop, [])
        endpoints = {}
        recovery = {}
        shared_preparation_seconds = time.monotonic() - started
        for arm in ARMS:
            charged[arm] += shared_preparation_seconds / len(ARMS)
        write(args.output / "shared-preparation.json", dict(seconds=shared_preparation_seconds,
            charged_equally_to_arms=True))
        for step in range(1, STEPS + 1):
            current = {}
            for arm, loop in loops.items():
                tick = time.monotonic()
                row = all_edit(loop)
                rows[arm].append(row)
                current[arm] = row
                trace_files[arm].write(json.dumps(diagnostic(row), allow_nan=False) + "\n")
                coverage[arm]["live_bank_updates"] += int(row["dense_gradient_rows"] > 0)
                coverage[arm]["dense_128_row_updates"] += int(row["dense_gradient_rows"] == 128)
                coverage[arm]["moves"] += int((row.get("move") or {}).get("moves", 0))
                if step % 400 == 0 or step in (*ENDPOINTS, 802):
                    require("frozen_owners_" + arm, immutable_owners(loop) == frozen[arm])
                    save_state(loop, args.output / arm / f"checkpoint-{step:05d}.pt")
                if step % 200 == 0:
                    probe = monitors[arm].evaluate(loop, rows[arm][-100:])
                    emit(event="progress", arm=arm, step=step, test_probe=probe["probes"]["edit"]["clean"]["frozen_start_D"],
                        loss_g=row["loss_g"], loss_d_game=row["loss_d_game"], sigma=row["output_sigma"],
                        bank_gradient_rows=row["dense_gradient_rows"], moves=coverage[arm]["moves"], seconds=time.monotonic() - started)
                if step in ENDPOINTS:
                    before = state_digest(checkpoint(loop))
                    with evaluation_modes(loop.policy):
                        result = full_evaluation(FastReference(loop), data, judges)
                    require("full_evaluation_immutable_" + arm + str(step), state_digest(checkpoint(loop)) == before)
                    write(args.output / arm / f"evaluation-{step:05d}.json", result)
                    endpoints[arm + "@" + str(step)] = result
                    with evaluation_modes(loop.policy):
                        ablated = full_evaluation(FastReference(loop, ablate=True), data, judges)
                        with mass_only_routing(loop.policy.router) as calls:
                            mass_only = full_evaluation(FastReference(loop), data, judges)
                    require("code_ablation_immutable_" + arm + str(step), state_digest(checkpoint(loop)) == before)
                    require("mass_only_all_sites_" + arm + str(step), calls[0] >= 71)
                    write(args.output / arm / f"particle-ablations-{step:05d}.json", dict(code_scores=ablated, mass_only_scores=mass_only,
                        zero_code_minus_live_test_game={name: ablated["test"][name] - result["test"][name] for name in judges},
                        mass_only_minus_live_test_game={name: mass_only["test"][name] - result["test"][name] for name in judges}))
                    snapshot = SimpleNamespace(generator=loop.policy.G, router=loop.policy.router, table=loop.policy.table,
                        encoder=loop.policy.encoder, models=loop.policy._training_modules(), routing=loop.policy.routed_control.spec,
                        source="fast", completed_steps=step)
                    export_path = args.output / arm / f"adapter-{step:05d}.safetensors"
                    export_served_adapter(snapshot, export_path,
                        extra_metadata={"particle_init": mode_names[arm], "training_schedule": "fresh_editing_only_v1"})
                    with torch.random.fork_rng(devices=[device.index or 0]):
                        loaded = load_particle_adapter(base, export_path, device=device)
                    with torch.no_grad(), evaluation_modes(loop.policy):
                        context = data["test"]["context"][:4].to(device)
                        z, t, ctx, mask, uctx, umask, strength = loop.policy.encoder.unpack(context)
                        actual = loaded.velocity(z, t, ctx, mask, uctx, umask, strength=strength)
                    # Compare the raw host before residual subtraction as well.
                    from supra.particle_training import raw_model_forward, features_for_rows
                    from particlegan import RoutedRows
                    raw_rows = RoutedRows(model_forward=raw_model_forward, features=features_for_rows, sites=loop.policy.G.sites)
                    with torch.no_grad(), evaluation_modes(loop.policy):
                        raw_expected = raw_rows.forward(loop.policy._training_modules(), context, loop.policy.routed_control.candidate())
                    require("export_reload_raw_exact_" + arm + str(step), torch.equal(actual, raw_expected))
                    require("export_reload_state_immutable_" + arm + str(step), state_digest(checkpoint(loop)) == before)
                    del loaded
                    gc.collect()
                charged[arm] += time.monotonic() - tick
            for key in ("batch_indices", "base_noise_sums", "paired_rng_digest", "hold", "game_weight"):
                require("matched_" + key, current[ARMS[0]][key] == current[ARMS[1]][key])
            if step == 802:
                for arm, loop in loops.items():
                    tick = time.monotonic()
                    before = state_digest(checkpoint(loop))
                    with torch.random.fork_rng(devices=[device.index or 0]):
                        replay = fresh(arm)
                        restore(replay, torch.load(args.output / arm / "checkpoint-00800.pt", map_location="cpu", weights_only=False))
                        actual = [all_edit(replay) for _ in range(2)]
                        require("recovery_rows_" + arm, state_digest(actual) == state_digest(rows[arm][800:802]))
                        require("recovery_state_" + arm, state_digest(checkpoint(replay)) == before)
                    require("recovery_authoritative_state_" + arm, state_digest(checkpoint(loop)) == before)
                    del replay
                    recovery[arm] = dict(rows_exact=True, state_exact=True, from_step=800, to_step=802)
                    gc.collect()
                    charged[arm] += time.monotonic() - tick
            if step % 25 == 0:
                report("training", step, **{arm: {"rolling": rolling_losses(rows[arm][-100:]), "coverage": coverage[arm]} for arm in ARMS})
            if step % 400 == 0:
                budget()
        particles = {}
        for arm, loop in loops.items():
            particles[arm] = dict(bank_changed=state_digest(loop.policy.table) != initial_bank[arm],
                router_changed=state_digest(loop.policy.router.state_dict()) != initial_router[arm],
                C_norms=[float(branch.bridge.weight[:, 16:].detach().norm()) for branch in loop.policy.G.particle_branches()],
                H_b_trainable=all(branch.bridge.weight.requires_grad and branch.bridge.bias.requires_grad for branch in loop.policy.G.particle_branches()))
            require("retained_particle_owners_" + arm, particles[arm]["bank_changed"] and particles[arm]["router_changed"]
                and particles[arm]["H_b_trainable"] and all(value > 0 for value in particles[arm]["C_norms"])
                and coverage[arm]["dense_128_row_updates"] > 0)
        require("immutable_inputs", all(sha(path) == input_hashes[key] for key, path in inputs.items()))
        budget()
        write(args.output / "receipt.json", dict(status="complete", plan=plan, checks=checks,
            seconds=time.monotonic() - started, charged_seconds=charged, recovery=recovery, coverage=coverage,
            particles=particles, final_native_digests={arm: state_digest(checkpoint(loop)) for arm, loop in loops.items()},
            final_checkpoint_sha256={arm: sha(args.output / arm / "checkpoint-06400.pt") for arm in ARMS},
            endpoint_test_scores={name: result["test"] for name, result in endpoints.items()},
            qualification_credit="none; real task comparison pending independent review"))
        report("complete", STEPS)
        emit(event="complete", seconds=time.monotonic() - started)
    except Exception as error:
        write(args.output / "failure.json", dict(error=type(error).__name__ + ": " + str(error), checks=checks,
            seconds=time.monotonic() - started))
        report("incomplete" if isinstance(error, TimeoutError) else "error", min((loop.policy.completed_steps for loop in loops.values()), default=0))
        raise
    finally:
        for stream in trace_files.values():
            stream.close()


if __name__ == "__main__":
    main()
