#!/usr/bin/env python3
"""Independent read-only CPU qualification of the complete native V2 run."""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import tarfile

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.review_e22_supra_training_controls import evaluate_check, finite, state_digest

PIN = "cabe2084284db923d525918cbf3e18de6f20faac"
START, END = 1856, 6400
ARCHITECTURE = "linear_modulated_v2"
BASELINE_SHA = "24a98ba055d0b5298136d67dc7491a339684da03416c510d1d443fe730eb7069"
POOLS = ("fit", "test", "holds", "preservation")
PANEL_SHA = {"edit": "a854a5f234968016ed38049858b9562e3379ebe6ee19ee57fafd9043c7edb4bf",
             "preservation": "de3b349c457cfd4f9abf8773faf21bcc06a56b6050a4398d2066e10f44d1e199"}


def require(value, message):
    if not value:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    content = Path(path).read_text()
    require(not content or content.endswith("\n"), f"incomplete trace: {path}")
    return [json.loads(line) for line in content.splitlines() if line.strip()]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def raw_digest(tensor):
    return hashlib.sha256(tensor.cpu().contiguous().numpy().tobytes()).hexdigest()


def close(actual, expected, message, tolerance=1e-12):
    require(math.isclose(actual, expected, abs_tol=tolerance, rel_tol=tolerance),
            f"{message}: {actual!r} != {expected!r}")


def immutable_weights(saved):
    policy = saved["policy"]
    result = {}
    for family in ("models", "averages"):
        result[family] = {}
        for role in ("generator", "encoder"):
            flags = policy["requires_grad"][role]
            # Include immutable encoder buffers as well as every frozen weight.
            result[family][role] = {name: value for name, value in policy[family][role].items()
                                    if not flags.get(name, False)}
    result["critic_scale"] = policy["models"]["critic"]["scale"]
    return result


def source_check(directory, plan, run):
    manifest = read(directory / "source/sha256.json")
    archived = subprocess.check_output([
        "git", "-C", "/ml2/hypergan/ParticleGAN-pr155-merge", "archive", PIN, "particlegan"])
    with tarfile.open(fileobj=io.BytesIO(archived)) as tree:
        committed = {str(Path(member.name).relative_to("particlegan")):
                     hashlib.sha256(tree.extractfile(member).read()).hexdigest()
                     for member in tree.getmembers() if member.isfile() and member.name.endswith(".py")}
    require(manifest["particlegan"] == committed, "native sources differ from the actual pinned Git tree")
    for name, expected in committed.items():
        require(sha(directory / "source/particlegan" / name) == expected
                == sha(Path(plan["particlegan_root"]) / "particlegan" / name),
                f"native source/snapshot changed: {name}")
    for name, expected in manifest["application"].items():
        require(sha(directory / "source" / name) == expected == sha(ROOT / name),
                f"application source/snapshot changed: {name}")
    digest = state_digest({name: value for name, value in committed.items() if "/" not in name})
    require(digest == plan["particlegan_source_digest"] == run["particlegan_source_digest"],
            "native module digest differs")
    return dict(actual_git_tree_matches=True, current_and_snapshotted_application_sources_match=True,
                native_module_digest=digest, manifest=manifest)


def trace_check(directory, receipt, initial, final, data, control):
    trace = rows(directory / "train.jsonl")
    require(len(trace) == END - START and [row["step"] for row in trace] == list(range(START + 1, END + 1)),
            "fixed update clock or full trace length differs")
    finite(trace)
    stream = torch.Generator().set_state(initial["data_rng"].cpu())
    program = []
    for row in trace:
        hold = row["step"] % 5 == 0
        indices = torch.randint(len(data["holds" if hold else "fit"]["context"]), (4,), generator=stream).tolist()
        require(row["hold"] is hold and row["batch_indices"] == indices,
                f"data draw or task differs at {row['step']}")
        require(row["game_weight"] == (.1 if hold else 1.) and row["penalty_calls"] == row["step"]
                and row["penalty_phase"] == "blend" and row["dense_gradient_rows"] == 128,
                f"native task weight or optimizer clock differs at {row['step']}")
        close(row["loss_d"], row["loss_d_game"] + row["penalty"], "D loss decomposition", tolerance=2e-6)
        require(len(row["base_noise_sums"]) == 2 and all(
            re.fullmatch(r"[0-9a-f]{64}", row[key]) for key in ("paired_rng_digest", "dv12_rng_digest")),
            "missing or malformed owned noise-program witness")
        program.append({key: row[key] for key in ("step", "hold", "batch_indices", "base_noise_sums",
                                                 "paired_rng_digest", "dv12_rng_digest")})
    require(torch.equal(stream.get_state(), final["data_rng"]), "final data RNG differs from exact CPU draw replay")
    for state, field, last_key in ((final["data_rng"], "final_data_rng_digest", None),
            (final["paired_noise_rng"], "final_paired_rng_digest", "paired_rng_digest"),
            (final["policy"]["streams"]["noise_generator"], "final_dv12_rng_digest", "dv12_rng_digest")):
        require(raw_digest(state) == receipt[field], f"saved final owned stream differs: {field}")
        if last_key:
            require(receipt[field] == trace[-1][last_key], f"final owned noise witness differs: {field}")
    reference = rows(control / "train.jsonl")
    require(len(reference) == 256 and [row["step"] for row in reference] == list(range(START + 1, START + 257)),
            "qualified 256-update control is incomplete")
    control_receipt = read(control / "receipt.json")
    require(control_receipt["initial_native_digest"] == receipt["plan"]["initial_native_digest"],
            "control starts from another native state")
    require(all(all(other[key] == value for key, value in native.items())
                for native, other in zip(trace[:256], reference)), "first256 native/noise trace differs from control")
    moves = [row["move"] for row in trace if isinstance(row["move"], dict) and row["move"].get("moves", 0)]
    require(moves == receipt["accepted_moves"], "accepted move receipt differs from complete trace")
    return trace, dict(updates=len(trace), edit_updates=sum(not row["hold"] for row in trace),
        preservation_updates=sum(row["hold"] for row in trace), complete_recorded_program_digest=state_digest(program),
        cpu_data_program_replayed_exactly=True, final_owned_streams_match_saved_state_and_receipt=True,
        first256_full_native_rows_match_qualified_control=True, accepted_move_updates=len(moves),
        paired_cuda_program_runtime_witness=receipt["data_and_paired_streams_matched"],
        dv12_limit="CPU validates recorded digests, clocks and saved endpoints, not CUDA perturbation draw values or full gradients.")


def rolling(trace, step):
    result = {}
    for name, hold in (("edit", False), ("preservation", True)):
        chosen = [row for row in trace if row["step"] <= step and row["hold"] is hold][-100:]
        if chosen:
            result[name] = dict(updates=len(chosen), first_step=chosen[0]["step"], last_step=chosen[-1]["step"],
                g_game=sum(row["loss_g"] / row["game_weight"] for row in chosen) / len(chosen),
                d_game=sum(row["loss_d_game"] / row["game_weight"] for row in chosen) / len(chosen),
                penalty=sum(row["penalty"] for row in chosen) / len(chosen),
                bank_grad_norm=sum(row["bank_grad_norm"] for row in chosen) / len(chosen))
    return result


def progress_check(directory, plan, data, trace):
    progress = rows(directory / "progress.jsonl")
    expected_steps = [START] + [step for step in range(START + 1, END + 1)
                               if step % plan["probe_every"] == 0 or step == END]
    require([row["step"] for row in progress] == expected_steps, "frozen-probe history is incomplete or has extra points")
    finite(progress)
    stream = torch.Generator().manual_seed(72)
    panels = {}
    for name, pool_name, count in (("edit", "fit", 2), ("preservation", "holds", 4)):
        pool = data[pool_name]["context"]
        indices = []
        for subject in pool[:, 4097].unique(sorted=True):
            available = (pool[:, 4097] == subject).nonzero().flatten()
            offsets = torch.linspace(0, len(available) - 1, min(count, len(available))).round().long()
            indices.extend(available[offsets].tolist())
        require(indices == read(directory / f"progress-{name}-indices.json"), f"frozen {name} probe selection differs")
        noise = torch.randn((4, len(indices), 256, 16), generator=stream)
        require(raw_digest(noise) == PANEL_SHA[name], "private CPU72 probe panel program differs from independent audit")
        panels[name] = dict(indices=indices, context_digest=state_digest(pool[indices]),
                            shape=list(noise.shape), raw_panel_sha256=raw_digest(noise))
    for index, row in enumerate(progress):
        require(row["source"] == "FAST current particle G" and row["output_sigma"] == .125
                and row["output_metrics_used"] is False and row["evaluation_only"] is True
                and row["native_state_unchanged"] is True, "probe serving/immutability witness differs")
        require(row["rolling"] == rolling(trace, row["step"]), "rolling task aggregation or units differ")
        require(set(row["probes"]) == {"edit", "preservation"}, "progress mixes or omits tasks")
        for task in row["probes"]:
            require(row["probes"][task]["contexts"] == 12, "probe context count differs")
            for mode in ("clean", "dv12"):
                require(set(row["probes"][task][mode]) == {"frozen_start_D", "live_D"}, "probe judges differ")
        if index < 3:
            require("plateau_diagnostic" not in row, "plateau reported before four probes")
        else:
            window = progress[index - 3:index + 1]
            expected = {}
            for task in ("edit", "preservation"):
                first, last = (window[j]["probes"][task]["clean"]["frozen_start_D"] for j in (0, -1))
                improvement = (first - last) / max(abs(first), 1e-12)
                expected[task] = dict(steps=window[-1]["step"] - window[0]["step"],
                    relative_improvement=improvement, possible_plateau=abs(improvement) < .002,
                    regressed=improvement < -.002)
            require(row["plateau_diagnostic"] == expected, "last-four task plateau aggregation differs")
    require(read(directory / "progress-latest.json") == progress[-1], "latest progress differs from full history")
    require((directory / "progress.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n"), "progress plot is missing")
    return dict(points=len(progress), steps=expected_steps, private_cpu72_panel_program=panels,
        rolling_and_last_four_task_aggregations_exact=True, final_probe=progress[-1],
        panel_limit="Panel values are reconstructed from the pinned monitor source; GPU probe outputs and immutability are recorded runtime witnesses.",
        dv12_probe="Each pool restarts a private clone of step1856 DV12 RNG, applies the current controller and leaves training streams unchanged.")


def history_check(final):
    policy = final["policy"]
    require(policy["controller"]["variant"] == "dv12" and policy["controller"]["updates"] == END,
            "native controller variant or update clock differs")
    require(len(policy["controller"]["latent_applications"]) <= 2, "DV12 diagnostic history exceeded native bound")
    for application in policy["controller"]["latent_applications"]:
        finite(application)
        require(0 <= application["radius_min"] <= application["radius_mean"] <= application["radius_max"]
                and application["perturbation_rms"] >= 0 and 0 <= application["clipped_fraction"] <= 1,
                "invalid saved DV12 application metrics")
    require(len(policy["surprise"]["log"]) <= 8, "optimizer surprise history exceeded native bound")
    settlement = []
    for names, row in zip(policy["roles"], policy["lr_settle"]):
        for name, tester in zip(names, row):
            if tester is not None:
                require(len(tester["log"]) <= 8 and len(tester["blocks"]) <= 3
                        and tester["blocks_in_window"] < 24,
                        f"settlement diagnostic history exceeded bound: {name}")
                require(len(tester["r_b"]) <= 11 and len(tester["r_2b"]) <= 5,
                        f"settlement pair window exceeded bound: {name}")
                settlement.append(dict(role=name, log=len(tester["log"]), blocks=len(tester["blocks"]),
                    pairs_b=len(tester["r_b"]), pairs_2b=len(tester["r_2b"]), windows=tester["windows"],
                    s=tester["s"], b=tester["b"]))
    routed = policy["routing"]
    capacity = routed["config"]["reservoir_size"]
    require(routed["config"]["max_context_harm"] == 0. and len(routed["config"]["sites"]) == 71
            and routed["table_shape"] == (128, 4), "native routed criterion or geometry differs")
    for name in ("fit", "guard"):
        require(0 <= routed[name + "_fill"] <= capacity and 0 <= routed[name + "_cursor"] < capacity,
                "routing reservoir bounds differ")
        require(routed["pools"][name + "_context"].shape == (capacity, 4100)
                and routed["pools"][name + "_targets"].shape == (capacity, 4, 32, 32),
                "routing reservoir allocation exceeds bound")
    require(routed["probe_clock"]["observed_updates"] == END
            and routed["config"]["probe_interval"] == 100
            and routed["probe_clock"]["last_probe_update"] == END
            and routed["counters"]["evals"] == END // routed["config"]["probe_interval"],
            "routed probe clock or evaluation counter differs")
    record = policy["optimizers"][1]["regularizer"]["record"]
    require(record["observed_steps"] == record["calls"] == END, "native penalty clock differs")
    require(len(record["sur_hist"]) <= 400, "KA2 surprise history exceeded native bound")
    for name, value in routed["evidence"]["tensors"].items():
        require(value.shape == ((128, 4) if name == "M" else (128,)), "routed evidence allocation differs")
    return dict(dv12_application_records=len(policy["controller"]["latent_applications"]),
        optimizer_surprise_log=len(policy["surprise"]["log"]), settlement=settlement,
        routing_reservoir_capacity=capacity, probe_clock=routed["probe_clock"], routing_counters=routed["counters"],
        ka2_surprise_history=len(record["sur_hist"]),
        native_ka2={key: record[key] for key in ("calls", "observed_steps", "w", "alpha", "ema_updates", "ema_skips", "ema_reseeds")})


def cpu_dv12_check(final, plan):
    sys.path.insert(0, plan["particlegan_root"])
    import particlegan
    from particlegan.continuous import DataDriftController
    require(Path(particlegan.__file__).resolve().parent.parent == Path(plan["particlegan_root"]).resolve(),
            "CPU DV12 imported another native source")
    controller = DataDriftController("dv12")
    controller.load_state_dict(final["policy"]["controller"])
    table, mass = final["policy"]["table"].clone(), final["policy"]["models"]["router"]["log_mass"].clone()
    prior = controller.routed_prior(table, mass)
    latent = table[:4].repeat(256, 1)
    before = state_digest(dict(controller=controller.state_dict(), table=table, mass=mass, latent=latent))
    first, second = (torch.Generator().manual_seed(72) for _ in range(2))
    with torch.no_grad():
        left = controller.perturb_latent(latent, first, prior, record=False)
        right = controller.perturb_latent(latent, second, prior, record=False)
    require(torch.equal(left, right) and torch.equal(first.get_state(), second.get_state()),
            "cloned CPU DV12 perturbations or streams differ")
    require(before == state_digest(dict(controller=controller.state_dict(), table=table, mass=mass, latent=latent)),
            "read-only CPU DV12 changed its controller or native inputs")
    require(bool(torch.isfinite(left).all()), "CPU DV12 perturbations are nonfinite")
    return dict(cloned_cpu_streams_exact=True, controller_and_inputs_unchanged=True, finite=True,
        limit="CPU law repeatability on saved bank atoms only; no CUDA stream or full native forward replay.")


def export_check(path, receipt, final, run):
    source = receipt["served_source"]
    require(source in ("fast", "averaged"), "unsupported public serving source")
    selected = final["policy"]["models" if source == "fast" else "averages"]
    expected = {"generator." + name: selected["generator"][name] for name, flag in
                final["policy"]["requires_grad"]["generator"].items() if flag}
    expected.update({"router." + name: value for name, value in selected["router"].items() if name != "log_mass"})
    expected.update({"bank.table": final["policy"]["table" if source == "fast" else "averaged_table"],
                     "bank.log_mass": selected["router"]["log_mass"]})
    export = receipt["export"]
    require(len(expected) == export["tensors"] == 428 and path.stat().st_size == export["bytes"], "clean export size differs")
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
        config = json.loads(metadata["config"])
        require(metadata["format"] == "supra_particlegan_clean_v2" and config == export["config"]
                and config["architecture"] == ARCHITECTURE and config["completed_steps"] == END,
                "clean export version, architecture or horizon mislabeled")
        require(config["sites"] == run["config"]["sites"] and len(config["sites"]) == 71
                and (config["rank"], config["z_dim"], config["num_particles"], config["cfg"]) == (16, 4, 128, 3.),
                "clean export bank, routing sites or native geometry differs")
        dimensions = {site: selected["generator"][f"model.{site}.down.weight"].shape[1]
                      for site in config["sites"]}
        require(config["site_input_dims"] == dimensions and config["served_source"] == source,
                "clean export site dimensions or serving source mislabeled")
        require(config["sampling"] == "clean" and config["routed_geometry"] == "mass_atoms_v1"
                and config["extra"] == {"selection": "final fixed horizon; output metrics evaluation only", "fixed_updates": END},
                "clean export routing or selection metadata differs")
        pins = json.loads(metadata["native_pins"])
        require(pins["backend_sha256"] == read(ROOT / "backend.lock.json")["sha256"], "backend pin differs")
        for key, field in (("model_revision", "model_revision"), ("text_encoder_revision", "text_revision"), ("vae_revision", "vae_revision")):
            require(pins[key] == run[field], "native model revision differs")
        require(set(handle.keys()) == set(expected), "clean export tensor names differ")
        for name, reference in expected.items():
            value = handle.get_tensor(name)
            require(value.dtype == torch.float32 and torch.equal(value, reference), f"clean export tensor differs: {name}")
            valid = ((torch.isfinite(value) | torch.isneginf(value)).all() and torch.isfinite(value).any()
                     if name == "bank.log_mass" else torch.isfinite(value).all())
            require(bool(valid), f"invalid clean export tensor: {name}")
    require(sum(value.numel() for value in expected.values()) == export["parameters"], "export parameter count differs")
    return dict(tensors=428, selected_full_native_tensors_exact=True, architecture_and_schema_correct=True,
                source=source, sha256=sha(path))


def original_check(directory, native, data, plan, native_directory):
    if not (directory / "receipt.json").is_file():
        return dict(available=False, path=str(directory), reason="original shared-critic evaluator receipt not present")
    receipt, original_plan = read(directory / "receipt.json"), read(directory / "plan.json")
    require(receipt["plan"] == original_plan and original_plan["fixed_updates"] == END
            and original_plan["optimization"] is False, "original comparison plan differs")
    require(original_plan["particlegan_commit"] == PIN
            and original_plan["particlegan_source_digest"] == plan["particlegan_source_digest"], "original native evaluation source differs")
    require(set(original_plan["critic_labels"]) == {"fixed_start_D", "arm_final_D"}
            and original_plan["critic_labels"]["fixed_start_D"] == plan["critic_labels"]["fixed_start_D"]
            and str(END) in original_plan["critic_labels"]["arm_final_D"]
            and original_plan["pools"] == list(POOLS),
            "original common judges or pools differ")
    require(original_plan["input_sha256"]["original"] == BASELINE_SHA == plan["original_baseline_sha256"],
            "original fixed6400 snapshot differs")
    for name, path in original_plan["inputs"].items():
        require(sha(path) == original_plan["input_sha256"][name], f"original evaluator input changed: {name}")
    require(Path(original_plan["inputs"]["initial"]).resolve() == Path(plan["resumed_from"]) / "final.pt"
            and Path(original_plan["inputs"]["final"]).resolve() == native_directory / "final.pt",
            "original evaluator used different common-critic checkpoints")
    sources = read(directory / "source/sha256.json")
    require(sources == original_plan["source_sha256"], "original source manifest differs")
    for name, expected in sources.items():
        require(sha(directory / "source" / name) == expected == sha(ROOT / name), "original evaluator source changed")
    for flag in ("evaluation_state_unchanged", "input_files_unchanged", "source_files_unchanged",
                 "strength_zero_exact", "complete_pools_and_teachers_matched"):
        require(receipt[flag] is True, f"original runtime witness failed: {flag}")
    evaluation = read(directory / "evaluation.json")
    finite(evaluation)
    evaluate_check(evaluation, receipt, data)
    comparison = read(directory / "comparison.json")
    require(comparison == receipt["comparison"], "original comparison receipt differs")
    for pool in POOLS:
        old, new, summary = evaluation[pool], native[pool], comparison[pool]
        require(summary["count"] == old["count"] == new["count"] and summary["output_errors_diagnostic_only"] is True,
                "original comparison pool size or metric use differs")
        require(summary["original_rmse"] == old["rmse"] and summary["particle_rmse"] == new["rmse"]
                and summary["original_relative_rms"] == old["relative_rms"]
                and summary["particle_relative_rms"] == new["relative_rms"], "original output metric aggregate differs")
        require(all(all(first[key] == second[key] for key in ("index", "source_caption_id", "time", "teacher_rms"))
                    for first, second in zip(old["records"], new["records"])), "original native contexts or teacher pairing differs")
        for judge in ("fixed_start_D", "arm_final_D"):
            result = summary[judge]
            require(result["original_g_game"] == old[judge]["g_game"] and result["particle_g_game"] == new[judge]["g_game"],
                    "shared critic game aggregate differs")
            close(result["particle_minus_original"], new[judge]["g_game"] - old[judge]["g_game"], "shared game difference")
            require(result["particle_better_contexts"] == sum(second[judge]["g_game"] < first[judge]["g_game"]
                        for first, second in zip(old["records"], new["records"])), "context game comparison differs")
            subjects = sorted({row["source_caption_id"] for row in old["records"]})
            require(set(result["per_source_caption"]) == {str(subject) for subject in subjects}, "subject set differs")
            for subject in subjects:
                for name, records in (("original_g_game", old["records"]), ("particle_g_game", new["records"])):
                    values = [row[judge]["g_game"] for row in records if row["source_caption_id"] == subject]
                    close(result["per_source_caption"][str(subject)][name], sum(values) / len(values), "subject game mean")
    return dict(available=True, qualified=True, comparison=comparison,
        limits="The original ordinary-LoRA MSE recipe and particle native-game formulation differ in architecture and objective; this does not isolate optimizer effects.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "outputs/e22-particle-v2-6400")
    parser.add_argument("--control", type=Path, default=ROOT / "outputs/e22-particle-preservation-units-256/native")
    parser.add_argument("--original", type=Path, help="Default: RUN/evaluation-original-6400-game, reviewed when complete")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    require(__debug__, "run qualification without Python optimization; aggregate helper checks use assertions")
    directory = args.run.resolve()
    output = args.output or directory / "qualification-review.json"
    require(not output.exists(), "preserve the existing review; choose a fresh output")
    torch.set_num_threads(1)
    global_rng = torch.get_rng_state().clone()
    reviewer_sources = {str(path.relative_to(ROOT)): sha(path) for path in
                        (Path(__file__), ROOT / "scripts/review_e22_supra_training_controls.py")}
    required_files = ("receipt.json", "plan.json", "run.json", "status.json", "restore-review.json", "data.pt",
                      "final.pt", "train.jsonl", "progress.jsonl", "progress-latest.json", "progress.png",
                      "progress-edit-indices.json", "progress-preservation-indices.json", "evaluation.json",
                      "final.safetensors", "source/sha256.json")
    artifact_sha256 = {name: sha(directory / name) for name in required_files}
    receipt, plan, run, status = (read(directory / name) for name in ("receipt.json", "plan.json", "run.json", "status.json"))
    require(receipt["plan"] == run["continuation"] == plan, "final continuation plan differs")
    require((plan["start_step"], plan["fixed_updates"], plan["additional_updates"]) == (START, END, END - START), "fixed horizon differs")
    require(status == dict(phase="complete", step=END, steps=END), "run has not completed")
    require(plan["particlegan_commit"] == run["particlegan_commit"] == PIN
            and plan["architecture"] == ARCHITECTURE and run["fixed_updates"] == END, "native source or architecture differs")
    source = source_check(directory, plan, run)
    reference = Path(plan["resumed_from"])
    for name, expected in plan["input_sha256"].items():
        require(sha(reference / name) == expected, f"starting input changed: {name}")
    require(sha(directory / "data.pt") == plan["input_sha256"]["data.pt"], "copied data bytes differ")
    require(sha(plan["original_baseline_path"]) == plan["original_baseline_sha256"] == BASELINE_SHA, "declared fixed original snapshot changed")
    initial = torch.load(reference / "final.pt", map_location="cpu", weights_only=False, mmap=True)
    final = torch.load(directory / "final.pt", map_location="cpu", weights_only=False, mmap=True)
    data = torch.load(directory / "data.pt", map_location="cpu", weights_only=False)
    initial_digest = state_digest(initial)
    final_digest = state_digest(final)
    require(initial["policy"]["completed_steps"] == START and final["policy"]["completed_steps"] == END, "native checkpoint horizon differs")
    require(initial_digest == plan["initial_native_digest"] == read(reference / "receipt.json")["final_native_digest"], "complete initial native state differs")
    require(final_digest == receipt["final_native_digest"] and sha(directory / "final.pt") == receipt["final_checkpoint_sha256"], "complete final native state differs")
    require(final["config"] == initial["config"] and json.loads(json.dumps(final["config"])) == run["config"], "native configuration was changed")
    require(final["config"]["architecture"] == ARCHITECTURE and final["config"]["output_error_guard"] is False
            and final["config"]["max_feature_context_harm"] == 0., "architecture or structural game criterion differs")
    require(final["policy"]["recipe"] == initial["policy"]["recipe"]
            and final["policy"]["requires_grad"] == initial["policy"]["requires_grad"], "native recipe or frozen ownership changed")
    for field in ("schema", "roles", "row_semantics", "device", "dtype", "table_requires_grad", "table_location", "initial_lrs"):
        require(final["policy"][field] == initial["policy"][field], f"native fixed ownership metadata changed: {field}")
    require(final["policy"]["routing"]["config"] == initial["policy"]["routing"]["config"],
            "native routing specification changed")
    require(state_digest(data) == final["config"]["dataset_digest"], "complete evaluation pool digest differs")
    immutable_digest = state_digest(immutable_weights(initial))
    require(immutable_digest == state_digest(immutable_weights(final)), "frozen fast/averaged weights or immutable buffers changed")
    for flag in ("initial_native_state_exact", "two_update_replay_exact", "frozen_unchanged", "export_reload_exact",
                 "evaluation_state_unchanged", "data_and_paired_streams_matched", "source_and_inputs_unchanged"):
        require(receipt[flag] is True, f"native runtime witness failed: {flag}")
    require(read(directory / "restore-review.json") == dict(initial_native_state_exact=True, two_update_replay_exact=True), "exact restore preflight witness differs")
    trace, program = trace_check(directory, receipt, initial, final, data, args.control)
    progress = progress_check(directory, plan, data, trace)
    history = history_check(final)
    dv12 = cpu_dv12_check(final, plan)
    exported = export_check(directory / "final.safetensors", receipt, final, run)
    evaluation = read(directory / "evaluation.json")
    finite(evaluation)
    evaluate_check(evaluation, receipt, data)
    original = original_check(args.original or directory / "evaluation-original-6400-game", evaluation, data, plan, directory)
    require(all(sha(directory / name) == expected for name, expected in artifact_sha256.items()),
            "native review input changed during qualification")
    source_check(directory, plan, run)
    require(all(sha(ROOT / name) == expected for name, expected in reviewer_sources.items()),
            "reviewer source changed during qualification")
    require(torch.equal(global_rng, torch.get_rng_state()), "CPU reviewer changed global RNG")
    report = dict(schema="supra_e22_v2_6400_qualification_v1", qualified=True, gpu_used=False,
        source=source, start_step=START, final_step=END, initial_native_digest=initial_digest,
        final_native_digest=final_digest, immutable_weights_digest=immutable_digest,
        common_start_critic_digest=state_digest(initial["policy"]["models"]["critic"]),
        final_critic_digest=state_digest(final["policy"]["models"]["critic"]),
        native_configuration_unchanged=True, no_output_guard_and_zero_feature_context_harm=True,
        program=program, progress=progress, bounded_history=history, cpu_dv12_repeatability=dv12, export=exported,
        artifact_sha256=artifact_sha256, reviewer_source_sha256=reviewer_sources,
        complete_per_context_evaluation_aggregates_exact=True, metrics=receipt["metrics"], original_comparison=original,
        output_metrics_used_for_optimization_or_selection=False,
        limits=["CPU review replays the complete CPU data stream and validates saved full state, sources, tensors and aggregates; it does not replay GPU full gradients.",
                "Initial equality, two-update GPU replay, full paired CUDA program, probe immutability, export-forward parity and evaluation immutability are recorded runtime witnesses.",
                "Recorded DV12 digest endpoints and the exact first256 control prefix do not independently reconstruct every intervening CUDA perturbation draw.",
                "One fixed task and training stream; per-context results are correlated and learned final critics are endogenous."])
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(qualified=True, output=str(output), final_step=END, program=program,
                         metrics=receipt["metrics"], original_comparison_available=original["available"])), flush=True)


if __name__ == "__main__":
    main()
