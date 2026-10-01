#!/usr/bin/env python3
"""CPU artifact review of the fixed, two-arm Supra H/b initialization run.

No model construction, updates, forward calls, or quality-based selection.
GPU forward/recovery checks remain explicit held-source runtime witnesses.
Use --partial during training; that report cannot qualify completion.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.review_e22_supra_pr223 import committed_python_hashes, state_digest

PIN = "6ec7e5788e14ea15ddc3e16ac71110458108b6a6"
ARMS = ("sampled_control", "sampled_hb_neutral")
MODES = dict(zip(ARMS, ("sampled_v1", "sampled_hb_neutral_v1")))
COUNTS = dict(fit=240, test=240, holds=30, preservation=60)
JUDGES = ("D1856", "D6400")
HORIZON = 6400
ENDPOINTS = (5120, 6400)
CHECKPOINTS = (0, *range(400, 6401, 400), 802, 5120)
_SHA_CACHE = {}


class Reviewer:
    def __init__(self):
        self.checks = 0

    def require(self, condition, label):
        self.checks += 1
        if not condition:
            raise ValueError(label)

    def close(self, actual, expected, label):
        self.require(math.isfinite(actual) and math.isfinite(expected)
                     and math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-12), label)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    path = Path(path).resolve()
    stat = path.stat()
    identity = (str(path), stat.st_size, stat.st_mtime_ns)
    if identity not in _SHA_CACHE:
        with path.open("rb") as stream:
            _SHA_CACHE[identity] = hashlib.file_digest(stream, "sha256").hexdigest()
    return _SHA_CACHE[identity]


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False, mmap=True)


def source_digest(sources):
    """The frozen runner's named-file digest differs from tensor-state digest."""
    digest = hashlib.sha256()
    for name in sorted(sources):
        digest.update(name.encode())
        digest.update(json.dumps(sources[name], sort_keys=True).encode())
    return digest.hexdigest()


def canonical(value):
    return json.loads(json.dumps(value))


def check_checkpoint_contract(review, state, initial, declared_config, arm, step, data_digest):
    """The immutable control/init contract, separate from expensive state reads."""
    policy, config = state["policy"], state["config"]
    review.require(policy["completed_steps"] == step and config == initial["config"]
                   and canonical(config) == declared_config and config["particle_init"] == MODES[arm], arm + "/clock/config")
    review.require(config["architecture"] == "gated_particle_v3" and config["particle_profile"] == "pr223_shared_routed_v1"
                   and config["initialization"] == "particlegan.init.initialize_"
                   and config["initialization_method"] == "sample_distributions_v1"
                   and config["training_schedule"] == "fresh_editing_only_v1"
                   and config["preservation_game_weight"] == 0
                   and config["output_error_guard"] is False and config["max_feature_context_harm"] == 0
                   and config["dataset_digest"] == data_digest, arm + "/game/input/init contract")
    review.require(config["recipe"] == policy["recipe"] and policy["recipe"]["birth_death_backend"] == "auto"
                   and policy["recipe"]["reopen_guard"] == "settled"
                   and policy["requires_grad"] == initial["policy"]["requires_grad"]
                   and policy["roles"] == initial["policy"]["roles"], arm + "/unchanged roles and recipe")
    review.require(len(config["sites"]) == 71 and tuple(config["sites"]) == tuple(policy["routing"]["config"]["sites"])
                   and tuple(policy["table"].shape) == (128, 4) and policy["table_requires_grad"]
                   and policy["routing"]["config"]["max_context_harm"] == 0
                   and policy["routing"]["config"].get("output_error_guard", False) is False, arm + "/routed population/guard")


def trace(path):
    text = Path(path).read_text()
    lines = text.splitlines()
    if text and not text.endswith("\n"):
        lines = lines[:-1]
    return [json.loads(line) for line in lines]


def immutable(state):
    policy = state["policy"]
    return {family: {role: {name: value for name, value in policy[family][role].items()
                            if not policy["requires_grad"][role].get(name, False)}
                     for role in ("generator", "encoder")}
            for family in ("models", "averages")}


def check_evaluation(review, result, data, label):
    """Reduce every record after checking metadata against the actual input."""
    review.require(set(result) == set(COUNTS), label + "/all pools")
    summary = {}
    for pool, count in COUNTS.items():
        value, context = result[pool], data[pool]["context"]
        records = value["records"]
        review.require(value["count"] == len(records) == len(context) == count, label + "/count")
        subjects = defaultdict(list)
        for index, record in enumerate(records):
            review.require(record["index"] == index
                           and record["source_caption_id"] == int(context[index, 4097])
                           and record["time"] == float(context[index, 4096]), label + "/context")
            review.require(all(math.isfinite(record[name]) for name in (*JUDGES, "mse_diagnostic"))
                           and record["mse_diagnostic"] >= 0
                           and all(record[name] >= 0 for name in JUDGES), label + "/finite")
            subjects[str(record["source_caption_id"])].append(record)
        review.close(value["rmse_diagnostic"], math.sqrt(sum(row["mse_diagnostic"] for row in records) / count),
                     label + "/diagnostic reduction")
        for judge in JUDGES:
            review.close(value[judge], sum(row[judge] for row in records) / count, label + "/game reduction")
        summary[pool] = dict(count=count, **{judge: value[judge] for judge in JUDGES},
            rmse_diagnostic=value["rmse_diagnostic"],
            subjects={subject: dict(contexts=len(items),
                **{judge: sum(row[judge] for row in items) / len(items) for judge in JUDGES})
                for subject, items in sorted(subjects.items())})
    review.require(set(summary["test"]["subjects"]) == {"4", "5", "15", "16", "17", "18"},
                   label + "/all six held-out subjects")
    return summary


def paired_summary(review, first, second, label):
    result = {}
    for pool, count in COUNTS.items():
        pairs = list(zip(first[pool]["records"], second[pool]["records"]))
        review.require(len(pairs) == count, label + "/paired pool")
        subjects = defaultdict(list)
        for old, new in pairs:
            review.require(all(old[key] == new[key] for key in ("index", "source_caption_id", "time")),
                           label + "/paired exact context")
            subjects[str(old["source_caption_id"])].append((old, new))
        result[pool] = {}
        for judge in JUDGES:
            changes = [new[judge] - old[judge] for old, new in pairs]
            mean = sum(changes) / count
            review.close(mean, second[pool][judge] - first[pool][judge], label + "/paired mean")
            result[pool][judge] = dict(new_minus_reference=mean,
                contexts_improved=sum(change < 0 for change in changes),
                contexts_worsened=sum(change > 0 for change in changes),
                subjects={subject: dict(contexts=len(items),
                    new_minus_reference=sum(new[judge] - old[judge] for old, new in items) / len(items))
                    for subject, items in sorted(subjects.items())})
    return result


def review_export(review, path, state, arm, step, backend_hash, expected_pins=None):
    policy = state["policy"]
    selected = policy["models"]  # The primary head remains FAST regardless of native serving.
    expected = {"generator." + name: tensor for name, tensor in selected["generator"].items()
                if policy["requires_grad"]["generator"].get(name, False)}
    expected.update({"router." + name: tensor for name, tensor in selected["router"].items() if name != "log_mass"})
    expected.update({"bank.table": policy["table"], "bank.log_mass": selected["router"]["log_mass"]})
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
        config = json.loads(metadata["config"])
        review.require(metadata["format"] == "supra_particlegan_clean_v3"
                       and config["architecture"] == "gated_particle_v3"
                       and config["served_source"] == "fast" and config["completed_steps"] == step
                       and config["extra"] == dict(particle_init=MODES[arm], training_schedule="fresh_editing_only_v1"),
                       arm + "/explicit fresh initialization and FAST export")
        review.require(config["rank"] == 16 and config["z_dim"] == 4 and config["num_particles"] == 128
                       and config["cfg"] == 3 and config["sampling"] == "clean"
                       and config["routed_geometry"] == "mass_atoms_v1"
                       and config["sites"] == state["config"]["sites"], arm + "/export geometry")
        expected_dimensions = {site: selected["router"][f"site_queries.query_{index:03d}.weight"].shape[1]
                               for index, site in enumerate(config["sites"])}
        review.require(config["site_input_dims"] == expected_dimensions, arm + "/export exact site dimensions")
        pins = json.loads(metadata["native_pins"])
        review.require(pins["backend_sha256"] == backend_hash, arm + "/export backend source")
        if expected_pins is not None:
            review.require(pins == expected_pins, arm + "/all exported model/text/VAE source pins")
        review.require(set(handle.keys()) == set(expected) and len(expected) == 428, arm + "/all428 export names")
        for name, tensor in expected.items():
            actual = handle.get_tensor(name)
            review.require(actual.dtype == tensor.dtype == torch.float32 and torch.equal(actual, tensor),
                           arm + "/exact FAST export tensor " + name)
    return dict(sha256=sha(path), config=config, native_pins=pins, tensors=len(expected))


def build_review(directory, pg_root, partial=False):
    started, review = time.monotonic(), Reviewer()
    _SHA_CACHE.clear()
    reviewer_digest = sha(__file__)
    helper_path = ROOT / "scripts/review_e22_supra_pr223.py"
    helper_digest = sha(helper_path)
    directory, pg_root = Path(directory).resolve(), Path(pg_root).resolve()
    plan = read(directory / "plan.json")
    protocol = plan["protocol"]
    review.require(plan["schema"] == "supra_sampled_hb_neutral_matched_v1"
                   and plan["particlegan_commit"] == protocol["particlegan"]["commit"] == PIN
                   and plan["fixed_updates"] == HORIZON and plan["endpoints"] == list(ENDPOINTS), "fixed law and horizons")
    review.require(protocol["training"]["external_updates_per_arm"] == HORIZON
                   and protocol["training"]["endpoint_updates"] == list(ENDPOINTS)
                   and protocol["training"]["preservation_training_updates"] == 0
                   and protocol["training"]["output_objective"] is False
                   and protocol["training"]["output_error_guard"] is False
                   and protocol["training"]["max_feature_context_harm"] == 0
                   and protocol["training"]["optimizer_interventions"] is False, "editing-only native learned game")
    for name, digest in plan["application_source_sha256"].items():
        review.require(sha(ROOT / name) == sha(directory / "source" / name) == digest, "held/archived app " + name)
    card = ROOT / "docs/e22_supra_neutral_initialization_protocol.json"
    review.require(read(card) == protocol, "exact frozen card")
    committed = committed_python_hashes(pg_root, PIN)
    review.require(committed == plan["particlegan_source_sha256"], "native committed tree")
    for name, digest in committed.items():
        review.require(sha(pg_root / name) == sha(directory / "source/native" / name) == digest, "held/archived native " + name)
    review.require(source_digest(committed) == protocol["particlegan"]["python_source_digest"], "native full source digest")
    inputs = {name: Path(path) for name, path in plan["input_paths"].items()}
    for name, path in inputs.items():
        review.require(sha(path) == plan["input_sha256"][name], "actual immutable input " + name)
    for name, entry in protocol["inputs"].items():
        review.require(sha(entry["path"]) == entry["sha256"], "declared immutable input " + name)
    data = load(inputs["data"])
    review.require(state_digest(data) == protocol["data"]["digest"], "full input data content")
    judges = read(directory / "judges.json")
    for name, key in (("D1856", "D1856"), ("D6400", "V2D6400")):
        judge = load(inputs[name])
        review.require(judge["policy"]["completed_steps"] == (1856 if name == "D1856" else 6400)
                       and state_digest(judge["policy"]["models"]["critic"]) == judges[name]
                       == protocol["evaluation"]["judges"][key]["critic_tensor_digest"], "actual fixed judge " + name)
        del judge
    stream = torch.Generator().manual_seed(72)
    panels = {}
    for pool, count in COUNTS.items():
        digest = hashlib.sha256()
        for start in range(0, count, 4):
            panel = torch.randn(4, min(4, count - start), 256, 16, generator=stream)
            digest.update(panel.contiguous().numpy().tobytes(order="C"))
        panels[pool] = digest.hexdigest()
        review.require(panels[pool] == protocol["evaluation"]["gaussian_panels"]["unscaled_gaussian_sha256_by_pool"][pool],
                       "private evaluation panel " + pool)
    historical_probe = read(directory / "historical-probes.json")
    indices, test = [], data["test"]["context"]
    for subject in test[:, 4097].unique(sorted=True):
        available = (test[:, 4097] == subject).nonzero().flatten()
        indices.extend(available[torch.linspace(0, len(available) - 1, 2).round().long()].tolist())
    probe_panels = torch.randn(4, len(indices), 256, 16, generator=torch.Generator().manual_seed(72))
    review.require(indices == historical_probe["indices"] and len(indices) == 12
                   and state_digest(test[indices]) == historical_probe["context_digest"]
                   and state_digest(probe_panels) == historical_probe["panel_digest"], "predeclared progress contexts/panels")

    initial_paths = {arm: directory / arm / "checkpoint-00000.pt" for arm in ARMS}
    initial = {arm: load(path) for arm, path in initial_paths.items()}
    left, right = (initial[arm] for arm in ARMS)
    configs = [{key: value for key, value in item["config"].items() if key != "particle_init"} for item in (left, right)]
    review.require(configs[0] == configs[1], "initial configs differ only initialization tag")
    changed = []
    for family in ("models", "averages"):
        for name, value in left["policy"][family]["generator"].items():
            candidate = right["policy"][family]["generator"][name]
            review.require(value.shape == candidate.shape and value.dtype == candidate.dtype, "initial tensor schema")
            if name.endswith("bridge.weight"):
                review.require(value[:, :16].count_nonzero() > 0 and candidate[:, :16].count_nonzero() == 0
                               and torch.equal(value[:, 16:], candidate[:, 16:]) and candidate[:, 16:].count_nonzero() > 0,
                               "only H zero, C bit-exact/nonzero")
                changed.append(family + "/" + name + "[H]")
            elif name.endswith("bridge.bias"):
                review.require(value.count_nonzero() > 0 and candidate.count_nonzero() == 0, "only b zero")
                changed.append(family + "/" + name)
            else:
                review.require(torch.equal(value, candidate), "other initial G tensor " + name)
                if name.endswith("up.weight"):
                    review.require(value.count_nonzero() == 0, "fresh zero up")
        owners = [{key: value for key, value in item["policy"][family].items() if key != "generator"}
                  for item in (left, right)]
        review.require(state_digest(owners[0]) == state_digest(owners[1]), "all initial other models " + family)
    for section in ("policy", "caller"):
        values = [{key: value for key, value in (item["policy"] if section == "policy" else item).items()
                   if key not in (("models", "averages") if section == "policy" else ("policy", "config"))}
                  for item in (left, right)]
        review.require(state_digest(values[0]) == state_digest(values[1]), "all initial other " + section + " state/streams")
    review.require(len(changed) == 284, "all71 FAST/EMA H and b slices")
    old_review = read(directory / "independent-initial-review.json")
    review.require(old_review["qualified_initial_artifacts"] is True
                   and old_review["plan_sha256"] == sha(directory / "plan.json")
                   and old_review["declared_changed_coordinates"] == changed, "preserved independent initial review")
    for arm, path in initial_paths.items():
        review.require(sha(path) == old_review["initial_checkpoint_sha256"][arm], "qualified unchanged initial file " + arm)
    frozen = {arm: state_digest(immutable(item)) for arm, item in initial.items()}
    review.require(len(set(frozen.values())) == 1, "initial identical four frozen owners")
    teacher = load(inputs["teacher"])
    teacher_tensors = teacher["policy"]["models"]["encoder"]
    for arm, state in initial.items():
        for family in ("models", "averages"):
            for name, value in state["policy"][family]["encoder"].items():
                if name.startswith("teacher."):
                    review.require(torch.equal(value, teacher_tensors[name]), arm + "/frozen teacher tensor")
    del teacher, teacher_tensors
    initial_meta = {arm: dict(config=item["config"], policy=dict(
        requires_grad=item["policy"]["requires_grad"], roles=item["policy"]["roles"]),
        bank=item["policy"]["table"].clone(),
        router={name: tensor.clone() for name, tensor in item["policy"]["models"]["router"].items()})
        for arm, item in initial.items()}
    del initial, left, right, owners, values, candidate, value, state
    gc.collect()

    traces = {arm: trace(directory / arm / "train.jsonl") for arm in ARMS}
    common = min(map(len, traces.values()))
    review.require(common <= HORIZON, "bounded external updates")
    if not partial:
        review.require(all(len(items) == HORIZON for items in traces.values()), "both complete fixed-horizon traces")
    cpu_stream = torch.Generator().manual_seed(7)
    expected_rngs = {0: cpu_stream.get_state().clone()}
    for step in range(1, common + 1):
        expected = torch.randint(240, (4,), generator=cpu_stream).tolist()
        if step in CHECKPOINTS:
            expected_rngs[step] = cpu_stream.get_state().clone()
        a, b = (traces[arm][step - 1] for arm in ARMS)
        for row in (a, b):
            review.require(row["step"] == step and row["batch_indices"] == expected
                           and row["hold"] is False and row["game_weight"] == 1., "CPU7 editing update " + str(step))
            review.require(all(math.isfinite(row[name]) for name in
                               ("loss_g", "loss_d", "loss_d_game", "penalty", "bank_grad_norm", "output_sigma")),
                           "finite native losses/gradients " + str(step))
            review.require(row["penalty_calls"] == step and row["dense_gradient_rows"] == (0 if step == 1 else 128)
                           and (step == 1 or row["bank_grad_norm"] > 0), "native penalty/dense particle gradient " + str(step))
        review.require(all(a[name] == b[name] for name in ("batch_indices", "base_noise_sums", "paired_rng_digest", "hold", "game_weight")),
                       "matched data/Gaussian signatures " + str(step))

    checkpoints, exports, progress = {}, {}, {}
    for arm in ARMS:
        checkpoints[arm] = {}
        for step in sorted(CHECKPOINTS):
            path = directory / arm / f"checkpoint-{step:05d}.pt"
            if partial and (step > common or not path.exists()):
                continue
            state = load(path)
            policy, config = state["policy"], state["config"]
            check_checkpoint_contract(review, state, initial_meta[arm], plan["configs"][arm],
                arm, step, protocol["data"]["digest"])
            review.require(state_digest(immutable(state)) == frozen[arm], arm + "/four immutable frozen owners")
            for family in ("models", "averages"):
                encoder = policy[family]["encoder"]
                review.require(torch.equal(encoder["contexts"], data["text_contexts"])
                               and torch.equal(encoder["masks"], data["text_masks"]), arm + "/frozen text/mask")
            review.require(torch.equal(policy["models"]["critic"]["scale"], data["coordinate_scale"]), arm + "/coordinate scale")
            review.require(torch.equal(policy["optimizers"][1]["regularizer"]["ema"]["scale"], data["coordinate_scale"]),
                           arm + "/averaged critic coordinate scale")
            review.require(torch.equal(state["data_rng"], expected_rngs[step]), arm + "/exact CPU7 checkpoint stream")
            for family in ("models", "averages"):
                for role in ("generator", "router"):
                    for name, tensor in policy[family][role].items():
                        if policy["requires_grad"][role].get(name, False):
                            review.require(torch.isfinite(tensor).all().item(), arm + "/finite trainable parameter")
            review.require(torch.isfinite(policy["table"]).all().item()
                           and torch.isfinite(policy["averaged_table"]).all().item(), arm + "/finite FAST/averaged bank")
            checkpoints[arm][str(step)] = dict(file_sha256=sha(path), native_digest=state_digest(state),
                paired_rng_digest=state_digest(state["paired_noise_rng"]), native_moves=policy["routing"]["counters"]["moves"])
            if step in ENDPOINTS and (not partial or (directory / arm / f"adapter-{step:05d}.safetensors").exists()):
                backend_hash = read(ROOT / "backend.lock.json")["sha256"]
                expected_pins = dict(model_id=protocol["host"]["model_id"], model_revision=protocol["host"]["model_revision"],
                    text_encoder_id="google/flan-t5-base", text_encoder_revision=protocol["host"]["text_encoder_revision"],
                    vae_id="stabilityai/sd-vae-ft-mse", vae_revision=protocol["host"]["vae_revision"], backend_sha256=backend_hash)
                exports[arm + "@" + str(step)] = review_export(review, directory / arm / f"adapter-{step:05d}.safetensors",
                    state, arm, step, backend_hash, expected_pins)
            if step == HORIZON:
                review.require(state_digest(policy["table"]) != state_digest(initial_meta[arm]["bank"])
                               and state_digest(policy["models"]["router"]) != state_digest(initial_meta[arm]["router"]),
                               arm + "/learned bank and router")
                g, flags = policy["models"]["generator"], policy["requires_grad"]["generator"]
                bridges = [name for name in g if name.endswith("bridge.weight")]
                review.require(len(bridges) == 71 and all(g[name][:, 16:].count_nonzero() > 0 and flags[name]
                               and flags[name.replace("weight", "bias")] for name in bridges), arm + "/retained trainable71 C/H/b")
                del g, flags
                live_queries = [name for name, tensor in policy["models"]["router"].items()
                                if name.startswith("site_queries.")
                                and not torch.equal(tensor, initial_meta[arm]["router"][name])]
                checkpoints[arm][str(step)]["changed_site_query_parameters"] = live_queries
            del state, policy, config, encoder, tensor
            gc.collect()
        monitor = trace(directory / arm / "monitor/progress.jsonl")
        eligible = [item for item in monitor if item["step"] <= common]
        expected_probes = list(range(0, (common // 200) * 200 + 1, 200))
        if partial:
            expected_probes = expected_probes[:len(eligible)]
        review.require([item["step"] for item in eligible] == expected_probes, arm + "/fixed probes")
        review.require(read(directory / arm / "monitor/progress-edit-indices.json") == indices, arm + "/shared held-out probes")
        for item in eligible:
            review.require(item["source"] == "FAST current particle G" and item["output_metrics_used"] is False
                           and item["evaluation_only"] is True and item["native_state_unchanged"] is True
                           and set(item["probes"]) == {"edit"} and item["probes"]["edit"]["contexts"] == 12, arm + "/observational clean/DV12 probes")
            review.require(all(math.isfinite(item["probes"]["edit"][mode][judge]) for mode in ("clean", "dv12")
                               for judge in ("frozen_start_D", "live_D")), arm + "/finite labeled progress games")
            if item["step"]:
                rolling = item["rolling"]["edit"]
                source = traces[arm][item["step"] - 100:item["step"]]
                review.require(rolling["updates"] == len(source) == 100
                               and rolling["first_step"] == source[0]["step"] and rolling["last_step"] == source[-1]["step"], arm + "/rolling window")
                for field, key in (("g_game", "loss_g"), ("d_game", "loss_d_game"), ("penalty", "penalty"), ("bank_grad_norm", "bank_grad_norm")):
                    review.close(rolling[field], sum(row[key] for row in source) / len(source), arm + "/rolling reduction")
        progress[arm] = eligible
    for step in set(checkpoints[ARMS[0]]) & set(checkpoints[ARMS[1]]):
        review.require(checkpoints[ARMS[0]][step]["paired_rng_digest"] == checkpoints[ARMS[1]][step]["paired_rng_digest"],
                       "matched checkpoint paired CUDA stream")
    if not partial:
        review.require(all(set(items) == {str(step) for step in CHECKPOINTS} and len(items) == 19 for items in checkpoints.values()),
                       "all38 predeclared checkpoints")

    historical = read(directory / "historical-references.json")
    review.require(set(historical) == {"ordinary_lora_6400", "historical_particle_v2", "historical_particle_v3"}, "declared historical references")
    evaluations = {name: check_evaluation(review, result, data, name) for name, result in historical.items()}
    comparisons = {}
    for step in ENDPOINTS:
        endpoints = {}
        for arm in ARMS:
            path = directory / arm / f"evaluation-{step:05d}.json"
            ablation_path = directory / arm / f"particle-ablations-{step:05d}.json"
            if partial and (not path.exists() or not ablation_path.exists()):
                continue
            result = read(path)
            endpoints[arm] = result
            evaluations[arm + "@" + str(step)] = check_evaluation(review, result, data, arm + "@" + str(step))
            ablation = read(ablation_path)
            for key in ("code_scores", "mass_only_scores"):
                evaluations[arm + "@" + str(step) + "/" + key] = check_evaluation(review, ablation[key], data, arm + "/" + key)
            for name, key, field in (("zero_code", "code_scores", "zero_code_minus_live_test_game"),
                                     ("mass_only", "mass_only_scores", "mass_only_minus_live_test_game")):
                for judge in JUDGES:
                    review.close(ablation[field][judge], ablation[key]["test"][judge] - result["test"][judge], arm + "/ablation change " + name)
            for name, reference in historical.items():
                comparisons[arm + "@" + str(step) + "-" + name] = paired_summary(review, reference, result, arm + "/historical " + name)
        if len(endpoints) == 2:
            comparisons["neutral-control@" + str(step)] = paired_summary(review, endpoints[ARMS[0]], endpoints[ARMS[1]], "neutral-control")

    runtime = None
    if not partial:
        receipt, status = read(directory / "receipt.json"), read(directory / "status.json")
        review.require(receipt["status"] == status["phase"] == "complete" and status["step"] == status["steps"] == HORIZON
                       and status["editing_updates"] == HORIZON and status["preservation_updates"] == 0
                       and receipt["plan"] == plan and all(value is True for value in receipt["checks"].values()), "complete held GPU qualification")
        review.require(receipt["seconds"] <= 14400 and all(value <= 7200 for value in receipt["charged_seconds"].values()), "execution budgets")
        for arm in ARMS:
            review.require(receipt["recovery"][arm] == dict(rows_exact=True, state_exact=True, from_step=800, to_step=802), arm + "/held GPU recovery")
            review.require(receipt["final_native_digests"][arm] == checkpoints[arm]["6400"]["native_digest"]
                           and receipt["final_checkpoint_sha256"][arm] == checkpoints[arm]["6400"]["file_sha256"], arm + "/actual final native file/state")
            actual_coverage = dict(live_bank_updates=sum(row["dense_gradient_rows"] > 0 for row in traces[arm]),
                dense_128_row_updates=sum(row["dense_gradient_rows"] == 128 for row in traces[arm]),
                moves=sum((row["move"] or {}).get("moves", 0) for row in traces[arm]))
            review.require(receipt["coverage"][arm] == actual_coverage
                           and actual_coverage["moves"] == checkpoints[arm]["6400"]["native_moves"], arm + "/actual particle coverage")
            for step in ENDPOINTS:
                result = read(directory / arm / f"evaluation-{step:05d}.json")
                review.require(receipt["endpoint_test_scores"][arm + "@" + str(step)] == result["test"], arm + "/receipt endpoint records")
                for prefix in ("full_evaluation_immutable_", "code_ablation_immutable_", "mass_only_all_sites_",
                               "export_reload_raw_exact_", "export_reload_state_immutable_"):
                    review.require(receipt["checks"][prefix + arm + str(step)] is True, arm + "/held endpoint forward/state witness")
        runtime = dict(receipt_sha256=sha(directory / "receipt.json"), seconds=receipt["seconds"],
            charged_seconds=receipt["charged_seconds"], recovery=receipt["recovery"],
            qualification="Archived source and actual GPU witnesses; no CUDA forward/recovery re-execution by CPU reviewer.")
    review.require(sha(__file__) == reviewer_digest and sha(helper_path) == helper_digest,
                   "reviewer source/helper unchanged during review")
    return dict(schema="supra_neutral_initialization_independent_review_v1", qualified=not partial,
        partial=partial, checks=review.checks, reviewer_sha256=reviewer_digest,
        reviewer_dependency_sha256={str(helper_path): helper_digest}, plan_sha256=sha(directory / "plan.json"),
        protocol_sha256=sha(card), application_source_sha256=plan["application_source_sha256"],
        native_source_digest=source_digest(committed), input_sha256=plan["input_sha256"],
        common_matched_rows=common, trace_rows={arm: len(items) for arm, items in traces.items()},
        trace_prefix_digests={arm: state_digest(items[:common]) for arm, items in traces.items()},
        checkpoint_artifacts=checkpoints, declared_initial_changed_coordinates=changed, judges=judges,
        full_evaluation_panel_sha256=panels, progress=progress, evaluation_summaries=evaluations,
        comparisons=comparisons, export_artifacts=exports, gpu_runtime_witnesses=runtime,
        qualification_credit="CPU provenance/ownership/reduction only; no Forge/default or general task improvement claim.",
        native_or_generator_updates_by_reviewer=0, model_forward_calls_by_reviewer=0,
        output_metrics_used_for_optimizer_or_selection=False, review_seconds=time.monotonic() - started,
        external_cpu_review_excluded_from_gpu_runner_budget=True,
        gpu_runner_budget_scope="Training, checkpoints, probes, endpoint scoring, recovery, and shared references; external CPU qualification reported separately.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "outputs/e22-supra-neutral-initialization-6400")
    parser.add_argument("--particlegan-root", type=Path, default=ROOT.parent / "ParticleGAN-supra-neutral-develop")
    parser.add_argument("--partial", action="store_true", help="Inspect active prefix without qualifying completion.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    report = build_review(args.run, args.particlegan_root, partial=args.partial)
    target = args.output or args.run / ("independent-partial-review.json" if args.partial else "independent-review.json")
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(target)
    print(json.dumps(dict(qualified=report["qualified"], partial=report["partial"], checks=report["checks"],
        common_matched_rows=report["common_matched_rows"], review_seconds=report["review_seconds"], output=str(target))))


if __name__ == "__main__":
    main()
