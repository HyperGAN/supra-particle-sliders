#!/usr/bin/env python3
"""Independent CPU tensor-only qualification of expanded Supra data.

No application/native API imports, model reconstruction, teacher replay, updates,
quality scores or distributional-independence claim. Root freezes this law
before generation and later binds actual output hashes in --descriptor.
"""
import time
STARTED = time.monotonic()
import argparse
import hashlib
import json
import os
from pathlib import Path
import traceback

TASK = "supra_latent_coverage_saved_data_review_v1"
GENERATION = "supra_latent_coverage_data_preparation_v1"
SCHEMA = "supra_latent_coverage_data_v1"
SOURCES = (18, 4, 17, 5, 16, 15)
TARGETS = (11, 3, 2, 13, 12, 14)
TRAIN_SEEDS = tuple(range(7063, 7071))
TEST_SEEDS = (39001, 39002)
COUNTS = dict(teacher_B4_forwards=1080, Euler_B4_forwards=900,
              opposite_caption_B4_forwards=180, private_seed_draws=6,
              native_updates=0, optimizer_steps=0)
LIMIT, RESERVE = 120, 10
torch = None
CHECKS = 0


class CheckFailed(Exception):
    pass


def require(value, message):
    global CHECKS
    CHECKS += 1
    if not value:
        raise CheckFailed(message)


def deadline():
    if time.monotonic() - STARTED > LIMIT - RESERVE:
        raise TimeoutError("CPU120 saved-data review computation reserve")


def sha(path, *, reserve=True):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
            if reserve: deadline()
    return digest.hexdigest()


def write(path, value, *, replace=False):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp") if replace else path
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write("\n")
    if replace:
        temporary.replace(path)


def state_digest(value):
    """Exact independent copy of the qualified nested state digest LAW.

Tensor shape/dtype text + contiguous raw bytes; sorted dict key text;
list/tuple type text; scalar repr. No app helper is imported or called.
"""
    digest = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                digest.update(str(key).encode()); visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode())
            for child in item:
                visit(child)
        else:
            digest.update(repr(item).encode())
        deadline()
    visit(value)
    return digest.hexdigest()


def exact(a, b, name):
    if isinstance(a, torch.Tensor):
        require(isinstance(b, torch.Tensor) and a.shape == b.shape and a.dtype == b.dtype
                and a.device.type == b.device.type == "cpu", name + " tensor schema")
        require(torch.equal(a.contiguous().reshape(-1).view(torch.uint8),
                            b.contiguous().reshape(-1).view(torch.uint8)), name + " tensor bytes")
    elif isinstance(a, dict):
        require(isinstance(b, dict) and a.keys() == b.keys(), name + " keys")
        for key in a:
            exact(a[key], b[key], name + "/" + str(key))
    elif isinstance(a, (list, tuple)):
        require(type(a) is type(b) and len(a) == len(b), name + " sequence schema")
        for i, (left, right) in enumerate(zip(a, b)):
            exact(left, right, name + "/" + str(i))
    else:
        require(type(a) is type(b) and a == b, name + " scalar")
    deadline()


def tensor(value, shape, dtype, name):
    require(isinstance(value, torch.Tensor) and value.device.type == "cpu"
            and tuple(value.shape) == shape and value.dtype == dtype, name + " schema")
    require(bool(torch.isfinite(value).all()), name + " finite")


def finite_tree(value):
    if isinstance(value, torch.Tensor):
        require(value.device.type == "cpu", "data tensor CPU")
        if value.is_floating_point() or value.is_complex():
            require(bool(torch.isfinite(value).all()), "data tensor finite")
    elif isinstance(value, dict):
        for child in value.values(): finite_tree(child)
    elif isinstance(value, (tuple, list)):
        for child in value: finite_tree(child)
    deadline()


def bind(items, result):
    for item in items.values():
        path = str(Path(item["path"]).resolve(strict=True))
        actual = result.setdefault(path, item["sha256"])
        require(actual == item["sha256"], "conflicting byte aliases")
    return result


def envelope(descriptor):
    inputs = descriptor["inputs"]
    path = lambda name: Path(inputs[name]["path"])
    card = json.loads(path("generation_card").read_text())
    report = json.loads(path("generation_report").read_text())
    completion = json.loads(path("generation_completion").read_text())
    external = json.loads(path("generation_external").read_text())
    manifest = json.loads(path("generation_manifest").read_text())
    expected = {}
    bind(descriptor["sources"], expected); bind(inputs, expected)
    bind({name: dict(path=name, sha256=digest) for name,digest in manifest["files"].items()}, expected)
    for group in ("sources", "dependencies", "inputs"):
        bind(card[group], expected)
    require(str(Path(__file__).resolve()) in expected, "independent reader source bound")
    require(card["id"] == GENERATION and card["seconds"] == 600
            and card["precision"] == "student_final_bf16_v1", "fixed generation cohort")
    for name in ("data", "neutral_checkpoint"):
        require(inputs[name] == card["inputs"][name], "original input alias " + name)
    for name, value in expected.items():
        require(sha(name) == value, "authenticated bytes " + name)
    require(report["task"] == completion["task"] == GENERATION, "producer task")
    require(report["complete"] is completion["complete"] is True
            and report["qualification"] == completion["qualification"] == "PASS"
            and report["scientific_status"] is completion["scientific_status"] is None,
            "completed null-science producer")
    require(completion["exit_code"] == 0
            and completion["report_sha256"] == inputs["generation_report"]["sha256"], "producer companion")
    card_sha = inputs["generation_card"]["sha256"]
    require(report["protocol_sha256"] == completion["protocol_sha256"] == card_sha, "producer card")
    require(report["artifact_sha256"] == completion["artifact_sha256"] == inputs["artifact"]["sha256"], "artifact chain")
    require(Path(card["output"]).resolve() == path("artifact").resolve().parent
            and path("artifact").name == "latent-coverage-data.pt", "declared output")
    require(external["complete"] is True and external["exit_code"] == 0
            and external["timed_out"] is False and external["child_qualification"] == "PASS"
            and external["child_scientific_status"] is None
            and external["child_completion_sha256"] == inputs["generation_completion"]["sha256"], "qualified external producer")
    require(external["limit_seconds"] == 600 and external["seconds"] <= 600
            and external["manifest_sha256"] == inputs["generation_manifest"]["sha256"], "generation cap/manifest")
    require(Path(external["manifest"]).resolve() == path("generation_manifest").resolve(), "manifest path")
    require(manifest["seconds"] == 600 and Path(manifest["output"]).resolve() == Path(card["output"]).resolve(), "manifest cohort")
    require(external["checks_before"] and external["checks_before"] == external["checks_after"]
            and set(external["checks_before"]) == set(manifest["files"])
            and all(external["checks_before"].values()), "external byte guards")
    require(manifest["files"].get(str(path("generation_card").resolve())) == card_sha,
            "outer-bound generation card")
    for name, digest in expected.items():
        # Reader-only sources/descriptor may be finalized after generation;
        # the complete original generation closure must be outer-bound.
        if name in report["source_input_bindings"]:
            require(report["source_input_bindings"][name] == digest
                    and manifest["files"].get(name) == digest
                    and external["checks_before"].get(name) is True, "producer closure " + name)
    declared = {}
    for group in ("sources", "dependencies", "inputs"): bind(card[group], declared)
    require(report["source_input_bindings"] == declared, "complete producer input/source inventory")
    require(report["native_before_after_restore_exact"] is True
            and report["native_updates"] == report["optimizer_steps"] == report["training_student_forwards"] == 0
            and report["future_training_eligible"] is False, "producer zero-work/rollback proofs")
    return card, report, expected


def inspect(descriptor, card, report):
    load = lambda name: torch.load(descriptor["inputs"][name]["path"], map_location="cpu", weights_only=True, mmap=True)
    original, artifact, parent = load("data"), load("artifact"), load("neutral_checkpoint")
    require(artifact["schema"] == SCHEMA, "production artifact, not software fixture")
    data, m = artifact["data"], artifact["manifest"]
    exact(m, report["manifest"], "retained producer manifest")
    require(set(data) == set(original), "data top-level schema preserved")
    finite_tree(original); finite_tree(data)
    old_digest, expanded_digest = state_digest(original), state_digest(data)
    require(old_digest == card["data_digest"] == m["original_dataset_digest"], "original full data digest")
    require(expanded_digest == m["expanded_dataset_digest"], "expanded full data digest")
    require(state_digest(parent) == card["neutral_native_digest"], "authenticated complete original12800 state")
    config = parent["config"]
    require(parent["policy"]["completed_steps"] == 12800 and config["dataset_digest"] == old_digest
            and config["training_schedule"] == "fresh_editing_only_v1"
            and config["preservation_game_weight"] == 0. and config["branch_rank"] == 16
            and len(config["sites"]) == 71, "original config/clock geometry")
    encoder = parent["policy"]["models"]["encoder"]
    require(state_digest(encoder) == m["encoder_tensor_digest"], "original frozen encoder tensor digest")
    exact(original["text_contexts"], encoder["contexts"], "original encoder text")
    exact(original["text_masks"], encoder["masks"], "original encoder masks")
    bootstrap = m["provenance"]["bootstrap"]
    exact(bootstrap, report["original12800_bootstrap"], "bootstrap receipt")
    require(bootstrap["original_native_digest"] == card["neutral_native_digest"]
            and bootstrap["legacy_roundtrip_exact"] is True and bootstrap["completed_steps"] == 12800
            and bootstrap["initialized_after_restore"] is False
            and bootstrap["precision"]["mode"] == card["precision"], "source-qualified public origin proof")
    require(m["provenance"]["protocol_sha256"] == descriptor["inputs"]["generation_card"]["sha256"]
            and m["provenance"]["input_bindings"] == card["inputs"]
            and m["provenance"]["source_identity"] == card["sources"], "artifact provenance")
    for key in original:
        if key != "fit": exact(original[key], data[key], "non-FIT/" + key)
    indices = [160*s + 20*j + 10*p + k for s in range(6) for j in range(2) for p in range(2) for k in range(10)]
    require(m["original_indices"] == indices and m["schema"] == SCHEMA and m["software_only"] is False
            and m["source_order"] == list(SOURCES) and m["target_order"] == list(TARGETS)
            and m["original_seeds"] == [7063, 7064] and m["train_seeds"] == list(TRAIN_SEEDS)
            and m["TEST_seeds"] == list(TEST_SEEDS) and set(TRAIN_SEEDS).isdisjoint(TEST_SEEDS)
            and m["index_law"] == "160*s+20*j+10*p+k" and m["paths"] == ["neutral", "positive"]
            and m["Euler_steps"] == 50 and m["recorded_steps"] == list(range(0, 50, 5))
            and m["cfg"] == 3. and m["builder_batch"] == 4 and m["counts"] == COUNTS, "fixed960 mapping/call declarations")
    require(set(data["fit"]) == set(original["fit"]), "FIT field schema preserved")
    for name in ("context", "base", "targets", "hold"):
        exact(original["fit"][name], data["fit"][name][indices], "original240/" + name)
    for name in ("prompts", "target_prompts"):
        exact(original["fit"][name], [data["fit"][name][i] for i in indices], "original240/" + name)
    for split, n, seeds in ((original["fit"], 240, (7063,7064)), (original["test"],240,TEST_SEEDS), (data["fit"],960,TRAIN_SEEDS)):
        tensor(split["context"], (n,4100), torch.float32, "contexts")
        for name in ("base", "targets"): tensor(split[name], (n,4,32,32), torch.float32, name)
        tensor(split["hold"], (n,), torch.bool, "editing flags")
        require(not bool(split["hold"].any()), "editing-only FIT/TEST")
        exact(torch.arange(n), split["source_indices"], "canonical source indices")
        require(len(split["prompts"]) == len(split["target_prompts"]) == n, "prompt count")
        for s, (source, target) in enumerate(zip(SOURCES, TARGETS)):
            for j in range(len(seeds)):
                for p in range(2):
                    for k in range(10):
                        i = 20*len(seeds)*s + 20*j + 10*p + k
                        exact(torch.tensor([k/10.,source,target,1.]), split["context"][i,4096:], "ordered row tail")
                        require(split["prompts"][i] == original["fit"]["prompts"][40*s]
                                and split["target_prompts"][i] == original["fit"]["target_prompts"][40*s], "source/target prompt pairing")
        for j in range(len(seeds)):
            initial = split["context"][20*j,:4096]
            for s in range(6):
                for p in range(2):
                    exact(initial, split["context"][20*len(seeds)*s+20*j+10*p,:4096], "global seed reuse at time0")
    tensor(data["guard"]["context"], (64,4100), torch.float32, "fixed guard64")
    tensor(data["coordinate_scale"], (16,), torch.float32, "original scale16")
    def row_hashes(context, width):
        return [hashlib.sha256(row[:width].contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() for row in context]
    fit_context, test_context = row_hashes(data["fit"]["context"],4100), row_hashes(original["test"]["context"],4100)
    fit_latent, test_latent = row_hashes(data["fit"]["context"],4096), row_hashes(original["test"]["context"],4096)
    require(not set(fit_context).intersection(test_context), "no exact packed FIT/TEST overlap")
    require(not set(fit_latent).intersection(test_latent), "no exact latent FIT/TEST overlap")
    initial_hashes = [fit_latent[20*j] for j in range(8)]
    require(len(set(initial_hashes)) == 8, "eight distinct TRAIN initial latent values")
    return dict(original_dataset_digest=old_digest, expanded_dataset_digest=expanded_digest,
        manifest_digest=state_digest(m), original240_exact=True, all_non_FIT_exact=True,
        original_guard_scale_text_prompts_exact=True, FIT_rows=960, TEST_rows=240,
        original_subset_indices=indices, source_order=list(SOURCES), target_order=list(TARGETS),
        train_seeds=list(TRAIN_SEEDS), TEST_seeds=list(TEST_SEEDS), metadata_rows=960,
        global_time0_seed_reuse_exact=True, declared_teacher_calls=COUNTS,
        exact_FIT_TEST_context_overlap=0, exact_FIT_TEST_latent_overlap=0,
        distinct_FIT_contexts=len(set(fit_context)), distinct_FIT_latents=len(set(fit_latent)),
        artifact_tensor_digest_verified=True, original_config_tensor_digest_verified=True)


def main():
    global torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--descriptor", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(); out = None; result = None; error = None; code = 2
    bindings = {}; descriptor_sha = None; rng = None; threads = None
    try:
        descriptor = json.loads(args.descriptor.read_text()); descriptor_sha = sha(args.descriptor)
        require(descriptor["id"] == TASK and descriptor["seconds"] == LIMIT
                and descriptor["final_reserve_seconds"] == RESERVE
                and Path(descriptor["output"]).resolve() == args.out.resolve(), "fixed CPU descriptor")
        require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CUDA hidden")
        if args.out.exists(): raise FileExistsError("exclusive output required; earlier attempt preserved")
        args.out.mkdir(parents=True, exist_ok=False); out = args.out
        card, producer, bindings = envelope(descriptor)
        import torch as tensor_library
        torch = tensor_library
        require(not torch.cuda.is_initialized(), "CUDA never initialized")
        rng, threads = torch.get_rng_state().clone(), torch.get_num_threads()
        torch.set_num_threads(1)
        result = inspect(descriptor, card, producer)
        require(torch.equal(rng,torch.get_rng_state()), "global CPU RNG unchanged")
        code = 0
    except CheckFailed as exc:
        traceback.print_exc(); error = dict(type=type(exc).__name__,message=str(exc)); code = 1
    except BaseException as exc:
        traceback.print_exc(); error = dict(type=type(exc).__name__,message=str(exc)); code = 2
    finally:
        if torch is not None and rng is not None:
            if not torch.equal(rng,torch.get_rng_state()) or torch.cuda.is_initialized():
                error = dict(type="CallerStateError",message="reader altered RNG/CUDA"); code = 2
            torch.set_rng_state(rng); torch.set_num_threads(threads)
    if out is None: return 2
    try:
        for path,digest in bindings.items(): require(sha(path) == digest, "final unchanged bytes " + path)
        require(sha(args.descriptor) == descriptor_sha, "descriptor unchanged"); deadline()
    except BaseException as exc:
        error = dict(type=type(exc).__name__,message=str(exc)); code = 2
    report = dict(task=TASK,complete=code in (0,1),qualification={0:"PASS",1:"FAIL",2:"INCOMPLETE"}[code],
        scientific_status=None,checks=CHECKS,descriptor_sha256=descriptor_sha,source_input_bindings=bindings,
        result=result,error=error,seconds=time.monotonic()-STARTED,limit_seconds=LIMIT,
        models=0,model_forwards=0,teacher_forwards=0,ParticleGAN_API_calls=0,native_updates=0,optimizer_steps=0,CUDA=False,
        limits="Tensor equality/digests and no exact TEST overlap are independent. Seed/path realization,1080 actual teacher calls, pure-teacher execution and native/class/caller rollback are authenticated producer/source proofs. No model replay, quality verdict, statistical-independence or convergence claim.")
    write(out/"report.json",report)
    completion=dict(task=TASK,complete=report["complete"],qualification=report["qualification"],scientific_status=None,
        report_sha256=sha(out/"report.json",reserve=False),descriptor_sha256=descriptor_sha,exit_code=code,error=error,
        seconds=time.monotonic()-STARTED,limit_seconds=LIMIT)
    write(out/"completion.json",completion)
    if time.monotonic()-STARTED > LIMIT:
        error=dict(type="FinalWriteDeadline",message="startup-through-final-writes CPU120 cap exceeded")
        report.update(complete=False,qualification="INCOMPLETE",error=error,seconds=time.monotonic()-STARTED)
        write(out/"report.json",report,replace=True)
        completion.update(complete=False,qualification="INCOMPLETE",exit_code=2,error=error,
            report_sha256=sha(out/"report.json",reserve=False),seconds=time.monotonic()-STARTED)
        write(out/"completion.json",completion,replace=True);code=2
    return code


if __name__ == "__main__":
    raise SystemExit(main())
