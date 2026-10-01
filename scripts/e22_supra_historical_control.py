"""Read-only reconstruction of the qualified historical V2 control program.

The trace files are segments, not full trajectories. Explicit source versions,
review/receipt file hashes, complete native boundary digests and owned stream
witnesses link them; no historical file is rewritten or schema migrated.
"""
from pathlib import Path
import json

PREFIX_PIN = "f459cb6d6aaaabeb1af076ec53ad7a963618de90"
SUFFIX_PIN = "cabe2084284db923d525918cbf3e18de6f20faac"
PROGRAM_KEYS = ("step", "hold", "batch_indices", "base_noise_sums", "paired_rng_digest", "dv12_rng_digest")
PAIRED_KEYS = PROGRAM_KEYS[:-1]


def historical_inputs(prefix, middle, suffix):
    paths = {}
    for label, directory in (("prefix", Path(prefix)), ("middle", Path(middle)), ("suffix", Path(suffix))):
        for name in ("plan", "receipt", "run", "trace", "checkpoint", "source_manifest"):
            filename = {"trace": "train.jsonl", "checkpoint": "final.pt", "source_manifest": "source/sha256.json"}.get(name, name + ".json")
            paths[f"control_{label}_{name}"] = directory / filename
    paths["control_prefix_review"] = Path(prefix) / "qualification-review.json"
    paths["control_middle_review"] = Path(middle).parent / "qualification-comparison.json"
    paths["control_middle_sampling"] = Path(middle) / "sampling-program.json"
    paths["control_suffix_review"] = Path(suffix) / "qualification-review.json"
    return paths


def reconstruct(paths, data, *, torch, state_digest, sha, committed_python_hashes, repository, require):
    """Return full immutable-record program and its bounded provenance report.

    The caller supplies its content-digest implementation. The independent CPU
    reviewer supplies the reviewer digest, not the live training checkpoint code.
    """
    def check(condition, message):
        require(bool(condition), "historical_control_" + message)

    def read(label, name):
        return json.loads(Path(paths[f"control_{label}_{name}"]).read_text())

    def trace(label):
        return [json.loads(line) for line in Path(paths[f"control_{label}_trace"]).read_text().splitlines() if line.strip()]

    plans = {label: read(label, "plan") for label in ("prefix", "middle", "suffix")}
    receipts = {label: read(label, "receipt") for label in plans}
    runs = {label: read(label, "run") for label in plans}
    reviews = {label: read(label, "review") for label in plans}
    states = {label: torch.load(paths[f"control_{label}_checkpoint"], map_location="cpu", weights_only=False, mmap=True)
              for label in plans}
    parts = {label: trace(label) for label in plans}
    expected_ranges = dict(prefix=(1, 1600), middle=(1601, 1856), suffix=(1857, 6400))
    check(reviews["prefix"]["all_checks_passed"] and reviews["middle"]["all_checks_passed"]
          and reviews["suffix"]["qualified"], "qualified_contributors")
    hashes = {key: sha(path) for key, path in paths.items()}
    for name, expected in reviews["prefix"]["file_hashes"].items():
        key = {"plan.json": "plan", "receipt.json": "receipt", "run.json": "run", "train.jsonl": "trace", "final.pt": "checkpoint"}.get(name)
        if key is not None:
            check(hashes[f"control_prefix_{key}"] == expected, "prefix_review_link_" + key)
    middle_review = reviews["middle"]["arms"]["latest"]
    for name, expected in middle_review["file_sha256"].items():
        key = {"plan.json": "plan", "receipt.json": "receipt", "train.jsonl": "trace", "final.pt": "checkpoint"}.get(name)
        if key is not None:
            check(hashes[f"control_middle_{key}"] == expected, "middle_review_link_" + key)
    for name, expected in reviews["suffix"]["artifact_sha256"].items():
        key = {"plan.json": "plan", "receipt.json": "receipt", "run.json": "run", "train.jsonl": "trace",
               "final.pt": "checkpoint", "source/sha256.json": "source_manifest"}.get(name)
        if key is not None:
            check(hashes[f"control_suffix_{key}"] == expected, "suffix_review_link_" + key)
    check(plans["middle"]["input_sha256"]["checkpoint"] == hashes["control_prefix_checkpoint"]
          and plans["middle"]["input_sha256"]["run"] == hashes["control_prefix_run"]
          and plans["middle"]["input_sha256"]["qualification"] == hashes["control_prefix_receipt"], "prefix_to_middle_file_chain")
    check(plans["suffix"]["input_sha256"]["final.pt"] == hashes["control_middle_checkpoint"]
          and plans["suffix"]["input_sha256"]["run.json"] == hashes["control_middle_run"]
          and plans["suffix"]["input_sha256"]["receipt.json"] == hashes["control_middle_receipt"], "middle_to_suffix_file_chain")
    digests = {label: state_digest(state) for label, state in states.items()}
    check(digests["prefix"] == receipts["prefix"]["final_state_digest"]
          == reviews["prefix"]["final_native_state_digest_verified"]
          == plans["middle"]["initial_native_digest"] == receipts["middle"]["initial_native_digest"]
          == reviews["middle"]["initial_native_digest_verified"], "native_boundary1600")
    check(digests["middle"] == receipts["middle"]["final_native_digest"]
          == middle_review["final_native_digest_verified"] == plans["suffix"]["initial_native_digest"]
          == receipts["suffix"]["plan"]["initial_native_digest"] == reviews["suffix"]["initial_native_digest"], "native_boundary1856")
    check(digests["suffix"] == receipts["suffix"]["final_native_digest"] == reviews["suffix"]["final_native_digest"], "native_boundary6400")
    check(receipts["middle"]["initial_native_state_exact"] and receipts["middle"]["two_update_replay_exact"]
          and receipts["suffix"]["initial_native_state_exact"] and receipts["suffix"]["two_update_replay_exact"], "native_source_resume_witnesses")
    source_reports = {}
    for label, pin in (("prefix", PREFIX_PIN), ("middle", SUFFIX_PIN), ("suffix", SUFFIX_PIN)):
        check(runs[label]["particlegan_commit"] == plans[label]["particlegan_commit"] == pin, label + "_source_pin")
        committed = committed_python_hashes(repository, pin)
        source_root = Path(runs[label]["particlegan_root"])
        check(all(sha(source_root / name) == digest for name, digest in committed.items()), label + "_native_git_source")
        flat = {str(Path(name).relative_to("particlegan")): digest for name, digest in committed.items()}
        module_digest = state_digest({name: digest for name, digest in flat.items() if "/" not in name})
        check(module_digest == runs[label]["particlegan_source_digest"] == plans[label]["particlegan_source_digest"], label + "_native_source_digest")
        manifest = read(label, "source_manifest")
        applications = manifest if label == "prefix" else manifest["application"]
        declared = plans[label]["source_sha256"] if label == "prefix" else (
            plans[label]["application_source_sha256"] if label == "middle" else reviews[label]["source"]["manifest"]["application"])
        check(applications == declared, label + "_application_manifest")
        directory = Path(paths[f"control_{label}_trace"]).parent
        check(all(sha(directory / "source" / name) == digest for name, digest in applications.items()), label + "_application_snapshots")
        if label != "prefix":
            check(manifest["particlegan"] == flat
                  and all(sha(directory / "source/particlegan" / name) == digest for name, digest in flat.items()), label + "_native_snapshots")
        source_reports[label] = dict(particlegan_commit=pin, archive_root=str(source_root),
            native_module_digest=module_digest, native_git_python_files=len(committed), application_snapshot_files=len(applications))
    stream = torch.Generator().manual_seed(7)
    full, reports = [], []
    for label in ("prefix", "middle", "suffix"):
        first, last = expected_ranges[label]
        records = parts[label]
        check([row["step"] for row in records] == list(range(first, last + 1)), label + "_complete_step_range")
        state = states[label]
        check(state["policy"]["completed_steps"] == last and state_digest(data) == state["config"]["dataset_digest"], label + "_native_clock_data")
        for row in records:
            hold = row["step"] % 5 == 0
            pool = data["holds" if hold else "fit"]["context"]
            indices = torch.randint(len(pool), (4,), generator=stream).tolist()
            check(row["batch_indices"] == indices and row["hold"] is hold and row["game_weight"] == (.1 if hold else 1.)
                  and row["penalty_calls"] == row["step"] and row["dense_gradient_rows"] == (0 if row["step"] == 1 else 128),
                  "sampling_and_native_clock_" + str(row["step"]))
        check(torch.equal(stream.get_state(), state["data_rng"]), label + "_boundary_data_rng")
        paired_digest = sha_bytes(state["paired_noise_rng"])
        dv12_digest = sha_bytes(state["policy"]["streams"]["noise_generator"])
        check(records[-1]["paired_rng_digest"] == paired_digest and records[-1]["dv12_rng_digest"] == dv12_digest,
              label + "_boundary_noise_rng")
        program = [{key: row[key] for key in PROGRAM_KEYS} for row in records]
        if label == "middle":
            sampling = json.loads(Path(paths["control_middle_sampling"]).read_text())
            check(sampling == [{key: row[key] for key in PAIRED_KEYS} for row in records]
                  and state_digest(sampling) == receipts["middle"]["sampling_program_digest"], "middle_private_paired_program")
        if label == "suffix":
            check(state_digest(program) == reviews["suffix"]["program"]["complete_recorded_program_digest"], "suffix_qualified_program")
        full.extend(records)
        reports.append(dict(label=label, first_step=first, last_step=last, updates=len(records),
            trace_path=str(Path(paths[f"control_{label}_trace"]).resolve()), trace_sha256=hashes[f"control_{label}_trace"],
            review_path=str(Path(paths[f"control_{label}_review"]).resolve()), review_sha256=hashes[f"control_{label}_review"],
            final_checkpoint_sha256=hashes[f"control_{label}_checkpoint"], final_native_digest=digests[label],
            final_data_rng_digest=sha_bytes(state["data_rng"]), final_paired_rng_digest=paired_digest,
            final_dv12_rng_digest=dv12_digest, complete_recorded_program_digest=state_digest(program), source=source_reports[label]))
    check([row["step"] for row in full] == list(range(1, 6401)), "complete0_to6400_program")
    program = [{key: row[key] for key in PROGRAM_KEYS} for row in full]
    return full, dict(schema="supra_historical_v2_control_chain_v1", qualified=True,
        updates=6400, control_trace_parts=reports, complete_recorded_program_digest=state_digest(program),
        native_boundary_digests=digests, cpu_sampling_replayed_exactly=True,
        source_profile="legacy pr155_routed; f4590..1600 then cabe1601..6400",
        limits="No historical files rewritten. CPU verifies exact recorded ranges, source/review/receipt chains, complete native boundary values and CPU data sampling; CUDA Gaussian/DV12 draws are pinned recorded digests and saved owned states, not rerun on CPU.")


def sha_bytes(tensor):
    import hashlib
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
