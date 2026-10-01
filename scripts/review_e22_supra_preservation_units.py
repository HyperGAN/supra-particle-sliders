#!/usr/bin/env python3
"""Independent read-only CPU qualification of the fixed V2 game-units test."""
import argparse
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import torch
from safetensors import safe_open


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.review_e22_supra_training_controls import evaluate_check, finite, paired_pool, state_digest

PIN = "cabe2084284db923d525918cbf3e18de6f20faac"
START, END = 1856, 2112
MODES = ("native", "hold_units")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frozen(state):
    return {role: {name: state["policy"]["models"][role][name]
                   for name, trainable in state["policy"]["requires_grad"][role].items() if not trainable}
            for role in ("generator", "encoder")}


def export_check(path, receipt, final, run):
    export = receipt["export"]
    source = receipt["served_source"]
    selected = final["policy"]["models" if source == "fast" else "averages"]
    names = [name for name, trainable in final["policy"]["requires_grad"]["generator"].items() if trainable]
    expected = {"generator." + name: selected["generator"][name] for name in names}
    expected.update({"router." + name: value for name, value in selected["router"].items() if name != "log_mass"})
    expected.update({"bank.table": final["policy"]["table" if source == "fast" else "averaged_table"],
                     "bank.log_mass": selected["router"]["log_mass"]})
    assert source in ("fast", "averaged") and len(expected) == export["tensors"] == 428
    assert path.stat().st_size == export["bytes"]
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
        config = json.loads(metadata["config"])
        assert metadata["format"] == "supra_particlegan_clean_v2" and config["architecture"] == "linear_modulated_v2"
        assert config == export["config"] and config["completed_steps"] == END
        assert config["sites"] == run["config"]["sites"] and len(config["sites"]) == 71
        assert (config["rank"], config["z_dim"], config["num_particles"], config["cfg"]) == (16, 4, 128, 3.)
        assert config["sampling"] == "clean" and config["routed_geometry"] == "mass_atoms_v1"
        assert config["extra"]["mode"] == receipt["plan"]["mode"]
        assert config["extra"]["selection"] == "fixed 256-update continuation; output metrics evaluation only"
        pins = json.loads(metadata["native_pins"])
        assert pins["backend_sha256"] == read(ROOT / "backend.lock.json")["sha256"]
        for name, field in (("model_revision", "model_revision"), ("text_encoder_revision", "text_revision"), ("vae_revision", "vae_revision")):
            assert pins[name] == run[field]
        assert set(handle.keys()) == set(expected)
        for name, reference in expected.items():
            value = handle.get_tensor(name)
            assert value.dtype == torch.float32 and torch.equal(value, reference), f"export mismatch: {name}"
            valid = ((torch.isfinite(value) | torch.isneginf(value)).all() and torch.isfinite(value).any()
                     if name == "bank.log_mass" else torch.isfinite(value).all())
            assert bool(valid), f"invalid export values: {name}"
    assert sum(value.numel() for value in expected.values()) == export["parameters"]
    return dict(tensors=428, selected_native_tensors_exact=True, schema_and_pins_match=True, sha256=sha(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "outputs/e22-particle-preservation-units-256")
    parser.add_argument("--long-run", type=Path, default=ROOT / "outputs/e22-particle-v2-6400")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.run / "qualification-review.json"
    if output.exists():
        parser.error("preserve the previous review; choose a fresh output")
    torch.set_num_threads(1)
    paths = {mode: args.run / mode for mode in MODES}
    receipts = {mode: read(path / "receipt.json") for mode, path in paths.items()}
    reference = Path(receipts["native"]["plan"]["reference_run"])
    original = torch.load(reference / "final.pt", map_location="cpu", weights_only=False, mmap=True)
    data = torch.load(reference / "data.pt", map_location="cpu", weights_only=False)
    initial_digest = state_digest(original)
    frozen_digest = state_digest(frozen(original))
    assert original["policy"]["completed_steps"] == START
    assert state_digest(data) == original["config"]["dataset_digest"]
    archive = subprocess.check_output(["git", "-C", "/ml2/hypergan/ParticleGAN-pr155-merge", "archive", PIN, "particlegan"])
    with tarfile.open(fileobj=io.BytesIO(archive)) as tree:
        committed = {str(Path(member.name).relative_to("particlegan")):
                     hashlib.sha256(tree.extractfile(member).read()).hexdigest()
                     for member in tree.getmembers() if member.isfile() and member.name.endswith(".py")}
    reports, evaluations, traces, programs = {}, {}, {}, {}
    for mode, path in paths.items():
        receipt = receipts[mode]
        plan, run, status = (read(path / name) for name in ("plan.json", "run.json", "status.json"))
        assert receipt["plan"] == run["preservation_unit_experiment"] == plan
        assert plan["mode"] == mode and plan["particlegan_commit"] == run["particlegan_commit"] == PIN
        assert (plan["start_step"], plan["additional_updates"], plan["final_step"], run["fixed_updates"]) == (START, 256, END, END)
        assert status["phase"] == "complete" and status["step"] == END
        assert plan["initial_native_digest"] == receipt["initial_native_digest"] == initial_digest
        assert plan["task_gradient_weight"] == {"fit": 1., "holds": .1}
        expected_weight = .1 if mode == "native" else 1.
        assert plan["critic_payoff_weight"] == plan["controller_payoff_weight"] == {"fit": 1., "holds": expected_weight}
        for flag in ("initial_native_state_exact", "four_update_replay_exact", "replay_includes_preservation",
                     "data_and_paired_streams_matched", "frozen_unchanged", "export_reload_exact",
                     "evaluation_state_unchanged", "source_and_inputs_unchanged"):
            assert receipt[flag] is True, f"{mode}: failed witness {flag}"
        resume = read(path / "restore-review.json")
        assert resume == dict(initial_native_state_exact=True, initial_native_digest=initial_digest,
                              four_update_replay_exact=True, preflight_updates=4, includes_preservation=True)
        for label, expected in plan["input_sha256"].items():
            name = dict(checkpoint="final.pt", data="data.pt", run="run.json", qualification="receipt.json")[label]
            assert sha(reference / name) == expected, f"initial input changed: {label}"
        assert sha(path / "data.pt") == sha(reference / "data.pt")
        manifest = read(path / "source/sha256.json")
        assert manifest == dict(application=plan["application_source_sha256"], particlegan=plan["particlegan_source_sha256"])
        assert manifest["particlegan"] == committed, "executed native source differs from pinned Git tree"
        for name, expected in manifest["application"].items():
            assert sha(path / "source" / name) == sha(ROOT / name) == expected, f"application source changed: {name}"
        for name, expected in manifest["particlegan"].items():
            assert sha(path / "source/particlegan" / name) == sha(Path(plan["particlegan_root"]) / "particlegan" / name) == expected
        assert state_digest({name: value for name, value in committed.items() if "/" not in name}) == run["particlegan_source_digest"] == plan["particlegan_source_digest"]
        final = torch.load(path / "final.pt", map_location="cpu", weights_only=False, mmap=True)
        assert state_digest(final) == receipt["final_native_digest"] and sha(path / "final.pt") == receipt["final_checkpoint_sha256"]
        assert final["config"] == original["config"] and json.loads(json.dumps(final["config"])) == run["config"]
        assert final["config"]["output_error_guard"] is False and final["config"]["max_feature_context_harm"] == 0.
        assert final["policy"]["completed_steps"] == END and state_digest(frozen(final)) == frozen_digest
        assert final["policy"]["optimizers"][1]["regularizer"]["record"]["observed_steps"] == END
        rows = [json.loads(line) for line in (path / "train.jsonl").read_text().splitlines()]
        program = read(path / "sampling-program.json")
        assert len(rows) == len(program) == 256 and [row["step"] for row in rows] == list(range(START + 1, END + 1))
        assert state_digest(program) == receipt["sampling_program_digest"]
        finite(rows)
        data_stream = torch.Generator().set_state(original["data_rng"].cpu())
        for row, expected in zip(rows, program):
            hold = row["step"] % 5 == 0
            weight = .1 if hold else 1.
            dw = weight if mode == "native" else 1.
            indices = torch.randint(len(data["holds" if hold else "fit"]["context"]), (4,), generator=data_stream).tolist()
            assert all(row[key] == value for key, value in expected.items())
            assert row["mode"] == mode and row["hold"] is hold and row["batch_indices"] == indices
            assert row["game_weight"] == weight and row["critic_game_weight"] == row["controller_payoff_weight"] == dw
            expected_observed_g = weight * row["controller_observed_g"] if mode == "hold_units" else row["controller_observed_g"]
            assert abs(row["loss_g"] - expected_observed_g) < 1e-7
            assert row["loss_d_game"] == row["controller_observed_d"]
            assert row["penalty_calls"] == row["step"] and row["penalty_phase"] == "blend" and row["dense_gradient_rows"] == 128
            assert abs(row["loss_g"] - weight * row["loss_g_unweighted"]) < 1e-7
            assert abs(row["loss_d_game"] - dw * row["loss_d_game_unweighted"]) < 1e-7
            for geometry in row["generator_role_gradient_geometry"].values():
                assert abs(geometry["actual_gradient_norm"] - weight * geometry["unweighted_game_gradient_norm"]) < 1e-12
        assert torch.equal(data_stream.get_state(), final["data_rng"])
        assert hashlib.sha256(bytes(final["data_rng"].tolist())).hexdigest() == receipt["final_data_rng_digest"]
        assert hashlib.sha256(bytes(final["paired_noise_rng"].tolist())).hexdigest() == receipt["final_paired_rng_digest"] == rows[-1]["paired_rng_digest"]
        assert hashlib.sha256(bytes(final["policy"]["streams"]["noise_generator"].tolist())).hexdigest() == receipt["final_dv12_rng_digest"] == rows[-1]["dv12_rng_digest"]
        moves = [row["move"] for row in rows if isinstance(row["move"], dict) and row["move"].get("moves", 0)]
        assert moves == receipt["accepted_moves"]
        evaluation = read(path / "evaluation.json")
        finite(evaluation)
        evaluate_check(evaluation, receipt, data)
        reports[mode] = dict(full_final_native_digest_verified=True, final_digest=receipt["final_native_digest"],
                             frozen_parameters_exact=True, source_files_match_git_pin=True,
                             exact_resume_including_preservation_witness=True,
                             updates=256, preservation_updates=sum(row["hold"] for row in rows),
                             export=export_check(path / "final.safetensors", receipt, final, run),
                             final_critic_digest=state_digest(final["policy"]["models"]["critic"]),
                             full_budget_game_summary=receipt["full_budget_game_summary"])
        evaluations[mode], traces[mode], programs[mode] = evaluation, rows, program
    assert programs["native"] == programs["hold_units"], "paired application sampling differs between arms"
    assert all(first["dv12_rng_digest"] == second["dv12_rng_digest"] for first, second in zip(traces["native"], traces["hold_units"])), "DV12 draw clocks differ"
    assert all(first[key] == second[key] for first, second in zip(traces["native"][:3], traces["hold_units"][:3])
               for key in ("loss_d", "loss_d_game", "loss_g", "penalty", "bank_grad_norm", "output_sigma")), "arms differ before first preservation intervention"
    long_prefix = None
    if (args.long_run / "train.jsonl").is_file():
        long_rows = [json.loads(line) for line in (args.long_run / "train.jsonl").read_text().splitlines()[:256]]
        if len(long_rows) == 256:
            assert all(all(row[key] == value for key, value in native.items())
                       for row, native in zip(traces["native"], long_rows)), "telemetry control differs from unchanged long native trajectory"
            long_prefix = dict(first_256_unchanged_native_trace_rows_exact=True, run=str(args.long_run))
    pairing = {pool: paired_pool(evaluations["native"][pool], evaluations["hold_units"][pool], data["prompts"])
               for pool in ("fit", "test", "holds", "preservation")}
    report = dict(all_checks_passed=True, initial_native_digest=initial_digest, start_step=START, final_step=END,
                  arms=reports, matched_data_paired_and_dv12_clocks=True, native_long_run_prefix=long_prefix,
                  common_start_critic_digest=state_digest(original["policy"]["models"]["critic"]),
                  common_critic_pairing=pairing, metrics={mode: receipts[mode]["metrics"] for mode in MODES},
                  interpretation="Common fixed step1856 critic: corrected units improve preservation but do not establish an edit win. "
                                 "Each arm's own final critic changes with the intervention and cannot be compared as a common judge.",
                  limits=["Initial equality/GPU restore/export-forward/evaluation immutability are recorded runtime witnesses, not rerun here.",
                          "CPU review verifies complete saved native states, frozen tensors, streams, sources, exports and full-pool aggregates.",
                          "Single task and matched stream; contexts are correlated; learned game judges are endogenous."])
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(all_checks_passed=True, output=str(output), common_critic_pairing=pairing)), flush=True)


if __name__ == "__main__":
    main()
