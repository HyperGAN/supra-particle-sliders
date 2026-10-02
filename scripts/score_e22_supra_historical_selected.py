#!/usr/bin/env python3
"""Score fixed historical6400 and selected28000; zero training or selection.

This separate observation never modifies a running study or its frozen card.
Both models receive the qualified parent's complete pools and CPU72 panels.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import json
import os
from pathlib import Path
import sys
import time

STARTED = time.monotonic()
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import torch.nn.functional as F
from scripts import diagnose_e22_supra_endpoint_game_rays as rays
from scripts.e22_supra_neutral_continuation_contract import bind_code, nested_code, ordinary_metadata_contract
from scripts import experiment_e22_supra_neutral_initialization as held
from scripts.experiment_e22_supra_particle_gated import emit, sha, write
from scripts.review_e22_supra_pr223 import state_digest

CARD = ROOT / "docs/e22_supra_historical_selected_protocol.json"
LABELS = ("ordinary_lora_6400", "ordinary_lora_selected_28000")


def require(value, label):
    if not value:
        raise ValueError(label)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    card = rays.read(CARD)
    output = Path(card["output"])
    require(not output.exists(), "preserve existing observation")
    require(card["training_updates"] == 0 and card["selection"] is False
            and card["budget_seconds"] == 300 and tuple(card["artifacts"]) == LABELS,
            "fixed observation contract")
    files = {str(CARD): sha(CARD)}
    for path, digest in card["extra_input_and_source_sha256"].items():
        require(sha(path) == digest, "declared source/input changed: " + path)
        files[path] = digest
    parent = Path(card["parent_run"])
    raycard, plan, review, _, parent_files, data, evaluations = rays.preflight(
        parent, Path(card["particlegan_root"]), rays.read(rays.CARD), card["parent_review_sha256"])
    files.update(parent_files)
    files[str(rays.CARD)] = sha(rays.CARD)
    historical_status = rays.read(card["historical_status"])
    historical_metadata = rays.read(card["historical_metadata"])
    require(historical_status["phase"] == "complete" and historical_status["step"] == 28800
            and historical_status["best_step"] == historical_status["selected_step"] == 28000
            and historical_metadata["step"] == 28000 and historical_metadata["stopped_step"] == 28800,
            "historical selection identity")
    # This was historical selection, not a new choice based on this evaluation.
    require(historical_metadata["selection_score"] == historical_status["best_score"],
            "historical selection metadata agreement")
    from safetensors import safe_open
    from safetensors.torch import load_file
    for label, entry in card["artifacts"].items():
        require(sha(entry["path"]) == entry["sha256"], "fixed adapter SHA: " + label)
        with safe_open(entry["path"], framework="pt", device="cpu") as handle:
            ordinary_metadata_contract(handle.metadata(), plan["protocol"]["host"],
                                       entry["step"], card["prompts_sha256"])
    require(state_digest(load_file(card["artifacts"][LABELS[1]]["path"], device="cpu")) ==
            state_digest(load_file(card["historical_selected_checkpoint"], device="cpu")),
            "historical selected and checkpoint28000 tensors agree")
    rays.check_budget(STARTED, card["budget_seconds"])
    if args.preflight_only:
        require(not torch.cuda.is_initialized(), "CPU preflight must not initialize CUDA")
        emit(preflight_passed=True, training_updates=0, model_forward_calls=0,
             seconds=time.monotonic() - STARTED)
        return
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "0", "physical GPU0 binding")
    sys.path.insert(0, card["particlegan_root"])
    import particlegan
    from safetensors.torch import load_file
    from supra.runtime import TARGETS, MODEL_SOURCE, model_module, load_adapter_state
    from supra.particle_game import ConditionalTokenCritic, patchify
    from supra.particle_training_data import FrozenSliderContexts
    require(Path(particlegan.__file__).resolve().parent.parent == Path(card["particlegan_root"]),
            "actual native import")
    require(sha(MODEL_SOURCE) == plan["protocol"]["inputs"]["backend_model_source"]["sha256"],
            "actual backend source")
    device = torch.device("cuda:0")
    require(torch.cuda.is_available(), "actual BF16 host requires CUDA")
    torch.cuda.set_device(device)
    torch.set_num_threads(8)
    output.mkdir()
    checks, captures, results, probes, summaries = {}, {}, {}, {}, {}
    reduction_review = rays.Reviewer()
    calls = dict(student=0, live_teacher=0)
    def check(label, value):
        checks[label] = bool(value)
        require(value, label)
    def budget():
        return rays.check_budget(STARTED, card["budget_seconds"])
    rng = rays.rng_witness(device)
    try:
        saved = {name: torch.load(plan["input_paths"][name], map_location="cpu",
                                 weights_only=False, mmap=True) for name in rays.JUDGES}
        teacher = {key.removeprefix("teacher."): value for key, value in
                   saved["D6400"]["policy"]["models"]["encoder"].items()
                   if key.startswith("teacher.")}
        with torch.random.fork_rng(devices=[0]), torch.device("meta"):
            backend = model_module()
            base = backend.SupraDiT()
            backend.attach_supra_lora(base, rank=16, alpha=16, targets=TARGETS)
        base.load_state_dict(teacher, strict=True, assign=True)
        base.to(device).eval().requires_grad_(False)
        check("pure_frozen_teacher", all(not value.count_nonzero() for key, value in
                                         teacher.items() if key.endswith(".up.weight")))
        encoder = FrozenSliderContexts(data["text_contexts"], data["text_masks"], base, cfg=3.).to(device)
        encoder.eval().requires_grad_(False)
        judges = {}
        for name in rays.JUDGES:
            with torch.random.fork_rng(devices=[0]):
                judge = ConditionalTokenCritic(data["coordinate_scale"]).to(device)
            judge.load_state_dict(saved[name]["policy"]["models"]["critic"], strict=True)
            judges[name] = judge.eval().requires_grad_(False)
            check("frozen_judge_identity:" + name, state_digest(judge.state_dict()) ==
                  raycard["critic_tensor_digests"][name])
        owner_before = state_digest(dict(base=rays.module_witness(base), encoder=rays.module_witness(encoder),
                                        judges={name: rays.module_witness(value) for name, value in judges.items()}))
        data_before = state_digest(data)
        # Execute the original qualified evaluator's exact code and decorator.
        owners = dict(torch=torch, F=F, device=device, require=check, patchify=patchify,
                      protocol=plan["protocol"])
        full_evaluation = torch.no_grad()(bind_code(nested_code(held.main.__code__, "full_evaluation"),
                                                   vars(held), owners))
        indices = rays.read(card["probe_indices_path"])
        probe_context = data["test"]["context"][indices].to(device)
        probe_panels = torch.randn(4, len(indices), 256, 16,
                                  generator=torch.Generator().manual_seed(72)).to(device)
        clean_probe = torch.no_grad()(bind_code(nested_code(held.main.__code__, "clean_probe"),
            vars(held), dict(owners, judges=judges, probe_context=probe_context, probe_panels=probe_panels)))
        for label, entry in card["artifacts"].items():
            budget()
            model = deepcopy(base)
            load_adapter_state(model, load_file(entry["path"], device="cpu"))
            before = rays.module_witness(model)
            captured = []
            class Reference:
                def __init__(self):
                    self.encoder = encoder
                @torch.no_grad()
                def routed_forward(self, context):
                    budget()
                    z, t, text, mask, empty, emask, strength = encoder.unpack(context)
                    branches = [value for value in model.modules() if hasattr(value, "multiplier")]
                    prior = [value.multiplier for value in branches]
                    try:
                        for value in branches:
                            value.multiplier = strength
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            velocity = model(torch.cat((z, z)), torch.cat((t, t)),
                                torch.cat((text, empty.expand(len(z), -1, -1))),
                                torch.cat((mask, emask.expand(len(z), -1)))).float()
                        positive, negative = velocity.chunk(2)
                        residual = negative + 3 * (positive - negative) - encoder.teacher_velocity(context)
                    finally:
                        for value, multiplier in zip(branches, prior):
                            value.multiplier = multiplier
                    check("finite_capture:" + label + ":" + str(len(captured)), bool(torch.isfinite(residual).all()))
                    captured.append(residual.cpu())
                    calls["student"] += 1
                    calls["live_teacher"] += 1
                    return residual
            reference = Reference()
            results[label] = full_evaluation(reference, data, judges)
            summaries[label] = rays.check_evaluation(reduction_review, results[label], data, label)
            captures[label] = dict(full_pools=torch.cat(captured))
            captured.clear()
            probes[label] = clean_probe(reference)
            captures[label]["probe"] = torch.cat(captured)
            check("student_immutable:" + label, rays.module_witness(model) == before)
            check("global_rng_immutable:" + label, rays.rng_witness(device) == rng)
            del model
            gc.collect()
        check("reference6400_exact_qualified_replay", results[LABELS[0]] == evaluations[LABELS[0]])
        check("base_teacher_critics_immutable", owner_before == state_digest(dict(
            base=rays.module_witness(base), encoder=rays.module_witness(encoder),
            judges={name: rays.module_witness(value) for name, value in judges.items()})))
        check("dataset_immutable", state_digest(data) == data_before)
        check("final_rng_immutable", rays.rng_witness(device) == rng)
        for path, digest in files.items():
            budget()
            check("file_immutable:" + path, sha(path) == digest)
        raw_path = output / "residuals.pt"
        torch.save(dict(captures=captures, probe_indices=indices), raw_path)
        write(output / "evaluations.json", results)
        report = dict(schema="supra_historical_selected_observation_v1", qualified=True,
            card=card, card_sha256=sha(CARD), checks=checks, results=results, summaries=summaries,
            record_reduction_checks=reduction_review.checks, probes=probes,
            probe_contract=dict(count=len(indices), indices=indices, judge="D1856",
                context_metadata=probe_context[:, 4096:].cpu().tolist(),
                panel_sha256=state_digest(probe_panels), panel_law="separate CPU72 four-panel stream",
                meaning="Observational first/last context per source at times0/.9. Primary is all240 fulltest contexts, never this12-context probe."),
            raw_residual_sha256=sha(raw_path), evaluation_sha256=sha(output / "evaluations.json"),
            consumed_source_and_input_sha256=files, forward_calls=calls, training_updates=0,
            output_metrics_used_for_optimizer_or_selection=False, selection=False,
            limit="Historical MSE/AdamW validation-selected28000 is a separate best-model reference, not a matched optimizer comparison. Both frozen judges are learned within one architectural family.")
        rays.final_report(output, report, STARTED, card["budget_seconds"])
        emit(complete=True, seconds=time.monotonic() - STARTED, probes=probes,
             test={name: {key: value for key, value in pool["test"].items() if key != "records"}
                   for name, pool in results.items()})
    except Exception as error:
        write(output / "failure.json", dict(qualified=False, error=type(error).__name__ + ": " + str(error),
            seconds=time.monotonic() - STARTED, checks=checks, forward_calls=calls, training_updates=0))
        raise


if __name__ == "__main__":
    main()
