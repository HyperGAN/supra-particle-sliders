"""Synthetic mathematical and mutation checks; no Supra or checkpoint execution."""
from copy import deepcopy
import hashlib
import json
import math

import pytest
import torch
from torch import nn

from scripts.diagnose_e22_supra_endpoint_game_rays import (
    ALPHAS, ARMS, CARD, COUNTS, ENDPOINTS, POOLS, ROOT, final_report, module_witness, panel_batches,
    physical_gpu0, qualification_gate, qualified_evaluations, read, rng_witness, score_ray, summarize_ray, validate_card,
)
from scripts.experiment_e22_supra_particle_gated import write
from scripts.review_e22_supra_neutral_initialization import Reviewer, canonical, check_evaluation


class LinearCritic(nn.Module):
    def __init__(self, bias=0.):
        super().__init__()
        self.register_buffer("bias", torch.tensor(bias, dtype=torch.float64))

    def forward(self, error, condition):
        return error.mean((1, 2)).unsqueeze(1) + self.bias


def test_context_derivatives_are_not_divided_by_batch_and_bias_cancels():
    error = torch.tensor([.4, -.25, .1], dtype=torch.float64).reshape(3, 1, 1).expand(3, 2, 2)
    panel = torch.zeros(4, 3, 2, 2, dtype=torch.float64)
    condition = torch.zeros(3, 1, dtype=torch.float64)
    games, derivatives = score_ray(LinearCritic(9.), error, condition, panel)
    plain_games, plain_derivatives = score_ray(LinearCritic(), error, condition, panel)
    mean = error.mean((1, 2))
    for alpha in ALPHAS:
        torch.testing.assert_close(games[f"{alpha:g}"], torch.nn.functional.softplus(-alpha * mean), rtol=0, atol=2e-15)
        torch.testing.assert_close(games[f"{alpha:g}"], plain_games[f"{alpha:g}"], rtol=0, atol=2e-15)
    torch.testing.assert_close(derivatives["0"], -.5 * mean, rtol=0, atol=2e-15)
    torch.testing.assert_close(derivatives["1"], -mean * torch.sigmoid(-mean), rtol=0, atol=2e-15)
    for key in derivatives:
        torch.testing.assert_close(derivatives[key], plain_derivatives[key], rtol=0, atol=2e-15)
    # This explicit fixed critic is a counterexample to interpreting sub-log2 as exact matching.
    assert games["0.01"][0] < math.log(2)
    assert derivatives["0"][0] < 0 and error[0].count_nonzero() > 0


def test_signed_summary_retains_origin_drift_and_subject_reversals():
    error = torch.tensor([.4, -.25], dtype=torch.float64).reshape(2, 1, 1).expand(2, 2, 2)
    games, derivatives = score_ray(LinearCritic(), error, torch.zeros(2, 1), torch.zeros(4, 2, 2, 2, dtype=torch.float64))
    summary = summarize_ray(games, derivatives, torch.tensor([4, 5]), 1e-6)
    assert summary["radial_derivatives"]["0"]["negative_contexts"] == 1
    assert summary["subjects"]["4"]["positive_minus_negative_game"]["1"] < 0
    assert summary["subjects"]["5"]["positive_minus_negative_game"]["1"] > 0
    assert all(value == 1 for value in summary["positive_ray_reversals"].values())
    assert all(value == 0 for value in summary["subjects"]["5"]["positive_ray_reversals"].values())


def test_scoring_preserves_inputs_rng_and_frozen_judge():
    critic = LinearCritic().eval().requires_grad_(False)
    error, condition = torch.ones(2, 2, 2, dtype=torch.float64), torch.zeros(2, 1)
    panel = torch.arange(32, dtype=torch.float64).reshape(4, 2, 2, 2) / 17
    clones = [value.clone() for value in (error, condition, panel)]
    before, rng = module_witness(critic), rng_witness()
    score_ray(critic, error, condition, panel)
    assert module_witness(critic) == before and rng_witness() == rng
    for value, expected in zip((error, condition, panel), clones):
        assert torch.equal(value, expected) and value.grad is None


@pytest.mark.parametrize("mutation", ["tensor", "mode", "ownership", "multiplier", "routing_frame"])
def test_module_witness_detects_relevant_mutation(mutation):
    module = nn.Linear(2, 1).eval().requires_grad_(False)
    module.multiplier, module.frame, module._in_forward = 1., None, False
    before = module_witness(module)
    if mutation == "tensor":
        module.weight.add_(1.)
    elif mutation == "mode":
        module.train()
    elif mutation == "ownership":
        module.weight.requires_grad_(True)
    elif mutation == "multiplier":
        module.multiplier = .5
    else:
        module._in_forward = True
    assert module_witness(module) != before


def gate_fixture():
    card = read(CARD)
    card["runtime_review_sha256"] = "r" * 64
    plan = dict(application_source_sha256={"held": "source"}, input_sha256={"data": "input"})
    status = dict(phase="complete", step=6400, steps=6400, editing_updates=6400, preservation_updates=0)
    receipt = dict(status="complete", plan=deepcopy(plan), checks={"witness": True})
    review = dict(schema="supra_neutral_initialization_independent_review_v1", qualified=True, partial=False,
        protocol_sha256=card["parent_protocol_sha256"], plan_sha256=card["parent_plan_sha256"],
        reviewer_sha256=card["reviewer_sources_sha256"]["scripts/review_e22_supra_neutral_initialization.py"],
        reviewer_dependency_sha256={str(ROOT / name): digest for name, digest in card["reviewer_sources_sha256"].items()
                                   if name != "scripts/review_e22_supra_neutral_initialization.py"},
        application_source_sha256=plan["application_source_sha256"], native_source_digest=card["native_source_digest"],
        input_sha256=plan["input_sha256"])
    return card, plan, status, receipt, review


@pytest.mark.parametrize("mutation", ["active", "partial", "unqualified", "review_identity", "failed_runtime"])
def test_gate_rejects_incomplete_or_unbound_evidence(mutation):
    card, plan, status, receipt, review = gate_fixture()
    qualification_gate(card, plan, status, receipt, review, "r" * 64)
    if mutation == "active":
        status["phase"] = "training"
    elif mutation == "partial":
        review["partial"] = True
    elif mutation == "unqualified":
        review["qualified"] = False
    elif mutation == "review_identity":
        card["runtime_review_sha256"] = "s" * 64
    else:
        receipt["checks"]["witness"] = False
    with pytest.raises(ValueError):
        qualification_gate(card, plan, status, receipt, review, "r" * 64)


@pytest.mark.parametrize("field,value", [("alphas", [0, .5, 1]), ("execution_budget_seconds", 1201),
    ("judges", ["D1856"]), ("editing_subjects", [4, 5]), ("alpha1_match_tolerance", dict(atol=.1, rtol=.1))])
def test_card_rejects_changed_law_or_missing_denominator(field, value):
    card = read(CARD)
    validate_card(card)
    card[field] = value
    with pytest.raises(ValueError):
        validate_card(card)


def test_panels_continue_across_pools_in_original_b4_order():
    # Seven test contexts ensure the tail B3 is preserved, rather than padded/bulk-drawn.
    data = {pool: dict(context=torch.zeros(count, 4100)) for pool, count in zip(POOLS, (8, 7, 4, 2))}
    stream = torch.Generator().manual_seed(72)
    expected, batches = {}, {}
    for pool in POOLS:
        values = [torch.randn(4, min(4, len(data[pool]["context"]) - start), 256, 16, generator=stream)
                  for start in range(0, len(data[pool]["context"]), 4)]
        batches[pool] = values
        expected[pool] = hashlib.sha256(b"".join(value.numpy().tobytes() for value in values)).hexdigest()
    rng = rng_witness()
    actual, hashes, _ = panel_batches(data, expected)
    assert hashes == expected and rng_witness() == rng
    assert actual["test"][1].shape[1] == 3
    for pool in POOLS:
        for a, b in zip(actual[pool], batches[pool]):
            assert torch.equal(a, b)
    reset_test = torch.randn(4, 4, 256, 16, generator=torch.Generator().manual_seed(72))
    assert not torch.equal(actual["test"][0], reset_test)
    # A whole-pool draw assigns a different stream ordering to contexts.
    bulk_fit = torch.randn(4, 8, 256, 16, generator=torch.Generator().manual_seed(72))
    assert not torch.equal(torch.cat(actual["fit"], dim=1), bulk_fit)
    with pytest.raises(ValueError, match="Gaussian hashes"):
        panel_batches(data, {**expected, "test": "changed"})


@pytest.mark.parametrize("mutation", ["metadata", "changed_valid_reduction"])
def test_original_records_bind_to_qualified_review_before_capture(tmp_path, mutation):
    data, evaluation = {}, {}
    for pool, count in COUNTS.items():
        context = torch.zeros(count, 4100)
        context[:, 4097] = torch.tensor([4, 5, 15, 16, 17, 18])[torch.arange(count) % 6]
        data[pool] = dict(context=context)
        records = [dict(index=index, source_caption_id=int(context[index, 4097]), time=0.,
            mse_diagnostic=.01, D1856=.8, D6400=.9) for index in range(count)]
        evaluation[pool] = dict(count=count, records=records, rmse_diagnostic=math.sqrt(sum(row["mse_diagnostic"] for row in records)/count),
            D1856=sum(row["D1856"] for row in records)/count, D6400=sum(row["D6400"] for row in records)/count)
    summary = canonical(check_evaluation(Reviewer(), evaluation, data, "fixture"))
    labels = [arm + "@" + str(step) for arm in ARMS for step in ENDPOINTS] + ["ordinary_lora_6400"]
    review = dict(evaluation_summaries={label: deepcopy(summary) for label in labels})
    (tmp_path / "historical-references.json").write_text(json.dumps(dict(ordinary_lora_6400=evaluation)))
    for arm in ARMS:
        (tmp_path / arm).mkdir()
        for step in ENDPOINTS:
            (tmp_path / arm / f"evaluation-{step:05d}.json").write_text(json.dumps(evaluation))
    qualified_evaluations(tmp_path, review, data)
    changed = deepcopy(evaluation)
    if mutation == "metadata":
        changed["fit"]["records"][0]["time"] = .1
    else:
        changed["fit"]["records"][0]["D1856"] += .1
        changed["fit"]["D1856"] = sum(row["D1856"] for row in changed["fit"]["records"])/COUNTS["fit"]
    (tmp_path / "historical-references.json").write_text(json.dumps(dict(ordinary_lora_6400=changed)))
    with pytest.raises(ValueError, match="context|qualified all-pool record reductions"):
        qualified_evaluations(tmp_path, review, data)


@pytest.mark.parametrize("visible", [None, "1", "0,1"])
def test_gpu0_execution_refuses_ambiguous_or_different_physical_mapping(visible):
    physical_gpu0({"CUDA_VISIBLE_DEVICES": "0"})
    with pytest.raises(ValueError, match="physical GPU0"):
        physical_gpu0({} if visible is None else {"CUDA_VISIBLE_DEVICES": visible})


@pytest.mark.parametrize("delays", [(1.1, 0.), (.6, .6)])
def test_postwrite_budget_overrun_preserves_incomplete_receipts(tmp_path, delays):
    clock, writes = [0.], [0]
    def delayed_write(path, value):
        write(path, value)
        clock[0] += delays[min(writes[0], len(delays)-1)]
        writes[0] += 1
    with pytest.raises(TimeoutError):
        final_report(tmp_path, dict(qualified=True, results={"retained": True}), started=0., limit=1.,
                     clock=lambda: clock[0], writer=delayed_write)
    report, completion = read(tmp_path / "report.json"), read(tmp_path / "completion.json")
    assert report["qualified"] is False and report["results"]["retained"] is True
    assert completion["complete"] is False and completion["budget_overrun"] is True


def test_completion_receipt_binds_report_after_serialization(tmp_path):
    clock = [0.]
    def delayed_write(path, value):
        write(path, value)
        clock[0] += .1
    final_seconds = final_report(tmp_path, dict(qualified=True), started=0., limit=1.,
        clock=lambda: clock[0], writer=delayed_write)
    completion = read(tmp_path / "completion.json")
    assert completion["complete"] is True and completion["wall_seconds"] == .1
    assert final_seconds == .2
    assert completion["report_sha256"] == hashlib.sha256((tmp_path / "report.json").read_bytes()).hexdigest()
