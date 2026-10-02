"""Small CPU algebra and observer-isolation checks; no Supra/GPU training."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

PATH = Path(__file__).resolve().parents[1] / "scripts/diagnose_e22_supra_cfg_gradient.py"
SPEC = importlib.util.spec_from_file_location("cfg_diagnostic", PATH)
DIAG = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DIAG)


def test_saved_checkpoint_config_matches_its_json_plan_without_mutation():
    config = {"sites": ("first", "second"), "recipe": {"betas": (0., .999)},
              "particle_init": "sampled_hb_neutral_v1"}
    state = {"config": config, "policy": {"completed_steps": 800}}
    declared = json.loads(json.dumps(config))
    DIAG.check_checkpoint_contract(state, declared, 800)
    assert isinstance(config["sites"], tuple)
    assert isinstance(config["recipe"]["betas"], tuple)
    changed = {**declared, "particle_init": "sampled_v1"}
    with pytest.raises(ValueError, match="clock/config"):
        DIAG.check_checkpoint_contract(state, changed, 800)
    with pytest.raises(ValueError, match="clock/config"):
        DIAG.check_checkpoint_contract(state, declared, 802)


def test_same_actual_graph_half_vjps_reconstruct_guided():
    parameter = torch.tensor([[.3, -.4], [.7, .2]], dtype=torch.float64, requires_grad=True)
    conditional = torch.tensor([[1., 2.], [3., -1.]], dtype=torch.float64) @ parameter.T
    unconditional = torch.tensor([[.5, -1.], [2., 1.]], dtype=torch.float64) @ parameter.T
    guided = unconditional + 3 * (conditional - unconditional)
    force = torch.tensor([[.2, -.8], [.4, .1]], dtype=torch.float64)
    before = parameter.detach().clone()
    c, u, g = DIAG.half_vjps(conditional, unconditional, guided, force, {"parameter": parameter})
    torch.testing.assert_close(c["parameter"] + u["parameter"], g["parameter"], rtol=1e-14, atol=1e-14)
    assert parameter.grad is None and torch.equal(parameter, before)


def test_bf16_output_boundary_decomposition_records_rounding():
    parameter = torch.tensor([[.3, -.4], [.7, .2]], requires_grad=True)
    value = torch.tensor([[1., 2.], [3., -1.], [.5, -1.], [2., 1.]])
    raw = (value @ parameter.T).to(torch.bfloat16).float()
    conditional, unconditional = raw.chunk(2)
    guided = unconditional + 3 * (conditional - unconditional)
    force = torch.tensor([[.2, -.8], [.4, .1]])
    c, u, g = DIAG.half_vjps(conditional, unconditional, guided, force, {"parameter": parameter})
    result = DIAG.geometry(c, u, g)
    DIAG.verify_decomposition(result)
    assert result["decomposition_half_norm_relative_error"] <= 2e-4
    assert parameter.grad is None


def test_bf16_output_boundary_with_strong_cancellation_passes_half_norm_bound():
    parameter = torch.tensor([[.3, -.4], [.7, .2]], requires_grad=True)
    value = torch.tensor([[1., 2.], [3., -1.], [1.4999, 2.9999], [4.4999, -1.4999]])
    raw = (value @ parameter.T).to(torch.bfloat16).float()
    conditional, unconditional = raw.chunk(2)
    guided = unconditional + 3 * (conditional - unconditional)
    force = torch.tensor([[.2, -.8], [.4, .1]])
    result = DIAG.geometry(*DIAG.half_vjps(conditional, unconditional, guided, force, {"p": parameter}))
    assert result["cancellation_ratio"] < .001
    assert result["decomposition_verified"]
    assert result["half_sum_cancellation_faithful_to_guided"]
    DIAG.verify_decomposition(result)
    assert parameter.grad is None


def test_half_norm_bound_does_not_divide_by_nearly_cancelled_guided_norm():
    # Deliberately introduce a small reduction discrepancy. This calibrates the
    # acceptance rule, separately from the actual-graph identity tests above.
    c, u, g = ({"x": torch.tensor([x], dtype=torch.float64)} for x in (1., -1., 1e-5))
    result = DIAG.geometry(c, u, g)
    assert result["decomposition_relative_error"] == pytest.approx(1.)
    assert result["decomposition_error_norm"] > 1e-12 + 2e-4 * result["guided_gradient_norm"]
    assert result["decomposition_half_norm_relative_error"] == pytest.approx(5e-6)
    DIAG.verify_decomposition(result)


def test_half_norm_bound_rejects_wrong_guided_gradient():
    c, u, g = ({"x": torch.tensor([x], dtype=torch.float64)} for x in (1., -1., .1))
    result = DIAG.geometry(c, u, g)
    assert not result["decomposition_within_tolerance"]
    with pytest.raises(AssertionError, match="half-norm-scaled"):
        DIAG.verify_decomposition(result)


def test_bf16_matrix_reductions_are_recorded_unverified_without_losing_actual_guided_norm():
    # Unlike the real FP32 particle branch, this extra witness also performs
    # BF16 matrix reductions. Its discrepancy is a numerical limitation,
    # not evidence of an incorrect gradient or of actual Supra parity failure.
    parameter = torch.tensor([[.3, -.4], [.7, .2]], requires_grad=True)
    value = torch.tensor([[1., 2.], [3., -1.], [1.4999, 2.9999], [4.4999, -1.4999]])
    with torch.autocast("cpu", dtype=torch.bfloat16):
        raw = (value @ parameter.T).float()
    conditional, unconditional = raw.chunk(2)
    guided = unconditional + 3 * (conditional - unconditional)
    force = torch.tensor([[.2, -.8], [.4, .1]])
    result = DIAG.geometry(*DIAG.half_vjps(conditional, unconditional, guided, force, {"p": parameter}))
    assert result["cancellation_ratio"] < .005
    assert result["decomposition_half_norm_relative_error"] > 2e-4
    assert not result["decomposition_verified"]
    assert not result["half_sum_cancellation_faithful_to_guided"]
    assert "unverified" in result["decomposition_interpretation"]
    assert result["decomposition_relative_error"] > 1.
    assert result["guided_gradient_norm"] > 0
    assert result["guided_to_half_norm_ratio"] == pytest.approx(result["guided_gradient_norm"] / (
        result["conditional_gradient_norm"] + result["unconditional_gradient_norm"]))
    assert result["guided_to_half_norm_ratio"] < result["half_sum_cancellation_ratio"]
    assert parameter.grad is None


def test_cfg_identical_halves_have_known_cancellation():
    c, u, g = ({"x": torch.tensor([x], dtype=torch.float64)} for x in (3., -2., 1.))
    result = DIAG.geometry(c, u, g)
    assert result["half_cosine"] == -1
    assert result["cancellation_ratio"] == pytest.approx(.2)
    assert result["half_sum_cancellation_ratio"] == pytest.approx(.2)
    assert result["guided_to_half_norm_ratio"] == pytest.approx(.2)
    assert result["guided_to_conditional_norm"] == pytest.approx(1/3)


@pytest.mark.parametrize("magnitude", [0., 1e-14])
def test_zero_and_near_zero_norm_ratios_are_undefined(magnitude):
    value = {"x": torch.tensor([magnitude], dtype=torch.float64)}
    result = DIAG.geometry(value, value, value)
    assert all(result[name] is None for name in ["half_cosine", "cancellation_ratio",
               "guided_to_conditional_norm", "decomposition_relative_error",
               "decomposition_half_norm_relative_error", "half_sum_cancellation_ratio",
               "guided_to_half_norm_ratio"])


def test_per_context_forces_are_distinct_from_batch_cancellation():
    parameter = torch.tensor(1., dtype=torch.float64, requires_grad=True)
    conditional = parameter * torch.tensor([[1.], [-1.]], dtype=torch.float64)
    unconditional = conditional * .5
    guided = unconditional + 3 * (conditional - unconditional)
    totals = []
    for row in range(2):
        force = torch.zeros_like(guided); force[row] = 1
        c, u, g = DIAG.half_vjps(conditional, unconditional, guided, force, {"p": parameter})
        totals.append(g["p"])
        assert DIAG.geometry(c, u, g)["guided_gradient_norm"] == pytest.approx(2.)
    assert torch.equal(totals[0] + totals[1], torch.zeros_like(parameter))


class Branch(nn.Module):
    def __init__(self, site):
        super().__init__()
        self.site, self.frame = site, None
        self.down, self.bridge = nn.Linear(3, 2), nn.Linear(4, 2)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.first, self.second = Branch("first"), Branch("second")

    def forward(self, value, routing):
        for branch in (self.first, self.second):
            routing.mix(branch.site, torch.zeros(len(value)//2, 2, value.shape[1], 3))
            branch.down(value)
        return value


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Model()
        self.sites, self._in_forward = ("first", "second"), False

    def particle_branches(self):
        return (self.model.first, self.model.second)


class Routing:
    def __init__(self):
        self.sites = []

    def mix(self, site, logits):
        self.sites.append(site)
        return torch.ones(*logits.shape[:-1], 2)


@pytest.mark.parametrize("raise_after_forward", [False, True])
def test_observer_delegates_once_restores_hooks_and_does_not_write_grads(raise_after_forward):
    generator = Generator()
    original_modes = [module.training for module in generator.modules()]
    original_hooks = [list(module._forward_hooks) for module in generator.modules()]
    values = {name: tensor.clone() for name, tensor in generator.state_dict().items()}
    stream = torch.get_rng_state().clone()
    routing = Routing()
    def callback(models, value, candidate, routing):
        output = models["generator"].model(value, routing)
        if raise_after_forward:
            raise RuntimeError("fixture failure")
        return output
    value = torch.ones(4, 3, 3)
    try:
        with DIAG.observe_forward(generator, callback) as (observed, captured):
            output = observed({"generator": generator}, value, None, routing)
            assert output is value and captured["calls"] == 1
            assert set(captured["gates"]) == {"first", "second"}
    except RuntimeError:
        assert raise_after_forward
    assert routing.sites == ["first", "second"]
    assert original_hooks == [list(module._forward_hooks) for module in generator.modules()]
    assert original_modes == [module.training for module in generator.modules()]
    assert all(torch.equal(values[name], tensor) for name, tensor in generator.state_dict().items())
    assert all(parameter.grad is None for parameter in generator.parameters())
    assert torch.equal(stream, torch.get_rng_state())


def test_private_panels_match_frozen_raw_bytes_without_advancing_global_rng():
    before = torch.get_rng_state().clone()
    assert DIAG.raw_sha(DIAG.private_panels()) == "a854a5f234968016ed38049858b9562e3379ebe6ee19ee57fafd9043c7edb4bf"
    assert torch.equal(before, torch.get_rng_state())


def test_source_only_card_refuses_execution_before_reading_parent_or_loading_cuda(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr("sys.argv", [str(PATH), "--output", str(tmp_path/"uncreated")])
    with pytest.raises(SystemExit) as error:
        DIAG.main()
    assert error.value.code == 2
    assert "source-only diagnostic" in capsys.readouterr().err
    assert not (tmp_path/"uncreated").exists()


@pytest.fixture
def review_fixture(tmp_path):
    sources = {}
    for name in DIAG.REVIEWER_SOURCES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fixed CPU test reviewer fixture\n" + name + "\n")
        sources[name] = DIAG.sha(path)
    main, helper = DIAG.REVIEWER_SOURCES
    review = dict(schema="supra_neutral_initialization_independent_review_v1", qualified=True,
        partial=False, reviewer_sha256=sources[main],
        reviewer_dependency_sha256={str((tmp_path / helper).resolve()): sources[helper]})
    path = tmp_path / "independent-review.json"
    path.write_text(json.dumps(review))
    return tmp_path, {"reviewer_sources_sha256": sources}, path, review


def test_final_review_gate_binds_exact_review_and_actual_reviewer_sources(review_fixture):
    root, card, path, review = review_fixture
    assert DIAG.qualified_review_identity(card, path, DIAG.sha(path), root=root) == review


@pytest.mark.parametrize("claimed", [None, "", "0" * 64, "G" * 64])
def test_final_review_gate_rejects_missing_invalid_or_wrong_frozen_sha(review_fixture, claimed):
    root, card, path, _ = review_fixture
    with pytest.raises(ValueError, match="SHA256"):
        DIAG.qualified_review_identity(card, path, claimed, root=root)


@pytest.mark.parametrize("field,value", [("schema", "other"), ("qualified", False),
    ("qualified", 1), ("partial", True), ("partial", None),
    ("reviewer_sha256", "0" * 64), ("reviewer_dependency_sha256", {})])
def test_final_review_gate_rejects_unqualified_or_wrong_reviewer_even_with_matching_file_sha(review_fixture, field, value):
    root, card, path, review = review_fixture
    review[field] = value
    path.write_text(json.dumps(review))
    with pytest.raises(ValueError):
        DIAG.qualified_review_identity(card, path, DIAG.sha(path), root=root)


def test_final_review_gate_rejects_changed_actual_reviewer_source(review_fixture):
    root, card, path, _ = review_fixture
    (root / DIAG.REVIEWER_SOURCES[1]).write_text("changed reviewer source\n")
    with pytest.raises(ValueError, match="actual independent reviewer"):
        DIAG.qualified_review_identity(card, path, DIAG.sha(path), root=root)


def test_authorized_execution_requires_explicit_frozen_review_sha_before_parent_load(monkeypatch, capsys, tmp_path):
    card = json.loads(DIAG.CARD.read_text())
    card["execution"]["authorized"] = True
    card["parent_run"] = str(tmp_path / "nonexistent-parent")
    path = tmp_path / "authorized-card.json"
    path.write_text(json.dumps(card))
    monkeypatch.setattr("sys.argv", [str(PATH), "--execute", "--card", str(path),
                                  "--output", str(tmp_path / "uncreated")])
    with pytest.raises(SystemExit) as error:
        DIAG.main()
    assert error.value.code == 2
    assert "--qualified-review-sha256 is required" in capsys.readouterr().err
    assert not (tmp_path / "uncreated").exists()


def test_authorized_execution_rejects_wrong_review_sha_before_cuda_or_checkpoint_load(monkeypatch, capsys, tmp_path):
    run = tmp_path / "parent"
    run.mkdir()
    (run / "status.json").write_text(json.dumps({"phase": "complete", "step": 6400}))
    (run / "independent-review.json").write_text(json.dumps({"qualified": True, "partial": False}))
    card = json.loads(DIAG.CARD.read_text())
    card["execution"]["authorized"] = True
    card["parent_run"] = str(run)
    path = tmp_path / "authorized-card.json"
    path.write_text(json.dumps(card))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    def forbidden_cuda_check():
        raise AssertionError("CUDA must not be touched before exact review identity matches")
    monkeypatch.setattr(torch.cuda, "is_available", forbidden_cuda_check)
    monkeypatch.setattr("sys.argv", [str(PATH), "--execute", "--card", str(path),
        "--qualified-review-sha256", "0" * 64, "--output", str(tmp_path / "uncreated")])
    with pytest.raises(SystemExit) as error:
        DIAG.main()
    assert error.value.code == 2
    assert "does not match frozen SHA256" in capsys.readouterr().err
    assert not (tmp_path / "uncreated").exists()


def test_runtime_identity_explicitly_observes_ordinary_lora_multiplier(monkeypatch):
    # Isolate this runtime-attribute check from tensor hashing/native imports.
    monkeypatch.setitem(sys.modules, "supra.particle_pilot", SimpleNamespace(state_digest=lambda value: "fixture-digest"))
    generator = Generator()
    ordinary = nn.Module()
    ordinary.down, ordinary.up, ordinary.multiplier = nn.Identity(), nn.Identity(), .7
    generator.model.ordinary = ordinary
    models = {"generator": generator, "encoder": nn.Identity(), "router": nn.Identity()}
    table = torch.ones(2, 2, requires_grad=True)
    before = DIAG.runtime_identity(models, table)
    assert before["runtime"]["generator/model.ordinary"]["ordinary_lora_multiplier"] == .7
    ordinary.multiplier = .9
    assert DIAG.runtime_identity(models, table) != before
    ordinary.multiplier = .7
    assert DIAG.runtime_identity(models, table) == before


def test_completion_receipt_binds_report_and_final_clock_includes_receipt_write(tmp_path):
    state = {"now": 0.}
    def writer(path, value):
        DIAG.write(path, value)
        state["now"] += 30.
    elapsed = DIAG.final_report(tmp_path, {"complete": True}, 0., clock=lambda: state["now"], writer=writer)
    receipt = json.loads((tmp_path / "completion.json").read_text())
    report = json.loads((tmp_path / "report.json").read_text())
    assert elapsed == 60.
    assert receipt["wall_seconds"] == 30.
    assert receipt["report_sha256"] == DIAG.sha(tmp_path / "report.json")
    assert receipt["complete"] and report["completion_receipt_required"]
    assert receipt["quality_qualification_credit"] == "none"


@pytest.mark.parametrize("late_stage", ["report", "hash", "receipt"])
def test_completion_overrun_retains_observations_but_marks_report_and_receipt_incomplete(tmp_path, monkeypatch, late_stage):
    state = {"now": 0., "advanced": False}
    real_sha = DIAG.sha
    def advance(stage):
        if stage == late_stage and not state["advanced"]:
            state.update(now=901., advanced=True)
    def writer(path, value):
        DIAG.write(path, value)
        advance("report" if path.name == "report.json" else "receipt")
    def timed_sha(path):
        value = real_sha(path)
        advance("hash")
        return value
    monkeypatch.setattr(DIAG, "sha", timed_sha)
    with pytest.raises(TimeoutError, match="900-second budget"):
        DIAG.final_report(tmp_path, {"complete": True, "results": ["retained witness"]},
            0., clock=lambda: state["now"], writer=writer)
    report = json.loads((tmp_path / "report.json").read_text())
    receipt = json.loads((tmp_path / "completion.json").read_text())
    assert not report["complete"] and report["budget_overrun"]
    assert not receipt["complete"] and receipt["budget_overrun"]
    assert report["results"] == ["retained witness"]
    assert receipt["report_sha256"] == real_sha(tmp_path / "report.json")
    assert receipt["wall_seconds"] == 901.
